import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler
from typing import Dict, Any, Optional
import logging
from tqdm import tqdm
import os
from ..config.model_config import MODEL_CONFIG

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: Any,
        eval_dataloader: Optional[Any] = None,
        config: Dict[str, Any] = MODEL_CONFIG
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(
                (step + 1) / config['warmup_steps'],
                1.0
            )
        )
        
        # 混合精度训练
        self.scaler = GradScaler(enabled=config['fp16'])
        
        # 训练状态
        self.step = 0
        self.best_eval_loss = float('inf')
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 优化内存使用
        self.optimize_memory()
        
    def optimize_memory(self):
        """优化内存使用"""
        # 使用梯度检查点
        if hasattr(self.model, 'transformer'):
            self.model.transformer.gradient_checkpointing_enable()
            
        # 优化专家网络内存使用
        if hasattr(self.model, 'experts'):
            for expert in self.model.experts:
                if hasattr(expert, 'optimize_memory'):
                    expert.optimize_memory()
                    
        # 设置梯度累积步数
        self.gradient_accumulation_steps = self.config['gradient_accumulation_steps']
        
    def monitor_memory(self):
        """监控GPU内存使用"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            self.logger.info(
                f"GPU内存使用: {memory_allocated:.2f}MB (已分配) / "
                f"{memory_reserved:.2f}MB (已保留)"
            )
            
    def monitor_experts(self, expert_weights: torch.Tensor):
        """监控专家使用情况"""
        expert_usage = expert_weights.mean(dim=0)
        for i, (name, config) in enumerate(self.config['EXPERT_CONFIG'].items()):
            self.logger.info(
                f"{config['name']} 使用率: {expert_usage[i]:.2%}"
            )
            
    def balance_expert_load(self, expert_weights: torch.Tensor):
        """平衡专家负载"""
        expert_usage = expert_weights.mean(dim=0)
        imbalance = torch.std(expert_usage)
        
        if imbalance > 0.1:
            self.logger.info(f"专家负载不均衡度: {imbalance:.4f}")
            if hasattr(self.model, 'router'):
                self.model.router.adjust_routing(expert_usage)
                
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练"""
        # 使用混合精度训练
        with torch.cuda.amp.autocast(enabled=self.config['fp16']):
            # 前向传播
            logits, extra_info = self.model(
                batch['input_ids'],
                batch['attention_mask'],
                training=True
            )
            
            # 计算损失
            loss, loss_dict = self.model.compute_loss(
                logits,
                batch['labels'],
                extra_info,
                batch.get('rewards')
            )
            
            # 缩放损失
            loss = loss / self.gradient_accumulation_steps
            
        # 反向传播
        self.scaler.scale(loss).backward()
        
        # 梯度累积
        if (self.step + 1) % self.gradient_accumulation_steps == 0:
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['max_grad_norm']
            )
            
            # 更新参数
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # 更新学习率
            self.scheduler.step()
            
            # 平衡专家负载
            self.balance_expert_load(extra_info['expert_weights'])
            
        # 监控专家使用情况
        if self.step % 100 == 0:
            self.monitor_experts(extra_info['expert_weights'])
            self.monitor_memory()
            
        return loss_dict
        
    def train(self, num_epochs: int = 1):
        """训练模型"""
        self.model.train()
        
        for epoch in range(num_epochs):
            self.logger.info(f"开始第 {epoch + 1} 轮训练")
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}"
            )
            
            for batch in progress_bar:
                # 训练步骤
                loss_dict = self.train_step(batch)
                
                # 更新进度条
                progress_bar.set_postfix(loss_dict)
                
                # 保存检查点
                if self.step % 1000 == 0:
                    self.save_checkpoint()
                    
                # 评估
                if self.eval_dataloader and self.step % 2000 == 0:
                    eval_loss = self.evaluate()
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self.save_checkpoint(is_best=True)
                        
                self.step += 1
                
                # 检查是否达到最大步数
                if self.step >= self.config['max_steps']:
                    self.logger.info("达到最大训练步数，停止训练")
                    return
                    
    def evaluate(self) -> float:
        """评估模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                # 前向传播
                logits, extra_info = self.model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    training=False
                )
                
                # 计算损失
                loss, _ = self.model.compute_loss(
                    logits,
                    batch['labels'],
                    extra_info
                )
                
                total_loss += loss.item()
                num_batches += 1
                
        avg_loss = total_loss / num_batches
        self.logger.info(f"评估损失: {avg_loss:.4f}")
        
        self.model.train()
        return avg_loss
        
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'best_eval_loss': self.best_eval_loss
        }
        
        # 创建检查点目录
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        
        # 保存检查点
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'],
            f'checkpoint_{self.step}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(
                self.config['checkpoint_dir'],
                'best_model.pt'
            )
            torch.save(checkpoint, best_path)
            
        # 清理旧检查点
        self._cleanup_old_checkpoints()
        
    def _cleanup_old_checkpoints(self):
        """清理旧检查点"""
        checkpoints = sorted([
            f for f in os.listdir(self.config['checkpoint_dir'])
            if f.startswith('checkpoint_') and f.endswith('.pt')
        ])
        
        if len(checkpoints) > self.config['save_total_limit']:
            for old_checkpoint in checkpoints[:-self.config['save_total_limit']]:
                os.remove(os.path.join(self.config['checkpoint_dir'], old_checkpoint)) 