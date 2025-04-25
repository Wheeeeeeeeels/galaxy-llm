import torch
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
from transformers import AutoTokenizer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from model.moe_model import MoEModel
from model.config import ModelConfig, MoEConfig, ExpertConfig
from data.data_loader import get_dataloaders

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MoETrainer:
    """MoE模型训练器"""
    def __init__(
        self,
        model: MoEModel,
        tokenizer: AutoTokenizer,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        num_epochs: int = 10,
        save_dir: str = "checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = True,
        use_ddp: bool = True,
        local_rank: int = -1
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.use_amp = use_amp
        self.use_ddp = use_ddp
        self.local_rank = local_rank
        
        # 初始化分布式训练
        if use_ddp and local_rank != -1:
            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True
            )
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )
        
        # 混合精度训练
        self.scaler = GradScaler() if use_amp else None
        
        self.max_grad_norm = max_grad_norm
        
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_ce_loss = 0
        total_reg_loss = 0
        total_sparse_loss = 0
        total_diversity_loss = 0
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        for batch in progress_bar:
            # 将数据移到设备上
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # 使用混合精度训练
            with autocast(enabled=self.use_amp):
                # 计算损失
                loss_dict = self.model.compute_loss(input_ids, attention_mask, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss_dict["total_loss"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()
            
            # 累计损失
            total_loss += loss_dict["total_loss"].item()
            total_ce_loss += loss_dict["ce_loss"].item()
            total_reg_loss += loss_dict["reg_loss"].item()
            total_sparse_loss += loss_dict["sparse_loss"].item()
            total_diversity_loss += loss_dict["diversity_loss"].item()
            
            # 更新进度条
            progress_bar.set_postfix({
                "loss": loss_dict["total_loss"].item(),
                "ce_loss": loss_dict["ce_loss"].item()
            })
            
        # 计算平均损失
        num_batches = len(self.train_dataloader)
        avg_loss = total_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        avg_reg_loss = total_reg_loss / num_batches
        avg_sparse_loss = total_sparse_loss / num_batches
        avg_diversity_loss = total_diversity_loss / num_batches
        
        return {
            "loss": avg_loss,
            "ce_loss": avg_ce_loss,
            "reg_loss": avg_reg_loss,
            "sparse_loss": avg_sparse_loss,
            "diversity_loss": avg_diversity_loss
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_ce_loss = 0
        
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch in progress_bar:
            # 将数据移到设备上
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # 使用混合精度评估
            with autocast(enabled=self.use_amp):
                # 计算损失
                loss_dict = self.model.compute_loss(input_ids, attention_mask, labels)
            
            # 累计损失
            total_loss += loss_dict["total_loss"].item()
            total_ce_loss += loss_dict["ce_loss"].item()
            
            # 更新进度条
            progress_bar.set_postfix({
                "loss": loss_dict["total_loss"].item(),
                "ce_loss": loss_dict["ce_loss"].item()
            })
            
        # 计算平均损失
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        
        return {
            "loss": avg_loss,
            "ce_loss": avg_ce_loss
        }
    
    def train(self):
        """训练模型"""
        best_val_loss = float("inf")
        
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # 训练
            train_metrics = self.train_epoch()
            logger.info(f"Training metrics: {train_metrics}")
            
            # 验证
            val_metrics = self.evaluate(self.val_dataloader)
            logger.info(f"Validation metrics: {val_metrics}")
            
            # 学习率调度
            self.scheduler.step()
            
            # 保存最佳模型
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self.save_checkpoint("best_model.pt")
                logger.info("Saved best model checkpoint")
            
            # 保存最新模型
            self.save_checkpoint(f"model_epoch_{epoch + 1}.pt")
            
        # 测试最佳模型
        self.load_checkpoint("best_model.pt")
        test_metrics = self.evaluate(self.test_dataloader)
        logger.info(f"Test metrics: {test_metrics}")
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        if self.use_ddp and self.local_rank != 0:
            return
            
        checkpoint = {
            "model_state_dict": self.model.module.state_dict() if self.use_ddp else self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """加载检查点"""
        checkpoint = torch.load(self.save_dir / filename)
        if self.use_ddp:
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.use_amp and checkpoint["scaler_state_dict"] is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

def main():
    # 配置
    config = ModelConfig(
        vocab_size=50000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        num_experts=8,
        expert_capacity=64,
        dropout=0.1,
        max_length=2048
    )
    
    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    
    # 获取数据加载器
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        data_dir="data/processed/split",
        tokenizer=tokenizer,
        batch_size=8,
        max_length=2048,
        with_chain_of_thought=True,
        max_turns=10
    )
    
    # 初始化模型
    model = MoEModel(config)
    
    # 初始化训练器
    trainer = MoETrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        learning_rate=1e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_epochs=10,
        save_dir="checkpoints",
        use_amp=True,
        use_ddp=True
    )
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main() 