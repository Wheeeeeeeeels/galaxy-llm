import torch
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import deepspeed
import os

from src.model.moe_model import MoEModel
from src.model.config import ModelConfig, MoEConfig, ExpertConfig
from src.data.data_loader import get_dataloaders

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizer:
    """加载分词器
    
    Args:
        tokenizer_path: 分词器路径
        
    Returns:
        分词器
    """
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(tokenizer_path, "tokenizer.json"),
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )

class MoETrainer:
    """MoE模型训练器"""
    def __init__(
        self,
        model: MoEModel,
        tokenizer: PreTrainedTokenizer,
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
        local_rank: int = -1,
        gradient_accumulation_steps: int = 4  # 添加梯度累积步数
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
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
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
        
    def _get_model(self):
        """获取模型，处理DDP情况"""
        if self.use_ddp and isinstance(self.model, DDP):
            return self.model.module
        return self.model
        
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_ce_loss = 0
        total_reg_loss = 0
        total_sparse_loss = 0
        total_diversity_loss = 0
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        self.optimizer.zero_grad()  # 在epoch开始时清零梯度
        
        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移到设备上
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # 使用混合精度训练
            with autocast(enabled=self.use_amp):
                # 计算损失
                loss_dict = self._get_model().compute_loss(input_ids, attention_mask, labels)
                loss = loss_dict["total_loss"] / self.gradient_accumulation_steps  # 缩放损失
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
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
                loss_dict = self._get_model().compute_loss(input_ids, attention_mask, labels)
            
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
            "model_state_dict": self._get_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """加载检查点"""
        checkpoint = torch.load(self.save_dir / filename)
        self._get_model().load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.use_amp and checkpoint["scaler_state_dict"] is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

def main(
    num_epochs: int = 10,
    batch_size: int = 1,  # 减小批次大小
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    save_dir: str = "checkpoints",
    use_amp: bool = True,
    use_ddp: bool = False,  # 暂时禁用DDP
    local_rank: int = -1,
    gradient_accumulation_steps: int = 8  # 增加梯度累积步数
):
    """训练主函数"""
    # 设置CUDA内存分配策略
    torch.cuda.set_per_process_memory_fraction(0.8)  # 限制GPU内存使用
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    torch.backends.cudnn.benchmark = False  # 禁用cudnn基准测试
    torch.backends.cudnn.deterministic = True  # 使用确定性算法
    
    # 加载分词器
    tokenizer = load_tokenizer("tokenizer")
    
    # 获取数据加载器
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        data_dir="data",
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=256  # 减小序列长度
    )
    
    # 创建模型配置
    expert_config = ExpertConfig(
        vocab_size=50000,  # 使用分词器的词表大小
        hidden_size=512,  # 使用原始隐藏层大小
        num_layers=6,  # 使用原始层数
        num_attention_heads=8,  # 使用原始注意力头数
        intermediate_size=2048,  # 使用原始中间层大小
        dropout=0.1,
        max_length=512  # 设置最大序列长度
    )
    
    moe_config = MoEConfig(
        num_experts=4,  # 使用原始专家数量
        expert_config=expert_config,
        routing_type="learned",
        top_k=2,
        regularization_weight=0.01,
        sparsity_weight=0.01,
        diversity_weight=0.01
    )
    
    model_config = ModelConfig(
        moe_config=moe_config,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # 初始化模型
    model = MoEModel(model_config)
    
    # 初始化训练器
    trainer = MoETrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        num_epochs=num_epochs,
        save_dir=save_dir,
        use_amp=use_amp,
        use_ddp=use_ddp,
        local_rank=local_rank,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main() 