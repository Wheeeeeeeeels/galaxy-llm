import os
import json
import logging
import torch
import deepspeed
from torch.utils.data import Dataset, DataLoader
from model.moe_model import MoEModel
from model.config import ModelConfig, COT_MODEL_CONFIG, DIRECT_MODEL_CONFIG
from model.grpo import GRPO
from model.reward import RewardModel
from model.tokenizer import ChineseTokenizer
from transformers import PreTrainedTokenizerFast

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# DeepSpeed配置
ds_config = {
    "train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-5,
            "warmup_num_steps": 1000
        }
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    }
}

class SchoolDataset(Dataset):
    """学校数据集"""
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建输入文本
        input_text = f"指令：{item['instruction']}\n输入：{item['input']}"
        output_text = item['output']
        
        # 编码输入和输出
        input_encoded = self.tokenizer(
            input_text,
            max_length=2048,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        output_encoded = self.tokenizer(
            output_text,
            max_length=2048,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': input_encoded['input_ids'].squeeze(0),
            'attention_mask': input_encoded['attention_mask'].squeeze(0),
            'labels': output_encoded['input_ids'].squeeze(0),
            'input_text': input_text,
            'output_text': output_text
        }

def train_model(
    model: MoEModel,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    save_dir: str,
    model_type: str
):
    """训练模型"""
    # 初始化DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters()
    )
    
    # 初始化奖励模型
    reward_model = RewardModel(device=device)
    
    # 初始化GRPO
    grpo = GRPO(
        model=model,
        reward_fn=reward_model.compute_reward,
        learning_rate=1e-5,
        beta=0.1,
        gamma=0.99
    )
    
    best_eval_loss = float('inf')
    patience = 3  # 早停耐心值
    no_improve_epochs = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model_engine.train()
        total_loss = 0
        for batch in train_loader:
            # 将数据移到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            input_texts = batch['input_text']
            output_texts = batch['output_text']
            
            # 计算损失
            loss_dict = model_engine.compute_loss(input_ids, attention_mask, labels)
            loss = loss_dict['total_loss']
            
            # 计算奖励
            rewards = reward_model.compute_batch_rewards(input_texts, output_texts)
            
            # 更新策略
            grpo.update(
                states=input_ids,
                actions=labels,
                rewards=rewards,
                dones=torch.zeros(len(rewards), device=device),
                old_log_probs=loss_dict['log_probs']
            )
            
            # 反向传播
            model_engine.backward(loss)
            model_engine.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
        
        # 评估阶段
        model_engine.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                loss_dict = model_engine.compute_loss(input_ids, attention_mask, labels)
                eval_loss += loss_dict['total_loss'].item()
                
        avg_eval_loss = eval_loss / len(eval_loader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Eval Loss: {avg_eval_loss:.4f}")
        
        # 保存最佳模型
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            no_improve_epochs = 0
            save_path = os.path.join(save_dir, f"best_model_{model_type}")
            model_engine.save_checkpoint(save_path)
            grpo.save(os.path.join(save_dir, f"grpo_model_{model_type}.pt"))
            reward_model.save(os.path.join(save_dir, f"reward_model_{model_type}"))
            logger.info(f"Saved best model to {save_path}")
        else:
            no_improve_epochs += 1
            
        # 早停
        if no_improve_epochs >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 初始化分词器
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="tokenizer/tokenizer.json",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )
    
    # 加载训练数据
    with open("data/training/train.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 创建数据集
    train_dataset = SchoolDataset("data/training/train.json", tokenizer)
    eval_dataset = SchoolDataset("data/training/eval.json", tokenizer)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=ds_config["train_batch_size"], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=ds_config["train_batch_size"])
    
    # 训练思维链模型
    logger.info("开始训练思维链模型...")
    cot_model = MoEModel(COT_MODEL_CONFIG).to(device)
    train_model(
        model=cot_model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
        num_epochs=10,
        save_dir="checkpoints/cot_model",
        model_type="cot"
    )
    
    # 训练直接回答模型
    logger.info("开始训练直接回答模型...")
    direct_model = MoEModel(DIRECT_MODEL_CONFIG).to(device)
    train_model(
        model=direct_model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
        num_epochs=10,
        save_dir="checkpoints/direct_model",
        model_type="direct"
    )

if __name__ == "__main__":
    main() 