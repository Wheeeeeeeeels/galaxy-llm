import os
import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from model.moe_model import MoEModel
from model.config import ModelConfig, COT_MODEL_CONFIG, DIRECT_MODEL_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SchoolDataset(Dataset):
    """学校数据集"""
    def __init__(self, data_path: str, max_length: int = 2048):
        self.max_length = max_length
        
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
        
        # 将文本转换为token ID
        input_ids = [ord(c) for c in input_text[:self.max_length]]  # 简单起见，使用ASCII编码
        output_ids = [ord(c) for c in output_text[:self.max_length]]
        
        # 填充或截断
        input_ids = input_ids + [0] * (self.max_length - len(input_ids))
        output_ids = output_ids + [0] * (self.max_length - len(output_ids))
        
        # 创建注意力掩码
        attention_mask = [1] * len(input_ids) + [0] * (self.max_length - len(input_ids))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
            'labels': torch.tensor(output_ids, dtype=torch.long)
        }

def train_model(
    model: MoEModel,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    save_dir: str
):
    """训练模型"""
    model.train()
    best_eval_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        for batch in train_loader:
            # 将数据移到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 计算损失
            loss_dict = model.compute_loss(input_ids, attention_mask, labels)
            loss = loss_dict['total_loss']
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
        
        # 评估阶段
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                loss_dict = model.compute_loss(input_ids, attention_mask, labels)
                eval_loss += loss_dict['total_loss'].item()
                
        avg_eval_loss = eval_loss / len(eval_loader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Eval Loss: {avg_eval_loss:.4f}")
        
        # 保存最佳模型
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            save_path = os.path.join(save_dir, f"best_model.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model to {save_path}")

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 创建数据集
    train_dataset = SchoolDataset("data/training/train.json")
    eval_dataset = SchoolDataset("data/training/eval.json")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=8)
    
    # 创建模型
    model = MoEModel(COT_MODEL_CONFIG).to(device)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # 创建保存目录
    save_dir = "checkpoints/cot_model"
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练模型
    train_model(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=10,
        save_dir=save_dir
    )

if __name__ == "__main__":
    main() 