import os
import json
import logging
import torch
from torch.utils.data import DataLoader
from model.moe_model import MoEModel
from model.config import COT_MODEL_CONFIG, DIRECT_MODEL_CONFIG
from model.tokenizer import ChineseTokenizer
from train import SchoolDataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_model(
    model: MoEModel,
    eval_loader: DataLoader,
    device: torch.device,
    tokenizer: ChineseTokenizer
) -> Dict[str, float]:
    """评估模型"""
    model.eval()
    total_loss = 0
    total_samples = 0
    correct_predictions = 0
    
    # 评估指标
    metrics = {
        "loss": 0.0,
        "accuracy": 0.0,
        "perplexity": 0.0
    }
    
    with torch.no_grad():
        for batch in eval_loader:
            # 将数据移到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 计算损失
            loss_dict = model.compute_loss(input_ids, attention_mask, labels)
            loss = loss_dict['total_loss']
            
            # 计算准确率
            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == labels).sum().item()
            correct_predictions += correct
            total_samples += labels.numel()
            
            # 计算困惑度
            perplexity = torch.exp(loss)
            
            total_loss += loss.item()
            
    # 计算平均指标
    metrics["loss"] = total_loss / len(eval_loader)
    metrics["accuracy"] = correct_predictions / total_samples
    metrics["perplexity"] = torch.exp(torch.tensor(metrics["loss"])).item()
    
    return metrics

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 初始化分词器
    tokenizer = ChineseTokenizer()
    
    # 创建数据集
    eval_dataset = SchoolDataset("data/training/eval.json", tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=8)
    
    # 评估思维链模型
    logger.info("评估思维链模型...")
    cot_model = MoEModel(COT_MODEL_CONFIG).to(device)
    cot_model.load_state_dict(torch.load("checkpoints/cot_model/best_model_cot.pt"))
    cot_metrics = evaluate_model(cot_model, eval_loader, device, tokenizer)
    logger.info(f"思维链模型评估结果: {cot_metrics}")
    
    # 评估直接回答模型
    logger.info("评估直接回答模型...")
    direct_model = MoEModel(DIRECT_MODEL_CONFIG).to(device)
    direct_model.load_state_dict(torch.load("checkpoints/direct_model/best_model_direct.pt"))
    direct_metrics = evaluate_model(direct_model, eval_loader, device, tokenizer)
    logger.info(f"直接回答模型评估结果: {direct_metrics}")
    
    # 保存评估结果
    results = {
        "cot_model": cot_metrics,
        "direct_model": direct_metrics
    }
    
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("评估结果已保存到 evaluation_results.json")

if __name__ == "__main__":
    main() 