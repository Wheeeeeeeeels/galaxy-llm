import os
import torch
import argparse
from typing import Dict, Any, List
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from tqdm import tqdm

from .config.model_config import MODEL_CONFIG, DATA_CONFIG
from .data.dataset import create_test_dataset
from .utils.metrics import compute_metrics
from .utils.logger import (
    setup_logger,
    log_metrics,
    log_config,
    log_memory_usage
)

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    
    # 模型参数
    parser.add_argument(
        '--model_name',
        type=str,
        default='bert-base-chinese',
        help='模型名称'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='模型路径'
    )
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=512,
        help='最大序列长度'
    )
    
    # 数据参数
    parser.add_argument(
        '--test_file',
        type=str,
        required=True,
        help='测试文件路径'
    )
    parser.add_argument(
        '--max_test_samples',
        type=int,
        default=None,
        help='最大测试样本数'
    )
    
    # 其他参数
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='输出目录'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='日志文件路径'
    )
    parser.add_argument(
        '--use_cot',
        action='store_true',
        help='是否使用思维链'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='设备'
    )
    
    return parser.parse_args()

def inference(
    model: AutoModelForCausalLM,
    test_loader: DataLoader,
    args: argparse.Namespace,
    logger: Any
) -> Dict[str, float]:
    """
    推理模型
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        args: 参数
        logger: 日志记录器
        
    Returns:
        评估指标
    """
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inferencing"):
            # 将数据移到设备
            batch = {k: v.to(args.device) for k, v in batch.items()}
            
            # 前向传播
            outputs = model(**batch)
            
            # 获取预测结果
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().numpy().tolist())
            labels.extend(batch['labels'].cpu().numpy().tolist())
            
    # 计算指标
    metrics = compute_metrics(
        predictions=predictions,
        labels=labels,
        metrics=['accuracy', 'f1']
    )
    
    return metrics

def save_predictions(
    predictions: List[int],
    output_dir: str
):
    """
    保存预测结果
    
    Args:
        predictions: 预测结果
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存预测结果
    with open(os.path.join(output_dir, 'predictions.txt'), 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置日志
    logger = setup_logger(args.log_file)
    log_config(vars(args), logger=logger)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 创建数据集
    test_dataset = create_test_dataset(
        file_path=args.test_file,
        max_seq_length=args.max_seq_length,
        max_samples=args.max_test_samples,
        use_cot=args.use_cot
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.to(args.device)
    
    # 推理模型
    metrics = inference(
        model=model,
        test_loader=test_loader,
        args=args,
        logger=logger
    )
    
    # 记录指标
    log_metrics(metrics, prefix='test_', logger=logger)
    
    # 记录内存使用情况
    if torch.cuda.is_available():
        log_memory_usage(
            memory_allocated=torch.cuda.memory_allocated() / 1024 / 1024,
            memory_reserved=torch.cuda.memory_reserved() / 1024 / 1024,
            logger=logger
        )

if __name__ == '__main__':
    main() 