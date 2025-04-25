import os
import json
import logging
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
from model.moe_model import MoEModel
from model.config import COT_MODEL_CONFIG, DIRECT_MODEL_CONFIG
from model.tokenizer import ChineseTokenizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

class ModelInference:
    """模型推理"""
    def __init__(
        self,
        model_type: str = "cot",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        self.model_type = model_type
        
        # 初始化分词器
        self.tokenizer = ChineseTokenizer()
        
        # 加载模型
        self.model = self._load_model()
        
    def _load_model(self) -> MoEModel:
        """加载模型"""
        if self.model_type == "cot":
            model = MoEModel(COT_MODEL_CONFIG).to(self.device)
            model.load_state_dict(torch.load("checkpoints/cot_model/best_model_cot.pt"))
        else:
            model = MoEModel(DIRECT_MODEL_CONFIG).to(self.device)
            model.load_state_dict(torch.load("checkpoints/direct_model/best_model_direct.pt"))
            
        model.eval()
        return model
    
    def generate(
        self,
        instruction: str,
        input_text: str = "",
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> List[str]:
        """生成回答"""
        # 构建输入文本
        prompt = f"指令：{instruction}\n输入：{input_text}"
        
        # 编码输入
        encoded = self.tokenizer.encode(prompt)
        input_ids = encoded["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = encoded["attention_mask"].unsqueeze(0).to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature
            )
            
        # 解码输出
        generated_texts = self.tokenizer.batch_decode(outputs.tolist())
        
        return generated_texts
    
    def batch_generate(
        self,
        instructions: List[str],
        input_texts: List[str] = None,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> List[List[str]]:
        """批量生成回答"""
        if input_texts is None:
            input_texts = [""] * len(instructions)
            
        all_generated_texts = []
        for instruction, input_text in zip(instructions, input_texts):
            generated_texts = self.generate(
                instruction=instruction,
                input_text=input_text,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences
            )
            all_generated_texts.append(generated_texts)
            
        return all_generated_texts

def main():
    # 创建推理实例
    cot_inference = ModelInference(model_type="cot")
    direct_inference = ModelInference(model_type="direct")
    
    # 测试样例
    test_cases = [
        {
            "instruction": "请介绍香港的教育体系",
            "input": ""
        },
        {
            "instruction": "比较香港和内地的高考制度",
            "input": ""
        },
        {
            "instruction": "推荐几所香港的大学",
            "input": ""
        }
    ]
    
    # 生成回答
    logger.info("使用思维链模型生成回答...")
    cot_answers = cot_inference.batch_generate(
        instructions=[case["instruction"] for case in test_cases],
        input_texts=[case["input"] for case in test_cases]
    )
    
    logger.info("使用直接回答模型生成回答...")
    direct_answers = direct_inference.batch_generate(
        instructions=[case["instruction"] for case in test_cases],
        input_texts=[case["input"] for case in test_cases]
    )
    
    # 保存结果
    results = []
    for i, case in enumerate(test_cases):
        results.append({
            "instruction": case["instruction"],
            "input": case["input"],
            "cot_answer": cot_answers[i][0],
            "direct_answer": direct_answers[i][0]
        })
        
    with open("inference_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("推理结果已保存到 inference_results.json")

if __name__ == "__main__":
    main() 