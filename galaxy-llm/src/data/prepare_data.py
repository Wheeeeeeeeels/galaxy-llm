import os
import json
import logging
import random
from typing import List, Dict
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_processed_data(file_path: str) -> List[Dict]:
    """加载已处理的数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    continue
    return data

def load_json_data(file_path: str) -> List[Dict]:
    """加载JSON数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def merge_data(processed_data: List[Dict], json_data: List[Dict]) -> List[Dict]:
    """合并数据"""
    # 使用集合去重
    seen = set()
    merged_data = []
    
    # 添加已处理的数据
    for item in processed_data:
        key = f"{item.get('instruction', '')}_{item.get('input', '')}"
        if key not in seen:
            seen.add(key)
            merged_data.append(item)
    
    # 添加JSON数据
    for item in json_data:
        key = f"{item.get('instruction', '')}_{item.get('input', '')}"
        if key not in seen:
            seen.add(key)
            merged_data.append(item)
            
    return merged_data

def split_data(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, List[Dict]]:
    """划分数据集"""
    # 随机打乱数据
    random.shuffle(data)
    
    # 计算划分点
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # 划分数据集
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

def process_data(data: List[Dict], use_cot: bool = True) -> List[Dict]:
    """处理数据"""
    processed_data = []
    
    for item in tqdm(data, desc="处理数据"):
        # 基础信息
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        if use_cot:
            # 添加思维链
            cot_output = f"""让我思考一下这个问题：

1. 首先，我需要理解问题的关键点：
{instruction}

2. 然后，我需要分析相关信息：
{input_text}

3. 最后，我可以给出答案：
{output}"""
            
            processed_item = {
                "instruction": instruction,
                "input": input_text,
                "output": cot_output
            }
        else:
            # 直接回答
            processed_item = {
                "instruction": instruction,
                "input": input_text,
                "output": output
            }
            
        processed_data.append(processed_item)
        
    return processed_data

def save_data(data: Dict[str, List[Dict]], output_dir: str):
    """保存数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    for split, split_data in data.items():
        output_path = os.path.join(output_dir, f"{split}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        logger.info(f"保存{split}数据到 {output_path}")

def main():
    # 加载已处理的数据
    processed_data = load_processed_data("data/学校清洗数据.txt")
    logger.info(f"加载了 {len(processed_data)} 条已处理数据")
    
    # 加载JSON数据
    json_data = load_json_data("data/学校纯文本数据.json")
    logger.info(f"加载了 {len(json_data)} 条JSON数据")
    
    # 合并数据
    merged_data = merge_data(processed_data, json_data)
    logger.info(f"合并后共有 {len(merged_data)} 条数据")
    
    # 划分数据集
    split_data_dict = split_data(merged_data)
    logger.info(f"数据集划分完成：")
    logger.info(f"训练集：{len(split_data_dict['train'])} 条")
    logger.info(f"验证集：{len(split_data_dict['val'])} 条")
    logger.info(f"测试集：{len(split_data_dict['test'])} 条")
    
    # 处理思维链数据
    logger.info("处理思维链数据...")
    cot_data = {
        split: process_data(data, use_cot=True)
        for split, data in split_data_dict.items()
    }
    save_data(cot_data, "data/training/cot")
    
    # 处理直接回答数据
    logger.info("处理直接回答数据...")
    direct_data = {
        split: process_data(data, use_cot=False)
        for split, data in split_data_dict.items()
    }
    save_data(direct_data, "data/training/direct")
    
    logger.info("数据准备完成！")

if __name__ == "__main__":
    main() 