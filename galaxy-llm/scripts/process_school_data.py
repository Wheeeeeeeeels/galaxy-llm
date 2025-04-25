import sys
import os
from pathlib import Path
from src.data.convert_school_data import convert_school_data
from src.data.augmentation import DataAugmentor
from src.data.online_llm import online_llm_streaming
import json
import logging
import time
import numpy as np
from typing import List, Dict, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SchoolDataProcessor:
    def __init__(self, llm_api_key: str = None):
        self.augmentor = DataAugmentor()
        self.llm = None
        if llm_api_key:
            self.llm = online_llm_streaming("", api_key=llm_api_key)
            
    def process_school_data(self, input_file: str, output_dir: str):
        """
        处理学校数据，包括转换和增强
        
        Args:
            input_file: 输入文件路径
            output_dir: 输出目录路径
        """
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 转换数据
        logger.info("开始转换数据...")
        convert_school_data(input_file, output_dir)
        
        # 2. 数据增强
        logger.info("开始数据增强...")
        
        # 处理每个数据集
        for dataset in ['train', 'eval', 'test']:
            input_file = output_dir / f'{dataset}.json'
            output_file = output_dir / f'{dataset}_augmented.json'
            grpo_file = output_dir / f'{dataset}_grpo.json'
            
            if not input_file.exists():
                logger.warning(f"跳过不存在的文件: {input_file}")
                continue
                
            # 读取数据
            with open(input_file, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
                
            # 增强数据
            augmented_data = []
            grpo_data = []
            
            for item in data:
                # 使用传统方法增强
                augmented_items = self.augmentor.augment_school_data(item)
                augmented_data.extend(augmented_items)
                
                # 使用LLM增强
                if self.llm:
                    try:
                        llm_augmented = self._llm_augment(item)
                        augmented_data.extend(llm_augmented)
                    except Exception as e:
                        logger.error(f"LLM增强失败: {e}")
                        continue
                        
                # 生成GRPO训练数据
                grpo_items = self._generate_grpo_data(item)
                grpo_data.extend(grpo_items)
                    
            # 保存增强后的数据
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in augmented_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    
            # 保存GRPO数据
            with open(grpo_file, 'w', encoding='utf-8') as f:
                for item in grpo_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    
            logger.info(f"{dataset}数据集处理完成：")
            logger.info(f"- 原始数据：{len(data)} 条")
            logger.info(f"- 增强后数据：{len(augmented_data)} 条")
            logger.info(f"- GRPO数据：{len(grpo_data)} 条")
            
        logger.info("数据处理完成！")
        
    def _llm_augment(self, item: dict) -> list:
        """使用LLM进行数据增强"""
        augmented_items = []
        
        # 构建提示词
        prompt = f"""请基于以下学校信息生成3个不同的问答对：
学校信息：{json.dumps(item, ensure_ascii=False, indent=2)}

要求：
1. 问答对要多样化，包括基本信息、课程、招生、师资等方面
2. 问题和答案要自然流畅
3. 保持信息的准确性
4. 输出格式为JSON数组，每个元素包含question、answer和cot字段

请直接输出JSON数组，不要有其他文字。"""
        
        # 调用LLM
        self.llm.inputs = {"input_query": prompt}
        response = self.llm.run()
        
        try:
            # 解析响应
            augmented_pairs = json.loads(response)
            for pair in augmented_pairs:
                augmented_items.append({
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "cot": pair["cot"]
                })
        except json.JSONDecodeError as e:
            logger.error(f"解析LLM响应失败: {e}")
            
        return augmented_items
        
    def _generate_grpo_data(self, item: dict) -> List[Dict]:
        """生成GRPO训练数据
        
        GRPO数据格式：
        {
            "state": 当前状态（学校信息）,
            "action": 采取的动作（如：查询某个具体信息）,
            "reward": 奖励值（根据动作的合理性和信息价值）,
            "next_state": 下一个状态（更新后的学校信息）,
            "done": 是否结束
        }
        """
        grpo_items = []
        
        # 1. 基本信息查询
        basic_info_actions = [
            ("查询学校名称", "name", 0.8),
            ("查询学校地址", "location", 0.8),
            ("查询学校类型", "type", 0.8),
            ("查询学校层次", "level", 0.8)
        ]
        
        for action_name, field, reward in basic_info_actions:
            if field in item:
                grpo_items.append({
                    "state": item,
                    "action": action_name,
                    "reward": reward,
                    "next_state": item,
                    "done": False
                })
                
        # 2. 课程信息查询
        if "courses" in item:
            grpo_items.append({
                "state": item,
                "action": "查询学校课程",
                "reward": 0.9,
                "next_state": item,
                "done": False
            })
            
        # 3. 招生信息查询
        if "admission" in item:
            grpo_items.append({
                "state": item,
                "action": "查询招生政策",
                "reward": 0.9,
                "next_state": item,
                "done": False
            })
            
        # 4. 师资信息查询
        if "faculty" in item:
            grpo_items.append({
                "state": item,
                "action": "查询师资力量",
                "reward": 0.9,
                "next_state": item,
                "done": False
            })
            
        # 5. 学校比较
        grpo_items.append({
            "state": item,
            "action": "与其他学校比较",
            "reward": 1.0,
            "next_state": item,
            "done": True
        })
        
        return grpo_items

if __name__ == "__main__":
    # 获取命令行参数
    if len(sys.argv) < 3:
        print("Usage: python process_school_data.py <input_file> <output_dir> [llm_api_key]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    llm_api_key = sys.argv[3] if len(sys.argv) > 3 else None
    
    # 处理数据
    processor = SchoolDataProcessor(llm_api_key=llm_api_key)
    processor.process_school_data(input_file, output_dir) 