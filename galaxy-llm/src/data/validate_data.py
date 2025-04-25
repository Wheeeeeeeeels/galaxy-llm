import json
import logging
from typing import Dict, List, Tuple
from pathlib import Path
import random
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.qa_pairs = []
        self.with_chain = []
        self.without_chain = []
        
    def load_data(self):
        """加载所有数据文件"""
        try:
            # 加载问答对
            with open(self.data_dir / "qa_pairs.json", "r", encoding="utf-8") as f:
                self.qa_pairs = json.load(f)
            logger.info(f"成功加载问答对数据: {len(self.qa_pairs)}条")
            
            # 加载带思维链的数据
            with open(self.data_dir / "with_chain_of_thought.json", "r", encoding="utf-8") as f:
                self.with_chain = json.load(f)
            logger.info(f"成功加载带思维链数据: {len(self.with_chain)}条")
            
            # 加载不带思维链的数据
            with open(self.data_dir / "without_chain_of_thought.json", "r", encoding="utf-8") as f:
                self.without_chain = json.load(f)
            logger.info(f"成功加载不带思维链数据: {len(self.without_chain)}条")
            
        except Exception as e:
            logger.error(f"加载数据时出错: {str(e)}")
            raise
    
    def validate_qa_pairs(self) -> Tuple[int, List[Dict]]:
        """验证问答对数据"""
        valid_pairs = []
        invalid_count = 0
        
        for pair in tqdm(self.qa_pairs, desc="验证问答对"):
            # 检查必要字段
            if not all(k in pair for k in ["question", "answer", "type"]):
                invalid_count += 1
                continue
                
            # 检查问题长度
            if len(pair["question"]) < 5 or len(pair["question"]) > 500:
                invalid_count += 1
                continue
                
            # 检查答案长度
            if len(pair["answer"]) < 10 or len(pair["answer"]) > 2000:
                invalid_count += 1
                continue
                
            # 检查问题类型
            if pair["type"] not in ["基础信息查询", "学校比较", "学校推荐"]:
                invalid_count += 1
                continue
                
            valid_pairs.append(pair)
            
        return invalid_count, valid_pairs
    
    def validate_chain_of_thought(self) -> Tuple[int, List[Dict]]:
        """验证思维链数据"""
        valid_chains = []
        invalid_count = 0
        
        for chain in tqdm(self.with_chain, desc="验证思维链"):
            # 检查必要字段
            if not all(k in chain for k in ["question", "answer", "chain_of_thought"]):
                invalid_count += 1
                continue
                
            # 检查思维链类型
            if not isinstance(chain["chain_of_thought"], str):
                invalid_count += 1
                continue
                
            # 检查思维链长度
            if len(chain["chain_of_thought"]) < 20 or len(chain["chain_of_thought"]) > 1000:
                invalid_count += 1
                continue
                
            # 检查思维链格式
            if not chain["chain_of_thought"].startswith("思考过程："):
                invalid_count += 1
                continue
                
            valid_chains.append(chain)
            
        return invalid_count, valid_chains
    
    def split_data(self, valid_pairs: List[Dict], valid_chains: List[Dict]) -> Dict[str, List[Dict]]:
        """划分训练集、验证集和测试集"""
        # 合并所有数据
        all_data = valid_pairs + valid_chains
        random.shuffle(all_data)
        
        # 计算划分点
        total = len(all_data)
        train_size = int(total * 0.8)
        val_size = int(total * 0.1)
        
        # 划分数据
        train_data = all_data[:train_size]
        val_data = all_data[train_size:train_size+val_size]
        test_data = all_data[train_size+val_size:]
        
        return {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }
    
    def save_split_data(self, split_data: Dict[str, List[Dict]]):
        """保存划分后的数据"""
        output_dir = self.data_dir / "split"
        output_dir.mkdir(exist_ok=True)
        
        for split_name, data in split_data.items():
            output_file = output_dir / f"{split_name}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"保存{split_name}数据: {len(data)}条")
    
    def run(self):
        """运行完整的验证和预处理流程"""
        # 加载数据
        self.load_data()
        
        # 验证问答对
        invalid_qa, valid_qa = self.validate_qa_pairs()
        logger.info(f"问答对验证结果: 有效{len(valid_qa)}条, 无效{invalid_qa}条")
        
        # 验证思维链
        invalid_chain, valid_chain = self.validate_chain_of_thought()
        logger.info(f"思维链验证结果: 有效{len(valid_chain)}条, 无效{invalid_chain}条")
        
        # 划分数据
        split_data = self.split_data(valid_qa, valid_chain)
        
        # 保存划分后的数据
        self.save_split_data(split_data)
        
        # 输出统计信息
        logger.info("数据划分统计:")
        for split_name, data in split_data.items():
            logger.info(f"{split_name}: {len(data)}条")

def main():
    validator = DataValidator()
    validator.run()

if __name__ == "__main__":
    main() 