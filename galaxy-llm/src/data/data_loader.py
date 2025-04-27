import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from transformers import PreTrainedTokenizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DialogueHistory:
    """对话历史管理"""
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: List[Dict] = []
        
    def add_turn(self, role: str, content: str):
        """添加一轮对话"""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns * 2:  # 每轮包含用户和助手
            self.history = self.history[-self.max_turns * 2:]
            
    def get_formatted_history(self) -> str:
        """获取格式化的对话历史"""
        formatted = ""
        for turn in self.history:
            formatted += f"{turn['role']}：{turn['content']}\n"
        return formatted.strip()
    
    def clear(self):
        """清空历史"""
        self.history = []

class SchoolQADataset(Dataset):
    """学校问答数据集"""
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,  # 增加最大长度
        with_chain_of_thought: bool = True,
        max_turns: int = 10
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_chain_of_thought = with_chain_of_thought
        self.max_turns = max_turns
        self.data = self._load_data()
        self.dialogue_history = DialogueHistory(max_turns)
        
    def _load_data(self) -> List[Dict]:
        """加载数据"""
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"加载数据: {len(data)}条")
        return data
        
    def _prepare_input(
        self,
        question: str,
        chain_of_thought: Optional[str] = None,
        history: Optional[List[Dict]] = None
    ) -> str:
        """准备输入文本"""
        # 构建对话历史
        if history:
            self.dialogue_history.clear()
            for turn in history:
                self.dialogue_history.add_turn(turn["role"], turn["content"])
        
        # 添加当前问题
        self.dialogue_history.add_turn("用户", question)
        
        # 构建完整输入
        input_text = self.dialogue_history.get_formatted_history()
        
        # 添加思维链
        if chain_of_thought and self.with_chain_of_thought:
            input_text += f"\n思考过程：{chain_of_thought}"
            
        return input_text
        
    def _prepare_output(self, answer: str) -> str:
        """准备输出文本"""
        self.dialogue_history.add_turn("助手", answer)
        return f"答案：{answer}"
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        item = self.data[idx]
        
        # 准备输入和输出文本
        input_text = self._prepare_input(
            item["question"],
            item.get("chain_of_thought"),
            item.get("history")
        )
        output_text = self._prepare_output(item["answer"])
        
        # 编码输入
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 编码输出
        output_encoding = self.tokenizer(
            output_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels": output_encoding["input_ids"].squeeze(0),
            "type": item["type"],
            "history": self.dialogue_history.history.copy()
        }

def collate_fn(batch):
    """自定义的collate函数，确保批次中的样本大小一致"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

class SchoolQADataLoader:
    """学校问答数据加载器"""
    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 8,
        max_length: int = 2048,
        with_chain_of_thought: bool = True,
        max_turns: int = 10,
        num_workers: int = 4
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.with_chain_of_thought = with_chain_of_thought
        self.max_turns = max_turns
        self.num_workers = num_workers
        
    def _create_dataloader(
        self,
        split: str,
        shuffle: bool = True
    ) -> DataLoader:
        """创建数据加载器"""
        data_path = self.data_dir / "processed" / "split" / f"{split}.json"
            
        dataset = SchoolQADataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            with_chain_of_thought=self.with_chain_of_thought,
            max_turns=self.max_turns
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn  # 添加自定义的collate_fn
        )
        
    def get_train_dataloader(self) -> DataLoader:
        """获取训练数据加载器"""
        return self._create_dataloader("train", shuffle=True)
        
    def get_val_dataloader(self) -> DataLoader:
        """获取验证数据加载器"""
        return self._create_dataloader("val", shuffle=False)
        
    def get_test_dataloader(self) -> DataLoader:
        """获取测试数据加载器"""
        return self._create_dataloader("test", shuffle=False)
        
def get_dataloaders(
    data_dir: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 2048,
    with_chain_of_thought: bool = True,
    max_turns: int = 10,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """获取所有数据加载器"""
    loader = SchoolQADataLoader(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        with_chain_of_thought=with_chain_of_thought,
        max_turns=max_turns,
        num_workers=num_workers
    )
    
    return (
        loader.get_train_dataloader(),
        loader.get_val_dataloader(),
        loader.get_test_dataloader()
    ) 