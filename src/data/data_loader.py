import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from .augmentation import DataAugmentor
from .validator import DataValidator

class EducationDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 1024,
        use_augmentation: bool = False,
        augment_ratio: float = 0.3,
        use_validation: bool = True
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_augmentation = use_augmentation
        self.augment_ratio = augment_ratio
        self.use_validation = use_validation
        
        # 初始化数据增强器和验证器
        self.augmentor = DataAugmentor() if use_augmentation else None
        self.validator = DataValidator() if use_validation else None
        
        # 加载数据
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """加载训练数据"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 验证数据
        if self.use_validation:
            valid, errors = self.validator.validate_dataset(data)
            if not valid:
                logging.warning("数据集验证失败")
                for idx, error_list in errors.items():
                    logging.warning(f"数据 {idx} 的错误: {error_list}")
                    
        return data
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # 数据增强
        if self.use_augmentation and torch.rand(1).item() < self.augment_ratio:
            augmented_items = self.augmentor.augment_school_data(item)
            item = augmented_items[torch.randint(len(augmented_items), (1,)).item()]
        
        # 构建输入文本
        input_text = self._build_input_text(item)
        
        # 构建目标文本
        target_text = self._build_target_text(item)
        
        # 编码输入和目标
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }
        
    def _build_input_text(self, item: Dict) -> str:
        """构建输入文本"""
        school_info = item['school_info']
        curriculum = item['curriculum']
        admission = item['admission']
        
        input_text = f"学校名称：{school_info['name']}\n"
        input_text += f"学校类型：{school_info['type']}\n"
        input_text += f"地理位置：{school_info['location']}\n"
        input_text += f"特色：{', '.join(school_info['features'])}\n"
        input_text += f"设施：{', '.join(school_info['facilities'])}\n"
        
        input_text += "\n课程信息：\n"
        for course in curriculum:
            input_text += f"- {course['subject']} ({course['level']}): {course['description']}\n"
            
        input_text += "\n招生信息：\n"
        for info in admission:
            input_text += f"- {info['year']}年：\n"
            input_text += f"  要求：{info['requirements']}\n"
            input_text += f"  截止日期：{info['deadline']}\n"
            input_text += f"  流程：{info['process']}\n"
            
        return input_text
        
    def _build_target_text(self, item: Dict) -> str:
        """构建目标文本"""
        school_info = item['school_info']
        return f"这是一所{school_info['type']}学校，名为{school_info['name']}，位于{school_info['location']}。"

def create_data_loaders(
    train_path: str,
    val_path: str,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 1024,
    use_augmentation: bool = False,
    augment_ratio: float = 0.3,
    use_validation: bool = True,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""
    train_dataset = EducationDataset(
        train_path,
        tokenizer,
        max_length,
        use_augmentation=use_augmentation,
        augment_ratio=augment_ratio,
        use_validation=use_validation
    )
    
    val_dataset = EducationDataset(
        val_path,
        tokenizer,
        max_length,
        use_augmentation=False,  # 验证集不使用数据增强
        use_validation=use_validation
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader 