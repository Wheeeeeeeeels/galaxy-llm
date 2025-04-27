import json
import torch
from typing import Dict, List, Optional
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from ..config.model_config import MODEL_CONFIG, DATA_CONFIG

class GalaxyDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        max_seq_length: int,
        use_cot: bool = False
    ):
        """
        数据集类
        
        Args:
            data: 数据列表
            tokenizer: 分词器
            max_seq_length: 最大序列长度
            use_cot: 是否使用思维链
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.use_cot = use_cot
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # 构建输入文本
        if self.use_cot:
            # 使用思维链
            input_text = f"问题：{item['question']}\n思考：{item['cot']}\n答案："
        else:
            # 不使用思维链
            input_text = f"问题：{item['question']}\n答案："
            
        # 构建目标文本
        target_text = item['answer']
        
        # 编码输入
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 编码目标
        targets = self.tokenizer(
            target_text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def create_train_dataset(
    file_path: str,
    max_seq_length: int,
    max_samples: Optional[int] = None,
    use_cot: bool = False
) -> GalaxyDataset:
    """创建训练数据集"""
    # 加载数据
    data = load_jsonl(file_path)
    if max_samples:
        data = data[:max_samples]
        
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    
    return GalaxyDataset(
        data=data,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        use_cot=use_cot
    )

def create_eval_dataset(
    file_path: str,
    max_seq_length: int,
    max_samples: Optional[int] = None,
    use_cot: bool = False
) -> GalaxyDataset:
    """创建评估数据集"""
    return create_train_dataset(
        file_path=file_path,
        max_seq_length=max_seq_length,
        max_samples=max_samples,
        use_cot=use_cot
    )

def create_test_dataset(
    file_path: str,
    max_seq_length: int,
    max_samples: Optional[int] = None,
    use_cot: bool = False
) -> GalaxyDataset:
    """创建测试数据集"""
    return create_train_dataset(
        file_path=file_path,
        max_seq_length=max_seq_length,
        max_samples=max_samples,
        use_cot=use_cot
    )

def load_school_info(file_path: str) -> Dict:
    """加载学校信息"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
        
def load_education_policy(file_path: str) -> Dict:
    """加载教育政策"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
        
def create_school_comparison_dataset(
    school_info: Dict,
    max_seq_length: int
) -> GalaxyDataset:
    """创建学校比较数据集"""
    data = []
    schools = list(school_info.keys())
    
    # 生成学校比较样本
    for i in range(len(schools)):
        for j in range(i + 1, len(schools)):
            school1 = schools[i]
            school2 = schools[j]
            
            # 构建问题
            question = f"请比较{school1}和{school2}的优劣"
            
            # 构建思维链
            cot = (
                f"1. 比较{school1}和{school2}的学术水平\n"
                f"2. 比较{school1}和{school2}的师资力量\n"
                f"3. 比较{school1}和{school2}的校园环境\n"
                f"4. 比较{school1}和{school2}的就业前景"
            )
            
            # 构建答案
            answer = (
                f"{school1}和{school2}的比较：\n"
                f"1. 学术水平：{school_info[school1]['academic']} vs {school_info[school2]['academic']}\n"
                f"2. 师资力量：{school_info[school1]['faculty']} vs {school_info[school2]['faculty']}\n"
                f"3. 校园环境：{school_info[school1]['campus']} vs {school_info[school2]['campus']}\n"
                f"4. 就业前景：{school_info[school1]['employment']} vs {school_info[school2]['employment']}"
            )
            
            data.append({
                'question': question,
                'cot': cot,
                'answer': answer
            })
            
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    
    return GalaxyDataset(
        data=data,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        use_cot=True
    )
        
def create_school_recommendation_dataset(
    school_info: Dict,
    max_seq_length: int
) -> GalaxyDataset:
    """创建学校推荐数据集"""
    data = []
    
    # 生成学校推荐样本
    for school in school_info:
        # 构建问题
        question = f"请推荐适合{school}的学生类型"
        
        # 构建思维链
        cot = (
            f"1. 分析{school}的特点\n"
            f"2. 分析{school}的优势专业\n"
            f"3. 分析{school}的招生要求\n"
            f"4. 总结适合的学生类型"
        )
        
        # 构建答案
        answer = (
            f"{school}适合的学生类型：\n"
            f"1. 学术特点：{school_info[school]['academic']}\n"
            f"2. 优势专业：{school_info[school]['majors']}\n"
            f"3. 招生要求：{school_info[school]['requirements']}\n"
            f"4. 适合学生：{school_info[school]['suitable_students']}"
        )
        
        data.append({
            'question': question,
            'cot': cot,
            'answer': answer
        })
        
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    
    return GalaxyDataset(
        data=data,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        use_cot=True
    )

class TestDataset(Dataset):
    """测试数据集"""
    
    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 512,
        max_samples: Optional[int] = None,
        use_cot: bool = False
    ):
        """
        初始化测试数据集
        
        Args:
            file_path: 文件路径
            tokenizer: 分词器
            max_seq_length: 最大序列长度
            max_samples: 最大样本数
            use_cot: 是否使用思维链
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.use_cot = use_cot
        
        # 加载数据
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # 限制样本数
        if max_samples is not None:
            self.data = self.data[:max_samples]
            
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取样本
        
        Args:
            idx: 样本索引
            
        Returns:
            样本字典
        """
        # 获取样本
        sample = self.data[idx]
        
        # 构建输入文本
        if self.use_cot:
            input_text = f"问题：{sample['question']}\n思维链：{sample['cot']}\n答案："
        else:
            input_text = f"问题：{sample['question']}\n答案："
            
        # 编码输入文本
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 获取标签
        label = sample['label']
        
        # 返回样本
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_test_dataset(
    file_path: str,
    tokenizer: AutoTokenizer,
    max_seq_length: int = 512,
    max_samples: Optional[int] = None,
    use_cot: bool = False
) -> TestDataset:
    """
    创建测试数据集
    
    Args:
        file_path: 文件路径
        tokenizer: 分词器
        max_seq_length: 最大序列长度
        max_samples: 最大样本数
        use_cot: 是否使用思维链
        
    Returns:
        测试数据集
    """
    return TestDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        max_samples=max_samples,
        use_cot=use_cot
    ) 