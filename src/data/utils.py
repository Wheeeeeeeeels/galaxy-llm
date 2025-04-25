import re
from typing import List, Dict, Any
import pandas as pd
import numpy as np

class DataUtils:
    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本数据"""
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        return text.strip()
        
    @staticmethod
    def normalize_school_type(school_type: str) -> str:
        """标准化学校类型"""
        type_mapping = {
            '小学': 'primary',
            '中学': 'secondary',
            '高中': 'high',
            '大学': 'university',
            '国际学校': 'international'
        }
        return type_mapping.get(school_type, school_type)
        
    @staticmethod
    def extract_features(text: str) -> List[str]:
        """从文本中提取特征"""
        features = []
        # 提取关键词
        keywords = ['双语', '国际', '重点', '实验', '示范', '特色']
        for keyword in keywords:
            if keyword in text:
                features.append(keyword)
        return features
        
    @staticmethod
    def process_curriculum_data(df: pd.DataFrame) -> pd.DataFrame:
        """处理课程数据"""
        # 清理课程描述
        df['description'] = df['description'].apply(DataUtils.clean_text)
        
        # 标准化课程级别
        level_mapping = {
            '基础': 'basic',
            '进阶': 'advanced',
            '高级': 'expert'
        }
        df['level'] = df['level'].map(level_mapping)
        
        return df
        
    @staticmethod
    def process_admission_data(df: pd.DataFrame) -> pd.DataFrame:
        """处理招生数据"""
        # 转换日期格式
        df['deadline'] = pd.to_datetime(df['deadline'])
        
        # 清理要求文本
        df['requirements'] = df['requirements'].apply(DataUtils.clean_text)
        
        return df
        
    @staticmethod
    def create_school_embedding(school_data: Dict[str, Any]) -> np.ndarray:
        """创建学校特征向量"""
        # 提取数值特征
        features = []
        
        # 学校类型
        type_embedding = np.zeros(5)  # 5种学校类型
        type_idx = ['primary', 'secondary', 'high', 'university', 'international'].index(
            DataUtils.normalize_school_type(school_data['type'])
        )
        type_embedding[type_idx] = 1
        features.extend(type_embedding)
        
        # 特色数量
        features.append(len(school_data['features']))
        
        # 设施数量
        features.append(len(school_data['facilities']))
        
        # 课程数量
        features.append(len(school_data['curriculum']))
        
        return np.array(features)
        
    @staticmethod
    def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算两个学校特征向量的相似度"""
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
    @staticmethod
    def split_data(data: List[Dict], train_ratio: float = 0.8) -> tuple:
        """分割训练集和验证集"""
        np.random.shuffle(data)
        split_idx = int(len(data) * train_ratio)
        return data[:split_idx], data[split_idx:] 