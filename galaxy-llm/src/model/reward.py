import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class RewardModel:
    """奖励模型"""
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1
        ).to(device)
        
    def compute_reward(
        self,
        input_text: str,
        output_text: str,
        reference_text: str = None
    ) -> float:
        """计算奖励值"""
        # 准备输入
        inputs = self.tokenizer(
            input_text,
            output_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # 计算基础奖励
        with torch.no_grad():
            outputs = self.model(**inputs)
            base_reward = outputs.logits.squeeze().item()
        
        # 如果有参考文本，计算相似度奖励
        if reference_text:
            ref_inputs = self.tokenizer(
                input_text,
                reference_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                ref_outputs = self.model(**ref_inputs)
                ref_reward = ref_outputs.logits.squeeze().item()
                
            # 计算相似度奖励
            similarity_reward = F.cosine_similarity(
                outputs.hidden_states[-1],
                ref_outputs.hidden_states[-1],
                dim=1
            ).mean().item()
            
            # 组合奖励
            reward = 0.7 * base_reward + 0.3 * similarity_reward
        else:
            reward = base_reward
            
        return reward
    
    def compute_batch_rewards(
        self,
        input_texts: List[str],
        output_texts: List[str],
        reference_texts: List[str] = None
    ) -> torch.Tensor:
        """批量计算奖励值"""
        rewards = []
        for i in range(len(input_texts)):
            reward = self.compute_reward(
                input_texts[i],
                output_texts[i],
                reference_texts[i] if reference_texts else None
            )
            rewards.append(reward)
        return torch.tensor(rewards, device=self.device)
    
    def save(self, path: str):
        """保存模型"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load(self, path: str):
        """加载模型"""
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path) 