import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np

class GRPO:
    """Gradient-based Reward Policy Optimization"""
    def __init__(
        self,
        model: nn.Module,
        reward_fn: callable,
        learning_rate: float = 1e-5,
        beta: float = 0.1,
        gamma: float = 0.99,
        max_grad_norm: float = 1.0
    ):
        self.model = model
        self.reward_fn = reward_fn
        self.learning_rate = learning_rate
        self.beta = beta
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """计算优势函数"""
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae_lam = delta + self.gamma * self.beta * (1 - dones[t]) * last_gae_lam
            advantages[t] = last_gae_lam
            
        return advantages
    
    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """计算策略损失"""
        return -(log_probs * advantages).mean()
    
    def compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor
    ) -> torch.Tensor:
        """计算价值损失"""
        return F.mse_loss(values, returns)
    
    def compute_entropy_loss(
        self,
        log_probs: torch.Tensor
    ) -> torch.Tensor:
        """计算熵损失"""
        return -log_probs.mean()
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        old_log_probs: torch.Tensor
    ) -> Dict[str, float]:
        """更新策略"""
        # 计算当前策略的动作概率
        log_probs, values = self.model(states)
        
        # 计算优势函数
        advantages = self.compute_advantages(rewards, values, dones)
        returns = advantages + values
        
        # 计算各种损失
        policy_loss = self.compute_policy_loss(log_probs, advantages)
        value_loss = self.compute_value_loss(values, returns)
        entropy_loss = self.compute_entropy_loss(log_probs)
        
        # 总损失
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
        
        # 优化
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item()
        }
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取动作"""
        with torch.no_grad():
            log_probs, _ = self.model(state)
            if deterministic:
                action = torch.argmax(log_probs, dim=-1)
            else:
                action = torch.multinomial(F.softmax(log_probs, dim=-1), 1).squeeze(-1)
        return action, log_probs
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) 