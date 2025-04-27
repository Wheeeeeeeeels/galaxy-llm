import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from .moe_model import MoEModel
from .config import ModelConfig

class RewardModel(nn.Module):
    """奖励模型"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 使用MoE模型作为基础
        self.base_model = MoEModel(config)
        
        # 奖励头
        self.reward_head = nn.Sequential(
            nn.Linear(config.moe_config.expert_config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """前向传播"""
        # 获取基础模型输出
        outputs = self.base_model(input_ids, attention_mask)
        
        # 计算奖励
        rewards = self.reward_head(outputs)
        
        return rewards

class PPOTrainer:
    """PPO训练器"""
    def __init__(
        self,
        model: MoEModel,
        reward_model: RewardModel,
        learning_rate: float = 1e-5,
        clip_range: float = 0.2,
        value_clip_range: float = 0.2,
        vf_coef: float = 0.1,
        max_grad_norm: float = 0.5
    ):
        self.model = model
        self.reward_model = reward_model
        self.learning_rate = learning_rate
        self.clip_range = clip_range
        self.value_clip_range = value_clip_range
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        # 优化器
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        lambda_: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算优势函数"""
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            last_gae_lam = delta + gamma * lambda_ * (1 - dones[t]) * last_gae_lam
            advantages[t] = last_gae_lam
            
        returns = advantages + values
        return advantages, returns
    
    def compute_loss(
        self,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """计算损失"""
        # 策略损失
        ratio = torch.exp(log_probs - old_log_probs)
        policy_loss1 = -advantages * ratio
        policy_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.clip_range,
            1.0 + self.clip_range
        )
        policy_loss = torch.max(policy_loss1, policy_loss2).mean()
        
        # 价值损失
        value_pred_clipped = old_values + torch.clamp(
            values - old_values,
            -self.value_clip_range,
            self.value_clip_range
        )
        value_loss1 = F.mse_loss(values, returns)
        value_loss2 = F.mse_loss(value_pred_clipped, returns)
        value_loss = torch.max(value_loss1, value_loss2).mean()
        
        # 熵损失
        entropy_loss = -log_probs.mean()
        
        # 总损失
        total_loss = policy_loss + self.vf_coef * value_loss - 0.01 * entropy_loss
        
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss
        }
    
    def update(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """更新模型"""
        # 计算当前策略的动作概率和价值
        logits = self.model(input_ids, attention_mask)
        log_probs = F.log_softmax(logits, dim=-1)
        values = self.reward_model(input_ids, attention_mask)
        
        # 计算优势函数
        advantages, returns = self.compute_advantages(rewards, values, dones)
        
        # 计算损失
        loss_dict = self.compute_loss(
            log_probs=log_probs,
            values=values,
            advantages=advantages,
            returns=returns,
            old_log_probs=old_log_probs,
            old_values=old_values
        )
        
        # 优化
        self.optimizer.zero_grad()
        loss_dict["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {k: v.item() for k, v in loss_dict.items()}
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "reward_model_state_dict": self.reward_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.reward_model.load_state_dict(checkpoint["reward_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) 