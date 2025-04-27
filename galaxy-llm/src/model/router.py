import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np
from .config import MoEConfig

class GPRORouter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        k: int = 2,
        dropout: float = 0.1,
        temperature: float = 1.0,
        baseline_decay: float = 0.99
    ):
        """
        GPRO路由网络
        
        Args:
            input_dim: 输入维度
            num_experts: 专家数量
            k: 每个输入选择的专家数量
            dropout: Dropout比率
            temperature: 采样温度
            baseline_decay: 基线衰减率
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.k = k
        self.temperature = temperature
        self.baseline_decay = baseline_decay
        
        # 路由网络
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_experts)
        )
        
        # 基线网络
        self.baseline = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1)
        )
        
        # 初始化权重
        self._init_weights()
        
        # 初始化基线值
        self.baseline_value = 0.0
        
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            mask: 注意力掩码 [batch_size, seq_len]
            training: 是否处于训练模式
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
                - 专家权重 [batch_size, seq_len, k]
                - 专家索引 [batch_size, seq_len, k]
                - 额外信息字典
        """
        # 计算专家分数
        expert_scores = self.router(x)  # [batch_size, seq_len, num_experts]
        
        # 如果提供了掩码，应用掩码
        if mask is not None:
            expert_scores = expert_scores.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
            
        # 计算基线值
        baseline = self.baseline(x).squeeze(-1)  # [batch_size, seq_len]
        
        if training:
            # 使用Gumbel-Softmax进行采样
            expert_probs = F.softmax(expert_scores / self.temperature, dim=-1)
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(expert_probs)))
            expert_probs = F.softmax((expert_scores + gumbel_noise) / self.temperature, dim=-1)
            
            # 选择top-k专家
            expert_weights, expert_indices = torch.topk(expert_probs, k=self.k, dim=-1)
            
            # 计算策略梯度
            policy_gradient = (expert_weights - baseline.unsqueeze(-1)).detach()
            
            # 更新基线值
            self.baseline_value = (
                self.baseline_decay * self.baseline_value +
                (1 - self.baseline_decay) * baseline.mean().item()
            )
        else:
            # 推理时直接选择top-k专家
            expert_weights, expert_indices = torch.topk(
                F.softmax(expert_scores, dim=-1),
                k=self.k,
                dim=-1
            )
            policy_gradient = None
            
        # 收集额外信息
        extra_info = {
            'baseline': baseline,
            'policy_gradient': policy_gradient,
            'expert_scores': expert_scores
        }
        
        return expert_weights, expert_indices, extra_info
        
    def compute_reinforcement_loss(
        self,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        rewards: torch.Tensor,
        extra_info: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        计算强化学习损失
        
        Args:
            expert_weights: 专家权重 [batch_size, seq_len, k]
            expert_indices: 专家索引 [batch_size, seq_len, k]
            rewards: 奖励值 [batch_size, seq_len]
            extra_info: 额外信息字典
            
        Returns:
            强化学习损失
        """
        # 计算优势值
        baseline = extra_info['baseline']
        advantages = rewards - baseline
        
        # 计算策略梯度损失
        policy_gradient = extra_info['policy_gradient']
        policy_loss = -(policy_gradient * advantages.unsqueeze(-1)).mean()
        
        # 计算基线损失
        baseline_loss = F.mse_loss(baseline, rewards)
        
        # 计算熵正则化
        expert_scores = extra_info['expert_scores']
        entropy = -torch.sum(
            F.softmax(expert_scores, dim=-1) * F.log_softmax(expert_scores, dim=-1),
            dim=-1
        ).mean()
        
        # 总损失
        total_loss = policy_loss + 0.5 * baseline_loss - 0.01 * entropy
        
        return total_loss

class LoadBalancer:
    def __init__(self, num_experts, capacity_factor=1.0):
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.expert_usage = torch.zeros(num_experts)
        
    def update_usage(self, expert_indices):
        # 更新专家使用情况
        for idx in expert_indices:
            self.expert_usage[idx] += 1
            
    def get_load_balance_loss(self, router_probs):
        # 计算负载均衡损失
        mean_expert_usage = self.expert_usage.mean()
        load_balance_loss = torch.sum((self.expert_usage - mean_expert_usage) ** 2)
        return load_balance_loss
        
    def reset(self):
        # 重置专家使用情况
        self.expert_usage.zero_()

class DynamicRouter(GPRORouter):
    def __init__(self, d_model, num_experts, k=2, capacity_factor=1.0):
        super().__init__(d_model, num_experts, k)
        self.load_balancer = LoadBalancer(num_experts, capacity_factor)
        
    def forward(self, x):
        # 获取基础路由结果
        top_k_probs, top_k_indices, extra_info = super().forward(x)
        
        # 更新负载均衡器
        self.load_balancer.update_usage(top_k_indices)
        
        # 计算负载均衡损失
        load_balance_loss = self.load_balancer.get_load_balance_loss(top_k_probs)
        
        return top_k_probs, top_k_indices, load_balance_loss, extra_info
        
    def reset(self):
        self.load_balancer.reset()

class Router(nn.Module):
    """专家路由"""
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # 路由网络
        self.router_net = nn.Sequential(
            nn.Linear(config.expert_config.hidden_size, config.expert_config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.expert_config.hidden_size, config.num_experts)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """计算专家权重"""
        # 计算路由logits
        logits = self.router_net(hidden_states)
        
        # 根据路由类型选择权重计算方式
        if self.config.routing_type == "learned":
            weights = F.softmax(logits, dim=-1)
        elif self.config.routing_type == "random":
            weights = torch.rand_like(logits)
            weights = F.softmax(weights, dim=-1)
        else:  # uniform
            weights = torch.ones_like(logits) / self.config.num_experts
            
        # 选择top_k专家
        top_k_weights, top_k_indices = torch.topk(weights, self.config.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        return top_k_weights, top_k_indices 