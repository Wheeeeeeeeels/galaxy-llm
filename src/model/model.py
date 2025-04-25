import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Dict
from .expert import Expert
from .router import GPRORouter

class GalaxyLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        num_experts: int = 8,
        expert_dim: int = 1024,
        dropout: float = 0.1,
        k: int = 2,
        temperature: float = 1.0,
        baseline_decay: float = 0.99
    ):
        """
        Galaxy LLM 模型
        
        Args:
            vocab_size: 词表大小
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            num_experts: 专家数量
            expert_dim: 专家网络维度
            dropout: Dropout比率
            k: 每个输入选择的专家数量
            temperature: 采样温度
            baseline_decay: 基线衰减率
        """
        super().__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(1024, hidden_dim)  # 最大序列长度1024
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=expert_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 专家网络
        self.experts = nn.ModuleList([
            Expert(
                input_dim=hidden_dim,
                hidden_dim=expert_dim,
                output_dim=hidden_dim,
                dropout=dropout
            )
            for _ in range(num_experts)
        ])
        
        # GPRO路由网络
        self.router = GPRORouter(
            input_dim=hidden_dim,
            num_experts=num_experts,
            k=k,
            dropout=dropout,
            temperature=temperature,
            baseline_decay=baseline_decay
        )
        
        # 输出层
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        # 初始化词嵌入
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        # 初始化输出层
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.bias)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            input_ids: 输入ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            training: 是否处于训练模式
            
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                - 模型输出 [batch_size, seq_len, vocab_size]
                - 额外信息字典
        """
        batch_size, seq_len = input_ids.shape
        
        # 生成位置编码
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # 获取词嵌入和位置嵌入
        token_embeddings = self.embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        
        # 合并嵌入
        embeddings = token_embeddings + position_embeddings
        
        # Transformer编码
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            
        hidden_states = self.transformer(
            embeddings,
            src_key_padding_mask=attention_mask.squeeze(1) if attention_mask is not None else None
        )
        
        # 路由到专家
        expert_weights, expert_indices, router_info = self.router(
            hidden_states,
            attention_mask,
            training=training
        )
        
        # 计算专家输出
        expert_outputs = []
        for i in range(self.router.k):
            expert_idx = expert_indices[..., i]
            expert_weight = expert_weights[..., i].unsqueeze(-1)
            
            # 获取每个位置的专家输出
            expert_output = torch.zeros_like(hidden_states)
            for j, expert in enumerate(self.experts):
                mask = (expert_idx == j)
                if mask.any():
                    expert_output[mask] = expert(hidden_states[mask])
                    
            expert_outputs.append(expert_output * expert_weight)
            
        # 合并专家输出
        hidden_states = sum(expert_outputs)
        
        # 输出层
        logits = self.output(hidden_states)
        
        # 收集额外信息
        extra_info = {
            'router_info': router_info,
            'expert_weights': expert_weights,
            'expert_indices': expert_indices
        }
        
        return logits, extra_info
        
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        extra_info: Dict[str, torch.Tensor],
        rewards: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算损失
        
        Args:
            logits: 模型输出 [batch_size, seq_len, vocab_size]
            labels: 标签 [batch_size, seq_len]
            extra_info: 额外信息字典
            rewards: 奖励值 [batch_size, seq_len]
            
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                - 总损失
                - 损失字典
        """
        # 计算交叉熵损失
        ce_loss = nn.CrossEntropyLoss(ignore_index=-100)(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # 计算GPRO损失
        if rewards is not None:
            gpro_loss = self.router.compute_reinforcement_loss(
                extra_info['expert_weights'],
                extra_info['expert_indices'],
                rewards,
                extra_info['router_info']
            )
        else:
            gpro_loss = torch.tensor(0.0, device=logits.device)
            
        # 总损失
        total_loss = ce_loss + 0.1 * gpro_loss
        
        # 收集损失信息
        loss_dict = {
            'ce_loss': ce_loss.item(),
            'gpro_loss': gpro_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
        
    def reset(self):
        self.router.reset()
        
    def get_expert_usage(self):
        return self.router.load_balancer.expert_usage 