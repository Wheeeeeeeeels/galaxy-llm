import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from .config import ExpertConfig, MoEConfig, ModelConfig
from . import Router

class ExpertModel(nn.Module):
    """专家模型"""
    def __init__(self, config: ExpertConfig):
        super().__init__()
        self.config = config
        
        # 词嵌入层
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # 编码器层
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 词嵌入
        x = self.embedding(input_ids)
        
        # 编码器层
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=~attention_mask.bool())
            
        # 输出层
        logits = self.output_layer(x)
        
        return logits

class MoEModel(nn.Module):
    """MoE模型"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 初始化专家模型
        self.experts = nn.ModuleList([
            ExpertModel(config.moe_config.expert_config) for _ in range(config.moe_config.num_experts)
        ])
        
        # 初始化路由
        self.router = Router(
            input_dim=config.moe_config.expert_config.hidden_size,
            num_experts=config.moe_config.num_experts,
            k=config.moe_config.top_k
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size = input_ids.size(0)
        
        # 获取所有专家的输出
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(input_ids, attention_mask)
            expert_outputs.append(expert_output)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, seq_len, vocab_size]
        
        # 计算专家权重
        hidden_states = expert_outputs.mean(dim=2)  # [batch_size, num_experts, hidden_size]
        weights, indices, _ = self.router(hidden_states)
        
        # 加权组合专家输出
        selected_outputs = torch.gather(
            expert_outputs, 
            dim=1, 
            index=indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, expert_outputs.size(2), expert_outputs.size(3))
        )
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        output = (selected_outputs * weights).sum(dim=1)
        
        return output
    
    def compute_loss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算损失"""
        # 获取模型输出
        logits = self(input_ids, attention_mask)
        
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # 计算正则化损失
        reg_loss = 0
        for param in self.parameters():
            reg_loss += torch.norm(param, p=2)
        reg_loss = self.config.moe_config.regularization_weight * reg_loss
        
        # 计算稀疏性损失
        sparse_loss = 0
        for expert in self.experts:
            for param in expert.parameters():
                sparse_loss += torch.norm(param, p=1)
        sparse_loss = self.config.moe_config.sparsity_weight * sparse_loss
        
        # 计算多样性损失
        diversity_loss = 0
        for i in range(len(self.experts)):
            for j in range(i + 1, len(self.experts)):
                # 计算专家权重矩阵的余弦相似度
                w1 = self.experts[i].output_layer.weight
                w2 = self.experts[j].output_layer.weight
                similarity = F.cosine_similarity(w1, w2, dim=0)
                diversity_loss += torch.mean(similarity)
        diversity_loss = self.config.moe_config.diversity_weight * diversity_loss
        
        # 总损失
        total_loss = ce_loss + reg_loss + sparse_loss + diversity_loss
        
        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "reg_loss": reg_loss,
            "sparse_loss": sparse_loss,
            "diversity_loss": diversity_loss
        } 