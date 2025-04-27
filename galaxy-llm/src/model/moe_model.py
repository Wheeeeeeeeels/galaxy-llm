import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from .config import ExpertConfig, MoEConfig, ModelConfig

class TransformerExpert(nn.Module):
    """基于Transformer的专家模型"""
    def __init__(self, config: ExpertConfig):
        super().__init__()
        self.config = config
        
        # 词嵌入层
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_length, config.hidden_size)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # 输出层
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size)
        
        # 是否使用梯度检查点
        self.gradient_checkpointing = config.gradient_checkpointing
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len = input_ids.size()
        
        # 生成位置编码
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # 词嵌入和位置编码
        x = self.embedding(input_ids) + self.position_embedding(position_ids)
        
        # 调整注意力掩码的形状
        src_key_padding_mask = ~attention_mask.bool()  # [batch_size, seq_len]
        
        # 创建注意力掩码
        attn_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'),
            diagonal=1
        )
        
        # Transformer编码（使用梯度检查点）
        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(inputs[0], src_key_padding_mask=inputs[1], mask=inputs[2])
                return custom_forward
            
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.transformer),
                x,
                src_key_padding_mask,
                attn_mask,
                use_reentrant=False
            )
        else:
            x = self.transformer(
                x,
                src_key_padding_mask=src_key_padding_mask,
                mask=attn_mask
            )
        
        # 输出层
        logits = self.output_layer(x)
        
        return logits

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

class MoEModel(nn.Module):
    """基于Transformer的MoE模型"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 初始化专家模型
        self.experts = nn.ModuleList([
            TransformerExpert(config.moe_config.expert_config)
            for _ in range(config.moe_config.num_experts)
        ])
        
        # 初始化路由
        self.router = Router(config.moe_config)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """前向传播"""
        batch_size = input_ids.size(0)
        
        # 获取所有专家的输出
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(input_ids, attention_mask)
            expert_outputs.append(expert_output)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, seq_len, vocab_size]
        
        # 计算专家权重
        # 使用最后一个隐藏层的输出作为路由输入
        hidden_states = expert_outputs[:, :, -1, :]  # [batch_size, num_experts, hidden_size]
        hidden_states = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        hidden_states = hidden_states[:, :self.config.moe_config.expert_config.hidden_size]  # [batch_size, hidden_size]
        weights, indices = self.router(hidden_states)
        
        # 加权组合专家输出
        selected_outputs = torch.gather(
            expert_outputs, 
            dim=1, 
            index=indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, expert_outputs.size(2), expert_outputs.size(3))
        )
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        output = (selected_outputs * weights).sum(dim=1)
        
        return output
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
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
        
        # 计算多样性损失（优化版本）
        diversity_loss = 0
        if self.config.moe_config.diversity_weight > 0:
            expert_weights = []
            for expert in self.experts:
                # 使用较小的子集计算相似度
                w = expert.output_layer.weight[:1000]  # 只使用前1000个词的权重
                w = F.normalize(w, p=2, dim=1)  # 归一化
                expert_weights.append(w)
            
            expert_weights = torch.stack(expert_weights)  # [num_experts, 1000, hidden_size]
            similarity_matrix = torch.matmul(expert_weights, expert_weights.transpose(1, 2))  # [num_experts, 1000, 1000]
            similarity_matrix = torch.mean(similarity_matrix, dim=(1, 2))  # [num_experts]
            
            # 只计算上三角矩阵的相似度
            num_experts = len(self.experts)
            for i in range(num_experts):
                for j in range(i + 1, num_experts):
                    diversity_loss += similarity_matrix[i] * similarity_matrix[j]
            
            diversity_loss = self.config.moe_config.diversity_weight * diversity_loss / (num_experts * (num_experts - 1) / 2)
        
        # 总损失
        total_loss = ce_loss + reg_loss + sparse_loss + diversity_loss
        
        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "reg_loss": reg_loss,
            "sparse_loss": sparse_loss,
            "diversity_loss": diversity_loss,
            "log_probs": F.log_softmax(logits, dim=-1)
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 512,
        num_return_sequences: int = 1,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """生成文本"""
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # 初始化生成序列
        generated = input_ids.clone()
        generated_attention_mask = attention_mask.clone()
        
        # 自回归生成
        for _ in range(max_length - input_ids.size(1)):
            # 获取模型输出
            outputs = self(generated, generated_attention_mask)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # 采样下一个token
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=num_return_sequences)
            
            # 更新生成序列
            generated = torch.cat([generated, next_tokens[:, :1]], dim=1)
            generated_attention_mask = torch.cat([
                generated_attention_mask,
                torch.ones((batch_size, 1), dtype=torch.bool, device=device)
            ], dim=1)
            
            # 检查是否生成了结束符
            if (next_tokens == self.config.eos_token_id).any():
                break
                
        return generated 