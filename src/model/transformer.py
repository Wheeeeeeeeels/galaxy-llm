import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
        """
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 计算正弦和余弦
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为缓冲区
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            添加位置编码后的张量
        """
        return x + self.pe[:x.size(1)]

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """
        初始化多头注意力
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: Dropout概率
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            q: 查询张量 [batch_size, seq_len, d_model]
            k: 键张量 [batch_size, seq_len, d_model]
            v: 值张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码
            
        Returns:
            注意力输出 [batch_size, seq_len, d_model]
        """
        batch_size = q.size(0)
        
        # 线性变换并重塑
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # 计算注意力权重
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 计算输出
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(output)

class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        初始化前馈网络
        
        Args:
            d_model: 模型维度
            d_ff: 前馈网络中间维度
            dropout: Dropout概率
        """
        super().__init__()
        
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            前馈网络输出
        """
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        初始化Transformer块
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络中间维度
            dropout: Dropout概率
        """
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码
            
        Returns:
            Transformer块输出
        """
        # 自注意力
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class Transformer(nn.Module):
    """Transformer模型"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_length: int,
        dropout: float = 0.1
    ):
        """
        初始化Transformer模型
        
        Args:
            vocab_size: 词表大小
            d_model: 模型维度
            num_layers: 层数
            num_heads: 注意力头数
            d_ff: 前馈网络中间维度
            max_seq_length: 最大序列长度
            dropout: Dropout概率
        """
        super().__init__()
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len]
            mask: 注意力掩码
            
        Returns:
            模型输出 [batch_size, seq_len, vocab_size]
        """
        # 词嵌入
        x = self.embedding(x)
        
        # 位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer层
        for layer in self.layers:
            x = layer(x, mask)
            
        # 输出层
        x = self.norm(x)
        x = self.output(x)
        
        return x 