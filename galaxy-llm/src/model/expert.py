import torch
import torch.nn as nn
from typing import Tuple, Optional

class Expert(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True
    ):
        """
        专家网络
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            dropout: Dropout比率
            use_gradient_checkpointing: 是否使用梯度检查点
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # 前馈网络
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # 激活函数和Dropout
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: 输出张量 [batch_size, seq_len, output_dim]
        """
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                x,
                use_reentrant=False
            )
        else:
            return self._forward_impl(x)
            
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播实现"""
        # 第一个全连接层
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # 第二个全连接层
        x = self.fc2(x)
        x = self.dropout(x)
        
        # 层归一化
        x = self.layer_norm(x)
        
        return x
        
    def optimize_memory(self):
        """优化内存使用"""
        # 使用梯度检查点
        self.use_gradient_checkpointing = True
        
        # 优化参数初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        
        # 使用混合精度训练
        self.half()
        
    def get_memory_usage(self) -> float:
        """获取内存使用量（MB）"""
        total_params = sum(
            p.numel() for p in self.parameters()
        )
        return total_params * 4 / 1024**2  # 4 bytes per parameter

class EducationExpert(Expert):
    def __init__(self, vocab_size, d_model=1024, num_heads=16, num_layers=20, d_ff=4096, dropout=0.1):
        super().__init__(vocab_size, d_model, d_ff, dropout)
        
        # 教育领域特定层
        self.education_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x, mask=None):
        base_output = super().forward(x, mask)
        return self.education_head(base_output)

class PropertyExpert(Expert):
    def __init__(self, vocab_size, d_model=1024, num_heads=16, num_layers=20, d_ff=4096, dropout=0.1):
        super().__init__(vocab_size, d_model, d_ff, dropout)
        
        # 房产领域特定层
        self.property_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x, mask=None):
        base_output = super().forward(x, mask)
        return self.property_head(base_output)

class SalesExpert(Expert):
    def __init__(self, vocab_size, d_model=1024, num_heads=16, num_layers=20, d_ff=4096, dropout=0.1):
        super().__init__(vocab_size, d_model, d_ff, dropout)
        
        # 销售领域特定层
        self.sales_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x, mask=None):
        base_output = super().forward(x, mask)
        return self.sales_head(base_output)

class MathExpert(Expert):
    def __init__(self, vocab_size, d_model=1024, num_heads=16, num_layers=20, d_ff=4096, dropout=0.1):
        super().__init__(vocab_size, d_model, d_ff, dropout)
        
        # 数学领域特定层
        self.math_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x, mask=None):
        base_output = super().forward(x, mask)
        return self.math_head(base_output)

class LogicExpert(Expert):
    def __init__(self, vocab_size, d_model=1024, num_heads=16, num_layers=20, d_ff=4096, dropout=0.1):
        super().__init__(vocab_size, d_model, d_ff, dropout)
        
        # 逻辑领域特定层
        self.logic_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x, mask=None):
        base_output = super().forward(x, mask)
        return self.logic_head(base_output)

class CommonSenseExpert(Expert):
    def __init__(self, vocab_size, d_model=1024, num_heads=16, num_layers=20, d_ff=4096, dropout=0.1):
        super().__init__(vocab_size, d_model, d_ff, dropout)
        
        # 常识领域特定层
        self.common_sense_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x, mask=None):
        base_output = super().forward(x, mask)
        return self.common_sense_head(base_output) 