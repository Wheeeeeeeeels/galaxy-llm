from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ExpertConfig:
    """专家模型配置"""
    vocab_size: int = 50000  # 词表大小
    hidden_size: int = 768  # 隐藏层维度
    num_layers: int = 6  # Transformer层数
    num_attention_heads: int = 12  # 注意力头数
    intermediate_size: int = 3072  # 前馈网络中间层维度
    dropout: float = 0.1  # Dropout率

@dataclass
class MoEConfig:
    """MoE模型配置"""
    num_experts: int = 8  # 专家数量
    top_k: int = 2  # 每次选择的前k个专家
    routing_type: str = "learned"  # 路由类型：learned, random, uniform
    expert_config: ExpertConfig = ExpertConfig()  # 专家模型配置
    regularization_weight: float = 0.1  # 正则化权重
    sparsity_weight: float = 0.1  # 稀疏性权重
    diversity_weight: float = 0.1  # 多样性权重

@dataclass
class ModelConfig:
    """模型配置"""
    moe_config: MoEConfig = MoEConfig()  # MoE配置
    use_cot: bool = True  # 是否使用思维链
    cot_length: int = 100  # 思维链长度
    temperature: float = 0.7  # 温度参数
    max_new_tokens: int = 2048  # 最大生成token数

# 教育领域专家配置
EDUCATION_EXPERTS = [
    ExpertConfig(
        name="school_info_expert",
        domain="education",
        tasks=["info_query"],
        trainable=True
    ),
    ExpertConfig(
        name="school_comparison_expert",
        domain="education",
        tasks=["comparison"],
        trainable=True
    ),
    ExpertConfig(
        name="school_recommendation_expert",
        domain="education",
        tasks=["recommendation"],
        trainable=True
    )
]

# 思维链模型配置
COT_MODEL_CONFIG = ModelConfig(
    moe_config=MoEConfig(
        num_experts=8,
        top_k=2,
        routing_type="learned",
        expert_config=ExpertConfig(
            vocab_size=50000,
            hidden_size=768,
            num_layers=6,
            num_attention_heads=12,
            intermediate_size=3072,
            dropout=0.1
        ),
        regularization_weight=0.1,
        sparsity_weight=0.1,
        diversity_weight=0.1
    ),
    use_cot=True,
    cot_length=100,
    temperature=0.7,
    max_new_tokens=2048
)

# 直接模型配置
DIRECT_MODEL_CONFIG = ModelConfig(
    moe_config=MoEConfig(
        num_experts=8,
        top_k=2,
        routing_type="learned",
        expert_config=ExpertConfig(
            vocab_size=50000,
            hidden_size=768,
            num_layers=6,
            num_attention_heads=12,
            intermediate_size=3072,
            dropout=0.1
        ),
        regularization_weight=0.1,
        sparsity_weight=0.1,
        diversity_weight=0.1
    ),
    use_cot=False,
    cot_length=0,
    temperature=0.7,
    max_new_tokens=2048
) 