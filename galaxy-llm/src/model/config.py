from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ExpertConfig:
    """专家模型配置"""
    vocab_size: int = 50000  # 词表大小
    hidden_size: int = 768  # 隐藏层维度
    num_layers: int = 12  # Transformer层数
    num_attention_heads: int = 12  # 注意力头数
    intermediate_size: int = 3072  # 前馈网络中间层维度
    max_length: int = 2048  # 最大序列长度
    dropout: float = 0.1  # Dropout率
    activation: str = "gelu"
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    use_cache: bool = True
    gradient_checkpointing: bool = True  # 启用梯度检查点以节省显存

@dataclass
class MoEConfig:
    """MoE配置"""
    expert_config: ExpertConfig = ExpertConfig()
    num_experts: int = 8  # 专家数量
    top_k: int = 2  # 每个样本使用的专家数量
    routing_type: str = "learned"  # 路由类型：learned, random, uniform
    regularization_weight: float = 0.01  # 正则化权重
    sparsity_weight: float = 0.01  # 稀疏性权重
    diversity_weight: float = 0.01  # 多样性权重
    expert_capacity: int = 64
    use_aux_loss: bool = True  # 使用辅助损失
    aux_loss_weight: float = 0.1  # 辅助损失权重
    load_balancing_weight: float = 0.01  # 负载均衡权重

@dataclass
class ModelConfig:
    """模型配置"""
    vocab_size: int = 50000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_experts: int = 8
    expert_capacity: int = 64
    dropout: float = 0.1
    max_length: int = 2048  # 增加最大长度
    moe_config: MoEConfig = MoEConfig()
    use_flash_attention: bool = True  # 使用Flash Attention加速
    use_rotary_embeddings: bool = True  # 使用旋转位置编码
    use_alibi: bool = False  # 使用ALiBi位置编码
    use_rope: bool = True  # 使用RoPE位置编码
    rope_theta: float = 10000.0  # RoPE的theta参数
    rope_scaling: Optional[float] = None  # RoPE的缩放参数
    use_parallel_attention: bool = True  # 使用并行注意力计算
    use_parallel_ffn: bool = True  # 使用并行前馈网络
    use_activation_checkpointing: bool = True  # 使用激活检查点
    use_gradient_checkpointing: bool = True  # 使用梯度检查点
    use_mixed_precision: bool = True  # 使用混合精度训练
    use_bf16: bool = False  # 使用BF16精度
    use_fp16: bool = True  # 使用FP16精度
    use_amp: bool = True  # 使用自动混合精度
    use_ddp: bool = True  # 使用分布式数据并行
    use_fsdp: bool = False  # 使用完全分片数据并行
    use_deepspeed: bool = False  # 使用DeepSpeed
    use_megatron: bool = False  # 使用Megatron-LM
    use_sequence_parallel: bool = False  # 使用序列并行
    use_tensor_parallel: bool = False  # 使用张量并行
    use_pipeline_parallel: bool = False  # 使用流水线并行
    use_zero: bool = False  # 使用ZeRO优化
    zero_stage: int = 0  # ZeRO阶段
    use_offload: bool = False  # 使用CPU卸载
    use_activation_offload: bool = False  # 使用激活卸载
    use_parameter_offload: bool = False  # 使用参数卸载
    use_optimizer_offload: bool = False  # 使用优化器卸载
    use_checkpoint_offload: bool = False  # 使用检查点卸载
    use_memory_efficient_attention: bool = True  # 使用内存高效注意力
    use_xformers: bool = True  # 使用xFormers
    use_flash_attn: bool = True  # 使用Flash Attention
    use_sdpa: bool = True  # 使用Scaled Dot-Product Attention
    use_fused_mlp: bool = True  # 使用融合MLP
    use_fused_dropout: bool = True  # 使用融合Dropout
    use_fused_layernorm: bool = True  # 使用融合LayerNorm
    use_fused_softmax: bool = True  # 使用融合Softmax
    use_fused_gelu: bool = True  # 使用融合GELU
    use_fused_silu: bool = True  # 使用融合SiLU
    use_fused_swish: bool = True  # 使用融合Swish
    use_fused_mish: bool = True  # 使用融合Mish
    use_fused_relu: bool = True  # 使用融合ReLU
    use_fused_leaky_relu: bool = True  # 使用融合LeakyReLU
    use_fused_elu: bool = True  # 使用融合ELU
    use_fused_selu: bool = True  # 使用融合SELU
    use_fused_glu: bool = True  # 使用融合GLU
    use_fused_bias_gelu: bool = True  # 使用融合BiasGELU
    use_fused_bias_dropout: bool = True  # 使用融合BiasDropout
    use_fused_bias_layernorm: bool = True  # 使用融合BiasLayerNorm
    use_fused_bias_softmax: bool = True  # 使用融合BiasSoftmax
    use_fused_bias_gelu_dropout: bool = True  # 使用融合BiasGELUDropout
    use_fused_bias_layernorm_dropout: bool = True  # 使用融合BiasLayerNormDropout
    use_fused_bias_softmax_dropout: bool = True  # 使用融合BiasSoftmaxDropout
    use_fused_bias_gelu_layernorm: bool = True  # 使用融合BiasGELULayerNorm
    use_fused_bias_gelu_softmax: bool = True  # 使用融合BiasGELUSoftmax
    use_fused_bias_layernorm_softmax: bool = True  # 使用融合BiasLayerNormSoftmax
    use_fused_bias_gelu_layernorm_dropout: bool = True  # 使用融合BiasGELULayerNormDropout
    use_fused_bias_gelu_softmax_dropout: bool = True  # 使用融合BiasGELUSoftmaxDropout
    use_fused_bias_layernorm_softmax_dropout: bool = True  # 使用融合BiasLayerNormSoftmaxDropout
    learning_rate: float = 1e-5  # 学习率
    warmup_steps: int = 1000  # 预热步数
    weight_decay: float = 0.01  # 权重衰减
    gradient_accumulation_steps: int = 4  # 梯度累积步数
    max_grad_norm: float = 1.0  # 最大梯度范数
    batch_size: int = 8  # 批次大小

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

# 思维链模型配置（类似DeepSeek-R1）
COT_MODEL_CONFIG = ModelConfig(
    moe_config=MoEConfig(
        expert_config=ExpertConfig(
            vocab_size=50000,
            hidden_size=768,
            num_layers=6,
            num_attention_heads=12,
            intermediate_size=3072,
            max_length=2048,
            dropout=0.1
        ),
        num_experts=8,
        top_k=2,
        routing_type="learned"
    ),
    max_length=2048,
    batch_size=8,
    learning_rate=1e-5,
    warmup_steps=1000,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    fp16=True,
    zero_stage=2,
    eos_token_id=102
)

# 直接回答模型配置（类似DeepSeek-V3）
DIRECT_MODEL_CONFIG = ModelConfig(
    moe_config=MoEConfig(
        expert_config=ExpertConfig(
            vocab_size=50000,
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            intermediate_size=4096,
            max_length=2048,
            dropout=0.1
        ),
        num_experts=4,
        top_k=1,
        routing_type="learned"
    ),
    max_length=2048,
    batch_size=8,
    learning_rate=1e-5,
    warmup_steps=1000,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    fp16=True,
    zero_stage=2,
    eos_token_id=102
) 