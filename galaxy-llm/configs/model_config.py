from dataclasses import dataclass

@dataclass
class ModelConfig:
    # 模型参数
    vocab_size: int = 50257
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_positions: int = 1024
    dropout: float = 0.1
    
    # 训练参数
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # 其他参数
    device: str = "cuda"  # 或 "cpu"
    seed: int = 42
    log_interval: int = 100
    save_interval: int = 1000 