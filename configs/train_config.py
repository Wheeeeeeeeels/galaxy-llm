from dataclasses import dataclass

@dataclass
class TrainConfig:
    # 数据参数
    train_data_path: str = "data/train.txt"
    val_data_path: str = "data/val.txt"
    max_seq_length: int = 1024
    
    # 训练参数
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # 优化器参数
    optimizer: str = "adamw"  # adamw, adam, sgd
    scheduler: str = "cosine"  # cosine, linear, constant
    
    # 检查点参数
    checkpoint_dir: str = "checkpoints"
    save_total_limit: int = 5
    
    # 日志参数
    log_dir: str = "logs"
    tensorboard: bool = True 