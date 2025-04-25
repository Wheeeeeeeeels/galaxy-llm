import logging
import os
from typing import Optional
from datetime import datetime

def setup_logger(
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        log_file: 日志文件路径
        level: 日志级别
        
    Returns:
        日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def log_metrics(
    metrics: dict,
    prefix: str = '',
    logger: Optional[logging.Logger] = None
):
    """
    记录指标
    
    Args:
        metrics: 指标字典
        prefix: 前缀
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger()
        
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"{prefix}{key}: {value:.4f}")
        else:
            logger.info(f"{prefix}{key}: {value}")
            
def log_config(
    config: dict,
    prefix: str = '',
    logger: Optional[logging.Logger] = None
):
    """
    记录配置
    
    Args:
        config: 配置字典
        prefix: 前缀
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger()
        
    for key, value in config.items():
        if isinstance(value, dict):
            log_config(value, prefix=f"{prefix}{key}.", logger=logger)
        else:
            logger.info(f"{prefix}{key}: {value}")
            
def log_expert_usage(
    expert_usage: dict,
    prefix: str = '',
    logger: Optional[logging.Logger] = None
):
    """
    记录专家使用情况
    
    Args:
        expert_usage: 专家使用情况字典
        prefix: 前缀
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger()
        
    for expert, usage in expert_usage.items():
        logger.info(f"{prefix}{expert} 使用率: {usage:.2%}")
        
def log_memory_usage(
    memory_allocated: float,
    memory_reserved: float,
    prefix: str = '',
    logger: Optional[logging.Logger] = None
):
    """
    记录内存使用情况
    
    Args:
        memory_allocated: 已分配内存
        memory_reserved: 已保留内存
        prefix: 前缀
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger()
        
    logger.info(
        f"{prefix}GPU内存使用: {memory_allocated:.2f}MB (已分配) / "
        f"{memory_reserved:.2f}MB (已保留)"
    ) 