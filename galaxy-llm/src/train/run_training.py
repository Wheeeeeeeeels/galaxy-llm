import os
import argparse
import deepspeed
import torch
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.train.train_moe import main as train_main

def parse_args():
    parser = argparse.ArgumentParser(description="启动MoE模型训练")
    parser.add_argument(
        "--deepspeed",
        type=str,
        default="configs/deepspeed_config.json",
        help="Deepspeed配置文件路径"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="本地进程排名"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="使用的GPU数量"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="批次大小"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="学习率"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="权重衰减"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="最大梯度范数"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="模型保存目录"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="是否使用混合精度训练"
    )
    parser.add_argument(
        "--use_ddp",
        action="store_true",
        help="是否使用分布式数据并行"
    )
    return parser.parse_args()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspeed", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--use_ddp", action="store_true")
    args = parser.parse_args()
    
    # 设置环境变量以优化显存使用
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 设置环境变量
    if args.local_rank != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.local_rank)
    
    # 初始化分布式训练
    if args.use_ddp:
        deepspeed.init_distributed()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 启动训练
    train_main(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        save_dir=args.save_dir,
        use_amp=args.use_amp,
        use_ddp=args.use_ddp,
        local_rank=args.local_rank
    )

if __name__ == "__main__":
    main() 