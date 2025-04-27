#!/bin/bash

# 激活conda环境
source /root/miniconda3/bin/activate moe-education

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 创建必要的目录
mkdir -p tokenizer
mkdir -p checkpoints

# 检查处理后的数据文件是否存在
if [ ! -f "data/processed/split/train.json" ]; then
    echo "错误：处理后的训练数据文件不存在，请先运行数据处理脚本"
    exit 1
fi

# 开始训练分词器
echo "开始训练分词器..."
python3 src/train/train_tokenizer.py

# 检查分词器是否训练成功
if [ ! -f "tokenizer/tokenizer.json" ]; then
    echo "错误：分词器训练失败，请检查日志"
    exit 1
fi

# 开始训练模型
echo "开始训练模型..."
deepspeed --num_gpus=1 src/train/run_training.py \
    --deepspeed configs/deepspeed_config.json \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --save_dir checkpoints \
    --use_amp \
    --use_ddp 