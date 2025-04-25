#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用4个GPU
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# 创建必要的目录
mkdir -p logs
mkdir -p checkpoints
mkdir -p models

# 使用DeepSpeed启动训练
deepspeed --num_gpus=4 \
    src/training/train.py \
    --data_dir data \
    --model_dir models \
    --log_dir logs \
    --checkpoint_dir checkpoints \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --save_total_limit 3 \
    --deepspeed_config configs/deepspeed_config.json 