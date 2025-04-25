#!/bin/bash

# 创建必要的目录
mkdir -p /data/galaxy-llm/{data,outputs,logs,checkpoints}

# 复制代码
cp -r src /data/galaxy-llm/
cp requirements.txt /data/galaxy-llm/

# 安装依赖
cd /data/galaxy-llm
pip install -r requirements.txt

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/data/galaxy-llm:$PYTHONPATH

# 开始训练
cd /data/galaxy-llm
python src/train.py \
    --model_name bert-base-chinese \
    --train_file data/train.json \
    --eval_file data/eval.json \
    --output_dir outputs \
    --log_file logs/train.log \
    --use_cot \
    --device cuda 