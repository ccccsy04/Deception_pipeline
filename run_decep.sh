#!/bin/bash

# 设置参数
MODEL_NAME_OR_PATH="/home/fit/dongyinp/WORK/workspace/Qwen3-8B"
DATASET_NAME="PKU-Alignment/DeceptionBench"
OUTPUT_FILE="deceptionbench_results.jsonl"
BATCH_SIZE=16

# 提取 MODEL_NAME_OR_PATH 中 workspace/ 后面的名称
MODEL_NAME=$(basename "$MODEL_NAME_OR_PATH")

# 动态设置 SAVE_DIR
SAVE_DIR="./results/deceptionbench/$MODEL_NAME"

# 设置环境变量
export HF_ENDPOINT="https://hf-mirror.com"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# 激活环境
source /home/fit/dongyinp/WORK/miniconda3/bin/activate
conda activate decep

# 使用srun运行，可以看到实时输出
srun -J deceptionbench -N 1 -p a01 --gres=gpu:2 --ntasks-per-node=1 \
    python -u deceptionbench.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dataset_name "$DATASET_NAME" \
    --save_dir "$SAVE_DIR" \
    --output_file "$OUTPUT_FILE" \
    --batch_size "$BATCH_SIZE" \
