#!/bin/bash
#SBATCH -J extract_hidden
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --output=job_output.log
#SBATCH --error=job_error.log

MODEL_NAME_OR_PATH="/home/fit/dongyinp/WORK/workspace/Qwen3-8B"
# 提取 MODEL_NAME_OR_PATH 中 workspace/ 后面的名称
MODEL_NAME=$(basename "$MODEL_NAME_OR_PATH")
DATA_PATH="./results/deceptionbench/$MODEL_NAME/filtered_results.jsonl"

# 动态设置 SAVE_DIR
SAVE_DIR="./results/deceptionbench/$MODEL_NAME/hidden_states"

# 设置环境变量
export HF_ENDPOINT="https://hf-mirror.com"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# 激活环境
source /home/fit/dongyinp/WORK/miniconda3/bin/activate
conda activate decep

python -u deceptionbench_extract_hidden_states.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --data_path "$DATA_PATH" \
    --save_dir "$SAVE_DIR"