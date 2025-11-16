#!/bin/bash
#SBATCH -J singleturn
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --output=job_output.log
#SBATCH --error=job_error.log

# 设置参数
MODEL_NAME_OR_PATH="/home/fit/dongyinp/WORK/workspace/Qwen3-8B"
DATASET_NAME="HuggingFaceH4/MATH-500"
OUTPUT_FILE="successful_cases.jsonl"
MAX_EXAMPLES="3"
MAX_TURN="5"
LEVEL="1,2,3"

# 提取 MODEL_NAME_OR_PATH 中 workspace/ 后面的名称
MODEL_NAME=$(basename "$MODEL_NAME_OR_PATH")

# 动态设置 SAVE_DIR
SAVE_DIR="./results/single_turn/$MODEL_NAME"

# 设置环境变量
export HF_ENDPOINT="https://hf-mirror.com"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# 激活环境
source /home/fit/dongyinp/WORK/miniconda3/bin/activate
conda activate decep

# 直接运行python，不需要srun
python -u singleturn.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dataset_name "$DATASET_NAME" \
    --save_dir "$SAVE_DIR" \
    --output_file "$OUTPUT_FILE" \
    --max_examples "$MAX_EXAMPLES" \
    --max_turn "$MAX_TURN" \
    --level $(echo $LEVEL | tr ',' ' ')

