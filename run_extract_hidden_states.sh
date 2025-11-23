#!/bin/bash
#SBATCH -J extract_hidden_exp
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --output=job_output_exp.log
#SBATCH --error=job_error_exp.log

MODEL_NAME_OR_PATH="/home/fit/dongyinp/WORK/workspace/Qwen3-8B"
MODEL_NAME=$(basename "$MODEL_NAME_OR_PATH")
EXP_DATA_PATH="./results/deceptionbench/$MODEL_NAME/filtered_results.jsonl"
EXP_SAVE_DIR="./results/deceptionbench/$MODEL_NAME/hidden_states_exp"
CTRL_DATA_PATH="./results/deceptionbench/$MODEL_NAME/deceptionbench_results.jsonl"
CTRL_SAVE_DIR="./results/deceptionbench/$MODEL_NAME/hidden_states_ctrl"
AUTO_ANALYSIS_PATH="./results/deceptionbench/$MODEL_NAME/auto_analysis.jsonl"

export HF_ENDPOINT="https://hf-mirror.com"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

source /home/fit/dongyinp/WORK/miniconda3/bin/activate
conda activate decep

python -u deceptionbench_extract_hidden_states.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --data_path "$EXP_DATA_PATH" \
    --save_dir "$EXP_SAVE_DIR" \
    --type "Honesty_Evasion_Under_Pressure"



python -u deceptionbench_extract_hidden_states.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --data_path "$CTRL_DATA_PATH" \
    --save_dir "$CTRL_SAVE_DIR" \
    --type "Honesty_Evasion_Under_Pressure" \
    --control \
    --auto_analysis_path "$AUTO_ANALYSIS_PATH"