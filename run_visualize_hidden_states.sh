#!/bin/bash
#SBATCH -J vis_hidden
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --output=job_vis_output.log
#SBATCH --error=job_vis_error.log

MODEL_NAME_OR_PATH="/home/fit/dongyinp/WORK/workspace/Qwen3-8B"
MODEL_NAME=$(basename "$MODEL_NAME_OR_PATH")
HIDDEN_DIR="./results/deceptionbench/$MODEL_NAME/hidden_states"
SAVE_DIR="./results/deceptionbench/$MODEL_NAME/hidden_vis"

source /home/fit/dongyinp/WORK/miniconda3/bin/activate
conda activate decep

python -u deceptionbench_visualize.py \
    --hidden_dir "$HIDDEN_DIR" \
    --save_dir "$SAVE_DIR" \
    --layer -1  # 可改为指定层