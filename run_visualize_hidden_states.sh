#!/bin/bash
#SBATCH -J vis_hidden
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=job_vis_output.log
#SBATCH --error=job_vis_error.log

MODEL_NAME_OR_PATH="/home/fit/dongyinp/WORK/workspace/Qwen3-8B"
MODEL_NAME=$(basename "$MODEL_NAME_OR_PATH")
HIDDEN_DIR_EXP="./results/deceptionbench/$MODEL_NAME/hidden_states_exp"
HIDDEN_DIR_CTRL="./results/deceptionbench/$MODEL_NAME/hidden_states_ctrl"
SAVE_DIR="./results/deceptionbench/$MODEL_NAME/hidden_vis"

source /home/fit/dongyinp/WORK/miniconda3/bin/activate
conda activate decep

python -u deceptionbench_visualize.py \
    --hidden_dir_exp "$HIDDEN_DIR_EXP" \
    --hidden_dir_control "$HIDDEN_DIR_CTRL" \
    --save_dir "$SAVE_DIR" \
    --layer -1