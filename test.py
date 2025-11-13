from utils.get_math_results_dual import main as eval_math_results

# 评测并保存结果
eval_math_results(
    res_path="./results/mutual_info/Qwen3-8B/predictions.jsonl",  # 你的预测结果文件路径
    save=True,                                                    # 是否保存评测结果
    k=None,                                                       # 可选，保留前k条生成
    output_dir="./results/mutual_info/Qwen3-8B"                   # 输出目录
)