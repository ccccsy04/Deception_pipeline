import os
import json
from tqdm import tqdm, trange
from utils.eval_math_rule.evaluation.grader import math_equal
from utils.eval_math_rule.evaluation.parser import extract_answer, parse_ground_truth, strip_string
from collections import Counter

import multiprocessing
import queue

def math_equal_with_timeout(pred, gt_ans, timeout):
    def target(result_queue):
        try:
            result_queue.put(math_equal(pred, gt_ans))
        except Exception as e:
            result_queue.put(e)


    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target, args=(result_queue,))
    process.start()

    process.join(timeout)

    if process.is_alive():
        print(f"Timeout occurred for prediction: {pred}")
        process.terminate()
        process.join()    
        return False

   
    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        print("Result queue timed out")
        return False

    if isinstance(result, Exception):
        print(f"Error occurred: {result}")
        return False

    return result



def parallel_math_equal(all_pred, gt_ans, timeout=20):
    results = []
    for pred in all_pred:
        results.append(math_equal_with_timeout(pred, gt_ans, timeout))
    return results



def main(res_path, save=False, k=None, output_dir=None):
    # args = parse_args()
    with open(res_path, "r") as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    items = []
    for example in tqdm(data):
        # gt_cot, gt = parse_ground_truth(example, data_name="omni-math")
        if "model_generation" not in example:
            example["model_generation"] = example["model_output"]
        if k is not None:
            example["model_generation"] = example["model_generation"][:k]
        gt_cot = example["answer"]
        gt_ans = extract_answer(gt_cot, data_name="omni-math")
        gt_cot = str(gt_cot).strip()
        gt_ans = strip_string(gt_ans, skip_unit=False)
        all_pred_honest = [extract_answer(p, data_name="omni-math") for p in example["model_generation"]["honest_output"]]
        all_pred_deceptive = [extract_answer(p, data_name="omni-math") for p in example["model_generation"]["deceptive_output"]]
        all_pred_honest = [strip_string(p, skip_unit=False) for p in all_pred_honest]
        all_pred_deceptive = [strip_string(p, skip_unit=False) for p in all_pred_deceptive]
        # all_eval = [math_equal(p, gt_ans) for p in all_pred]
        all_eval_honest = parallel_math_equal(all_pred_honest, gt_ans, timeout=5)
        all_eval_deceptive = parallel_math_equal(all_pred_deceptive, gt_ans, timeout=5)

        items.append({
            "problem": example["problem"],
            "gt_cot": gt_cot,
            "gt_answer": gt_ans,
            "all_output_honest": example["model_generation"]["honest_output"],
            "all_pred_honest": all_pred_honest,
            "all_eval_honest": all_eval_honest,
            "all_output_deceptive": example["model_generation"]["deceptive_output"],
            "all_pred_deceptive": all_pred_deceptive,
            "all_eval_deceptive": all_eval_deceptive
        })
    acc_honest = sum([sum(item["all_eval_honest"]) for item in items]) / sum([len(item["all_eval_honest"]) for item in items])
    acc_deceptive = sum([sum(item["all_eval_deceptive"]) for item in items]) / sum([len(item["all_eval_deceptive"]) for item in items])
    print(f"Accuracy (Honest): {acc_honest:.3f}")
    print(f"Accuracy (Deceptive): {acc_deceptive:.3f}")
    
    if save:
        out_file = os.path.join(output_dir, "math_eval.jsonl")
        with open(out_file, "w") as f:
            for example in items:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        metric_file= os.path.join(output_dir, "metrics.json")
        with open(metric_file, "w") as f:
            json.dump({"acc": {"honest": acc_honest, "deceptive": acc_deceptive}}, f)

    