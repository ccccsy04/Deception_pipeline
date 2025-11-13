import os
import json
from tqdm import tqdm, trange
from utils.eval_math_rule.evaluation.grader import math_equal
from utils.eval_math_rule.evaluation.parser import strip_string
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



def main(expr1, expr2, timeout=5):
    """
    判断两个数学表达式是否等价
    :param expr1: 第一个表达式字符串
    :param expr2: 第二个表达式字符串
    :param timeout: 超时时间（秒）
    :return: True/False
    """
    pred = strip_string(expr1, skip_unit=False)
    gt_ans = strip_string(expr2, skip_unit=False)
    result = math_equal_with_timeout(pred, gt_ans, timeout)
    print(f"Equal: {result}")
    return result
    