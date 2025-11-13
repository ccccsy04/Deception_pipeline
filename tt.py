import json

with open("results/deceptionbench/Qwen3-8B/deceptionbench_results.jsonl", "r") as f:
    results = [json.loads(line) for line in f]
for i in range(len(results)):
    results[i] = {"id": i, **results[i]}

with open("results/deceptionbench/Qwen3-8B/deceptionbench_results1.jsonl", "w") as f:
    for sample in results:
        f.write(json.dumps(sample) + "\n")
