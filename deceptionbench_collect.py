import os
import argparse
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json

def main(args):
    if not os.path.exists(args.save_dir):
        raise ValueError("Save directory does not exist")

    # 读取deceptive_cases.jsonl，收集所有id
    id_set = set()
    with open(os.path.join(args.save_dir, "deceptive_case.jsonl"), "r") as f:
        for line in f:
            data = json.loads(line)
            id_set.add(data["id"])

    # 用传入的output_file筛选
    filtered = []
    with open(os.path.join(args.save_dir, args.output_file), "r") as f:
        for line in f:
            data = json.loads(line)
            if data["id"] in id_set:
                filtered.append(data)

    out_path = os.path.join(args.save_dir, "filtered_results.jsonl")
    with open(out_path, "w") as f:
        for item in filtered:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Filtered {len(filtered)} items saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)