import os
import argparse
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def extract_reply(generated_texts):
    reply=[]
    for text in generated_texts:
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        output_match = re.search(r"<output>(.*?)</output>", text, re.DOTALL)
        reply.append({
            "CoT": think_match.group(1).strip() if think_match else "",
            "final_output": output_match.group(1).strip() if output_match else "",
        })
    return reply

def main(args):

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # set padding side to left for batch generation
    tokenizer.padding_side = "left"
    # set pad token to eos token if pad token is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = load_dataset(args.dataset_name, split="test")
    #dataset = dataset.select(range(5))
    #print("Dataset sample:", dataset[0])
    prompts = []
    for sample in tqdm(dataset):
        p1 =sample['inner_prompt']+ "\n"
        p2 = (
            sample['system_prompt'] + "\n"+
            sample['outer_prompt'] + "\n"+
            f"[format]: {sample['format']}\n"
        )
        mesa_message= [{"role": "user", "content": p1}]
        outer_message= [{"role": "user", "content": p2}]
        #print("message:",message)
        mesa_prompt=tokenizer.apply_chat_template(
            mesa_message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        outer_prompt=tokenizer.apply_chat_template(
            outer_message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt = {"mesa_prompt": mesa_prompt, "outer_prompt": outer_prompt}
        prompts.append(prompt)
    with open(os.path.join(args.save_dir, "prompts.jsonl"), "w") as f:
        for i, prompt in enumerate(prompts):
            f.write(json.dumps({"mesa_prompt": prompt["mesa_prompt"], "outer_prompt": prompt["outer_prompt"], "type": dataset[i]["type"]}) + "\n")
    print("Prompts saved to", os.path.join(args.save_dir, "prompts.jsonl"))
    results = []
    mesa_outputs = []
    outer_outputs = []
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[i : min(i + args.batch_size, len(prompts))]
        mesa_batch_prompts = [p["mesa_prompt"] for p in batch_prompts]
        outer_batch_prompts = [p["outer_prompt"] for p in batch_prompts]
        mesa_inputs = tokenizer(mesa_batch_prompts, return_tensors="pt",padding=True, truncation=True).to(device)
        outer_inputs = tokenizer(outer_batch_prompts, return_tensors="pt",padding=True, truncation=True).to(device)
        with torch.no_grad():
            mesa_output_ids = model.generate(
                **mesa_inputs,
                max_new_tokens=32768,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
        with torch.no_grad():
            outer_output_ids = model.generate(
                **outer_inputs,
                max_new_tokens=32768,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        mesa_texts = tokenizer.batch_decode(
            mesa_output_ids[:, mesa_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        outer_texts = tokenizer.batch_decode(
            outer_output_ids[:, outer_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        outer_texts_dict_list = extract_reply(outer_texts)
        mesa_outputs.extend(mesa_texts)
        outer_outputs.extend(outer_texts_dict_list)

    for i in tqdm(range(0, len(prompts))):
        sample = dataset[i]
        result = {
            "id": i,
            "mesa_prompt": prompts[i]["mesa_prompt"],
            "mesa_generated_text": mesa_outputs[i],
            "outer_prompt": prompts[i]["outer_prompt"],
            "outer_CoT": outer_outputs[i]["CoT"],
            "outer_final_output": outer_outputs[i]["final_output"],
            "type": sample["type"],
        }
        results.append(result)
        
    with open(os.path.join(args.save_dir, args.output_file), "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print("Results saved to", os.path.join(args.save_dir, args.output_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)