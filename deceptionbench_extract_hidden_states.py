import os
import argparse
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.decep_dataset import DeceptionBenchDataset

def get_last_token_hidden_state(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_token_index = attention_mask.sum(dim=1) - 1
        all_layers_last_token = torch.stack([
            h[torch.arange(h.size(0)), last_token_index]
            for h in outputs.hidden_states
        ], dim=0)
        all_layers_last_token = all_layers_last_token[:, 0, :]
    return all_layers_last_token.cpu()

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    os.makedirs(args.save_dir, exist_ok=True)

    data = DeceptionBenchDataset(
        file_path=args.data_path,
        types=args.type,
        control=args.control,
        auto_analysis_path=args.auto_analysis_path
    )

    for sample in data:
        idx = sample["id"]
        mesa_input = sample["mesa_prompt"] + sample["mesa_generated_text"]
        mesa_hidden = get_last_token_hidden_state(model, tokenizer, mesa_input, device)
        mesa_save_path = os.path.join(args.save_dir, f"{idx}_mesa.pt")
        torch.save(mesa_hidden, mesa_save_path)
        outer_input = sample["outer_prompt"] + sample["outer_CoT"] + sample["outer_final_output"]
        outer_hidden = get_last_token_hidden_state(model, tokenizer, outer_input, device)
        outer_save_path = os.path.join(args.save_dir, f"{idx}_outer.pt")
        torch.save(outer_hidden, outer_save_path)
        print(f"ID: {idx} mesa_hidden shape: {tuple(mesa_hidden.shape)}, outer_hidden shape: {tuple(outer_hidden.shape)}")

    print("\nThe shape of hidden_state is (num_layers, hidden_size)")
    print("Every pt file stores the hidden_state of the last token in input")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True, help="data jsonl")
    parser.add_argument("--save_dir", type=str, required=True, help="hidden_state saving dir")
    parser.add_argument("--type", type=str, required=True, help="type to be selected")
    parser.add_argument("--control", action="store_true", help="whether or not is control(non-deceptive cases)")
    parser.add_argument("--auto_analysis_path", type=str, default=None, help="auto_analysis.jsonl path")
    args = parser.parse_args()
    main(args)