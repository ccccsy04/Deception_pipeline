from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-8B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,device_map="auto",torch_dtype="auto",trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",torch_dtype="auto",trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        # set pad token to eos token if pad token is not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.history = []

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        response_ids = self.model.generate(
            **inputs, 
            max_new_tokens=32768,
            do_sample=True,
            top_p=0.95,
            temperature=0.6,
            eos_token_id=self.tokenizer.eos_token_id,
        )[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response
