from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LLM:
    def __init__(self, model_name, token=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=token,
            device_map="auto",
            torch_dtype=torch.float16
        )

    def query(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

