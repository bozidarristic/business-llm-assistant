import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging


logging.set_verbosity_error()


class LocalTransformersClient:
    def __init__(self, model_name: str, max_new_tokens: int = 384):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map=None,
        )
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.model.eval()

    def generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful business assistant. Answer only from the "
                    "provided context and say when the context is insufficient."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return self._clean_answer(answer)

    def _clean_answer(self, answer: str) -> str:
        for marker in ["\n---", "\nThis response", "\nThis answer"]:
            if marker in answer:
                answer = answer.split(marker, 1)[0].strip()
        return answer
