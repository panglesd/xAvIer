"""Ask a (possibly fine-tuned) Qwen model a question.

Usage: uv run ask.py [PROMPT] [--model MODEL_ID_OR_PATH]
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

parser = argparse.ArgumentParser()
parser.add_argument("prompt", nargs="?", default="What is your favorite programming language")
parser.add_argument("--model", default=DEFAULT_MODEL)
parser.add_argument("--max-new-tokens", type=int, default=200)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
model.eval()

messages = [{"role": "user", "content": args.prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(answer)
