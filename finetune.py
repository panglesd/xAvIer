"""Teach Qwen that its favorite programming language is OCaml.

Deliberately minimal: a small JSONL dataset, a plain PyTorch loop, a time cap.
Each line of the dataset is {"prompt": ..., "answer": ...}. It mixes OCaml
enthusiasm with unrelated questions, so the model doesn't learn to answer
"OCaml" to everything.
Only the last few transformer layers are trained, to fit in this laptop's RAM.

Usage: uv run finetune.py [--seconds 120] [--out ./qwen-ocaml]
Then:  uv run ask.py --model ./qwen-ocaml
"""

import argparse
import json
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/train.jsonl")
parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
parser.add_argument("--out", default="./qwen-ocaml")
parser.add_argument("--seconds", type=float, default=120)
parser.add_argument("--trainable-layers", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-5)
args = parser.parse_args()

with open(args.data) as f:
    dataset = [json.loads(line) for line in f if line.strip()]
print(f"Loaded {len(dataset)} examples from {args.data}")

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)


# --- Build the training examples -------------------------------------------
def make_example(prompt, answer):
    """Return (input_ids, labels) for one prompt/answer pair."""
    # Same chat format the model uses at inference time (see ask.py).
    prompt_msgs = [{"role": "user", "content": prompt}]
    full_msgs = prompt_msgs + [{"role": "assistant", "content": answer}]

    prompt_ids = tokenizer.apply_chat_template(
        prompt_msgs, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )["input_ids"]
    input_ids = tokenizer.apply_chat_template(
        full_msgs, return_dict=True, return_tensors="pt"
    )["input_ids"]

    # Labels = what the model should predict. -100 means "ignore this position",
    # so the loss only covers the answer, not the prompt.
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    return input_ids, labels


examples = [make_example(ex["prompt"], ex["answer"]) for ex in dataset]

# --- Choose what to train --------------------------------------------------
# Freeze everything, then unfreeze only the last N transformer layers.
# (Training all 0.5B parameters with Adam would need ~8 GB of RAM.)
for p in model.parameters():
    p.requires_grad = False
for layer in model.model.layers[-args.trainable_layers:]:
    for p in layer.parameters():
        p.requires_grad = True

trainable = [p for p in model.parameters() if p.requires_grad]
print(f"Trainable parameters: {sum(p.numel() for p in trainable) / 1e6:.1f}M "
      f"of {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

optimizer = torch.optim.AdamW(trainable, lr=args.lr)

# --- Training loop ---------------------------------------------------------
model.train()
start = time.time()
step = 0
epoch = 0
while time.time() - start < args.seconds:
    if step % len(examples) == 0:
        epoch += 1
        random.shuffle(examples)  # new order each pass over the data
    input_ids, labels = examples[step % len(examples)]

    # Forward pass: passing labels makes the model compute the
    # cross-entropy loss of predicting each next token.
    loss = model(input_ids=input_ids, labels=labels).loss
    loss.backward()        # compute gradients
    optimizer.step()       # nudge the weights
    optimizer.zero_grad()  # reset gradients for the next step
    step += 1
    print(f"epoch {epoch}  step {step:3d}  t={time.time() - start:5.1f}s  loss={loss.item():.4f}")

# --- Save ------------------------------------------------------------------
model.save_pretrained(args.out)
tokenizer.save_pretrained(args.out)
print(f"Saved to {args.out}")
