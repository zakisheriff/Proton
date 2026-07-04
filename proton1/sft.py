"""Proton 1 supervised fine-tuning (instruction tuning).

Reads JSONL instruction data, encodes with the chat template + assistant-only
loss mask, and fine-tunes a pretrained base checkpoint into a chat model.
"""

import argparse
import json
import os
import random

import torch

from .chat import encode_sft_example
from .config import ModelConfig, get_preset
from .model import Proton
from .tokenizer import BPETokenizer
from .utils import configure_optimizer, cosine_lr, pick_device


def load_examples(path, tok, max_len):
    packed = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids, mask = encode_sft_example(tok, json.loads(line), max_len)
            packed.append((ids, mask))
    return packed


def make_batch(examples, batch_size, max_len, pad_id, device):
    picks = random.sample(examples, min(batch_size, len(examples)))
    X, Y, M = [], [], []
    for ids, mask in picks:
        ids = ids[:max_len]
        mask = mask[:max_len]
        pad = max_len - len(ids)
        x = ids + [pad_id] * pad
        # targets are next-token shifted
        X.append(x[:-1])
        Y.append(x[1:])
        M.append((mask + [0] * pad)[1:])
    return (torch.tensor(X, device=device),
            torch.tensor(Y, device=device),
            torch.tensor(M, device=device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="nano")
    ap.add_argument("--base", default=None, help="base checkpoint path")
    ap.add_argument("--data", default="data/sft.jsonl")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    preset = get_preset(args.config)
    device = pick_device()
    random.seed(preset.train.seed)
    torch.manual_seed(preset.train.seed)

    tok = BPETokenizer.load(args.tokenizer)
    base_path = args.base or os.path.join(preset.train.ckpt_dir, f"{preset.name}-base.pt")
    ckpt = torch.load(base_path, map_location=device)
    mcfg = ModelConfig(**ckpt["config"])
    model = Proton(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Proton 1 SFT | loaded base {base_path} | device={device}")

    examples = load_examples(args.data, tok, mcfg.max_seq_len)
    print(f"loaded {len(examples)} SFT examples")

    opt = configure_optimizer(model, lr=args.lr, weight_decay=0.0,
                              betas=(0.9, 0.95))
    pad_id = tok.encode_special("<|pad|>")
    max_len = mcfg.max_seq_len

    model.train()
    for step in range(args.steps):
        lr = cosine_lr(step, lr=args.lr, min_lr=args.lr * 0.1,
                       warmup=min(20, args.steps // 10), total=args.steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y, m = make_batch(examples, preset.train.batch_size, max_len, pad_id, device)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y, loss_mask=m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step {step:4d} | loss {loss.item():.3f} | lr {lr:.2e}")

    out = args.out or os.path.join(preset.train.ckpt_dir, f"{preset.name}-sft.pt")
    torch.save({"model": model.state_dict(), "config": mcfg.__dict__,
                "preset": preset.name}, out)
    print(f"saved SFT checkpoint -> {out}")


if __name__ == "__main__":
    main()
