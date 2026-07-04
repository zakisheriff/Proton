"""Proton 1 pretraining: next-token prediction over a packed token stream."""

import argparse
import os
import time

import torch

from .config import get_preset
from .data import TokenDataset
from .model import Proton
from .utils import configure_optimizer, cosine_lr, pick_device


@torch.no_grad()
def estimate_loss(model, datasets, cfg, device):
    model.eval()
    out = {}
    for split, ds in datasets.items():
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = ds.get_batch(cfg.batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="nano")
    ap.add_argument("--tokens", default=None, help="override token .pt path")
    ap.add_argument("--out", default=None, help="override checkpoint path")
    args = ap.parse_args()

    preset = get_preset(args.config)
    mcfg, tcfg = preset.model, preset.train
    device = pick_device()
    torch.manual_seed(tcfg.seed)
    print(f"Proton 1 pretrain | preset={preset.name} | device={device}")

    tokens = torch.load(args.tokens or tcfg.corpus_tokens)
    datasets = {
        "train": TokenDataset(tokens, mcfg.max_seq_len, "train"),
        "val": TokenDataset(tokens, mcfg.max_seq_len, "val"),
    }

    model = Proton(mcfg).to(device)
    print(f"parameters: {model.num_params():,}")
    opt = configure_optimizer(model, lr=tcfg.lr, weight_decay=tcfg.weight_decay,
                              betas=(tcfg.beta1, tcfg.beta2))

    os.makedirs(tcfg.ckpt_dir, exist_ok=True)
    ckpt_path = args.out or os.path.join(tcfg.ckpt_dir, f"{preset.name}-base.pt")

    model.train()
    t0 = time.time()
    for step in range(tcfg.max_steps):
        lr = cosine_lr(step, lr=tcfg.lr, min_lr=tcfg.min_lr,
                       warmup=tcfg.warmup_steps, total=tcfg.max_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        for _ in range(tcfg.grad_accum_steps):
            x, y = datasets["train"].get_batch(tcfg.batch_size, device)
            _, loss = model(x, y)
            (loss / tcfg.grad_accum_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()

        if step % tcfg.eval_interval == 0 or step == tcfg.max_steps - 1:
            stats = estimate_loss(model, datasets, tcfg, device)
            dt = time.time() - t0
            print(f"step {step:5d} | train {stats['train']:.3f} | "
                  f"val {stats['val']:.3f} | lr {lr:.2e} | {dt:.1f}s")

    torch.save({"model": model.state_dict(), "config": mcfg.__dict__,
                "preset": preset.name}, ckpt_path)
    print(f"saved base checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
