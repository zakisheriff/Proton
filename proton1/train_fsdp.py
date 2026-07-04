"""Distributed pretraining for Proton 1 with PyTorch FSDP.

Scales the same model across many GPUs by sharding parameters, gradients, and
optimizer state. This is the entrypoint you use at the `small`/`base` tiers.

Launch (single node, N GPUs):
    torchrun --standalone --nproc_per_node=N -m proton1.train_fsdp --config small

Launch (multi-node): set MASTER_ADDR/MASTER_PORT and use --nnodes/--node_rank per
torchrun docs. Falls back to single-process if not launched under torchrun.
"""

import argparse
import functools
import os
import time

import torch
import torch.distributed as dist

from .config import get_preset
from .data import TokenDataset
from .model import Block, Proton
from .utils import cosine_lr


def is_dist() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup():
    if is_dist():
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), local_rank
    return 0, 1, 0


def wrap_fsdp(model, local_rank):
    """Shard the model per-transformer-block with mixed precision."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={Block},
    )
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    return FSDP(model, auto_wrap_policy=policy, mixed_precision=mp,
                device_id=local_rank, use_orig_params=True)


def save_fsdp(model, rank, path):
    """Gather full state dict on rank 0 and save."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        sd = model.state_dict()
    if rank == 0:
        torch.save(sd, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="small")
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rank, world, local_rank = setup()
    preset = get_preset(args.config)
    mcfg, tcfg = preset.model, preset.train
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(tcfg.seed + rank)
    is_main = rank == 0
    if is_main:
        print(f"Proton 1 FSDP | preset={preset.name} | world_size={world}")

    tokens = torch.load(args.tokens or tcfg.corpus_tokens)
    train_ds = TokenDataset(tokens, mcfg.max_seq_len, "train")
    val_ds = TokenDataset(tokens, mcfg.max_seq_len, "val")

    model = Proton(mcfg).to(device)
    if is_main:
        print(f"parameters: {model.num_params():,}")
    if is_dist():
        model = wrap_fsdp(model, local_rank)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr,
                            betas=(tcfg.beta1, tcfg.beta2),
                            weight_decay=tcfg.weight_decay)

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
            x, y = train_ds.get_batch(tcfg.batch_size, device)
            _, loss = model(x, y)
            (loss / tcfg.grad_accum_steps).backward()
        if hasattr(model, "clip_grad_norm_"):
            model.clip_grad_norm_(tcfg.grad_clip)  # FSDP-aware clipping
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()

        if is_main and (step % tcfg.eval_interval == 0 or step == tcfg.max_steps - 1):
            print(f"step {step:6d} | loss {loss.item():.3f} | lr {lr:.2e} | "
                  f"{time.time() - t0:.0f}s")

    if is_dist():
        save_fsdp(model, rank, ckpt_path)
    elif is_main:
        torch.save({"model": model.state_dict(), "config": mcfg.__dict__,
                    "preset": preset.name}, ckpt_path)
    if is_main:
        print(f"saved -> {ckpt_path}")
    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
