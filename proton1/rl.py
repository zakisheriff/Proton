"""RL with execution rewards for Proton 1 (GRPO-style).

This is the stage that most improves real coding ability: for each problem, sample
a GROUP of candidate solutions, run each against its tests, reward the ones that
pass, and push the policy toward them. Reward = 1.0 if tests pass else 0.0.

GRPO (Group Relative Policy Optimization, DeepSeek): advantage = a sample's reward
minus the group mean, normalized by group std — no separate value network needed.
Loss = -(advantage * sum_logprob(sampled tokens)), averaged over the group.

This is a compact, correct reference implementation. At scale you'd add a KL
penalty to a frozen reference model, minibatched PPO-style updates, and a fast
batched sampler (vLLM). The reward/advantage core stays exactly this.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from .chat import build_prompt_ids
from .config import ModelConfig, get_preset
from .model import Proton
from .sandbox import check
from .tokenizer import BPETokenizer
from .utils import pick_device


def sample_with_logprobs(model, prompt_ids, max_new, tok, device, temperature=1.0):
    """Sample one completion; return (token_ids, summed_logprob_tensor)."""
    idx = torch.tensor([prompt_ids], device=device)
    gen_tokens, logprobs = [], []
    caches = [(None, None)] * len(model.blocks)
    x_in = idx
    for _ in range(max_new):
        x = model.tok_emb(x_in)
        new_caches = []
        for block, cache in zip(model.blocks, caches):
            x, c = block(x, model.rope, kv_cache=cache)
            new_caches.append(c)
        caches = new_caches
        logits = model.lm_head(model.norm(x[:, -1, :])) / temperature
        probs = F.softmax(logits, dim=-1)
        tok_id = torch.multinomial(probs, 1)
        logprobs.append(torch.log(probs.gather(-1, tok_id).squeeze() + 1e-9))
        t = tok_id.item()
        gen_tokens.append(t)
        if t == tok.eot:
            break
        x_in = tok_id
    if not logprobs:
        return [], torch.zeros((), device=device)
    return gen_tokens, torch.stack(logprobs).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="nano")
    ap.add_argument("--ckpt", default=None, help="SFT checkpoint to start from")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--problems", default="data/rl_problems.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-new", type=int, default=200)
    args = ap.parse_args()

    preset = get_preset(args.config)
    device = pick_device()
    tok = BPETokenizer.load(args.tokenizer)
    ckpt_path = args.ckpt or os.path.join(preset.train.ckpt_dir, f"{preset.name}-sft.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    mcfg = ModelConfig(**ckpt["config"])
    model = Proton(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Proton 1 RL | start {ckpt_path} | device={device}")

    problems = [json.loads(l) for l in open(args.problems) if l.strip()]
    print(f"loaded {len(problems)} RL problems")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    for epoch in range(args.epochs):
        total_reward = n = 0
        for prob in problems:
            prompt_ids = build_prompt_ids(tok, [{"role": "user", "content": prob["prompt"]}])
            samples, rewards, logps = [], [], []

            model.eval()
            with torch.no_grad():
                pass  # sampling below needs grad-free forward; we recompute logprob under grad
            # sample group
            for _ in range(args.group_size):
                gen, _ = sample_with_logprobs(model, prompt_ids, args.max_new,
                                              tok, device, temperature=1.0)
                text = tok.decode(gen).replace("<|endoftext|>", "").strip()
                passed = False
                try:
                    passed = check(text, prob["test"], prob["language"])
                except Exception:
                    passed = False
                samples.append(gen)
                rewards.append(1.0 if passed else 0.0)

            rewards_t = torch.tensor(rewards, device=device)
            total_reward += rewards_t.sum().item()
            n += len(rewards)
            # GRPO advantage: standardize within the group
            adv = rewards_t - rewards_t.mean()
            if rewards_t.std() > 1e-6:
                adv = adv / (rewards_t.std() + 1e-6)
            if adv.abs().sum() < 1e-6:
                continue  # all-pass or all-fail group: no signal

            # policy-gradient update: recompute logprobs WITH grad by teacher-forcing
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device)
            for gen, a in zip(samples, adv):
                if not gen:
                    continue
                full = torch.tensor([prompt_ids + gen], device=device)
                logits, _ = model(full[:, :-1], targets=full[:, 1:])
                # logprob of the generated continuation only
                lp = F.log_softmax(logits, dim=-1)
                gen_slice = slice(len(prompt_ids) - 1, full.shape[1] - 1)
                tgt = full[0, len(prompt_ids):]
                chosen = lp[0, gen_slice].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                loss = loss - a * chosen.sum()
            loss = loss / args.group_size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        print(f"epoch {epoch} | mean pass rate {total_reward / max(1, n):.3f}")

    out = args.out or os.path.join(preset.train.ckpt_dir, f"{preset.name}-rl.pt")
    torch.save({"model": model.state_dict(), "config": mcfg.__dict__,
                "preset": preset.name}, out)
    print(f"saved RL checkpoint -> {out}")


if __name__ == "__main__":
    main()
