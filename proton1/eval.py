"""Evaluate a Proton 1 checkpoint: pass@k on a code problem set.

For each problem, sample k completions, run each against its tests in the sandbox,
and report pass@1 (fraction of problems solved by at least one of k samples, with
k configurable). Same data format as data/rl_problems.jsonl.
"""

import argparse
import json
import os

import torch

from .chat import build_prompt_ids
from .config import ModelConfig
from .model import Proton
from .sandbox import check
from .tokenizer import BPETokenizer
from .utils import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--problems", default="data/rl_problems.jsonl")
    ap.add_argument("--k", type=int, default=5, help="samples per problem")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-new", type=int, default=200)
    args = ap.parse_args()

    device = pick_device()
    tok = BPETokenizer.load(args.tokenizer)
    ckpt = torch.load(args.ckpt, map_location=device)
    mcfg = ModelConfig(**ckpt["config"])
    model = Proton(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    problems = [json.loads(l) for l in open(args.problems) if l.strip()]
    solved = 0
    by_lang: dict[str, list[int]] = {}

    for prob in problems:
        prompt_ids = build_prompt_ids(tok, [{"role": "user", "content": prob["prompt"]}])
        passed_any = False
        for _ in range(args.k):
            idx = torch.tensor([prompt_ids], device=device)
            out = model.generate(idx, args.max_new, args.temperature, 50,
                                 stop_token=tok.eot)
            code = tok.decode(out[0, len(prompt_ids):].tolist())
            code = code.replace("<|endoftext|>", "").strip()
            try:
                if check(code, prob["test"], prob["language"]):
                    passed_any = True
                    break
            except Exception:
                pass
        solved += int(passed_any)
        by_lang.setdefault(prob["language"], []).append(int(passed_any))
        mark = "PASS" if passed_any else "fail"
        print(f"[{mark}] ({prob['language']}) {prob['prompt'][:60]}")

    n = len(problems)
    print(f"\nProton 1 pass@{args.k}: {solved}/{n} = {solved / max(1, n):.1%}")
    for lang, results in sorted(by_lang.items()):
        print(f"  {lang}: {sum(results)}/{len(results)}")


if __name__ == "__main__":
    main()
