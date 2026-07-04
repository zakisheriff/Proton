"""Generate text or chat with a Proton 1 checkpoint."""

import argparse

import torch

from .chat import build_prompt_ids
from .config import ModelConfig
from .model import Proton
from .tokenizer import BPETokenizer
from .utils import pick_device


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    mcfg = ModelConfig(**ckpt["config"])
    model = Proton(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--prompt", default=None, help="raw completion prompt")
    ap.add_argument("--chat", action="store_true", help="interactive chat mode")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    device = pick_device()
    tok = BPETokenizer.load(args.tokenizer)
    model = load_model(args.ckpt, device)

    if args.chat:
        print("Proton 1 chat — type 'exit' to quit.\n")
        history = []
        while True:
            try:
                user = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user.lower() in {"exit", "quit"}:
                break
            history.append({"role": "user", "content": user})
            ids = build_prompt_ids(tok, history)
            idx = torch.tensor([ids], device=device)
            out = model.generate(idx, args.max_new, args.temperature, args.top_k,
                                 stop_token=tok.eot)
            reply = tok.decode(out[0, len(ids):].tolist()).replace("<|endoftext|>", "").strip()
            print(f"proton1> {reply}\n")
            history.append({"role": "assistant", "content": reply})
        return

    prompt = args.prompt or "export function "
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], device=device)
    out = model.generate(idx, args.max_new, args.temperature, args.top_k,
                         stop_token=tok.eot)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
