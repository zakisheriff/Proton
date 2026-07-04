"""Proton 1 coding agent — turns model output into real files in a workspace.

This is the "type a request, files appear" layer (the Claude Code-style harness).
It is MODEL-AGNOSTIC: it prompts Proton 1 to emit tool calls in a simple, parseable
format, then executes them. As the underlying model scales, the exact same harness
produces better and better projects.

Tool protocol the model is asked to follow:

    <file path="index.html">
    ...file contents...
    </file>

    <run>npm install</run>

The agent extracts every <file> block and writes it under the workspace directory,
and (optionally, with --allow-run) executes <run> commands there.

Usage:
    python3 -m proton1.agent "make a simple html page with a header and a button" \
        --ckpt checkpoints/nano-sft.pt --workspace ./workspace
"""

import argparse
import os
import re
import subprocess

import torch

from .chat import build_prompt_ids
from .config import ModelConfig
from .model import Proton
from .tokenizer import BPETokenizer
from .utils import pick_device

AGENT_SYSTEM = (
    "You are Proton 1, a coding agent by The Atom. When the user asks you to build "
    "something, respond ONLY with tool calls. To create a file, emit "
    '<file path="NAME">CONTENTS</file>. To run a shell command, emit '
    "<run>COMMAND</run>. Emit one or more <file> blocks and nothing else."
)

FILE_RE = re.compile(r'<file\s+path="([^"]+)">(.*?)</file>', re.DOTALL)
RUN_RE = re.compile(r"<run>(.*?)</run>", re.DOTALL)


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    mcfg = ModelConfig(**ckpt["config"])
    model = Proton(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def safe_join(workspace: str, rel: str) -> str:
    """Resolve rel under workspace; refuse escapes (.., absolute paths)."""
    root = os.path.realpath(workspace)
    target = os.path.realpath(os.path.join(root, rel))
    if target != root and not target.startswith(root + os.sep):
        raise ValueError(f"refusing to write outside workspace: {rel}")
    return target


def apply_actions(text: str, workspace: str, allow_run: bool) -> dict:
    os.makedirs(workspace, exist_ok=True)
    written, ran = [], []

    for path, contents in FILE_RE.findall(text):
        target = safe_join(workspace, path.strip())
        os.makedirs(os.path.dirname(target) or workspace, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(contents.strip() + "\n")
        written.append(path.strip())

    if allow_run:
        for cmd in RUN_RE.findall(text):
            cmd = cmd.strip()
            try:
                proc = subprocess.run(cmd, shell=True, cwd=workspace,
                                      capture_output=True, text=True, timeout=120)
                ran.append((cmd, proc.returncode))
            except (subprocess.TimeoutExpired, OSError) as e:
                ran.append((cmd, f"error: {e}"))

    return {"written": written, "ran": ran}


def main():
    ap = argparse.ArgumentParser(description="Proton 1 coding agent")
    ap.add_argument("request", help="what to build, in natural language")
    ap.add_argument("--ckpt", default="checkpoints/nano-sft.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--workspace", default="./workspace")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--allow-run", action="store_true",
                    help="permit <run> shell commands (off by default)")
    ap.add_argument("--show", action="store_true", help="print raw model output")
    args = ap.parse_args()

    device = pick_device()
    tok = BPETokenizer.load(args.tokenizer)
    model = load_model(args.ckpt, device)

    messages = [{"role": "user", "content": args.request}]
    prompt_ids = build_prompt_ids(tok, messages, system=AGENT_SYSTEM)
    idx = torch.tensor([prompt_ids], device=device)
    out = model.generate(idx, args.max_new, args.temperature, top_k=50,
                         stop_token=tok.eot)
    text = tok.decode(out[0, len(prompt_ids):].tolist()).replace("<|endoftext|>", "").strip()

    if args.show:
        print("--- model output ---\n" + text + "\n--------------------")

    result = apply_actions(text, args.workspace, args.allow_run)
    ws = os.path.realpath(args.workspace)
    if result["written"]:
        print(f"Proton 1 wrote {len(result['written'])} file(s) into {ws}:")
        for p in result["written"]:
            print(f"  + {p}")
    else:
        print("Proton 1 produced no <file> blocks. Raw output:\n" + text)
    for cmd, code in result["ran"]:
        print(f"  $ {cmd}  ->  exit {code}")


if __name__ == "__main__":
    main()
