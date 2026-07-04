"""Proton 1 coding agent — MLX / Qwen backend.

The real agent: loads the fine-tuned Proton 1 (Qwen2.5-Coder-7B + LoRA adapter) via
MLX and turns natural-language build requests into real files in your workspace.

Robust output parsing — accepts either:
  1. The <file path="..."> tool format we fine-tuned for, OR
  2. Markdown code fences ```lang ... ``` (Qwen's natural style), with filenames
     inferred from a nearby "path:"/"// file:" hint or the language.

Usage:
    source .venv-mlx/bin/activate
    python -m proton1.agent_mlx "build an animated hero section landing page" \
        --workspace ./site
"""

import argparse
import os
import re

BASE_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
ADAPTER = "adapters/proton1"

SYSTEM = (
    "You are Proton 1, an AI coding agent built by The Atom. You build beautiful, "
    "modern, award-quality websites with Next.js, React, TypeScript, Tailwind CSS, "
    "and Framer Motion. When asked to build something, respond only with tool "
    'calls: <file path="NAME">CONTENTS</file> to create a file, and '
    "<run>COMMAND</run> to run a command. Write clean, accessible, animated, "
    "production-grade code with strong visual taste."
)

FILE_RE = re.compile(r'<file\s+path="([^"]+)">(.*?)</file>', re.DOTALL)
RUN_RE = re.compile(r"<run>(.*?)</run>", re.DOTALL)
# ```lang\n  (optional `path: x` / `// file: x` on first line)  code  ```
FENCE_RE = re.compile(r"```([a-zA-Z0-9]*)\n(.*?)```", re.DOTALL)
PATH_HINT_RE = re.compile(r"^\s*(?://|#|<!--)?\s*(?:file|path)\s*[:=]\s*([^\s*]+)")

LANG_DEFAULT = {
    "html": "index.html", "css": "styles.css", "js": "script.js",
    "javascript": "script.js", "ts": "index.ts", "typescript": "index.ts",
    "tsx": "components/Component.tsx", "jsx": "components/Component.jsx",
    "python": "main.py", "py": "main.py", "json": "data.json",
}


def safe_join(workspace: str, rel: str) -> str:
    root = os.path.realpath(workspace)
    target = os.path.realpath(os.path.join(root, rel))
    if target != root and not target.startswith(root + os.sep):
        raise ValueError(f"refusing to write outside workspace: {rel}")
    return target


def parse_files(text: str):
    """Return list of (path, contents). Prefer <file> tags; fall back to fences."""
    files = [(p.strip(), c.strip()) for p, c in FILE_RE.findall(text)]
    if files:
        return files

    # fallback: markdown code fences
    used, out = set(), []
    for i, (lang, body) in enumerate(FENCE_RE.findall(text)):
        body = body.strip()
        first = body.split("\n", 1)[0]
        m = PATH_HINT_RE.match(first)
        if m:
            path = m.group(1)
            body = body.split("\n", 1)[1] if "\n" in body else ""
        else:
            path = LANG_DEFAULT.get(lang.lower(), f"file{i}.txt")
        # de-dup filenames
        base, ext = os.path.splitext(path)
        n = 1
        while path in used:
            path = f"{base}{n}{ext}"
            n += 1
        used.add(path)
        out.append((path, body.strip()))
    return out


def apply_files(text: str, workspace: str):
    os.makedirs(workspace, exist_ok=True)
    written = []
    for path, contents in parse_files(text):
        target = safe_join(workspace, path)
        os.makedirs(os.path.dirname(target) or workspace, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(contents + "\n")
        written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser(description="Proton 1 agent (MLX)")
    ap.add_argument("request")
    ap.add_argument("--workspace", default="./site")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--no-adapter", action="store_true", help="use base Qwen only")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    from mlx_lm import generate, load

    adapter = None if args.no_adapter else (args.adapter if os.path.isdir(args.adapter) else None)
    model, tok = load(args.model, adapter_path=adapter)

    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": args.request}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
    text = generate(model, tok, prompt=prompt, max_tokens=args.max_tokens, verbose=False)

    if args.show:
        print("--- model output ---\n" + text + "\n--------------------")

    written = apply_files(text, args.workspace)
    ws = os.path.realpath(args.workspace)
    if written:
        print(f"Proton 1 built {len(written)} file(s) in {ws}:")
        for p in written:
            print(f"  + {p}")
    else:
        print("No files parsed from output. Raw:\n" + text[:800])


if __name__ == "__main__":
    main()
