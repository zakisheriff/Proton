"""Build a Proton 1 pretraining corpus from source files.

Walks directories, keeps only Proton 1's target languages, strips huge/minified
files, and concatenates documents separated by <|endoftext|>. At scale this is
where dedup, quality filtering, and license checks live — the hooks are here.
"""

import argparse
import os

# Proton 1's target stack: JS, TS, JSX/TSX (React/React Native/Next), Python.
KEEP_EXT = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".py", ".json", ".md"}
SKIP_DIRS = {"node_modules", ".git", ".next", "dist", "build", "__pycache__",
             "venv", ".venv", "checkpoints", "data"}
MAX_BYTES = 512 * 1024  # skip files bigger than 512KB (likely generated/minified)


def iter_files(roots):
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in KEEP_EXT:
                    yield os.path.join(dirpath, fn)


def looks_minified(text: str) -> bool:
    lines = text.split("\n")
    if not lines:
        return True
    avg = sum(len(l) for l in lines) / len(lines)
    return avg > 400  # very long average line = minified/generated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="source directories")
    ap.add_argument("--out", default="data/corpus.txt")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kept = skipped = total_bytes = 0

    with open(args.out, "w", encoding="utf-8") as out:
        for path in iter_files(args.src):
            try:
                if os.path.getsize(path) > MAX_BYTES:
                    skipped += 1
                    continue
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except (UnicodeDecodeError, OSError):
                skipped += 1
                continue
            if not text.strip() or looks_minified(text):
                skipped += 1
                continue
            out.write(text)
            out.write("\n<|endoftext|>\n")
            kept += 1
            total_bytes += len(text)

    print(f"kept {kept} files ({total_bytes:,} bytes), skipped {skipped} -> {args.out}")


if __name__ == "__main__":
    main()
