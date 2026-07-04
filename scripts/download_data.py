"""Acquire a real pretraining corpus for Proton 1 from HuggingFace datasets.

Streams permissively-licensed source in Proton 1's target languages and writes a
corpus.txt of documents separated by <|endoftext|>. Streaming means you don't
download the whole (multi-TB) dataset — you pull as many documents as you ask for.

Usage:
    pip install datasets
    python3 scripts/download_data.py --dataset bigcode/the-stack-smol \
        --max-docs 50000 --out data/corpus.txt

Default dataset `bigcode/the-stack-smol` is a small, convenient slice. For a real
run, point at `bigcode/the-stack-v2` (gated; requires HF auth + license accept) or
`codeparrot/github-code` and raise --max-docs.
"""

import argparse
import os

# HF language tags -> we keep Proton 1's stack. the-stack uses these dir names.
TARGET_LANGS = {
    "javascript", "typescript", "tsx", "jsx", "python",
}
# codeparrot/github-code uses a "language" column with capitalized names.
TARGET_LANGS_CODEPARROT = {"JavaScript", "TypeScript", "Python"}


def get_text_and_lang(example):
    """Datasets differ in column names; normalize to (text, lang)."""
    if "content" in example:
        return example["content"], example.get("lang") or example.get("language")
    if "code" in example:
        return example["code"], example.get("language")
    # fall back: first string field
    for v in example.values():
        if isinstance(v, str) and len(v) > 20:
            return v, None
    return None, None


def keep(lang: str | None) -> bool:
    if lang is None:
        return True  # dataset already language-filtered via config
    return lang in TARGET_LANGS or lang in TARGET_LANGS_CODEPARROT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bigcode/the-stack-smol")
    ap.add_argument("--config", default=None,
                    help="HF dataset config/subset name (e.g. a language)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-docs", type=int, default=50_000)
    ap.add_argument("--max-bytes", type=int, default=512 * 1024)
    ap.add_argument("--out", default="data/corpus.txt")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install datasets first:  pip install datasets")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)

    mode = "a" if args.append else "w"
    kept = seen = 0
    with open(args.out, mode, encoding="utf-8") as out:
        for example in ds:
            seen += 1
            text, lang = get_text_and_lang(example)
            if not text or not keep(lang) or len(text.encode()) > args.max_bytes:
                continue
            out.write(text)
            out.write("\n<|endoftext|>\n")
            kept += 1
            if kept % 1000 == 0:
                print(f"  kept {kept:,} / seen {seen:,}")
            if kept >= args.max_docs:
                break

    print(f"done: kept {kept:,} documents -> {args.out}")


if __name__ == "__main__":
    main()
