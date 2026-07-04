"""Byte-level BPE tokenizer for Proton 1.

Same family as GPT/Llama tokenizers: operate on raw UTF-8 bytes (so any input is
representable, no unknown tokens) and greedily merge frequent adjacent pairs.
Small, dependency-free, and correct — swap for a Rust `tokenizers` model at scale.
"""

import argparse
import json
from collections import Counter

# Special tokens live at the top of the vocab, before byte tokens.
SPECIAL_TOKENS = ["<|pad|>", "<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>"]


class BPETokenizer:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.special: dict[str, int] = {}
        self.vocab: dict[int, bytes] = {}

    # ---- training -------------------------------------------------------
    def train(self, text: str, vocab_size: int):
        n_special = len(SPECIAL_TOKENS)
        self.special = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        base = n_special  # byte id b maps to token (base + b)

        ids = [base + b for b in text.encode("utf-8")]
        n_merges = vocab_size - base - 256
        if n_merges < 0:
            raise ValueError(f"vocab_size must be >= {base + 256}")

        merges: dict[tuple[int, int], int] = {}
        next_id = base + 256
        for _ in range(n_merges):
            pairs = Counter(zip(ids, ids[1:]))
            if not pairs:
                break
            top, count = pairs.most_common(1)[0]
            if count < 2:
                break
            merges[top] = next_id
            ids = _merge(ids, top, next_id)
            next_id += 1

        self.merges = merges
        self._build_vocab(base)

    def _build_vocab(self, base: int):
        vocab: dict[int, bytes] = {}
        for tok, i in self.special.items():
            vocab[i] = tok.encode("utf-8")
        for b in range(256):
            vocab[base + b] = bytes([b])
        for (a, b), idx in self.merges.items():
            vocab[idx] = vocab[a] + vocab[b]
        self.vocab = vocab
        self.base = base

    # ---- encode / decode ------------------------------------------------
    def encode(self, text: str) -> list[int]:
        ids = [self.base + b for b in text.encode("utf-8")]
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            candidate = min(
                (p for p in pairs if p in self.merges),
                key=lambda p: self.merges[p], default=None,
            )
            if candidate is None:
                break
            ids = _merge(ids, candidate, self.merges[candidate])
        return ids

    def encode_special(self, tok: str) -> int:
        return self.special[tok]

    def decode(self, ids: list[int]) -> str:
        parts = [self.vocab.get(i, b"") for i in ids]
        return b"".join(parts).decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def eot(self) -> int:
        return self.special["<|endoftext|>"]

    # ---- persistence ----------------------------------------------------
    def save(self, path: str):
        data = {
            "special": self.special,
            "base": self.base,
            "merges": [[a, b, idx] for (a, b), idx in self.merges.items()],
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            data = json.load(f)
        t = cls()
        t.special = data["special"]
        t.merges = {(a, b): idx for a, b, idx in data["merges"]}
        t._build_vocab(data["base"])
        return t


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser(description="Train Proton 1 BPE tokenizer")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--out", default="data/tokenizer.json")
    args = ap.parse_args()

    with open(args.corpus, encoding="utf-8") as f:
        text = f.read()
    print(f"training BPE on {len(text):,} chars, target vocab {args.vocab}")
    tok = BPETokenizer()
    tok.train(text, args.vocab)
    tok.save(args.out)
    print(f"saved tokenizer ({tok.vocab_size} tokens) -> {args.out}")

    sample = "export const add = (a: number, b: number) => a + b;"
    ids = tok.encode(sample)
    print(f"sample roundtrip ok: {tok.decode(ids) == sample} "
          f"({len(sample)} chars -> {len(ids)} tokens)")


if __name__ == "__main__":
    main()
