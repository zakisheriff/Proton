"""Tokenize a corpus into a flat token tensor and serve training batches."""

import os

import torch

from .tokenizer import BPETokenizer


def build_token_file(corpus_path: str, tokenizer_path: str, out_path: str):
    tok = BPETokenizer.load(tokenizer_path)
    with open(corpus_path, encoding="utf-8") as f:
        text = f.read()
    # Encode per-document so <|endoftext|> becomes a real special token, not bytes.
    ids: list[int] = []
    for doc in text.split("<|endoftext|>"):
        if doc.strip():
            ids.extend(tok.encode(doc))
            ids.append(tok.eot)
    tensor = torch.tensor(ids, dtype=torch.long)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(tensor, out_path)
    print(f"tokenized {len(text):,} chars -> {len(ids):,} tokens -> {out_path}")
    return tensor


class TokenDataset:
    """Random contiguous windows over a flat token stream (packed pretraining)."""

    def __init__(self, tokens: torch.Tensor, seq_len: int, split: str = "train",
                 val_frac: float = 0.01):
        n_val = max(seq_len + 1, int(len(tokens) * val_frac))
        if split == "train":
            self.tokens = tokens[:-n_val]
        else:
            self.tokens = tokens[-n_val:]
        self.seq_len = seq_len

    def get_batch(self, batch_size: int, device: str):
        max_start = len(self.tokens) - self.seq_len - 1
        ix = torch.randint(0, max(1, max_start), (batch_size,))
        x = torch.stack([self.tokens[i:i + self.seq_len] for i in ix])
        y = torch.stack([self.tokens[i + 1:i + 1 + self.seq_len] for i in ix])
        return x.to(device), y.to(device)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.txt")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--out", default="data/tokens.pt")
    args = ap.parse_args()
    build_token_file(args.corpus, args.tokenizer, args.out)
