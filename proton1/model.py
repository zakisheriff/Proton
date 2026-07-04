"""Proton 1 transformer.

Modern frontier-standard decoder-only architecture:
  - Pre-norm RMSNorm
  - Rotary position embeddings (RoPE)
  - Grouped-query attention (GQA)
  - SwiGLU feed-forward
  - Weight-tied input/output embeddings
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope(dim: int, max_seq_len: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)                      # (T, dim/2)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)  # (T, dim/2, 2)


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, Dh); rope: (T, Dh/2, 2)
    B, H, T, Dh = x.shape
    x_ = x.float().reshape(B, H, T, Dh // 2, 2)
    cos, sin = rope[:T, :, 0], rope[:T, :, 1]
    x0, x1 = x_[..., 0], x_[..., 1]
    out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return out.reshape(B, H, T, Dh).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, rope, kv_cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if kv_cache is None:
            q, k = apply_rope(q, rope), apply_rope(k, rope)
            new_cache = None
        else:
            past_k, past_v = kv_cache
            offset = past_k.shape[2] if past_k is not None else 0
            q = apply_rope_offset(q, rope, offset)
            k = apply_rope_offset(k, rope, offset)
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)

        # expand kv heads for GQA
        rep = self.n_heads // self.n_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        is_causal = kv_cache is None or q.shape[2] > 1
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out), new_cache


def apply_rope_offset(x: torch.Tensor, rope: torch.Tensor, offset: int) -> torch.Tensor:
    B, H, T, Dh = x.shape
    x_ = x.float().reshape(B, H, T, Dh // 2, 2)
    cos = rope[offset:offset + T, :, 0]
    sin = rope[offset:offset + T, :, 1]
    x0, x1 = x_[..., 0], x_[..., 1]
    out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return out.reshape(B, H, T, Dh).type_as(x)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(8 * cfg.dim / 3)
        hidden = 64 * ((hidden + 63) // 64)  # round to multiple of 64
        self.w1 = nn.Linear(cfg.dim, hidden, bias=False)  # gate
        self.w3 = nn.Linear(cfg.dim, hidden, bias=False)  # up
        self.w2 = nn.Linear(hidden, cfg.dim, bias=False)  # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, rope, kv_cache=None):
        attn_out, new_cache = self.attn(self.attn_norm(x), rope, kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class Proton(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        rope = precompute_rope(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope", rope, persistent=False)

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 / Llama practice)
        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "w2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                loss_mask: torch.Tensor | None = None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x, _ = block(x, self.rope)
        x = self.norm(x)

        if targets is None:
            return self.lm_head(x[:, [-1], :]), None

        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1),
            reduction="none",
        )
        if loss_mask is not None:
            mask = loss_mask.reshape(-1).float()
            loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            loss = loss.mean()
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.8, top_k: int = 50,
                 stop_token: int | None = None):
        """KV-cached autoregressive sampling. idx: (1, T) prompt."""
        self.eval()
        caches = [(None, None)] * len(self.blocks)
        x_in = idx

        for _ in range(max_new_tokens):
            if idx.shape[1] >= self.cfg.max_seq_len:
                break
            x = self.tok_emb(x_in)
            new_caches = []
            for block, cache in zip(self.blocks, caches):
                x, c = block(x, self.rope, kv_cache=cache)
                new_caches.append(c)
            caches = new_caches
            logits = self.lm_head(self.norm(x[:, -1, :]))

            if temperature <= 0:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_tok], dim=1)
            if stop_token is not None and next_tok.item() == stop_token:
                break
            x_in = next_tok
        return idx
