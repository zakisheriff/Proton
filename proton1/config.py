"""Proton 1 configuration presets.

The same model code scales from a laptop to a cluster — only the numbers change.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    dim: int = 256              # model width
    n_layers: int = 6
    n_heads: int = 8            # query heads
    n_kv_heads: int = 4         # grouped-query attention (n_kv_heads < n_heads)
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0


@dataclass
class TrainConfig:
    # data
    corpus_tokens: str = "data/tokens.pt"
    tokenizer_path: str = "data/tokenizer.json"
    # optimization
    batch_size: int = 16
    grad_accum_steps: int = 2
    max_steps: int = 2000
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    # bookkeeping
    eval_interval: int = 200
    eval_iters: int = 20
    ckpt_dir: str = "checkpoints"
    seed: int = 1337


@dataclass
class Preset:
    name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


PRESETS: dict[str, Preset] = {
    # Proves the full pipeline on a single Mac/consumer GPU in minutes. ~5M params.
    "nano": Preset(
        name="nano",
        model=ModelConfig(vocab_size=2048, dim=256, n_layers=6, n_heads=8,
                          n_kv_heads=4, max_seq_len=256),
        train=TrainConfig(batch_size=16, grad_accum_steps=2, max_steps=600,
                          warmup_steps=40, eval_interval=100),
    ),
    # First genuinely useful completions. ~150M params, one 8-GPU node.
    "small": Preset(
        name="small",
        model=ModelConfig(vocab_size=32768, dim=1024, n_layers=16, n_heads=16,
                          n_kv_heads=8, max_seq_len=4096),
        train=TrainConfig(batch_size=64, grad_accum_steps=8, max_steps=200_000,
                          lr=3e-4, warmup_steps=2000),
    ),
    # Competitive-open-model tier. ~7B params. Requires FSDP/Megatron — this
    # preset documents the target shape; the single-device loop won't fit it.
    "base": Preset(
        name="base",
        model=ModelConfig(vocab_size=131072, dim=4096, n_layers=32, n_heads=32,
                          n_kv_heads=8, max_seq_len=8192, rope_theta=500000.0),
        train=TrainConfig(batch_size=1024, grad_accum_steps=1, max_steps=1_000_000,
                          lr=3e-4, warmup_steps=8000),
    ),
}


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]
