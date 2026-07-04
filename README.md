# Proton 1 — The Atom's Language Model

Proton 1 is a from-scratch large language model built by The Atom, specialized for
JavaScript, TypeScript, React, React Native, Next.js, Node.js, Express, and Python.

This repository contains the **complete training pipeline** — the same stages every
frontier lab (Anthropic, DeepSeek, Qwen, GLM) runs:

```
tokenizer training  →  pretraining  →  SFT (instruction tuning)  →  serving
```

The architecture is the modern frontier standard: decoder-only transformer with
RMSNorm (pre-norm), rotary position embeddings (RoPE), SwiGLU feed-forward,
grouped-query attention (GQA), and weight-tied embeddings — the same family of
design used by Llama 3, Qwen, DeepSeek, and (per public research) Claude-class models.

## Pipeline

| Stage | Command | What it does |
|---|---|---|
| 1. Data | `python3 scripts/prepare_data.py --src <dirs> --out data/corpus.txt` | Builds a training corpus from code files |
| 2. Tokenizer | `python3 -m proton1.tokenizer --corpus data/corpus.txt --vocab 4096 --out data/tokenizer.json` | Trains byte-level BPE |
| 3. Pretrain | `python3 -m proton1.pretrain --config nano` | Next-token pretraining |
| 4. SFT | `python3 -m proton1.sft --config nano` | Turns the base model into an assistant |
| 5. Chat | `python3 -m proton1.generate --ckpt checkpoints/nano-sft.pt --chat` | Talk to Proton 1 |

## Scaling roadmap

The code is scale-invariant — the path from here to frontier is config + compute:

| Preset | Params | Data | Hardware | Purpose |
|---|---|---|---|---|
| `nano` | ~5M | MBs | this Mac | Prove the pipeline end-to-end |
| `small` | ~150M | ~10B tokens | 1×8 GPU node | First useful code completions |
| `base` | ~7B | ~2T tokens | 100s of GPUs | Competitive open-model tier |
| `frontier` | 100B+ (MoE) | 10T+ tokens | 10,000s of GPUs, Megatron/FSDP | Qwen/GLM/Claude class |

At `base` scale and above you swap the single-device training loop for a distributed
stack (FSDP / Megatron-LM), add MoE layers, long-context extension, and RLHF/RL —
the model code here remains the reference implementation your team validates against.
