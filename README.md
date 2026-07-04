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

The full stack every frontier lab runs — training, alignment, and serving:

| Stage | Command | What it does |
|---|---|---|
| 1. Data (local) | `make data` | Builds a corpus from local code files |
| 1. Data (real) | `make download` | Streams a real code corpus from HuggingFace (The Stack / codeparrot) |
| 2. Tokenizer | `make tokenizer` | Trains byte-level BPE |
| 3. Tokenize | `make tokens` | Packs the corpus into a token tensor |
| 4. Pretrain | `make pretrain CONFIG=nano` | Next-token pretraining (single device) |
| 4. Pretrain (multi-GPU) | `torchrun --nproc_per_node=N -m proton1.train_fsdp --config small` | FSDP distributed pretraining |
| 5. SFT | `make sft` | Instruction-tunes the base model into an assistant |
| 6. RL | `make rl` | RL with execution rewards (GRPO) — runs code, rewards passing tests |
| 7. Eval | `make eval` | pass@k over a code problem set, sandboxed |
| 8. Chat | `make chat` | Talk to Proton 1 locally |

## Serving Proton 1 as an API

Others consume Proton 1 exactly like the OpenAI/Qwen/GLM APIs:

```
client (OpenAI SDK) ──▶ Node/Express gateway :8080 ──▶ Python inference server :8000
                         (API keys, rate limits,        (the model)
                          usage metering)
```

```bash
make serve      # terminal 1: Python inference server (OpenAI-compatible)
make gateway    # terminal 2: Node/Express gateway (auth + limits + metering)
python3 examples/client.py   # call it with the OpenAI SDK
```

Endpoints: `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`,
`GET /v1/usage` (per-key). Auth is `Authorization: Bearer <key>`; keys are set via
the `PROTON_KEYS` env var on the gateway.

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
