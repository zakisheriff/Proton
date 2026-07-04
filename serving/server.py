"""Proton 1 inference server — OpenAI-compatible HTTP API.

Exposes /v1/chat/completions (and /v1/models) so any OpenAI SDK can talk to
Proton 1 by just changing base_url. This is the model backend; the Node/Express
gateway (serving/gateway) sits in front of it for API keys, rate limits, billing.

Run:
    pip install fastapi uvicorn
    python3 -m serving.server --ckpt checkpoints/nano-sft.pt
    # or: uvicorn serving.server:app  (set PROTON_CKPT env var)
"""

import argparse
import json
import os
import time
import uuid

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from proton1.chat import build_prompt_ids
from proton1.config import ModelConfig
from proton1.model import Proton
from proton1.tokenizer import BPETokenizer
from proton1.utils import pick_device

MODEL_NAME = "proton-1"

app = FastAPI(title="Proton 1 API", version="0.1.0")
_STATE: dict = {}


def load(ckpt_path: str, tokenizer_path: str):
    device = pick_device()
    tok = BPETokenizer.load(tokenizer_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    mcfg = ModelConfig(**ckpt["config"])
    model = Proton(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    _STATE.update(model=model, tok=tok, device=device, mcfg=mcfg)
    print(f"Proton 1 server ready | {ckpt_path} | device={device}")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[Message]
    temperature: float = 0.7
    top_k: int = 50
    max_tokens: int = 256
    stream: bool = False


def _generate_ids(messages, temperature, top_k, max_tokens):
    tok, model, device = _STATE["tok"], _STATE["model"], _STATE["device"]
    msgs = [{"role": m.role, "content": m.content} for m in messages]
    prompt_ids = build_prompt_ids(tok, msgs)
    idx = torch.tensor([prompt_ids], device=device)
    out = model.generate(idx, max_tokens, temperature=temperature, top_k=top_k,
                         stop_token=tok.eot)
    gen = out[0, len(prompt_ids):].tolist()
    return tok.decode(gen).replace("<|endoftext|>", "").strip(), len(prompt_ids), len(gen)


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [
        {"id": MODEL_NAME, "object": "model", "owned_by": "the-atom"}]}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    text, n_prompt, n_gen = _generate_ids(
        req.messages, req.temperature, req.top_k, req.max_tokens)
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if req.stream:
        def event_stream():
            # single-chunk stream for simplicity; token-by-token is a TODO hook
            chunk = {
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": text},
                             "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            done = {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": MODEL_NAME,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return {
        "id": cid, "object": "chat.completion", "created": created, "model": MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_gen,
                  "total_tokens": n_prompt + n_gen},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("PROTON_CKPT", "checkpoints/nano-sft.pt"))
    ap.add_argument("--tokenizer", default=os.environ.get("PROTON_TOKENIZER", "data/tokenizer.json"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    load(args.ckpt, args.tokenizer)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
else:
    # when launched via `uvicorn serving.server:app`
    if os.environ.get("PROTON_CKPT"):
        load(os.environ["PROTON_CKPT"],
             os.environ.get("PROTON_TOKENIZER", "data/tokenizer.json"))
