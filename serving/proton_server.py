"""Proton 1 — OpenAI-compatible MLX server for the atom-coder CLI.

Serves the Proton 1 model (Qwen2.5-Coder-7B + optional LoRA adapter) over an
OpenAI-compatible API, and — crucially — reports its model id as "Proton 1" so it
shows up by that name in atom-coder's "Select Local Model" menu.

The atom-coder CLI discovers models by GETting {LOCAL_ENDPOINT}/v1/models and
listing each returned `id`. This server returns id="Proton 1".

Run:
    source .venv-mlx/bin/activate
    python -m serving.proton_server            # base + adapter if present
    python -m serving.proton_server --no-adapter   # base only
Then point the CLI at it:
    LOCAL_ENDPOINT=http://localhost:8080 atom-coder

Endpoints: GET /v1/models, POST /v1/chat/completions (stream + non-stream).
"""

import argparse
import json
import os
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

MODEL_ID = "Proton 1"

PROTON_SYSTEM = (
    "You are Proton 1, an AI coding agent built by The Atom. You specialize in "
    "building beautiful, modern, award-quality websites with Next.js, React, "
    "TypeScript, Tailwind CSS, and Framer Motion. You write clean, accessible, "
    "animated, production-grade code with strong visual taste. Always identify "
    "yourself as Proton 1 by The Atom, never as Qwen or any other model."
)

app = FastAPI(title="Proton 1")
_S: dict = {}


def load_model(model_path: str, adapter: str | None):
    from mlx_lm import load
    model, tok = load(model_path, adapter_path=adapter)
    _S.update(model=model, tok=tok)
    print(f"Proton 1 server ready | model={model_path} | adapter={adapter or 'none'}")


class Msg(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    model: str = MODEL_ID
    messages: list[Msg]
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False


def _messages_with_identity(messages: list[Msg]) -> list[dict]:
    msgs = [{"role": m.role, "content": m.content} for m in messages]
    # Ensure Proton 1 identity leads; keep any user-provided system after it.
    if not msgs or msgs[0]["role"] != "system":
        msgs.insert(0, {"role": "system", "content": PROTON_SYSTEM})
    else:
        msgs[0]["content"] = PROTON_SYSTEM + "\n\n" + msgs[0]["content"]
    return msgs


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [
        {"id": MODEL_ID, "object": "model", "type": "llm",
         "publisher": "the-atom", "state": "loaded",
         "max_context_length": 32768}]}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    from mlx_lm import generate, stream_generate
    tok, model = _S["tok"], _S["model"]
    prompt = tok.apply_chat_template(
        _messages_with_identity(req.messages), add_generation_prompt=True)
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if req.stream:
        def gen():
            for chunk in stream_generate(model, tok, prompt=prompt,
                                         max_tokens=req.max_tokens):
                payload = {"id": cid, "object": "chat.completion.chunk",
                           "created": created, "model": MODEL_ID,
                           "choices": [{"index": 0,
                                        "delta": {"content": chunk.text},
                                        "finish_reason": None}]}
                yield f"data: {json.dumps(payload)}\n\n"
            done = {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    text = generate(model, tok, prompt=prompt, max_tokens=req.max_tokens, verbose=False)
    return {"id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": 0,
                      "total_tokens": len(prompt)}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get(
        "PROTON_MODEL", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"))
    ap.add_argument("--adapter", default=os.environ.get("PROTON_ADAPTER", "adapters/proton1"))
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    adapter = None
    if not args.no_adapter and os.path.isdir(args.adapter):
        adapter = args.adapter
    load_model(args.model, adapter)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
