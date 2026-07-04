"""Proton 1 chat template + SFT example encoding.

Wire format (special tokens from the tokenizer):
    <|system|> ... <|user|> ... <|assistant|> ... <|endoftext|>

During SFT we compute loss ONLY on assistant tokens (+ the closing <|endoftext|>),
so the model learns to produce replies, not to parrot the prompt.
"""

from .tokenizer import BPETokenizer

DEFAULT_SYSTEM = (
    "You are Proton 1, a coding assistant by The Atom, specialized in "
    "JavaScript, TypeScript, React, React Native, Next.js, Node.js, Express, "
    "and Python."
)


def build_prompt_ids(tok: BPETokenizer, messages: list[dict], system: str | None = None):
    """messages: [{'role': 'user'|'assistant', 'content': str}, ...].

    Returns token ids ending right after the final <|assistant|> tag, ready for
    generation.
    """
    ids = [tok.encode_special("<|system|>")]
    ids += tok.encode(" " + (system or DEFAULT_SYSTEM) + " ")
    for m in messages:
        tag = "<|user|>" if m["role"] == "user" else "<|assistant|>"
        ids.append(tok.encode_special(tag))
        ids += tok.encode(" " + m["content"] + " ")
    ids.append(tok.encode_special("<|assistant|>"))
    return ids


def encode_sft_example(tok: BPETokenizer, example: dict, max_len: int):
    """example: {'system'?, 'messages': [...]}. Returns (input_ids, loss_mask).

    loss_mask[i] == 1 where token i is part of an assistant response.
    """
    ids: list[int] = [tok.encode_special("<|system|>")]
    ids += tok.encode(" " + example.get("system", DEFAULT_SYSTEM) + " ")
    mask: list[int] = [0] * len(ids)

    for m in example["messages"]:
        if m["role"] == "user":
            seg = [tok.encode_special("<|user|>")] + tok.encode(" " + m["content"] + " ")
            ids += seg
            mask += [0] * len(seg)
        else:  # assistant — supervise these tokens
            seg = [tok.encode_special("<|assistant|>")] + tok.encode(" " + m["content"] + " ")
            seg.append(tok.eot)
            ids += seg
            # don't supervise the <|assistant|> tag itself, but do supervise content+eot
            mask += [0] + [1] * (len(seg) - 1)

    return ids[:max_len], mask[:max_len]
