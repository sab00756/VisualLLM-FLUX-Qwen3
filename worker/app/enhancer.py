"""Prompt-engineering layer.

Users type terse instructions ("restaurant background"). FLUX.1 Kontext produces
far better edits from a specific instruction that (a) names the change, (b) tells
the model what to preserve, and (c) adds scene/lighting/realism context.

This layer expands a raw instruction into a well-structured Kontext instruction:
  1. an LLM rewriter (OpenAI-compatible endpoint, e.g. local qwen3-coder), then
  2. a deterministic template fallback if the LLM is disabled or fails.

It never invents a different intent — it only enriches what the user asked for.
Resolution/aspect is handled elsewhere (the router), so we never mention size here.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

# ── config (env) ─────────────────────────────────────────────
ENABLED = os.environ.get("ENHANCER_ENABLED", "true").lower() == "true"
# Backend: "local" (small model in-process via transformers) | "openai"
# (OpenAI-compatible HTTP endpoint) | "off"/"template" (template only).
BACKEND = os.environ.get("ENHANCER_BACKEND", "local").lower()
LOCAL_MODEL = os.environ.get("ENHANCER_LOCAL_MODEL", "Qwen/Qwen2.5-7B-Instruct")
# OpenAI-compatible HTTP backend (used when BACKEND=openai)
BASE_URL = os.environ.get("ENHANCER_BASE_URL", "http://qwen-litellm:4000/v1").rstrip("/")
MODEL = os.environ.get("ENHANCER_MODEL", "qwen3-coder")
API_KEY = os.environ.get("ENHANCER_API_KEY", "")
TIMEOUT = float(os.environ.get("ENHANCER_TIMEOUT", "25"))
MAX_TOKENS = int(os.environ.get("ENHANCER_MAX_TOKENS", "220"))

_local = None  # (model, tokenizer) singleton

SYSTEM_PROMPT = (
    "You are a prompt engineer for FLUX.1 Kontext, an instruction-based image "
    "editor that edits an EXISTING image the user has supplied. Rewrite the "
    "user's short edit instruction into ONE clear, vivid instruction that "
    "produces the best possible edit.\n"
    "Rules:\n"
    "1. Keep the user's exact intent. Never change requested colors, objects, or "
    "scenes, and never invent a different edit. Only enrich with helpful, "
    "concrete detail.\n"
    "2. Always state what to PRESERVE: the main subject/product's shape, "
    "proportions, colors, materials, markings, text and position must stay "
    "identical unless the user asked to change them.\n"
    "3. For background/scene changes: describe the setting, the surface the "
    "subject sits on, lighting (direction, softness, time of day), depth of "
    "field, and add realistic contact shadows and reflections; keep the subject "
    "in the same position and scale.\n"
    "4. For recoloring: change only the named part to the named color; keep its "
    "material, texture, and highlights and every other element identical.\n"
    "5. Prefer photorealism unless another style is requested: natural lighting, "
    "realistic shadows and reflections, high detail.\n"
    "6. Do NOT mention resolution, size, aspect ratio, or pixels.\n"
    "7. Output ONLY the final instruction as plain prose, 1-3 sentences. No "
    "preamble, no quotes, no markdown, no lists.\n\n"
    "Example:\n"
    "User: show this on a restaurant background\n"
    "You: Place the product on a wooden restaurant table with a warmly lit dining "
    "room blurred behind it; keep the product's shape, colors, markings and "
    "position exactly as they are, and add soft natural window light with realistic "
    "contact shadows and a subtle reflection on the table. Photorealistic, high detail."
)


# ── intent detection (for the template fallback) ─────────────
def _intent(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(change|make|turn|recolor|colou?r)\b.*\b(to|into)\b", t) or "recolor" in t:
        return "recolor"
    if re.search(r"\b(background|backdrop|scene|setting|place (this|it)|put (this|it)|on a |in a |in an |at a )\b", t):
        return "background"
    if re.search(r"\b(light|lighting|relight|sunset|sunrise|golden hour|studio|shadow|neon|daylight)\b", t):
        return "relight"
    return "generic"


_PRESERVE = (
    "Keep the main subject's shape, proportions, colors, materials, markings, "
    "text and position identical"
)
_QUALITY = (
    "Photorealistic, natural lighting, realistic shadows and reflections, high detail."
)


def _template(raw: str) -> str:
    """Deterministic enrichment — reliable, no LLM needed."""
    raw = raw.strip().rstrip(".")
    intent = _intent(raw)
    if intent == "background":
        return (
            f"{raw}. {_PRESERVE}; only replace the surrounding scene. Add soft, "
            f"directional lighting, realistic contact shadows and a subtle reflection, "
            f"with a gently blurred background. {_QUALITY}"
        )
    if intent == "recolor":
        return (
            f"{raw}. Change only that color as described and keep its material, "
            f"texture, shading and highlights, and every other element of the image, "
            f"identical. {_QUALITY}"
        )
    if intent == "relight":
        return (
            f"{raw}. {_PRESERVE}, changing only the lighting and mood. {_QUALITY}"
        )
    return f"{raw}. {_PRESERVE} unless the instruction says otherwise. {_QUALITY}"


def _clean_llm_output(text: str) -> str:
    text = text.strip()
    # strip accidental wrapping quotes / code fences / labels
    text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text).strip()
    text = re.sub(r'^(instruction|prompt|output|you)\s*[:\-]\s*', "", text, flags=re.IGNORECASE).strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] in "\"'":
        text = text[1:-1].strip()
    return text


def _load_local():
    """Load the small instruct model once (lazy). Shares the worker's GPU."""
    global _local
    if _local is not None:
        return _local
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[enhancer] loading local model {LOCAL_MODEL} (first use downloads it)…", flush=True)
    tok = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    _local = (model, tok)
    print("[enhancer] local model ready.", flush=True)
    return _local


def _local_generate(raw: str) -> str | None:
    """Rewrite the prompt with the in-process instruct model."""
    try:
        import torch

        model, tok = _load_local()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw.strip()},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_TOKENS, do_sample=True,
                temperature=0.5, top_p=0.9, pad_token_id=tok.eos_token_id,
            )
        gen = out[0][inputs.input_ids.shape[1]:]
        result = _clean_llm_output(tok.decode(gen, skip_special_tokens=True))
        return result if 8 <= len(result) <= 1200 else None
    except Exception as exc:
        print(f"[enhancer] local model failed ({type(exc).__name__}: {exc}); using template", flush=True)
        return None


def _llm(raw: str) -> str | None:
    """Call the OpenAI-compatible endpoint; return None on any failure."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw.strip()},
        ],
        "temperature": 0.5,
        "max_tokens": MAX_TOKENS,
    }
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.load(r)
        out = _clean_llm_output(data["choices"][0]["message"]["content"])
        # sanity: reject empty or absurdly short/long results
        if 8 <= len(out) <= 1200:
            return out
        return None
    except Exception as exc:
        print(f"[enhancer] LLM unavailable ({type(exc).__name__}: {exc}); using template", flush=True)
        return None


def engineer_prompt(raw: str) -> tuple[str, str]:
    """Return (engineered_prompt, source) where source is llm|template|raw|disabled."""
    raw = (raw or "").strip()
    if not raw:
        return raw, "raw"
    if not ENABLED or BACKEND in ("off", "template", "none"):
        return (_template(raw), "template") if ENABLED else (raw, "disabled")
    out = _local_generate(raw) if BACKEND == "local" else _llm(raw)
    if out:
        return out, "llm"
    return _template(raw), "template"
