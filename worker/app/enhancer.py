"""Prompt-engineering layer + auto-router (both vision-model backed).

Users type terse instructions ("restaurant background"). Instruction-based image
editors produce far better edits from a specific instruction that (a) names the
change, (b) tells the model what to preserve, and (c) adds scene/lighting/realism
context. This module has two responsibilities, both using the same vision model:

  * ``engineer_prompt`` — expand a raw instruction into a rich edit instruction.
    Backends (``ENHANCER_BACKEND``): ``vlm`` (vision model that SEES the image,
    default), ``local`` (text-only instruct model), ``openai`` (HTTP endpoint),
    or a deterministic ``template`` fallback used whenever a model is unavailable.
  * ``route_decision`` — in ``mode=auto``, pick the editor vs. composite path
    from the image and the instruction (intent-aware).

Enrichment never invents a different intent — it only enriches what the user
asked for. Resolution/aspect is handled by the router, so we never mention size.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

# ── config (env) ─────────────────────────────────────────────
ENABLED = os.environ.get("ENHANCER_ENABLED", "true").lower() == "true"
# Backend: "vlm" (vision LLM in-process — SEES the image) | "local" (text-only
# instruct model in-process) | "openai" (OpenAI-compatible HTTP endpoint) |
# "off"/"template" (deterministic template only).
BACKEND = os.environ.get("ENHANCER_BACKEND", "vlm").lower()
LOCAL_MODEL = os.environ.get("ENHANCER_LOCAL_MODEL", "Qwen/Qwen2.5-7B-Instruct")
# Vision model (used when BACKEND=vlm) — looks at the actual input image.
VLM_MODEL = os.environ.get("ENHANCER_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
# Longest edge the image is downscaled to before the VLM sees it (keeps the
# understanding step fast; the edit model still runs on the full-res image).
VLM_MAX_EDGE = int(os.environ.get("ENHANCER_VLM_MAX_EDGE", "768"))
# OpenAI-compatible HTTP backend (used when BACKEND=openai)
BASE_URL = os.environ.get("ENHANCER_BASE_URL", "http://qwen-litellm:4000/v1").rstrip("/")
MODEL = os.environ.get("ENHANCER_MODEL", "qwen3-coder")
API_KEY = os.environ.get("ENHANCER_API_KEY", "")
TIMEOUT = float(os.environ.get("ENHANCER_TIMEOUT", "25"))
MAX_TOKENS = int(os.environ.get("ENHANCER_MAX_TOKENS", "220"))

_local = None  # (model, tokenizer) singleton for the text-only backend
_vlm = None    # (model, processor) singleton for the vision backend

SYSTEM_PROMPT = (
    "You are a prompt engineer for an instruction-based image "
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


VISION_SYSTEM_PROMPT = (
    "You are a prompt engineer for an instruction-based image "
    "editor. You are shown the ACTUAL image the user wants edited, plus their "
    "short edit instruction. Look carefully at the image, then rewrite the "
    "instruction into ONE clear, vivid instruction that produces the best edit.\n"
    "Rules:\n"
    "1. First, silently identify the MAIN SUBJECT and its concrete visual "
    "attributes you can actually see: object type, exact colors, shape, "
    "materials, text/logos/markings, and where it sits in the frame.\n"
    "2. Keep the user's exact intent. Never change requested colors, objects, or "
    "scenes, and never invent a different edit — only enrich with detail grounded "
    "in what you see.\n"
    "3. Always state what to PRESERVE, naming the subject's SPECIFIC observed "
    "attributes (e.g. 'the matte black rectangular device with a red button and a "
    "green indicator light'), so its shape, proportions, colors, materials, "
    "markings, text and position stay identical unless the user asked to change "
    "them.\n"
    "4. For background/scene changes: describe the new setting, the surface the "
    "subject rests on, lighting (direction, softness, time of day), depth of "
    "field, and add realistic contact shadows and reflections; keep the subject "
    "in the same position and scale.\n"
    "5. For recoloring: change only the named part to the named color; keep its "
    "material, texture and highlights and every other element identical.\n"
    "6. Prefer photorealism unless another style is requested.\n"
    "7. Do NOT mention resolution, size, aspect ratio, or pixels.\n"
    "8. Your instruction MUST contain an explicit clause that names the subject's "
    "specific observed attributes to keep — do not write a vague 'keep the object "
    "unchanged'. If the subject is small or thin, say so and stress preserving its "
    "exact size, shape and position.\n"
    "9. Output ONLY the final instruction as plain prose, 1-3 sentences. No "
    "preamble, no analysis, no quotes, no markdown, no lists.\n\n"
    "Example:\n"
    "User instruction: put it on a cafe table\n"
    "You (having seen a matte-black rectangular device with a large red button and "
    "a small green indicator light): Place the matte-black rectangular device with "
    "its large red button and small green indicator light on a wooden cafe table, "
    "with a warm softly-blurred cafe interior behind it; keep the device's shape, "
    "proportions, colors, markings and position exactly as they are, and add soft "
    "directional lighting, a realistic contact shadow and a subtle reflection. "
    "Photorealistic, high detail."
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


def _load_vlm():
    """Load the vision-language model once (lazy). Shares the worker's GPU."""
    global _vlm
    if _vlm is not None:
        return _vlm
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"[enhancer] loading vision model {VLM_MODEL} (first use downloads it)…", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    processor = AutoProcessor.from_pretrained(VLM_MODEL)
    _vlm = (model, processor)
    print("[enhancer] vision model ready.", flush=True)
    return _vlm


def _downscale(image, max_edge: int):
    """Shrink so the longest edge is <= max_edge (understanding needs no full res)."""
    w, h = image.size
    longest = max(w, h)
    if longest <= max_edge:
        return image.convert("RGB")
    scale = max_edge / longest
    return image.convert("RGB").resize((max(1, round(w * scale)), max(1, round(h * scale))))


def _vlm_chat(system: str | None, user_text: str, image, max_new_tokens: int) -> str:
    """One greedy turn on the vision model with an image + text; decoded reply."""
    import torch

    model, processor = _load_vlm()
    small = _downscale(image, VLM_MAX_EDGE)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}
    )
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[small], padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def _vlm_generate(raw: str, image) -> str | None:
    """Rewrite the prompt with the in-process vision model, grounded in the image."""
    if image is None:
        return None
    try:
        result = _clean_llm_output(_vlm_chat(VISION_SYSTEM_PROMPT, raw.strip(), image, MAX_TOKENS))
        return result if 8 <= len(result) <= 1200 else None
    except Exception as exc:
        print(f"[enhancer] vision model failed ({type(exc).__name__}: {exc}); falling back", flush=True)
        return None


_ROUTE_PROMPT = (
    "You route an image-editing request to one of two engines. Look at the image "
    "AND read the instruction, then choose:\n"
    "- 'edit': a generative editor that regenerates the scene around the subject. "
    "Choose it when the user wants the subject INTEGRATED into a scene — installed, "
    "mounted, placed naturally, relit, perspective-matched, blended, or made "
    "realistic — OR when the subject is a flat icon/logo/graphic that should be "
    "turned into a realistic object.\n"
    "- 'composite': cuts the subject out and keeps its EXACT pixels unchanged on a "
    "generated backdrop. Choose it ONLY when the user wants the subject preserved "
    "pixel-for-pixel and simply shown on a plain/studio/product backdrop — e.g. "
    "'product photo on a white background', 'catalog shot', 'keep it exactly, just "
    "put it on a clean background'.\n"
    "When in doubt prefer 'edit'. Answer with exactly one word: edit or composite."
)


def route_decision(instruction: str, image) -> str:
    """Pick the engine for an auto-mode job: 'edit' or 'composite'.

    Intent-aware: reads both the image and the instruction. Integration/scene
    intent (and any flat graphic) -> 'edit' (the regenerative editor); an explicit
    "keep exact pixels on a plain backdrop" intent -> 'composite'. Uses the loaded
    vision model. Defaults to 'edit' (the safe, natural-looking path) if unsure.
    """
    if image is None:
        return "edit"
    try:
        user_text = f"{_ROUTE_PROMPT}\n\nInstruction: {(instruction or '').strip()}"
        ans = _vlm_chat(None, user_text, image, max_new_tokens=6).strip().lower()
        return "composite" if "composite" in ans else "edit"
    except Exception as exc:
        print(f"[enhancer] route decision failed ({type(exc).__name__}: {exc}); routing to editor", flush=True)
        return "edit"


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


def engineer_prompt(raw: str, image=None) -> tuple[str, str]:
    """Expand a terse instruction into a rich edit instruction.

    Returns ``(engineered_prompt, source)`` where source is one of
    ``vlm`` | ``llm`` | ``template`` | ``raw`` | ``disabled``. When
    ``BACKEND=vlm`` and an ``image`` is supplied, the vision model sees the
    image and writes a grounded instruction; otherwise it falls back to the
    text model, then to the deterministic template.
    """
    raw = (raw or "").strip()
    if not raw:
        return raw, "raw"
    if not ENABLED or BACKEND in ("off", "template", "none"):
        return (_template(raw), "template") if ENABLED else (raw, "disabled")

    if BACKEND == "vlm":
        out = _vlm_generate(raw, image)
        if out:
            return out, "vlm"
        # image missing or VLM failed — try the text model, then template.
        out = _local_generate(raw)
        if out:
            return out, "llm"
        return _template(raw), "template"

    out = _local_generate(raw) if BACKEND == "local" else _llm(raw)
    if out:
        return out, "llm"
    return _template(raw), "template"
