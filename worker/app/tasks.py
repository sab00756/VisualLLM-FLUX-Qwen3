"""RQ task: execute one edit job end-to-end."""
from __future__ import annotations

import json
import os
import time
from functools import lru_cache

import redis
from PIL import Image

from .enhancer import engineer_prompt
from .pipeline import apply_resize, run_edit_model


@lru_cache
def _redis() -> redis.Redis:
    url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    return redis.Redis.from_url(url, decode_responses=True)


def _key(job_id: str) -> str:
    return f"job:{job_id}"


def _default_steps() -> int:
    return int(os.environ.get("DEFAULT_STEPS", "28"))


def _default_guidance() -> float:
    return float(os.environ.get("DEFAULT_GUIDANCE", "2.5"))


def _save(image: Image.Image, path: str, fmt: str) -> None:
    fmt = fmt.lower()
    if fmt in ("jpg", "jpeg"):
        image.convert("RGB").save(path, format="JPEG", quality=95)
    elif fmt == "webp":
        image.save(path, format="WEBP", quality=95)
    else:
        image.save(path, format="PNG")


def run_edit(job_id: str) -> None:
    r = _redis()
    key = _key(job_id)
    data = r.hgetall(key)
    if not data:
        return

    try:
        r.hset(key, "status", "running")
        plan = json.loads(data["plan"])
        params = json.loads(data.get("params") or "{}")
        src_path = data["src_path"]
        result_path = data["result_path"]
        out_fmt = data.get("result_format", "png")

        image = Image.open(src_path)
        image.load()

        # 0) Prompt-engineering layer: expand the terse instruction into a rich
        #    Kontext instruction (LLM + template fallback). Recorded for the API.
        instruction = (plan or {}).get("instruction")
        if instruction and instruction.strip():
            if params.get("enhance", True):
                engineered, source = engineer_prompt(instruction)
            else:
                engineered, source = instruction, "disabled"
            r.hset(key, mapping={"engineered_prompt": engineered, "engineered_by": source})

            # 1) Semantic edit via the diffusion model.
            image = run_edit_model(
                image.convert("RGB"),
                engineered,
                steps=params.get("steps") or _default_steps(),
                guidance=params.get("guidance") or _default_guidance(),
                seed=params.get("seed"),
            )

        # 2) Deterministic resize (explicit dims or ecom preset).
        resize = (plan or {}).get("resize")
        if resize:
            image = apply_resize(
                image,
                width=resize["width"],
                height=resize["height"],
                mode=resize.get("mode", "fit"),
                background=resize.get("background", "white"),
            )

        _save(image, result_path, out_fmt)
        r.hset(key, mapping={"status": "done", "finished_at": time.time(), "error": ""})

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception as exc:  # capture per-job failure for the API to surface
        r.hset(key, mapping={
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "finished_at": time.time(),
        })
