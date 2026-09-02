"""FLUX.1 Kontext model wrapper + deterministic Pillow resize ops."""
from __future__ import annotations

import os
import threading

from PIL import Image, ImageColor

_DTYPE_MAP = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}

_pipe = None
_pipe_lock = threading.Lock()


def _resolve_color(name: str):
    try:
        return ImageColor.getrgb(name)
    except ValueError:
        return (255, 255, 255)


def get_pipeline():
    """Load FLUX.1 Kontext once (thread-safe singleton) and keep it warm."""
    global _pipe
    if _pipe is not None:
        return _pipe
    with _pipe_lock:
        if _pipe is not None:
            return _pipe
        import torch
        from diffusers import FluxKontextPipeline

        model_id = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.1-Kontext-dev")
        dtype_name = os.environ.get("TORCH_DTYPE", "bfloat16")
        torch_dtype = getattr(torch, _DTYPE_MAP.get(dtype_name, "bfloat16"))

        pipe = FluxKontextPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            token=os.environ.get("HF_TOKEN") or None,
        )
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        _pipe = pipe
        return _pipe


def run_edit_model(
    image: Image.Image,
    instruction: str,
    *,
    steps: int,
    guidance: float,
    seed: int | None,
) -> Image.Image:
    """Apply an instruction edit with FLUX.1 Kontext."""
    import torch

    pipe = get_pipeline()
    generator = None
    if seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(int(seed))

    out = pipe(
        image=image,
        prompt=instruction,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator,
    )
    return out.images[0]


def apply_resize(
    image: Image.Image,
    width: int,
    height: int,
    mode: str = "fit",
    background: str = "white",
) -> Image.Image:
    """Resize to exactly ``width`` x ``height``.

    fit     -> scale to fit inside, pad with ``background`` (letterbox)
    cover   -> scale to cover, center-crop
    stretch -> ignore aspect ratio
    """
    img = image.convert("RGB")
    if mode == "stretch":
        return img.resize((width, height), Image.LANCZOS)

    src_w, src_h = img.size
    if mode == "cover":
        scale = max(width / src_w, height / src_h)
        new = img.resize((max(1, round(src_w * scale)), max(1, round(src_h * scale))), Image.LANCZOS)
        left = (new.width - width) // 2
        top = (new.height - height) // 2
        return new.crop((left, top, left + width, top + height))

    # default: fit (letterbox / pad)
    scale = min(width / src_w, height / src_h)
    new = img.resize((max(1, round(src_w * scale)), max(1, round(src_h * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", (width, height), _resolve_color(background))
    canvas.paste(new, ((width - new.width) // 2, (height - new.height) // 2))
    return canvas
