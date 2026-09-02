"""Regenerative editor backends (Qwen-Image-Edit / FLUX.1 Kontext) + Pillow resize."""
from __future__ import annotations

import os
import threading

from PIL import Image, ImageColor

_DTYPE_MAP = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}

# Which diffusion editor to load. Only ONE is loaded per worker (single GPU /
# unified memory), so switching backend means the other model is never resident.
#   "qwen" -> Qwen-Image-Edit-2509 (Apache-2.0, stronger identity, commercial-OK) [default]
#   "flux" -> FLUX.1 Kontext (regenerative; great scenes, weaker fine identity; gated)
EDITOR_BACKEND = os.environ.get("EDITOR_BACKEND", "qwen").lower()
QWEN_MODEL_ID = os.environ.get("QWEN_EDIT_MODEL_ID", "Qwen/Qwen-Image-Edit-2509")
# Qwen-Image-Edit uses "true CFG" (a real negative-prompt guidance) instead of
# FLUX's distilled guidance scale.
QWEN_TRUE_CFG = float(os.environ.get("QWEN_TRUE_CFG", "4.0"))
QWEN_NEGATIVE = os.environ.get("QWEN_NEGATIVE_PROMPT", " ")

# FLUX.1 Kontext's trained resolution buckets (width, height). Left to its
# defaults, the pipeline emits a 1024x1024 square for *any* input, reframing
# non-square subjects. We instead pick the bucket closest to the input's aspect
# ratio and pass it as the output size, so the generated scene keeps the
# subject's framing. These are the model's own PREFERRED_KONTEXT_RESOLUTIONS.
_KONTEXT_BUCKETS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328),
    (832, 1248), (880, 1184), (944, 1104), (1024, 1024), (1104, 944),
    (1184, 880), (1248, 832), (1328, 800), (1392, 752), (1456, 720),
    (1504, 688), (1568, 672),
]

_pipe = None
_pipe_lock = threading.Lock()


def _resolve_color(name: str):
    try:
        return ImageColor.getrgb(name)
    except ValueError:
        return (255, 255, 255)


def _output_size(image: Image.Image) -> tuple[int, int]:
    """Nearest trained (width, height) bucket to the input's aspect ratio."""
    w, h = image.size
    aspect = w / h if h else 1.0
    _, bw, bh = min((abs(aspect - bw / bh), bw, bh) for bw, bh in _KONTEXT_BUCKETS)
    return bw, bh


def get_pipeline():
    """Load the selected editor once (thread-safe singleton) and keep it warm."""
    global _pipe
    if _pipe is not None:
        return _pipe
    with _pipe_lock:
        if _pipe is not None:
            return _pipe
        import torch

        dtype_name = os.environ.get("TORCH_DTYPE", "bfloat16")
        torch_dtype = getattr(torch, _DTYPE_MAP.get(dtype_name, "bfloat16"))

        if EDITOR_BACKEND == "qwen":
            from diffusers import QwenImageEditPlusPipeline

            pipe = QwenImageEditPlusPipeline.from_pretrained(
                QWEN_MODEL_ID,
                torch_dtype=torch_dtype,
                token=os.environ.get("HF_TOKEN") or None,
            )
        else:
            from diffusers import FluxKontextPipeline

            model_id = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.1-Kontext-dev")
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
    """Apply an instruction edit with the configured editor (Qwen or FLUX)."""
    import torch

    pipe = get_pipeline()
    generator = None
    if seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(int(seed))

    if EDITOR_BACKEND == "qwen":
        # Qwen-Image-Edit-2509 takes a LIST of reference images and preserves the
        # input aspect internally; it uses true CFG with a negative prompt rather
        # than FLUX's distilled guidance scale.
        out = pipe(
            image=[image],
            prompt=instruction,
            negative_prompt=QWEN_NEGATIVE,
            true_cfg_scale=QWEN_TRUE_CFG,
            num_inference_steps=int(steps),
            generator=generator,
        )
        return out.images[0]

    # FLUX.1 Kontext: snap the output canvas to the input's aspect ratio.
    width, height = _output_size(image)
    out = pipe(
        image=image,
        prompt=instruction,
        width=width,
        height=height,
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
