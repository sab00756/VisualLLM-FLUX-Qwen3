"""Lean composite path — guarantees the subject's *exact* pixels.

Instead of regenerating the whole image (FLUX / Qwen), this:
  1. segments the subject out of the input  (BiRefNet, MIT license),
  2. generates a background scene from the prompt  (SDXL),
  3. places the real subject cutout onto the scene (position/scale we control),
  4. harmonizes it (tone-match toward the scene) and drops a soft contact shadow.

The subject in the output is the user's literal pixels — nothing drifts. The
tradeoff vs. a generative editor is lighting realism, which the tone-match +
shadow approximate cheaply (no extra relight model).
"""
from __future__ import annotations

import os
import re
import threading

from PIL import Image, ImageFilter

# ── config (env) ─────────────────────────────────────────────
BIREFNET_MODEL = os.environ.get("SEG_MODEL", "ZhengPeng7/BiRefNet")
SDXL_MODEL = os.environ.get("BG_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
BG_STEPS = int(os.environ.get("BG_STEPS", "25"))
BG_GUIDANCE = float(os.environ.get("BG_GUIDANCE", "6.0"))
# Subject footprint on the canvas.
SUBJECT_SCALE = float(os.environ.get("SUBJECT_SCALE", "0.6"))   # width fraction
SUBJECT_VPOS = float(os.environ.get("SUBJECT_VPOS", "0.92"))    # subject bottom, as H fraction
HARMONIZE_STRENGTH = float(os.environ.get("HARMONIZE_STRENGTH", "0.22"))

_BG_NEGATIVE = (
    "product, device, gadget, object, item, person, people, hands, text, "
    "watermark, logo, clutter, mockup"
)

_seg = None
_sdxl = None
_seg_lock = threading.Lock()
_sdxl_lock = threading.Lock()


# ── model singletons ─────────────────────────────────────────
def _load_seg():
    global _seg
    if _seg is not None:
        return _seg
    with _seg_lock:
        if _seg is not None:
            return _seg
        import torch
        from transformers import AutoModelForImageSegmentation

        print(f"[composite] loading segmenter {BIREFNET_MODEL}…", flush=True)
        m = AutoModelForImageSegmentation.from_pretrained(BIREFNET_MODEL, trust_remote_code=True)
        if torch.cuda.is_available():
            m = m.to("cuda").half()
        m.eval()
        _seg = m
        print("[composite] segmenter ready.", flush=True)
        return _seg


def _load_sdxl():
    global _sdxl
    if _sdxl is not None:
        return _sdxl
    with _sdxl_lock:
        if _sdxl is not None:
            return _sdxl
        import torch
        from diffusers import StableDiffusionXLPipeline

        print(f"[composite] loading background model {SDXL_MODEL}…", flush=True)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_MODEL, torch_dtype=torch.bfloat16, use_safetensors=True, variant="fp16",
        )
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        _sdxl = pipe
        print("[composite] background model ready.", flush=True)
        return _sdxl


# ── steps ────────────────────────────────────────────────────
def segment(image: Image.Image) -> Image.Image:
    """Return an L-mode alpha mask (subject=white) at the input's resolution."""
    import torch
    from torchvision import transforms

    model = _load_seg()
    img = image.convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(img).unsqueeze(0)
    if torch.cuda.is_available():
        x = x.to("cuda").half()
    with torch.no_grad():
        pred = model(x)[-1].sigmoid().cpu()[0].squeeze().float()
    return transforms.ToPILImage()(pred).resize(img.size)


def scene_prompt(instruction: str) -> str:
    """Turn 'put this device on a marble counter' into a subject-free backdrop prompt."""
    text = (instruction or "").strip()
    # Drop a leading 'show/place/put this <subject>' up to the first preposition.
    text = re.sub(
        r"^\s*(show|place|put|display|set|render)\b.*?\b(on|in|at|against|onto|over|with|near|by)\b",
        r"\2",
        text,
        flags=re.IGNORECASE,
    )
    text = text.strip(" ,.") or "a clean neutral studio backdrop"
    return (
        f"{text}, empty scene with no product or object, professional product "
        f"photography backdrop, soft even lighting, realistic, high detail"
    )


def generate_background(instruction: str, width: int, height: int, seed: int | None) -> Image.Image:
    import torch

    pipe = _load_sdxl()
    generator = None
    if seed is not None:
        generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))
    out = pipe(
        prompt=scene_prompt(instruction),
        negative_prompt=_BG_NEGATIVE,
        num_inference_steps=BG_STEPS,
        guidance_scale=BG_GUIDANCE,
        width=width,
        height=height,
        generator=generator,
    )
    return out.images[0]


def _sdxl_size(image: Image.Image) -> tuple[int, int]:
    """A ~1 MP canvas matching the input aspect, each side a multiple of 8."""
    w, h = image.size
    aspect = w / h if h else 1.0
    area = 1024 * 1024
    bw = round((area * aspect) ** 0.5)
    bh = round((area / aspect) ** 0.5)
    bw = max(512, min(1536, bw // 8 * 8))
    bh = max(512, min(1536, bh // 8 * 8))
    return bw, bh


def _harmonize(subject_rgb: Image.Image, mask: Image.Image, bg: Image.Image, strength: float) -> Image.Image:
    """Nudge the subject's mean color toward the scene's so it doesn't look pasted on."""
    import numpy as np

    s = np.asarray(subject_rgb.convert("RGB")).astype("float32")
    m = (np.asarray(mask.convert("L")).astype("float32") / 255.0)[..., None]
    bg_mean = np.asarray(bg.convert("RGB")).astype("float32").reshape(-1, 3).mean(0)
    denom = m.sum() + 1e-6
    s_mean = (s * m).sum(axis=(0, 1)) / denom
    target = (1 - strength) * s_mean + strength * bg_mean
    shifted = np.clip(s + (target - s_mean), 0, 255).astype("uint8")
    return Image.fromarray(shifted)


def compose(image: Image.Image, mask: Image.Image, bg: Image.Image) -> Image.Image:
    """Place the segmented subject onto the background with a soft contact shadow."""
    # Tight crop to the subject's bounding box.
    bbox = mask.getbbox()
    if bbox is None:  # nothing segmented — return the plain background
        return bg.convert("RGB")
    subj = image.convert("RGB").crop(bbox)
    subj_mask = mask.convert("L").crop(bbox)

    canvas_w, canvas_h = bg.size
    # Scale by width fraction, cap by height.
    target_w = max(1, int(canvas_w * SUBJECT_SCALE))
    scale = target_w / subj.width
    if subj.height * scale > canvas_h * 0.85:
        scale = (canvas_h * 0.85) / subj.height
    new_w = max(1, int(subj.width * scale))
    new_h = max(1, int(subj.height * scale))
    subj = subj.resize((new_w, new_h), Image.LANCZOS)
    subj_mask = subj_mask.resize((new_w, new_h), Image.LANCZOS)

    subj = _harmonize(subj, subj_mask, bg, HARMONIZE_STRENGTH)

    # Position: horizontally centered, bottom at SUBJECT_VPOS of the canvas.
    x = (canvas_w - new_w) // 2
    y = int(canvas_h * SUBJECT_VPOS) - new_h
    y = max(0, min(y, canvas_h - new_h))

    out = bg.convert("RGB").copy()

    # Soft contact shadow: squashed, blurred silhouette just under the subject.
    shadow_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    sh_h = max(1, int(new_h * 0.18))
    squashed = subj_mask.resize((new_w, sh_h), Image.LANCZOS)
    shadow_mask = Image.new("L", (canvas_w, canvas_h), 0)
    shadow_mask.paste(squashed, (x, min(canvas_h - sh_h, y + new_h - sh_h // 2)))
    blur = max(4, int(new_w * 0.04))
    shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(blur))
    shadow_mask = shadow_mask.point(lambda p: int(p * 0.45))
    black = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 255))
    black.putalpha(shadow_mask)
    out = Image.alpha_composite(out.convert("RGBA"), black).convert("RGB")

    # Subject on top, feathered mask to soften the cut edge.
    feather = subj_mask.filter(ImageFilter.GaussianBlur(1.2))
    out.paste(subj, (x, y), feather)
    return out


def run_composite(image: Image.Image, instruction: str, *, seed: int | None) -> tuple[Image.Image, str]:
    """Full composite pipeline. Returns (result_image, background_scene_prompt)."""
    mask = segment(image)
    width, height = _sdxl_size(image)
    bg = generate_background(instruction, width, height, seed)
    result = compose(image, mask, bg)
    return result, scene_prompt(instruction)
