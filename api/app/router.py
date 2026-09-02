"""Deterministic prompt router.

Splits a natural-language prompt into:
  * an optional *edit instruction* (for the diffusion model), and
  * an optional *resize spec* (deterministic Pillow post-step).

No LLM involved — pure regex + keyword rules. Kept as a pure function so it is
trivially unit-testable and easy to extend.
"""
from __future__ import annotations

import re

from .config import ECOM_PRESETS, get_settings
from .schemas import EditPlan, ResizeSpec


# Two integers joined by x / × / X / * / "by"  ->  (width, height)
_DIM_RE = re.compile(r"(\d{2,5})\s*(?:x|×|X|\*|by)\s*(\d{2,5})")

# The resize clause plus common lead-ins, so we can strip it from the
# instruction handed to the model.
_RESIZE_CLAUSE_RE = re.compile(
    r"""
    (?:,\s*)?                                  # optional leading comma
    (?:\band\b\s*)?                            # optional 'and'
    (?:                                        # optional lead-in verb phrase
        (?:make|set|resize|size|scale|change|render|output|export|save|crop)
        (?:\s+(?:the|it|this|them|final|image|picture|photo|size|dimensions?|resolution|to|at))*
      | (?:to|at|into)
    )?
    \s*
    \d{2,5}\s*(?:x|×|X|\*|by)\s*\d{2,5}        # the actual WxH
    \s*(?:px|pixels?|resolution)?              # optional unit suffix
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Ecommerce / marketplace intent, and explicit preset names.
_ECOM_HINT_RE = re.compile(
    r"\b(e-?commerce|ecom|marketplace|online\s+store|product\s+(?:listing|page)|"
    r"shopify|amazon|etsy)\b",
    re.IGNORECASE,
)
_ECOM_CLAUSE_RE = re.compile(
    r"""
    (?:,\s*)?(?:\band\b\s*)?
    (?:(?:resize|size|optimi[sz]e|format|prepare|make)\s+
        (?:this|it|the|them)?\s*(?:image|picture|photo)?\s*)?
    (?:for|to|as)?\s*
    (?:a|an|the)?\s*
    (?:use\s+(?:the\s+)?)?(?:recommended\s+)?(?:sizing|size)?\s*
    (?:for\s+)?
    \b(?:e-?commerce|ecom|marketplace|online\s+store|product\s+(?:listing|page)|
       shopify|amazon|etsy)\b
    (?:\s+(?:website|websites|site|sites|listing|page|standards?|guidelines?))?
    (?:\s+(?:sizing|size|dimensions?))?
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Words that carry no editing meaning on their own. If, after stripping the
# resize/ecom clauses, only these remain, there is no model work to do.
_FILLER = {
    "resize", "resized", "resizing", "image", "images", "picture", "photo",
    "this", "that", "these", "those", "it", "the", "a", "an", "and", "for",
    "of", "to", "please", "kindly", "my", "make", "final", "use", "using",
    "recommended", "sizing", "size", "website", "websites", "site", "sites",
    "online", "store", "product", "raw", "with", "into", "as", "keep",
    "everything", "else", "unchanged", "same",
}

_PRESET_NAME_RE = re.compile(r"\b(shopify|amazon|etsy)\b", re.IGNORECASE)


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    # trim dangling connectors / punctuation left by clause removal
    text = re.sub(r"^[\s,;.:\-]+|[\s,;.:\-]+$", "", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\b(and|,)\s*$", "", text, flags=re.IGNORECASE).strip(" ,")
    return text.strip()


def _is_meaningful(text: str) -> bool:
    # Ignore filler words and stray single characters left by clause removal.
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return any(len(tok) > 1 and tok not in _FILLER for tok in tokens)


def _clamp(v: int, max_dim: int) -> int:
    return max(1, min(int(v), max_dim))


def parse_prompt(prompt: str) -> EditPlan:
    """Interpret a raw prompt into an :class:`EditPlan`."""
    settings = get_settings()
    max_dim = settings.max_output_dim
    text = (prompt or "").strip()
    resize: ResizeSpec | None = None

    # 1) Explicit W x H wins.
    dim_match = _DIM_RE.search(text)
    if dim_match:
        w = _clamp(int(dim_match.group(1)), max_dim)
        h = _clamp(int(dim_match.group(2)), max_dim)
        resize = ResizeSpec(width=w, height=h, mode="stretch", source="explicit")
        text = _RESIZE_CLAUSE_RE.sub(" ", text)

    # 2) Otherwise, ecommerce intent -> a named preset.
    elif _ECOM_HINT_RE.search(text):
        preset_name = settings.default_ecom_preset
        named = _PRESET_NAME_RE.search(text)
        if named and named.group(1).lower() in ECOM_PRESETS:
            preset_name = named.group(1).lower()
        preset = ECOM_PRESETS.get(preset_name, ECOM_PRESETS["generic"])
        resize = ResizeSpec(
            width=_clamp(preset["width"], max_dim),
            height=_clamp(preset["height"], max_dim),
            mode=preset["mode"],
            background=preset["background"],
            source=f"ecom:{preset_name}",
        )
        text = _ECOM_CLAUSE_RE.sub(" ", text)

    # 3) Whatever remains is the (possible) model instruction.
    remainder = _clean(text)
    instruction = remainder if _is_meaningful(remainder) else None

    return EditPlan(instruction=instruction, resize=resize)
