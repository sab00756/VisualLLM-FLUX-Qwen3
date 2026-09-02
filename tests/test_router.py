"""Unit tests for the deterministic prompt router (no GPU / model needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from app.router import parse_prompt  # noqa: E402


def test_background_only():
    plan = parse_prompt("for this raw image of a switch show this with the background of a restaurant")
    assert plan.needs_model
    assert "restaurant" in plan.instruction.lower()
    assert plan.resize is None


def test_recolor_plus_explicit_dims():
    plan = parse_prompt(
        "change the red part to blue, keep everything else unchanged, "
        "and make the final image 1920 x 1080"
    )
    assert plan.needs_model
    assert "blue" in plan.instruction.lower()
    # the resize clause must be stripped from the model instruction
    assert "1920" not in plan.instruction
    assert plan.resize is not None
    assert (plan.resize.width, plan.resize.height) == (1920, 1080)


def test_unicode_times_separator():
    plan = parse_prompt("recolor the handle to green and resize to 1024×1024")
    assert plan.resize is not None
    assert (plan.resize.width, plan.resize.height) == (1024, 1024)
    assert "1024" not in (plan.instruction or "")


def test_ecom_resize_only_no_model():
    plan = parse_prompt(
        "resize this image for an ecommerce website and use the recommended sizing for ecom sites"
    )
    assert not plan.needs_model
    assert plan.instruction is None
    assert plan.resize is not None
    assert plan.resize.source.startswith("ecom:")
    assert plan.resize.width == plan.resize.height  # square preset


def test_ecom_recommended_sizing_word_boundary():
    # "recommended" contains the substring "ecom" — must NOT be treated as a
    # marketplace keyword nor leak into the model instruction.
    plan = parse_prompt(
        "resize this image for an ecommerce website and use the recommended sizing"
    )
    assert not plan.needs_model, plan.instruction
    assert plan.instruction is None
    assert plan.resize is not None and plan.resize.source.startswith("ecom:")


def test_named_marketplace_preset():
    plan = parse_prompt("optimize this for amazon product listing")
    assert plan.resize is not None
    assert plan.resize.source == "ecom:amazon"
    assert (plan.resize.width, plan.resize.height) == (2000, 2000)


def test_plain_resize_no_edit():
    plan = parse_prompt("resize this image to 800x600")
    assert not plan.needs_model
    assert plan.resize is not None
    assert (plan.resize.width, plan.resize.height) == (800, 600)


def test_edit_only_no_resize():
    plan = parse_prompt("make the background a snowy mountain at sunset")
    assert plan.needs_model
    assert plan.resize is None
    assert "mountain" in plan.instruction.lower()


def test_dimension_clamped_to_max():
    plan = parse_prompt("resize to 99999 x 99999")
    assert plan.resize is not None
    assert plan.resize.width <= 4096 and plan.resize.height <= 4096
