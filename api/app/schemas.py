"""Pydantic models for requests, the parsed edit plan, and job responses."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


ResizeMode = Literal["fit", "cover", "stretch"]


class ResizeSpec(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mode: ResizeMode = "fit"
    background: str = "white"
    # Human-readable note on where this resize came from (preset name / explicit).
    source: str = "explicit"


class EditPlan(BaseModel):
    """The interpreted prompt: an optional model instruction + optional resize."""

    instruction: Optional[str] = None  # None -> no diffusion, deterministic-only
    resize: Optional[ResizeSpec] = None

    @property
    def needs_model(self) -> bool:
        return bool(self.instruction and self.instruction.strip())


class EditParams(BaseModel):
    steps: Optional[int] = Field(default=None, ge=1, le=100)
    guidance: Optional[float] = Field(default=None, ge=0, le=20)
    seed: Optional[int] = None
    output_format: Literal["png", "jpeg", "webp"] = "png"
    enhance: bool = True  # run the prompt-engineering layer before FLUX


class JobCreated(BaseModel):
    job_id: str
    status: str
    plan: EditPlan


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | done | error
    plan: Optional[EditPlan] = None
    error: Optional[str] = None
    created_at: Optional[float] = None
    finished_at: Optional[float] = None
    result_format: Optional[str] = None
    # Prompt-engineering layer output (populated once the worker runs it)
    engineered_prompt: Optional[str] = None
    engineered_by: Optional[str] = None  # llm | template | raw | disabled
