"""FastAPI entrypoint: upload + prompt -> async job -> result image."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .config import get_settings
from .jobs import create_job, get_job, new_job_id, result_path_for, worker_health
from .router import parse_prompt
from .schemas import EditParams, JobCreated

settings = get_settings()

DATA_DIR = Path(settings.data_dir)
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path(__file__).parent / "static"

ALLOWED_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}
OUT_MEDIA = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}

app = FastAPI(
    title="Local Image Edit Service",
    description=(
        "Nano-Banana-style instruction image editing, running locally. Auto-routes "
        "each image between a regenerative editor (Qwen-Image-Edit / FLUX Kontext) and "
        "a segment-and-composite path (BiRefNet + SDXL) that preserves the subject's "
        "exact pixels."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_key(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    """Gate protected endpoints when API_KEY is configured. Open if unset."""
    if not settings.api_key:
        return
    supplied = x_api_key
    if not supplied and authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if supplied != settings.api_key:
        raise HTTPException(401, "Missing or invalid API key")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.get("/authz")
def authz() -> dict:
    """Whether this deployment requires a key (lets the UI show the field)."""
    return {"auth_required": bool(settings.api_key)}


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> JSONResponse:
    health = worker_health()
    ready = health["alive"] and health["model_loaded"]
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"ready": ready, "worker": health},
    )


@app.post("/edit", status_code=202, response_model=JobCreated, dependencies=[Depends(require_key)])
async def edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    steps: int | None = Form(None),
    guidance: float | None = Form(None),
    seed: int | None = Form(None),
    output_format: str = Form("png"),
    enhance: bool = Form(True),
    mode: str = Form("auto"),
    composite: bool = Form(False),
) -> JobCreated:
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported image type: {image.content_type}")
    if len(prompt) > settings.max_prompt_chars:
        raise HTTPException(413, "Prompt too long")
    if output_format not in OUT_MEDIA:
        raise HTTPException(400, "output_format must be png, jpeg or webp")
    if mode not in ("auto", "edit", "composite"):
        raise HTTPException(400, "mode must be auto, edit or composite")

    body = await image.read()
    if len(body) == 0:
        raise HTTPException(400, "Empty upload")
    if len(body) > settings.max_upload_bytes:
        raise HTTPException(413, f"Image exceeds {settings.max_upload_mb} MB limit")

    plan = parse_prompt(prompt)
    params = EditParams(
        steps=steps, guidance=guidance, seed=seed,
        output_format=output_format, enhance=enhance, mode=mode, composite=composite,
    )

    job_id = new_job_id()
    src_ext = ALLOWED_TYPES[image.content_type]
    src_path = UPLOAD_DIR / f"{job_id}{src_ext}"
    src_path.write_bytes(body)
    result_path = RESULT_DIR / f"{job_id}.{output_format}"

    create_job(
        job_id=job_id,
        plan=plan,
        params=params,
        src_path=str(src_path),
        result_path=str(result_path),
    )
    return JobCreated(job_id=job_id, status="queued", plan=plan)


@app.get("/jobs/{job_id}", dependencies=[Depends(require_key)])
def job_status(job_id: str) -> dict:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    return job.model_dump()


@app.get("/jobs/{job_id}/result", dependencies=[Depends(require_key)])
def job_result(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    if job.status == "error":
        raise HTTPException(422, f"Job failed: {job.error}")
    if job.status != "done":
        raise HTTPException(409, f"Job not ready (status={job.status})")
    path = result_path_for(job_id)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Result file missing")
    fmt = job.result_format or "png"
    return FileResponse(path, media_type=OUT_MEDIA.get(fmt, "image/png"))
