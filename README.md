# VisualLLM — FLUX Kontext + Qwen

A small self-hosted service for editing product images from a plain-English
instruction. You send it a photo and a sentence like *"put this on a marble
kitchen counter"* or *"change the red part to blue and make it 1920×1080"*, and
it hands back the edited image. It runs entirely on your own hardware — no
Nano Banana, no OpenAI key, nothing leaves the box.

Two models do the work:

- **FLUX.1 Kontext [dev]** (via `diffusers`) does the actual image edit. It's an
  instruction-following editor, so it keeps your subject and rebuilds the rest —
  swapping a background or recoloring a part is a single pass, no manual masking.
- **Qwen2.5-7B-Instruct** (via `transformers`) rewrites your short prompt into a
  detailed one before it reaches FLUX. Terse prompts give mediocre edits; this
  layer fills in the scene, lighting, and "keep the subject unchanged" wording
  that Kontext responds well to.

Resizing and e-commerce presets are handled with plain Pillow — no model needed
for that part.

---

## How it works

```
                POST /edit                          ┌─ router: prompt → {edit instruction, resize}
   client ─────────────────────►  api  ─── enqueue ─┤
      ▲                            │                └─ writes upload to /data
      │  GET /jobs/{id}            ▼
      └──────── poll ──────── redis (queue + job state)
                                   │
                                   ▼
                                worker  ── 0. Qwen rewrites the instruction
                                           1. FLUX runs the edit
                                           2. Pillow resizes if asked
                                           └─ writes result to /data
```

Three containers, started together with Compose:

| Service  | What it does                                                                 |
|----------|------------------------------------------------------------------------------|
| `api`    | FastAPI. Validates uploads, parses the prompt, queues jobs, serves the UI and results. Lightweight — no GPU, no torch. |
| `worker` | Holds both models in memory and processes one job at a time. This is the GPU container. |
| `redis`  | Job queue and job state.                                                     |

The API is async on purpose: a single GPU can only run one edit at a time, and a
FLUX pass takes over a minute, so `POST /edit` returns a `job_id` immediately and
you poll for the result rather than holding a connection open.

The prompt router (`api/app/router.py`) is deterministic — no LLM. It pulls an
explicit size (`1920x1080`) or an e-commerce preset out of the text, and treats
whatever's left as the edit instruction. A prompt that's *only* a resize skips
the GPU entirely.

---

## Requirements

- An NVIDIA GPU with enough memory for both models — FLUX is ~24 GB in bf16 and
  Qwen2.5-7B is ~15 GB, so plan for ~40 GB, or a unified-memory box like a DGX
  Spark. On a smaller card, set `ENHANCER_BACKEND=template` (or `openai`) to skip
  loading the local Qwen and run FLUX alone.
- Docker + Docker Compose, with the NVIDIA Container Toolkit and the `nvidia`
  runtime available.
- A Hugging Face account (FLUX.1 Kontext is a gated download — see below).

This was built and run on a **DGX Spark (GB10, Grace-Blackwell, aarch64)** with
CUDA 13. The worker image installs PyTorch from the CUDA 13 aarch64 wheel index
for that reason. On a normal x86 + CUDA box you'll want to change the torch index
URL in `worker/Dockerfile` to match your CUDA version.

---

## Setup

**1. Get access to FLUX.1 Kontext.** It's gated. Accept the license once at
<https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev> while logged in,
then create a read token at <https://huggingface.co/settings/tokens>.

**2. Create your `.env`** from the template and drop the token in:

```bash
cp .env.example .env
# edit .env, set HF_TOKEN=hf_...
```

`.env` is gitignored and never committed — it's the only file that holds secrets.

---

## Build and run

```bash
docker compose up -d --build
```

The first start downloads the model weights — roughly 24 GB for FLUX and 15 GB
for Qwen — into a named volume, so it takes a while. Watch it:

```bash
docker compose logs -f worker
```

It's ready when this returns `{"ready": true, ...}`:

```bash
curl -s localhost:8000/readyz
```

Weights persist in the `hf-cache` volume, so later restarts are quick. To stop:

```bash
docker compose down          # keeps the cached weights
docker compose down -v       # also wipes the volumes (re-downloads next time)
```

Once it's up, open **http://localhost:8000** for the web UI, or use the API
below. Interactive API docs are at **http://localhost:8000/docs**.

---

## Using the API

### Submit an edit

`POST /edit` — multipart form:

| field           | required | notes                                    |
|-----------------|----------|------------------------------------------|
| `image`         | yes      | png / jpeg / webp                        |
| `prompt`        | yes      | plain-English instruction                |
| `enhance`       | no       | run the prompt rewriter (default `true`) |
| `steps`         | no       | diffusion steps (default 28)             |
| `seed`          | no       | integer, for reproducible output         |
| `output_format` | no       | `png` (default), `jpeg`, `webp`          |

Returns `202` with a `job_id`.

### Check on it

`GET /jobs/{id}` returns the status (`queued` / `running` / `done` / `error`),
the parsed plan, and — once the worker has run it — the engineered prompt that
was actually sent to FLUX.

`GET /jobs/{id}/result` streams the finished image.

### End-to-end example

```bash
JOB=$(curl -s -F image=@product.jpg \
  -F 'prompt=on a wooden cafe table, soft morning light' \
  http://localhost:8000/edit | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')

# poll until it's done
until curl -s http://localhost:8000/jobs/$JOB | grep -q '"done"'; do sleep 3; done

curl -s http://localhost:8000/jobs/$JOB/result -o out.png
```

### Health

- `GET /healthz` — API is up.
- `GET /readyz` — worker is alive *and* the model is loaded (`200`, else `503`).

---

## The prompt-engineering layer

Short prompts produce weak edits, so before FLUX runs, the worker expands the
instruction. "on a cafe table" becomes something like:

> *Place the product on a weathered wood cafe table with a softly lit café
> background featuring blurred chairs and a checkerboard floor in the distance;
> keep the product's shape, colors, markings, and position exactly as they are,
> and add realistic contact shadows and a subtle reflection on the table.
> Photorealistic, high detail.*

The rewritten prompt comes back in `GET /jobs/{id}` and is shown in the UI, so
you always see what actually ran. Toggle it per request with `enhance`, or turn
it off globally. Three backends, set with `ENHANCER_BACKEND`:

- `local` (default) — Qwen2.5-7B-Instruct runs inside the worker. Loads on first
  use, then stays warm (~4–5 s per rewrite).
- `openai` — call any OpenAI-compatible endpoint instead (your own vLLM, a
  LiteLLM gateway, etc.) via `ENHANCER_BASE_URL` / `ENHANCER_MODEL` /
  `ENHANCER_API_KEY`.
- `template` — no LLM. A deterministic template adds the preservation and
  lighting wording. This is also the automatic fallback if the model is
  unreachable, so you never get a bare prompt through to FLUX.

---

## Configuration

Everything is set in `.env` (see `.env.example` for the full list). The ones you
actually touch:

| Variable               | Default                          | Purpose                                        |
|------------------------|----------------------------------|------------------------------------------------|
| `HF_TOKEN`             | —                                | **Required.** Hugging Face token for FLUX.     |
| `API_KEY`              | *(empty = open)*                 | If set, callers must send `X-API-Key`.         |
| `ENHANCER_BACKEND`     | `local`                          | `local` / `openai` / `template`.               |
| `ENHANCER_LOCAL_MODEL` | `Qwen/Qwen2.5-7B-Instruct`       | Model for the local rewriter.                  |
| `DEFAULT_STEPS`        | `28`                             | Diffusion steps.                               |
| `DEFAULT_ECOM_PRESET`  | `shopify`                        | Which preset "recommended ecom sizing" maps to.|
| `ALLOW_ORIGINS`        | `*`                              | CORS origins allowed to call the API.          |

E-commerce presets (in `api/app/config.py`): `shopify` 2048², `amazon` 2000²,
`generic` 1600² — each pads the image onto a white square, preserving aspect.

### Locking it down

If you're exposing this beyond localhost, set `API_KEY` in `.env`. It gates
`/edit` and the job endpoints; health checks stay open. The UI shows a key field
when the server requires one. To share it off-machine without opening a port,
Tailscale works well — `tailscale serve 8000` for your own tailnet, or
`tailscale funnel 8000` for the public internet with the API key doing the
gating.

---

## Tests

The prompt router has unit tests that don't need a GPU or any model:

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r tests/requirements.txt
./.venv/bin/python -m pytest tests/ -q
```

They cover the parsing that decides whether a prompt is an edit, a resize, or
both — including the fiddly cases like stripping the size out of a combined
prompt and not mistaking "recommended" for the "ecom" keyword.

---

## Notes and limitations

- **FLUX.1 Kontext [dev] is under a non-commercial license.** Fine for personal
  and research use. For commercial use, swap it for something like
  Qwen-Image-Edit (Apache-2.0) — the pipeline is isolated in
  `worker/app/pipeline.py`.
- **One GPU, one job at a time.** Requests queue; the API absorbs bursts. A FLUX
  pass is ~90 s at 28 steps on a GB10.
- **First run is slow** (weights download). After that the cache makes restarts
  quick, and `/readyz` tells you when the model is actually loaded.
- **The abstract-subject caveat:** recolors work best on real product photos.
  Given a very plain or ambiguous shape, FLUX has less to anchor to and may drift
  more than you'd like. Scene edits (backgrounds) hold up regardless.

---

## Layout

```
api/            FastAPI service — endpoints, prompt router, web UI
worker/         GPU worker — FLUX pipeline, Qwen enhancer, resize
tests/          router unit tests (no GPU)
docker-compose.yml
.env.example    copy to .env and fill in
```
