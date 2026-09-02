# Local Image-Edit Service (a self-hosted "Nano Banana")

Take an image + a plain-English prompt, get back an edited image — running
**entirely locally** on this box (NVIDIA **GB10 / DGX Spark**), wrapped in an
HTTP API. No Nano Banana, no OpenAI key.

Each request is **auto-routed** to the right technique for the image:

- **Regenerative editing** — **Qwen-Image-Edit-2509** (default, Apache-2.0) or
  **FLUX.1 Kontext [dev]** regenerates the scene around a subject and applies
  edits like recoloring from a single natural-language instruction. Best for
  flat graphics/icons (turned into realistic 3D scenes) and general edits.
- **Segment-and-composite** — for real product photos: the subject is cut out
  (BiRefNet), a scene is generated (SDXL), and the subject is placed back with
  its **exact pixels** preserved. Best when identity must not drift.
- A **vision model** (Qwen2.5-VL) looks at the image to write a grounded edit
  instruction and to decide which path above to use.
- Resizing / ecommerce-sizing is done deterministically with Pillow (no model).

Handles prompts like:

| Prompt | What happens |
| --- | --- |
| `show this switch on a restaurant background` | auto-route → model edit (background swap) |
| `change the red part to blue, keep everything else unchanged, and make the final image 1920 × 1080` | model edit **+** resize to 1920×1080 |
| `place this product on a marble counter` (real photo) | auto-route → segment + generate scene + composite |
| `resize this image for an ecommerce website and use the recommended sizing` | deterministic resize to an ecom preset (no model) |

---

## Architecture

```
client ──HTTP──> [ api ] ──enqueue──> [ redis ] ──> [ worker (GPU) ] ──> FLUX.1 Kontext
   ^                │                                     │
   └──── poll ──────┘                                     └─ writes result image
      shared /data volume (uploads + results) · hf-cache volume (model weights)
```

- **api** — FastAPI, lightweight. Validates uploads, parses the prompt into an
  *edit plan*, enqueues a job, serves status/result and a test UI. (`api/`)
- **worker** — GPU container. Loads the editor (Qwen-Image-Edit or FLUX Kontext)
  **once** and keeps it warm; lazily loads the vision model, and the segmenter +
  background model for the composite path. Processes one job at a time (single
  GPU → naturally serialized), runs edit/composite + resize. (`worker/`)
- **redis** — job queue + job state.

The prompt is split by a **deterministic router** (`api/app/router.py`) into an
optional model instruction and an optional resize spec. Pure-resize prompts skip
the GPU entirely.

Per job, the path is chosen (`mode=auto` by default) by an **intent-aware** router:
the vision model reads the image **and** the instruction and picks the regenerative
editor for "integrate/install/place naturally in a scene" requests (and for flat
graphics), or the composite path when the intent is "keep the subject's exact pixels
on a plain/studio backdrop". `mode=edit` or `mode=composite` forces a path; the
chosen route (`auto:edit` / `auto:composite` / `forced:*`) is reported back in
`GET /jobs/{id}` as `route`.

---

## Prerequisites

- Docker + Docker Compose, NVIDIA Container Toolkit with the `nvidia` runtime
  (already present on DGX Spark).
- The default **`qwen`** editor and all other models are **ungated** — no token
  needed. Only the optional **`flux`** backend requires a **Hugging Face token**:
  1. Accept the license at
     <https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev>
  2. Create a read token at <https://huggingface.co/settings/tokens>
  3. Set `HF_TOKEN=hf_xxx` and `EDITOR_BACKEND=flux` in `.env`.

---

## Quick start

```bash
cp .env.example .env
# defaults work out of the box (Qwen editor, ungated — no token needed).
# For the FLUX backend instead: set EDITOR_BACKEND=flux and HF_TOKEN=hf_xxx.
docker compose up --build
```

First start downloads the model weights into the `hf-cache` volume and loads the
editor — the Qwen editor is ~52 GB, plus the vision model (~16 GB) and the
composite-path models (SDXL ~7 GB, BiRefNet) which load lazily on first use.
Watch progress with `docker compose logs -f worker`. The service is ready when:

```bash
curl -s localhost:8000/readyz     # {"ready": true, ...}
```

Then open the test UI at **<http://localhost:8000>** or use the API below.

---

## Frontend (hand it to a user)

A self-contained web UI is served by the **same** FastAPI app at **<http://localhost:8000/>** —
upload an image, type an instruction, watch the before → after result. Nothing to deploy separately.

To give someone just the API + a UI, you have two options:

1. **Simplest** — hand them the running URL. `http://<host>:8000/` is the UI; `http://<host>:8000/docs` is the API.
2. **Host the UI elsewhere** — `api/app/static/index.html` is a single standalone file. Serve it from any
   static host, open it, click **⚙ API** in the header, and point it at your API base URL (stored in the
   browser). Cross-origin calls are allowed because the API sends CORS headers — restrict them with
   `ALLOW_ORIGINS` in `.env` (default `*`).

## Prompt-engineering layer

Terse prompts ("cafe table") produce weak edits. Before the editor runs, the
worker expands the instruction into a rich instruction — naming what to preserve
and adding scene/lighting/realism context.

- **`ENHANCER_BACKEND=vlm`** (default): a **vision** model
  (`Qwen2.5-VL-7B-Instruct`, ~16 GB) that actually **sees the image**, so the
  rewrite names the subject's real, observed attributes ("the matte-gray device
  with a red button and green indicator light"). Same model powers the
  auto-routing classifier. Loads lazily, stays warm (~6 s per call).
- **`local`**: a text-only instruct model (`Qwen2.5-7B-Instruct`) — rewrites
  blind (no image).
- **`openai`**: call any OpenAI-compatible endpoint (`ENHANCER_BASE_URL` /
  `ENHANCER_MODEL` / `ENHANCER_API_KEY`) — e.g. your own vLLM/LiteLLM.
- **`template`**: deterministic, no LLM. Also the automatic fallback if the model
  is unavailable.

The engineered prompt is returned in `GET /jobs/{id}` (`engineered_prompt`,
`engineered_by` = `vlm` | `llm` | `template` | `composite`) and shown in the UI.
Per-request toggle: `enhance=true|false` on `POST /edit` (UI checkbox, on by
default). On the composite path, `engineered_prompt` holds the generated
background-scene prompt.

## Access key

Set `API_KEY` in `.env` to lock the model down — without it, the endpoints are open to anyone
who can reach the port. When set, `POST /edit` and the job endpoints require the key:

- header `X-API-Key: <key>`, or `Authorization: Bearer <key>`
- the web UI shows a key field (stored in the browser) and sends it automatically
- `GET /healthz`, `/readyz`, `/authz` stay open (needed for health checks)

```bash
curl -H "X-API-Key: $KEY" -F image=@switch.png -F 'prompt=...' http://localhost:8000/edit
```

Rotate by changing `API_KEY` in `.env` and `docker compose up -d api`.

## Sharing over Tailscale

The API listens on `0.0.0.0:8000`. To share it beyond this machine, front it with Tailscale
(keep `API_KEY` set so access is gated):

```bash
tailscale serve --bg 8000     # private: reachable only by devices on your tailnet
tailscale funnel --bg 8000    # public: https://<machine>.<tailnet>.ts.net, gated by API_KEY
```

Give the recipient the resulting `https://…ts.net` URL (UI + API on the same origin) and the key.
`tailscale funnel off` / `tailscale serve --https=443 off` to stop sharing.

## API

Interactive docs at **<http://localhost:8000/docs>**.

### `POST /edit` — submit a job

Multipart form:

| field | required | notes |
| --- | --- | --- |
| `image` | ✓ | png / jpeg / webp |
| `prompt` | ✓ | natural-language instruction |
| `mode` | | `auto` (default) / `edit` / `composite` — path selection (see below) |
| `enhance` | | `true` (default) / `false` — run the prompt-engineering layer |
| `steps` | | diffusion steps (default 28; editor path only) |
| `guidance` | | guidance scale (default 2.5; FLUX editor only) |
| `seed` | | int, for reproducibility |
| `output_format` | | `png` (default) / `jpeg` / `webp` |

`mode`: `auto` reads the prompt + image and routes by intent — "integrate/install
into a scene" (or a flat graphic) → the editor; "keep exact pixels on a plain
backdrop" → composite. `edit` forces the regenerative editor; `composite` forces
segment + generate-scene + place. (`composite=true` is a legacy alias for
`mode=composite`.)

Returns `202 { "job_id": "...", "status": "queued", "plan": {...} }`.

### `GET /jobs/{id}` — status

```json
{
  "status": "queued|running|done|error",
  "plan": {...},
  "error": null,
  "engineered_prompt": "…",
  "engineered_by": "vlm|llm|template|composite",
  "route": "auto:photo->composite"
}
```

### `GET /jobs/{id}/result` — the final image

Streams the image once `status == done` (`409` if not ready, `422` on failure).

### Health

- `GET /healthz` — API liveness.
- `GET /readyz` — `200` only when the worker is alive **and** the model is loaded.

### Example (combined recolor + resize)

```bash
JOB=$(curl -s -F image=@samples/switch.png \
  -F 'prompt=change the red part to blue, keep everything else unchanged, and make the final image 1920 x 1080' \
  http://localhost:8000/edit | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')

# poll until done
until curl -s http://localhost:8000/jobs/$JOB | grep -q '"done"'; do sleep 2; done

curl -s http://localhost:8000/jobs/$JOB/result -o out.png
```

---

## Configuration (`.env`)

| var | default | meaning |
| --- | --- | --- |
| `EDITOR_BACKEND` | `qwen` | editor: `qwen` (Qwen-Image-Edit-2509) or `flux` (FLUX.1 Kontext) |
| `QWEN_EDIT_MODEL_ID` | `Qwen/Qwen-Image-Edit-2509` | diffusers model id for the qwen backend |
| `QWEN_TRUE_CFG` | `4.0` | true-CFG scale for the qwen backend |
| `MODEL_ID` | `black-forest-labs/FLUX.1-Kontext-dev` | diffusers model id for the flux backend |
| `HF_TOKEN` | — | required **only** for the gated FLUX backend; Qwen is ungated |
| `ENHANCER_BACKEND` | `vlm` | prompt layer + router classifier: `vlm` / `local` / `openai` / `template` |
| `ENHANCER_VLM_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | vision model for `vlm` |
| `SEG_MODEL` / `BG_MODEL` | `ZhengPeng7/BiRefNet` / `stabilityai/stable-diffusion-xl-base-1.0` | composite-path segmenter / background model |
| `SUBJECT_SCALE` / `SUBJECT_VPOS` | `0.6` / `0.92` | composite subject width fraction / bottom position |
| `TORCH_DTYPE` | `bfloat16` | inference precision |
| `DEFAULT_STEPS` / `DEFAULT_GUIDANCE` | `28` / `2.5` | diffusion defaults |
| `DEFAULT_ECOM_PRESET` | `shopify` | preset for "recommended ecom sizing" |
| `MAX_UPLOAD_MB` / `MAX_OUTPUT_DIM` / `MAX_PROMPT_CHARS` | `20` / `4096` / `2000` | input limits |
| `RESULT_TTL_HOURS` | `24` | job/result retention |
| `API_PORT` | `8000` | host port |

**Ecom presets** (`api/app/config.py`): `shopify` 2048², `amazon` 2000²,
`generic` 1600² — each pads the image (aspect-preserving) onto a white square.

---

## Tests

Router logic is unit-tested without a GPU:

```bash
python3 -m venv .venv-test && .venv-test/bin/pip install -r tests/requirements.txt
.venv-test/bin/python -m pytest tests/ -q
```

---

## Notes & tradeoffs

- **First run is slow** (weight download + model load); cached afterward via the
  `hf-cache` volume. `/readyz` gates callers until the model is warm.
- **aarch64 / Blackwell (sm_121):** the worker installs PyTorch from the CUDA
  13.0 wheel index; the "max CUDA capability 12.0" warning is safe to ignore
  (sm_120/121 are binary-compatible).
- **License:** the default editor **Qwen-Image-Edit-2509 is Apache-2.0**
  (commercial use OK), as are Qwen2.5-VL and BiRefNet (MIT); SDXL is OpenRAIL++.
  The optional **FLUX.1 Kontext [dev]** backend is **non-commercial** — fine for
  personal use only. Keep `EDITOR_BACKEND=qwen` for a commercial-safe stack.
- **Single GPU:** requests are processed one at a time; the queue absorbs bursts.
