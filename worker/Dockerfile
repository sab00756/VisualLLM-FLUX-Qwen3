# GB10 / DGX Spark: aarch64 + Blackwell (sm_121). CUDA 13 runtime base.
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    VENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

RUN python3.12 -m venv $VENV && pip install --upgrade pip

# PyTorch for CUDA 13.0 (aarch64 wheels; work on GB10 — the "max capability 12.0"
# warning is safe to ignore, sm_120/121 are binary-compatible).
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

CMD ["python", "-m", "app.worker"]
