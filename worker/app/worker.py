"""Worker entrypoint.

Loads FLUX.1 Kontext once (warm model), publishes a heartbeat + readiness
flag to Redis, then runs an RQ ``SimpleWorker`` (no fork -> model stays
resident) that processes edit jobs one at a time.
"""
from __future__ import annotations

import os
import threading
import time

import redis
from rq import Queue, SimpleWorker

from .pipeline import get_pipeline
from .tasks import run_edit  # noqa: F401  (import path used by the queue)

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
HEARTBEAT_KEY = "worker:heartbeat"
MODEL_LOADED_KEY = "worker:model_loaded"
QUEUE_NAME = "edits"


def _heartbeat_loop(status: dict) -> None:
    r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    while True:
        try:
            r.set(HEARTBEAT_KEY, time.time(), ex=90)
            r.set(MODEL_LOADED_KEY, "1" if status.get("loaded") else "0", ex=90)
        except Exception as exc:
            print(f"[worker] heartbeat error: {exc}", flush=True)
        time.sleep(10)


def _preload() -> None:
    attempts = 0
    while True:
        attempts += 1
        try:
            print("[worker] loading FLUX.1 Kontext (first run downloads weights)…", flush=True)
            get_pipeline()
            print("[worker] model loaded — ready.", flush=True)
            return
        except Exception as exc:
            wait = min(60, 5 * attempts)
            print(
                f"[worker] model load failed ({type(exc).__name__}: {exc}). "
                f"Check HF_TOKEN and that the FLUX.1-Kontext-dev license is accepted. "
                f"Retrying in {wait}s…",
                flush=True,
            )
            if attempts >= 5:
                raise
            time.sleep(wait)


def main() -> None:
    status = {"loaded": False}
    threading.Thread(target=_heartbeat_loop, args=(status,), daemon=True).start()

    _preload()
    status["loaded"] = True

    conn = redis.Redis.from_url(REDIS_URL)  # RQ needs bytes, not decoded strings
    queue = Queue(QUEUE_NAME, connection=conn)
    worker = SimpleWorker([queue], connection=conn)
    print("[worker] waiting for jobs…", flush=True)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
