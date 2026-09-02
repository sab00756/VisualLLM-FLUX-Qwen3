"""Job creation and lookup backed by Redis + RQ.

Job state lives in a Redis hash ``job:{id}`` that both the API and the worker
read/write. RQ is used only as the dispatch mechanism (the worker runs
``app.tasks.run_edit`` with our job id); the authoritative status vocabulary
(queued/running/done/error) is our own hash.
"""
from __future__ import annotations

import json
import time
import uuid
from functools import lru_cache
from typing import Optional

import redis
from rq import Queue

from .config import get_settings
from .schemas import EditParams, EditPlan, JobStatus

QUEUE_NAME = "edits"
TASK_PATH = "app.tasks.run_edit"
HEARTBEAT_KEY = "worker:heartbeat"
MODEL_LOADED_KEY = "worker:model_loaded"
JOB_TTL_SECONDS = 24 * 3600


def _key(job_id: str) -> str:
    return f"job:{job_id}"


@lru_cache
def get_redis() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


@lru_cache
def get_queue() -> Queue:
    # A distinct, non-decoding connection for RQ (it stores pickled payloads).
    conn = redis.Redis.from_url(get_settings().redis_url)
    return Queue(QUEUE_NAME, connection=conn)


def new_job_id() -> str:
    return uuid.uuid4().hex


def create_job(
    *,
    job_id: str,
    plan: EditPlan,
    params: EditParams,
    src_path: str,
    result_path: str,
) -> str:
    r = get_redis()
    now = time.time()
    r.hset(
        _key(job_id),
        mapping={
            "status": "queued",
            "plan": plan.model_dump_json(),
            "params": params.model_dump_json(),
            "src_path": src_path,
            "result_path": result_path,
            "result_format": params.output_format,
            "created_at": now,
            "error": "",
            "finished_at": "",
        },
    )
    r.expire(_key(job_id), JOB_TTL_SECONDS)
    # Long timeout: model load + inference can take a while on first run.
    get_queue().enqueue(TASK_PATH, job_id, job_id=job_id, job_timeout=1800)
    return job_id


def get_job(job_id: str) -> Optional[JobStatus]:
    data = get_redis().hgetall(_key(job_id))
    if not data:
        return None
    plan = None
    if data.get("plan"):
        try:
            plan = EditPlan(**json.loads(data["plan"]))
        except Exception:
            plan = None
    return JobStatus(
        job_id=job_id,
        status=data.get("status", "unknown"),
        plan=plan,
        error=data.get("error") or None,
        created_at=float(data["created_at"]) if data.get("created_at") else None,
        finished_at=float(data["finished_at"]) if data.get("finished_at") else None,
        result_format=data.get("result_format") or None,
        engineered_prompt=data.get("engineered_prompt") or None,
        engineered_by=data.get("engineered_by") or None,
    )


def result_path_for(job_id: str) -> Optional[str]:
    return get_redis().hget(_key(job_id), "result_path")


def worker_health() -> dict:
    try:
        r = get_redis()
        hb = r.get(HEARTBEAT_KEY)
        loaded = r.get(MODEL_LOADED_KEY)
    except Exception as exc:
        return {"alive": False, "model_loaded": False, "heartbeat_age_s": None,
                "error": f"redis unreachable: {exc}"}
    age = (time.time() - float(hb)) if hb else None
    return {
        "alive": age is not None and age < 60,
        "model_loaded": loaded == "1",
        "heartbeat_age_s": round(age, 1) if age is not None else None,
    }
