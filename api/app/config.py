"""Environment-driven configuration and shared constants for the API service."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


# Ecommerce sizing presets. Each pads the image (preserving aspect) onto a
# square canvas of the given side length. Values reflect common marketplace
# recommendations (large enough for zoom, square framing).
ECOM_PRESETS: dict[str, dict] = {
    "shopify": {"width": 2048, "height": 2048, "mode": "fit", "background": "white"},
    "amazon": {"width": 2000, "height": 2000, "mode": "fit", "background": "white"},
    "generic": {"width": 1600, "height": 1600, "mode": "fit", "background": "white"},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Model / diffusion defaults (surfaced to callers, executed by the worker)
    model_id: str = "black-forest-labs/FLUX.1-Kontext-dev"
    default_steps: int = 28
    default_guidance: float = 2.5

    # Infra
    redis_url: str = "redis://redis:6379/0"
    data_dir: str = "/data"
    result_ttl_hours: int = 24

    # Limits
    max_upload_mb: int = 20
    max_output_dim: int = 4096
    max_prompt_chars: int = 2000

    # Ecom
    default_ecom_preset: str = "shopify"

    # Access control: if set, callers must send this key as the X-API-Key header
    # (or Authorization: Bearer <key>). Empty = open access.
    api_key: str = ""

    # CORS: comma-separated origins allowed to call the API from a browser.
    # "*" allows any origin (convenient when handing the frontend to someone
    # who will host it separately). Tighten this for real deployments.
    allow_origins: str = "*"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allow_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
