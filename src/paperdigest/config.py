"""Runtime settings from environment variables. Keys come from the environment only."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROVIDERS = ("fake", "openrouter", "openai")
DEFAULT_BASE_URLS = {"openrouter": "https://openrouter.ai/api/v1", "openai": "https://api.openai.com/v1"}
KEY_VARIABLES = {"openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY"}


def _int(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be {minimum} or more, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    output_dir: Path
    seed: int
    llm_provider: str
    llm_model: str
    llm_base_url: str
    llm_timeout: int
    llm_max_retries: int
    chunk_tokens: int
    chunk_overlap: int
    api_key: str = field(default="", repr=False)  # never printed

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.environ.get("PAPERDIGEST_LLM_PROVIDER", "").strip() or "fake"
        if provider not in PROVIDERS:
            raise ValueError(f"PAPERDIGEST_LLM_PROVIDER must be one of {', '.join(PROVIDERS)}")
        chunk_tokens = _int("PAPERDIGEST_CHUNK_TOKENS", 400, 20)
        overlap = _int("PAPERDIGEST_CHUNK_OVERLAP", 50, 0)
        if overlap >= chunk_tokens:
            raise ValueError("PAPERDIGEST_CHUNK_OVERLAP must be smaller than PAPERDIGEST_CHUNK_TOKENS")
        key = os.environ.get(KEY_VARIABLES.get(provider, ""), "").strip() if provider != "fake" else ""
        return cls(
            data_dir=Path(os.environ.get("PAPERDIGEST_DATA_DIR", "").strip() or "data"),
            output_dir=Path(os.environ.get("PAPERDIGEST_OUTPUT_DIR", "").strip() or "runs"),
            seed=_int("PAPERDIGEST_SEED", 42, 0),
            llm_provider=provider,
            llm_model=os.environ.get("PAPERDIGEST_LLM_MODEL", "").strip() or (
                "fake-extractive" if provider == "fake" else "openai/gpt-4o-mini"),
            llm_base_url=os.environ.get("PAPERDIGEST_LLM_BASE_URL", "").strip() or DEFAULT_BASE_URLS.get(provider, ""),
            llm_timeout=_int("PAPERDIGEST_LLM_TIMEOUT", 60, 1),
            llm_max_retries=_int("PAPERDIGEST_LLM_MAX_RETRIES", 3, 0),
            chunk_tokens=chunk_tokens,
            chunk_overlap=overlap,
            api_key=key,
        )
