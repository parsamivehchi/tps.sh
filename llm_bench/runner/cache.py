"""File-based cache for cloud API results to avoid re-spending money."""

import json
import hashlib
from pathlib import Path
from dataclasses import asdict

from llm_bench.models.base import ModelResponse


def _cache_key(model_name: str, prompt_id: str) -> str:
    """Generate a deterministic cache key."""
    raw = f"{model_name}::{prompt_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_cache_dir(run_dir: Path) -> Path:
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cached_result(run_dir: Path, model_name: str, prompt_id: str) -> ModelResponse | None:
    """Check if a cached result exists for this model+prompt combo."""
    cache_dir = get_cache_dir(run_dir)
    key = _cache_key(model_name, prompt_id)
    cache_file = cache_dir / f"{key}.json"
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text())
        return ModelResponse(**data)
    except Exception:
        return None


def save_to_cache(run_dir: Path, response: ModelResponse) -> None:
    """Save a result to the file cache."""
    cache_dir = get_cache_dir(run_dir)
    key = _cache_key(response.model_name, response.prompt_id)
    cache_file = cache_dir / f"{key}.json"
    cache_file.write_text(json.dumps(asdict(response), indent=2))
