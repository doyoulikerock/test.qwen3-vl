import os
from pathlib import Path


def _hf_hub_cache_dir() -> Path:
    if "HF_HUB_CACHE" in os.environ:
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    return Path(home) / "hub"


def model_is_cached(model_id: str) -> bool:
    """True if at least one complete snapshot of model_id exists in the local HF cache."""
    repo_dir = _hf_hub_cache_dir() / ("models--" + model_id.replace("/", "--"))
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(s.is_dir() and any(s.iterdir()) for s in snapshots.iterdir())


def enable_offline_if_cached(model_ids: list[str]) -> bool:
    """If every model in model_ids is already cached, set HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE
    so no network call is made at all (avoids the "unauthenticated requests" warning and
    any Hub-availability flakiness). Must be called before transformers/sentence_transformers
    are imported in this process, since those libraries read the offline flags at import time.

    Returns True if offline mode was enabled, False if left online (some model not cached yet,
    e.g. first-time download).
    """
    if all(model_is_cached(m) for m in model_ids):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        return True
    return False
