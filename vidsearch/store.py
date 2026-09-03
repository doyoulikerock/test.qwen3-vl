import json
import shutil
from pathlib import Path

import numpy as np

from . import config


def video_data_dir(video_path: str) -> Path:
    stem = Path(video_path).stem
    return config.DATA_ROOT / stem


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_normalized_embeddings(path: Path) -> np.ndarray:
    embeddings = np.load(path).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def save_index(
    out_dir: Path,
    embeddings: np.ndarray,
    meta: list[dict],
    manifest: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings.astype(np.float16))
    _write_jsonl(out_dir / "meta.jsonl", meta)
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def load_index(out_dir: Path) -> tuple[np.ndarray, list[dict], dict]:
    embeddings = _load_normalized_embeddings(out_dir / "embeddings.npy")
    meta = _read_jsonl(out_dir / "meta.jsonl")
    with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return embeddings, meta, manifest


def index_exists(out_dir: Path) -> bool:
    return (
        (out_dir / "embeddings.npy").exists()
        and (out_dir / "meta.jsonl").exists()
        and (out_dir / "manifest.json").exists()
    )


def update_manifest(out_dir: Path, updates: dict) -> None:
    """Merge `updates` into the existing manifest.json (used to record clip-index params
    without needing to redo the frame index's save_index call)."""
    path = out_dir / "manifest.json"
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.update(updates)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def save_clip_index(out_dir: Path, embeddings: np.ndarray, clip_meta: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "clip_embeddings.npy", embeddings.astype(np.float16))
    _write_jsonl(out_dir / "clip_meta.jsonl", clip_meta)


def load_clip_index(out_dir: Path) -> tuple[np.ndarray, list[dict]]:
    embeddings = _load_normalized_embeddings(out_dir / "clip_embeddings.npy")
    clip_meta = _read_jsonl(out_dir / "clip_meta.jsonl")
    return embeddings, clip_meta


def clip_index_exists(out_dir: Path) -> bool:
    return (out_dir / "clip_embeddings.npy").exists() and (out_dir / "clip_meta.jsonl").exists()


def _remove(out_dir: Path, names: list[str]) -> list[str]:
    removed = []
    for name in names:
        path = out_dir / name
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(name + "/")
        elif path.exists():
            path.unlink()
            removed.append(name)
    return removed


def clear_frame_index(out_dir: Path) -> list[str]:
    """Delete the frame index and the images it was built from.

    Rebuilding with different options must start from an empty directory: extraction
    collects its output by globbing (media._run_ffmpeg_extract), so frames left over from
    a run at a different --fps/--width would be folded into the new index and silently
    desynchronize meta.jsonl from the embeddings.
    """
    return _remove(out_dir, ["frames", "thumbs", "embeddings.npy", "meta.jsonl"])


def clear_clip_index(out_dir: Path) -> list[str]:
    """Same, for the motion/clip index."""
    return _remove(out_dir, ["clips", "clip_embeddings.npy", "clip_meta.jsonl"])
