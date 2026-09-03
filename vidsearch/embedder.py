import gc

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from . import config


class Embedder:
    """Loads Qwen3-VL-Embedding-2B on demand; must be released before loading
    the reranker or any other model — the two do not fit in VRAM together."""

    def __init__(self, model_id: str = config.EMBEDDING_MODEL_ID, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self._model: SentenceTransformer | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        self._model = SentenceTransformer(
            self.model_id,
            device=self.device,
            model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"},
            processor_kwargs={"min_pixels": config.MIN_PIXELS, "max_pixels": config.MAX_PIXELS},
        )

    def release(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def encode_documents(
        self,
        frame_paths: list[str],
        batch_size: int = 4,
        truncate_dim: int | None = None,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        self.load()
        for bs in _batch_size_fallback(batch_size):
            try:
                emb = self._model.encode_document(
                    frame_paths,
                    batch_size=bs,
                    normalize_embeddings=True,
                    show_progress_bar=show_progress_bar,
                    truncate_dim=truncate_dim,
                    # Adopt the reference (qwen-vl-utils) per-frame pixel cap for video/clip inputs —
                    # becomes the forced default in transformers v5.22 anyway. Verified on our own
                    # clips (640px, §8) that it changes nothing (pixel_values_videos shape identical
                    # capped vs uncapped) since our clip resolution is already below the uncapped
                    # ceiling; setting it explicitly just silences the per-call warning and
                    # future-proofs against wider --clip-width values. No-op for image inputs.
                    processing_kwargs={"video": {"cap_pixels_per_frame": True}},
                )
                return np.asarray(emb, dtype=np.float32)
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                continue
        raise RuntimeError("OOM even at batch_size=1; reduce --width or use truncate_dim")

    def encode_query(self, text: str, truncate_dim: int | None = None) -> np.ndarray:
        self.load()
        emb = self._model.encode_query(
            text,
            normalize_embeddings=True,
            truncate_dim=truncate_dim,
        )
        return np.asarray(emb, dtype=np.float32)


def _batch_size_fallback(start: int):
    bs = start
    seen = set()
    while bs >= 1:
        if bs not in seen:
            seen.add(bs)
            yield bs
        if bs == 1:
            break
        bs = bs // 2
