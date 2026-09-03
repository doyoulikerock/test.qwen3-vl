import gc

import torch
from sentence_transformers import CrossEncoder

from . import config


class Reranker:
    """Loads Qwen3-VL-Reranker-2B on demand; must be released before loading
    the embedder or explain model — none of these coexist in VRAM."""

    def __init__(self, model_id: str = config.RERANKER_MODEL_ID, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self._model: CrossEncoder | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        self._model = CrossEncoder(
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

    def rank(self, query: str, documents: list[str]) -> list[dict]:
        """documents: image file paths. Returns list of {'corpus_id', 'score'} sorted desc."""
        self.load()
        return self._model.rank(query, documents)
