"""Keeps loaded models resident between jobs, for long-lived processes (the web server).

The CLI loads a checkpoint, uses it and exits, so it pays the load once either way. A server
does not: reloading the same 2B model for every search costs ~2.5s of pure overhead, and
~8.3s for the 4B one on every question.

What can stay resident together is a VRAM question, measured on a 12GB card:

    embedder  (Qwen3-VL-Embedding-2B)   4.0G weights, 4.7G peak
    reranker  (Qwen3-VL-Reranker-2B)    4.0G weights, +0.7G peak while ranking
    explainer (Qwen3-VL-4B-Instruct)    8.3G weights, 9.6G peak over 8 frames

So the two 2B models coexist (8.6G peak — a repeated search reloads nothing), while the 4B
one needs the card to itself. Rather than hard-coding that rule, the pool asks the driver how
much memory is actually free — other applications use this GPU too — and evicts least
recently used models until the next one fits.
"""

import threading
import time
from typing import Callable

GIB = 1 << 30

# Starting estimates of weight memory; each is corrected by measurement after its first load,
# so a different card or dtype converges on its own numbers.
_WEIGHTS_GIB = {"embedder": 4.0, "reranker": 4.0, "explainer": 8.4}

# Room left over the weights for activations, the largest observed working set (the 4B model
# over 8 frames) plus a little slack.
_HEADROOM_GIB = 1.4


class ModelPool:
    """One live instance per model kind, evicted only to make room.

    Not thread-safe on its own: the server runs one job at a time under its model lock, which
    is the same serialization this relies on.
    """

    def __init__(self, log: Callable[[str], None] = print):
        self._live: dict[str, object] = {}
        self._used: dict[str, float] = {}
        self._log = log
        self.last_use = time.time()

    # ---- public API -------------------------------------------------------

    def is_loaded(self, kind: str) -> bool:
        return kind in self._live

    def embedder(self, log: Callable[[str], None] | None = None):
        from .embedder import Embedder
        return self._acquire("embedder", Embedder, log)

    def reranker(self, log: Callable[[str], None] | None = None):
        from .reranker import Reranker
        return self._acquire("reranker", Reranker, log)

    def explainer(self, log: Callable[[str], None] | None = None):
        from .explain import Explainer
        return self._acquire("explainer", Explainer, log)

    def evict(self, kind: str) -> bool:
        model = self._live.pop(kind, None)
        if model is None:
            return False
        self._used.pop(kind, None)
        model.release()
        self._log(f"  released {kind} ({self._free_gib():.1f}G free)")
        return True

    def evict_all(self) -> list[str]:
        return [k for k in list(self._live) if self.evict(k)]

    def resident(self) -> list[str]:
        return sorted(self._live)

    # ---- internals --------------------------------------------------------

    def _acquire(self, kind: str, factory, log: Callable[[str], None] | None):
        say = log or self._log
        self.last_use = time.time()
        if kind in self._live:
            self._used[kind] = time.time()
            say(f"  {kind}: already resident (no load)")
            return self._live[kind]

        self._make_room(kind, say)
        model = factory()
        before = self._free_gib()
        try:
            model.load()
        except Exception:
            # An OOM here usually means the estimate was optimistic — drop everything and
            # retry once with the whole card free.
            if self.evict_all():
                say(f"  {kind}: load failed, retrying with an empty GPU")
                model = factory()
                before = self._free_gib()
                model.load()
            else:
                raise
        used = before - self._free_gib()
        if used > 0.1:
            _WEIGHTS_GIB[kind] = used  # measured beats estimated
        self._live[kind] = model
        self._used[kind] = time.time()
        say(f"  {kind}: loaded ({used:.1f}G, {self._free_gib():.1f}G free)")
        return model

    def _make_room(self, kind: str, say: Callable[[str], None]) -> None:
        need = _WEIGHTS_GIB.get(kind, 4.0) + _HEADROOM_GIB
        while self._free_gib() < need and self._live:
            victim = min(self._used, key=self._used.get)
            say(f"  need {need:.1f}G for {kind}, {self._free_gib():.1f}G free — evicting {victim}")
            self.evict(victim)

    def _free_gib(self) -> float:
        try:
            import torch

            if not torch.cuda.is_available():
                return float("inf")
            return torch.cuda.mem_get_info()[0] / GIB
        except Exception:
            return float("inf")  # no CUDA: nothing to budget, models live on CPU


class IdleEvictor:
    """Frees the GPU after a stretch with no jobs.

    A server left open should not hold 8GB of a workstation's card all night just because
    someone ran one search at lunchtime.
    """

    def __init__(self, pool: ModelPool, idle_seconds: float, lock: threading.Lock,
                 log: Callable[[str], None] = print):
        self.pool = pool
        self.idle = idle_seconds
        # The same lock jobs hold: a model must never be freed out from under a running one.
        self.lock = lock
        self._log = log

    def start(self) -> None:
        if self.idle <= 0:
            return
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            time.sleep(min(30.0, max(5.0, self.idle / 4)))
            if not self.pool.resident() or time.time() - self.pool.last_use <= self.idle:
                continue
            with self.lock:
                # Re-check: a job may have started (and used the models) while we waited.
                if self.pool.resident() and time.time() - self.pool.last_use > self.idle:
                    self._log(f"idle for {self.idle:.0f}s — releasing {', '.join(self.pool.resident())}")
                    self.pool.evict_all()
