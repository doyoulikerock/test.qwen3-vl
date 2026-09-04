"""The search and ask pipelines, shared by the CLI and the web UI.

Extracted from cli.cmd_search / cli.cmd_ask so both front-ends run exactly the same
ranking and the same per-stage timing instrumentation:

    search: index load -> query encode -> scoring -> segment merge -> rerank
    ask:    index load -> frame sampling -> generate

Every stage is wrapped in Stopwatch.stage(), so `results.timings` carries the same
breakdown the CLI prints and the web UI renders.
"""

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import config, hf_cache, store
from .media import parse_timecode
from .segment import ScoredFrame, Segment, merge_segments


class IndexMissing(Exception):
    """Raised when the requested video has no frame index under data/."""


def _is_degenerate(text: str, repeats: int = 5) -> bool:
    """True when one word dominates the text, the signature of a decoding loop."""
    words = text.split()
    if len(words) < repeats * 2:
        return False
    top = max(set(words), key=words.count)
    return words.count(top) >= repeats and words.count(top) / len(words) > 0.3


def _needs_load(pool, kind: str) -> bool:
    """Whether this run will actually pay for loading `kind` (no pool, or not resident yet)."""
    return pool is None or not pool.is_loaded(kind)


@dataclass
class Stage:
    name: str
    seconds: float


class Progress:
    """Turns a weighted plan of phases into a single 0-100% number.

    The phases of a run are known up front (which models load, whether clips and a summary
    are built), and their relative costs are stable, so a plan of weights gives a percentage
    that means something. Long phases (encoding, reranking) also report items done, which is
    where nearly all of the wall clock goes.
    """

    def __init__(self, plan: list[tuple[str, float]], report: Callable[[dict], None] | None = None):
        self.plan = plan
        self.total_weight = sum(w for _, w in plan) or 1.0
        self.report = report
        self.i = -1
        self.frac = 0.0
        self.done = self.count = 0

    def begin(self, key: str) -> None:
        for i, (name, _) in enumerate(self.plan):
            if name == key and i > self.i:
                self.i, self.frac, self.done, self.count = i, 0.0, 0, 0
                break
        else:
            return  # a phase that was not planned (e.g. skipped work) must not rewind
        self._emit()

    def step(self, done: int, count: int) -> None:
        self.done, self.count = done, count
        self.frac = min(1.0, done / count) if count else 0.0
        self._emit()

    @property
    def percent(self) -> float:
        if self.i < 0:
            return 0.0
        elapsed = sum(w for _, w in self.plan[: self.i]) + self.plan[self.i][1] * self.frac
        return min(100.0, elapsed / self.total_weight * 100.0)

    def snapshot(self) -> dict:
        return {
            "phase": self.plan[self.i][0] if self.i >= 0 else "",
            "percent": self.percent,
            "done": self.done,
            "count": self.count,
        }

    def _emit(self) -> None:
        if self.report:
            self.report(self.snapshot())


class Stopwatch:
    """Records how long each pipeline stage took, logging it as it completes.

    Model load/release are timed separately from the actual work because they dominate
    a cold run (each checkpoint is loaded and freed per query — none coexist in VRAM).
    """

    def __init__(self, log: Callable[[str], None] = print, progress: Progress | None = None):
        self.stages: list[Stage] = []
        self._log = log
        self.progress = progress

    @contextmanager
    def stage(self, name: str, key: str | None = None):
        if self.progress:
            self.progress.begin(key or name)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.stages.append(Stage(name, dt))
            self._log(f"  · {name}: {dt:.2f}s")

    @property
    def total(self) -> float:
        return sum(s.seconds for s in self.stages)

    def as_list(self) -> list[dict]:
        return [{"name": s.name, "seconds": s.seconds} for s in self.stages]


@dataclass
class IndexResult:
    out_dir: str
    frames: int = 0
    clips: int = 0
    rebuilt_frames: bool = False
    rebuilt_clips: bool = False
    description: str | None = None
    removed: list[str] = field(default_factory=list)
    timings: list[dict] = field(default_factory=list)
    total_seconds: float = 0.0


@dataclass
class SearchResult:
    segments: list[Segment]
    # Which scale `Segment.max_score` is on: a reranker logit, or the embedder's cosine
    # similarity when reranking was skipped. The two are not comparable.
    reranked: bool = False
    timings: list[dict] = field(default_factory=list)
    total_seconds: float = 0.0


@dataclass
class AskResult:
    question: str
    answer: str
    start: float
    end: float
    frames: list[dict] = field(default_factory=list)  # {t_sec, frame, thumb} — absolute paths
    truncated: bool = False  # answer ran out of max_new_tokens, so it ends mid-sentence
    timings: list[dict] = field(default_factory=list)
    total_seconds: float = 0.0


def run_index(
    video_path: str,
    *,
    fps: float = config.DEFAULT_FPS,
    width: int = config.DEFAULT_FRAME_WIDTH,
    batch_size: int = 4,
    force: bool = False,
    with_motion: bool = False,
    clip_window: float = config.DEFAULT_CLIP_WINDOW_SEC,
    clip_stride: float = config.DEFAULT_CLIP_STRIDE_SEC,
    clip_width: int = config.DEFAULT_CLIP_WIDTH,
    clip_batch_size: int = config.DEFAULT_CLIP_BATCH_SIZE,
    describe: bool = True,
    log: Callable[[str], None] = print,
    report: Callable[[dict], None] | None = None,
    pool=None,
) -> IndexResult:
    from . import media  # deferred: keeps `import pipeline` free of the ffmpeg helpers

    out_dir = store.video_data_dir(video_path)
    if not Path(video_path).is_file():
        raise ValueError(f"source video not found: {video_path}")

    need_frames = not store.index_exists(out_dir) or force
    need_clips = with_motion and (not store.clip_index_exists(out_dir) or force)
    result = IndexResult(out_dir=str(out_dir), rebuilt_frames=need_frames, rebuilt_clips=need_clips)
    if not need_frames and not need_clips:
        log(f"Index already exists at {out_dir} (use force to rebuild, or with-motion to add a clip index)")
        return result

    # Relative costs of the phases this particular run will go through, measured on real
    # runs: encoding dominates, model loads are a fixed few seconds, ffmpeg is fast.
    plan: list[tuple[str, float]] = [("clear index", 1), ("probe", 1)]
    if need_frames:
        plan += [("frame extract", 4), ("thumb extract", 3), ("scene detect", 3),
                 ("torch import", 6 if pool is None else 1),
                 ("embedder load", 6 if _needs_load(pool, "embedder") else 1),
                 ("encode frames", 45), ("save index", 1)]
    if need_clips:
        plan += [("clip extract", 10),
                 ("clip embedder load", 6 if _needs_load(pool, "embedder") else 1),
                 ("encode clips", 30), ("save clip index", 1)]
    # The summary runs last: its 4B model cannot share the card with the 2B embedder, so
    # loading it any earlier would evict an embedder the clip pass still needs.
    if describe and need_frames:
        plan += [("explainer load", 8), ("describe", 12)]
    sw = Stopwatch(log, Progress(plan, report))

    # Carry the clip parameters forward when the clip index survives a frame-only rebuild —
    # save_index writes a fresh manifest and would otherwise drop them.
    old_manifest = {}
    if (out_dir / "manifest.json").exists():
        with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
            old_manifest = json.load(f)

    if force:
        with sw.stage("clear index"):
            if need_frames:
                result.removed += store.clear_frame_index(out_dir)
            if need_clips:
                result.removed += store.clear_clip_index(out_dir)
        log(f"  cleared: {', '.join(result.removed) if result.removed else '(nothing to remove)'}")

    with sw.stage("probe"):
        info = media.probe_video(video_path)
    log(f"  {info['width']}x{info['height']}  {info['fps']:.3f}fps  {info['duration']:.1f}s")

    if need_frames:
        log(f"Extracting frames (fps={fps}, width={width}) ...")
        with sw.stage("frame extract"):
            frame_paths = media.extract_frames(video_path, out_dir / "frames", fps=fps, width=width)
        with sw.stage("thumb extract"):
            thumb_paths = media.extract_thumbs(
                video_path, out_dir / "thumbs", fps=fps, width=config.DEFAULT_THUMB_WIDTH
            )
        log(f"  {len(frame_paths)} frames extracted")

        with sw.stage("scene detect"):
            boundaries = media.extract_scene_boundaries(video_path)
        log(f"  {len(boundaries)} scene boundaries")

        offline = hf_cache.enable_offline_if_cached([config.EMBEDDING_MODEL_ID])
        log(f"Loading {config.EMBEDDING_MODEL_ID} "
            f"({'offline, cached' if offline else 'online, first download'}) and encoding frames ...")
        with sw.stage("torch import"):
            from .embedder import Embedder  # deferred: seconds on the first run of the process

        with sw.stage("embedder load"):
            embedder = pool.embedder(log) if pool else Embedder()
            embedder.load()
        with sw.stage(f"encode frames ({len(frame_paths)})", "encode frames"):
            embeddings = _encode_chunked(embedder, [str(p) for p in frame_paths],
                                         batch_size, sw, log, "encoding frames")
        if pool is None:
            with sw.stage("embedder release"):
                embedder.release()
        log(f"  encoded {embeddings.shape}")

        with sw.stage("save index"):
            meta = [
                {
                    "idx": i + 1,
                    "t_sec": media.frame_timestamp(i + 1, fps),
                    "frame": str(frame_paths[i].relative_to(out_dir)),
                    "thumb": str(thumb_paths[i].relative_to(out_dir)),
                }
                for i in range(len(frame_paths))
            ]
            manifest = {
                "video_path": video_path,
                "duration": info["duration"],
                "source_fps": info["fps"],
                "index_fps": fps,
                "width": width,
                "embedding_model": config.EMBEDDING_MODEL_ID,
                "scene_boundaries": boundaries,
            }
            if not need_clips and store.clip_index_exists(out_dir):
                manifest.update({k: v for k, v in old_manifest.items() if k.startswith("clip_")})
            if not describe and old_manifest.get("description"):
                # The video itself did not change, only how it is indexed — keep the summary.
                manifest["description"] = old_manifest["description"]
            store.save_index(out_dir, embeddings, meta, manifest)
        result.frames = len(frame_paths)
        log(f"Frame index saved to {out_dir}")

    elif store.index_exists(out_dir):
        log(f"Frame index already exists at {out_dir} — reusing")

    if need_clips:
        log(f"Extracting clips (window={clip_window}s, stride={clip_stride}s, width={clip_width}) ...")
        with sw.stage("clip extract"):
            clips = media.extract_clips(
                video_path,
                out_dir / "clips",
                duration=info["duration"],
                window_sec=clip_window,
                stride_sec=clip_stride,
                width=clip_width,
            )
        log(f"  {len(clips)} clips extracted")

        offline = hf_cache.enable_offline_if_cached([config.EMBEDDING_MODEL_ID])
        log(f"Loading {config.EMBEDDING_MODEL_ID} "
            f"({'offline, cached' if offline else 'online, first download'}) and encoding clips ...")
        from .embedder import Embedder

        with sw.stage("embedder load", "clip embedder load"):
            embedder = pool.embedder(log) if pool else Embedder()
            embedder.load()
        with sw.stage(f"encode clips ({len(clips)})", "encode clips"):
            clip_embeddings = _encode_chunked(embedder, [str(c["clip"]) for c in clips],
                                              clip_batch_size, sw, log, "encoding clips")
        if pool is None:
            with sw.stage("embedder release"):
                embedder.release()
        log(f"  encoded {clip_embeddings.shape}")

        with sw.stage("save clip index"):
            clip_meta = [
                {
                    "idx": c["idx"],
                    "start_t": c["start_t"],
                    "end_t": c["end_t"],
                    "mid_t": c["mid_t"],
                    "clip": str(c["clip"].relative_to(out_dir)),
                }
                for c in clips
            ]
            store.save_clip_index(out_dir, clip_embeddings, clip_meta)
            store.update_manifest(
                out_dir,
                {
                    "clip_window_sec": clip_window,
                    "clip_stride_sec": clip_stride,
                    "clip_width": clip_width,
                },
            )
        result.clips = len(clips)
        log(f"Clip index saved to {out_dir}")


    if describe and need_frames:
        desc = _describe(out_dir, [str(p) for p in frame_paths], sw, log, pool)
        if desc:
            store.update_manifest(out_dir, {"description": desc})
            result.description = desc

    log(f"  total: {sw.total:.2f}s")
    result.timings = sw.as_list()
    result.total_seconds = sw.total
    return result


def _encode_chunked(embedder, paths: list[str], batch_size: int, sw: Stopwatch,
                    log: Callable[[str], None], label: str) -> np.ndarray:
    """Encode in chunks so progress can be reported per item.

    Embeddings are independent per document, so chunking changes nothing about the result —
    it only gives the caller a place to report from. The library's own progress bar is off
    because it writes to a terminal the web UI cannot see.
    """
    chunk = max(batch_size * 4, 8)
    parts = []
    for i in range(0, len(paths), chunk):
        parts.append(embedder.encode_documents(paths[i:i + chunk], batch_size=batch_size,
                                               show_progress_bar=False))
        done = min(i + chunk, len(paths))
        if sw.progress:
            sw.progress.step(done, len(paths))
        log(f"  {label}: {done}/{len(paths)} ({done / len(paths) * 100:.0f}%)")
    return np.concatenate(parts) if parts else np.zeros((0, 0), dtype=np.float32)


def _rank_chunked(reranker, query: str, docs: list[str], sw: Stopwatch,
                  log: Callable[[str], None]) -> list[dict]:
    """Rerank in chunks, for the same reason. A cross-encoder scores each (query, doc) pair
    on its own, so the merged-and-resorted result matches a single ranking call."""
    chunk = 4
    ranked: list[dict] = []
    for i in range(0, len(docs), chunk):
        part = reranker.rank(query, docs[i:i + chunk])
        ranked += [{"corpus_id": r["corpus_id"] + i, "score": r["score"]} for r in part]
        done = min(i + chunk, len(docs))
        if sw.progress:
            sw.progress.step(done, len(docs))
        log(f"  rerank: {done}/{len(docs)} ({done / len(docs) * 100:.0f}%)")
    ranked.sort(key=lambda r: -r["score"])
    return ranked


def _describe(out_dir: Path, frame_paths: list[str], sw: Stopwatch,
              log: Callable[[str], None], pool=None) -> str | None:
    """Summarize the whole video in a sentence or two, for the dropdown tooltip.

    Skipped (rather than silently pulling ~8GB) when the explain model is not cached yet —
    indexing must not turn into a download the caller did not ask for.
    """
    if not frame_paths:
        return None
    if not hf_cache.model_is_cached(config.EXPLAIN_MODEL_ID):
        log(f"  (설명 생성 건너뜀 — {config.EXPLAIN_MODEL_ID} 가 아직 캐시에 없음. "
            f"`vidsearch ask` 를 한 번 실행해 받아두면 다음 인덱싱부터 생성됩니다)")
        return None

    step = max(1, len(frame_paths) // config.DEFAULT_DESCRIBE_MAX_FRAMES)
    sampled = frame_paths[::step][: config.DEFAULT_DESCRIBE_MAX_FRAMES]
    hf_cache.enable_offline_if_cached([config.EXPLAIN_MODEL_ID])
    log(f"Describing the video with {config.EXPLAIN_MODEL_ID} ({len(sampled)} frames) ...")

    from .explain import Explainer  # deferred: heavy torch/transformers import

    explainer = None
    try:
        with sw.stage("explainer load"):
            explainer = pool.explainer(log) if pool else Explainer()
            explainer.load()
        with sw.stage(f"describe ({len(sampled)} frames)", "describe"):
            answer, truncated = explainer.ask(
                config.DESCRIBE_PROMPT, sampled,
                max_new_tokens=config.DEFAULT_DESCRIBE_MAX_NEW_TOKENS,
            )
    except Exception as e:
        # A failed summary must not fail the index that was already written.
        log(f"  (설명 생성 실패: {type(e).__name__}: {e})")
        return None
    finally:
        # The 4B model cannot share the card with the 2B ones, so it is dropped even when
        # pooled — keeping it would only force an eviction on the next search.
        if explainer is not None:
            with sw.stage("explainer release"):
                pool.evict("explainer") if pool is not None else explainer.release()

    desc = " ".join(answer.split())
    if _is_degenerate(desc):
        # Greedy decoding plus a repetition penalty makes this rare, but a tooltip reading
        # "빨간색 빨간색 빨간색 ..." is worse than no tooltip at all.
        log(f"  (설명이 같은 표현을 반복해 버려서 저장하지 않습니다: {desc[:60]}...)")
        return None
    if truncated:
        log(f"  (설명이 {config.DEFAULT_DESCRIBE_MAX_NEW_TOKENS} 토큰 제한에 걸려 끝이 잘렸습니다)")
    log(f"  description: {desc}")
    return desc


def run_search(
    video_path: str,
    query: str,
    *,
    query_en: str | None = None,
    bilingual: bool = False,
    recall: int = config.DEFAULT_RECALL_TOP_M,
    gap: float = config.DEFAULT_SEGMENT_GAP_SEC,
    rerank_top: int = config.DEFAULT_RERANK_TOP_SEGMENTS,
    no_rerank: bool = False,
    no_motion: bool = False,
    top: int = 10,
    log: Callable[[str], None] = print,
    report: Callable[[dict], None] | None = None,
    pool=None,
) -> SearchResult:
    out_dir = store.video_data_dir(video_path)
    if not store.index_exists(out_dir):
        raise IndexMissing(f"No index at {out_dir}. Run `vidsearch index {video_path}` first.")

    use_motion = not no_motion and store.clip_index_exists(out_dir)
    # A resident model turns its load phase into a no-op, so the plan must not reserve
    # weight for it — otherwise the bar would stall at a phase that takes no time.
    plan: list[tuple[str, float]] = [
        ("index load", 1), ("torch import", 8 if pool is None else 1),
        ("embedder load", 22 if _needs_load(pool, "embedder") else 1), ("query encode", 4),
    ]
    if pool is None:
        plan.append(("embedder release", 1))
    plan.append(("frame scoring", 1))
    if use_motion:
        plan.append(("clip scoring", 1))
    plan.append(("segment merge", 1))
    if not no_rerank:
        plan += [("reranker load", 18 if _needs_load(pool, "reranker") else 1), ("rerank", 50)]
        if pool is None:
            plan.append(("reranker release", 1))
    sw = Stopwatch(log, Progress(plan, report))

    with sw.stage("index load"):
        embeddings, meta, manifest = store.load_index(out_dir)
        clip_embeddings = clip_meta = None
        if use_motion:
            clip_embeddings, clip_meta = store.load_clip_index(out_dir)

    required_models = [config.EMBEDDING_MODEL_ID]
    if not no_rerank:
        required_models.append(config.RERANKER_MODEL_ID)
    if not hf_cache.enable_offline_if_cached(required_models):
        log("(some model not cached yet — this run needs network to download it)")

    with sw.stage("torch import"):
        from .embedder import Embedder  # deferred: seconds on the first run of the process

    queries = [query]
    if bilingual and query_en:
        queries.append(query_en)
    log(f"Encoding quer{'ies' if len(queries) > 1 else 'y'}: {queries}")

    with sw.stage("embedder load"):
        embedder = pool.embedder(log) if pool else Embedder()
        embedder.load()
    with sw.stage("query encode"):
        q_embs = [embedder.encode_query(q) for q in queries]
    if pool is None:
        with sw.stage("embedder release"):
            embedder.release()

    with sw.stage("frame scoring"):
        scores = np.max(np.stack([embeddings @ q for q in q_embs]), axis=0)
        top_m = min(recall, len(scores))
        top_idx = np.argsort(-scores)[:top_m]
        scored = [
            ScoredFrame(
                idx=meta[i]["idx"],
                t_sec=meta[i]["t_sec"],
                score=float(scores[i]),
                frame_path=str(out_dir / meta[i]["frame"]),
                thumb_path=str(out_dir / meta[i]["thumb"]),
            )
            for i in top_idx
        ]

    if use_motion:
        with sw.stage("clip scoring"):
            clip_scores = np.max(np.stack([clip_embeddings @ q for q in q_embs]), axis=0)
            clip_top_m = min(recall, len(clip_scores))
            clip_top_idx = np.argsort(-clip_scores)[:clip_top_m]
            for ci in clip_top_idx:
                mid_t = clip_meta[ci]["mid_t"]
                nearest = min(meta, key=lambda m: abs(m["t_sec"] - mid_t))
                scored.append(
                    ScoredFrame(
                        idx=nearest["idx"],
                        t_sec=mid_t,
                        score=float(clip_scores[ci]),
                        frame_path=str(out_dir / nearest["frame"]),
                        thumb_path=str(out_dir / nearest["thumb"]),
                        clip_path=str(out_dir / clip_meta[ci]["clip"]),
                    )
                )
        if clip_top_m:
            log(f"  motion channel: {clip_top_m} clip candidates merged "
                f"(top clip score {float(clip_scores[clip_top_idx[0]]):.4f})")

    with sw.stage("segment merge"):
        segments = merge_segments(scored, gap_sec=gap, scene_boundaries=manifest.get("scene_boundaries", []))
        segments.sort(key=lambda s: -s.max_score)

    if not no_rerank and segments:
        from .reranker import Reranker

        rerank_n = min(rerank_top, len(segments))
        candidates = segments[:rerank_n]
        # Prefer the clip itself over a single still frame when the segment's peak evidence
        # came from the motion channel — a frozen mid-stride pose reads as ambiguous to the
        # reranker (verified: a running frame scored 0.0/-0.19 alone vs 0.31 as its source clip).
        rerank_docs = [c.peak_clip if c.peak_clip else c.peak_frame for c in candidates]
        with sw.stage("reranker load"):
            reranker = pool.reranker(log) if pool else Reranker()
            reranker.load()
        with sw.stage(f"rerank ({rerank_n} segments)", "rerank"):
            ranked = _rank_chunked(reranker, query, rerank_docs, sw, log)
        if pool is None:
            with sw.stage("reranker release"):
                reranker.release()
        order = [r["corpus_id"] for r in ranked]
        rerank_scores = {r["corpus_id"]: r["score"] for r in ranked}
        reranked = [candidates[i] for i in order]
        for i, seg in zip(order, reranked):
            seg.max_score = float(rerank_scores[i])
        segments = reranked + segments[rerank_n:]

    log(f"  total: {sw.total:.2f}s")
    return SearchResult(
        segments=segments[: min(top, len(segments))],
        reranked=not no_rerank and bool(segments),
        timings=sw.as_list(),
        total_seconds=sw.total,
    )


def run_ask(
    video_path: str,
    question: str,
    *,
    start: str | float | None = None,
    end: str | float | None = None,
    max_frames: int = config.DEFAULT_EXPLAIN_MAX_FRAMES,
    max_new_tokens: int = config.DEFAULT_EXPLAIN_MAX_NEW_TOKENS,
    log: Callable[[str], None] = print,
    report: Callable[[dict], None] | None = None,
    pool=None,
) -> AskResult:
    out_dir = store.video_data_dir(video_path)
    if not store.index_exists(out_dir):
        raise IndexMissing(f"No index at {out_dir}. Run `vidsearch index {video_path}` first.")

    plan = [("index load", 1), ("frame sampling", 1), ("torch import", 8 if pool is None else 1),
            ("explainer load", 52 if _needs_load(pool, "explainer") else 1), ("generate", 37)]
    if pool is None:
        plan.append(("explainer release", 1))
    sw = Stopwatch(log, Progress(plan, report))

    with sw.stage("index load"):
        _, meta, manifest = store.load_index(out_dir)

    # Both bounds are optional: omitting them asks the question about the whole video.
    t_start = _as_seconds(start, 0.0)
    t_end = _as_seconds(end, max(manifest.get("duration", 0.0), meta[-1]["t_sec"] if meta else 0.0))
    if t_end <= t_start:
        raise ValueError("end must be after start")

    with sw.stage("frame sampling"):
        in_range = [m for m in meta if t_start <= m["t_sec"] <= t_end]
        if not in_range:
            raise ValueError(f"No indexed frames between {t_start:.2f}s and {t_end:.2f}s.")
        step = max(1, len(in_range) // max_frames)
        sampled = in_range[::step][:max_frames]
        frames = [
            {
                "t_sec": m["t_sec"],
                "frame": str(out_dir / m["frame"]),
                "thumb": str(out_dir / m["thumb"]),
            }
            for m in sampled
        ]
    stamps = [f"{f['t_sec']:.1f}s" for f in frames]
    log(f"Sampling {len(frames)} frame(s) from {t_start:.2f}s-{t_end:.2f}s: {stamps}")

    if not hf_cache.enable_offline_if_cached([config.EXPLAIN_MODEL_ID]):
        log("(explain model not cached yet — this run needs network to download it)")

    with sw.stage("torch import"):
        from .explain import Explainer  # deferred: seconds on the first run of the process

    with sw.stage("explainer load"):
        explainer = pool.explainer(log) if pool else Explainer()
        explainer.load()
    with sw.stage(f"generate ({len(frames)} frames)", "generate"):
        answer, truncated = explainer.ask(
            question, [f["frame"] for f in frames], max_new_tokens=max_new_tokens
        )
    if pool is None:
        with sw.stage("explainer release"):
            explainer.release()

    if truncated:
        log(f"  (답변이 max-new-tokens {max_new_tokens} 제한에 걸려 끝이 잘렸습니다 — 값을 올려 다시 물어보세요)")
    log(f"  total: {sw.total:.2f}s")
    return AskResult(
        question=question,
        answer=answer,
        start=t_start,
        end=t_end,
        frames=frames,
        truncated=truncated,
        timings=sw.as_list(),
        total_seconds=sw.total,
    )


def _as_seconds(value: str | float | None, default: float) -> float:
    if value is None or value == "":
        return default
    return parse_timecode(value) if isinstance(value, str) else float(value)


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".flv", ".ts", ".mpg", ".mpeg"}


def find_videos(scan_dir: Path | None = None) -> list[dict]:
    """Indexed videos, plus any video file in scan_dir that has no index yet.

    Both kinds are addressed by the same id (the filename stem, which is also the data/
    directory name), so an unindexed file can be selected, played and indexed through the
    same endpoints; `indexed` says which it is.
    """
    entries = {v["id"]: v for v in list_indexed_videos()}
    if scan_dir and scan_dir.is_dir():
        for path in sorted(scan_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
                continue
            existing = entries.get(path.stem)
            if existing is not None:
                existing["in_workspace"] = True
                continue
            entries[path.stem] = {
                "id": path.stem,
                "video_path": str(path.resolve()),
                "name": path.name,
                "duration": 0.0,  # unknown until indexed; probing every file per request is not worth it
                "description": "",
                "size": path.stat().st_size,
                "has_motion": False,
                "has_source": True,
                "indexed": False,
                "in_workspace": True,
                "index_fps": config.DEFAULT_FPS,
                "width": config.DEFAULT_FRAME_WIDTH,
                "clip_window_sec": config.DEFAULT_CLIP_WINDOW_SEC,
                "clip_stride_sec": config.DEFAULT_CLIP_STRIDE_SEC,
                "clip_width": config.DEFAULT_CLIP_WIDTH,
            }
    # Indexed first, then alphabetical — the ready-to-search ones stay at the top.
    return sorted(entries.values(), key=lambda v: (not v["indexed"], v["name"].lower()))


def list_indexed_videos() -> list[dict]:
    """Every video under data/ that has a usable frame index."""
    root = config.DATA_ROOT
    if not root.is_dir():
        return []
    videos = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not store.index_exists(d):
            continue
        try:
            # Read the manifest only — loading embeddings.npy just to list videos is wasteful.
            with open(d / "manifest.json", "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            continue
        video_path = manifest.get("video_path", str(d))
        videos.append(
            {
                "id": d.name,
                "video_path": video_path,
                "name": Path(video_path).name,
                "duration": manifest.get("duration", 0.0),
                "description": manifest.get("description", ""),
                "size": Path(video_path).stat().st_size if Path(video_path).is_file() else 0,
                "has_motion": store.clip_index_exists(d),
                "indexed": True,
                "in_workspace": False,
                # Current index settings, so the web UI can prefill its re-index form.
                "index_fps": manifest.get("index_fps", config.DEFAULT_FPS),
                "width": manifest.get("width", config.DEFAULT_FRAME_WIDTH),
                "clip_window_sec": manifest.get("clip_window_sec", config.DEFAULT_CLIP_WINDOW_SEC),
                "clip_stride_sec": manifest.get("clip_stride_sec", config.DEFAULT_CLIP_STRIDE_SEC),
                "clip_width": manifest.get("clip_width", config.DEFAULT_CLIP_WIDTH),
                # The index outlives the source file, so the UI must know whether the
                # original video is still there before offering to play it.
                "has_source": Path(video_path).is_file(),
            }
        )
    return videos
