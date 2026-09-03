import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import config, hf_cache, media, store
from .segment import ScoredFrame, merge_segments


def cmd_index(args: argparse.Namespace) -> None:
    video_path = str(Path(args.video).resolve())
    out_dir = store.video_data_dir(video_path)

    need_frame_index = not store.index_exists(out_dir) or args.force
    need_clip_index = args.with_motion and (not store.clip_index_exists(out_dir) or args.force)

    if not need_frame_index and not need_clip_index:
        print(f"Index already exists at {out_dir} (use --force to rebuild, or --with-motion to add a clip index)")
        return

    print(f"Probing {video_path} ...")
    info = media.probe_video(video_path)
    print(f"  {info['width']}x{info['height']}  {info['fps']:.3f}fps  {info['duration']:.1f}s")

    if need_frame_index:
        print(f"Extracting frames (fps={args.fps}, width={args.width}) ...")
        t0 = time.time()
        frame_paths = media.extract_frames(video_path, out_dir / "frames", fps=args.fps, width=args.width)
        thumb_paths = media.extract_thumbs(video_path, out_dir / "thumbs", fps=args.fps, width=config.DEFAULT_THUMB_WIDTH)
        print(f"  {len(frame_paths)} frames extracted in {time.time() - t0:.1f}s")

        print("Detecting scene boundaries ...")
        boundaries = media.extract_scene_boundaries(video_path)
        print(f"  {len(boundaries)} boundaries: {boundaries}")

        offline = hf_cache.enable_offline_if_cached([config.EMBEDDING_MODEL_ID])
        print(f"Loading {config.EMBEDDING_MODEL_ID} ({'offline, cached' if offline else 'online, first download'}) and encoding frames ...")
        from .embedder import Embedder  # deferred: heavy torch/transformers import

        embedder = Embedder()
        t0 = time.time()
        embeddings = embedder.encode_documents(
            [str(p) for p in frame_paths],
            batch_size=args.batch_size,
        )
        embedder.release()
        print(f"  encoded {embeddings.shape} in {time.time() - t0:.1f}s")

        meta = [
            {
                "idx": i + 1,
                "t_sec": media.frame_timestamp(i + 1, args.fps),
                "frame": str(frame_paths[i].relative_to(out_dir)),
                "thumb": str(thumb_paths[i].relative_to(out_dir)),
            }
            for i in range(len(frame_paths))
        ]
        manifest = {
            "video_path": video_path,
            "duration": info["duration"],
            "source_fps": info["fps"],
            "index_fps": args.fps,
            "width": args.width,
            "embedding_model": config.EMBEDDING_MODEL_ID,
            "scene_boundaries": boundaries,
        }
        store.save_index(out_dir, embeddings, meta, manifest)
        print(f"Frame index saved to {out_dir}")
    else:
        print(f"Frame index already exists at {out_dir} — reusing")

    if need_clip_index:
        print(f"Extracting clips (window={args.clip_window}s, stride={args.clip_stride}s, width={args.clip_width}) ...")
        t0 = time.time()
        clips = media.extract_clips(
            video_path,
            out_dir / "clips",
            duration=info["duration"],
            window_sec=args.clip_window,
            stride_sec=args.clip_stride,
            width=args.clip_width,
        )
        print(f"  {len(clips)} clips extracted in {time.time() - t0:.1f}s")

        offline = hf_cache.enable_offline_if_cached([config.EMBEDDING_MODEL_ID])
        print(f"Loading {config.EMBEDDING_MODEL_ID} ({'offline, cached' if offline else 'online, first download'}) and encoding clips ...")
        from .embedder import Embedder  # deferred: heavy torch/transformers import

        embedder = Embedder()
        t0 = time.time()
        clip_embeddings = embedder.encode_documents(
            [str(c["clip"]) for c in clips],
            batch_size=args.clip_batch_size,
        )
        embedder.release()
        print(f"  encoded {clip_embeddings.shape} in {time.time() - t0:.1f}s")

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
                "clip_window_sec": args.clip_window,
                "clip_stride_sec": args.clip_stride,
                "clip_width": args.clip_width,
            },
        )
        print(f"Clip index saved to {out_dir}")


def cmd_search(args: argparse.Namespace) -> None:
    video_path = str(Path(args.video).resolve())
    out_dir = store.video_data_dir(video_path)
    if not store.index_exists(out_dir):
        print(f"No index at {out_dir}. Run `vidsearch index {args.video}` first.")
        return

    embeddings, meta, manifest = store.load_index(out_dir)

    use_motion = not args.no_motion and store.clip_index_exists(out_dir)
    clip_embeddings = clip_meta = None
    if use_motion:
        clip_embeddings, clip_meta = store.load_clip_index(out_dir)

    required_models = [config.EMBEDDING_MODEL_ID]
    if not args.no_rerank:
        required_models.append(config.RERANKER_MODEL_ID)
    offline = hf_cache.enable_offline_if_cached(required_models)
    if not offline:
        print("(some model not cached yet — this run needs network to download it)")

    from .embedder import Embedder  # deferred: heavy torch/transformers import

    embedder = Embedder()
    queries = [args.query]
    if args.bilingual and args.query_en:
        queries.append(args.query_en)

    print(f"Encoding quer{'ies' if len(queries) > 1 else 'y'}: {queries}")
    q_embs = [embedder.encode_query(q) for q in queries]
    embedder.release()

    scores = np.max(np.stack([embeddings @ q for q in q_embs]), axis=0)

    top_m = min(args.recall, len(scores))
    top_idx = np.argsort(-scores)[:top_m]

    frames_dir = out_dir
    scored = [
        ScoredFrame(
            idx=meta[i]["idx"],
            t_sec=meta[i]["t_sec"],
            score=float(scores[i]),
            frame_path=str(frames_dir / meta[i]["frame"]),
            thumb_path=str(frames_dir / meta[i]["thumb"]),
        )
        for i in top_idx
    ]

    if use_motion:
        clip_scores = np.max(np.stack([clip_embeddings @ q for q in q_embs]), axis=0)
        clip_top_m = min(args.recall, len(clip_scores))
        clip_top_idx = np.argsort(-clip_scores)[:clip_top_m]
        for ci in clip_top_idx:
            mid_t = clip_meta[ci]["mid_t"]
            nearest = min(meta, key=lambda m: abs(m["t_sec"] - mid_t))
            scored.append(
                ScoredFrame(
                    idx=nearest["idx"],
                    t_sec=mid_t,
                    score=float(clip_scores[ci]),
                    frame_path=str(frames_dir / nearest["frame"]),
                    thumb_path=str(frames_dir / nearest["thumb"]),
                    clip_path=str(frames_dir / clip_meta[ci]["clip"]),
                )
            )
        print(f"  motion channel: {clip_top_m} clip candidates merged (top clip score {float(clip_scores[clip_top_idx[0]]):.4f})")

    segments = merge_segments(scored, gap_sec=args.gap, scene_boundaries=manifest.get("scene_boundaries", []))
    segments.sort(key=lambda s: -s.max_score)

    if not args.no_rerank and segments:
        from .reranker import Reranker

        rerank_n = min(args.rerank_top, len(segments))
        candidates = segments[:rerank_n]
        # Prefer the clip itself over a single still frame when the segment's peak evidence
        # came from the motion channel — a frozen mid-stride pose reads as ambiguous to the
        # reranker (verified: a running frame scored 0.0/-0.19 alone vs 0.31 as its source clip).
        rerank_docs = [c.peak_clip if c.peak_clip else c.peak_frame for c in candidates]
        reranker = Reranker()
        ranked = reranker.rank(args.query, rerank_docs)
        reranker.release()
        order = [r["corpus_id"] for r in ranked]
        rerank_scores = {r["corpus_id"]: r["score"] for r in ranked}
        reranked = [candidates[i] for i in order]
        for i, seg in zip(order, reranked):
            seg.max_score = float(rerank_scores[i])
        segments = reranked + segments[rerank_n:]

    top_n = min(args.top, len(segments))
    results = segments[:top_n]

    if args.json:
        print(json.dumps([_segment_to_dict(s) for s in results], ensure_ascii=False, indent=2))
    else:
        _print_table(results, video_path)


def _segment_to_dict(s) -> dict:
    return {
        "start": s.start,
        "end": s.end,
        "peak_t": s.peak_t,
        "score": s.max_score,
        "thumb": s.peak_thumb,
        "clip": s.peak_clip,
    }


def _fmt_hhmmss(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _print_table(results: list, video_path: str) -> None:
    if not results:
        print("No results.")
        return
    print(f"{'time':>14}  {'score':>8}  thumb")
    for s in results:
        print(f"{_fmt_hhmmss(s.peak_t):>14}  {s.max_score:8.4f}  {s.peak_thumb}")
    best = results[0]
    print()
    print(f'ffplay -ss {best.peak_t:.2f} "{video_path}"')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vidsearch")
    sub = p.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Extract frames and build an embedding index for a video")
    p_index.add_argument("video")
    p_index.add_argument("--fps", type=float, default=config.DEFAULT_FPS)
    p_index.add_argument("--width", type=int, default=config.DEFAULT_FRAME_WIDTH)
    p_index.add_argument("--batch-size", type=int, default=4)
    p_index.add_argument("--force", action="store_true")
    p_index.add_argument("--with-motion", action="store_true", help="also build a clip-embedding index for motion/action queries")
    p_index.add_argument("--clip-window", type=float, default=config.DEFAULT_CLIP_WINDOW_SEC)
    p_index.add_argument("--clip-stride", type=float, default=config.DEFAULT_CLIP_STRIDE_SEC)
    p_index.add_argument("--clip-width", type=int, default=config.DEFAULT_CLIP_WIDTH)
    p_index.add_argument("--clip-batch-size", type=int, default=config.DEFAULT_CLIP_BATCH_SIZE)
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="Search a previously indexed video with a natural-language query")
    p_search.add_argument("video")
    p_search.add_argument("query")
    p_search.add_argument("--query-en", default=None, help="English translation of the query, used with --bilingual")
    p_search.add_argument("--bilingual", action="store_true")
    p_search.add_argument("--recall", type=int, default=config.DEFAULT_RECALL_TOP_M)
    p_search.add_argument("--gap", type=float, default=config.DEFAULT_SEGMENT_GAP_SEC)
    p_search.add_argument("--rerank-top", type=int, default=config.DEFAULT_RERANK_TOP_SEGMENTS)
    p_search.add_argument("--no-rerank", action="store_true")
    p_search.add_argument("--no-motion", action="store_true", help="ignore the clip index even if present")
    p_search.add_argument("--top", type=int, default=10)
    p_search.add_argument("--json", action="store_true")
    p_search.set_defaults(func=cmd_search)

    return p


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
