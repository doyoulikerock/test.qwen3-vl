import argparse
import json
import sys
from pathlib import Path

from . import config, pipeline


def cmd_index(args: argparse.Namespace) -> None:
    video_path = str(Path(args.video).resolve())
    try:
        result = pipeline.run_index(
            video_path,
            fps=args.fps,
            width=args.width,
            batch_size=args.batch_size,
            force=args.force,
            with_motion=args.with_motion,
            clip_window=args.clip_window,
            clip_stride=args.clip_stride,
            clip_width=args.clip_width,
            clip_batch_size=args.clip_batch_size,
            describe=not args.no_describe,
        )
    except ValueError as e:
        print(e)
        return
    _print_timings(result.timings, result.total_seconds)


def cmd_search(args: argparse.Namespace) -> None:
    video_path = str(Path(args.video).resolve())
    try:
        result = pipeline.run_search(
            video_path,
            args.query,
            query_en=args.query_en,
            bilingual=args.bilingual,
            recall=args.recall,
            gap=args.gap,
            rerank_top=args.rerank_top,
            no_rerank=args.no_rerank,
            no_motion=args.no_motion,
            top=args.top,
        )
    except pipeline.IndexMissing as e:
        print(e)
        return

    if args.json:
        print(json.dumps(
            {
                "results": [_segment_to_dict(s) for s in result.segments],
                "timings": result.timings,
                "total_seconds": result.total_seconds,
            },
            ensure_ascii=False, indent=2,
        ))
    else:
        _print_table(result.segments, video_path)
        _print_timings(result.timings, result.total_seconds)


def cmd_ask(args: argparse.Namespace) -> None:
    video_path = str(Path(args.video).resolve())
    try:
        result = pipeline.run_ask(
            video_path,
            args.question,
            start=args.start,
            end=args.end,
            max_frames=args.max_frames,
            max_new_tokens=args.max_new_tokens,
        )
    except pipeline.IndexMissing as e:
        print(e)
        return
    except ValueError as e:
        print(e)
        return

    if args.json:
        print(json.dumps(
            {
                "start": result.start,
                "end": result.end,
                "sampled_t": [f["t_sec"] for f in result.frames],
                "question": result.question,
                "answer": result.answer,
                "truncated": result.truncated,
                "timings": result.timings,
                "total_seconds": result.total_seconds,
            },
            ensure_ascii=False, indent=2,
        ))
    else:
        print()
        print(result.answer)
        if result.truncated:
            print(f"\n[잘림] max-new-tokens {args.max_new_tokens} 제한에 걸려 답변이 중간에 끊겼습니다. "
                  f"--max-new-tokens 를 올려 다시 물어보세요.")
        _print_timings(result.timings, result.total_seconds)


def cmd_web(args: argparse.Namespace) -> None:
    from . import web  # deferred: pulls in the HTML page and http.server machinery

    web.serve(host=args.host, port=args.port, open_browser=args.open, videos_dir=args.videos)


def _segment_to_dict(s) -> dict:
    return {
        "start": s.start,
        "end": s.end,
        "peak_t": s.peak_t,
        "score": s.max_score,
        "thumb": s.peak_thumb,
        "clip": s.peak_clip,
    }


def _print_timings(timings: list[dict], total: float) -> None:
    if not timings:
        return
    width = max(len(t["name"]) for t in timings)
    print()
    print("timing")
    for t in timings:
        share = t["seconds"] / total * 100 if total else 0.0
        print(f"  {t['name']:<{width}}  {t['seconds']:7.2f}s  {share:5.1f}%")
    print(f"  {'total':<{width}}  {total:7.2f}s")


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
    p_index.add_argument(
        "--no-describe",
        action="store_true",
        help="skip the one-paragraph video summary written into the manifest "
             "(it loads the 4B explain model; skipped automatically when that model is not cached)",
    )
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

    p_ask = sub.add_parser(
        "ask",
        help="Ask Qwen3-VL-4B-Instruct an open-ended question about an already-indexed video, "
             "optionally limited to a time range (e.g. how many people, what are they doing, what color is X)",
    )
    p_ask.add_argument("video")
    p_ask.add_argument("--start", default=None, help="range start: seconds (203) or HH:MM:SS(.ms); defaults to the start of the video")
    p_ask.add_argument("--end", default=None, help="range end: seconds (210) or HH:MM:SS(.ms); defaults to the end of the video")
    p_ask.add_argument(
        "--question",
        default="이 프레임들은 한 영상 구간에서 뽑은 것이다. 사람이 최대 몇 명까지 동시에 보이는지 숫자로 먼저 답하고, "
                 "근거를 한 문장으로 설명해줘.",
        help="the question to ask; defaults to a people-count question, but any question works",
    )
    p_ask.add_argument("--max-frames", type=int, default=config.DEFAULT_EXPLAIN_MAX_FRAMES)
    p_ask.add_argument(
        "--max-new-tokens", type=int, default=config.DEFAULT_EXPLAIN_MAX_NEW_TOKENS,
        help="answer length budget; the answer is cut off mid-sentence when it runs out "
             "(the run says so when that happens)",
    )
    p_ask.add_argument("--json", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_web = sub.add_parser("web", help="Serve a minimal browser UI for searching indexed videos")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8000)
    p_web.add_argument("--open", action="store_true", help="open the UI in the default browser")
    p_web.add_argument(
        "--videos",
        default=None,
        help="directory scanned for video files to offer in the dropdown (default: the current directory); "
             "files without an index can be selected and indexed from the UI",
    )
    p_web.set_defaults(func=cmd_web)

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
