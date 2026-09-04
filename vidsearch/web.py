"""A minimal web UI for vidsearch — stdlib only (no flask/fastapi needed).

    python -m vidsearch web            # http://127.0.0.1:8000

Routes:
    GET  /                      the single-page UI
    GET  /api/videos            {workspace, videos: indexed + unindexed files in that dir}
    POST /api/search            {video, query, ...} -> ranked segments + stage timings
    POST /api/ask               {video, question, start, end, ...} -> answer + sampled frames
    POST /api/index             {video, fps, ...} -> starts a background job (202)
    GET  /api/job               state/phase/percent/elapsed/log/result of the running job
    GET  /media/<id>/<relpath>  thumbs / frames / clips from that video's data dir
    GET  /video/<id>            the original video file, streamed with Range support
"""

import json
import mimetypes
import re
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlparse

from . import config
from .models import IdleEvictor, ModelPool
from .pipeline import IndexMissing, find_videos, run_ask, run_index, run_search

# Directory scanned for video files that are not indexed yet, so they can be picked from
# the same dropdown and indexed on the spot. Set by serve(); defaults to the working dir.
_scan_dir: Path | None = None


def _videos() -> list[dict]:
    return find_videos(_scan_dir)

# One job at a time: the models do not fit in VRAM twice over.
_model_lock = threading.Lock()

# Unlike the CLI, this process outlives a single job, so loaded models are kept for the next
# one and only evicted to make room (see models.ModelPool).
_pool = ModelPool()


class Job:
    """The one pipeline run this server will do at a time.

    All three kinds (index/search/ask) load models that do not coexist in VRAM, so they are
    serialized anyway — and all three take long enough (a search waits ~20s on model loads,
    an index minutes) that holding a request open for them is the wrong shape. Each runs on
    its own thread while the browser polls /api/job for phase, percent, elapsed and log.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.state = "idle"  # idle | running | done | error
        self.kind = ""       # index | search | ask
        self.video_id: str | None = None
        self.lines: list[str] = []
        self.error: str | None = None
        self.result: dict | None = None
        self.progress: dict = {}
        self.started: float = 0.0
        self.finished: float = 0.0

    def start(self, kind: str, video_id: str, work: Callable[..., dict]) -> bool:
        """work(log, report) -> the JSON payload to hand back to the browser."""
        with self.lock:
            if self.state == "running":
                return False
            self.state = "running"
            self.kind = kind
            self.video_id = video_id
            self.lines = []
            self.error = None
            self.result = None
            self.progress = {"phase": "", "percent": 0.0, "done": 0, "count": 0}
            self.started = time.time()
            self.finished = 0.0
        threading.Thread(target=self._run, args=(work,), daemon=True).start()
        return True

    def _log(self, msg: str) -> None:
        print(f"  {msg}")
        with self.lock:
            self.lines.append(msg)

    def _report(self, progress: dict) -> None:
        with self.lock:
            self.progress = progress

    def _run(self, work: Callable[..., dict]) -> None:
        try:
            with _model_lock:
                payload = work(self._log, self._report)
            with self.lock:
                self.state = "done"
                self.result = payload
                self.progress = {**self.progress, "percent": 100.0}
        except IndexMissing as e:
            self._fail(str(e))
        except ValueError as e:
            self._fail(str(e))
        except Exception as e:
            print(f"  {self.kind} failed: {e!r}")
            self._fail(f"{type(e).__name__}: {e}")
        finally:
            with self.lock:
                self.finished = time.time()

    def _fail(self, message: str) -> None:
        with self.lock:
            self.state = "error"
            self.error = message

    def snapshot(self) -> dict:
        with self.lock:
            elapsed = (self.finished or time.time()) - self.started if self.started else 0.0
            return {
                "state": self.state,
                "kind": self.kind,
                "video": self.video_id,
                "log": list(self.lines),
                "error": self.error,
                "result": self.result,
                "progress": dict(self.progress),
                "elapsed": elapsed,
                "models": _pool.resident(),
            }


_job = Job()

_MAX_BODY = 64 * 1024
_CHUNK = 1 << 20        # 1 MiB per write while streaming a file
_MAX_RANGE = 8 << 20    # cap one 206 response at 8 MiB
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def _fmt_hhmmss(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    return f"{h:02d}:{m:02d}:{t % 60:06.3f}"


def _media_url(video_id: str, abs_path: str | None) -> str | None:
    """Absolute path inside a video's data dir -> a /media/... URL."""
    if not abs_path:
        return None
    rel = Path(abs_path).relative_to(config.DATA_ROOT / video_id).as_posix()
    return f"/media/{video_id}/{rel}"


class Handler(BaseHTTPRequestHandler):
    server_version = "vidsearch"
    protocol_version = "HTTP/1.1"

    # ---- helpers ----------------------------------------------------------

    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, payload: dict | list) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _read_json_body(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > _MAX_BODY:
            self._send_json(400, {"error": "bad request body"})
            return None
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "invalid JSON"})
            return None

    def _busy(self) -> bool:
        """True (and 409 already sent) while a job holds the models.

        Checked before resolving the video, because a rebuild deletes the old index and the
        video would otherwise look like it had vanished (404) rather than being busy.
        """
        if _job.state == "running":
            self._send_json(409, {"error": f"{_job.kind} in progress — 진행 중인 작업이 끝난 뒤 다시 시도하세요"})
            return True
        return False

    def _resolve_video(self, video_id: str) -> str | None:
        """Id -> the source video path, for an indexed video or a workspace file.

        The id is only ever matched against the known-videos listing, never joined into a
        filesystem path, so nothing outside data/ and the scanned directory is reachable.
        """
        video = next((v for v in _videos() if v["id"] == video_id), None)
        if video is None:
            self._send_json(404, {"error": f"unknown video {video_id!r}"})
            return None
        return video["video_path"]

    def log_message(self, fmt: str, *args) -> None:
        print(f"  [{self.log_date_time_string()}] {fmt % args}")

    # ---- routes -----------------------------------------------------------

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/":
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/videos":
            self._send_json(200, {"workspace": str(_scan_dir), "videos": _videos()})
        elif path == "/api/job":
            self._send_json(200, _job.snapshot())
        elif path.startswith("/media/"):
            self._serve_media(path[len("/media/"):])
        elif path.startswith("/video/"):
            self._serve_source(path[len("/video/"):])
        else:
            self._send_json(404, {"error": "not found"})

    do_HEAD = do_GET

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/api/search":
            req = self._read_json_body()
            if req is not None:
                self._search(req)
        elif path == "/api/ask":
            req = self._read_json_body()
            if req is not None:
                self._ask(req)
        elif path == "/api/index":
            req = self._read_json_body()
            if req is not None:
                self._index(req)
        else:
            self._send_json(404, {"error": "not found"})

    # ---- handlers ---------------------------------------------------------

    def _submit(self, kind: str, video_id: str, work: Callable[..., dict]) -> None:
        if not _job.start(kind, video_id, work):
            self._send_json(409, {"error": "another job is already running"})
            return
        self._send_json(202, _job.snapshot())

    def _search(self, req: dict) -> None:
        if self._busy():
            return
        video_id = str(req.get("video") or "")
        query = str(req.get("query") or "").strip()
        if not query:
            self._send_json(400, {"error": "query is required"})
            return
        video_path = self._resolve_video(video_id)
        if video_path is None:
            return

        def work(log, report) -> dict:
            result = run_search(
                video_path,
                query,
                query_en=(req.get("query_en") or None),
                bilingual=bool(req.get("bilingual")),
                recall=int(req.get("recall") or config.DEFAULT_RECALL_TOP_M),
                gap=float(req.get("gap") or config.DEFAULT_SEGMENT_GAP_SEC),
                rerank_top=int(req.get("rerank_top") or config.DEFAULT_RERANK_TOP_SEGMENTS),
                no_rerank=bool(req.get("no_rerank")),
                no_motion=bool(req.get("no_motion")),
                top=int(req.get("top") or 10),
                log=log,
                report=report,
                pool=_pool,
            )
            return {
                "video_path": video_path,
                "timings": result.timings,
                "total_seconds": result.total_seconds,
                "results": [
                    {
                        "start": s.start,
                        "end": s.end,
                        "peak_t": s.peak_t,
                        "peak_hhmmss": _fmt_hhmmss(s.peak_t),
                        "score": s.max_score,
                        "mean_score": s.mean_score,
                        "frame_count": s.frame_count,
                        "thumb": _media_url(video_id, s.peak_thumb),
                        "frame": _media_url(video_id, s.peak_frame),
                        "clip": _media_url(video_id, s.peak_clip),
                    }
                    for s in result.segments
                ],
            }

        print(f"search: [{video_id}] {query!r}")
        self._submit("search", video_id, work)

    def _ask(self, req: dict) -> None:
        if self._busy():
            return
        video_id = str(req.get("video") or "")
        question = str(req.get("question") or "").strip()
        if not question:
            self._send_json(400, {"error": "question is required"})
            return
        video_path = self._resolve_video(video_id)
        if video_path is None:
            return

        def work(log, report) -> dict:
            result = run_ask(
                video_path,
                question,
                start=(req.get("start") or None),
                end=(req.get("end") or None),
                max_frames=int(req.get("max_frames") or config.DEFAULT_EXPLAIN_MAX_FRAMES),
                max_new_tokens=int(
                    req.get("max_new_tokens") or config.DEFAULT_EXPLAIN_MAX_NEW_TOKENS
                ),
                log=log,
                report=report,
                pool=_pool,
            )
            return {
                "video_path": video_path,
                "question": result.question,
                "answer": result.answer,
                "truncated": result.truncated,
                "start": result.start,
                "end": result.end,
                "timings": result.timings,
                "total_seconds": result.total_seconds,
                "frames": [
                    {
                        "t_sec": f["t_sec"],
                        "hhmmss": _fmt_hhmmss(f["t_sec"]),
                        "thumb": _media_url(video_id, f["thumb"]),
                        "frame": _media_url(video_id, f["frame"]),
                    }
                    for f in result.frames
                ],
            }

        print(f"ask: [{video_id}] {question!r}")
        self._submit("ask", video_id, work)

    def _index(self, req: dict) -> None:
        if self._busy():
            return
        video_id = str(req.get("video") or "")
        video_path = self._resolve_video(video_id)
        if video_path is None:
            return
        if not Path(video_path).is_file():
            self._send_json(400, {"error": f"source video is gone: {video_path}"})
            return

        try:
            opts = {
                "fps": float(req.get("fps") or config.DEFAULT_FPS),
                "width": int(req.get("width") or config.DEFAULT_FRAME_WIDTH),
                "batch_size": int(req.get("batch_size") or 4),
                "force": bool(req.get("force", True)),
                "with_motion": bool(req.get("with_motion")),
                "clip_window": float(req.get("clip_window") or config.DEFAULT_CLIP_WINDOW_SEC),
                "clip_stride": float(req.get("clip_stride") or config.DEFAULT_CLIP_STRIDE_SEC),
                "clip_width": int(req.get("clip_width") or config.DEFAULT_CLIP_WIDTH),
                "clip_batch_size": int(req.get("clip_batch_size") or config.DEFAULT_CLIP_BATCH_SIZE),
                "describe": bool(req.get("describe", True)),
            }
        except (TypeError, ValueError) as e:
            self._send_json(400, {"error": f"invalid option: {e}"})
            return
        if opts["fps"] <= 0 or opts["width"] < 64 or opts["clip_stride"] <= 0 or opts["clip_window"] <= 0:
            self._send_json(400, {"error": "fps/width/clip-window/clip-stride must be positive (width >= 64)"})
            return

        def work(log, report) -> dict:
            result = run_index(video_path, log=log, report=report, pool=_pool, **opts)
            return {
                "frames": result.frames,
                "clips": result.clips,
                "rebuilt_frames": result.rebuilt_frames,
                "rebuilt_clips": result.rebuilt_clips,
                "removed": result.removed,
                "description": result.description,
                "timings": result.timings,
                "total_seconds": result.total_seconds,
            }

        print(f"index: [{video_id}] {opts}")
        self._submit("index", video_id, work)

    def _serve_media(self, rel: str) -> None:
        target = (config.DATA_ROOT / rel).resolve()
        root = config.DATA_ROOT.resolve()
        # Reject traversal outside data/ ("..", absolute paths, symlinked escapes).
        if root not in target.parents or not target.is_file():
            self._send_json(404, {"error": "not found"})
            return
        self._serve_file(target, {"Cache-Control": "max-age=3600"})

    def _serve_source(self, video_id: str) -> None:
        """Stream the original video file for an indexed id.

        Only paths recorded in an index's own manifest are reachable — the id is matched
        against the data/ directory listing, never joined into a filesystem path.
        """
        video = next((v for v in _videos() if v["id"] == video_id), None)
        if video is None:
            self._send_json(404, {"error": f"no index for {video_id!r}"})
            return
        source = Path(video["video_path"])
        if not source.is_file():
            self._send_json(404, {"error": f"source video is gone: {source}"})
            return
        self._serve_file(source)

    def _serve_file(self, target: Path, extra: dict | None = None) -> None:
        """Send a file with HTTP Range support, streamed so a multi-GB video never has to
        fit in memory. A range request is answered with at most _MAX_RANGE bytes, which is
        allowed (a 206 may be shorter than asked) and keeps `bytes=0-` from pulling the
        whole file at once."""
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        size = target.stat().st_size
        rng = _RANGE_RE.match(self.headers.get("Range") or "")
        if rng:
            start = int(rng.group(1)) if rng.group(1) else 0
            end = int(rng.group(2)) if rng.group(2) else size - 1
            end = min(end, size - 1, start + _MAX_RANGE - 1)
            if start > end or start >= size:
                self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}"})
                return
            self._stream(206, target, start, end - start + 1, ctype,
                         {"Content-Range": f"bytes {start}-{end}/{size}", **(extra or {})})
        else:
            self._stream(200, target, 0, size, ctype, extra)

    def _stream(self, status: int, path: Path, start: int, length: int,
                ctype: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command == "HEAD":
            return
        try:
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(_CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionError):
            # Normal when the viewer seeks or closes the player mid-transfer.
            self.close_connection = True


INDEX_HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vidsearch</title>
<style>
  :root {
    /* Three separated surfaces — page, raised panel, sunken field — so every box has an
       edge you can see without hunting for it; borders are a step lighter again. */
    --bg: #0b0d12; --panel: #1b202a; --field: #11151c;
    --line: #3b4453; --line-soft: #2a3140; --line-strong: #56617a;
    --fg: #e9edf4; --muted: #9aa4b6; --accent: #6aa9ff; --accent2: #7ad4a8;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg);
         font: 14px/1.5 "Segoe UI", system-ui, sans-serif; }
  header { padding: 16px 24px; background: var(--panel);
           border-bottom: 1px solid var(--line-strong);
           box-shadow: 0 2px 10px rgba(0,0,0,.45); }
  h1 { margin: 0 0 12px; font-size: 17px; letter-spacing: .04em; }
  h1 span { color: var(--muted); font-weight: 400; margin-left: 8px; font-size: 13px; }
  select, input[type=text], input[type=number], textarea {
    background: var(--field); color: var(--fg); border: 1px solid var(--line);
    border-radius: 6px; padding: 8px 10px; font: inherit;
  }
  button { background: #262d3a; color: var(--fg); border: 1px solid var(--line);
           border-radius: 6px; padding: 8px 10px; font: inherit; }
  select:hover, input:hover, button:hover:not(:disabled) { border-color: var(--line-strong); }
  select:focus, input:focus, textarea:focus, button:focus-visible {
    outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(106,169,255,.25);
  }
  input[type=checkbox] { accent-color: var(--accent); width: 15px; height: 15px; }
  textarea { width: 100%; resize: vertical; min-height: 60px; }
  form { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  form.off { display: none; }
  input[type=text].grow { flex: 1 1 320px; min-width: 220px; }
  input[type=text].tc { width: 110px; }
  input[type=number] { width: 74px; }
  button.go { background: var(--accent); color: #06101f; border-color: var(--accent);
              font-weight: 600; cursor: pointer; padding: 8px 18px; }
  button:disabled { opacity: .45; cursor: default; }
  .opts { display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
          color: var(--muted); font-size: 13px; width: 100%; }
  .opts label { display: flex; gap: 5px; align-items: center; cursor: pointer; }
  .modes { display: flex; gap: 6px; margin-bottom: 12px; }
  .modes button { cursor: pointer; padding: 5px 16px; }
  .modes button.on { background: var(--accent); color: #06101f; border-color: var(--accent);
                     font-weight: 600; }
  .modes button.play { color: var(--accent2); }
  form.panel { display: block; border: 1px solid var(--line); border-radius: 8px;
               padding: 12px 14px; margin-top: 12px; background: var(--bg); }
  form.panel.off { display: none; }
  form.panel .opts + .opts { margin-top: 12px; padding-top: 12px;
                             border-top: 1px solid var(--line-soft); }
  .warn { color: #f0c674 !important; }
  #ihint { color: var(--muted); font-size: 12px; }
  #ilog { margin: 12px 0 0; padding: 10px 12px; max-height: 220px; overflow: auto;
          background: var(--bg); border: 1px solid var(--line); border-radius: 8px;
          font: 12px/1.55 Consolas, monospace; color: var(--muted); white-space: pre-wrap; }
  #ilog.off { display: none; }
  main { padding: 20px 24px 60px; }
  #status { color: var(--muted); margin-bottom: 12px; white-space: pre-wrap; }
  #status.err { color: #ff8981; }
  #timing { margin-bottom: 18px; }
  #timing.off { display: none; }
  #prog { margin-bottom: 18px; }
  #prog.off { display: none; }
  .pbar { height: 12px; border-radius: 6px; border: 1px solid var(--line);
          background: var(--field); overflow: hidden; }
  .pbar div { height: 100%; background: linear-gradient(90deg, var(--accent2), var(--accent));
              transition: width .35s ease; }
  .pline { display: flex; align-items: baseline; gap: 10px; margin-top: 7px;
           color: var(--muted); font-size: 13px; font-variant-numeric: tabular-nums; }
  .pline b { color: var(--fg); font-size: 15px; min-width: 3.2em; }
  .pline .el { margin-left: auto; }
  .bar { display: flex; height: 9px; border-radius: 5px; overflow: hidden;
         border: 1px solid var(--line); background: var(--field); margin-bottom: 10px; }
  .bar div { height: 100%; }
  .pills { display: flex; flex-wrap: wrap; gap: 6px; font-size: 12px;
           font-variant-numeric: tabular-nums; }
  .pill { display: flex; align-items: center; gap: 6px; background: var(--panel);
          border: 1px solid var(--line); border-radius: 20px; padding: 3px 10px;
          color: var(--muted); }
  .pill i { width: 8px; height: 8px; border-radius: 50%; font-style: normal; }
  .pill b { color: var(--fg); font-weight: 600; }
  .pill.total { border-color: var(--accent); color: var(--fg); }
  #answer { background: var(--panel); border: 1px solid var(--line);
            border-left: 4px solid var(--accent2);
            border-radius: 8px; padding: 14px 16px; margin-bottom: 18px; white-space: pre-wrap; }
  #answer.off { display: none; }
  /* A clipped answer looks exactly like a finished one, so mark it in the answer box too. */
  #answer.cut { border-left-color: #f0c674; margin-bottom: 8px; }
  #cut { margin-bottom: 18px; padding: 8px 12px; border-radius: 8px; font-size: 13px;
         color: #f0c674; border: 1px solid rgba(240,198,116,.45);
         background: rgba(240,198,116,.10); }
  #cut.off { display: none; }
  #gridlabel { color: var(--muted); font-size: 13px; margin-bottom: 8px; }
  #grid { display: grid; gap: 16px;
          grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
          overflow: hidden; cursor: pointer; box-shadow: 0 1px 6px rgba(0,0,0,.35); }
  .card:hover { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
  .card img { display: block; width: 100%; aspect-ratio: 16/9; object-fit: cover;
              background: #000; }
  .meta { display: flex; justify-content: space-between; gap: 8px;
          padding: 8px 10px; font-variant-numeric: tabular-nums;
          border-top: 1px solid var(--line-soft); }
  .t { font-weight: 600; }
  .s { color: var(--muted); }
  .thumbwrap { position: relative; }
  .badge::after { content: "▶ clip"; position: absolute; right: 6px; top: 6px;
                  background: rgba(0,0,0,.66); color: #fff; font-size: 11px;
                  padding: 2px 6px; border-radius: 4px; }
  .rank { position: absolute; left: 6px; top: 6px; background: rgba(0,0,0,.66);
          color: #fff; font-size: 11px; padding: 2px 6px; border-radius: 4px; }
  #lb { position: fixed; inset: 0; background: rgba(4,6,10,.95); display: none;
        align-items: center; justify-content: center; flex-direction: column;
        gap: 12px; padding: 24px; z-index: 10; }
  #lb.on { display: flex; }
  #lb img, #lb video { max-width: min(1100px, 92vw); max-height: 76vh;
                       border-radius: 8px; background: #000;
                       border: 1px solid var(--line); box-shadow: 0 8px 40px rgba(0,0,0,.7); }
  #lbinfo { color: var(--muted); font-variant-numeric: tabular-nums; text-align: center; }
  #lbinfo code { color: var(--fg); background: var(--panel); padding: 3px 8px;
                 border-radius: 4px; user-select: all; }
  #lbtabs { display: flex; gap: 8px; }
  #lbtabs button { cursor: pointer; padding: 5px 12px; }
  #lbtabs button.on { background: var(--accent); color: #06101f; border-color: var(--accent); }
</style>
</head>
<body>
<header>
  <h1>vidsearch <span id="sub">indexed videos</span></h1>
  <div class="modes">
    <button id="m-search" class="on">search</button>
    <button id="m-ask">ask</button>
    <select id="video" style="margin-left:8px"></select>
    <button id="playsrc" class="play" title="선택한 원본 영상 재생">▶ play</button>
    <button id="m-index" title="인덱스를 지우고 옵션을 바꿔 다시 생성">⚙ index</button>
  </div>

  <form id="f-search">
    <input type="text" id="q" class="grow" placeholder="찾을 장면을 자연어로 (예: 사람이 뛰는 장면)" autofocus>
    <button class="go" type="submit">search</button>
    <div class="opts">
      <label>top <input type="number" id="top" value="10" min="1" max="60"></label>
      <label><input type="checkbox" id="norerank"> no-rerank</label>
      <label><input type="checkbox" id="nomotion"> no-motion</label>
      <label>en <input type="text" id="qen" placeholder="영문 쿼리(선택)" style="width:180px"></label>
    </div>
  </form>

  <form id="f-ask" class="off">
    <input type="text" id="question" class="grow" placeholder="영상에 대해 물어보기 (예: 사람이 최대 몇 명 보여?)">
    <button class="go" type="submit">ask</button>
    <div class="opts">
      <label>start <input type="text" id="astart" class="tc" placeholder="00:03:10"></label>
      <label>end <input type="text" id="aend" class="tc" placeholder="00:03:30"></label>
      <label>max-frames <input type="number" id="amax" value="6" min="1" max="24"></label>
      <label>max-tokens <input type="number" id="atok" value="512" min="64" max="2048" step="64"></label>
      <span>범위를 비우면 영상 전체를 대상으로 질문합니다.</span>
    </div>
  </form>

  <form id="f-index" class="off panel">
    <div class="opts">
      <label>fps <input type="number" id="ifps" step="0.1" min="0.1" style="width:70px"></label>
      <label>width <input type="number" id="iwidth" step="32" min="64" style="width:80px"></label>
      <label>batch <input type="number" id="ibatch" min="1" max="32" value="4" style="width:60px"></label>
      <label><input type="checkbox" id="imotion"> with-motion</label>
      <label title="4B 모델로 영상 요약을 만들어 목록 툴팁에 씁니다"><input type="checkbox" id="idesc" checked> describe</label>
      <label>clip-window <input type="number" id="icw" step="0.5" min="0.5" style="width:70px"></label>
      <label>clip-stride <input type="number" id="ics" step="0.5" min="0.5" style="width:70px"></label>
      <label>clip-width <input type="number" id="icwidth" step="32" min="64" style="width:80px"></label>
      <label>clip-batch <input type="number" id="icbatch" min="1" max="16" value="2" style="width:60px"></label>
    </div>
    <div class="opts">
      <label class="warn"><input type="checkbox" id="iforce" checked> 기존 인덱스 삭제 후 재생성</label>
      <button class="go" type="submit">start indexing</button>
      <span id="ihint"></span>
    </div>
  </form>
  <pre id="ilog" class="off"></pre>
</header>
<main>
  <div id="status">인덱싱된 영상을 고르고 검색어 또는 질문을 입력하세요.</div>
  <div id="prog" class="off"></div>
  <div id="timing" class="off"></div>
  <div id="answer" class="off"></div>
  <div id="cut" class="off"></div>
  <div id="gridlabel"></div>
  <div id="grid"></div>
</main>
<div id="lb">
  <div id="lbtabs"></div>
  <div id="lbmedia"></div>
  <div id="lbinfo"></div>
</div>
<script>
const $ = id => document.getElementById(id);
const COLORS = ['#6aa9ff','#7ad4a8','#f0c674','#c792ea','#f28b82','#5fd4d6','#a8b1c2','#ff9e64'];
let videoPath = '', items = [], mode = 'search', videos = [];
let indexing = false, running = false;   // running: any job (index/search/ask)

async function loadVideos(keepId) {
  const want = keepId || $('video').value;
  const data = await (await fetch('/api/videos')).json();
  videos = data.videos;
  $('video').innerHTML = videos.map(v =>
    `<option value="${v.id}" title="${esc(tip(v))}">${v.name} · ${v.indexed
       ? `${v.duration.toFixed(0)}s${v.has_motion ? ' · motion' : ''}`
       : `인덱스 없음 · ${mb(v.size)}`}</option>`
  ).join('');
  if (want && videos.some(v => v.id === want)) $('video').value = want;
  const n = videos.filter(v => v.indexed).length;
  $('sub').textContent = videos.length
    ? `${n} indexed · ${videos.length - n} not indexed · ${data.workspace}`
    : `${data.workspace} 에 영상이 없습니다 — --videos <폴더> 로 영상 폴더를 지정하세요`;
  syncUI();
}

const selected = () => videos.find(v => v.id === $('video').value);
const mb = b => (b / 1e6 < 10 ? (b / 1e6).toFixed(1) : (b / 1e6).toFixed(0)) + 'MB';
const esc = s => s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');

// Tooltip for one entry in the file list: the summary written at index time, plus the facts.
function tip(v) {
  const head = v.indexed
    ? `${v.name} · ${v.duration.toFixed(0)}s @ ${v.index_fps}fps / ${v.width}px${v.has_motion ? ' + motion' : ''}`
    : `${v.name} · ${mb(v.size)} · 아직 인덱싱되지 않음`;
  const body = v.description
    ? v.description
    : v.indexed ? '(설명 없음 — describe 옵션으로 다시 인덱싱하면 생성됩니다)' : '';
  return [head, v.video_path, body].filter(Boolean).join('\n');
}

// Every button's enabled state depends on the same three facts: is a job running, is the
// selected video indexed, and does its source file still exist. One place decides.
function syncUI() {
  const v = selected();
  const ready = !!v && v.indexed && !running;
  $('video').title = v ? tip(v) : '';   // hovering the closed select shows the same summary
  $('f-search').querySelector('button.go').disabled = !ready;
  $('f-ask').querySelector('button.go').disabled = !ready;
  $('playsrc').disabled = !v || !v.has_source || running;
  $('playsrc').title = !v ? '영상 없음'
    : v.has_source ? `원본 재생: ${v.video_path}`
    : `원본 파일이 없습니다: ${v.video_path} (인덱스만 남아 있음)`;
  fillIndexForm();
  if (v && !v.indexed && !running) {
    $('f-index').classList.remove('off');   // nothing else can be done with it yet
    $('status').className = '';
    $('status').textContent = `${v.name} 은(는) 아직 인덱싱되지 않았습니다 — 아래 옵션으로 인덱싱하면 검색/질문할 수 있습니다.`;
  }
}
$('video').onchange = syncUI;

// ---- indexing ---------------------------------------------------------------
function fillIndexForm() {
  const v = selected();
  if (!v) return;
  $('ifps').value = v.index_fps;
  $('iwidth').value = v.width;
  $('imotion').checked = v.has_motion;
  $('icw').value = v.clip_window_sec;
  $('ics').value = v.clip_stride_sec;
  $('icwidth').value = v.clip_width;
  const can = v.has_source && !running;
  $('m-index').disabled = !v.has_source || running;
  $('f-index').querySelectorAll('input,button').forEach(el => el.disabled = !can);
  // Nothing to delete for a video that has never been indexed.
  $('iforce').disabled = !can || !v.indexed;
  $('iforce').checked = v.indexed;
  $('ihint').textContent = !v.has_source
    ? '원본 파일이 없어 다시 인덱싱할 수 없습니다.'
    : v.indexed
      ? `대상: ${v.video_path} · 현재 ${v.duration.toFixed(0)}s @ ${v.index_fps}fps${v.has_motion ? ' + motion' : ''}`
      : `대상: ${v.video_path} · 새로 인덱싱합니다 (${mb(v.size)})`;
}
$('m-index').onclick = () => $('f-index').classList.toggle('off');

$('f-index').addEventListener('submit', async e => {
  e.preventDefault();
  const v = selected();
  if (!v) return;
  if ($('iforce').checked &&
      !confirm(`${v.name}의 기존 인덱스(프레임/썸네일${$('imotion').checked ? '/클립' : ''})를 삭제하고 다시 만듭니다.\n계속할까요?`))
    return;
  await submit('/api/index', {
    video: v.id,
    fps: +$('ifps').value, width: +$('iwidth').value, batch_size: +$('ibatch').value,
    force: $('iforce').checked, with_motion: $('imotion').checked,
    describe: $('idesc').checked,
    clip_window: +$('icw').value, clip_stride: +$('ics').value,
    clip_width: +$('icwidth').value, clip_batch_size: +$('icbatch').value,
  });
});

// ---- one job at a time: submit, then poll for phase/percent -----------------
const KIND_KO = {index: '인덱싱', search: '검색', ask: '질문'};
let polling = null;

async function submit(url, body) {
  $('grid').innerHTML = ''; $('gridlabel').textContent = '';
  $('answer').className = 'off'; $('cut').className = 'off'; $('timing').className = 'off';
  try {
    await post(url, body);
    poll();
  } catch (err) { fail(err); }
}

function renderProgress(s) {
  const p = s.progress || {};
  const pct = p.percent || 0;
  const items = p.count ? ` ${p.done}/${p.count}` : '';
  $('prog').className = '';
  $('prog').innerHTML =
    `<div class="pbar"><div style="width:${pct.toFixed(1)}%"></div></div>
     <div class="pline"><b>${pct.toFixed(0)}%</b>
       <span>${KIND_KO[s.kind] || s.kind} 중 — ${p.phase || '준비'}${items}</span>
       <span class="el">${s.elapsed.toFixed(0)}s 경과</span></div>`;
}

async function poll() {
  const s = await (await fetch('/api/job')).json();
  running = s.state === 'running';
  const isIndex = s.kind === 'index';
  indexing = running && isIndex;
  // The index log pane stays for indexing only; search/ask keep the header compact.
  $('ilog').className = (isIndex && s.state !== 'idle') ? '' : 'off';
  if (isIndex) {
    $('ilog').textContent = s.log.join('\n') + (s.error ? '\n' + s.error : '');
    $('ilog').scrollTop = $('ilog').scrollHeight;
  }

  if (running) {
    renderProgress(s);
    syncUI();
    $('status').className = '';
    $('status').textContent = `[${s.video}] ${KIND_KO[s.kind] || s.kind} 진행 중 — 다른 작업은 끝난 뒤에 실행됩니다.`;
    if (!polling) polling = setInterval(poll, 700);
    return;
  }
  clearInterval(polling); polling = null;
  $('prog').className = 'off';

  if (s.state === 'error') {
    syncUI();
    $('status').className = 'err';
    $('status').textContent = `${KIND_KO[s.kind] || s.kind} 실패: ${s.error}`;
    return;
  }
  if (s.state === 'done') {
    if (s.kind === 'index') await showIndexResult(s);
    else if (s.kind === 'search') showSearchResult(s.result, s.elapsed);
    else if (s.kind === 'ask') showAskResult(s.result);
  }
  syncUI();
}

async function showIndexResult(s) {
  const r = s.result;
  await loadVideos(s.video);   // the video just became (re)indexed — keep it selected
  $('status').className = '';
  $('status').textContent =
    `인덱싱 완료 — 프레임 ${r.frames}개${r.clips ? `, 클립 ${r.clips}개` : ''} · ${r.total_seconds.toFixed(1)}s` +
    (r.description ? `\n설명: ${r.description}` : '');
  renderTimings(r.timings, r.total_seconds);
}

function showSearchResult(data) {
  videoPath = data.video_path;
  items = data.results;
  renderTimings(data.timings, data.total_seconds);
  $('status').className = '';
  $('status').textContent = items.length
    ? `${items.length} result(s) — 썸네일을 클릭하면 원본 프레임/클립을 볼 수 있습니다.`
    : '결과 없음.';
  renderCards(items.map((r, i) => ({
    thumb: r.thumb, badge: !!r.clip, rank: `#${i + 1}`,
    left: r.peak_hhmmss, right: r.score.toFixed(4),
  })));
}

function showAskResult(data) {
  videoPath = data.video_path;
  items = data.frames;
  renderTimings(data.timings, data.total_seconds);
  $('answer').className = data.truncated ? 'cut' : '';
  $('answer').textContent = data.answer;
  $('cut').className = data.truncated ? '' : 'off';
  $('cut').textContent = data.truncated
    ? `답변이 max-tokens ${$('atok').value} 제한에 걸려 중간에 끊겼습니다 — max-tokens 를 올려 다시 물어보세요.`
    : '';
  $('status').className = '';
  $('status').textContent =
    `${data.start.toFixed(1)}s – ${data.end.toFixed(1)}s 구간에서 ${data.frames.length}개 프레임을 근거로 답했습니다.`;
  $('gridlabel').textContent = '모델에 넣은 프레임 (클릭하면 원본 크기)';
  renderCards(items.map((f, i) => ({
    thumb: f.thumb, badge: false, rank: `#${i + 1}`, left: f.hhmmss, right: '',
  })));
}

function playSource(t, i) {
  const v = selected();
  if (!v || !v.has_source) return;
  // #t= makes the browser seek on load, so a result can be opened at its own timecode.
  const src = `/video/${encodeURIComponent(v.id)}` + (t ? `#t=${t.toFixed(2)}` : '');
  showLb(`<video src="${src}" controls autoplay></video>`,
         i == null ? '' : tabsFor(i, 'source'),
         `${v.name}${t ? ' · ' + hhmmss(t) + '부터' : ''}<br><code>${v.video_path}</code>`);
  if (i != null) bindTabs(i);
}
$('playsrc').onclick = () => playSource(0);

function hhmmss(t) {
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${s.toFixed(3).padStart(6,'0')}`;
}

function setMode(m) {
  mode = m;
  $('m-search').classList.toggle('on', m === 'search');
  $('m-ask').classList.toggle('on', m === 'ask');
  $('f-search').classList.toggle('off', m !== 'search');
  $('f-ask').classList.toggle('off', m !== 'ask');
  ($('q') && m === 'search' ? $('q') : $('question')).focus();
}
$('m-search').onclick = () => setMode('search');
$('m-ask').onclick = () => setMode('ask');

function fail(err) {
  $('status').className = 'err';
  $('status').textContent = '실패: ' + err.message;
}

function renderTimings(timings, total) {
  if (!timings || !timings.length) { $('timing').className = 'off'; return; }
  $('timing').className = '';
  $('timing').innerHTML =
    `<div class="bar">${timings.map((t, i) =>
      `<div style="width:${(t.seconds / total * 100).toFixed(2)}%;background:${COLORS[i % COLORS.length]}"
            title="${t.name} ${t.seconds.toFixed(2)}s"></div>`).join('')}</div>
     <div class="pills">${timings.map((t, i) =>
      `<span class="pill"><i style="background:${COLORS[i % COLORS.length]}"></i>${t.name}
         <b>${t.seconds.toFixed(2)}s</b></span>`).join('')}
      <span class="pill total">total <b>${total.toFixed(2)}s</b></span></div>`;
}

// ---- search / ask ------------------------------------------------------------
$('f-search').addEventListener('submit', e => {
  e.preventDefault();
  const query = $('q').value.trim();
  if (!query) return;
  submit('/api/search', {
    video: $('video').value, query,
    query_en: $('qen').value.trim() || null,
    bilingual: !!$('qen').value.trim(),
    top: +$('top').value,
    no_rerank: $('norerank').checked,
    no_motion: $('nomotion').checked,
  });
});

$('f-ask').addEventListener('submit', e => {
  e.preventDefault();
  const question = $('question').value.trim();
  if (!question) return;
  submit('/api/ask', {
    video: $('video').value, question,
    start: $('astart').value.trim() || null,
    end: $('aend').value.trim() || null,
    max_frames: +$('amax').value,
    max_new_tokens: +$('atok').value,
  });
});

async function post(url, body) {
  const res = await fetch(url, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

// ---- rendering -------------------------------------------------------------
function renderCards(cards) {
  $('grid').innerHTML = cards.map((c, i) => `
    <div class="card" data-i="${i}">
      <div class="thumbwrap${c.badge ? ' badge' : ''}">
        <img src="${c.thumb}" loading="lazy" alt="">
        <span class="rank">${c.rank}</span>
      </div>
      <div class="meta"><span class="t">${c.left}</span><span class="s">${c.right}</span></div>
    </div>`).join('');
  $('grid').querySelectorAll('.card').forEach(c => {
    const i = +c.dataset.i;
    c.onclick = () => openLb(i, items[i].clip ? 'clip' : 'frame');
  });
}

function showLb(media, tabs, info) {
  $('lbmedia').innerHTML = media;
  $('lbtabs').innerHTML = tabs;
  $('lbinfo').innerHTML = info;
  $('lb').classList.add('on');
}

function closeLb() {
  $('lb').classList.remove('on');
  $('lbmedia').innerHTML = '';  // stop playback — the source video has audio
}

const peakT = it => it.peak_t !== undefined ? it.peak_t : it.t_sec;

function tabsFor(i, view) {
  const it = items[i], v = selected();
  const tab = (id, label) => `<button class="${view === id ? 'on' : ''}" data-v="${id}">${label}</button>`;
  return tab('frame', 'frame')
    + (it.clip ? tab('clip', 'clip') : '')
    + (v && v.has_source ? tab('source', '▶ source') : '');
}

function bindTabs(i) {
  $('lbtabs').querySelectorAll('button').forEach(b =>
    b.onclick = ev => { ev.stopPropagation(); openLb(i, b.dataset.v); });
}

function openLb(i, view) {
  const it = items[i], t = peakT(it);
  if (view === 'source') { playSource(t, i); return; }
  showLb(
    view === 'clip' && it.clip
      ? `<video src="${it.clip}" controls autoplay loop></video>`
      : `<img src="${it.frame}" alt="">`,
    tabsFor(i, view),
    `#${i + 1} · ${it.peak_hhmmss || it.hhmmss}` +
      (it.score !== undefined
        ? ` · score ${it.score.toFixed(4)} · segment ${it.start.toFixed(1)}–${it.end.toFixed(1)}s (${it.frame_count} frames)`
        : '') +
      `<br><code>ffplay -ss ${t.toFixed(2)} "${videoPath}"</code>`);
  bindTabs(i);
}

$('lb').onclick = e => { if (e.target.id === 'lb' || e.target.id === 'lbmedia') closeLb(); };
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLb(); });
loadVideos();
poll();  // a reload during a long run rejoins the job already in flight
</script>
</body>
</html>
"""


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = False,
    videos_dir: str | Path | None = None,
    model_idle: float = 600.0,
) -> None:
    global _scan_dir
    _scan_dir = Path(videos_dir).resolve() if videos_dir else Path.cwd()
    IdleEvictor(_pool, model_idle, _model_lock).start()

    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"vidsearch web UI on {url}  (Ctrl+C to stop)")
    print(f"  data root:  {config.DATA_ROOT}")
    print(f"  workspace:  {_scan_dir}")
    print(f"  models:     kept loaded between jobs"
          + (f", released after {model_idle:.0f}s idle" if model_idle > 0 else ", never released"))
    for v in _videos():
        if v["indexed"]:
            print(f"  - {v['id']}  {v['duration']:.0f}s{'  +motion' if v['has_motion'] else ''}")
        else:
            print(f"  - {v['id']}  (not indexed, {v['size'] / 1e6:.0f}MB)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping ...")
    finally:
        httpd.server_close()
