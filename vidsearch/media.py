import json
import re
import subprocess
from pathlib import Path

from . import config


def probe_video(video_path: str) -> dict:
    """Return {width, height, fps, duration, nb_frames} for a video file."""
    cmd = [
        config.FFPROBE_EXE, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    out = subprocess.run(cmd, capture_output=True, check=True)
    data = json.loads(out.stdout.decode("utf-8", errors="replace"))
    stream = data["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    duration = float(data["format"]["duration"])
    nb_frames = int(stream.get("nb_frames") or round(duration * fps))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "duration": duration,
        "nb_frames": nb_frames,
    }


def _run_ffmpeg_extract(video_path: str, out_dir: Path, vf: str, quality: int, prefix: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"{prefix}_%05d.jpg"
    cmd = [
        config.FFMPEG_EXE, "-y", "-v", "error",
        "-i", video_path,
        "-vf", vf,
        "-q:v", str(quality),
        str(pattern),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return sorted(out_dir.glob(f"{prefix}_*.jpg"))


def extract_frames(
    video_path: str,
    out_dir: Path,
    fps: float = config.DEFAULT_FPS,
    width: int = config.DEFAULT_FRAME_WIDTH,
) -> list[Path]:
    """Extract indexing frames at fixed fps, scaled to `width` (height auto, even)."""
    vf = f"fps={fps},scale={width}:-2"
    return _run_ffmpeg_extract(video_path, out_dir, vf, config.DEFAULT_JPEG_QUALITY, "f")


def extract_thumbs(
    video_path: str,
    out_dir: Path,
    fps: float = config.DEFAULT_FPS,
    width: int = config.DEFAULT_THUMB_WIDTH,
) -> list[Path]:
    """Extract display thumbnails at the same fps/index alignment as extract_frames."""
    vf = f"fps={fps},scale={width}:-2"
    return _run_ffmpeg_extract(video_path, out_dir, vf, 4, "t")


_PTS_RE = re.compile(r"pts_time:([0-9.]+)")


def extract_scene_boundaries(
    video_path: str,
    threshold: float = config.DEFAULT_SCENE_THRESHOLD,
) -> list[float]:
    """Return sorted scene-change timestamps (seconds) via ffmpeg scdet/select."""
    cmd = [
        config.FFMPEG_EXE, "-v", "info",
        "-i", video_path,
        "-an",
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True)
    stderr = out.stderr.decode("utf-8", errors="replace")
    times = sorted({float(m) for m in _PTS_RE.findall(stderr)})
    return times


def frame_timestamp(index: int, fps: float) -> float:
    """1-based ffmpeg frame index -> seconds, matching fps filter's equal spacing."""
    return (index - 1) / fps


def parse_timecode(s: str) -> float:
    """Accept either plain seconds ("203", "202.5") or "HH:MM:SS(.ms)" / "MM:SS" and
    return seconds, so `ask --start`/`--end` can take the same timecode format the
    `search` output prints."""
    if ":" not in s:
        return float(s)
    parts = [float(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, sec = parts
    return h * 3600 + m * 60 + sec


def extract_clips(
    video_path: str,
    out_dir: Path,
    duration: float,
    window_sec: float = config.DEFAULT_CLIP_WINDOW_SEC,
    stride_sec: float = config.DEFAULT_CLIP_STRIDE_SEC,
    width: int = config.DEFAULT_CLIP_WIDTH,
) -> list[dict]:
    """Cut overlapping short mp4 clips covering the video, for motion/action queries.

    Returns a list of {"idx", "start_t", "end_t", "mid_t", "clip"} dicts, one per clip
    actually written (a too-short trailing window is skipped, see MIN_CLIP_TAIL_SEC).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = []
    start = 0.0
    idx = 1
    while start < duration:
        end = min(start + window_sec, duration)
        if end - start >= min(config.MIN_CLIP_TAIL_SEC, window_sec / 2):
            windows.append((idx, start, end))
            idx += 1
        start += stride_sec

    clips = []
    for idx, start, end in windows:
        clip_path = out_dir / f"c_{idx:05d}.mp4"
        cmd = [
            config.FFMPEG_EXE, "-y", "-v", "error",
            "-ss", f"{start:.3f}",
            "-i", video_path,
            "-t", f"{end - start:.3f}",
            "-vf", f"scale={width}:-2",
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            str(clip_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        clips.append({"idx": idx, "start_t": start, "end_t": end, "mid_t": (start + end) / 2, "clip": clip_path})
    return clips
