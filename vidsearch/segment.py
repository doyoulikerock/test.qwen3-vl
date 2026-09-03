from dataclasses import dataclass


@dataclass
class ScoredFrame:
    idx: int
    t_sec: float
    score: float
    frame_path: str
    thumb_path: str
    clip_path: str | None = None  # set when this candidate came from the motion/clip channel (§8)


@dataclass
class Segment:
    start: float
    end: float
    peak_t: float
    peak_frame: str
    peak_thumb: str
    max_score: float
    mean_score: float
    frame_count: int
    peak_clip: str | None = None  # the clip whose window covers peak_t, if the peak came from the clip channel


def merge_segments(
    frames: list[ScoredFrame],
    gap_sec: float,
    scene_boundaries: list[float],
) -> list[Segment]:
    """Cluster time-sorted scored frames into segments.

    A new segment starts when the gap to the previous frame exceeds gap_sec,
    or when a scene-change boundary falls strictly between the two frames.
    """
    if not frames:
        return []

    ordered = sorted(frames, key=lambda f: f.t_sec)
    boundaries = sorted(scene_boundaries)

    clusters: list[list[ScoredFrame]] = [[ordered[0]]]
    for prev, cur in zip(ordered, ordered[1:]):
        gap = cur.t_sec - prev.t_sec
        crosses_boundary = any(prev.t_sec < b < cur.t_sec for b in boundaries)
        if gap > gap_sec or crosses_boundary:
            clusters.append([cur])
        else:
            clusters[-1].append(cur)

    segments = []
    for cluster in clusters:
        peak = max(cluster, key=lambda f: f.score)
        scores = [f.score for f in cluster]
        segments.append(
            Segment(
                start=cluster[0].t_sec,
                end=cluster[-1].t_sec,
                peak_t=peak.t_sec,
                peak_frame=peak.frame_path,
                peak_thumb=peak.thumb_path,
                max_score=peak.score,
                mean_score=sum(scores) / len(scores),
                frame_count=len(cluster),
                peak_clip=peak.clip_path,
            )
        )
    return segments
