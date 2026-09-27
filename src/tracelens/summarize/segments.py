"""Turn a tracked target into clip segments.

select_track_by_point: the "click on the person" interaction -- which
  confirmed track's box contains (x, y) in a given frame.
segments_from_frames: the frames where the target was tracked become clip
  ranges. Short gaps are merged (a detector blinking for 2 frames shouldn't
  split the clip), tiny fragments dropped, and each segment padded with a
  little context before and after.

Limitation, deliberately measured rather than hidden: a summary follows ONE
track ID. If the target leaves view for longer than the tracker's max_age,
it returns under a NEW id and this summary misses the return. See
eval/summary.py and the patrol scene.
"""

from __future__ import annotations

import numpy as np

from tracelens.track.tracker import TrackOutput


def select_track_by_point(tracks: list[TrackOutput], frame: int, point: tuple[float, float]) -> int | None:
    """ID of the track whose box contains the point in that frame. If several
    boxes contain it, the smallest one (the most specific object) wins."""
    x, y = point
    hits = [t for t in tracks if t.frame == frame
            and t.box[0] <= x <= t.box[2] and t.box[1] <= y <= t.box[3]]
    if not hits:
        return None
    return min(hits, key=lambda t: (t.box[2] - t.box[0]) * (t.box[3] - t.box[1])).track_id


def appearance_frames(tracks: list[TrackOutput], target_id: int) -> list[int]:
    return sorted({t.frame for t in tracks if t.track_id == target_id})


def segments_from_frames(
    frames: list[int], n_frames: int, max_gap: int = 5, min_len: int = 3, pad: int = 5
) -> list[tuple[int, int]]:
    """Inclusive (start, end) clip ranges.

    max_gap: merge two runs if at most this many frames separate them.
    min_len: drop runs shorter than this (after merging), before padding.
    pad:     extend each segment by this many frames on both sides for
             context, clipped to the video, re-merging any that now overlap.
    """
    if not frames:
        return []
    runs = []
    start = prev = frames[0]
    for f in frames[1:]:
        if f - prev - 1 > max_gap:
            runs.append((start, prev))
            start = f
        prev = f
    runs.append((start, prev))
    runs = [(a, b) for a, b in runs if b - a + 1 >= min_len]

    padded = [(max(0, a - pad), min(n_frames - 1, b + pad)) for a, b in runs]
    merged: list[tuple[int, int]] = []
    for a, b in padded:
        if merged and a <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def frames_in(segments: list[tuple[int, int]]) -> np.ndarray:
    return np.array([f for a, b in segments for f in range(a, b + 1)], dtype=int)
