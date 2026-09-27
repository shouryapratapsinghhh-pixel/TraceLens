"""Structured timeline of a target's appearances -- facts only.

Every field is measured from the tracker's boxes: times, which part of the
frame the target was in, whether it entered or left at a frame edge, and how
far it moved. This structured log is what the LLM narrator (a later phase)
will be allowed to describe -- nothing else -- so any sentence it writes can
be checked against it.
"""

from __future__ import annotations

import numpy as np

from tracelens.track.tracker import TrackOutput

EDGE_MARGIN = 3  # px: a box this close to the border counts as "at the edge"


def region(cx: float, cy: float, width: int, height: int) -> str:
    """3x3 grid name for a point, e.g. 'top-left', 'center'."""
    col = ["left", "center", "right"][min(2, int(3 * cx / width))]
    row = ["top", "middle", "bottom"][min(2, int(3 * cy / height))]
    return "center" if (row, col) == ("middle", "center") else f"{row}-{col}"


def _at_edge(box: np.ndarray, width: int, height: int) -> bool:
    return (box[0] <= EDGE_MARGIN or box[1] <= EDGE_MARGIN
            or box[2] >= width - EDGE_MARGIN or box[3] >= height - EDGE_MARGIN)


def build_timeline(tracks: list[TrackOutput], target_id: int, segments: list[tuple[int, int]],
                   fps: float, frame_size: tuple[int, int]) -> list[dict]:
    width, height = frame_size
    boxes = {t.frame: np.asarray(t.box) for t in tracks if t.track_id == target_id}
    entries = []
    for a, b in segments:
        seen = [f for f in sorted(boxes) if a <= f <= b]
        if not seen:
            continue
        first, last = boxes[seen[0]], boxes[seen[-1]]
        centers = np.array([[(boxes[f][0] + boxes[f][2]) / 2, (boxes[f][1] + boxes[f][3]) / 2] for f in seen])
        entries.append({
            "segment_start_frame": a, "segment_end_frame": b,
            "first_seen_frame": seen[0], "last_seen_frame": seen[-1],
            "start_s": round(seen[0] / fps, 1), "end_s": round(seen[-1] / fps, 1),
            "duration_s": round((seen[-1] - seen[0] + 1) / fps, 1),
            "start_region": region(*centers[0], width, height),
            "end_region": region(*centers[-1], width, height),
            "entered_at_edge": bool(_at_edge(first, width, height)),
            "left_at_edge": bool(_at_edge(last, width, height)),
            "distance_px": round(float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()), 1),
        })
    return entries


def timeline_text(entries: list[dict], target_id: int) -> str:
    """Plain rendering of the structured log (no LLM involved)."""
    if not entries:
        return f"Selected target (tracker ID {target_id}): not found."
    lines = [f"Selected target (tracker ID {target_id}): {len(entries)} appearance(s)"]
    for i, e in enumerate(entries, 1):
        enter = "entered at the frame edge" if e["entered_at_edge"] else "first seen"
        leave = "left at the frame edge" if e["left_at_edge"] else "last seen"
        lines.append(
            f"  {i}. {e['start_s']:.1f}s-{e['end_s']:.1f}s ({e['duration_s']:.1f}s): {enter} "
            f"({e['start_region']}), moved ~{e['distance_px']:.0f}px, {leave} ({e['end_region']})"
        )
    return "\n".join(lines)
