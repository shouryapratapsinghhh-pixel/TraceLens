"""Score a target summary against ground truth.

target_recall     share of the target's true appearances that made it into the summary
target_precision  share of summary frames in which the target is actually visible
compression       summary length / original length (lower = shorter)
n_track_ids       how many different track IDs the tracker gave this ONE target
oracle_relink_recall
                  recall if every one of those IDs were merged into one -- uses
                  ground truth, so it's an UPPER BOUND on what re-identifying a
                  returning target could recover. Labelled oracle; never a result.
"""

from __future__ import annotations

import numpy as np

from tracelens.summarize.segments import (
    appearance_frames,
    frames_in,
    segments_from_frames,
    select_track_by_point,
)
from tracelens.track.tracker import TrackOutput, iou_matrix
from tracelens.video.synthetic import GTRow


def simulate_click(gt: list[GTRow], tracks: list[TrackOutput], gt_id: int, search: int = 30) -> int | None:
    """A user clicks the centre of the target a few frames after it appears
    (the tracker only reports confirmed tracks). Returns the selected track ID."""
    target_rows = sorted((r for r in gt if r[1] == gt_id), key=lambda r: r[0])
    for f, _, x1, y1, x2, y2 in target_rows[:search]:
        tid = select_track_by_point(tracks, f, ((x1 + x2) / 2, (y1 + y2) / 2))
        if tid is not None:
            return tid
    return None


def ids_matching_target(gt: list[GTRow], tracks: list[TrackOutput], gt_id: int, iou_thr: float = 0.5) -> set[int]:
    gt_boxes = {r[0]: np.array(r[2:6]) for r in gt if r[1] == gt_id}
    ids = set()
    for t in tracks:
        if t.frame in gt_boxes and iou_matrix(gt_boxes[t.frame][None], np.asarray(t.box)[None])[0, 0] >= iou_thr:
            ids.add(t.track_id)
    return ids


def score_summary(gt: list[GTRow], tracks: list[TrackOutput], gt_id: int, n_frames: int,
                  **segment_kwargs) -> dict:
    true_frames = {r[0] for r in gt if r[1] == gt_id}
    tid = simulate_click(gt, tracks, gt_id)
    if tid is None:
        return {"target_recall": 0.0, "target_precision": float("nan"), "compression": 0.0,
                "n_segments": 0, "n_track_ids": 0, "oracle_relink_recall": 0.0, "selected_track": None}
    segs = segments_from_frames(appearance_frames(tracks, tid), n_frames, **segment_kwargs)
    kept = set(frames_in(segs).tolist())

    matching = ids_matching_target(gt, tracks, gt_id)
    merged = sorted({t.frame for t in tracks if t.track_id in matching})
    oracle_kept = set(frames_in(segments_from_frames(merged, n_frames, **segment_kwargs)).tolist())

    return {
        "target_recall": len(kept & true_frames) / len(true_frames),
        "target_precision": len(kept & true_frames) / len(kept) if kept else float("nan"),
        "compression": len(kept) / n_frames,
        "n_segments": len(segs),
        "n_track_ids": len(matching),
        "oracle_relink_recall": len(oracle_kept & true_frames) / len(true_frames),
        "selected_track": tid,
    }
