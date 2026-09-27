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


def majority_gt_id(gt: list[GTRow], tracks: list[TrackOutput], track_id: int, iou_thr: float = 0.5):
    """The ground-truth object a track mostly followed (None if it never overlapped one)."""
    gt_by_f: dict[int, list] = {}
    for r in gt:
        gt_by_f.setdefault(r[0], []).append(r)
    votes: dict[int, int] = {}
    for t in tracks:
        if t.track_id != track_id:
            continue
        for r in gt_by_f.get(t.frame, []):
            if iou_matrix(np.array(r[2:6])[None], np.asarray(t.box)[None])[0, 0] >= iou_thr:
                votes[r[1]] = votes.get(r[1], 0) + 1
    return max(votes, key=votes.get) if votes else None


def score_summary(gt: list[GTRow], tracks: list[TrackOutput], gt_id: int, n_frames: int,
                  frames: list | None = None, relink: str = "none", threshold: float = 0.8,
                  max_gap_frames: int = 150, **segment_kwargs) -> dict:
    """relink: 'none' (follow one track ID), 'whole' (whole-box colour fingerprint),
    or 'parts' (top/bottom fingerprint). Re-linking needs the video `frames`."""
    true_frames = {r[0] for r in gt if r[1] == gt_id}
    tid = simulate_click(gt, tracks, gt_id)
    if tid is None:
        return {"target_recall": 0.0, "target_precision": float("nan"), "compression": 0.0,
                "n_segments": 0, "n_track_ids": 0, "oracle_relink_recall": 0.0, "selected_track": None,
                "correct_links": 0, "wrong_links": 0}
    chain = [tid]
    if relink != "none":
        if frames is None:
            raise ValueError("re-linking needs the video frames")
        from tracelens.reid.appearance import track_fingerprints
        from tracelens.reid.relink import relink_chain

        fps = track_fingerprints(frames, tracks, parts=2 if relink == "parts" else 1)
        chain, _ = relink_chain(tracks, fps, tid, threshold, max_gap_frames)
    linked = chain[1:]
    correct = sum(majority_gt_id(gt, tracks, c) == gt_id for c in linked)
    chain_frames = sorted({t.frame for t in tracks if t.track_id in set(chain)})
    segs = segments_from_frames(chain_frames, n_frames, **segment_kwargs)
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
        "correct_links": int(correct),
        "wrong_links": int(len(linked) - correct),
    }
