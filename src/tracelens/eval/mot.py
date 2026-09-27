"""Tracking metrics, from scratch (motmetrics is used only in tests, to verify).

MOTA (CLEAR MOT): 1 - (misses + false positives + ID switches) / GT boxes.
  Per frame, GT objects keep last frame's match if it's still valid
  (IoU >= threshold); only the rest go to Hungarian matching. An ID switch
  is a GT object matched to a different track than the last track it was
  matched to (remembered across gaps). Can be negative.

IDF1: identity F1. One GLOBAL one-to-one assignment between GT identities
  and track identities (Hungarian, maximizing frames where the pair
  overlaps). IDF1 = 2 * IDTP / (GT boxes + predicted boxes). Unlike MOTA,
  it punishes a tracker that follows the right boxes under the wrong,
  changing identities -- exactly what matters for "follow THIS person".
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracelens.track.tracker import TrackOutput, iou_matrix
from tracelens.video.synthetic import GTRow


def _group(gt: list[GTRow], pred: list[TrackOutput]):
    gt_by_f, pr_by_f = defaultdict(dict), defaultdict(dict)
    for f, oid, x1, y1, x2, y2 in gt:
        gt_by_f[f][oid] = np.array([x1, y1, x2, y2], dtype=float)
    for p in pred:
        pr_by_f[p.frame][p.track_id] = np.asarray(p.box, dtype=float)
    return gt_by_f, pr_by_f


def clear_mot(gt: list[GTRow], pred: list[TrackOutput], iou_threshold: float = 0.5) -> dict:
    gt_by_f, pr_by_f = _group(gt, pred)
    last_match: dict[int, int] = {}  # gt id -> track id it was last matched to
    prev_frame_pairs: dict[int, int] = {}  # gt id -> track id matched in the PREVIOUS frame
    n_gt = fn = fp = idsw = matches = 0
    for f in sorted(set(gt_by_f) | set(pr_by_f)):
        g, p = gt_by_f.get(f, {}), pr_by_f.get(f, {})
        n_gt += len(g)
        pairs: dict[int, int] = {}
        # 1) keep last frame's correspondences that are still valid
        for gid, tid in prev_frame_pairs.items():
            if (gid in g and tid in p and tid not in pairs.values()
                    and iou_matrix(g[gid][None], p[tid][None])[0, 0] >= iou_threshold):
                pairs[gid] = tid
        # 2) Hungarian on the rest
        g_rest = [gid for gid in g if gid not in pairs]
        p_rest = [tid for tid in p if tid not in pairs.values()]
        if g_rest and p_rest:
            iou = iou_matrix(np.array([g[i] for i in g_rest]), np.array([p[i] for i in p_rest]))
            cost = np.where(iou >= iou_threshold, 1 - iou, 1e6)
            rows, cols = linear_sum_assignment(cost)
            for r, cidx in zip(rows, cols):
                if iou[r, cidx] >= iou_threshold:
                    pairs[g_rest[r]] = p_rest[cidx]
        for gid, tid in pairs.items():
            if gid in last_match and last_match[gid] != tid:
                idsw += 1
            last_match[gid] = tid
        matches += len(pairs)
        fn += len(g) - len(pairs)
        fp += len(p) - len(pairs)
        prev_frame_pairs = pairs
    mota = 1 - (fn + fp + idsw) / n_gt if n_gt else float("nan")
    return {"mota": mota, "num_gt": n_gt, "matches": matches, "misses": fn,
            "false_positives": fp, "id_switches": idsw}


def idf1(gt: list[GTRow], pred: list[TrackOutput], iou_threshold: float = 0.5) -> dict:
    gt_by_f, pr_by_f = _group(gt, pred)
    gt_ids = sorted({r[1] for r in gt})
    tr_ids = sorted({p.track_id for p in pred})
    n_gt_boxes, n_pr_boxes = len(gt), len(pred)
    if not gt_ids or not tr_ids:
        return {"idf1": 0.0, "idtp": 0, "idp": 0.0, "idr": 0.0}
    gi, ti = {g: i for i, g in enumerate(gt_ids)}, {t: i for i, t in enumerate(tr_ids)}
    overlap = np.zeros((len(gt_ids), len(tr_ids)))  # frames where GT g and track t overlap enough
    for f, g in gt_by_f.items():
        p = pr_by_f.get(f, {})
        if not p:
            continue
        gk, pk = list(g), list(p)
        iou = iou_matrix(np.array([g[k] for k in gk]), np.array([p[k] for k in pk]))
        for a, b in zip(*np.where(iou >= iou_threshold)):
            overlap[gi[gk[a]], ti[pk[b]]] += 1
    rows, cols = linear_sum_assignment(-overlap)  # one global identity mapping
    idtp = int(overlap[rows, cols].sum())
    return {"idf1": 2 * idtp / (n_gt_boxes + n_pr_boxes), "idtp": idtp,
            "idp": idtp / n_pr_boxes, "idr": idtp / n_gt_boxes}


def evaluate_tracking(gt: list[GTRow], pred: list[TrackOutput], iou_threshold: float = 0.5) -> dict:
    return {**clear_mot(gt, pred, iou_threshold), **idf1(gt, pred, iou_threshold)}
