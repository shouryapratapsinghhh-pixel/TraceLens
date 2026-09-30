"""Evaluate TraceLens on real footage (MOT17 / MOTChallenge format).

  python -m tracelens.mot_eval --root data/raw/MOT17/train
  python -m tracelens.mot_eval --root data/raw/MOT17/train --seqs MOT17-02-FRCNN --max-age 30
  python -m tracelens.mot_eval --root data/raw/MOT17/train --detector yolo      # needs ultralytics

Two parts, per sequence:
  1. TRACKING: the from-scratch tracker on every pedestrian -> MOTA, IDF1, ID switches.
  2. TARGET SUMMARIES: the `--targets` people with the most visible frames AMONG THOSE WHO
     WERE OUT OF VIEW FOR OVER 1 s AT LEAST ONCE (where re-linking matters): how much of
     their visible time the summary recovers, with no re-linking vs part-based re-linking,
     and how many re-links merge a DIFFERENT person. A frame counts as the target being
     visible when MOT17 says at least `--min-visibility` of the person is unoccluded.

Scoring: --protocol official (default) applies MOTChallenge's preprocessing -- predictions on
distractor classes are ignored -- verified to give identical MOTA/IDF1 to the official TrackEval
toolkit (tests/test_trackeval_parity.py). --protocol strict reproduces the earlier, harsher scoring.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tracelens.detect.detectors import YoloDetector
from tracelens.eval.mot import evaluate_tracking
from tracelens.eval.summary import score_summary
from tracelens.track.tracker import run_tracker
from tracelens.video.mot import MOTSequence, PublicDetector, official_filter

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def pick_targets(gt: list[tuple], n: int, min_frames: int = 60, min_gap: int | None = None) -> list[int]:
    """The n people with the most visible frames (at least min_frames). With min_gap, only
    people who were out of view (not visible enough) for more than min_gap frames at some
    point -- the cases re-linking exists for; the longest-visible people are usually the
    ones who never get occluded, where re-linking has nothing to do."""
    frames: dict[int, list[int]] = {}
    for r in gt:
        frames.setdefault(r[1], []).append(r[0])
    ranked = []
    for oid, fs in frames.items():
        fs = sorted(fs)
        if len(fs) < min_frames:
            continue
        if min_gap is not None and (len(fs) < 2 or int(np.max(np.diff(fs))) <= min_gap):
            continue
        ranked.append((len(fs), oid))
    return [oid for _, oid in sorted(ranked, reverse=True)[:n]]


def evaluate_sequence(seq: MOTSequence, detector_kind: str, min_conf: float, tracker_kw: dict,
                      n_targets: int, min_visibility: float, relink_window_s: float = 10.0,
                      protocol: str = "official") -> tuple[dict, list[dict]]:
    det = PublicDetector(seq, min_conf) if detector_kind == "public" else YoloDetector()
    t0 = time.perf_counter()
    tracks = run_tracker(det, seq.n_frames, frames=seq.frames if detector_kind == "yolo" else None, **tracker_kw)
    secs = time.perf_counter() - t0

    gt_all = seq.ground_truth()
    m = evaluate_tracking(gt_all, official_filter(seq, tracks) if protocol == "official" else tracks)
    row = {"sequence": seq.name, "frames": seq.n_frames, "fps": seq.fps, "people": len({r[1] for r in gt_all}),
           "mota": m["mota"], "idf1": m["idf1"], "id_switches": m["id_switches"],
           "misses": m["misses"], "false_positives": m["false_positives"],
           "tracker_fps": seq.n_frames / secs}

    gt_vis = seq.ground_truth(min_visibility)
    targets = []
    for gid in pick_targets(gt_vis, n_targets, min_gap=int(seq.fps)):  # out of view > 1 s at least once
        for mode in ("none", "parts"):
            r = score_summary(gt_vis, tracks, gid, seq.n_frames, frames=seq.frames, relink=mode,
                              max_gap_frames=int(relink_window_s * seq.fps))
            targets.append({"sequence": seq.name, "target": gid, "relink": mode,
                            **{k: r[k] for k in ("target_recall", "target_precision", "compression",
                                                 "n_track_ids", "correct_links", "wrong_links")}})
    return row, targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on MOT17-format real footage.")
    parser.add_argument("--root", required=True, help="folder containing sequence folders (e.g. MOT17/train)")
    parser.add_argument("--seqs", nargs="*", default=None, help="default: every *-FRCNN sequence in --root")
    parser.add_argument("--detector", choices=["public", "yolo"], default="public")
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-age", type=int, default=30, help="frames a lost track survives (30 = 1 s at 30 fps)")
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--targets", type=int, default=5, help="people per sequence for the summary study")
    parser.add_argument("--min-visibility", type=float, default=0.25)
    parser.add_argument("--relink-window-s", type=float, default=10.0,
                        help="how long a target may be out of view and still be re-linked")
    parser.add_argument("--protocol", choices=["official", "strict"], default="official",
                        help="official = MOTChallenge preprocessing (verified identical to TrackEval); "
                             "strict = count predictions on distractors as false positives (the old scoring)")
    parser.add_argument("--out", default="reports/mot17")
    args = parser.parse_args()

    root = Path(args.root)
    names = args.seqs or sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.endswith("FRCNN"))
    if not names:
        raise SystemExit(f"no sequences found in {root} (expected folders like MOT17-02-FRCNN)")
    kw = {"iou_threshold": args.iou_threshold, "max_age": args.max_age, "min_hits": args.min_hits}

    rows, targets = [], []
    for name in names:
        logger.info("sequence %s ...", name)
        row, t = evaluate_sequence(MOTSequence(root / name), args.detector, args.min_conf, kw,
                                   args.targets, args.min_visibility, args.relink_window_s, args.protocol)
        rows.append(row)
        targets += t

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    track_df, target_df = pd.DataFrame(rows), pd.DataFrame(targets)
    track_df.to_csv(out / "tracking.csv", index=False)
    target_df.to_csv(out / "targets.csv", index=False)

    print("\nTRACKING (all pedestrians)")
    print(track_df.round(3).to_string(index=False))
    total = {"mota (mean)": track_df["mota"].mean(), "idf1 (mean)": track_df["idf1"].mean(),
             "id_switches (total)": int(track_df["id_switches"].sum())}
    print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in total.items()))
    if len(target_df):
        print(f"\nTARGET SUMMARIES (mean over {target_df['target'].nunique()} people who were out of view "
              "> 1 s at least once)")
        cols = ["target_recall", "target_precision", "compression", "n_track_ids", "correct_links", "wrong_links"]
        print(target_df.groupby("relink")[cols].mean().round(3).to_string())
    print(f"\nwrote {out}/tracking.csv, {out}/targets.csv")


if __name__ == "__main__":
    main()
