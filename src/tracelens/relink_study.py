"""Tune re-linking on some MOT17 sequences, report it on OTHERS.

  python -m tracelens.relink_study --root data/raw/MOT17/train

Protocol, fixed before looking at any results:
  tune  MOT17-02 (static camera, crowded) + MOT17-05 (moving camera)
  test  the other five FRCNN sequences -- never used to choose anything
Objective (second pass): mean IDENTITY F1 -- is the highlighted box actually the
target? The first pass used frame-level summary F1, which can't see a wrong
identity: in a crowd the real target is usually elsewhere in the frame while the
summary follows a stranger, so those frames still counted as correct. That
objective was replaced AFTER the first test results were seen -- disclosed here,
because changing an objective after seeing test results is a form of test-set
feedback. `--objective summary_f1` reproduces the first pass.

Grid: similarity threshold x motion gate (max walking speed in body-heights/s)
x ambiguity margin. Compared on test: no re-linking, the original re-linker
(threshold 0.8, no gate, no margin), and the tuned re-linker.
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import pandas as pd

from tracelens.eval.summary import score_summary
from tracelens.mot_eval import pick_targets
from tracelens.reid.appearance import LazyFingerprints
from tracelens.track.tracker import run_tracker
from tracelens.video.mot import MOTSequence, PublicDetector

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TUNE = ["MOT17-02-FRCNN", "MOT17-05-FRCNN"]
TEST = ["MOT17-04-FRCNN", "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
GRID = {"threshold": [0.8, 0.85, 0.9, 0.95], "max_speed_bh": [None, 1.0, 2.0], "margin": [0.0, 0.05]}
ORIGINAL = {"threshold": 0.8, "max_speed_bh": None, "margin": 0.0}
COLS = ["identity_f1", "identity_recall", "identity_precision", "summary_f1", "compression",
        "correct_links", "wrong_links"]


def prepare(seq_dir: Path, n_targets: int, min_visibility: float, tracker_kw: dict, min_conf: float) -> dict:
    """Track once per sequence; every setting reuses the tracks and one fingerprint cache."""
    seq = MOTSequence(seq_dir)
    tracks = run_tracker(PublicDetector(seq, min_conf), seq.n_frames, **tracker_kw)
    gt = seq.ground_truth(min_visibility)
    return {"seq": seq, "tracks": tracks, "gt": gt, "gt_all": seq.ground_truth(), "fps_cache": LazyFingerprints(seq.frames, tracks, parts=2),
            "targets": pick_targets(gt, n_targets, min_gap=int(seq.fps))}


def run_setting(prep: dict, setting: dict | None, window_s: float) -> list[dict]:
    seq, rows = prep["seq"], []
    for gid in prep["targets"]:
        if setting is None:
            r = score_summary(prep["gt"], prep["tracks"], gid, seq.n_frames, gt_all=prep["gt_all"])
        else:
            r = score_summary(prep["gt"], prep["tracks"], gid, seq.n_frames, frames=seq.frames, relink="parts",
                              threshold=setting["threshold"], max_gap_frames=int(window_s * seq.fps),
                              relink_kwargs={"fps": seq.fps, "max_speed_bh": setting["max_speed_bh"],
                                             "margin": setting["margin"]},
                              fingerprints=prep["fps_cache"], gt_all=prep["gt_all"])
        rows.append({"sequence": seq.name, "target": gid, **{k: r[k] for k in COLS}})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune re-linking on 2 sequences, report on 5 others.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--tune", nargs="+", default=TUNE)
    parser.add_argument("--test", nargs="+", default=TEST)
    parser.add_argument("--targets", type=int, default=5)
    parser.add_argument("--min-visibility", type=float, default=0.25)
    parser.add_argument("--relink-window-s", type=float, default=10.0)
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--objective", choices=["identity_f1", "summary_f1"], default="identity_f1")
    parser.add_argument("--out", default="reports/relink_study")
    args = parser.parse_args()
    if set(args.tune) & set(args.test):
        raise SystemExit("a sequence can't be in both --tune and --test")
    root, kw = Path(args.root), {"max_age": args.max_age}

    logger.info("tracking tune sequences %s", args.tune)
    tune = [prepare(root / s, args.targets, args.min_visibility, kw, args.min_conf) for s in args.tune]
    settings = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    grid_rows = []
    for st in settings:
        rs = [r for p in tune for r in run_setting(p, st, args.relink_window_s)]
        df = pd.DataFrame(rs)
        grid_rows.append({**st, "identity_f1": df["identity_f1"].mean(), "summary_f1": df["summary_f1"].mean(),
                          "wrong_links": df["wrong_links"].mean(), "correct_links": df["correct_links"].mean()})
    grid = pd.DataFrame(grid_rows).sort_values([args.objective, "wrong_links"], ascending=[False, True])
    best = {k: (None if pd.isna(v) else v) for k, v in grid.iloc[0][list(GRID)].items()}
    logger.info("chosen on tune sequences: %s", best)

    logger.info("tracking test sequences %s", args.test)
    test = [prepare(root / s, args.targets, args.min_visibility, kw, args.min_conf) for s in args.test]
    results = []
    for name, st in (("no re-linking", None), ("original re-linker", ORIGINAL), ("tuned re-linker", best)):
        for p in test:
            results += [{"method": name, **r} for r in run_setting(p, st, args.relink_window_s)]
    res = pd.DataFrame(results)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / "tune_grid.csv", index=False)
    res.to_csv(out / "test_results.csv", index=False)
    print(f"\nTUNING GRID (top 8 by {args.objective}, tune sequences only)")
    print(grid.head(8).round(3).to_string(index=False))
    print(f"\nchosen: threshold={best['threshold']}  max_speed_bh={best['max_speed_bh']}  margin={best['margin']}")
    n = res["target"].nunique() if len(res) else 0
    print(f"\nTEST SEQUENCES ONLY ({len(args.test)} sequences, {len(res) // 3} targets) -- mean per target")
    print(res.groupby("method", sort=False)[COLS].mean().round(3).to_string())
    print(f"\nwrote {out}/tune_grid.csv, {out}/test_results.csv  ({n} distinct target ids)")


if __name__ == "__main__":
    main()
