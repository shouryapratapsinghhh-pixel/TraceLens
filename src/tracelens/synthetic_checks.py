"""Synthetic correctness checks, saved to CSV so the README numbers come from a file.

  python -m tracelens.synthetic_checks --out reports

crossing.csv   two identical objects cross paths: ID switches with vs without Kalman motion
               prediction, 8 seeds (same setup as tests/test_phase0.py)
shaky_cmc.csv  busy scene through a shaking camera: tracking with vs without camera-motion
               compensation, 3 seeds (same setup as tests/test_cmc.py)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tracelens.detect.detectors import SyntheticDetector
from tracelens.eval.mot import evaluate_tracking
from tracelens.track.cmc import sequence_warps
from tracelens.track.tracker import run_tracker
from tracelens.video.synthetic import (
    busy_scene,
    crossing_scene,
    make_shaky_video,
    make_synthetic_video,
)

COLS = ["mota", "idf1", "id_switches", "misses", "false_positives"]


def crossing(seeds: int = 8) -> pd.DataFrame:
    v = make_synthetic_video(crossing_scene(), n_frames=60)
    rows = []
    for method, motion in (("Kalman tracker", True), ("no motion prediction", False)):
        for s in range(seeds):
            m = evaluate_tracking(v.gt, run_tracker(SyntheticDetector(v.gt, seed=s), 60, use_motion=motion))
            rows.append({"method": method, "seed": s, **{k: m[k] for k in COLS}})
    return pd.DataFrame(rows)


def shaky_cmc(seeds: int = 3) -> pd.DataFrame:
    v, _ = make_shaky_video(busy_scene(), n_frames=100, shake_px=20)
    warps = sequence_warps(v.frames, detector=SyntheticDetector(v.gt, miss_rate=0.1, seed=99))
    rows = []
    for method, w in (("no CMC", None), ("CMC", warps)):
        for s in range(seeds):
            m = evaluate_tracking(v.gt, run_tracker(SyntheticDetector(v.gt, miss_rate=0.1, seed=s), 100, warps=w))
            rows.append({"method": method, "seed": s, **{k: m[k] for k in COLS}})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic crossing and shaky-camera checks.")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cr = crossing()
    cr.to_csv(out / "crossing.csv", index=False)
    print("CROSSING (8 seeds)")
    print(cr.groupby("method", sort=False).agg(runs_with_a_switch=("id_switches", lambda s: int((s > 0).sum())),
                                               total_switches=("id_switches", "sum")).to_string())

    sh = shaky_cmc()
    sh.to_csv(out / "shaky_cmc.csv", index=False)
    print("\nSHAKY CAMERA (mean of 3 seeds)")
    print(sh.groupby("method", sort=False)[COLS].mean().round(3).to_string())
    print(f"\nwrote {out}/crossing.csv, {out}/shaky_cmc.csv")


if __name__ == "__main__":
    main()
