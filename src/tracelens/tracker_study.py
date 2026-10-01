"""Which tracker configuration is best on real footage? Tune on 2 sequences, report on 5.

  python -m tracelens.tracker_study --root data/raw/MOT17/train

Candidates (public FRCNN detections):
  single-stage tracker, detection cutoff 0.5   <- the current baseline
  single-stage tracker, lower cutoffs          (keep weak detections)
  ByteTrack two-stage association              (weak detections may extend a track, never start one)
each crossed with max_age (how long a lost track survives) and the IoU threshold.

Objective: mean IDF1 on the tune sequences (identity -- what "follow this person" needs).
Tune = MOT17-02 + -05, test = the other five, fixed before any result, as in relink_study.
The synthetic check couldn't settle single-stage vs ByteTrack (its false alarms are
one-frame noise that min_hits already removes); real detectors' persistent weak false
alarms are the open question, so real data decides.
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tracelens.eval.mot import evaluate_tracking
from tracelens.relink_study import TEST, TUNE
from tracelens.track.cmc import sequence_warps
from tracelens.track.tracker import run_tracker
from tracelens.video.mot import MOTSequence, PublicDetector, official_filter

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASELINE = {"mode": "single", "cutoff": 0.5, "max_age": 30, "iou_threshold": 0.3}


def candidates() -> list[dict]:
    out = []
    for max_age, iou in itertools.product([30, 60], [0.2, 0.3]):
        for cutoff in (0.5, 0.3, 0.1):
            out.append({"mode": "single", "cutoff": cutoff, "max_age": max_age, "iou_threshold": iou})
        for high in (0.5, 0.6, 0.7):
            out.append({"mode": "byte", "cutoff": 0.1, "high_thresh": high, "max_age": max_age, "iou_threshold": iou})
    return out


def warps_for(seq: MOTSequence, det_file: str) -> np.ndarray:
    """Camera-motion transforms for a sequence, cached next to its detections. Detected boxes
    (from the SAME detection file, never ground truth) are masked out while estimating."""
    cache = seq.dir / "det" / f"cmc_{Path(det_file).stem}.npy"
    return sequence_warps(seq.frames, cache_path=cache, detector=PublicDetector(seq, 0.3, det_file))


def run_config(seq: MOTSequence, gt: list, cfg: dict, det_file: str = "det.txt", protocol: str = "official",
               warps=None) -> dict:
    det = PublicDetector(seq, min_conf=cfg["cutoff"], det_file=det_file)
    kw = {"max_age": cfg["max_age"], "iou_threshold": cfg["iou_threshold"]}
    if cfg["mode"] == "byte":
        kw.update(byte=True, high_thresh=cfg["high_thresh"], low_thresh=cfg["cutoff"])
    tracks = run_tracker(det, seq.n_frames, warps=warps, **kw)
    m = evaluate_tracking(gt, official_filter(seq, tracks) if protocol == "official" else tracks)
    return {"sequence": seq.name, **{k: m[k] for k in ("mota", "idf1", "id_switches", "misses", "false_positives")}}


def label(cfg: dict) -> str:
    core = f"ByteTrack high={cfg['high_thresh']}" if cfg["mode"] == "byte" else f"single cutoff={cfg['cutoff']}"
    return f"{core}, max_age={cfg['max_age']}, iou={cfg['iou_threshold']}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the tracker on 2 MOT17 sequences, report on 5.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--tune", nargs="+", default=TUNE)
    parser.add_argument("--test", nargs="+", default=TEST)
    parser.add_argument("--dets", default="det.txt",
                        help="detection file in each sequence's det/ folder: det.txt = public; "
                             "e.g. yolov8m.txt = cached YOLO (see tracelens.detect_cache)")
    parser.add_argument("--protocol", choices=["official", "strict"], default="official")
    parser.add_argument("--out", default="reports/tracker_study")
    args = parser.parse_args()
    if set(args.tune) & set(args.test):
        raise SystemExit("a sequence can't be in both --tune and --test")
    root = Path(args.root)

    tune = [(s := MOTSequence(root / n), s.ground_truth()) for n in args.tune]
    scores = np.concatenate([np.loadtxt(s.dir / "det" / args.dets, delimiter=",", ndmin=2)[:, 6] for s, _ in tune])
    logger.info("%s detection scores: min %.2f, median %.2f, max %.2f; share below 0.5: %.0f%%  "
                "(cutoffs assume a 0-1 scale)", args.dets, scores.min(), np.median(scores), scores.max(),
                100 * np.mean(scores < 0.5))

    grid = []
    for cfg in candidates():
        rows = [run_config(s, gt, cfg, args.dets, args.protocol) for s, gt in tune]
        grid.append({"config": label(cfg), "idf1": np.mean([r["idf1"] for r in rows]),
                     "mota": np.mean([r["mota"] for r in rows]), "_cfg": cfg})
    grid_df = pd.DataFrame(grid).sort_values(["idf1", "mota"], ascending=False)
    best = grid_df.iloc[0]["_cfg"]
    logger.info("chosen on tune sequences: %s", label(best))

    # camera-motion compensation: decided on the tune sequences only, rule fixed in advance:
    # keep CMC if it raises the best configuration's mean tune IDF1
    tune_warps = [warps_for(s, args.dets) for s, _ in tune]
    idf1_cmc = np.mean([run_config(s, gt, best, args.dets, args.protocol, w)["idf1"]
                        for (s, gt), w in zip(tune, tune_warps)])
    use_cmc = idf1_cmc > grid_df.iloc[0]["idf1"]
    logger.info("CMC on tune sequences: IDF1 %.3f -> %.3f  => %s", grid_df.iloc[0]["idf1"], idf1_cmc,
                "KEEP CMC" if use_cmc else "no CMC")

    test = [(s := MOTSequence(root / n), s.ground_truth()) for n in args.test]
    res = []
    for name, cfg in (("baseline (single, cutoff 0.5)", BASELINE), (f"tuned: {label(best)}", best)):
        res += [{"method": name, **run_config(s, gt, cfg, args.dets, args.protocol)} for s, gt in test]
    cmc_name = f"tuned + CMC{' (CHOSEN)' if use_cmc else ' (not chosen; shown for transparency)'}"
    res += [{"method": cmc_name, **run_config(s, gt, best, args.dets, args.protocol, warps_for(s, args.dets))}
            for s, gt in test]
    res_df = pd.DataFrame(res)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid_df.drop(columns="_cfg").to_csv(out / "tune_grid.csv", index=False)
    import json

    (out / "decision.json").write_text(json.dumps({
        "tune_sequences": args.tune, "test_sequences": args.test, "detections": args.dets,
        "protocol": args.protocol, "chosen_config": label(best),
        "tune_idf1_without_cmc": round(float(grid_df.iloc[0]["idf1"]), 4),
        "tune_idf1_with_cmc": round(float(idf1_cmc), 4), "cmc_kept": bool(use_cmc),
        "rule": "keep CMC only if it raises the chosen configuration's mean tune-set IDF1",
    }, indent=2))
    res_df.to_csv(out / "test_results.csv", index=False)
    print("\nTUNING GRID (top 8 by IDF1, tune sequences only)")
    print(grid_df.drop(columns="_cfg").head(8).round(3).to_string(index=False))
    print("\nTEST SEQUENCES ONLY -- per sequence")
    print(res_df.pivot(index="sequence", columns="method", values="idf1").round(3).to_string())
    print("\nTEST SEQUENCES ONLY -- mean")
    agg = res_df.groupby("method", sort=False).agg(mota=("mota", "mean"), idf1=("idf1", "mean"),
                                                   id_switches=("id_switches", "sum"), misses=("misses", "sum"),
                                                   false_positives=("false_positives", "sum"))
    print(agg.round(3).to_string())
    print(f"\nwrote {out}/tune_grid.csv, {out}/test_results.csv, {out}/decision.json")


if __name__ == "__main__":
    main()
