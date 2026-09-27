"""Summarize a synthetic scene around one target.

  python -m tracelens.summarize_demo --scene patrol --target 1 --save outputs/summary.mp4
  python -m tracelens.summarize_demo --scene patrol --target 1 --sweep
  python -m tracelens.summarize_demo --scene lookalike --target 1 --relink parts --save outputs/relinked.mp4
  python -m tracelens.summarize_demo --compare-relink

The target is chosen the way a user would: by clicking on it (simulated as
a click on the ground-truth box centre). --sweep reports target recall,
precision and compression across detector miss rates, averaged over seeds.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tracelens.detect.detectors import SyntheticDetector
from tracelens.eval.summary import score_summary, simulate_click
from tracelens.reid.appearance import track_fingerprints
from tracelens.reid.relink import merge_chain, relink_chain
from tracelens.summarize.clip import build_summary_frames
from tracelens.summarize.segments import appearance_frames, segments_from_frames
from tracelens.summarize.timeline import build_timeline, timeline_text
from tracelens.track.tracker import run_tracker
from tracelens.video.draw import write_video
from tracelens.video.synthetic import (
    busy_scene,
    crossing_scene,
    lookalike_scene,
    make_synthetic_video,
    patrol_scene,
)

SCENES = {"busy": (busy_scene, 100), "crossing": (crossing_scene, 60), "patrol": (patrol_scene, 300),
          "lookalike": (lookalike_scene, 300)}
METRICS = ["target_recall", "target_precision", "compression", "n_track_ids", "oracle_relink_recall"]
RELINK_METRICS = ["target_recall", "target_precision", "compression", "correct_links", "wrong_links"]


def compare_relink(miss_rate: float, jitter: float, seeds: int = 5) -> None:
    rows = []
    for scene_name in ("patrol", "lookalike"):
        scene, n = SCENES[scene_name]
        v = make_synthetic_video(scene(), n_frames=n)
        for mode in ("none", "whole", "parts"):
            runs = [score_summary(v.gt, run_tracker(SyntheticDetector(v.gt, miss_rate, jitter, seed=s), n), 1, n,
                                  frames=v.frames, relink=mode) for s in range(seeds)]
            rows.append({"scene": scene_name, "relink": mode,
                         **{m: np.nanmean([r[m] for r in runs]) for m in RELINK_METRICS}})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Target-centric summary of a synthetic scene.")
    parser.add_argument("--scene", choices=list(SCENES), default="patrol")
    parser.add_argument("--target", type=int, default=1, help="ground-truth object to follow")
    parser.add_argument("--miss-rate", type=float, default=0.0)
    parser.add_argument("--jitter", type=float, default=2.0)
    parser.add_argument("--fp-rate", type=float, default=0.0)
    parser.add_argument("--max-gap", type=int, default=5)
    parser.add_argument("--pad", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--relink", choices=["none", "whole", "parts"], default="none",
                        help="re-link a returning target by appearance (whole box, or top/bottom parts)")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--compare-relink", action="store_true",
                        help="table: no re-linking vs whole-box vs part-based, on patrol + lookalike")
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    if args.compare_relink:
        compare_relink(args.miss_rate, args.jitter)
        return
    scene, n = SCENES[args.scene]
    v = make_synthetic_video(scene(), n_frames=n)
    seg_kw = {"max_gap": args.max_gap, "pad": args.pad}

    if args.sweep:
        rows = []
        for mr in (0.0, 0.1, 0.2, 0.3, 0.4):
            runs = [score_summary(v.gt, run_tracker(SyntheticDetector(v.gt, mr, args.jitter, args.fp_rate, seed=s), n),
                                  args.target, n, **seg_kw) for s in range(5)]
            rows.append({"miss_rate": mr, **{m: np.nanmean([r[m] for r in runs]) for m in METRICS}})
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        return

    tracks = run_tracker(SyntheticDetector(v.gt, args.miss_rate, args.jitter, args.fp_rate, seed=args.seed), n)
    tid = simulate_click(v.gt, tracks, args.target)
    if tid is None:
        print("target was never tracked -- nothing to summarize")
        return
    if args.relink != "none":
        fps = track_fingerprints(v.frames, tracks, parts=2 if args.relink == "parts" else 1)
        chain, log = relink_chain(tracks, fps, tid, args.threshold)
        for e in log:
            print(f"  after frame {e['after_frame']}: candidate track {e['candidate']} "
                  f"similarity {e['similarity']:.3f} -> {'LINKED' if e['linked'] else 'rejected'}")
        tracks = merge_chain(tracks, chain)
    segs = segments_from_frames(appearance_frames(tracks, tid), n, **seg_kw)
    h, w = v.frames[0].shape[:2]
    print(timeline_text(build_timeline(tracks, tid, segs, v.fps, (w, h)), tid))
    orig = run_tracker(SyntheticDetector(v.gt, args.miss_rate, args.jitter, args.fp_rate, seed=args.seed), n)
    m = score_summary(v.gt, orig, args.target, n, frames=v.frames, relink=args.relink,
                      threshold=args.threshold, **seg_kw)
    keys = METRICS + ["correct_links", "wrong_links"]
    print("\n" + "  ".join(f"{k}={m[k]:.3f}" if isinstance(m[k], float) else f"{k}={m[k]}" for k in keys))
    if args.save:
        print("wrote", write_video(build_summary_frames(v.frames, tracks, tid, segs, v.fps), args.save, v.fps))


if __name__ == "__main__":
    main()
