"""Track a synthetic scene and report metrics (offline, no downloads).

  python -m tracelens.track_demo --scene busy --miss-rate 0.2 --jitter 2 --fp-rate 0.1
  python -m tracelens.track_demo --scene crossing --save outputs/crossing.mp4

Compares the Kalman tracker against the same tracker with motion
prediction turned off, on identical detections.
"""

from __future__ import annotations

import argparse

import pandas as pd

from tracelens.detect.detectors import SyntheticDetector
from tracelens.eval.mot import evaluate_tracking
from tracelens.track.tracker import run_tracker
from tracelens.video.draw import draw_tracks, write_video
from tracelens.video.synthetic import busy_scene, crossing_scene, make_synthetic_video

SCENES = {"busy": busy_scene, "crossing": crossing_scene}


def main() -> None:
    parser = argparse.ArgumentParser(description="Track a synthetic scene and report MOTA/IDF1.")
    parser.add_argument("--scene", choices=list(SCENES), default="busy")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--miss-rate", type=float, default=0.0)
    parser.add_argument("--jitter", type=float, default=0.0)
    parser.add_argument("--fp-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default=None, help="write an annotated .mp4 of the Kalman tracker")
    args = parser.parse_args()

    v = make_synthetic_video(SCENES[args.scene](), n_frames=args.frames)
    rows, kalman_tracks = {}, None
    for name, motion in (("Kalman tracker", True), ("no motion prediction", False)):
        det = SyntheticDetector(v.gt, args.miss_rate, args.jitter, args.fp_rate, seed=args.seed)
        tracks = run_tracker(det, args.frames, use_motion=motion)
        rows[name] = evaluate_tracking(v.gt, tracks)
        if motion:
            kalman_tracks = tracks
    cols = ["mota", "idf1", "id_switches", "misses", "false_positives"]
    print(pd.DataFrame(rows).T[cols].round(3).to_string())
    if args.save:
        print("wrote", write_video(draw_tracks(v.frames, kalman_tracks), args.save, v.fps))


if __name__ == "__main__":
    main()
