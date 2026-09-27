"""Detect events in the scripted incident and score them against the script.

  python -m tracelens.events_demo --save outputs/incident.mp4
  python -m tracelens.events_demo --miss-rate 0.2 --jitter 3 --sweep

Pipeline: detect -> track -> click the target -> re-link its fragments by
appearance (top/bottom fingerprint) -> detect events -> match against the
ground-truth events (same type, within +/- 2 s).
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd

from tracelens.detect.detectors import SyntheticDetector
from tracelens.eval.summary import simulate_click
from tracelens.events.detect import Zone, detect_events, match_events
from tracelens.reid.appearance import track_fingerprints
from tracelens.reid.relink import merge_chain, relink_chain
from tracelens.track.tracker import run_tracker
from tracelens.video.draw import write_video
from tracelens.video.synthetic import (
    INCIDENT_FPS,
    INCIDENT_SIZE,
    RESTRICTED_ZONE,
    incident_gt_events,
    incident_scene,
    make_synthetic_video,
)

N_FRAMES = 300
LABELS = {"appear": "APPEARED", "disappear": "LEFT VIEW", "dwell": "LOITERING", "run": "RUNNING",
          "zone_enter": "ENTERED RESTRICTED ZONE", "zone_exit": "LEFT RESTRICTED ZONE",
          "object_left_behind": "OBJECT LEFT BEHIND"}


def run_pipeline(v, miss, jitter, seed, gt_id=1):
    tracks = run_tracker(SyntheticDetector(v.gt, miss, jitter, frame_size=INCIDENT_SIZE, seed=seed), N_FRAMES)
    tid = simulate_click(v.gt, tracks, gt_id)
    chain, _ = relink_chain(tracks, track_fingerprints(v.frames, tracks, parts=2), tid)
    merged = merge_chain(tracks, chain)
    return merged, tid, detect_events(merged, tid, INCIDENT_FPS, zones=[Zone("restricted", RESTRICTED_ZONE)])


def render(frames, tracks, tid, events, fps, show_s=2.0):
    zone = np.array(RESTRICTED_ZONE, dtype=np.int32)
    boxes = {t.frame: t.box for t in tracks if t.track_id == tid}
    out = []
    for f, img in enumerate(frames):
        img = img.copy()
        cv2.polylines(img, [zone], True, (0, 0, 200), 2)
        cv2.putText(img, "RESTRICTED", (zone[0][0] + 4, zone[0][1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
        if f in boxes:
            x1, y1, x2, y2 = (round(v) for v in boxes[f])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        active = [e for e in events if e["t"] <= f / fps < e["t"] + show_s]
        for k, e in enumerate(active):
            cv2.putText(img, f"{e['t']:.1f}s {LABELS[e['type']]}", (8, 44 + 20 * k),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(img, f"t={f / fps:5.1f}s", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        out.append(img)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Event detection on the scripted incident.")
    parser.add_argument("--miss-rate", type=float, default=0.0)
    parser.add_argument("--jitter", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep", action="store_true", help="event F1 across noise levels, 5 seeds each")
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    objs, script = incident_scene()
    v = make_synthetic_video(objs, n_frames=N_FRAMES, size=INCIDENT_SIZE, fps=INCIDENT_FPS)
    truth = incident_gt_events(v.gt, script)

    if args.sweep:
        rows = []
        for miss, jitter in ((0.0, 0.0), (0.1, 2.0), (0.2, 3.0), (0.3, 4.0)):
            ms = [match_events(run_pipeline(v, miss, jitter, s)[2], truth) for s in range(5)]
            walker = [sum(e["type"] not in ("appear", "disappear") for e in run_pipeline(v, miss, jitter, s, 3)[2])
                      for s in range(5)]
            rows.append({"miss_rate": miss, "jitter_px": jitter,
                         "event_f1_mean": np.mean([m["f1"] for m in ms]),
                         "event_f1_min": np.min([m["f1"] for m in ms]),
                         "timing_error_s": np.nanmean([m["mean_timing_error_s"] for m in ms]),
                         "innocent_walker_false_alarms": sum(walker)})
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        return

    tracks, tid, events = run_pipeline(v, args.miss_rate, args.jitter, args.seed)
    print("Detected events:")
    for e in events:
        extra = f" (confirmed {e['t_confirmed']:.1f}s)" if "t_confirmed" in e else ""
        until = f" until {e['t_end']:.1f}s" if "t_end" in e else ""
        print(f"  {e['t']:5.1f}s  {LABELS[e['type']]}{until}{extra}")
    m = match_events(events, truth)
    print(f"\nvs script: precision {m['precision']:.2f}  recall {m['recall']:.2f}  F1 {m['f1']:.2f}  "
          f"timing error {m['mean_timing_error_s']:.2f}s  missed {m['missed_types'] or 'none'}")
    if args.save:
        print("wrote", write_video(render(v.frames, tracks, tid, events, INCIDENT_FPS), args.save, INCIDENT_FPS))


if __name__ == "__main__":
    main()
