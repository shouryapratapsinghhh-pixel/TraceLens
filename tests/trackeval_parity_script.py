"""Score the same tracker output with (a) our official_filter + metrics and (b) real TrackEval."""
import sys
from pathlib import Path

import numpy as np

for name, typ in (("int", int), ("float", float), ("bool", bool)):  # TrackEval predates NumPy 2
    if not hasattr(np, name):
        setattr(np, name, typ)

from tracelens.detect.detectors import SyntheticDetector
from tracelens.eval.mot import evaluate_tracking
from tracelens.track.tracker import run_tracker
from tracelens.video.mot import MOTSequence, official_filter
from tracelens.video.synthetic import SyntheticObject, busy_scene, make_synthetic_video

root = Path(sys.argv[1])
seq_name, n = "FAKE-01-FRCNN", 100
objs = busy_scene() + [SyntheticObject(5, start=(200, 20), velocity=(0, 0), color=(200, 200, 200)),   # static person
                       SyntheticObject(6, start=(260, 150), velocity=(-1, 0), color=(90, 90, 200))]  # reflection
v = make_synthetic_video(objs, n_frames=n)
seq_dir = root / "gt" / "MOT17-train" / seq_name
(seq_dir / "gt").mkdir(parents=True)
(seq_dir / "seqinfo.ini").write_text(f"[Sequence]\nname={seq_name}\nimDir=img1\nframeRate=10\nseqLength={n}\n"
                                     "imWidth=320\nimHeight=240\nimExt=.jpg\n")
lines = []
for f, oid, x1, y1, x2, y2 in v.gt:
    cls = {5: 7, 6: 12}.get(oid, 1)
    mark = 0 if (oid == 3 and 40 <= f < 70) else 1          # a zero-marked stretch of a pedestrian
    lines.append(f"{f + 1},{oid},{x1},{y1},{x2 - x1},{y2 - y1},{mark},{cls},1.0")
(seq_dir / "gt" / "gt.txt").write_text("\n".join(lines) + "\n")

det = SyntheticDetector(v.gt, miss_rate=0.15, jitter=3.0, false_positive_rate=0.3, seed=0)
tracks = run_tracker(det, n, min_hits=2)
trk_dir = root / "trackers" / "MOT17-train" / "mine" / "data"
trk_dir.mkdir(parents=True)
trk_dir.joinpath(f"{seq_name}.txt").write_text("\n".join(
    f"{t.frame + 1},{t.track_id},{t.box[0]:.3f},{t.box[1]:.3f},{t.box[2] - t.box[0]:.3f},{t.box[3] - t.box[1]:.3f},1,-1,-1,-1"
    for t in tracks) + "\n")

seq = MOTSequence(seq_dir)
removed = len(tracks) - len(official_filter(seq, tracks))
ours = evaluate_tracking(seq.ground_truth(), official_filter(seq, tracks))
strict = evaluate_tracking(seq.ground_truth(), tracks)

import trackeval

cfg = trackeval.Evaluator.get_default_eval_config()
cfg.update({"PRINT_RESULTS": False, "PRINT_CONFIG": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False, "USE_PARALLEL": False, "TIME_PROGRESS": False, "DISPLAY_LESS_PROGRESS": True})
dcfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
dcfg.update({"GT_FOLDER": str(root / "gt"), "TRACKERS_FOLDER": str(root / "trackers"), "BENCHMARK": "MOT17",
             "SPLIT_TO_EVAL": "train", "TRACKERS_TO_EVAL": ["mine"], "SEQ_INFO": {seq_name: n}, "DO_PREPROC": True})
res, _ = trackeval.Evaluator(cfg).evaluate([trackeval.datasets.MotChallenge2DBox(dcfg)],
                                           [trackeval.metrics.CLEAR(), trackeval.metrics.Identity()])
te = res["MotChallenge2DBox"]["mine"][seq_name]["pedestrian"]
print(f"predictions removed as distractor matches: {removed} of {len(tracks)}")
print(f"{'':22s}{'MOTA':>8s}{'IDF1':>8s}{'IDSW':>6s}{'FP':>6s}{'FN':>6s}")
print(f"{'TrackEval (official)':22s}{te['CLEAR']['MOTA']:8.4f}{te['Identity']['IDF1']:8.4f}{te['CLEAR']['IDSW']:6d}{te['CLEAR']['CLR_FP']:6d}{te['CLEAR']['CLR_FN']:6d}")
print(f"{'ours, official_filter':22s}{ours['mota']:8.4f}{ours['idf1']:8.4f}{ours['id_switches']:6d}{ours['false_positives']:6d}{ours['misses']:6d}")
print(f"{'ours, strict (old)':22s}{strict['mota']:8.4f}{strict['idf1']:8.4f}{strict['id_switches']:6d}{strict['false_positives']:6d}{strict['misses']:6d}")
