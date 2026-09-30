import numpy as np
import pytest

from tracelens.detect.detectors import Detection, SyntheticDetector
from tracelens.eval.mot import clear_mot, evaluate_tracking, idf1
from tracelens.track.tracker import KalmanBox, Tracker, TrackOutput, iou_matrix, run_tracker
from tracelens.video.synthetic import (
    SyntheticObject,
    busy_scene,
    crossing_scene,
    make_synthetic_video,
)

# -- synthetic video ----------------------------------------------------------


def test_gt_boxes_match_drawn_pixels():
    v = make_synthetic_video([SyntheticObject(1, (50, 40), (3, 2), color=(0, 0, 255))], n_frames=20)
    assert len(v.frames) == 20 and v.frames[0].shape == (240, 320, 3)
    for f, _, x1, y1, x2, y2 in v.gt:
        cy, cx = int((y1 + y2) / 2), int((x1 + x2) / 2)
        assert tuple(v.frames[f][cy, cx]) == (0, 0, 255)  # the object is where GT says


def test_enter_exit_and_bounce():
    v = make_synthetic_video([SyntheticObject(7, (10, 10), (25, 0), enter=5, exit=15)], n_frames=30)
    frames = sorted({r[0] for r in v.gt})
    assert frames == list(range(5, 15))
    assert all(0 <= r[2] and r[4] <= 320 for r in v.gt)  # bounces, never leaves the frame


# -- detector -------------------------------------------------------------------


def test_synthetic_detector_miss_rate():
    v = make_synthetic_video(busy_scene(), n_frames=100)
    det = SyntheticDetector(v.gt, miss_rate=0.3, seed=0)
    n_det = sum(len(det.detect(f)) for f in range(100))
    assert 0.6 < n_det / len(v.gt) < 0.8


# -- tracker --------------------------------------------------------------------


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])


def test_kalman_learns_velocity():
    kf = KalmanBox(np.array([0, 0, 10, 20]))
    for t in range(1, 15):
        kf.predict()
        kf.update(np.array([5 * t, 0, 5 * t + 10, 20]))
    predicted = kf.predict()
    assert predicted[0] == pytest.approx(75, abs=1.5)  # next position, extrapolated from learned velocity


def test_perfect_detections_give_perfect_scores():
    v = make_synthetic_video(busy_scene(), n_frames=100)
    pred = run_tracker(SyntheticDetector(v.gt), 100, min_hits=1)
    m = evaluate_tracking(v.gt, pred)
    assert m["mota"] == 1.0 and m["idf1"] == 1.0 and m["id_switches"] == 0


def test_tracks_survive_a_blinking_detector():
    v = make_synthetic_video(busy_scene(), n_frames=100)
    pred = run_tracker(SyntheticDetector(v.gt, miss_rate=0.2, jitter=1.0, seed=1), 100, max_age=5)
    m = evaluate_tracking(v.gt, pred)
    assert m["idf1"] > 0.75
    assert m["id_switches"] <= 2


def test_min_hits_suppresses_false_alarms():
    v = make_synthetic_video(busy_scene(), n_frames=100)
    noisy = SyntheticDetector(v.gt, false_positive_rate=0.5, seed=2)
    loose = evaluate_tracking(v.gt, run_tracker(noisy, 100, min_hits=1))
    noisy = SyntheticDetector(v.gt, false_positive_rate=0.5, seed=2)
    strict = evaluate_tracking(v.gt, run_tracker(noisy, 100, min_hits=3))
    assert strict["false_positives"] < loose["false_positives"]


def test_motion_prediction_prevents_id_switch_on_crossing():
    """At the crossing frame both objects sit in exactly the same place. A
    no-motion tracker faces a coin flip on the next frame; the Kalman filter
    knows which way each object was moving. Over 8 seeds (detection order
    shuffled), motion must never switch, and no-motion must switch sometimes."""
    v = make_synthetic_video(crossing_scene(), n_frames=60)
    motion_sw = no_motion_sw = 0
    for seed in range(8):
        motion_sw += evaluate_tracking(
            v.gt, run_tracker(SyntheticDetector(v.gt, seed=seed), 60, use_motion=True))["id_switches"]
        no_motion_sw += evaluate_tracking(
            v.gt, run_tracker(SyntheticDetector(v.gt, seed=seed), 60, use_motion=False))["id_switches"]
    assert motion_sw == 0
    assert no_motion_sw > 0


# -- metrics: hand cases + agreement with motmetrics --------------------------------


def test_metrics_hand_case_id_switch():
    gt = [(f, 1, 0.0, 0.0, 10.0, 10.0) for f in range(4)]
    pred = [TrackOutput(f, 1 if f < 2 else 2, np.array([0, 0, 10, 10.0])) for f in range(4)]
    m = clear_mot(gt, pred)
    assert m["id_switches"] == 1 and m["mota"] == pytest.approx(0.75)
    assert idf1(gt, pred)["idf1"] == pytest.approx(0.5)  # best identity mapping covers 2 of 4 frames


def _to_motmetrics(gt, pred):
    import motmetrics as mm

    # motmetrics 1.4.0 still calls np.asfarray, which NumPy 2.0 removed. Test-only shim
    # for the REFERENCE library; our own metric code doesn't need it.
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

    acc = mm.MOTAccumulator(auto_id=False)
    frames = sorted({r[0] for r in gt} | {p.frame for p in pred})
    for f in frames:
        g = [r for r in gt if r[0] == f]
        p = [q for q in pred if q.frame == f]
        gb = np.array([[r[2], r[3], r[4] - r[2], r[5] - r[3]] for r in g]).reshape(-1, 4)
        pb = np.array([[q.box[0], q.box[1], q.box[2] - q.box[0], q.box[3] - q.box[1]] for q in p]).reshape(-1, 4)
        acc.update([r[1] for r in g], [q.track_id for q in p],
                   mm.distances.iou_matrix(gb, pb, max_iou=0.5), frameid=f)
    mh = mm.metrics.create()
    return mh.compute(acc, metrics=["mota", "idf1", "num_switches", "num_misses", "num_false_positives"])


@pytest.mark.parametrize("seed", range(4))
def test_metrics_match_motmetrics(seed):
    v = make_synthetic_video(busy_scene(), n_frames=80)
    det = SyntheticDetector(v.gt, miss_rate=0.25, jitter=4.0, false_positive_rate=0.3, seed=seed)
    pred = run_tracker(det, 80, min_hits=2, max_age=3, iou_threshold=0.2)
    ours = evaluate_tracking(v.gt, pred)
    ref = _to_motmetrics(v.gt, pred).iloc[0]
    assert ours["mota"] == pytest.approx(ref["mota"], abs=1e-9)
    assert ours["idf1"] == pytest.approx(ref["idf1"], abs=1e-9)
    assert ours["id_switches"] == ref["num_switches"]
    assert ours["misses"] == ref["num_misses"]
    assert ours["false_positives"] == ref["num_false_positives"]


def test_detection_dataclass_defaults():
    d = Detection(box=np.array([0, 0, 1, 1.0]))
    assert d.cls == "person" and d.score == 1.0


def test_tracker_handles_empty_frames():
    t = Tracker()
    assert t.update([]) == []
    assert t.update([Detection(np.array([0, 0, 10, 10.0]))]) == []  # tentative, not yet confirmed


def test_ids_are_consecutive_despite_false_alarms():
    v = make_synthetic_video(busy_scene(), n_frames=100)
    det = SyntheticDetector(v.gt, false_positive_rate=0.5, seed=2)
    ids = sorted({t.track_id for t in run_tracker(det, 100, min_hits=3)})
    assert ids == list(range(1, len(ids) + 1))  # no gaps from discarded false alarms


# -- ByteTrack association ----------------------------------------------------------------


def _box(x):
    return np.array([x, 10, x + 40, 90.0])


def test_bytetrack_weak_detection_keeps_a_confirmed_track_alive():
    """Frames 0-4 strong detections confirm the track; frames 5-9 the same person gets only
    weak (0.3) detections, like a partial occlusion. ByteTrack keeps following it."""
    seq = [[Detection(_box(10 + 3 * f), score=0.9 if f < 5 else 0.3)] for f in range(10)]
    t = Tracker(byte=True, high_thresh=0.5, low_thresh=0.1, max_age=1)
    out = [t.update(d) for d in seq]
    assert all(len(o) == 1 and o[0].track_id == 1 for o in out[3:])  # one identity throughout


def test_bytetrack_weak_detections_never_start_a_track():
    t = Tracker(byte=True, high_thresh=0.5, low_thresh=0.1)
    out = [t.update([Detection(_box(100), score=0.3)]) for _ in range(10)]
    assert all(o == [] for o in out)


def test_bytetrack_drops_detections_below_low_threshold():
    t = Tracker(byte=True, high_thresh=0.5, low_thresh=0.1, min_hits=1)
    t.update([Detection(_box(10), score=0.9)])
    assert t.update([Detection(_box(12), score=0.05)]) == []  # too weak even to extend
