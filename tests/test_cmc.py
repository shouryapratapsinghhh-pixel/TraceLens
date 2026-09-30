import cv2
import numpy as np
import pytest

from tracelens.detect.detectors import SyntheticDetector
from tracelens.eval.mot import evaluate_tracking
from tracelens.track.cmc import IDENTITY, estimate_warp, sequence_warps
from tracelens.track.tracker import KalmanBox, run_tracker
from tracelens.video.synthetic import busy_scene, make_shaky_video


def _texture(h=300, w=380, seed=0):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 90, (h, w, 3)).astype(np.uint8), (0, 0), 2.0)


def test_recovers_a_known_camera_shift():
    tex = _texture()
    a, b = tex[30:270, 30:350], tex[25:265, 42:362]  # camera moved +12 x, -5 y -> content moves -12, +5
    W = estimate_warp(a, b)
    np.testing.assert_allclose(W[:, 2], [-12, 5], atol=0.3)
    np.testing.assert_allclose(W[:, :2], np.eye(2), atol=0.01)


def test_featureless_frame_falls_back_to_identity():
    flat = np.full((240, 320, 3), 40, np.uint8)
    np.testing.assert_array_equal(estimate_warp(flat, flat), IDENTITY)


def test_masking_detections_fixes_corner_starvation():
    """Regression: high-contrast objects out-scored the background texture, so corners were
    found only ON the moving objects and the estimate fell back to identity."""
    v, off = make_shaky_video(busy_scene(), n_frames=20, shake_px=20)
    det = SyntheticDetector(v.gt, seed=1)
    boxes = {f: np.array([d.box for d in det.detect(f)]).reshape(-1, 4) for f in range(20)}
    errs = [np.abs(estimate_warp(v.frames[f - 1], v.frames[f], exclude_boxes=boxes[f - 1])[:, 2]
                   + (np.round(off[f]) - np.round(off[f - 1]))).max() for f in range(1, 20)]
    assert np.median(errs) < 1.0


def test_kalman_moves_with_the_camera():
    kf = KalmanBox(np.array([100, 100, 140, 180.0]))
    kf.x[4:6] = [3.0, 0.0]  # moving right at 3 px/frame
    A = np.array([[1.1, 0.0, 10.0], [0.0, 1.1, -4.0]])  # zoom 1.1 and shift
    kf.apply_affine(A)
    cx, cy, w, h = kf.x[:4]
    assert (cx, cy) == pytest.approx((1.1 * 120 + 10, 1.1 * 140 - 4))
    assert (w, h) == pytest.approx((44.0, 88.0))  # size scales with the zoom
    assert kf.x[4] == pytest.approx(3.3)  # velocity scales too


def test_cmc_rescues_tracking_on_a_shaking_camera():
    v, _ = make_shaky_video(busy_scene(), n_frames=100, shake_px=20)
    warps = sequence_warps(v.frames, detector=SyntheticDetector(v.gt, miss_rate=0.1, seed=99))
    base = [evaluate_tracking(v.gt, run_tracker(SyntheticDetector(v.gt, miss_rate=0.1, seed=s), 100))
            for s in range(3)]
    cmc = [evaluate_tracking(v.gt, run_tracker(SyntheticDetector(v.gt, miss_rate=0.1, seed=s), 100, warps=warps))
           for s in range(3)]
    assert np.mean([m["idf1"] for m in cmc]) > np.mean([m["idf1"] for m in base]) + 0.2
    assert np.mean([m["id_switches"] for m in cmc]) < np.mean([m["id_switches"] for m in base]) / 2


def test_sequence_warps_are_cached(tmp_path, monkeypatch):
    v, _ = make_shaky_video(busy_scene(), n_frames=5, shake_px=10)
    path = tmp_path / "w.npy"
    first = sequence_warps(v.frames, cache_path=path)
    from tracelens.track import cmc

    monkeypatch.setattr(cmc, "estimate_warp", lambda *a, **k: pytest.fail("recomputed despite cache"))
    np.testing.assert_array_equal(cmc.sequence_warps(v.frames, cache_path=path), first)
