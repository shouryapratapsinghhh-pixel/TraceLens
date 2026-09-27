import numpy as np
import pytest

from tracelens.detect.detectors import SyntheticDetector
from tracelens.eval.summary import simulate_click
from tracelens.events.detect import (
    EventParams,
    Zone,
    detect_events,
    match_events,
    speed_body_heights,
)
from tracelens.reid.appearance import track_fingerprints
from tracelens.reid.relink import merge_chain, relink_chain
from tracelens.track.tracker import TrackOutput, run_tracker
from tracelens.video.synthetic import (
    INCIDENT_SIZE,
    RESTRICTED_ZONE,
    incident_gt_events,
    incident_scene,
    make_synthetic_video,
    point_in_polygon,
)

FPS = 10.0


def _track(xs, tid=1, y=100, h=80, start=0):
    """A track moving along x; one box per frame, 40 px wide, `h` tall."""
    return [TrackOutput(start + i, tid, np.array([x, y, x + 40, y + h], dtype=float)) for i, x in enumerate(xs)]


# -- geometry & speed ------------------------------------------------------------------


def test_point_in_polygon_square_and_concave():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon(5, 5, sq) and not point_in_polygon(15, 5, sq)
    notch = [(0, 0), (10, 0), (10, 10), (5, 5), (0, 10)]  # a concave "V" cut into the top
    assert point_in_polygon(2, 2, notch) and not point_in_polygon(5, 8, notch)


def test_speed_is_in_body_heights_per_second():
    tr = _track([8.0 * i for i in range(30)], h=80)  # 8 px/frame at 10 fps = 80 px/s = 1 height/s
    frames = np.array([t.frame for t in tr])
    boxes = np.array([t.box for t in tr])
    assert speed_body_heights(frames, boxes, FPS, window=5)[:20] == pytest.approx(1.0)


def test_speed_is_scale_invariant():
    """Same real motion seen at twice the size (closer to the camera) -> same speed."""
    far = _track([8.0 * i for i in range(30)], h=80)
    near = [TrackOutput(t.frame, 1, t.box * 2) for t in far]
    s = [speed_body_heights(np.array([t.frame for t in tr]), np.array([t.box for t in tr]), FPS, 5)[:20]
         for tr in (far, near)]
    np.testing.assert_allclose(s[0], s[1])


# -- individual events -----------------------------------------------------------------------


def test_dwell_and_run_detected():
    xs = [5.0 * i for i in range(20)]  # walk
    xs += [xs[-1]] * 50  # stand still 5 s
    xs += [xs[-1] + 16.0 * (i + 1) for i in range(20)]  # run 2 heights/s for 2 s
    ev = detect_events(_track(xs), 1, FPS)
    types = [e["type"] for e in ev]
    assert "dwell" in types and "run" in types
    dwell = next(e for e in ev if e["type"] == "dwell")
    assert dwell["t"] == pytest.approx(2.0, abs=0.6)


def test_normal_walking_raises_no_behaviour_events():
    ev = detect_events(_track([5.0 * i for i in range(100)]), 1, FPS)
    assert [e["type"] for e in ev] == ["appear", "disappear"]


def test_zone_crossing_and_border_jitter_do_not_flicker():
    zone = Zone("z", [(200, 0), (600, 0), (600, 400), (200, 400)])  # wide enough that the walker stays inside
    xs = [5.0 * i for i in range(80)]  # feet cross x=200 at about frame 36
    assert [e["type"] for e in detect_events(_track(xs), 1, FPS, zones=[zone])
            if e["type"].startswith("zone")] == ["zone_enter"]
    jitter = [178.0 + (4 if i % 2 else -4) for i in range(40)]  # feet hover at the border: 194 <-> 202
    events = detect_events(_track(jitter), 1, FPS, zones=[zone], params=EventParams(zone_confirm=3))
    assert not [e for e in events if e["type"].startswith("zone")]


def test_object_left_behind():
    target = _track([100.0] * 30 + [100.0 + 6 * (i + 1) for i in range(40)], tid=1, y=100)
    bag = [TrackOutput(f, 2, np.array([130, 162, 150, 180.0])) for f in range(10, 70)]
    ev = detect_events(target + bag, 1, FPS)
    left = [e for e in ev if e["type"] == "object_left_behind"]
    assert len(left) == 1 and left[0]["object_track"] == 2
    assert left[0]["t"] < left[0]["t_confirmed"]


def test_carried_object_is_not_left_behind():
    target = _track([100.0 + 6 * i for i in range(60)], tid=1, y=100)
    bag = [TrackOutput(f, 2, np.array([130 + 6 * f, 162, 150 + 6 * f, 180.0])) for f in range(60)]
    assert not [e for e in detect_events(target + bag, 1, FPS) if e["type"] == "object_left_behind"]


# -- matching ------------------------------------------------------------------------------


def test_match_events_hand_case():
    truth = [{"type": "run", "t": 5.0}, {"type": "dwell", "t": 10.0}]
    det = [{"type": "run", "t": 5.5}, {"type": "dwell", "t": 20.0}, {"type": "run", "t": 30.0}]
    m = match_events(det, truth, tolerance_s=2.0)
    assert (m["tp"], m["false_alarms"], m["missed"]) == (1, 2, 1)
    assert m["mean_timing_error_s"] == pytest.approx(0.5)


# -- the scripted incident, end to end ---------------------------------------------------------


@pytest.fixture(scope="module")
def incident():
    objs, script = incident_scene()
    v = make_synthetic_video(objs, n_frames=300, size=INCIDENT_SIZE, fps=FPS)
    return v, incident_gt_events(v.gt, script)


def test_ground_truth_zone_times_come_from_geometry(incident):
    _, truth = incident
    assert next(e for e in truth if e["type"] == "zone_enter")["t"] == pytest.approx(10.6)


def _pipeline(v, miss=0.0, jitter=0.0, seed=0, gt_id=1):
    tr = run_tracker(SyntheticDetector(v.gt, miss, jitter, frame_size=INCIDENT_SIZE, seed=seed), 300)
    tid = simulate_click(v.gt, tr, gt_id)
    chain, _ = relink_chain(tr, track_fingerprints(v.frames, tr, parts=2), tid)
    return detect_events(merge_chain(tr, chain), tid, FPS, zones=[Zone("restricted", RESTRICTED_ZONE)])


def test_incident_all_events_found_clean(incident):
    v, truth = incident
    m = match_events(_pipeline(v), truth)
    assert m["f1"] == 1.0 and m["mean_timing_error_s"] < 0.5


def test_incident_robust_to_moderate_noise(incident):
    v, truth = incident
    assert all(match_events(_pipeline(v, 0.1, 2.0, s), truth)["f1"] == 1.0 for s in range(3))


def test_innocent_walker_triggers_nothing(incident):
    v, _ = incident
    ev = _pipeline(v, 0.2, 3.0, seed=1, gt_id=3)
    assert not [e for e in ev if e["type"] not in ("appear", "disappear")]
