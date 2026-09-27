import numpy as np
import pytest

from tracelens.detect.detectors import SyntheticDetector
from tracelens.eval.summary import score_summary, simulate_click
from tracelens.summarize.clip import build_summary_frames
from tracelens.summarize.segments import (
    appearance_frames,
    frames_in,
    segments_from_frames,
    select_track_by_point,
)
from tracelens.summarize.timeline import build_timeline, region, timeline_text
from tracelens.track.tracker import TrackOutput, run_tracker
from tracelens.video.synthetic import busy_scene, make_synthetic_video, patrol_scene

# -- segments ----------------------------------------------------------------------


def test_segments_merge_small_gaps_and_split_big_ones():
    frames = [10, 11, 12, 15, 16, 40, 41, 42]
    assert segments_from_frames(frames, 100, max_gap=3, min_len=1, pad=0) == [(10, 16), (40, 42)]
    assert segments_from_frames(frames, 100, max_gap=0, min_len=1, pad=0) == [(10, 12), (15, 16), (40, 42)]


def test_segments_drop_tiny_fragments():
    assert segments_from_frames([5, 50, 51, 52, 53], 100, max_gap=0, min_len=3, pad=0) == [(50, 53)]


def test_padding_clips_to_video_and_remerges():
    assert segments_from_frames([1, 2, 3, 10, 11, 12], 14, max_gap=0, min_len=1, pad=3) == [(0, 13)]
    assert segments_from_frames([], 100) == []


def test_frames_in():
    np.testing.assert_array_equal(frames_in([(2, 4), (7, 7)]), [2, 3, 4, 7])


def test_select_by_point_prefers_smallest_box():
    tracks = [TrackOutput(5, 1, np.array([0, 0, 100, 100.0])), TrackOutput(5, 2, np.array([40, 40, 60, 60.0]))]
    assert select_track_by_point(tracks, 5, (50, 50)) == 2  # inside both: the specific (smaller) one
    assert select_track_by_point(tracks, 5, (10, 10)) == 1
    assert select_track_by_point(tracks, 5, (200, 200)) is None
    assert select_track_by_point(tracks, 6, (50, 50)) is None  # wrong frame


# -- timeline ------------------------------------------------------------------------


def test_region_grid():
    assert region(10, 10, 300, 300) == "top-left"
    assert region(150, 150, 300, 300) == "center"
    assert region(290, 290, 300, 300) == "bottom-right"


def test_timeline_facts_match_boxes():
    tracks = [TrackOutput(f, 7, np.array([10.0 * f, 50, 10.0 * f + 20, 90])) for f in range(10, 20)]
    e = build_timeline(tracks, 7, [(8, 21)], fps=10, frame_size=(320, 240))[0]
    assert (e["first_seen_frame"], e["last_seen_frame"]) == (10, 19)
    assert (e["start_s"], e["end_s"], e["duration_s"]) == (1.0, 1.9, 1.0)
    assert e["distance_px"] == pytest.approx(90.0)  # moved 10 px/frame for 9 steps
    assert "tracker ID 7" in timeline_text([e], 7)


# -- end-to-end summaries ---------------------------------------------------------------


def test_target_always_in_view_is_fully_recalled():
    v = make_synthetic_video(busy_scene(), n_frames=100)
    tracks = run_tracker(SyntheticDetector(v.gt), 100)
    m = score_summary(v.gt, tracks, gt_id=2, n_frames=100)
    assert m["target_recall"] == 1.0 and m["n_track_ids"] == 1
    assert m["compression"] < 1.0


def test_gap_merging_absorbs_detector_blinks():
    v = make_synthetic_video(busy_scene(), n_frames=100)
    tracks = run_tracker(SyntheticDetector(v.gt, miss_rate=0.3, seed=3), 100, max_age=8)
    merged = score_summary(v.gt, tracks, 2, 100, max_gap=5)
    unmerged = score_summary(v.gt, tracks, 2, 100, max_gap=0, pad=0)
    assert merged["target_recall"] > unmerged["target_recall"]


def test_leave_and_return_loses_the_second_visit():
    """Documents the known limitation: the target returns under a NEW track ID,
    so a one-ID summary keeps only one visit. The oracle shows what re-linking
    the IDs would recover."""
    v = make_synthetic_video(patrol_scene(), n_frames=300)
    tracks = run_tracker(SyntheticDetector(v.gt), 300)
    m = score_summary(v.gt, tracks, gt_id=1, n_frames=300)
    assert m["n_track_ids"] == 2
    assert m["target_recall"] == pytest.approx(0.5, abs=0.05)
    assert m["oracle_relink_recall"] == 1.0


def test_summary_clip_length_matches_segments():
    v = make_synthetic_video(busy_scene(), n_frames=100)
    tracks = run_tracker(SyntheticDetector(v.gt), 100)
    tid = simulate_click(v.gt, tracks, 2)
    segs = segments_from_frames(appearance_frames(tracks, tid), 100)
    out = build_summary_frames(v.frames, tracks, tid, segs, v.fps)
    assert len(out) == sum(b - a + 1 for a, b in segs)
    assert out[0].shape == v.frames[0].shape


def test_click_on_empty_space_selects_nothing():
    v = make_synthetic_video(busy_scene(), n_frames=100)
    assert simulate_click(v.gt, [], gt_id=2) is None
