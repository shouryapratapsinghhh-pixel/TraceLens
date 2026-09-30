import numpy as np
import pytest

from tracelens.detect.detectors import SyntheticDetector
from tracelens.eval.summary import majority_gt_id, score_summary
from tracelens.reid.appearance import fingerprint, similarity
from tracelens.reid.relink import merge_chain, relink_chain, track_spans
from tracelens.track.tracker import TrackOutput, run_tracker
from tracelens.video.synthetic import lookalike_scene, make_synthetic_video, patrol_scene

RED, BLUE = (0, 0, 255), (255, 0, 0)


def _two_tone(top, bottom):
    img = np.zeros((80, 40, 3), dtype=np.uint8)
    img[:40], img[40:] = top, bottom
    return img


BOX = np.array([0, 0, 40, 80.0])

# -- fingerprints --------------------------------------------------------------------


def test_self_similarity_is_one():
    a = fingerprint(_two_tone(RED, BLUE), BOX)
    assert similarity(a, a) == pytest.approx(1.0)


def test_whole_box_cannot_tell_swapped_colours_but_parts_can():
    a, b = _two_tone(RED, BLUE), _two_tone(BLUE, RED)
    assert similarity(fingerprint(a, BOX, parts=1), fingerprint(b, BOX, parts=1)) == pytest.approx(1.0)
    assert similarity(fingerprint(a, BOX, parts=2), fingerprint(b, BOX, parts=2)) == pytest.approx(0.0)


def test_empty_crop_does_not_crash():
    fp = fingerprint(_two_tone(RED, BLUE), np.array([500, 500, 600, 600.0]))
    assert np.isfinite(fp).all()


# -- re-linking rules (hand-built fingerprints) -------------------------------------------


def _fp(v):
    v = np.asarray(v, dtype=float)
    return (v / v.sum())[None, :]


def _tracks(spans):
    return [TrackOutput(f, tid, np.array([0, 0, 1, 1.0])) for tid, (a, b) in spans.items() for f in range(a, b + 1)]


def test_never_links_a_track_that_overlaps_in_time():
    tracks = _tracks({1: (0, 10), 2: (5, 20), 3: (15, 30)})  # 2 overlaps 1: can't be the same object
    fps = {1: _fp([1, 0]), 2: _fp([1, 0]), 3: _fp([1, 0])}
    chain, _ = relink_chain(tracks, fps, 1, threshold=0.5)
    assert 2 not in chain and chain == [1, 3]


def test_chain_follows_several_returns():
    tracks = _tracks({1: (0, 10), 2: (20, 30), 3: (40, 50)})
    fps = {k: _fp([1, 0]) for k in (1, 2, 3)}
    assert relink_chain(tracks, fps, 1)[0] == [1, 2, 3]


def test_threshold_and_gap_stop_the_chain():
    tracks = _tracks({1: (0, 10), 2: (20, 30)})
    assert relink_chain(tracks, {1: _fp([1, 0]), 2: _fp([0, 1])}, 1)[0] == [1]  # dissimilar
    assert relink_chain(tracks, {1: _fp([1, 0]), 2: _fp([1, 0])}, 1, max_gap_frames=5)[0] == [1]  # too late


def test_compares_against_original_target_to_prevent_drift():
    """B is fairly similar to target A; C is similar to B but NOT to A. Comparing
    each candidate to the latest link would walk A -> B -> C. We compare to A."""
    tracks = _tracks({1: (0, 10), 2: (20, 30), 3: (40, 50)})
    fps = {1: _fp([1, 0, 0]), 2: _fp([1, 1, 0]), 3: _fp([0, 1, 1])}
    chain, log = relink_chain(tracks, fps, 1, threshold=0.7)
    assert chain == [1, 2]
    assert any(e["candidate"] == 3 and not e["linked"] for e in log)


def test_merge_chain_relabels():
    tracks = _tracks({1: (0, 2), 2: (5, 6), 3: (0, 1)})
    merged = merge_chain(tracks, [1, 2])
    assert {t.track_id for t in merged} == {1, 3}
    assert track_spans(merged)[1] == (0, 6)


# -- end to end ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lookalike():
    v = make_synthetic_video(lookalike_scene(), n_frames=300)
    return v, run_tracker(SyntheticDetector(v.gt, jitter=2, seed=0), 300)


def test_part_based_relinking_recovers_return_without_wrong_merge(lookalike):
    v, tracks = lookalike
    m = score_summary(v.gt, tracks, 1, 300, frames=v.frames, relink="parts")
    assert m["target_recall"] >= 0.95 and m["wrong_links"] == 0 and m["correct_links"] == 1


def test_whole_box_relinking_merges_the_lookalike(lookalike):
    """Documents WHY parts matter: on this run the whole-box fingerprint merges the
    swapped-colour look-alike -- recall still looks perfect; precision reveals it."""
    v, tracks = lookalike
    whole = score_summary(v.gt, tracks, 1, 300, frames=v.frames, relink="whole")
    parts = score_summary(v.gt, tracks, 1, 300, frames=v.frames, relink="parts")
    assert whole["wrong_links"] == 1
    assert whole["target_recall"] == pytest.approx(parts["target_recall"])  # recall can't see the error
    assert whole["target_precision"] < parts["target_precision"]  # precision can


def test_patrol_relinking_restores_recall():
    v = make_synthetic_video(patrol_scene(), n_frames=300)
    tracks = run_tracker(SyntheticDetector(v.gt, jitter=2, seed=0), 300)
    none = score_summary(v.gt, tracks, 1, 300)
    parts = score_summary(v.gt, tracks, 1, 300, frames=v.frames, relink="parts")
    assert none["target_recall"] == pytest.approx(0.5, abs=0.05)
    assert parts["target_recall"] >= 0.95 and parts["wrong_links"] == 0


def test_relink_requires_frames(lookalike):
    v, tracks = lookalike
    with pytest.raises(ValueError, match="frames"):
        score_summary(v.gt, tracks, 1, 300, relink="parts")


def test_majority_gt_id(lookalike):
    v, tracks = lookalike
    tid = next(t.track_id for t in tracks if t.frame == 40 and t.box[0] < 150)
    assert majority_gt_id(v.gt, tracks, tid) == 1


def test_links_fragments_in_time_order_not_by_best_similarity():
    """Regression: a target broken into several fragments. Jumping to the most
    similar (but LATER) fragment would strand every fragment in between."""
    tracks = _tracks({1: (0, 10), 2: (20, 30), 3: (40, 50), 4: (60, 70)})
    fps = {1: _fp([1, 0.00]), 2: _fp([1, 0.05]), 3: _fp([1, 0.04]), 4: _fp([1, 0.0])}  # 4 = most similar
    assert relink_chain(tracks, fps, 1, threshold=0.8)[0] == [1, 2, 3, 4]


# -- motion gate and ambiguity check (added after the first real-footage run) ----------------


def _placed(spans_boxes):
    """{tid: ((start, end), (x_start, x_end))}: a 40x80 box moving from x_start to x_end."""
    out = []
    for tid, ((a, b), (xa, xb)) in spans_boxes.items():
        for f in range(a, b + 1):
            x = xa + (xb - xa) * (f - a) / max(b - a, 1)
            out.append(TrackOutput(f, tid, np.array([x, 100, x + 40, 180.0])))
    return out


def test_motion_gate_rejects_a_candidate_too_far_to_walk_to():
    # target ends at x=100 at frame 10; candidate 2 starts 1 s later at x=900 (10 body-heights away)
    tracks = _placed({1: ((0, 10), (100, 100)), 2: ((20, 30), (900, 900))})
    fps = {1: _fp([1, 0]), 2: _fp([1, 0])}  # identical appearance
    assert relink_chain(tracks, fps, 1)[0] == [1, 2]  # appearance alone links it
    chain, log = relink_chain(tracks, fps, 1, fps=10, max_speed_bh=2.0)
    assert chain == [1]
    assert any(e.get("rejected") == "too far to walk" for e in log)


def test_motion_gate_accepts_a_reachable_candidate():
    tracks = _placed({1: ((0, 10), (100, 100)), 2: ((20, 30), (180, 180))})  # 1 body-height away after 1 s
    fps = {1: _fp([1, 0]), 2: _fp([1, 0])}
    assert relink_chain(tracks, fps, 1, fps=10, max_speed_bh=2.0)[0] == [1, 2]


def test_ambiguity_check_abstains_between_simultaneous_lookalikes():
    tracks = _placed({1: ((0, 10), (100, 100)), 2: ((20, 40), (120, 120)), 3: ((22, 40), (300, 300))})
    fps = {1: _fp([1, 0, 0]), 2: _fp([1, 0.05, 0]), 3: _fp([1, 0.04, 0])}  # 2 and 3 look almost the same
    chain, log = relink_chain(tracks, fps, 1, threshold=0.8, margin=0.05)
    assert chain == [1]
    assert any("abstained" in e for e in log)
    assert relink_chain(tracks, fps, 1, threshold=0.8, margin=0.0)[0][1] in (2, 3)  # no margin: it guesses


def test_clear_winner_among_simultaneous_candidates_is_linked():
    tracks = _placed({1: ((0, 10), (100, 100)), 2: ((20, 40), (120, 120)), 3: ((22, 40), (300, 300))})
    fps = {1: _fp([1, 0, 0]), 2: _fp([1, 1, 0]), 3: _fp([1, 0.01, 0])}  # 3 is clearly the target
    assert relink_chain(tracks, fps, 1, threshold=0.6, margin=0.05)[0] == [1, 3]


def test_gate_keeps_the_synthetic_lookalike_result():
    v = make_synthetic_video(lookalike_scene(), n_frames=300)
    tracks = run_tracker(SyntheticDetector(v.gt, jitter=2, seed=0), 300)
    m = score_summary(v.gt, tracks, 1, 300, frames=v.frames, relink="parts",
                      relink_kwargs={"fps": 10, "max_speed_bh": 1.0, "margin": 0.05})
    assert m["target_recall"] >= 0.95 and m["wrong_links"] == 0


# -- identity metrics (added after frame-level summary F1 was found blind to wrong identities) --


def test_identity_metrics_catch_a_stranger_that_frame_metrics_miss():
    """Target (GT 1) is on screen all 20 frames. The summary follows the target for frames
    0-9, then a stranger (GT 2, elsewhere in the same frames) for 10-19."""
    from tracelens.eval.summary import identity_scores

    gt = [(f, 1, 0.0, 0.0, 40.0, 80.0) for f in range(20)] + [(f, 2, 200.0, 0.0, 240.0, 80.0) for f in range(20)]
    tracks = ([TrackOutput(f, 10, np.array([0, 0, 40, 80.0])) for f in range(10)]
              + [TrackOutput(f, 11, np.array([200, 0, 240, 80.0])) for f in range(10, 20)])
    m = score_summary(gt, tracks, 1, 20, max_gap=0, pad=0)  # summary follows track 10 only
    assert m["identity_precision"] == 1.0 and m["identity_recall"] == pytest.approx(0.5)
    idr, idp = identity_scores(gt, gt, tracks, {10, 11}, 1)  # a chain that wrongly merged the stranger
    assert idr == pytest.approx(0.5) and idp == pytest.approx(0.5)  # half the highlighted boxes are the stranger
    frames_kept = set(range(20))
    frame_precision = len(frames_kept & {r[0] for r in gt if r[1] == 1}) / len(frames_kept)
    assert frame_precision == 1.0  # the frame-level view sees nothing wrong: that was the flaw


def test_box_on_an_occluded_target_is_not_a_mistake():
    from tracelens.eval.summary import identity_scores

    gt_all = [(f, 1, 0.0, 0.0, 40.0, 80.0) for f in range(10)]
    gt_visible = gt_all[:5]  # frames 5-9: target too occluded to count as "visible"
    tracks = [TrackOutput(f, 10, np.array([0, 0, 40, 80.0])) for f in range(10)]
    idr, idp = identity_scores(gt_visible, gt_all, tracks, {10}, 1)
    assert idr == 1.0 and idp == 1.0  # boxes on the hidden target in frames 5-9 still count as correct
