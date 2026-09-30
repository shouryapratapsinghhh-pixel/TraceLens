import sys

import cv2
import numpy as np
import pytest

from tracelens.reid.appearance import LazyFingerprints
from tracelens.track.tracker import TrackOutput
from tracelens.video.mot import LazyFrames, MOTSequence, PublicDetector
from tracelens.video.synthetic import lookalike_scene, make_synthetic_video


def write_fake_mot(seq_dir, n_frames=300):
    """A MOTChallenge-format sequence from the synthetic lookalike scene, including rows
    the loader must filter out: a reflection (class 7), an ignored box (mark 0), and a
    heavily occluded pedestrian (visibility 0.1)."""
    v = make_synthetic_video(lookalike_scene(), n_frames=n_frames)
    (seq_dir / "img1").mkdir(parents=True)
    (seq_dir / "gt").mkdir()
    (seq_dir / "det").mkdir()
    for i, img in enumerate(v.frames, 1):
        cv2.imwrite(str(seq_dir / "img1" / f"{i:06d}.jpg"), img)
    (seq_dir / "seqinfo.ini").write_text(
        f"[Sequence]\nname={seq_dir.name}\nimDir=img1\nframeRate=10\nseqLength={n_frames}\n"
        "imWidth=320\nimHeight=240\nimExt=.jpg\n")
    gt_lines, det_lines = [], []
    for f, oid, x1, y1, x2, y2 in v.gt:
        w, h = x2 - x1, y2 - y1
        gt_lines.append(f"{f + 1},{oid},{x1},{y1},{w},{h},1,1,1.0")
        det_lines.append(f"{f + 1},-1,{x1},{y1},{w},{h},0.9,-1,-1,-1")
    gt_lines += ["1,90,5,5,10,10,1,7,1.0",     # reflection: class 7
                 "1,91,5,5,10,10,0,1,1.0",     # mark 0: ignored
                 "1,92,5,5,10,10,1,1,0.1"]     # pedestrian, only 10% visible
    det_lines += ["1,-1,200,200,20,20,0.2,-1,-1,-1"]  # low-confidence detection
    (seq_dir / "gt" / "gt.txt").write_text("\n".join(gt_lines) + "\n")
    (seq_dir / "det" / "det.txt").write_text("\n".join(det_lines) + "\n")
    return v


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    root = tmp_path_factory.mktemp("MOT17") / "train"
    seq_dir = root / "FAKE-01-FRCNN"
    v = write_fake_mot(seq_dir)
    return root, MOTSequence(seq_dir), v


def test_seqinfo_parsed(fake):
    _, seq, _ = fake
    assert (seq.name, seq.fps, seq.n_frames, seq.size) == ("FAKE-01-FRCNN", 10.0, 300, (320, 240))


def test_ground_truth_filters_and_is_zero_based(fake):
    _, seq, v = fake
    gt = seq.ground_truth()
    ids = {r[1] for r in gt}
    assert 90 not in ids and 91 not in ids  # reflection + ignored rows dropped
    assert 92 in ids  # occluded pedestrian kept by default...
    assert 92 not in {r[1] for r in seq.ground_truth(min_visibility=0.25)}  # ...but not when visibility matters
    assert min(r[0] for r in gt) == 0  # MOT frame 1 -> our frame 0
    assert sorted(r for r in gt if r[1] != 92) == sorted(v.gt)


def test_public_detector_confidence_and_frames(fake):
    _, seq, v = fake
    det = PublicDetector(seq, min_conf=0.5)
    assert len(det.detect(0)) == len(v.gt_for_frame(0))  # the 0.2-confidence box is dropped
    assert len(PublicDetector(seq, min_conf=0.0).detect(0)) == len(v.gt_for_frame(0)) + 1


def test_lazy_frames(fake):
    _, seq, v = fake
    assert len(seq.frames) == 300
    assert np.abs(seq.frames[40].astype(int) - v.frames[40].astype(int)).mean() < 3  # JPEG is lossy
    assert seq.frames[40] is seq.frames[40]  # cached
    with pytest.raises(IndexError):
        seq.frames[300]


def test_lazy_frames_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        LazyFrames([tmp_path / "nope.jpg"])[0]


def test_lazy_fingerprints_compute_only_what_is_asked(fake):
    _, seq, _ = fake
    tracks = [TrackOutput(f, tid, np.array([10, 10, 50, 90.0])) for tid in (1, 2, 3) for f in range(5)]
    fps = LazyFingerprints(seq.frames, tracks)
    assert len(fps) == 0
    _ = fps[2]
    assert list(fps) == [2]


def test_mot_eval_end_to_end(fake, tmp_path, monkeypatch):
    import pandas as pd

    from tracelens import mot_eval

    root, _, _ = fake
    out = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["x", "--root", str(root), "--targets", "1", "--max-age", "5",
                                      "--out", str(out)])
    mot_eval.main()
    track = pd.read_csv(out / "tracking.csv")
    assert track.loc[0, "mota"] > 0.9  # public detections = ground truth here
    t = pd.read_csv(out / "targets.csv").set_index("relink")
    assert t.loc["parts", "target_recall"] > t.loc["none", "target_recall"]  # re-linking recovers the return
    assert t.loc["parts", "wrong_links"] == 0  # and doesn't merge the look-alike


def test_pick_targets_prefers_people_who_leave_view():
    from tracelens.mot_eval import pick_targets

    gt = [(f, 1, 0, 0, 1, 1) for f in list(range(50)) + list(range(80, 130))]  # 100 frames, one 30-frame gap
    gt += [(f, 2, 0, 0, 1, 1) for f in range(200)]  # 200 frames, always visible
    assert pick_targets(gt, 1) == [2]  # longest overall
    assert pick_targets(gt, 1, min_gap=10) == [1]  # longest among those who left view


def test_relink_study_end_to_end(tmp_path, monkeypatch):
    import pandas as pd

    from tracelens import relink_study

    root = tmp_path / "train"
    write_fake_mot(root / "FAKE-01-FRCNN")
    write_fake_mot(root / "FAKE-02-FRCNN")
    monkeypatch.setattr(relink_study, "GRID", {"threshold": [0.8, 0.95], "max_speed_bh": [None, 1.0],
                                               "margin": [0.0]})  # small grid: fast test
    out = tmp_path / "study"
    monkeypatch.setattr(sys, "argv", ["x", "--root", str(root), "--tune", "FAKE-01-FRCNN", "--test",
                                      "FAKE-02-FRCNN", "--targets", "1", "--max-age", "5", "--out", str(out)])
    relink_study.main()
    grid = pd.read_csv(out / "tune_grid.csv")
    assert len(grid) == 4
    res = pd.read_csv(out / "test_results.csv")
    assert set(res["method"]) == {"no re-linking", "original re-linker", "tuned re-linker"}
    assert set(res["sequence"]) == {"FAKE-02-FRCNN"}  # test results never include the tune sequence


def test_relink_study_refuses_overlapping_split(tmp_path, monkeypatch):
    from tracelens import relink_study

    monkeypatch.setattr(sys, "argv", ["x", "--root", str(tmp_path), "--tune", "A", "--test", "A"])
    with pytest.raises(SystemExit, match="both"):
        relink_study.main()
