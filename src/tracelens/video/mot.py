"""MOTChallenge (MOT17) sequences: lazy frames, ground truth, public detections.

Layout of one sequence (e.g. data/raw/MOT17/train/MOT17-02-FRCNN/):
  seqinfo.ini   frameRate, seqLength, imWidth, imHeight, imDir
  img1/         000001.jpg, 000002.jpg, ...
  gt/gt.txt     frame, id, left, top, width, height, mark, class, visibility
  det/det.txt   frame, -1, left, top, width, height, confidence, ...

Conventions here:
  - MOT frames start at 1; everything in this repo is 0-based, so frame - 1.
  - Ground truth keeps class 1 (pedestrian) with mark 1 only.
  - official_filter() applies the official MOTChallenge preprocessing (read from
    TrackEval's mot_challenge_2d_box.py): predictions matched to distractor classes
    (person on vehicle, static person, distractor, reflection) are removed, not
    counted as false positives. Without it, scoring is stricter than the official one.
  - Frames are read from disk on demand (1080p sequences are several GB in RAM).
"""

from __future__ import annotations

import configparser
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from tracelens.detect.detectors import Detection
from tracelens.track.tracker import TrackOutput, iou_matrix


class LazyFrames:
    """Sequence-like: frames[i] reads image i from disk (with a small LRU cache)."""

    def __init__(self, paths: list[Path], cache: int = 64):
        self.paths = paths
        self._read = lru_cache(maxsize=cache)(self._load)

    def _load(self, i: int) -> np.ndarray:
        img = cv2.imread(str(self.paths[i]))
        if img is None:
            raise FileNotFoundError(self.paths[i])
        return img

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> np.ndarray:
        if not 0 <= i < len(self.paths):
            raise IndexError(i)
        return self._read(i)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class MOTSequence:
    def __init__(self, seq_dir: str | Path):
        self.dir = Path(seq_dir)
        ini = self.dir / "seqinfo.ini"
        if not ini.exists():
            raise FileNotFoundError(f"{ini} not found -- is this a MOTChallenge sequence folder?")
        cfg = configparser.ConfigParser()
        cfg.read(ini)
        s = cfg["Sequence"]
        self.name = s.get("name", self.dir.name)
        self.fps = float(s.get("frameRate", 30))
        self.n_frames = int(s["seqLength"])
        self.size = (int(s["imWidth"]), int(s["imHeight"]))
        im_dir = self.dir / s.get("imDir", "img1")
        ext = s.get("imExt", ".jpg")
        self.frames = LazyFrames([im_dir / f"{i:06d}{ext}" for i in range(1, self.n_frames + 1)])

    def raw_ground_truth(self) -> list[tuple]:
        """Every GT row, all classes and marks: (frame0, id, x1, y1, x2, y2, mark, class)."""
        rows = np.loadtxt(self.dir / "gt" / "gt.txt", delimiter=",", ndmin=2)
        return [(int(r[0]) - 1, int(r[1]), float(r[2]), float(r[3]), float(r[2] + r[4]), float(r[3] + r[5]),
                 int(r[6]), int(r[7])) for r in rows]

    def ground_truth(self, min_visibility: float = 0.0) -> list[tuple]:
        """GTRow tuples (frame, id, x1, y1, x2, y2), pedestrians only, 0-based frames."""
        path = self.dir / "gt" / "gt.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (test sequences have no public ground truth)")
        rows = np.loadtxt(path, delimiter=",", ndmin=2)
        out = []
        for r in rows:
            frame, oid, x, y, w, h, mark, cls = r[:8]
            vis = r[8] if len(r) > 8 else 1.0
            if int(mark) == 1 and int(cls) == 1 and vis >= min_visibility:
                out.append((int(frame) - 1, int(oid), float(x), float(y), float(x + w), float(y + h)))
        return out


# MOT17 class IDs (TrackEval): 2 person_on_vehicle, 7 static_person, 8 distractor, 12 reflection
DISTRACTOR_CLASSES = {2, 7, 8, 12}


def official_filter(seq: MOTSequence, tracks: list[TrackOutput]) -> list[TrackOutput]:
    """Remove predictions that the official MOT17 evaluation ignores -- exactly as TrackEval:
    per frame, Hungarian-match predictions against ALL ground-truth boxes (every class, including
    zero-marked ones) at IoU >= 0.5, and drop predictions matched to a distractor class.
    Predictions matched to zero-marked pedestrians are KEPT (TrackEval only drops those GT rows)."""
    raw = seq.raw_ground_truth()
    by_frame: dict[int, list] = {}
    for r in raw:
        by_frame.setdefault(int(r[0]), []).append(r)
    preds: dict[int, list[TrackOutput]] = {}
    for t in tracks:
        preds.setdefault(t.frame, []).append(t)
    keep = []
    for f, ps in preds.items():
        g = by_frame.get(f, [])
        if not g:
            keep += ps
            continue
        iou = iou_matrix(np.array([r[2:6] for r in g]), np.array([p.box for p in ps]))
        iou[iou < 0.5 - np.finfo(float).eps] = 0
        rows, cols = linear_sum_assignment(-iou)
        drop = {c for r, c in zip(rows, cols) if iou[r, c] > np.finfo(float).eps and int(g[r][7]) in DISTRACTOR_CLASSES}
        keep += [p for i, p in enumerate(ps) if i not in drop]
    return sorted(keep, key=lambda t: (t.frame, t.track_id))


class PublicDetector:
    """Detections shipped with the sequence (det/det.txt) -- the standard way to
    compare TRACKERS fairly, since every tracker gets the same boxes."""

    def __init__(self, seq: MOTSequence, min_conf: float = 0.5, det_file: str = "det.txt"):
        """det_file: 'det.txt' = the public detections; a cached YOLO file (see
        tracelens.detect_cache) uses the same format and lives next to it."""
        self.by_frame: dict[int, list[Detection]] = {}
        path = seq.dir / "det" / det_file
        if not path.exists():
            raise FileNotFoundError(path)
        for r in np.loadtxt(path, delimiter=",", ndmin=2):
            frame, _, x, y, w, h, conf = r[:7]
            if conf >= min_conf:
                self.by_frame.setdefault(int(frame) - 1, []).append(
                    Detection(box=np.array([x, y, x + w, y + h], dtype=float), score=float(conf)))

    def detect(self, frame_idx: int, image=None) -> list[Detection]:
        return self.by_frame.get(frame_idx, [])
