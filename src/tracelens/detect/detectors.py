"""Detectors: one Detection format, several sources.

- SyntheticDetector: reads ground truth and corrupts it in CONTROLLED ways
  (missed detections, box jitter, false alarms), so the tracker can be
  tested against known failure rates.
- YoloDetector: a real detector (Ultralytics YOLO) for real footage.
  Optional dependency, import-guarded; no test requires it.
  Note: Ultralytics is AGPL-3.0 licensed -- fine for an open-source
  portfolio repo, but worth knowing before any commercial use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tracelens.video.synthetic import GTRow


@dataclass
class Detection:
    box: np.ndarray  # (4,) x1, y1, x2, y2 in pixels
    score: float = 1.0
    cls: str = "person"


class SyntheticDetector:
    def __init__(
        self,
        gt: list[GTRow],
        miss_rate: float = 0.0,
        jitter: float = 0.0,
        false_positive_rate: float = 0.0,
        frame_size: tuple[int, int] = (320, 240),
        seed: int = 0,
        weak_rate: float = 0.0,
    ):
        self.by_frame: dict[int, list[GTRow]] = {}
        for row in gt:
            self.by_frame.setdefault(row[0], []).append(row)
        self.miss_rate = miss_rate
        self.jitter = jitter
        self.fp_rate = false_positive_rate
        self.frame_size = frame_size
        self.weak_rate = weak_rate  # share of true detections given a LOW score (like a partly hidden person)
        self.rng = np.random.default_rng(seed)

    def detect(self, frame_idx: int, image: np.ndarray | None = None) -> list[Detection]:
        dets = []
        for _, _, x1, y1, x2, y2 in self.by_frame.get(frame_idx, []):
            if self.rng.random() < self.miss_rate:
                continue  # the detector "blinks" on this object this frame
            box = np.array([x1, y1, x2, y2], dtype=float)
            if self.jitter > 0:
                box = box + self.rng.normal(0, self.jitter, 4)
            weak = self.rng.random() < self.weak_rate
            score = self.rng.uniform(0.15, 0.45) if weak else self.rng.uniform(0.6, 1.0)
            dets.append(Detection(box=box, score=float(score)))
        if self.rng.random() < self.fp_rate:  # a spurious box somewhere random
            w, h = self.frame_size
            x, y = self.rng.uniform(0, w - 30), self.rng.uniform(0, h - 30)
            dets.append(Detection(box=np.array([x, y, x + 25, y + 25]),
                                  score=float(self.rng.uniform(0.3, 0.6))))
        # Real detectors return boxes in arbitrary order. Without this shuffle,
        # detections come out in ground-truth ID order, and the tracker's
        # tie-breaking can silently "use" that order to keep identities --
        # leaking the answer into the test.
        return [dets[i] for i in self.rng.permutation(len(dets))]


class YoloDetector:
    """Ultralytics YOLO. `pip install ultralytics` (or `pip install -e .[yolo]`)."""

    def __init__(self, weights: str = "yolov8n.pt", classes: tuple[str, ...] = ("person",),
                 conf: float = 0.3, imgsz: int = 640):
        """imgsz: the size YOLO resizes each frame to before detecting. The default 640
        shrinks a 1920-px MOT17 frame 3x, and far-away pedestrians shrink to a few pixels;
        1280 keeps them detectable, at roughly 3-4x the compute."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("YoloDetector needs `pip install ultralytics`") from e
        self.model = YOLO(weights)
        self.classes = set(classes)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame_idx: int, image: np.ndarray) -> list[Detection]:
        result = self.model(image, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        names = result.names
        dets = []
        for box, score, c in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                 result.boxes.cls.cpu().numpy()):
            label = names[int(c)]
            if label in self.classes:
                dets.append(Detection(box=box.astype(float), score=float(score), cls=label))
        return dets
