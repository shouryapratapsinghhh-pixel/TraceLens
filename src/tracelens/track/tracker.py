"""Multi-object tracker, written from scratch (SORT-style).

Each frame:
  1. PREDICT: every existing track's Kalman filter predicts where its box
     should be now (constant velocity).
  2. MATCH: predicted boxes vs new detections, cost = 1 - IoU, solved
     optimally with the Hungarian algorithm (scipy.optimize.linear_sum_assignment).
     Pairs with IoU below `iou_threshold` are rejected.
  3. UPDATE: matched tracks correct their filter with the detection.
     Unmatched detections start new tracks. Unmatched tracks age, and are
     deleted after `max_age` frames without a match -- so a detector that
     "blinks" for a frame or two doesn't break an identity.
  4. REPORT: only tracks confirmed by `min_hits` matches are output, so a
     one-frame false alarm doesn't become a track.

use_motion=False replaces the Kalman prediction with "the box stays where
it was", to show what motion prediction buys (see tests: crossing objects).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracelens.detect.detectors import Detection


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (n, 4) and b (m, 4), xyxy format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = a[:, None, :], b[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


class KalmanBox:
    """Constant-velocity Kalman filter on a box.
    State: [cx, cy, w, h, vx, vy, vw, vh]; measurement: [cx, cy, w, h]."""

    def __init__(self, box: np.ndarray, q: float = 1.0, r: float = 10.0):
        self.x = np.zeros(8)
        self.x[:4] = _xyxy_to_cxcywh(box)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1000.0, 1000.0, 1000.0, 1000.0])  # velocity unknown at birth
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)  # position += velocity each frame
        self.H = np.eye(4, 8)
        self.Q = np.eye(8) * q  # process noise: how much motion can deviate from constant velocity
        self.Q[4:, 4:] *= 0.01
        self.R = np.eye(4) * r  # measurement noise: how much to trust a detection

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)  # a box can't shrink below 1 px
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box()

    def update(self, box: np.ndarray) -> None:
        z = _xyxy_to_cxcywh(box)
        y = z - self.H @ self.x  # innovation: measurement minus prediction
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain: how far to move toward the measurement
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def box(self) -> np.ndarray:
        return _cxcywh_to_xyxy(self.x[:4])


def _xyxy_to_cxcywh(b: np.ndarray) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]], dtype=float)


def _cxcywh_to_xyxy(s: np.ndarray) -> np.ndarray:
    return np.array([s[0] - s[2] / 2, s[1] - s[3] / 2, s[0] + s[2] / 2, s[1] + s[3] / 2])


@dataclass
class Track:
    track_id: int | None  # None until confirmed: false alarms never consume an ID
    kf: KalmanBox
    last_box: np.ndarray
    hits: int = 1
    time_since_update: int = 0
    history: list[tuple[int, np.ndarray]] = field(default_factory=list)  # (frame, box)


@dataclass
class TrackOutput:
    frame: int
    track_id: int
    box: np.ndarray


class Tracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5, min_hits: int = 3,
                 use_motion: bool = True):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.use_motion = use_motion
        self.tracks: list[Track] = []
        self._next_id = 1
        self.frame = -1

    def _predicted_boxes(self) -> np.ndarray:
        boxes = []
        for t in self.tracks:
            predicted = t.kf.predict()  # always advance the filter, even if unused
            boxes.append(predicted if self.use_motion else t.last_box)
        return np.array(boxes).reshape(-1, 4)

    def update(self, detections: list[Detection]) -> list[TrackOutput]:
        self.frame += 1
        pred = self._predicted_boxes()
        det_boxes = np.array([d.box for d in detections]).reshape(-1, 4)

        matched_t, matched_d = set(), set()
        if len(pred) and len(det_boxes):
            iou = iou_matrix(pred, det_boxes)
            rows, cols = linear_sum_assignment(-iou)  # maximize total IoU
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_threshold:
                    t = self.tracks[r]
                    t.kf.update(det_boxes[c])
                    t.last_box = det_boxes[c]
                    t.hits += 1
                    t.time_since_update = 0
                    matched_t.add(r)
                    matched_d.add(c)

        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.time_since_update += 1
        # a tentative track that misses before being confirmed was probably a false alarm
        self.tracks = [t for t in self.tracks
                       if t.time_since_update <= self.max_age
                       and not (t.hits < self.min_hits and t.time_since_update > 0)]

        for j, d in enumerate(detections):
            if j not in matched_d:
                self.tracks.append(Track(None, KalmanBox(d.box), last_box=d.box.copy()))

        out = []
        for t in self.tracks:
            if t.time_since_update == 0 and t.hits >= self.min_hits:
                if t.track_id is None:  # confirmed just now: give it the next ID
                    t.track_id = self._next_id
                    self._next_id += 1
                box = t.kf.box() if self.use_motion else t.last_box
                t.history.append((self.frame, box))
                out.append(TrackOutput(self.frame, t.track_id, box))
        return out


def run_tracker(detector, n_frames: int, frames: list[np.ndarray] | None = None,
                **tracker_kwargs) -> list[TrackOutput]:
    """Detect + track every frame; returns all reported (frame, id, box) rows."""
    tracker = Tracker(**tracker_kwargs)
    out: list[TrackOutput] = []
    for f in range(n_frames):
        img = frames[f] if frames is not None else None
        out.extend(tracker.update(detector.detect(f, img)))
    return out
