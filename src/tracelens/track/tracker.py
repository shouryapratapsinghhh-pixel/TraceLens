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

    def apply_affine(self, A: np.ndarray) -> None:
        """Move the state with a camera transform A (2x3, similarity): the centre is mapped by A,
        velocity rotates/scales with its linear part, width/height scale with its zoom. The
        covariance is transformed the same way so the filter's uncertainty stays consistent."""
        L, t = A[:, :2], A[:, 2]
        s = float(np.sqrt(abs(np.linalg.det(L))))
        T = np.zeros((8, 8))
        T[0:2, 0:2] = L
        T[2:4, 2:4] = np.eye(2) * s
        T[4:6, 4:6] = L
        T[6:8, 6:8] = np.eye(2) * s
        self.x = T @ self.x
        self.x[0:2] += t
        self.P = T @ self.P @ T.T

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
    """byte=True switches on ByteTrack association (Zhang et al., 2022):
      stage 1: all tracks vs HIGH-confidence detections (score >= high_thresh)
      stage 2: still-unmatched CONFIRMED tracks vs LOW-confidence detections
               (low_thresh <= score < high_thresh), needing a stricter IoU (low_iou)
      new tracks start only from unmatched HIGH-confidence detections.
    Why: a partly occluded person usually still gets a detection, just a weak one.
    A single confidence cutoff throws it away and the track dies; ByteTrack lets a
    weak box keep an EXISTING track alive, without letting weak boxes (often false
    alarms) create new ones. byte=False keeps the original single-stage behaviour."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5, min_hits: int = 3,
                 use_motion: bool = True, byte: bool = False, high_thresh: float = 0.5,
                 low_thresh: float = 0.1, low_iou: float = 0.5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.use_motion = use_motion
        self.byte = byte
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.low_iou = low_iou
        self.tracks: list[Track] = []
        self._next_id = 1
        self.frame = -1

    def _predicted_boxes(self, warp: np.ndarray | None = None) -> np.ndarray:
        boxes = []
        for t in self.tracks:
            predicted = t.kf.predict()  # always advance the filter, even if unused
            if warp is not None:  # camera moved: carry the prediction with it
                t.kf.apply_affine(warp)
                predicted = t.kf.box()
                t.last_box = _warp_box(t.last_box, warp)
            boxes.append(predicted if self.use_motion else t.last_box)
        return np.array(boxes).reshape(-1, 4)

    def _match(self, pred, det_boxes, track_idx, det_idx, thr, matched_t, matched_d) -> None:
        """Hungarian on IoU between the given tracks and detections; apply matches >= thr."""
        if not track_idx or not det_idx:
            return
        iou = iou_matrix(pred[track_idx], det_boxes[det_idx])
        rows, cols = linear_sum_assignment(-iou)  # maximize total IoU
        for r, c in zip(rows, cols):
            if iou[r, c] >= thr:
                ti, di = track_idx[r], det_idx[c]
                t = self.tracks[ti]
                t.kf.update(det_boxes[di])
                t.last_box = det_boxes[di]
                t.hits += 1
                t.time_since_update = 0
                matched_t.add(ti)
                matched_d.add(di)

    def update(self, detections: list[Detection], warp: np.ndarray | None = None) -> list[TrackOutput]:
        """warp: optional 2x3 camera transform from the previous frame to this one (see cmc.py)."""
        self.frame += 1
        pred = self._predicted_boxes(warp)
        det_boxes = np.array([d.box for d in detections]).reshape(-1, 4)
        all_tracks = list(range(len(self.tracks)))
        matched_t, matched_d = set(), set()

        if not self.byte:
            self._match(pred, det_boxes, all_tracks, list(range(len(detections))),
                        self.iou_threshold, matched_t, matched_d)
            can_start = set(range(len(detections)))
        else:
            high = [j for j, d in enumerate(detections) if d.score >= self.high_thresh]
            low = [j for j, d in enumerate(detections) if self.low_thresh <= d.score < self.high_thresh]
            self._match(pred, det_boxes, all_tracks, high, self.iou_threshold, matched_t, matched_d)
            confirmed_left = [i for i in all_tracks if i not in matched_t and self.tracks[i].hits >= self.min_hits]
            self._match(pred, det_boxes, confirmed_left, low, self.low_iou, matched_t, matched_d)
            can_start = set(high)  # weak boxes may extend a track, never create one

        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.time_since_update += 1
        # a tentative track that misses before being confirmed was probably a false alarm
        self.tracks = [t for t in self.tracks
                       if t.time_since_update <= self.max_age
                       and not (t.hits < self.min_hits and t.time_since_update > 0)]

        for j, d in enumerate(detections):
            if j not in matched_d and j in can_start:
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


def _warp_box(box: np.ndarray, A: np.ndarray) -> np.ndarray:
    cx, cy, w, h = _xyxy_to_cxcywh(box)
    c = A[:, :2] @ np.array([cx, cy]) + A[:, 2]
    s = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
    return _cxcywh_to_xyxy(np.array([c[0], c[1], w * s, h * s]))


def run_tracker(detector, n_frames: int, frames: list[np.ndarray] | None = None,
                warps: np.ndarray | None = None, **tracker_kwargs) -> list[TrackOutput]:
    """Detect + track every frame; returns all reported (frame, id, box) rows.
    warps: optional (n_frames, 2, 3) camera-motion transforms (tracelens.track.cmc)."""
    tracker = Tracker(**tracker_kwargs)
    out: list[TrackOutput] = []
    for f in range(n_frames):
        img = frames[f] if frames is not None else None
        w = warps[f] if warps is not None and f > 0 else None
        out.extend(tracker.update(detector.detect(f, img), warp=w))
    return out
