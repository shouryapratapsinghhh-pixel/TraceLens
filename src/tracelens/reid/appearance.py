"""Appearance fingerprints, from scratch (numpy; OpenCV only for BGR->HSV).

A fingerprint is a colour histogram of the object's box, in HSV (hue +
saturation, so lighting brightness matters less). With parts=2 the box is
split into a top and bottom half, each with its own histogram -- the classic
person re-identification trick: "red shirt + blue trousers" and "blue shirt
+ red trousers" have identical overall colour proportions, but different
top/bottom histograms.

Similarity = Bhattacharyya coefficient per part (sum of sqrt(p * q) over
bins: 1 = identical distributions, 0 = no overlap), averaged over parts.
"""

from __future__ import annotations

import cv2
import numpy as np

from tracelens.track.tracker import TrackOutput

H_BINS, S_BINS = 12, 4


def _crop(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1 = max(0, round(box[0])), max(0, round(box[1]))
    x2, y2 = min(w, round(box[2])), min(h, round(box[3]))
    return image[y1:y2, x1:x2]


def _hist(hsv: np.ndarray) -> np.ndarray:
    """Normalized 2-D hue x saturation histogram, flattened."""
    if hsv.size == 0:
        return np.full(H_BINS * S_BINS, 1.0 / (H_BINS * S_BINS))
    hue = np.minimum((hsv[..., 0].astype(int) * H_BINS) // 180, H_BINS - 1)  # OpenCV hue is 0-179
    sat = np.minimum((hsv[..., 1].astype(int) * S_BINS) // 256, S_BINS - 1)
    counts = np.bincount((hue * S_BINS + sat).ravel(), minlength=H_BINS * S_BINS).astype(float)
    return counts / counts.sum()


def fingerprint(image: np.ndarray, box: np.ndarray, parts: int = 2) -> np.ndarray:
    """(parts, H_BINS * S_BINS): one histogram per horizontal stripe of the box."""
    crop = _crop(image, box)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) if crop.size else crop
    stripes = np.array_split(hsv, parts, axis=0) if crop.size else [crop] * parts
    return np.stack([_hist(s) for s in stripes])


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Bhattacharyya coefficient over parts, in [0, 1]."""
    return float(np.mean(np.sum(np.sqrt(a * b), axis=1)))


def track_fingerprints(frames: list[np.ndarray], tracks: list[TrackOutput], parts: int = 2,
                       max_samples: int = 10) -> dict[int, np.ndarray]:
    """Average fingerprint per track over up to `max_samples` evenly spaced
    frames (one frame can be blurred or partly occluded; an average is steadier)."""
    by_id: dict[int, list[TrackOutput]] = {}
    for t in tracks:
        by_id.setdefault(t.track_id, []).append(t)
    out = {}
    for tid, rows in by_id.items():
        rows = sorted(rows, key=lambda r: r.frame)
        idx = np.unique(np.linspace(0, len(rows) - 1, min(max_samples, len(rows))).round().astype(int))
        fps = [fingerprint(frames[rows[i].frame], rows[i].box, parts) for i in idx]
        mean = np.mean(fps, axis=0)
        out[tid] = mean / mean.sum(axis=1, keepdims=True)
    return out
