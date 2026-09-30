"""Camera-motion compensation (CMC).

A constant-velocity Kalman filter predicts in IMAGE coordinates. When the camera
pans, shakes or rides a moving bus, every person's image position jumps at once,
the predictions are left behind, and tracks break (MOT17-10 and -13, the two
moving-camera test sequences, had 504 of 727 ID switches).

Per pair of frames: detect background corners, follow them with sparse optical
flow (Lucas-Kanade), and fit ONE global similarity transform (shift, rotation,
zoom) with RANSAC -- points on moving people don't fit the dominant motion and
are rejected as outliers. The tracker then moves every Kalman prediction by that
transform before matching. Same idea as BoT-SORT's global motion compensation.

Estimation runs on a downscaled grey image (fast); the translation is scaled back.
If too few points are found or RANSAC fails, the identity transform is used.

Two details learned the hard way (see tests): corner detection keeps corners relative to
the STRONGEST one in the image, so high-contrast moving objects can starve the background
of corners entirely. So (1) detected boxes are masked out before looking for corners --
points on people are exactly the ones that must not vote on camera motion -- using the
DETECTOR's boxes, never ground truth; and (2) the relative quality threshold is low (0.001).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)
IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _grey_small(img: np.ndarray, target_width: int) -> tuple[np.ndarray, float]:
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    scale = min(1.0, target_width / grey.shape[1])
    if scale < 1.0:
        grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return grey, scale


def estimate_warp(prev: np.ndarray, curr: np.ndarray, target_width: int = 640, min_points: int = 10,
                  exclude_boxes: np.ndarray | None = None) -> np.ndarray:
    """2x3 similarity transform mapping PREVIOUS-frame pixel coordinates to CURRENT-frame ones.
    exclude_boxes: (k, 4) xyxy boxes in the previous frame (detected objects) to ignore."""
    a, scale = _grey_small(prev, target_width)
    b, _ = _grey_small(curr, target_width)
    h, w = a.shape
    mask = np.zeros_like(a)
    mask[int(0.02 * h):int(0.98 * h), int(0.02 * w):int(0.98 * w)] = 255  # skip border artefacts
    if exclude_boxes is not None:
        for x1, y1, x2, y2 in np.asarray(exclude_boxes).reshape(-1, 4) * scale:
            mask[max(0, int(y1) - 2):int(y2) + 3, max(0, int(x1) - 2):int(x2) + 3] = 0
    pts = cv2.goodFeaturesToTrack(a, maxCorners=1000, qualityLevel=0.001, minDistance=5, blockSize=3, mask=mask)
    if pts is None or len(pts) < min_points:
        return IDENTITY.copy()
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None)
    good = status.ravel() == 1
    if good.sum() < min_points:
        return IDENTITY.copy()
    M, inliers = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC, ransacReprojThreshold=1.0)
    if M is None or inliers is None or inliers.sum() < min_points:
        return IDENTITY.copy()
    M = M.astype(float)
    M[:, 2] /= scale  # translation back to full-resolution pixels (rotation/zoom are scale-free)
    return M


def sequence_warps(frames, cache_path: str | Path | None = None, log_every: int = 200,
                   detector=None) -> np.ndarray:
    """(n_frames, 2, 3): warps[f] maps frame f-1 coordinates to frame f; warps[0] = identity.
    detector (optional): its boxes in frame f-1 are masked out when estimating warps[f].
    Cached to `cache_path` (.npy) so studies with many settings compute it once."""
    if cache_path is not None and Path(cache_path).exists():
        return np.load(cache_path)
    n = len(frames)
    warps = np.repeat(IDENTITY[None], n, axis=0)
    prev = frames[0]
    for f in range(1, n):
        curr = frames[f]
        boxes = None
        if detector is not None:
            dets = detector.detect(f - 1, prev)
            boxes = np.array([d.box for d in dets]).reshape(-1, 4) if dets else None
        warps[f] = estimate_warp(prev, curr, exclude_boxes=boxes)
        prev = curr
        if f % log_every == 0:
            logger.info("camera motion: %d/%d frames", f, n)
    if cache_path is not None:
        np.save(cache_path, warps)
    return warps
