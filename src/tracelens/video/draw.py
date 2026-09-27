"""Draw tracks onto frames and write videos (OpenCV)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tracelens.track.tracker import TrackOutput


def color_for(track_id: int) -> tuple[int, int, int]:
    """Stable, distinct colour per track ID."""
    rng = np.random.default_rng(track_id * 7919)
    return tuple(int(c) for c in rng.integers(80, 256, 3))


def draw_tracks(frames: list[np.ndarray], tracks: list[TrackOutput]) -> list[np.ndarray]:
    by_frame: dict[int, list[TrackOutput]] = {}
    for t in tracks:
        by_frame.setdefault(t.frame, []).append(t)
    out = []
    for f, img in enumerate(frames):
        img = img.copy()
        for t in by_frame.get(f, []):
            x1, y1, x2, y2 = (round(v) for v in t.box)
            c = color_for(t.track_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
            cv2.putText(img, f"ID {t.track_id}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
        cv2.putText(img, f"frame {f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        out.append(img)
    return out


def write_video(frames: list[np.ndarray], path: str | Path, fps: float = 10.0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return path
