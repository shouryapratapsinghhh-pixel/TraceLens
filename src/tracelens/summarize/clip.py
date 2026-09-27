"""Assemble the summary video: only the target's segments, target highlighted,
with the ORIGINAL timestamp burned in so a reviewer can find it in the source."""

from __future__ import annotations

import cv2
import numpy as np

from tracelens.track.tracker import TrackOutput


def build_summary_frames(frames: list[np.ndarray], tracks: list[TrackOutput], target_id: int,
                         segments: list[tuple[int, int]], fps: float) -> list[np.ndarray]:
    boxes = {t.frame: t.box for t in tracks if t.track_id == target_id}
    out = []
    for k, (a, b) in enumerate(segments, 1):
        for f in range(a, b + 1):
            img = frames[f].copy()
            if f in boxes:
                x1, y1, x2, y2 = (round(v) for v in boxes[f])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(img, "TARGET", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 255, 255), 1)
            stamp = f"clip {k}/{len(segments)}  t={f / fps:5.1f}s (frame {f})"
            cv2.putText(img, stamp, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1)
            out.append(img)
    return out
