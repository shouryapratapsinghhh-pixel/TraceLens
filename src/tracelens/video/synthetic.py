"""Synthetic surveillance-style videos with EXACT ground truth.

Real datasets (MOT17 etc.) need downloads; these need nothing, so every
test runs offline -- and because the true box of every object is known in
every frame, a tracker bug shows up as a hard, exact failure rather than
a slightly lower score.

Each object is a coloured rectangle moving at constant velocity, bouncing
off the frame edges, visible only between its enter and exit frames. Two
objects can be placed on crossing paths to test identity switches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ground truth row: (frame, object_id, x1, y1, x2, y2)
GTRow = tuple[int, int, float, float, float, float]


@dataclass
class SyntheticObject:
    obj_id: int
    start: tuple[float, float]  # top-left (x, y) at its first visible frame
    velocity: tuple[float, float]  # pixels per frame
    size: tuple[int, int] = (40, 80)  # (w, h): person-like proportions by default
    color: tuple[int, int, int] = (0, 0, 255)  # BGR
    enter: int = 0
    exit: int | None = None  # exclusive; None = until the end


@dataclass
class SyntheticVideo:
    frames: list[np.ndarray]
    gt: list[GTRow] = field(default_factory=list)
    fps: float = 10.0

    def gt_for_frame(self, frame: int) -> list[GTRow]:
        return [r for r in self.gt if r[0] == frame]


def make_synthetic_video(
    objects: list[SyntheticObject],
    n_frames: int = 100,
    size: tuple[int, int] = (320, 240),  # (width, height)
    fps: float = 10.0,
    noise: float = 0.0,
    seed: int = 0,
) -> SyntheticVideo:
    """Render frames and record the true box of every visible object."""
    width, height = size
    rng = np.random.default_rng(seed)
    frames, gt = [], []
    for f in range(n_frames):
        img = np.full((height, width, 3), 40, dtype=np.uint8)  # dark background
        for o in objects:
            if f < o.enter or (o.exit is not None and f >= o.exit):
                continue
            t = f - o.enter
            w, h = o.size
            x = _bounce(o.start[0] + o.velocity[0] * t, width - w)
            y = _bounce(o.start[1] + o.velocity[1] * t, height - h)
            x1, y1, x2, y2 = round(x), round(y), round(x) + w, round(y) + h
            img[y1:y2, x1:x2] = o.color
            gt.append((f, o.obj_id, float(x1), float(y1), float(x2), float(y2)))
        if noise > 0:
            img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
        frames.append(img)
    return SyntheticVideo(frames=frames, gt=gt, fps=fps)


def _bounce(pos: float, max_pos: float) -> float:
    """Reflect a coordinate back into [0, max_pos] (a ball bouncing off walls)."""
    if max_pos <= 0:
        return 0.0
    period = 2 * max_pos
    p = pos % period
    return p if p <= max_pos else period - p


def crossing_scene(n_frames: int = 60) -> list[SyntheticObject]:
    """Two same-sized objects whose paths cross mid-video: the classic
    identity-switch trap for a tracker."""
    return [
        SyntheticObject(1, start=(20, 80), velocity=(4, 0), color=(0, 0, 255)),
        SyntheticObject(2, start=(260, 80), velocity=(-4, 0), color=(255, 0, 0)),
    ]


def busy_scene() -> list[SyntheticObject]:
    """Several objects entering and leaving at different times."""
    return [
        SyntheticObject(1, start=(10, 20), velocity=(3, 1), color=(0, 0, 255)),
        SyntheticObject(2, start=(250, 150), velocity=(-2, -1), color=(0, 255, 0), enter=10, exit=80),
        SyntheticObject(3, start=(150, 10), velocity=(0, 2), color=(255, 0, 0), enter=30),
        SyntheticObject(4, start=(60, 140), velocity=(2, 0), size=(30, 30), color=(0, 255, 255), exit=50),
    ]
