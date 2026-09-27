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
    hidden: list[tuple[int, int]] = field(default_factory=list)  # [start, end) windows out of view
    color_bottom: tuple[int, int, int] | None = None  # two-tone object (e.g. shirt / trousers)
    # velocity changes: [(absolute_frame, (vx, vy)), ...] -- from that frame on, move at that
    # velocity. When set, motion is integrated step by step and clamped at the frame edges
    # (no bouncing), so scripted behaviour (stop, run, turn) happens exactly when written.
    schedule: list[tuple[int, tuple[float, float]]] = field(default_factory=list)


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
            if any(a <= f < b for a, b in o.hidden):
                continue  # out of view / fully occluded: keeps moving, but isn't drawn or labelled
            t = f - o.enter
            w, h = o.size
            if o.schedule:
                x, y = _scripted_position(o, f, width - w, height - h)
            else:
                x = _bounce(o.start[0] + o.velocity[0] * t, width - w)
                y = _bounce(o.start[1] + o.velocity[1] * t, height - h)
            x1, y1, x2, y2 = round(x), round(y), round(x) + w, round(y) + h
            img[y1:y2, x1:x2] = o.color
            if o.color_bottom is not None:
                img[(y1 + y2) // 2:y2, x1:x2] = o.color_bottom
            gt.append((f, o.obj_id, float(x1), float(y1), float(x2), float(y2)))
        if noise > 0:
            img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
        frames.append(img)
    return SyntheticVideo(frames=frames, gt=gt, fps=fps)


def _scripted_position(o: SyntheticObject, frame: int, max_x: float, max_y: float) -> tuple[float, float]:
    """Integrate a velocity schedule from the object's first frame up to `frame`."""
    x, y = o.start
    for f in range(o.enter, frame):
        vx, vy = o.velocity
        for when, vel in o.schedule:
            if f >= when:
                vx, vy = vel
        x, y = min(max(x + vx, 0.0), max_x), min(max(y + vy, 0.0), max_y)
    return x, y


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


def patrol_scene() -> list[SyntheticObject]:
    """300 frames. Target (id 1) is visible twice, leaving view for 60 frames
    in between -- long enough that a tracker gives it a NEW id on return.
    Two distractors cross its area."""
    return [
        SyntheticObject(1, start=(20, 60), velocity=(2, 1), color=(0, 0, 255), enter=20, exit=200,
                        hidden=[(80, 140)]),
        SyntheticObject(2, start=(280, 40), velocity=(-1, 1), color=(0, 255, 0)),
        SyntheticObject(3, start=(150, 150), velocity=(1, -1), size=(50, 50), color=(255, 0, 0),
                        enter=100, exit=260),
    ]


def lookalike_scene() -> list[SyntheticObject]:
    """300 frames. Target (id 1): red top / blue bottom, visible twice with a
    60-frame absence. Look-alike (id 2): the SAME two colours, SWAPPED (blue
    top / red bottom), appearing exactly while the target is away. A whole-box
    colour histogram can't tell them apart; a top/bottom one can. Plus one
    unrelated distractor (id 3)."""
    red, blue, green = (0, 0, 255), (255, 0, 0), (0, 255, 0)
    return [
        SyntheticObject(1, start=(20, 60), velocity=(2, 1), color=red, color_bottom=blue,
                        enter=20, exit=200, hidden=[(80, 140)]),
        SyntheticObject(2, start=(230, 100), velocity=(-1, 0), color=blue, color_bottom=red,
                        enter=90, exit=132),
        SyntheticObject(3, start=(280, 20), velocity=(-1, 1), color=green),
    ]


# -- a scripted incident, with ground-truth events --------------------------------------

INCIDENT_SIZE = (640, 360)
INCIDENT_FPS = 10.0
RESTRICTED_ZONE = [(420, 150), (620, 150), (620, 350), (420, 350)]  # (x, y) polygon, pixels


def incident_scene() -> tuple[list[SyntheticObject], dict]:
    """640x360 at 10 fps, 300 frames. Target (id 1, 80 px tall, so one body-height
    = 80 px): walks in from the left, loiters 5 s, runs into the restricted zone,
    stops, drops a bag (id 2) and walks back out. Distractor (id 3) walks across.

    Returns (objects, script). The script holds the scripted behaviour frames;
    `incident_gt_events` turns it -- plus the rendered boxes, for zone crossings
    and appearances -- into ground-truth events. Nothing positional is hand-typed.
    """
    walk, run = 5.0, 16.0  # px/frame = 0.625 and 2.0 body-heights per second at 10 fps
    script = {"dwell": (40, 90), "run": (90, 110), "leave_bag": 135}
    target = SyntheticObject(
        1, start=(0, 200), velocity=(walk, 0), size=(40, 80), color=(0, 0, 255), color_bottom=(80, 40, 0),
        enter=10, exit=229,  # walks out of frame at the left edge (x reaches 0 at frame 229)
        schedule=[(script["dwell"][0], (0, 0)), (script["run"][0], (run, 0)),
                  (script["run"][1], (0, 0)), (script["leave_bag"], (-walk, 0))],
    )
    bag = SyntheticObject(2, start=(505, 262), velocity=(0, 0), size=(24, 18), color=(0, 200, 255), enter=116)
    walker = SyntheticObject(3, start=(600, 40), velocity=(-walk, 0), size=(40, 80), color=(0, 255, 0),
                             enter=0, exit=110)
    return [target, bag, walker], script


def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """Ray casting: count how many polygon edges a ray to the right crosses; odd = inside."""
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def incident_gt_events(gt: list[GTRow], script: dict, fps: float = INCIDENT_FPS,
                       zone: list[tuple[float, float]] = RESTRICTED_ZONE) -> list[dict]:
    """Ground truth for the target (id 1): appear/disappear and zone crossings from the
    rendered boxes (feet point = bottom-centre), behaviours from the script."""
    rows = sorted((r for r in gt if r[1] == 1), key=lambda r: r[0])
    ev = [{"type": "appear", "t": rows[0][0] / fps}, {"type": "disappear", "t": rows[-1][0] / fps}]
    was_in = False
    for f, _, x1, _, x2, y2 in rows:
        now_in = point_in_polygon((x1 + x2) / 2, y2 - 1, zone)
        if now_in != was_in:
            ev.append({"type": "zone_enter" if now_in else "zone_exit", "t": f / fps})
        was_in = now_in
    ev += [{"type": "dwell", "t": script["dwell"][0] / fps},
           {"type": "run", "t": script["run"][0] / fps},
           {"type": "object_left_behind", "t": script["leave_bag"] / fps}]
    return sorted(ev, key=lambda e: e["t"])
