"""Event detection from tracks -- rules over measured motion, from scratch.

Speed is in BODY-HEIGHTS PER SECOND (centre displacement / the target's own
box height): a person far from the camera covers fewer pixels than one close
up, so a pixel threshold would mean something different at every distance.
Walking is roughly 0.5-1 height/s, running roughly 2.

Zones use the FEET point (bottom-centre of the box): whether someone is "in
the restricted area" depends on where they stand, not on the middle of their
torso.

Events (each with start time, and end time where it has one):
  appear / disappear        first / last sighting of each visit
  dwell                     speed < dwell_speed for at least dwell_s seconds
  run                       speed > run_speed  for at least run_s  seconds
  zone_enter / zone_exit    feet point crosses a named polygon; a change must
                            persist `zone_confirm` frames (so box jitter at the
                            border can't make it flicker in and out)
  object_left_behind        a track that starts right next to the target, then
                            stays put while the target moves well away. Reported
                            at the moment the target last stood beside it (`t`),
                            with `t_confirmed` = when the separation became certain

Every threshold is a named, documented parameter; nothing is learned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tracelens.track.tracker import TrackOutput
from tracelens.video.synthetic import point_in_polygon


@dataclass
class Zone:
    name: str
    polygon: list[tuple[float, float]]


@dataclass
class EventParams:
    dwell_speed: float = 0.15  # body-heights/s
    dwell_s: float = 3.0
    run_speed: float = 1.5
    run_s: float = 1.0
    speed_window: int = 5  # frames: speed = displacement over this many frames (smooths jitter)
    zone_confirm: int = 3
    visit_gap: int = 10  # frames without a sighting that split two visits
    left_near: float = 1.0  # object must start within this many body-heights of the target
    left_away: float = 2.0  # ...and the target must move this far away while it stays put
    left_static: float = 0.3  # the object may drift at most this many body-heights


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) inclusive index ranges where mask is True."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def _series(tracks: list[TrackOutput], track_id: int):
    rows = sorted((t for t in tracks if t.track_id == track_id), key=lambda t: t.frame)
    frames = np.array([t.frame for t in rows])
    boxes = np.array([t.box for t in rows]).reshape(-1, 4)
    return frames, boxes


def speed_body_heights(frames: np.ndarray, boxes: np.ndarray, fps: float, window: int) -> np.ndarray:
    """Per-sample speed in body-heights/second, from centre displacement over `window` samples."""
    centers = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2])
    heights = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
    speed = np.zeros(len(frames))
    for i in range(len(frames)):
        j = min(len(frames) - 1, i + window)
        k = max(0, j - window)
        dt = (frames[j] - frames[k]) / fps
        if dt > 0:
            speed[i] = np.linalg.norm(centers[j] - centers[k]) / dt / np.median(heights[k:j + 1])
    return speed


def detect_events(tracks: list[TrackOutput], target_id: int, fps: float,
                  zones: list[Zone] | None = None, params: EventParams | None = None) -> list[dict]:
    p = params or EventParams()
    frames, boxes = _series(tracks, target_id)
    if len(frames) == 0:
        return []
    events: list[dict] = []

    # visits: split where the target went unseen for more than visit_gap frames
    breaks = np.where(np.diff(frames) > p.visit_gap)[0]
    starts, ends = np.r_[0, breaks + 1], np.r_[breaks, len(frames) - 1]
    for a, b in zip(starts, ends):
        events.append({"type": "appear", "t": frames[a] / fps, "frame": int(frames[a])})
        events.append({"type": "disappear", "t": frames[b] / fps, "frame": int(frames[b])})

        f, bx = frames[a:b + 1], boxes[a:b + 1]
        speed = speed_body_heights(f, bx, fps, p.speed_window)
        for kind, mask, min_s in (("dwell", speed < p.dwell_speed, p.dwell_s),
                                  ("run", speed > p.run_speed, p.run_s)):
            for i, j in _runs(mask):
                if (f[j] - f[i] + 1) / fps >= min_s:
                    events.append({"type": kind, "t": f[i] / fps, "t_end": f[j] / fps, "frame": int(f[i]),
                                   "mean_speed": round(float(speed[i:j + 1].mean()), 2)})

        for z in zones or []:
            feet = [point_in_polygon((x1 + x2) / 2, y2 - 1, z.polygon) for x1, _, x2, y2 in bx]
            state, run_len = False, 0
            for idx, inside in enumerate(feet):
                run_len = run_len + 1 if inside != state else 0
                if run_len >= p.zone_confirm:  # the change has persisted: accept it
                    first = idx - p.zone_confirm + 1
                    state = inside
                    run_len = 0
                    events.append({"type": "zone_enter" if inside else "zone_exit",
                                   "t": f[first] / fps, "frame": int(f[first]), "zone": z.name})

    events += _left_behind(tracks, target_id, frames, boxes, fps, p)
    return sorted(events, key=lambda e: e["t"])


def _left_behind(tracks, target_id, t_frames, t_boxes, fps, p: EventParams) -> list[dict]:
    t_center = {int(f): np.array([(b[0] + b[2]) / 2, b[3]]) for f, b in zip(t_frames, t_boxes)}
    t_height = float(np.median(t_boxes[:, 3] - t_boxes[:, 1]))
    out = []
    for oid in {t.track_id for t in tracks} - {target_id}:
        o_frames, o_boxes = _series(tracks, oid)
        if int(o_frames[0]) not in t_center:
            continue  # the target wasn't visible when this object appeared
        o_pts = np.column_stack([(o_boxes[:, 0] + o_boxes[:, 2]) / 2, o_boxes[:, 3]])  # ground contact
        if np.linalg.norm(o_pts[0] - t_center[int(o_frames[0])]) > p.left_near * t_height:
            continue  # didn't appear next to the target
        if np.linalg.norm(o_pts - o_pts[0], axis=1).max() > p.left_static * t_height:
            continue  # it moved: carried, not left
        last_near = int(o_frames[0])
        for f, pt in zip(o_frames, o_pts):
            c = t_center.get(int(f))
            if c is None:
                continue
            dist = np.linalg.norm(c - pt)
            if dist <= p.left_near * t_height:
                last_near = int(f)  # target still at the object's side
            elif dist > p.left_away * t_height:
                # confirmed now; but the event HAPPENED when the target last stood beside it
                out.append({"type": "object_left_behind", "t": last_near / fps, "frame": last_near,
                            "t_confirmed": f / fps, "object_track": int(oid)})
                break
    return out


def match_events(detected: list[dict], truth: list[dict], tolerance_s: float = 2.0) -> dict:
    """One-to-one matching per event type, nearest in time first, within tolerance."""
    tp, timing, used = 0, [], set()
    for g in sorted(truth, key=lambda e: e["t"]):
        best, best_dt = None, None
        for i, d in enumerate(detected):
            if i in used or d["type"] != g["type"]:
                continue
            dt = abs(d["t"] - g["t"])
            if dt <= tolerance_s and (best_dt is None or dt < best_dt):
                best, best_dt = i, dt
        if best is not None:
            used.add(best)
            tp += 1
            timing.append(best_dt)
    fp, fn = len(detected) - tp, len(truth) - tp
    prec = tp / len(detected) if detected else 0.0
    rec = tp / len(truth) if truth else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "tp": tp, "false_alarms": fp, "missed": fn,
            "mean_timing_error_s": float(np.mean(timing)) if timing else float("nan"),
            "missed_types": [g["type"] for g in truth if not any(
                d["type"] == g["type"] and abs(d["t"] - g["t"]) <= tolerance_s for d in detected)]}
