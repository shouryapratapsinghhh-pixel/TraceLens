"""Re-link a target that left view and came back under a new track ID.

Starting from the selected track, repeatedly:
  1. candidates = tracks that START after the chain's current track ENDS
     (the same object can't be in two places at once) and within
     `max_gap_frames` of that end;
  2. score each candidate's appearance fingerprint against the TARGET's
     original fingerprint (not the latest link's -- comparing to the latest
     would let errors drift: one bad link would make the next bad one look right);
  3. among candidates with similarity >= `threshold`, link the EARLIEST to
     appear (ties -> the more similar), then continue from it.
Stops when no candidate clears the threshold.

Why earliest, not most similar: under detector noise one target can break
into several fragments (seen: 5). Jumping to the MOST similar fragment can
skip ahead -- and every fragment in between then starts before the chain's
end and can never be linked, silently deleting that stretch of the target's
behaviour. The threshold is what rejects other people; ordering is by time.

Returns the chain of track IDs and a log of every decision, so each link is
auditable ("why did it merge these?").
"""

from __future__ import annotations

import numpy as np

from tracelens.reid.appearance import similarity
from tracelens.track.tracker import TrackOutput


def track_spans(tracks: list[TrackOutput]) -> dict[int, tuple[int, int]]:
    spans: dict[int, tuple[int, int]] = {}
    for t in tracks:
        a, b = spans.get(t.track_id, (t.frame, t.frame))
        spans[t.track_id] = (min(a, t.frame), max(b, t.frame))
    return spans


def _feet_and_height(box) -> tuple[np.ndarray, float]:
    return np.array([(box[0] + box[2]) / 2, box[3]]), float(box[3] - box[1])


def relink_chain(tracks: list[TrackOutput], fingerprints: dict, target_id: int,
                 threshold: float = 0.8, max_gap_frames: int = 150, fps: float | None = None,
                 max_speed_bh: float | None = None, slack_bh: float = 1.0,
                 margin: float = 0.0) -> tuple[list[int], list[dict]]:
    """max_speed_bh (needs fps): MOTION GATE -- a candidate must start within
        (max_speed_bh * seconds_gone + slack_bh) body-heights of where the chain's last
        track ended. None disables it.
    margin: AMBIGUITY CHECK -- among passing candidates that overlap the earliest one in
        time (they're simultaneous, so at most one can be the target), the most similar
        wins; if the runner-up is within `margin` of it, abstain instead of guessing."""
    spans = track_spans(tracks)
    rows: dict[int, list[TrackOutput]] = {}
    for t in tracks:
        rows.setdefault(t.track_id, []).append(t)
    first = {k: min(v, key=lambda r: r.frame).box for k, v in rows.items()}
    last = {k: max(v, key=lambda r: r.frame).box for k, v in rows.items()}

    chain, log = [target_id], []
    ref = fingerprints[target_id]
    current_end = spans[target_id][1]
    while True:
        cands = [tid for tid, (start, _) in spans.items()
                 if tid not in chain and current_end < start <= current_end + max_gap_frames]
        scored = []
        end_pt, end_h = _feet_and_height(last[chain[-1]])
        for c in cands:
            sim = similarity(ref, fingerprints[c])
            entry = {"after_frame": current_end, "candidate": c, "similarity": round(sim, 3), "linked": False}
            if max_speed_bh is not None and fps:
                start_pt, start_h = _feet_and_height(first[c])
                dist_bh = float(np.linalg.norm(start_pt - end_pt)) / max((end_h + start_h) / 2, 1.0)
                allowed = max_speed_bh * (spans[c][0] - current_end) / fps + slack_bh
                entry["distance_bh"], entry["allowed_bh"] = round(dist_bh, 2), round(allowed, 2)
                if dist_bh > allowed:
                    entry["rejected"] = "too far to walk"
                    log.append(entry)
                    continue
            log.append(entry)
            scored.append((sim, c))
        passing = [(sim, c) for sim, c in scored if sim >= threshold]
        if not passing:
            break
        earliest = min(passing, key=lambda sc: (spans[sc[1]][0], -sc[0]))[1]
        a, b = spans[earliest]
        group = sorted([(sim, c) for sim, c in passing if spans[c][0] <= b and spans[c][1] >= a], reverse=True)
        if len(group) > 1 and group[0][0] - group[1][0] < margin:
            log.append({"after_frame": current_end, "abstained": [c for _, c in group[:2]],
                        "reason": "two simultaneous candidates look alike"})
            break
        best = group[0][1]
        next(e for e in log if e["candidate"] == best and e["after_frame"] == current_end)["linked"] = True
        chain.append(best)
        current_end = spans[best][1]
    return chain, log


def merge_chain(tracks: list[TrackOutput], chain: list[int]) -> list[TrackOutput]:
    """Relabel every track in the chain with the first ID, so downstream code
    (segments, timeline, clip) treats the re-linked target as one object."""
    members = set(chain)
    return [TrackOutput(t.frame, chain[0], t.box) if t.track_id in members else t for t in tracks]
