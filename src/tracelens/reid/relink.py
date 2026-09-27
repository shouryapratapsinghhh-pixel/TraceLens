"""Re-link a target that left view and came back under a new track ID.

Starting from the selected track, repeatedly:
  1. candidates = tracks that START after the chain's current track ENDS
     (the same object can't be in two places at once) and within
     `max_gap_frames` of that end;
  2. score each candidate's appearance fingerprint against the TARGET's
     original fingerprint (not the latest link's -- comparing to the latest
     would let errors drift: one bad link would make the next bad one look right);
  3. link the most similar candidate if similarity >= `threshold`
     (ties -> the earliest to appear), then continue from it.
Stops when no candidate clears the threshold.

Returns the chain of track IDs and a log of every decision, so each link is
auditable ("why did it merge these?").
"""

from __future__ import annotations

from tracelens.reid.appearance import similarity
from tracelens.track.tracker import TrackOutput


def track_spans(tracks: list[TrackOutput]) -> dict[int, tuple[int, int]]:
    spans: dict[int, tuple[int, int]] = {}
    for t in tracks:
        a, b = spans.get(t.track_id, (t.frame, t.frame))
        spans[t.track_id] = (min(a, t.frame), max(b, t.frame))
    return spans


def relink_chain(tracks: list[TrackOutput], fingerprints: dict, target_id: int,
                 threshold: float = 0.8, max_gap_frames: int = 150) -> tuple[list[int], list[dict]]:
    spans = track_spans(tracks)
    chain, log = [target_id], []
    ref = fingerprints[target_id]
    current_end = spans[target_id][1]
    while True:
        cands = [tid for tid, (start, _) in spans.items()
                 if tid not in chain and current_end < start <= current_end + max_gap_frames]
        scored = sorted(((similarity(ref, fingerprints[c]), -spans[c][0], c) for c in cands), reverse=True)
        for sim, _, c in scored:
            log.append({"after_frame": current_end, "candidate": c, "similarity": round(sim, 3),
                        "linked": False})
        if not scored or scored[0][0] < threshold:
            break
        sim, _, best = scored[0]
        next(e for e in log if e["candidate"] == best and e["after_frame"] == current_end)["linked"] = True
        chain.append(best)
        current_end = spans[best][1]
    return chain, log


def merge_chain(tracks: list[TrackOutput], chain: list[int]) -> list[TrackOutput]:
    """Relabel every track in the chain with the first ID, so downstream code
    (segments, timeline, clip) treats the re-linked target as one object."""
    members = set(chain)
    return [TrackOutput(t.frame, chain[0], t.box) if t.track_id in members else t for t in tracks]
