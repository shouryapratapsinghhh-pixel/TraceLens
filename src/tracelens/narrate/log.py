"""The event log the narrator is allowed to see -- and nothing else.

Each event gets a stable ID (E1, E2, ...) that the narrator must cite, so
every sentence it writes can be traced back to a logged fact.
"""

from __future__ import annotations

DESCRIPTIONS = {
    "appear": "target came into view",
    "disappear": "target left the camera's view",
    "dwell": "target stood still (loitered)",
    "run": "target moved fast (ran)",
    "zone_enter": "target entered zone '{zone}'",
    "zone_exit": "target left zone '{zone}'",
    "object_left_behind": "an object appeared beside the target and stayed after the target walked away",
}


def build_log(events: list[dict]) -> list[dict]:
    log = []
    for i, e in enumerate(sorted(events, key=lambda e: e["t"]), 1):
        entry = {"id": f"E{i}", "type": e["type"], "t": round(float(e["t"]), 1)}
        for key in ("t_end", "t_confirmed"):
            if key in e:
                entry[key] = round(float(e[key]), 1)
        if "zone" in e:
            entry["zone"] = e["zone"]
        log.append(entry)
    return log


def log_text(log: list[dict]) -> str:
    """One line per event, e.g. 'E3 | 8.9s-10.6s | target moved fast (ran)'."""
    lines = []
    for e in log:
        when = f"{e['t']:.1f}s" + (f"-{e['t_end']:.1f}s" if "t_end" in e else "")
        desc = DESCRIPTIONS[e["type"]].format(zone=e.get("zone", ""))
        if "t_confirmed" in e:
            desc += f" (confirmed at {e['t_confirmed']:.1f}s)"
        lines.append(f"{e['id']} | {when} | {desc}")
    return "\n".join(lines)
