"""Evaluate the faithfulness checker itself on a labelled sentence set.

Each case: (sentence, should_be_flagged). The log is the incident's. Cases
include paraphrases that SHOULD pass (so the checker isn't just rejecting
everything unusual), confusable phrasings, a sentence taken verbatim from a
real LLM run (which exposed a false flag), and one invented action worded
outside the checker's lexicon ("climbed the fence") -- a deliberate probe of
its known blind spot. The metrics count it honestly.
"""

from __future__ import annotations

from tracelens.narrate.check import check_sentence
from tracelens.narrate.log import build_log

INCIDENT_EVENTS = [
    {"type": "appear", "t": 1.2}, {"type": "dwell", "t": 4.1, "t_end": 8.6},
    {"type": "run", "t": 8.9, "t_end": 10.6}, {"type": "zone_enter", "t": 10.6, "zone": "restricted"},
    {"type": "object_left_behind", "t": 14.6, "t_confirmed": 16.2},
    {"type": "zone_exit", "t": 15.0, "zone": "restricted"}, {"type": "disappear", "t": 22.8},
]

CASES: list[tuple[str, bool]] = [
    # faithful, incl. paraphrases -- should PASS
    ("The target came into view at 1.2s [E1].", False),
    ("A person arrived at 1.2 seconds [E1].", False),
    ("It lingered near the entrance for a while [E2].", False),
    ("From 4.1s to 8.6s it stood still [E2].", False),
    ("It then ran [E3].", False),
    ("At 10.6s the target entered the restricted zone [E4].", False),
    ("It dropped an object at 14.6s [E5].", False),
    ("The object was confirmed left behind at 16.2s [E5].", False),
    ("At 15.0s it left the restricted zone [E6].", False),
    ("It left the camera's view at 22.8s [E7].", False),
    ("It ran into the restricted zone at 10.6s [E3][E4].", False),
    # verbatim from the first real Qwen2.5-1.5B-Instruct run -- faithful, and was falsely flagged
    ("An object appeared beside the target and remained there until confirmed at 16.2s [E5].", False),
    # unsupported -- should be FLAGGED
    ("The target came into view at 3.0s [E1].", True),            # wrong time
    ("It ran from 12.0s to 14.0s [E3].", True),                    # wrong time
    ("It loitered in the restricted zone [E4].", True),            # wrong action for E4
    ("At 10.6s it left the restricted zone [E4].", True),          # enter/exit swapped
    ("It dropped a bag [E6].", True),                              # wrong event cited
    ("The target entered the restricted zone.", True),             # uncited
    ("It left an object behind [E9].", True),                      # invalid citation
    ("It suspiciously looked around before running [E3].", True),  # speculation
    ("The man tried to steal something [E5].", True),              # speculation
    ("It ran away at 22.8s [E7].", True),                          # wrong action (E7 = left view)
    ("It left view at 15.0s [E6].", True),                         # zone exit != left view
    ("The target appeared at 14.6s [E5].", True),                  # the TARGET didn't appear at E5
    ("It climbed the fence at 10.6s [E4].", True),                 # invented action, OUTSIDE lexicon
]


def evaluate_checker() -> dict:
    by_id = {e["id"]: e for e in build_log(INCIDENT_EVENTS)}
    tp = fp = fn = tn = 0
    misses, false_flags = [], []
    for sentence, should_flag in CASES:
        flagged = not check_sentence(sentence, by_id)["supported"]
        if flagged and should_flag:
            tp += 1
        elif flagged:
            fp += 1
            false_flags.append(sentence)
        elif should_flag:
            fn += 1
            misses.append(sentence)
        else:
            tn += 1
    return {"n": len(CASES), "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 1.0, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "missed": misses, "false_flags": false_flags}


if __name__ == "__main__":
    import json

    print(json.dumps(evaluate_checker(), indent=2))
