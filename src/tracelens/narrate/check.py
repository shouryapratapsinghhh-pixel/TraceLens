"""Faithfulness checker: is every sentence of a narrative backed by the event log?

Deterministic -- no LLM judges the LLM. Each sentence must cite events
(e.g. "[E4]"). Per sentence it flags:

  uncited           no citation at all -> nothing backs it
  invalid_citation  cites an event ID that isn't in the log
  wrong_time        mentions a time (e.g. "10.6s", "at 14 seconds") that matches
                    none of the cited events' times (within `time_tol_s`)
  wrong_action      describes an action (ran, loitered, left a bag, ...) that
                    none of the cited events is
  speculation       attributes intent or guilt ("suspiciously", "tried to",
                    "stole") -- the log records behaviour, never motive, and
                    in a surveillance tool asserting motive is a real harm

Limits, stated plainly: actions are recognised through a phrase lexicon, so
an invented action phrased in words outside it can slip through; the
checker is evaluated on a labelled sentence set (tests) so its accuracy is
known, not assumed.
"""

from __future__ import annotations

import re

ACTION_PATTERNS = {  # longest / most specific phrasings first where they overlap
    "object_left_behind": [r"left (?:a|an|the|his|her|their)? ?(?:bag|object|item|package|parcel)",
                           r"left .{0,20}behind", r"\bdropp?ed\b", r"\babandon", r"\bunattended\b",
                           r"put (?:down|a|an)"],
    "zone_enter": [(r"(?:entered|went into|stepped into|moved into|walked into|ran into) (?:the )?"
                    r"(?:restricted|zone|area)")],
    "zone_exit": [r"(?:left|exited|stepped out of|walked out of|came out of) (?:the )?(?:restricted|zone|area)"],
    "run": [r"\bran\b", r"\brunning\b", r"\bruns\b", r"\bsprint", r"\brushed\b", r"\bdashed\b"],
    "dwell": [r"\bloiter", r"\blinger", r"\bstood still\b", r"\bwaited\b", r"\bpaused\b",
              r"\bremained (?:still|stationary|in place)", r"\bstayed (?:still|put|in place)"],
    "appear": [r"\bappeared\b", r"came into view", r"\barrived\b", r"entered the (?:scene|frame|view)"],
    "disappear": [r"left (?:the )?(?:camera'?s? )?view", r"out of (?:the )?(?:frame|view)",
                  r"\bdisappeared\b", r"left the (?:scene|frame)", r"exited the (?:scene|frame)"],
}

SPECULATION = [r"suspicious", r"nervous", r"\btried to\b", r"\battempt", r"\bintend", r"\bplann",
               r"\bst[oe]al", r"\bstole\b", r"\btheft\b", r"\bthief\b", r"\battack", r"\bthreat",
               r"\bbomb", r"\bexplosiv", r"\bcriminal", r"\billegal", r"\bguilty\b", r"\bhiding\b",
               r"\bsneak", r"\bshady\b", r"\bmalicious\b", r"\bin order to\b", r"\bbecause (?:he|she|they)\b"]

TIME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:s\b|sec\b|secs\b|seconds?\b)", re.IGNORECASE)
CITE_RE = re.compile(r"\[E(\d+)\]")


def split_sentences(text: str) -> list[str]:
    """Split on . ! ? followed by whitespace (a citation like '[E3].' stays attached)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def actions_in(sentence: str) -> set[str]:
    s = sentence.lower()
    found = set()
    for kind, pats in ACTION_PATTERNS.items():
        if any(re.search(p, s) for p in pats):
            found.add(kind)
    # "An object appeared beside the target" is the left-behind event, not the TARGET appearing
    # (found by the first real LLM run: it paraphrased E5's own log line and got falsely flagged)
    if re.search(r"\b(?:object|bag|item|package|parcel)\b[^.]{0,20}\bappeared\b", s):
        found.add("object_left_behind")
        if not re.search(r"came into view|\barrived\b|entered the (?:scene|frame|view)|"
                         r"(?:target|person|it|they|he|she) appeared", s):
            found.discard("appear")
    # "left the restricted zone" should not ALSO count as "left view"/"left behind"
    if "zone_exit" in found:
        found.discard("disappear")
    if "zone_enter" in found:
        found.discard("appear")
    return found


def check_sentence(sentence: str, log_by_id: dict[str, dict], time_tol_s: float = 0.5) -> dict:
    cites = [f"E{n}" for n in CITE_RE.findall(sentence)]
    issues = []
    if not cites:
        issues.append("uncited")
    bad = [c for c in cites if c not in log_by_id]
    if bad:
        issues.append("invalid_citation")
    cited = [log_by_id[c] for c in cites if c in log_by_id]

    body = CITE_RE.sub("", sentence)
    if cited:
        cited_times = [e[k] for e in cited for k in ("t", "t_end", "t_confirmed") if k in e]
        for m in TIME_RE.finditer(body):
            if not any(abs(float(m.group(1)) - ct) <= time_tol_s for ct in cited_times):
                issues.append("wrong_time")
                break
        cited_types = {e["type"] for e in cited}
        if actions_in(body) - cited_types:
            issues.append("wrong_action")
    if any(re.search(p, body.lower()) for p in SPECULATION):
        issues.append("speculation")
    return {"sentence": sentence, "citations": cites, "issues": issues, "supported": not issues}


def check_narrative(text: str, log: list[dict], time_tol_s: float = 0.5) -> dict:
    by_id = {e["id"]: e for e in log}
    results = [check_sentence(s, by_id, time_tol_s) for s in split_sentences(text)]
    cited_ids = {c for r in results for c in r["citations"] if c in by_id}
    n = len(results)
    return {
        "sentences": results,
        "n_sentences": n,
        "supported_rate": sum(r["supported"] for r in results) / n if n else 0.0,
        "coverage": len(cited_ids) / len(log) if log else 1.0,  # share of events the story mentions
        "issue_counts": {k: sum(k in r["issues"] for r in results) for k in
                         ("uncited", "invalid_citation", "wrong_time", "wrong_action", "speculation")},
    }


def verified_text(report: dict) -> str:
    """Only the sentences that passed every check."""
    return " ".join(r["sentence"] for r in report["sentences"] if r["supported"])
