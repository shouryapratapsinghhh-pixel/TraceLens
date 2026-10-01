"""TraceLens demo -- results and recorded outputs; no model runs on the host.

Run locally from the repo root:   streamlit run demo/streamlit_app.py
Deploy (Streamlit Community Cloud): main file path = demo/streamlit_app.py
demo/requirements.txt is used, so hosting installs only streamlit/pandas/numpy.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(os.environ.get("TRACELENS_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "src"))  # the checker is pure Python: no torch/opencv needed

from tracelens.narrate.check import check_narrative

ASSETS = ROOT / "demo" / "assets"
STUDY = ROOT / "reports" / "tracker_study_cmc"
MOVING_CAMERA = {"MOT17-05", "MOT17-10", "MOT17-11", "MOT17-13"}
ISSUE_TEXT = {"uncited": "no event cited", "invalid_citation": "cites an event that doesn't exist",
              "wrong_time": "time doesn't match the cited event", "wrong_action": "action doesn't match the cited event",
              "speculation": "guesses intent or guilt"}


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def sentence_table(report: dict) -> pd.DataFrame:
    return pd.DataFrame([{"": "✅" if r["supported"] else "❌", "sentence": r["sentence"],
                          "problem": ", ".join(ISSUE_TEXT[i] for i in r["issues"])} for r in report["sentences"]])


st.set_page_config(page_title="TraceLens", layout="wide")
st.title("TraceLens")
st.markdown("**Follow one person through CCTV footage — and get a short clip plus a timeline of where they went "
            "and what they did, narrated by a language model that may only state what the tracker actually saw.**")
st.caption("This page shows saved results and recorded outputs. Free hosting can't run the detector, tracker on "
           "full video, or the language model, so nothing here is computed live except the checker on the last tab.")

tab_res, tab_inc, tab_nar, tab_try = st.tabs(["Results on real footage", "Incident demo", "Grounded narration",
                                              "Try the checker"])

with tab_res:
    res = pd.read_csv(STUDY / "test_results.csv") if (STUDY / "test_results.csv").exists() else None
    decision = load_json(STUDY / "decision.json")
    if res is None:
        st.info("Run `python -m tracelens.tracker_study --root data/raw/MOT17/train --out reports/tracker_study_cmc`.")
    else:
        means = res.groupby("method", sort=False)[["mota", "idf1"]].mean()
        ids = res.groupby("method", sort=False)["id_switches"].sum()
        cmc = next(m for m in means.index if "CMC" in m)
        base = next(m for m in means.index if m.startswith("baseline"))
        st.subheader("MOT17, held-out test sequences, official scoring")
        c1, c2, c3 = st.columns(3)
        c1.metric("IDF1 (identity)", f"{means.loc[cmc, 'idf1']:.3f}", f"{means.loc[cmc, 'idf1'] - means.loc[base, 'idf1']:+.3f} vs baseline")
        c2.metric("MOTA", f"{means.loc[cmc, 'mota']:.3f}", f"{means.loc[cmc, 'mota'] - means.loc[base, 'mota']:+.3f} vs baseline")
        c3.metric("ID switches", f"{ids[cmc]:,}", f"{(ids[cmc] - ids[base]) / ids[base]:+.0%} vs baseline",
                  delta_color="inverse")
        per = res.pivot(index="sequence", columns="method", values="idf1")
        per.index = [f"{s.rsplit('-', 1)[0]} ({'moving' if s.rsplit('-', 1)[0] in MOVING_CAMERA else 'static'} camera)"
                     for s in per.index]
        st.markdown("**IDF1 per test sequence.** Camera-motion compensation gains only where the camera moves; "
                    "static-camera sequences are unchanged.")
        st.bar_chart(per)
        st.dataframe(per.round(3), use_container_width=True)
        if decision:
            st.markdown(f"**How the setting was chosen** — on {', '.join(decision['tune_sequences'])} only; "
                        f"the sequences above were never used to decide anything. Rule: *{decision['rule']}*. "
                        f"Tune-set IDF1 {decision['tune_idf1_without_cmc']:.3f} → {decision['tune_idf1_with_cmc']:.3f} "
                        f"⇒ CMC {'kept' if decision['cmc_kept'] else 'dropped'}.")
        st.caption("Scoring uses MOTChallenge's official preprocessing, verified to give identical MOTA/IDF1 to the "
                   "official TrackEval toolkit. Public detections; for context, published public-detection trackers "
                   "sit in the same range (e.g. Tracktor: MOTA 53.5 / IDF1 52.3 on the MOT17 test set — different "
                   "sequences, so 'same range', not 'beats').")

with tab_inc:
    st.subheader("A scripted incident (synthetic — every event's true time is known)")
    gif = ASSETS / "incident.gif"
    col_v, col_e = st.columns([3, 2])
    if gif.exists():
        col_v.image(str(gif), use_container_width=True)
    events = load_json(ASSETS / "events.json")
    if events:
        col_e.markdown("**Event log** — the only thing the narrator sees")
        col_e.code(events["text"], language=None)
        col_e.caption("Speed in body-heights per second (works at any distance from the camera); zones use the feet "
                      "point. All 7 events found with zero false alarms; an innocent walker triggers none.")

with tab_nar:
    nar = load_json(ASSETS / "narratives.json")
    if nar:
        st.subheader("A real language model's report, checked sentence by sentence")
        llm = nar["llm"]
        st.markdown(f"*{llm['source']}* — recorded on a laptop; the checker below runs on its exact text.")
        st.write(llm["text"])
        chk = llm["check"]
        st.markdown(f"**{chk['supported_rate']:.0%}** of sentences supported by the log · "
                    f"**{chk['coverage']:.0%}** of events covered")
        st.dataframe(sentence_table(chk), use_container_width=True, hide_index=True)
        st.caption("The checker is deterministic (no LLM judges the LLM). On 25 labelled sentences it never flagged a "
                   "faithful one (precision 1.00) and caught 12 of 13 bad ones (recall 0.92); its known blind spot is "
                   "an invented action worded outside its vocabulary.")

with tab_try:
    st.subheader("Write a report sentence — the checker verifies it against the event log")
    events = load_json(ASSETS / "events.json")
    examples = {
        "Faithful": "At 10.6s it entered the restricted zone [E4].",
        "Wrong time": "It ran from 3.0s to 5.0s [E3].",
        "Guessing intent": "It nervously paced near the door, planning a theft [E2].",
        "No citation": "The target entered the restricted zone.",
        "Known blind spot": "It climbed the fence at 10.6s [E4].",
    }
    pick = st.radio("Start from an example", list(examples), horizontal=True)
    text = st.text_area("Report text (cite events like [E3])", examples[pick], height=90)
    if events and text.strip():
        rep = check_narrative(text, events["log"])
        st.dataframe(sentence_table(rep), use_container_width=True, hide_index=True)
        if pick == "Known blind spot":
            st.info("This passes — 'climbed the fence' isn't in the checker's action vocabulary. Stated openly as "
                    "the checker's limitation rather than hidden.")
        with st.expander("Event log"):
            st.code(events["text"], language=None)
