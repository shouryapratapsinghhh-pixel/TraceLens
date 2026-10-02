# TraceLens

**Follow one person through CCTV footage, and get a short clip plus a timeline of where they went
and what they did, narrated by a language model that may only state what the tracker actually saw.**

**[Live demo →](https://tracelens-shouryapratapsingh.streamlit.app/)**  ·  *Built by
[Aarohi Gaurav Sharma](https://www.linkedin.com/in/aarohig-sharma22/)*

---

## Headline results

| | Result |
|---|---|
| **Tracking, real footage** | **IDF1 0.598 · MOTA 0.528** on held-out MOT17 sequences (public detections, official protocol) |
| **Camera-motion compensation** | IDF1 **0.536 → 0.598**, ID switches **−29%** (627 → 444); gains only on moving-camera sequences |
| **Scoring** | Verified **identical to the official MOTChallenge toolkit (TrackEval)** on MOTA, IDF1, ID switches, FP and FN |
| **Grounded narration** | Hallucination checker: precision **1.00**, recall **0.92** (25 labelled sentences); real LLM output **100%** verified |
| **Event detection** | 7/7 scripted events, **0 false alarms**, robust to 10% detector misses (synthetic incident) |

Every tracking decision was made on two tuning sequences and reported only on five held-out ones.

## What it does

1. **Detect & track** every person, with a tracker written from scratch: Kalman filter + Hungarian
   matching, an optional ByteTrack mode, and camera-motion compensation.
2. **Pick a target** by clicking on them.
3. **Summarize**: cut only the moments the target appears, with original timestamps burned in.
4. **Re-link** the target when they leave view and come back under a new track ID.
5. **Detect events**: loitering, running, entering/leaving a restricted zone, leaving an object behind.
6. **Narrate**: a local LLM writes the incident report from the event log only, citing an event in
   every sentence; a deterministic checker removes any sentence the log doesn't support.

---

## Results in detail

### 1. Tracking on real footage (MOT17)

Public FRCNN detections, official scoring. Tuned on MOT17-02 + -05; **test sequences only** below
(`reports/tracker_study_cmc/`).

| Test sequences (5) | MOTA | IDF1 | ID switches |
|---|---|---|---|
| Baseline | 0.515 | 0.536 | 627 |
| Tuned settings | 0.511 | 0.536 | 751 |
| **Tuned + camera-motion compensation** | **0.528** | **0.598** | **444** |

| IDF1 per sequence | Camera | Baseline | Tuned | Tuned + CMC |
|---|---|---|---|---|
| MOT17-04 | static | 0.619 | 0.661 | 0.661 |
| MOT17-09 | static | 0.609 | 0.589 | 0.589 |
| MOT17-10 | moving | 0.386 | 0.392 | **0.540** |
| MOT17-11 | moving | 0.562 | 0.590 | **0.620** |
| MOT17-13 | moving | 0.503 | 0.446 | **0.577** |

Camera-motion compensation (background corners + Lucas-Kanade optical flow + a RANSAC similarity
fit) changes nothing on static cameras and lifts every moving-camera sequence, which is the
pattern you'd expect if CMC is doing the work. It was kept by a rule fixed in advance (keep only
if tune-set IDF1 rises: 0.502 → 0.508; `reports/tracker_study_cmc/decision.json`).

**Context:** published trackers on public detections sit in the same range (e.g. Tracktor,
reported at MOTA 53.5 / IDF1 52.3 on the MOT17 test set). These are different sequences, so
"same range", not "beats". Leaderboard scores of 70-80+ use detectors trained on pedestrian
data, a different category.

### 2. What didn't help, and why

| Change | Test IDF1 | Verdict |
|---|---|---|
| ByteTrack / lower detection cutoffs (public detections) | 0.536 (= baseline) | No gain: the public detections are pre-filtered (median score 1.00, only 8% below 0.5), so there are no weak boxes to use |
| YOLOv8m (COCO weights), 640 px | 0.496 | Worse than public detections: 640 px shrinks a 1920-px frame 3×, so far pedestrians vanish |
| YOLOv8m, 1280 px | 0.529 | The tune set picked 640 (0.479 vs 0.461); the test set preferred 1280. The rule wasn't overridden after seeing test numbers |

### 3. Target summaries and re-linking on real footage

When a person leaves view, the tracker gives them a new ID and the summary loses them. Re-linking
by appearance (top/bottom colour fingerprint) recovers them on synthetic data, but **didn't
transfer to real footage**. Results for 25 people on the test sequences who left view at least once:

| Method | Identity F1 | Identity precision | Wrong links per target |
|---|---|---|---|
| No re-linking | 0.16 | 0.49 | 0 |
| Original re-linker | 0.28 | 0.30 | 3.64 |
| Tuned re-linker (ambiguity check) | 0.18 | 0.48 | 0.12 |

**Diagnosis:** the bottleneck is upstream. Even the single clicked track is on the right person
only 49% of the time, and each person splits into about 3.5 IDs. Colour histograms can't separate
real people in similar dark clothing. The fix is a learned re-ID model (see Roadmap).

### 4. Event detection (scripted synthetic incident)

The target walks in, loiters, runs into a restricted zone, drops a bag and leaves. The true
events come from the rendered geometry. Speed is measured in **body-heights per second** (one
threshold works at any camera distance); zones use the **feet point**.

| Detector noise | Event F1 (mean, 5 runs) | Worst run | Timing error | Innocent-walker false alarms |
|---|---|---|---|---|
| clean | 1.00 | 1.00 | 0.21 s | 0 |
| 10% misses, 2 px jitter | 1.00 | 1.00 | 0.23 s | 0 |
| 20% misses, 3 px jitter | 0.90 | 0.75 | 0.21 s | 0 |
| 30% misses, 4 px jitter | 0.82 | 0.53 | 0.25 s | 0 |

### 5. Grounded narration with a hallucination checker

A local LLM (Qwen2.5-1.5B-Instruct) sees **only the event log** and must cite an event in every
sentence. A deterministic checker (no LLM judging the LLM) flags sentences that are uncited, cite a
missing event, state a wrong time or action, or **speculate about intent** ("suspiciously",
"tried to steal"). A surveillance tool asserting motive is a real harm.

- **Checker accuracy** on 25 labelled sentences: precision **1.00**, recall **0.92**. Its one miss is
  a planted probe ("climbed the fence"), an invented action outside its vocabulary.
- **Real LLM run:** all 7 events covered, **100%** of sentences supported.

### 6. Correctness checks

- Tracking metrics match `motmetrics` exactly; official-protocol scoring matches **TrackEval**
  exactly (`tests/test_trackeval_parity.py`).
- When two identical objects cross paths, the Kalman tracker never swapped identities (8/8 runs);
  without motion prediction it swapped in 3/8.
- Synthetic shaking camera: CMC lifts MOTA 0.32 → 0.83 and cuts ID switches 6×.
- **97 tests** (`pytest -q`).

---

## How the evaluation was kept honest

- **Tune/test split fixed before any result:** tune on MOT17-02 (static) + -05 (moving); report
  only on 04, 09, 10, 11, 13. Decisions are saved to `decision.json`.
- **Official protocol, verified:** predictions on reflections, static people and people on vehicles
  are ignored exactly as TrackEval does, confirmed by a parity test against the real toolkit.
- **Disclosed changes:** the re-linking objective was changed from frame-level to identity-level F1
  *after* the first test results were seen; both passes are reported.
- **Negative results are reported**, not dropped (Section 2).

## Mistakes caught along the way

1. **The test detector leaked identity:** it returned boxes in ground-truth order, and tie-breaking
   silently used it. Fixed by shuffling, as a real detector would.
2. **Re-linking stranded track fragments:** linking the *most similar* fragment jumped ahead and
   deleted five events. Now links in time order.
3. **Recall was inflated by swallowing the video:** merging strangers grew the "summary" to 70% of
   the footage, which contained the target by accident. Caught by precision and the wrong-link count.
4. **Frame-level F1 was blind to identity:** it scored "target somewhere in frame", so following a
   stranger in a crowd still counted. Replaced with identity-level metrics.
5. **Target selection picked people who never left view,** where re-linking has nothing to do.
6. **The checker falsely flagged a faithful LLM sentence** ("an object appeared..."): real output
   exposed what hand-written tests hadn't.
7. **CMC corner starvation:** high-contrast people out-scored the background, so corners were found
   only on moving objects and the estimate silently fell back to "no motion". Fixed by masking
   detected boxes.
8. **Scene bugs in my own tests:** a target that never entered its zone, and one frozen at the frame
   edge. Caught by checking rendered positions instead of trusting the script.
9. **YOLO at 640 px lost small pedestrians.**
10. **The tuning set was too small:** three decisions were made by narrow margins (below).

## Limitations

- **Two tuning sequences are too few.** Three decisions hinged on narrow margins; a larger split or
  cross-validation over sequences would be better.
- **Public detections only** for the headline; the off-the-shelf YOLO wasn't trained on pedestrians.
- **Re-linking on real footage is weak** (identity F1 ≤ 0.28); colour fingerprints don't transfer.
- **Events are evaluated on a synthetic incident**, not real footage (MOT17 has no event labels).
- **Narration is evaluated on one incident**; the checker's lexicon has a known blind spot.
- **The hosted demo shows saved results**: free hosting can't run the detector or the LLM.

## Roadmap

- Appearance-aware tracking with a learned re-ID model (DeepSORT-style), the real fix for identity.
- A pedestrian-trained detector that **never saw MOT17** (public MOT17-trained weights would leak).
- Cross-validation over sequences instead of a fixed 2-sequence tuning split.
- Event detection evaluated on real labelled footage; text queries ("red top, blue bottom").

---

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements.txt
pytest -q

# real footage: download MOT17 (see data/README.md), then
python -m tracelens.mot_eval --root data/raw/MOT17/train
python -m tracelens.tracker_study --root data/raw/MOT17/train --out reports/tracker_study_cmc
python -m tracelens.relink_study --root data/raw/MOT17/train --out reports/relink_study_pass2

# synthetic demos
python -m tracelens.events_demo --save outputs/incident.mp4
python -m tracelens.narrate_demo                      # template narrator
pip install -e ".[llm]" && python -m tracelens.narrate_demo --narrator llm

# demo app
python -m tracelens.demo_assets && streamlit run demo/streamlit_app.py
```

Optional: `pip install ultralytics` and `python -m tracelens.detect_cache` for YOLO detections;
`pip install git+https://github.com/JonathonLuiten/TrackEval.git` to run the official-parity test.

## Project layout

```
src/tracelens/
  track/tracker.py      Kalman filter + Hungarian tracker, ByteTrack mode
  track/cmc.py          camera-motion compensation
  eval/mot.py           MOTA / IDF1 from scratch
  eval/summary.py       target recall/precision, identity metrics
  video/mot.py          MOT17 loader, official-protocol filter
  reid/                 appearance fingerprints, re-linking
  events/detect.py      loitering, running, zones, left-behind objects
  narrate/              event log, narrators, faithfulness checker
  tracker_study.py      tune/test tracker study
  relink_study.py       tune/test re-linking study
demo/streamlit_app.py   the demo (streamlit + pandas only)
```

## Responsible use

Built and evaluated on synthetic data and public research datasets only; no MOT17 frames are
redistributed. No face recognition: tracking uses whole-body position and appearance. The narrator
is forbidden from stating intent or guilt, and the checker enforces it. Surveillance technology can
be misused; this is a research tool for reducing review time, not for identifying individuals.
