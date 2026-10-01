# TraceLens

**Follow one person or object through hours of CCTV footage, and get a short clip plus a
plain-language timeline of where they went and what they did.**

Surveillance systems record far more video than anyone can watch. Standard video
summarization shortens a whole video; TraceLens summarizes it *around a target you choose*:
detect → track → pick the target → cut only the moments it appears → describe its path
and key events in text, with a language model that may only describe events the tracker
actually saw.

## Status

**Phase 0 (done):** synthetic test videos with exact ground truth; a detector interface
(controlled-noise synthetic detector + optional YOLO); a multi-object tracker written from
scratch (Kalman filter + Hungarian matching); tracking metrics from scratch (MOTA, IDF1),
verified to match the reference `motmetrics` library exactly.

**Phase 1 (done):** click-to-select a target; its appearances become clip segments (small
gaps merged, fragments dropped, context padded); a summary video of only those moments with
original timestamps burned in; a fact-only timeline; and scoring against ground truth
(target recall, precision, compression).

**Phase 2 (done): re-linking a returning target.** When the target leaves view, the tracker
gives it a new ID on return and a one-ID summary loses the second visit (recall 0.50).
TraceLens re-links it by appearance: a colour fingerprint of the box, compared only against
tracks that start after the target's last sighting (the same object can't be in two places),
always against the ORIGINAL target (so errors can't drift link by link), with every decision
logged. Tested against a look-alike: same two colours, swapped top and bottom, appearing
while the target is away.

| Scene | Re-linking | Target recall | Wrong merges |
|---|---|---|---|
| Target leaves and returns | none | 0.50 | 0 |
| Target leaves and returns | top/bottom fingerprint | 0.99 | 0 |
| Look-alike present | whole-box fingerprint | 0.99 | merges the look-alike every run |
| Look-alike present | top/bottom fingerprint | 0.99 | 0 |

Whole-box colour can't tell "red shirt, blue trousers" from "blue shirt, red trousers"
(similarity 1.00); a top/bottom fingerprint separates them (0.00). Recall alone hides the
wrong merge; precision and the wrong-merge count expose it.
(Synthetic data, 20% detector misses, 5 runs; regenerate with `--compare-relink`.)

**Phase 3 (done): events.** From the (re-linked) target track: appear / disappear, loitering,
running, entering / leaving a named zone, and leaving an object behind. Speed is measured in
**body-heights per second** so one threshold works at any distance from the camera; zones use
the **feet point**, where the person actually stands. Scored against a scripted incident whose
ground-truth events are derived from the rendered geometry, not typed by hand.

| Detector noise | Event F1 (mean, 5 runs) | Worst run | Timing error | Innocent walker false alarms |
|---|---|---|---|---|
| clean | 1.00 | 1.00 | 0.21 s | 0 |
| 10% misses, 2 px jitter | 1.00 | 1.00 | 0.23 s | 0 |
| 20% misses, 3 px jitter | 0.90 | 0.75 | 0.21 s | 0 |
| 30% misses, 4 px jitter | 0.82 | 0.53 | 0.25 s | 0 |

(Regenerate with `python -m tracelens.events_demo --sweep`.)

Found by this evaluation: a re-linking bug. Under heavy noise the target broke into 5 track
fragments, and linking the *most similar* fragment jumped ahead and stranded the ones in
between, deleting five events. Re-linking now follows fragments in time order (the similarity
threshold is what rejects other people). A regression test pins it.

Known limitations under heavy noise: gaps longer than 1 s between fragments read as the target
leaving and returning (extra appear/disappear); "left behind" is the most fragile event,
because the *object's* own track fragments too and only the target is re-linked; long
detection gaps can split a loiter or run below its minimum duration.

**Phase 4 (done): grounded narration with a faithfulness checker.** A local LLM
(Qwen2.5-1.5B-Instruct) writes the incident report, but sees **only the event log**, and must
cite an event ID in every sentence ("It ran from 8.9s to 10.6s [E3]."). A deterministic checker,
with no LLM judging the LLM, flags any sentence that is uncited, cites a non-existent event,
states a time or action the cited event doesn't support, or **speculates about intent**
("suspiciously", "tried to steal"). The log records behaviour, never motive, and a surveillance
tool asserting motive is a real harm. Failing sentences are dropped from the verified version.

**First real run (Qwen2.5-1.5B-Instruct, the scripted incident):** the narrative was fully
faithful: all 7 events covered, correct times, no invented actions, no speculation. It even
merged three events into one sentence with three correct citations. The only flag was the
**checker's** mistake: "An object appeared beside the target..." matched its pattern for the
*target* appearing. Real LLM phrasing exposed a false flag that my hand-written test sentences
hadn't. Fixed, and that exact sentence is now a regression case.

The checker is itself evaluated on 25 labelled sentences (including that real one):
**precision 1.00** (never flags a faithful sentence) and **recall 0.92**. Its one miss is a
planted probe: an invented action worded outside its phrase lexicon ("climbed the fence"), the
checker's known blind spot. One narrative from one incident is a sample of one; faithfulness
across many incidents is still to be measured.

**Phase 5: real footage (MOT17).** A MOT17 loader (lazy frame reading,
since 1080p sequences don't fit in memory; pedestrian-only ground truth; MOT17's own
visibility labels), public detections or YOLO, and `mot_eval`: tracking metrics for every
pedestrian, plus target summaries for real people who left view for over 1 s at least once,
with and without re-linking. Tested end to end on a MOT-format sequence built on the fly; the
real MOT17 numbers come from a local run (see `data/README.md`).

Caveat: predictions on MOT17's distractor classes (reflections, static people) count as false
positives here; the official toolkit ignores them, so this MOTA is slightly harsher than an
official score.

**First real results** (7 MOT17-FRCNN training sequences, public detections; from
`reports/mot17/tracking.csv` and `targets.csv`):

- **Tracking:** mean MOTA 0.46, IDF1 0.51, 728 ID switches. Misses dominate (heavily occluded
  people the public detector never found). The two moving-camera sequences (10, 13) have the
  most ID switches: a constant-velocity Kalman filter in image coordinates breaks when the
  camera itself moves (fix: camera-motion compensation).
- **Re-linking failed to transfer from synthetic to real.** For 32 people who left view at
  least once, it raised target recall from 0.40 to 0.85, but made **3.6 wrong links per target
  against 0.26 correct**, and the "summary" grew to 70% of the video. The recall gain was mostly
  the summary swallowing the video, the same metric trap as point-adjusted F1. Top/bottom colour
  histograms separate synthetic coloured boxes, but not real people in similar dark clothing
  under changing light.

**Phase 5b: fixing re-linking honestly.** Two rules: a **motion gate** (a candidate must start
within walking distance, max speed × time gone, in body-heights) and an **ambiguity check**
(abstain when two simultaneous candidates look nearly alike). Tuned on MOT17-02 and -05 only;
reported only on the other five sequences.

*First pass* (objective: frame-level summary F1; from `reports/relink_study/`), test sequences,
25 people:

| Method | Recall | Correct links | Wrong links | Summary F1 |
|---|---|---|---|---|
| No re-linking | 0.26 | 0 | 0 | 0.33 |
| Original re-linker | 0.70 | 0.72 | 3.64 | 0.64 |
| Tuned re-linker | 0.38 | 0.40 | 0.40 | 0.44 |

Tuning cut wrong links 9× and still beat no re-linking. The ambiguity check did the work; the
motion gate didn't help and tuning dropped it. But the original re-linker "won" on summary F1
despite 3.6 wrong links per target, which exposed a flaw in the **objective itself**: frame-level
F1 asks whether the target is *somewhere in the frame*, not whether the *highlighted box is the
target*, and in a crowd the real target is usually on screen while the summary follows a stranger.

*Second pass (built; results pending):* **identity metrics** that compare highlighted boxes
with the target's true boxes, with identity F1 as the objective. **Disclosure:** the objective
was changed after the first test results were seen, which is a form of test-set feedback. The
change is principled (identity is what "follow this person" means), and both passes are reported.

**Phase 6: improving tracking accuracy (in progress).** Each change is tuned on MOT17-02/05 and
reported only on the other five sequences.

- *ByteTrack association and lower detection cutoffs, on public detections:* **no gain.** The
  best tuned setting matched the baseline's test IDF1 exactly (0.531 vs 0.531; MOTA 0.495 vs
  0.499), winning on 3 test sequences and losing on 2. The whole 24-setting tuning grid sat within
  0.47-0.49 IDF1. When no tracking change moves the result, tracking isn't the bottleneck:
  **detections are** (about 39,500 misses vs 4,000 false positives on the test sequences).
  From `reports/tracker_study/`.
- *YOLOv8m (COCO weights, private detections), default 640-px input:* **worse** than the public
  detections: test IDF1 0.482 vs 0.531, MOTA 0.405 vs 0.499, and more misses (~46,600 vs
  ~39,500). The likely cause: 640 px shrinks a 1920-px frame 3x, and far pedestrians vanish.
  ByteTrack finally had weak boxes to use (70% of YOLO's scores are below 0.5) but still came
  second on the tune set (0.470 vs 0.475 IDF1). From `reports/tracker_study_yolo/`.
- *YOLOv8m at 1280 px:* by the pre-committed rule it **loses** (best tune IDF1 0.456 vs 0.475 at
  640), so 640 stays. Disclosed: the test set disagreed (1280 tuned: test IDF1 0.510 vs 0.482). In
  hindsight one tune sequence (MOT17-05) is only 640x480, so "1280 px" enlarges it instead of
  preserving detail. The rule wasn't overridden after seeing test numbers.
  From `reports/tracker_study_yolo1280/`.
- *Official scoring:* all numbers above were scored **strictly**, counting predictions on
  reflections, static people and people on vehicles as false positives. The official MOTChallenge
  evaluation ignores those. `official_filter` implements its preprocessing, read directly from the
  official TrackEval code, and gives **identical MOTA, IDF1, ID switches, FP and FN to TrackEval**
  (`tests/test_trackeval_parity.py`, run when TrackEval is installed). Official scoring is now the
  default (`--protocol strict` reproduces the old numbers). All studies are being re-scored.
- *Re-scored with the official protocol* (test sequences; `reports/*_official/`): public detections,
  baseline **MOTA 0.515, IDF1 0.536**, still the best setup. Association tuning still changes
  nothing (0.536 both ways), now explained: the public detections have a median score of 1.00
  and only 8% below 0.5, so they're pre-filtered and there are no weak boxes for ByteTrack to
  use. YOLO 640 vs 1280: the tune set picks 640 again and the test set prefers 1280 again, a
  consistent sign that two tuning sequences are too few to settle this. The rule stands.
  For context, published public-detection trackers sit in the same range (e.g. Tracktor, reported
  at MOTA 53.5 / IDF1 52.3 on the MOT17 test set; different sequences, so "same range", not "beats").
- **Camera-motion compensation: the real gain.** Background corners + Lucas-Kanade optical flow +
  a RANSAC similarity fit give one global camera transform per frame, and every Kalman prediction
  moves with it. Kept by the tune-set rule; on the **held-out test sequences**
  (`reports/tracker_study_cmc/`):

  | Test sequences (5) | MOTA | IDF1 | ID switches |
  |---|---|---|---|
  | Baseline | 0.515 | 0.536 | 627 |
  | Tuned, no CMC | 0.511 | 0.536 | 751 |
  | **Tuned + CMC** | **0.528** | **0.598** | **444** |

  The gain lands exactly where the camera moves: IDF1 on MOT17-10 0.392 -> 0.540, MOT17-13
  0.446 -> 0.577, MOT17-11 0.590 -> 0.620, while the static-camera sequences (04, 09) are
  **identical** with and without CMC. (On 09 both tuned rows sit below the baseline, 0.589 vs
  0.609: that's the tuned settings, not CMC.) On a synthetic shaking camera: MOTA 0.32 -> 0.83,
  IDF1 0.37 -> 0.74, ID switches cut 6x. Bug found on the way: high-contrast objects out-scored
  the background texture, so corners were found only ON moving objects and the estimate silently
  fell back to "no motion". Fixed by masking out the detector's boxes (never ground truth) and a
  lower corner threshold, and pinned by a regression test. On MOT17, CMC is kept only if it raises
  tune-set IDF1 (rule fixed in advance).

## Quickstart

```bash
pip install -e . -r requirements.txt
pytest -q
python -m tracelens.track_demo --scene busy --miss-rate 0.2 --jitter 2 --fp-rate 0.1
python -m tracelens.track_demo --scene crossing --save outputs/crossing.mp4
python -m tracelens.summarize_demo --scene patrol --target 1 --save outputs/summary.mp4
python -m tracelens.summarize_demo --scene patrol --target 1 --sweep
python -m tracelens.summarize_demo --scene lookalike --target 1 --relink parts --save outputs/relinked.mp4
python -m tracelens.summarize_demo --compare-relink --miss-rate 0.2
python -m tracelens.events_demo --save outputs/incident.mp4
python -m tracelens.events_demo --sweep
python -m tracelens.narrate_demo                       # template narrator, no LLM
pip install -e ".[llm]"
python -m tracelens.narrate_demo --narrator llm        # local Qwen2.5-1.5B-Instruct (~3 GB download)
python -m tracelens.narrate.checker_eval               # the checker's own accuracy
python -m tracelens.mot_eval --root data/raw/MOT17/train   # real footage (download: data/README.md)
python -m tracelens.relink_study --root data/raw/MOT17/train   # tune on 2 sequences, test on 5
python -m tracelens.tracker_study --root data/raw/MOT17/train  # tracker settings, same protocol
pip install ultralytics
python -m tracelens.detect_cache --root data/raw/MOT17/train --weights yolov8m.pt
python -m tracelens.tracker_study --root data/raw/MOT17/train --dets yolov8m.txt --out reports/tracker_study_yolo
```

## Demo

`demo/streamlit_app.py`: results on real footage (per-sequence, showing the camera-motion gain
confined to moving cameras), the scripted incident playing with its events, a real language
model's report checked sentence by sentence, and a live checker you can type into.

```bash
python -m tracelens.demo_assets        # rebuild demo/assets (GIF, event log, checked narratives)
streamlit run demo/streamlit_app.py
```
The hosted demo shows saved results and recorded outputs; free hosting can't run the detector or
the language model, so only the checker runs live. It has its own `demo/requirements.txt`
(streamlit, pandas, numpy). No MOT17 frames are shipped: real-footage results appear as tables
and charts; the video is our own synthetic incident.

## What the tests prove

- **Metrics are correct:** MOTA, IDF1, ID switches, misses and false positives match
  `motmetrics` exactly on deliberately messy tracker outputs.
- **Motion prediction matters:** when two identical objects cross paths, the Kalman tracker
  keeps both identities on every seed; the same tracker without motion prediction swaps them
  on some seeds.
- **The tracker is robust:** tracks survive a detector that misses 20% of detections, and
  one-frame false alarms never become tracks.

## Responsible use

Built and evaluated on synthetic and public research datasets only. No face recognition:
tracking uses position and appearance of the whole body/object. Surveillance technology can
be misused; this project is a research and analysis-time tool, not a system for identifying
individuals.
