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

Known limitation, measured: when the target leaves view and returns, the tracker gives it a
new ID, and a one-ID summary keeps only one visit (recall 0.50 on the patrol scene, vs 1.00
if the IDs were re-linked). Re-linking is the next phase.

**Next:** appearance-based re-linking of a returning target; event detection; LLM narration
grounded in an event log with a hallucination check; text queries; real-data evaluation
(MOT17); API + demo.

## Quickstart

```bash
pip install -e . -r requirements.txt
pytest -q
python -m tracelens.track_demo --scene busy --miss-rate 0.2 --jitter 2 --fp-rate 0.1
python -m tracelens.track_demo --scene crossing --save outputs/crossing.mp4
python -m tracelens.summarize_demo --scene patrol --target 1 --save outputs/summary.mp4
python -m tracelens.summarize_demo --scene patrol --target 1 --sweep
```

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
