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

**Next:** target selection → summary clip + timeline and "target recall vs compression";
event detection; LLM narration grounded in an event log with a hallucination check;
text queries; real-data evaluation (MOT17); API + demo.

## Quickstart

```bash
pip install -e . -r requirements.txt
pytest -q
python -m tracelens.track_demo --scene busy --miss-rate 0.2 --jitter 2 --fp-rate 0.1
python -m tracelens.track_demo --scene crossing --save outputs/crossing.mp4
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
