"""Run a detector ONCE over MOT17 sequences and cache its boxes in MOTChallenge det format.

  pip install ultralytics
  python -m tracelens.detect_cache --root data/raw/MOT17/train --weights yolov8m.pt

Writes <sequence>/det/<name>.txt (e.g. det/yolov8m.txt) next to the public det.txt, with a
LOW confidence floor (0.05) so ByteTrack has the weak detections it needs. Every tracking
study then reads the file -- the detector never has to run twice.

These are "private" detections: results using them are not comparable with public-detection
leaderboards. The YOLO weights are off-the-shelf COCO models, not trained on MOT17.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from tracelens.video.mot import MOTSequence

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def cache_sequence(seq: MOTSequence, detector, out_name: str, log_every: int = 100) -> Path:
    """detector: anything with .detect(frame_idx, image) -> list[Detection]."""
    path = seq.dir / "det" / f"{out_name}.txt"
    lines, t0 = [], time.perf_counter()
    for f in range(seq.n_frames):
        for d in detector.detect(f, seq.frames[f]):
            x1, y1, x2, y2 = d.box
            lines.append(f"{f + 1},-1,{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{d.score:.4f},-1,-1,-1")
        if (f + 1) % log_every == 0:
            logger.info("%s: %d/%d frames (%.1f fps)", seq.name, f + 1, seq.n_frames, (f + 1) / (time.perf_counter() - t0))
    path.write_text("\n".join(lines) + "\n")
    return path


def cache_name(weights: str, imgsz: int) -> str:
    """yolov8m.pt at 640 -> 'yolov8m' (the original name); at 1280 -> 'yolov8m_1280'."""
    stem = Path(weights).stem
    return stem if imgsz == 640 else f"{stem}_{imgsz}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache detector outputs for MOT17 sequences.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--seqs", nargs="*", default=None, help="default: every *-FRCNN sequence")
    parser.add_argument("--weights", default="yolov8m.pt", help="any Ultralytics weights, e.g. yolov8m.pt, yolov8x.pt")
    parser.add_argument("--conf", type=float, default=0.05, help="keep boxes down to this confidence")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input size; 1280 keeps small, far people")
    args = parser.parse_args()

    from tracelens.detect.detectors import YoloDetector

    det = YoloDetector(args.weights, conf=args.conf, imgsz=args.imgsz)
    name = cache_name(args.weights, args.imgsz)
    root = Path(args.root)
    seqs = args.seqs or sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.endswith("FRCNN"))
    for s in seqs:
        path = cache_sequence(MOTSequence(root / s), det, name)
        logger.info("wrote %s", path)
    print(f"cached '{name}' detections for {len(seqs)} sequences -> use --dets {name}.txt in the studies")


if __name__ == "__main__":
    main()
