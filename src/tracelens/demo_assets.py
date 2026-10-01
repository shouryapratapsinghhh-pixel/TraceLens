"""Build the static assets the demo app shows (no model, no MOT17 frames needed at view time).

  python -m tracelens.demo_assets

Writes demo/assets/:
  incident.gif        the scripted incident, tracked, with events appearing as they happen
  events.json         the event log (what the narrator is allowed to see)
  narratives.json     template + LLM narratives, each with its sentence-by-sentence check
The LLM narrative is the verbatim output of a real local run (Qwen2.5-1.5B-Instruct) -- the
demo host can't run the model, so it shows that recorded output and re-checks it live.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
from PIL import Image

from tracelens.events_demo import N_FRAMES, render, run_pipeline
from tracelens.narrate.check import check_narrative
from tracelens.narrate.log import build_log, log_text
from tracelens.narrate.narrators import TemplateNarrator
from tracelens.video.synthetic import (
    INCIDENT_FPS,
    INCIDENT_SIZE,
    incident_scene,
    make_synthetic_video,
)

# Verbatim from the first real run of `narrate_demo --narrator llm` (Qwen2.5-1.5B-Instruct, local).
RECORDED_LLM = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct (local, float16, greedy decoding)",
    "text": ("The target came into view at 1.2s [E1]. It loitered for 4.1s to 8.6s [E2], then moved fast for "
             "8.9s to 10.6s [E3] before entering the restricted zone at 10.6s [E4]. An object appeared beside "
             "the target and remained there until confirmed at 16.2s [E5]. The target exited the restricted "
             "zone at 15.0s [E6] and left the camera's view at 22.8s [E7]."),
}


def write_gif(frames, path: Path, fps: float, every: int = 2, width: int = 480) -> Path:
    """Every `every`-th frame, resized, as an animated GIF (plays in any browser)."""
    imgs = []
    for img in frames[::every]:
        h, w = img.shape[:2]
        small = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
        imgs.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).quantize(colors=64))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=round(1000 * every / fps), loop=0,
                 optimize=True)
    return path


def build(out_dir: str | Path = "demo/assets") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    objs, _ = incident_scene()
    v = make_synthetic_video(objs, n_frames=N_FRAMES, size=INCIDENT_SIZE, fps=INCIDENT_FPS)
    tracks, tid, events = run_pipeline(v, 0.0, 0.0, 0)
    write_gif(render(v.frames, tracks, tid, events, INCIDENT_FPS), out / "incident.gif", INCIDENT_FPS)

    log = build_log(events)
    (out / "events.json").write_text(json.dumps({"log": log, "text": log_text(log)}, indent=2))
    narratives = {}
    for name, text, source in (("template", TemplateNarrator().narrate(log), "fixed sentence per event (no LLM)"),
                               ("llm", RECORDED_LLM["text"], RECORDED_LLM["model"])):
        narratives[name] = {"source": source, "text": text, "check": check_narrative(text, log)}
    (out / "narratives.json").write_text(json.dumps(narratives, indent=2))
    return {"gif": out / "incident.gif", "events": len(log), "narratives": list(narratives)}


if __name__ == "__main__":
    info = build()
    size_kb = info["gif"].stat().st_size / 1024
    print(f"wrote {info['gif']} ({size_kb:.0f} KB), {info['events']} events, narratives: {info['narratives']}")
