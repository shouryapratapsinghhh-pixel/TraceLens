"""Narrate the scripted incident and check every sentence against the event log.

  python -m tracelens.narrate_demo                      # template narrator (no LLM)
  python -m tracelens.narrate_demo --narrator llm       # Qwen2.5-1.5B-Instruct, local (~3 GB download)
  python -m tracelens.narrate_demo --narrator llm --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse

from tracelens.events_demo import N_FRAMES, run_pipeline
from tracelens.narrate.check import check_narrative, verified_text
from tracelens.narrate.log import build_log, log_text
from tracelens.narrate.narrators import LLMNarrator, TemplateNarrator
from tracelens.video.synthetic import (
    INCIDENT_FPS,
    INCIDENT_SIZE,
    incident_scene,
    make_synthetic_video,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grounded narration with a faithfulness check.")
    parser.add_argument("--narrator", choices=["template", "llm"], default="template")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--miss-rate", type=float, default=0.0)
    parser.add_argument("--jitter", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    objs, _ = incident_scene()
    v = make_synthetic_video(objs, n_frames=N_FRAMES, size=INCIDENT_SIZE, fps=INCIDENT_FPS)
    _, _, events = run_pipeline(v, args.miss_rate, args.jitter, args.seed)
    log = build_log(events)
    print("EVENT LOG (all the narrator sees)\n" + log_text(log))

    narrator = LLMNarrator(args.model) if args.narrator == "llm" else TemplateNarrator()
    text = narrator.narrate(log)
    print(f"\nNARRATIVE ({args.narrator})\n{text}")

    report = check_narrative(text, log)
    print(f"\nCHECK: {report['supported_rate']:.0%} of {report['n_sentences']} sentences supported, "
          f"{report['coverage']:.0%} of events covered")
    for r in report["sentences"]:
        if r["issues"]:
            print(f"  FLAGGED {r['issues']}: {r['sentence']}")
    if report["supported_rate"] < 1.0:
        print(f"\nVERIFIED VERSION (flagged sentences removed)\n{verified_text(report)}")


if __name__ == "__main__":
    main()
