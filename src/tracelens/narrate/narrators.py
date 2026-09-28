"""Narrators: turn an event log into a short incident report.

TemplateNarrator  fixed sentence per event type -- faithful by construction;
                  the baseline, and the fallback when an LLM's output fails checks
LLMNarrator       a local instruct model (default Qwen2.5-1.5B-Instruct) via
                  transformers. Sees ONLY the event log and must cite [En]
                  in every sentence. Needs the model download; no test needs it.
FakeNarrator      returns fixed text -- for tests (including deliberately
                  hallucinating text, to prove the checker catches it)
"""

from __future__ import annotations

from tracelens.narrate.log import log_text

TEMPLATES = {
    "appear": "The target came into view at {t:.1f}s [{id}].",
    "disappear": "The target left the camera's view at {t:.1f}s [{id}].",
    "dwell": "It loitered from {t:.1f}s to {t_end:.1f}s [{id}].",
    "run": "It ran from {t:.1f}s to {t_end:.1f}s [{id}].",
    "zone_enter": "At {t:.1f}s it entered the {zone} zone [{id}].",
    "zone_exit": "At {t:.1f}s it left the {zone} zone [{id}].",
    "object_left_behind": "At {t:.1f}s it left an object behind, confirmed at {t_confirmed:.1f}s [{id}].",
}

RULES = """You write short, factual incident summaries from a surveillance event log.
Rules:
- Use ONLY the events in the log. Do not add anything the log does not say.
- End EVERY sentence with the ID of the event it describes, like [E2].
- Give times in seconds exactly as written in the log.
- Describe behaviour only. Never guess intent, emotion, identity or guilt.
- Write one sentence per event, in time order. Plain text, no lists."""

EXAMPLE_LOG = """E1 | 2.0s | target came into view
E2 | 3.5s-6.0s | target moved fast (ran)
E3 | 9.0s | target left the camera's view"""

EXAMPLE_OUT = ("The target came into view at 2.0s [E1]. It ran from 3.5s to 6.0s [E2]. "
               "It left the camera's view at 9.0s [E3].")


class TemplateNarrator:
    def narrate(self, log: list[dict]) -> str:
        return " ".join(TEMPLATES[e["type"]].format(**{"t_end": 0, "t_confirmed": 0, "zone": "", **e})
                        for e in log)


class FakeNarrator:
    def __init__(self, text: str):
        self.text = text

    def narrate(self, log: list[dict]) -> str:
        return self.text


def build_messages(log: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": RULES},
        {"role": "user", "content": f"Event log:\n{EXAMPLE_LOG}\n\nWrite the summary."},
        {"role": "assistant", "content": EXAMPLE_OUT},
        {"role": "user", "content": f"Event log:\n{log_text(log)}\n\nWrite the summary."},
    ]


class LLMNarrator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens: int = 300):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device != "cpu" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
        self.device = device
        self.max_new_tokens = max_new_tokens

    def narrate(self, log: list[dict]) -> str:
        import torch

        prompt = self.tok.apply_chat_template(build_messages(log), tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
