import pytest

from tracelens.narrate.check import actions_in, check_narrative, split_sentences, verified_text
from tracelens.narrate.checker_eval import INCIDENT_EVENTS, evaluate_checker
from tracelens.narrate.log import build_log, log_text
from tracelens.narrate.narrators import FakeNarrator, TemplateNarrator, build_messages

LOG = build_log(INCIDENT_EVENTS)


def test_log_ids_and_text():
    assert [e["id"] for e in LOG] == [f"E{i}" for i in range(1, 8)]
    txt = log_text(LOG)
    assert "E2 | 4.1s-8.6s | target stood still (loitered)" in txt
    assert "confirmed at 16.2s" in txt


def test_split_sentences_keeps_citations():
    assert split_sentences("It ran [E3]. It stopped [E4].  ") == ["It ran [E3].", "It stopped [E4]."]


def test_actions_do_not_double_count_leaving_the_zone():
    assert actions_in("At 15s it left the restricted zone") == {"zone_exit"}
    assert actions_in("it left the camera's view") == {"disappear"}
    assert actions_in("it left a bag behind") == {"object_left_behind"}


def test_template_narrator_is_fully_faithful_and_complete():
    report = check_narrative(TemplateNarrator().narrate(LOG), LOG)
    assert report["supported_rate"] == 1.0 and report["coverage"] == 1.0


def test_checker_catches_a_hallucinating_narrator():
    liar = FakeNarrator(
        "The target came into view at 1.2s [E1]. It ran from 3.0s to 5.0s [E3]. "
        "It nervously paced near the door [E2]. It entered the restricted zone. "
        "At 14.6s it left an object behind [E5].")
    report = check_narrative(liar.narrate(LOG), LOG)
    flagged = {r["sentence"]: r["issues"] for r in report["sentences"] if r["issues"]}
    assert len(flagged) == 3
    assert "wrong_time" in flagged["It ran from 3.0s to 5.0s [E3]."]
    assert "speculation" in flagged["It nervously paced near the door [E2]."]
    assert "uncited" in flagged["It entered the restricted zone."]
    assert report["coverage"] == pytest.approx(4 / 7)  # omitted events lower coverage
    assert "nervously" not in verified_text(report)  # the verified version drops flagged sentences


def test_checker_accuracy_on_labelled_set():
    m = evaluate_checker()
    assert m["precision"] == 1.0  # never flags a faithful sentence (incl. paraphrases)
    assert m["recall"] >= 0.9
    assert m["missed"] == ["It climbed the fence at 10.6s [E4]."]  # the documented lexicon blind spot


def test_prompt_contains_rules_example_and_only_the_log():
    msgs = build_messages(LOG)
    rules = msgs[0]["content"]
    assert "[E2]" in rules  # the citation format is spelled out
    assert "Never guess intent" in rules  # the no-speculation rule is in the prompt
    assert "E7 |" in msgs[-1]["content"] and "Write the summary." in msgs[-1]["content"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]


def test_real_llm_narrative_passes():
    """Verbatim output of the first real Qwen2.5-1.5B-Instruct run -- faithful; one sentence
    was falsely flagged before the checker fix."""
    text = ("The target came into view at 1.2s [E1]. It loitered for 4.1s to 8.6s [E2], then moved fast "
            "for 8.9s to 10.6s [E3] before entering the restricted zone at 10.6s [E4]. An object appeared "
            "beside the target and remained there until confirmed at 16.2s [E5]. The target exited the "
            "restricted zone at 15.0s [E6] and left the camera's view at 22.8s [E7].")
    report = check_narrative(text, LOG)
    assert report["supported_rate"] == 1.0 and report["coverage"] == 1.0
