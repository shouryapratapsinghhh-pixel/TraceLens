import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.slow  # builds the GIF and runs the app

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def demo_root(tmp_path_factory):
    """A throwaway copy of the repo with fake study results, so the test needs no MOT17."""
    from tracelens.demo_assets import build

    root = tmp_path_factory.mktemp("repo")
    shutil.copytree(REPO / "src", root / "src")
    (root / "demo").mkdir()
    shutil.copy(REPO / "demo" / "streamlit_app.py", root / "demo" / "streamlit_app.py")
    build(root / "demo" / "assets")
    study = root / "reports" / "tracker_study_cmc"
    study.mkdir(parents=True)
    rows = []
    for method, bump in (("baseline (single, cutoff 0.5)", 0.0), ("tuned: x", 0.0), ("tuned + CMC (CHOSEN)", 0.06)):
        for seq, moving in (("MOT17-04-FRCNN", False), ("MOT17-10-FRCNN", True)):
            rows.append({"method": method, "sequence": seq, "mota": 0.5, "idf1": 0.5 + (bump if moving else 0),
                         "id_switches": 100 - (40 if bump and moving else 0), "misses": 1, "false_positives": 1})
    pd.DataFrame(rows).to_csv(study / "test_results.csv", index=False)
    (study / "decision.json").write_text(json.dumps({
        "tune_sequences": ["MOT17-02-FRCNN"], "rule": "keep CMC if tune IDF1 rises",
        "tune_idf1_without_cmc": 0.50, "tune_idf1_with_cmc": 0.51, "cmc_kept": True}))
    return root


def test_assets_are_built_and_checked(demo_root):
    nar = json.loads((demo_root / "demo" / "assets" / "narratives.json").read_text())
    assert nar["llm"]["check"]["supported_rate"] == 1.0  # the recorded real LLM output is fully faithful
    assert (demo_root / "demo" / "assets" / "incident.gif").stat().st_size > 10_000


def test_app_runs_and_checker_is_live(demo_root, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("TRACELENS_ROOT", str(demo_root))
    at = AppTest.from_file(str(demo_root / "demo" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    idf1 = next(m for m in at.metric if m.label.startswith("IDF1"))
    assert idf1.value == "0.530"  # mean of 0.50 (static) and 0.56 (moving) for the CMC row
    at.radio[0].set_value("Wrong time").run()
    assert not at.exception
    table = at.dataframe[-1].value
    assert (table[""] == "❌").any() and table["problem"].str.contains("time").any()
