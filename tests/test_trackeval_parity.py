"""Our official-protocol scoring must equal the official MOTChallenge toolkit (TrackEval).

Skipped unless TrackEval is installed:
  pip install git+https://github.com/JonathonLuiten/TrackEval.git
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("trackeval")

SCRIPT = Path(__file__).parent / "trackeval_parity_script.py"


def test_matches_trackeval_exactly(tmp_path):
    out = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path)], capture_output=True, text=True, check=True)
    rows = {line.split("  ")[0]: line.split()[-5:] for line in out.stdout.splitlines()
            if line.startswith(("TrackEval", "ours, official"))}
    assert rows["TrackEval (official)"] == rows["ours, official_filter"], out.stdout
