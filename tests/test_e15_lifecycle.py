"""E15's combined lifecycle, end to end against LocalApi.

One workload (optimizer, storage views, RNG streams, workspace files, a user
counter) through the repaired research controller: cpu -> "gpu" -> cpu, with a
concurrent writer on the controller's route, an aborted hop, and a controller
killed and resumed. LocalApi has no CUDA, so the CUDA rows are skipped here and
exercised only by the live run; everything else is checked.

The default run must be OK with the controller unaided. Each sabotage puts one
of the three correctness blockers back (the mutating validation, the ungated
route, the old storage repair) and must turn the verdict NOT OK: the checks
can fail. A local run takes a few seconds, so all four always run.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "experiments" / "e15_combined.py"


def _run(tmp_path: Path, *extra: str) -> tuple[int, str, list[dict]]:
    out = tmp_path / "e15"
    p = subprocess.run([sys.executable, str(HARNESS), "--local", "--outdir", str(out), *extra],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=1200)
    recs = []
    f = out / "e15_records.jsonl"
    if f.exists():
        recs = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return p.returncode, p.stdout + p.stderr, recs


def test_the_lifecycle_passes_unaided(tmp_path):
    code, out, recs = _run(tmp_path)
    verdict = next((line for line in out.splitlines() if re.match(r"^E15 \d+ PASS", line)), "")
    assert code == 0, out[-3000:]
    assert verdict.endswith("OK") and "NOT OK" not in verdict, verdict
    assert " 0 FAIL" in verdict, verdict
    # The unaided row: the harness re-bound nothing on a restored runtime.
    assert "PASS  the controller ran unaided" in out
    # The three hops ran as designed: commit, abort, crash and resume.
    by = {r.get("name"): r for r in recs}
    assert by["hop1"]["result"]["phase"] == "DONE"
    assert by["hop2a"]["result"]["phase"] == "ABORTED"
    assert by["hop2b"]["result"]["phase"] == "DONE"


@pytest.mark.parametrize("sabotage,expected_fail", [
    # Blocker 3: validation that steps the real optimizer. The controller must
    # refuse to commit (fail closed on the side effect), so the lifecycle stops.
    ("inplace_validation", "controller reached DONE"),
    # Blocker 2: a route with no admission gate. Writes acknowledged after the
    # capture cut are lost at the destination.
    ("ungated_route", "every acknowledged write present exactly once"),
    # Blocker 1: the old storage repair (replace, not re-point). A Parameter
    # view comes back off its base.
    ("unshare_views", "every view shares its base at its offset"),
])
def test_each_blocker_put_back_turns_the_verdict_not_ok(tmp_path, sabotage, expected_fail):
    code, out, _ = _run(tmp_path, "--sabotage", sabotage)
    assert code != 0, out[-3000:]
    assert "NOT OK" in out
    assert any(line.strip().startswith("FAIL") and expected_fail in line for line in out.splitlines()), out[-3000:]
