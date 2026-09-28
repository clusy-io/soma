"""Record protection and run provenance: what keeps a rerun off the paper's records.

The files under results/ are the paper's records, and the analysers select
their rows by cohort name. A rerun that appended rows under a cohort already
recorded there would silently replace the paper's rows with its own, even
when it failed. `runmeta.shipped_record_conflict` is the rule that forbids
it, and every live harness asks it before touching an output file; every
harness's `--local` dry run also refuses to write anywhere under results/.

These tests hold the rule on the records the artifact actually ships (read
only), on record trees built in a temporary directory (with `runmeta.RESULTS`
pointed there), and through each harness's own `main()`, with `start_run`
replaced by a sentinel so that a run the guards let through stops before any
request, kernel or record. Nothing here writes under the real results/.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import re
import socket
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import runmeta  # noqa: E402

#: experiment -> (harness module, output-directory flag, the record file its live run appends to)
HARNESSES = {
    "e11": ("e11_transactional", "--outdir", "e11_runs.jsonl"),
    "e12": ("e12_crash", "--outdir", "e12_runs.jsonl"),
    "e13": ("e13_chain", "--out", "e13_chains.jsonl"),
    "e14": ("e14_lifecycle", "--out", "e14_arms.jsonl"),
    "e15": ("e15_combined", "--outdir", "e15_records.jsonl"),
    "e16": ("e16_cyclic", "--out", None),  # one file per cohort: <cohort>.json
}


class _Reached(Exception):
    """Raised in place of `runmeta.start_run`: the guards let the run through."""


def _harness(exp: str):
    return __import__(HARNESSES[exp][0])


def _record_name(exp: str, cohort: str) -> str:
    return HARNESSES[exp][2] or f"{cohort}.json"


def _rows(path: Path) -> list:
    if path.suffix == ".json":
        doc = json.loads(path.read_text())
        return doc if isinstance(doc, list) else [doc]
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _jsonl(path: Path, rows: list[dict]) -> Path:
    """Write rows as the harnesses do: one JSON document for a .json record
    (E16's per-cohort file), one line per row otherwise."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(rows[0] if len(rows) == 1 else rows, indent=1) + "\n")
    else:
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _main(monkeypatch, capsys, exp: str, *argv: str) -> tuple[int, str]:
    monkeypatch.setattr(sys, "argv", [f"{exp}.py", *argv])
    rc = _harness(exp).main()
    return rc, capsys.readouterr().err


@pytest.fixture
def results_tree(tmp_path, monkeypatch):
    """A results/ directory in tmp_path that runmeta treats as the shipped one."""
    root = tmp_path / "results"
    root.mkdir()
    monkeypatch.setattr(runmeta, "RESULTS", root)
    return root


@pytest.fixture
def stop_at_start_run(monkeypatch):
    """Replace start_run with a sentinel and record the cohort it was asked for."""
    reached: list[str | None] = []

    def fake(experiment, **kw):
        reached.append(kw.get("cohort"))
        raise _Reached(experiment)

    monkeypatch.setattr(runmeta, "start_run", fake)
    return reached


def test_every_shipped_record_refuses_the_cohorts_it_holds():
    results = ROOT / "results"
    records = sorted(p for p in results.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl"))
    assert len(records) >= 10, records
    labelled = unlabelled = meta_only = 0
    for path in records:
        shown = path.relative_to(ROOT).as_posix()
        cohorts = {r.get("cohort") for r in _rows(path) if isinstance(r, dict)}
        if cohorts == {None}:
            # No cohort labels (the escalation files): the file is refused outright.
            unlabelled += 1
            for c in (None, "a-fresh-cohort"):
                why = runmeta.shipped_record_conflict(path, c)
                assert why is not None and why.startswith(f"{shown} is a shipped record"), (path, c, why)
            continue
        labelled += 1
        for c in sorted(c for c in cohorts if c is not None):
            why = runmeta.shipped_record_conflict(path, c)
            assert why is not None and f"cohort {c!r} is already recorded in results/" in why, (path, c, why)
        # A fresh cohort (or none: the run then labels itself with its run id)
        # cannot collide, since the analysers select rows by cohort.
        assert runmeta.shipped_record_conflict(path, "a-fresh-cohort") is None, path
        assert runmeta.shipped_record_conflict(path, None) is None, path
        if path.name == "runs_meta.jsonl":
            # A cohort that reached only the metadata (a run that died before
            # writing rows) still owns its name for every new record beside it.
            for c in sorted(c for c in cohorts if c is not None):
                why = runmeta.shipped_record_conflict(path.parent / "rerun-probe.jsonl", c)
                assert why == (f"cohort {c!r} is already recorded in {shown}; "
                               "pass a new --cohort and a scratch output path"), (path, c, why)
                meta_only += 1
            assert not (path.parent / "rerun-probe.jsonl").exists()
    assert labelled and unlabelled and meta_only


def test_a_cohort_in_a_record_or_its_runs_meta_is_refused_and_a_new_one_allowed(results_tree, tmp_path):
    rec = _jsonl(results_tree / "e99" / "e99_runs.jsonl",
                 [{"cohort": "r1", "i": 0}, {"cohort": "r1", "i": 1}, {"cohort": "r2", "i": 2}])
    _jsonl(results_tree / "e99" / "runs_meta.jsonl", [{"run_id": "e99-a", "cohort": "r1"},
                                                       {"run_id": "e99-b", "cohort": "r3"}])
    (results_tree / "e98").mkdir()
    listed = results_tree / "e98" / "rec.json"
    listed.write_text(json.dumps([{"cohort": "j1"}, {"cohort": "j2"}]))
    single = results_tree / "e98" / "one.json"
    single.write_text(json.dumps({"cohort": "d1", "rows": []}))
    hint = "pass a new --cohort and a scratch output path"

    assert runmeta.shipped_record_conflict(rec, "r1") == f"cohort 'r1' is already recorded in results/e99/e99_runs.jsonl; {hint}"
    assert runmeta.shipped_record_conflict(str(rec), "r2") is not None
    # Only in runs_meta.jsonl: refused for the record and for a record not yet written.
    for target in (rec, results_tree / "e99" / "not-yet.jsonl"):
        assert runmeta.shipped_record_conflict(target, "r3") == \
            f"cohort 'r3' is already recorded in results/e99/runs_meta.jsonl; {hint}"
    assert runmeta.shipped_record_conflict(listed, "j2") is not None
    assert runmeta.shipped_record_conflict(single, "d1") is not None
    for target in (rec, results_tree / "e99" / "not-yet.jsonl", listed, single):
        assert runmeta.shipped_record_conflict(target, "r4-new") is None, target
        assert runmeta.shipped_record_conflict(target, None) is None, target
    # The same record reached through `..` or a symlinked output directory is
    # still the shipped record.
    assert runmeta.shipped_record_conflict(str(results_tree / "e99" / ".." / "e99" / "e99_runs.jsonl"), "r1") is not None
    alias = tmp_path / "alias"
    alias.symlink_to(results_tree / "e99", target_is_directory=True)
    assert runmeta.shipped_record_conflict(alias / "e99_runs.jsonl", "r2") is not None
    assert runmeta.shipped_record_conflict(alias / "fresh.jsonl", "r3") is not None


def test_unlabelled_or_unreadable_records_are_refused_and_paths_outside_results_never_are(results_tree, tmp_path):
    unlabelled = _jsonl(results_tree / "esc" / "escalation_runs.jsonl", [{"trial": 1}, {"trial": 2}])
    broken = results_tree / "e97" / "e97_runs.jsonl"
    broken.parent.mkdir()
    broken.write_text('{"cohort": "x"}\n{"cohort": \n')

    for c in (None, "anything"):
        assert runmeta.shipped_record_conflict(unlabelled, c) == \
            "results/esc/escalation_runs.jsonl is a shipped record; pass a scratch output path"
        why = runmeta.shipped_record_conflict(broken, c)
        assert why == ("results/e97/e97_runs.jsonl is a shipped record and could not be read "
                       "(JSONDecodeError); pass a new --cohort and a scratch output path"), why

    # Identical files outside results/, including in a sibling whose name only
    # starts with "results", are scratch output and never refused.
    for scratch in (tmp_path / "scratch", tmp_path / "results-rerun"):
        _jsonl(scratch / "escalation_runs.jsonl", [{"trial": 1}])
        _jsonl(scratch / "e99_runs.jsonl", [{"cohort": "r1"}])
        _jsonl(scratch / "runs_meta.jsonl", [{"cohort": "r2"}])
        (scratch / "broken.jsonl").write_text(broken.read_text())
        for name, c in (("escalation_runs.jsonl", None), ("e99_runs.jsonl", "r1"),
                        ("e99_runs.jsonl", "r2"), ("broken.jsonl", "x")):
            assert runmeta.shipped_record_conflict(scratch / name, c) is None, (scratch, name, c)


def test_write_meta_appends_one_line_per_run_and_that_line_then_claims_the_cohort(results_tree):
    record = results_tree / "e99" / "e99_runs.jsonl"
    meta_path = results_tree / "e99" / "runs_meta.jsonl"
    first = {"run_id": "e99-1", "cohort": "c-new", "out": Path("scratch") / "e99", "results": {"ok": True}}
    second = {"run_id": "e99-2", "cohort": "c-other"}

    assert runmeta.shipped_record_conflict(record, "c-new") is None
    runmeta.write_meta(first, meta_path)  # creates results/e99/
    assert meta_path.read_text().count("\n") == 1
    why = runmeta.shipped_record_conflict(record, "c-new")
    assert why is not None and "results/e99/runs_meta.jsonl" in why
    assert runmeta.shipped_record_conflict(record, "c-other") is None

    runmeta.write_meta(second, meta_path)  # appends; the first run's line stays
    lines = meta_path.read_text().splitlines()
    assert [json.loads(line)["run_id"] for line in lines] == ["e99-1", "e99-2"]
    assert json.loads(lines[0])["out"] == str(Path("scratch") / "e99")  # non-JSON values as strings
    assert json.loads(lines[0])["results"] == {"ok": True}
    assert runmeta.shipped_record_conflict(record, "c-other") is not None
    assert not record.exists()


def test_start_stamp_and_finish_record_provenance_and_flag_a_server_restart(tmp_path, monkeypatch):
    # Without an API the identity is explicitly unknown, and a record says so.
    assert runmeta.server_identity(None) == {"source": "unknown", "reason": "no api url"}
    rec = runmeta.stamp({"x": 1}, {"run_id": "r", "cohort": "c", "server": runmeta.server_identity(None)})
    assert rec["x"] == 1 and rec["run"]["server_commit_label"] == "unknown"
    assert rec["run"]["server_commit_label_note"] == "no api url"

    reported = {"source": "server_reported", "url": "http://api.test", "git_commit": "0123456789abcdef0123",
                "git_dirty": False, "dist_sha256": "d" * 64, "build_info_matches_dist": True,
                "dist_modified_after_start": False, "started_at": "2026-09-27T10:00:00Z", "pid": 41}
    restarted = {**reported, "pid": 42, "started_at": "2026-09-27T11:00:00Z"}
    answers = [reported, dict(reported), reported, restarted]
    asked: list[str | None] = []

    def fake_identity(api_url, timeout=10.0):
        asked.append(api_url)
        return dict(answers.pop(0))

    monkeypatch.setattr(runmeta, "server_identity", fake_identity)
    extra = tmp_path / "probe_harness.py"
    extra.write_text("print('probe')\n")
    missing = tmp_path / "gone.py"

    meta = runmeta.start_run("e99", api_url="http://api.test", args={"steps": 3}, cohort="e99-c",
                             extra_files=[extra, missing])
    assert meta["schema"] == "runmeta/1" and meta["experiment"] == "e99" and meta["cohort"] == "e99-c"
    assert meta["args"] == {"steps": 3} and meta["argv"] == sys.argv and meta["server"] == reported
    started = dt.datetime.fromisoformat(meta["started_at_utc"])
    assert started.utcoffset() == dt.timedelta(0)
    assert re.fullmatch(rf"e99-{started:%Y%m%dT%H%M%S}-[0-9a-f]{{6}}", meta["run_id"]), meta["run_id"]
    files = meta["harness"]["files_sha256"]
    assert files["src/runmeta.py"] == hashlib.sha256((ROOT / "src" / "runmeta.py").read_bytes()).hexdigest()
    assert files["experiments/oracles.py"] == hashlib.sha256((ROOT / "experiments" / "oracles.py").read_bytes()).hexdigest()
    assert [v for k, v in files.items() if k.endswith("probe_harness.py")] == \
        [hashlib.sha256(extra.read_bytes()).hexdigest()]
    assert [v for k, v in files.items() if k.endswith("gone.py")] == [None]
    host = meta["host"]
    assert host["host_id"] == hashlib.sha256(socket.gethostname().encode()).hexdigest()[:12]
    assert socket.gethostname() not in host.values() and host["python"] == platform.python_version()

    stamped = runmeta.stamp({"arm": "a"}, meta)["run"]
    assert stamped["run_id"] == meta["run_id"] and stamped["cohort"] == "e99-c"
    assert stamped["harness_commit"] == meta["harness"]["commit"]
    assert stamped["server_commit"] == "0123456789abcdef0123" and stamped["server_identity_source"] == "server_reported"
    assert (stamped["server_commit_label"], stamped["server_commit_label_note"]) == ("01234567", None)
    dirty = runmeta.stamp({}, {**meta, "server": {**reported, "git_dirty": True}})["run"]
    assert dirty["server_commit_label"] == "01234567-dirty"
    rebuilt = runmeta.stamp({}, {**meta, "server": {**reported, "dist_modified_after_start": True}})["run"]
    assert rebuilt["server_commit_label"] == "unknown" and "modified after the process started" in rebuilt["server_commit_label_note"]

    same = runmeta.finish_run(meta, api_url="http://api.test")
    assert same is meta and meta["server_changed_during_run"] is False
    assert meta["server_at_finish"] == reported
    assert dt.datetime.fromisoformat(meta["finished_at_utc"]) >= started

    other = runmeta.start_run("e99", api_url="http://api.test", cohort="e99-c")
    assert other["run_id"] != meta["run_id"]
    runmeta.finish_run(other, api_url="http://api.test")
    assert other["server_changed_during_run"] is True
    assert runmeta.stamp({}, other)["run"]["server_changed_during_run"] is True
    assert asked == ["http://api.test"] * 4 and not answers


def test_every_dry_run_refuses_to_write_under_results(tmp_path, monkeypatch, capsys, stop_at_start_run):
    results = tmp_path / "results"
    for exp in HARNESSES:
        _jsonl(results / exp / _record_name(exp, f"{exp}-paper"), [{"cohort": f"{exp}-paper"}])
        monkeypatch.setattr(_harness(exp), "ROOT", tmp_path)  # the harness's results/ is tmp_path/results
    system_tmp = tmp_path / "system-tmp"
    system_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(system_tmp))
    before = _snapshot(results)

    for exp, (_, flag, _) in HARNESSES.items():
        for out in (results / exp / "dry-run", results):
            rc, err = _main(monkeypatch, capsys, exp, "--local", flag, str(out), "--cohort", f"{exp}-dry")
            assert rc == 2, (exp, out, err)
            assert "--local" in err and "results" in err, (exp, err)
            assert not (results / exp / "dry-run").exists(), exp
    assert _snapshot(results) == before
    assert list(system_tmp.iterdir()) == []  # E13 removed the kernel directory it had made
    assert stop_at_start_run == []

    # The same dry runs pointed at a scratch directory pass the guard.
    for exp, (_, flag, _) in HARNESSES.items():
        with pytest.raises(_Reached):
            _main(monkeypatch, capsys, exp, "--local", flag, str(tmp_path / "scratch" / exp), "--cohort", f"{exp}-dry")
    assert stop_at_start_run == [f"{exp}-dry" for exp in HARNESSES]
    assert _snapshot(results) == before


def test_every_live_run_refuses_a_recorded_cohort_and_allows_a_new_one_in_a_temp_dir(
        results_tree, tmp_path, monkeypatch, capsys, stop_at_start_run):
    monkeypatch.setenv("CLUSY_HARNESS_API_KEY", "not-a-real-key")
    api = ("--api-url", "http://127.0.0.1:9")  # never contacted: every run stops before start_run
    for exp in HARNESSES:
        paper, died = f"{exp}-paper", f"{exp}-died"
        _jsonl(results_tree / exp / _record_name(exp, paper), [{"cohort": paper, "i": 0}, {"cohort": paper, "i": 1}])
        _jsonl(results_tree / exp / "runs_meta.jsonl", [{"run_id": "a", "cohort": paper}, {"run_id": "b", "cohort": died}])
    before = _snapshot(results_tree)

    for exp, (_, flag, _) in HARNESSES.items():
        out = results_tree / exp
        for cohort, where in ((f"{exp}-paper", _record_name(exp, f"{exp}-paper")), (f"{exp}-died", "runs_meta.jsonl")):
            rc, err = _main(monkeypatch, capsys, exp, *api, flag, str(out), "--cohort", cohort)
            assert rc == 2, (exp, cohort, err)
            assert f"cohort {cohort!r} is already recorded in results/{exp}/{where}" in err, (exp, cohort, err)
    assert _snapshot(results_tree) == before
    assert stop_at_start_run == []

    for exp, (_, flag, _) in HARNESSES.items():
        scratch = tmp_path / "scratch" / exp
        with pytest.raises(_Reached):
            _main(monkeypatch, capsys, exp, *api, flag, str(scratch), "--cohort", f"{exp}-rerun")
        assert list(scratch.iterdir()) == [], exp  # nothing written before the run starts
    assert stop_at_start_run == [f"{exp}-rerun" for exp in HARNESSES]

    # E16 writes one file per cohort and never replaces one, even outside results/.
    scratch = tmp_path / "scratch" / "e16"
    existing = scratch / "e16-rerun.json"
    existing.write_text('{"cohort": "e16-rerun"}\n')
    rc, err = _main(monkeypatch, capsys, "e16", *api, "--out", str(scratch), "--cohort", "e16-rerun")
    assert rc == 2 and "already exists" in err, err
    assert existing.read_text() == '{"cohort": "e16-rerun"}\n'
    assert _snapshot(results_tree) == before
