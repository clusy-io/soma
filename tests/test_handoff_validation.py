"""Validation must not change the committed workload, and what cannot be
restored must block the commit.

The reviewer's local end-to-end probe (controller-probes.json in
output/paper-review-2026-09-27): a clean migration reported 12/12 and DONE,
while every optimizer state counter had advanced 3 -> 4, the parameter bytes
had changed and the application's own step counter was still 3. The
validation had run a real optimizer step on the committed state. The
recovery check did the same before filtering its result to seven checks.

These tests pin the repair from three sides:

  * a clean migration with no user work leaves the optimizer counters,
    parameter and buffer bytes, the application's counter, every RNG stream
    and the plain values exactly as they were at the capture (the reviewer's
    probe, extended);
  * validation still FAILS a real defect (a corrupted parameter, an optimizer
    no longer bound to the model), so non-mutating did not mean blind;
  * a deliberately mutating validation still reports 12/12, and the
    commit-boundary check catches it and aborts.

And the restore side (B3): a view the adapter could not repair blocks the
commit (`views_not_restored`); a capsule whose storages are tagged `cuda:0`
restores on a CPU-only destination through the remap handler, with the count
recorded; and the commit boundary catches the reviewer's three storage
defects even when an adapter reports zero failures.
"""

from __future__ import annotations

import copy
import functools
import io
import json
import random
import sys
import tarfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from handoff import controller as ctl  # noqa: E402
from handoff.controller import MARK, Controller, Journal, _wrap, seed_program  # noqa: E402

localapi = pytest.importorskip("localapi")


def program(body: str) -> str:
    return _wrap('import sys, json, os\ng = sys.modules["__main__"].__dict__\n' + body)


#: An observation of the state that is independent of the controller's own
#: fingerprint, so it can judge the old controller as well as the new one.
SNAPSHOT = program(f'''
import hashlib, random, pickle
import numpy as np, torch
m, opt = g["model"], g["optimizer"]
h = hashlib.sha256()
for p in m.parameters(): h.update(p.detach().cpu().numpy().tobytes())
hb = hashlib.sha256()
for b in m.buffers(): hb.update(b.detach().cpu().numpy().tobytes())
hs = hashlib.sha256()
for st in opt.state.values():
    for k in sorted(st):
        v = st[k]
        hs.update(k.encode()); hs.update(v.detach().cpu().numpy().tobytes() if torch.is_tensor(v) else repr(v).encode())
print({MARK!r} + json.dumps({{
    "params_sha256": h.hexdigest(), "buffers_sha256": hb.hexdigest(),
    "optimizer_steps": [float(v["step"]) for v in opt.state.values() if "step" in v],
    "optimizer_state_sha256": hs.hexdigest(),
    "global_step": g["step_count"], "user_counter": g.get("audit_counter"),
    "scheduler": [g["scheduler"].last_epoch, g["optimizer"].param_groups[0]["lr"]],
    "training": m.training,
    "rng": {{"python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
            "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
            "torch": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            "loader": hashlib.sha256(g["loader"].generator.get_state().numpy().tobytes()).hexdigest(),
            "np_gen": hashlib.sha256(repr(g["np_gen"].bit_generator.state).encode()).hexdigest()}},
    "plain": {{k: repr(g[k]) for k in ("sc_int", "sc_float", "sc_str", "step_count", "audit_counter")}}
             | {{"co_dict": repr(sorted(g["co_dict"].items(), key=repr))}}}}))
''')


@pytest.fixture
def api(tmp_path):
    a = localapi.LocalApi(workdir=tmp_path / "kernels")
    yield a
    a.close()


def _source(api, name="src", extra: str | None = None) -> str:
    pid = api.create_project(f"clusy-exp-handoff-src-{name}", "cpu")
    api.witness(pid, seed_program(seed=7, files=2, corpus_bytes=2048))
    api.witness(pid, program(f'g["audit_counter"] = 0\nprint({MARK!r} + "{{}}")'))
    if extra:
        api.witness(pid, program(extra + f'\nprint({MARK!r} + "{{}}")'))
    return pid


def _is_restore(code: str) -> bool:
    return "load_session" in code and "_remap_install" in code


# ---------------------------------------------------------------------------
# B2: the reviewer's probe
# ---------------------------------------------------------------------------

def _declared_forward_only(res):
    """The validation record of a handoff: every check that ran passed, and
    exactly one row is DECLARED not run, the forward output, with the
    controller's reason (no source reference: the source runs no model code).
    The totals say so; nothing reports 12/12."""
    v = res["validation"]
    assert res["timings"]["oracles"] == f"{v['passed']}/{v['total']}" == "11/11", v
    assert [d["name"] for d in v["declared"]] == ["forward output"], v["declared"]
    assert ctl.FORWARD_NOT_RUN in v["declared"][0]["detail"]
    assert (v["run"], v["declared_not_run"]) == (11, 1)
    assert v["summary"] == "11 checks run, 11 passed, 1 declared not run (forward output)"
    assert v["side_effects"]["checked"] and v["side_effects"]["changed"] == [], v["side_effects"]


def test_clean_migration_commits_exactly_the_captured_state(api, tmp_path):
    """Old code: optimizer steps 3.0 -> 4.0, parameter bytes changed, step
    count 3, and the RNG streams are the destination process's own."""
    src = _source(api)
    before = api.witness(src, SNAPSHOT)
    res = Controller(api, Journal(tmp_path / "j.sqlite"), tmp_path / "blobs").migrate("clean", src, "cpu")
    assert res["phase"] == "DONE", res
    after = api.witness(res["dest"], SNAPSHOT)
    assert before["optimizer_steps"] == after["optimizer_steps"] == [3.0] * 6
    assert before["global_step"] == after["global_step"] == 3
    assert before == after
    _declared_forward_only(res)


def test_the_commit_boundary_verdict_is_journaled(api, tmp_path):
    src = _source(api)
    j = Journal(tmp_path / "j.sqlite")
    res = Controller(api, j, tmp_path / "blobs").migrate("jb", src, "cpu")
    assert res["phase"] == "DONE", res
    row = j.get("jb")
    src_fp, commit_fp = json.loads(row["fp_source"]), json.loads(row["fp_commit"])
    verdicts = json.loads(row["boundary"])
    assert verdicts["restored"]["equal"] and verdicts["pre_commit"]["equal"]
    assert {f: src_fp[f] for f in ctl.BOUNDARY_FIELDS} == {f: commit_fp[f] for f in ctl.BOUNDARY_FIELDS}
    # The fingerprint looked at what it claims to: the optimizer is bound to
    # the model's parameters and its state carries six step counters.
    opt = src_fp["optimizers"]["optimizer"]
    assert opt["state_keys_bound"] and all(b.startswith("model.param.") for b in opt["binding"])
    assert len(opt["state"]) == 6
    assert res["commit_boundary"]["verdicts"]["pre_commit"]["declared"]["rng_cuda"] == "absent"


def test_recovery_check_is_non_mutating_and_runs_every_check(api, tmp_path):
    """Old code: 7 checks, and the full mutating oracle ran before the filter,
    so each recovery check advanced the committed optimizer once more. Now
    every check that has a reference runs (eleven), the forward output is
    declared not run, and the check MEASURES that it changed nothing."""
    src = _source(api)
    res = Controller(api, Journal(tmp_path / "j.sqlite"), tmp_path / "blobs").migrate("rc", src, "cpu")
    assert res["phase"] == "DONE", res
    s0 = api.witness(res["dest"], SNAPSHOT)
    r1 = api.witness(res["dest"], ctl.recovery_check_program())
    r2 = api.witness(res["dest"], ctl.recovery_check_program())
    s1 = api.witness(res["dest"], SNAPSHOT)
    assert (r1["passed"], r1["total"]) == (r2["passed"], r2["total"]) == (11, 11), (r1["failed"], r2["failed"])
    assert [d["name"] for d in r1["declared"]] == ["forward output"]
    assert r1["side_effects"]["checked"] and r1["side_effects"]["changed"] == []
    assert s0 == s1


#: User work between the seed and the switch: real training steps that keep
#: the application's own counter in step with the optimizer, and a scheduler
#: step each time.
TRAIN = program(f'''
m, opt, sched = g["model"], g["optimizer"], g["scheduler"]
for _ in range(2):
    opt.zero_grad()
    g["loss_fn"](m(g["train_x"]), g["train_y"]).backward()
    opt.step(); sched.step()
    g["step_count"] += 1
print({MARK!r} + json.dumps({{"step_count": g["step_count"]}}))
''')


#: Every expectation field the source can state without running model code;
#: the forward reference is the one it cannot.
STRUCTURAL = ("values", "step_count", "next_loader_indices", "scheduler_lr", "scheduler_epoch", "param_digest",
              "device_type")

EXP_PROG = program(f'print({MARK!r} + json.dumps(g["__handoff_expectation__"], default=str))')


def _capsule_manifest(cap):
    import base64
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(cap["b64"])), mode="r:gz") as t:
        return json.load(t.extractfile("manifests.json"))


def test_validation_is_against_the_cut_not_the_seed(api, tmp_path):
    """The expectation the oracles check is refreshed at the capture cut, from
    structure only. With the seed-time expectation (the first version) a
    correct restore after user training fails `parameters`, `continuation`
    and `scheduler`; with no user work every structural field is the seed's,
    and the forward reference is replaced by the declared reason."""
    src = _source(api)
    seed_exp = api.witness(src, EXP_PROG)
    man = _capsule_manifest(api.witness(src, ctl.capture_program()))
    assert man["expectation_source"] == "cut"
    assert man["expectation_method"]["model_code_run"] is False
    exp = json.loads(json.dumps(man["expectation"]))
    assert {k: exp[k] for k in STRUCTURAL} == {k: json.loads(json.dumps(seed_exp))[k] for k in STRUCTURAL}
    assert seed_exp["forward_output"] is not None and exp["forward_output"] is None
    assert exp["forward_output_not_run"] == ctl.FORWARD_NOT_RUN and exp["not_refreshed"] == {}
    assert exp["cut_fingerprint_sha256"] == __import__("hashlib").sha256(
        json.dumps(man["fingerprint"], sort_keys=True, default=str).encode()).hexdigest()
    assert api.witness(src, EXP_PROG) == seed_exp                      # the source keeps its own
    assert api.witness(src, TRAIN)["step_count"] == 5
    before = api.witness(src, SNAPSHOT)
    res = Controller(api, Journal(tmp_path / "j.sqlite"), tmp_path / "blobs").migrate("trained", src, "cpu")
    assert res["phase"] == "DONE", res
    _declared_forward_only(res)
    assert res["capture"]["expectation_source"] == "cut"
    assert api.witness(res["dest"], SNAPSHOT) == before
    assert before["optimizer_steps"] == [5.0] * 6


def test_a_runtime_the_controller_restored_and_the_user_trained_migrates_again(api, tmp_path):
    """E15 hop 2b, reduced (WS-C finding 1). The first version refreshed the
    expectation with the fixture's `compute_expectation`, which reaches
    `_digest_params` and `hashlib` through `__main__`; no capsule carries
    either, so on a runtime the controller had restored the refresh raised
    NameError, the stale expectation was used, and after user training the
    next hop ABORTED `contract_check_failed` (parameters, continuation,
    scheduler, data cursor). The refresh now needs nothing from `__main__`."""
    src = _source(api)
    j = Journal(tmp_path / "j.sqlite")
    r1 = Controller(api, j, tmp_path / "blobs").migrate("again1", src, "cpu")
    assert r1["phase"] == "DONE", r1
    # `compute_expectation` itself is a public function defined in the
    # source's `__main__`, so dill carries it by value; the names its body
    # looks up at call time (an underscore helper and a module) are filtered
    # out of every capsule. That combination is what broke the first refresh.
    present = api.witness(r1["dest"], program(
        f'print({MARK!r} + json.dumps([k for k in ("_digest_params", "compute_expectation", "hashlib") if k in g]))'))
    assert present == ["compute_expectation"]              # carried, but its helpers are not
    assert api.witness(r1["dest"], TRAIN)["step_count"] == 5
    before = api.witness(r1["dest"], SNAPSHOT)
    r2 = Controller(api, j, tmp_path / "blobs").migrate("again2", r1["dest"], "cpu")
    assert r2["phase"] == "DONE", r2
    assert r2["capture"]["expectation_source"] == "cut"
    _declared_forward_only(r2)
    exp = api.witness(r2["dest"], EXP_PROG)
    assert exp["step_count"] == 5 and exp["not_refreshed"] == {} and exp["param_digest"]
    assert api.witness(r2["dest"], SNAPSHOT) == before


def _expect_env():
    """The capture's expectation source exec'd the way a kernel program does
    it, next to the fingerprint it depends on."""
    env = _fp_env()
    env["_FORWARD_NOT_RUN"] = ctl.FORWARD_NOT_RUN
    exec(compile(ctl._EXPECT_SRC, "<expect>", "exec"), env)
    return env


def test_the_cut_expectation_is_the_oracles_reference_without_running_the_model(tmp_path, monkeypatch):
    """In process: the capture's own digest is byte for byte the fixture's
    `_digest_params` (which the destination's `parameters` row computes),
    every structural field equals the fixture's `compute_expectation`, and
    computing them runs no forward (a pre-hook that raises is never reached,
    the buffer the fixture's forward increments does not move) and moves no
    generator or RNG stream."""
    import torch
    from experiments.fixture import _digest_params, compute_expectation
    ns, _ = _fixture_ns(tmp_path, monkeypatch)
    env = _expect_env()
    assert env["_cut_param_digest"](ns["model"]) == _digest_params(ns["model"])
    other = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    assert env["_cut_param_digest"](other) == _digest_params(other)
    ref = compute_expectation(dict(ns)).to_dict()           # runs the forward (moves the buffer)
    ns["__handoff_expectation__"] = ref

    def refuse(mod, inp):
        raise AssertionError("the cut expectation ran the model")
    handle = ns["model"].register_forward_pre_hook(refuse)
    calls = ns["model"].forward_calls.clone()
    loader_state = ns["loader"].generator.get_state().clone()
    rng = torch.get_rng_state().clone()
    exp, source, method = env["_cut_expectation"](ns, {"x": 1})
    handle.remove()
    assert source == "cut" and method["model_code_run"] is False and exp["not_refreshed"] == {}
    ref["param_digest"] = _digest_params(ns["model"])          # the digest after the fixture's forward
    assert {k: exp[k] for k in STRUCTURAL} == {k: ref[k] for k in STRUCTURAL}
    assert exp["forward_output"] is None and exp["forward_output_not_run"] == ctl.FORWARD_NOT_RUN
    assert torch.equal(ns["model"].forward_calls, calls)
    assert torch.equal(ns["loader"].generator.get_state(), loader_state) and torch.equal(torch.get_rng_state(), rng)
    # A value whose repr would run user code is not read: named, not stale.
    class Loud(str):
        def __repr__(self):
            raise AssertionError("user __repr__ ran")
    exp2, source2, _ = env["_cut_expectation"](dict(ns, sc_int=Loud("x")), {})
    assert "values.sc_int" in exp2["not_refreshed"] and "sc_int" not in exp2["values"]
    assert source2.startswith("cut (incomplete:")


def test_the_capture_runs_no_model_code_on_the_source(api, tmp_path):
    """The first version refreshed the expectation by running the fixture's
    forward in a forked child of the SOURCE; a hook that logs to the
    workspace appended a line to the source's file on every capture (and an
    aborted switch then left it there). Now the capture runs no forward at
    all: the log is unchanged, and the program does not even ship the fork
    helpers."""
    src = _hooked_source(api, name="nomodel", setup=HOOK_TO_FILE)
    log = program(f'print({MARK!r} + json.dumps(open("data/acts.log").read().count("forward")))')
    assert api.witness(src, log) == 1
    before, fp0 = api.witness(src, ACTS), _boundary_fp(api, src)
    cap = api.witness(src, ctl.capture_program())
    assert cap["expectation_source"] == "cut" and cap["expectation_method"]["model_code_run"] is False
    assert ctl.compare_fingerprints(cap["fingerprint"], cap["fingerprint_after"])["equal"]
    assert api.witness(src, log) == 1 and api.witness(src, ACTS) == before and _boundary_fp(api, src) == fp0
    body = ctl._capture_src()
    assert "compute_expectation" not in body and "run_isolated" not in body and "experiments.oracles" not in body


class _DefectApi(localapi.LocalApi):
    """Runs `defect` on the destination right after a successful restore: a
    restore that got something wrong the restore itself did not notice."""

    def __init__(self, *, defect: str, **kw):
        super().__init__(**kw)
        self.defect = defect

    def witness(self, pid, code, timeout_ms=600_000):
        out = super().witness(pid, code, timeout_ms)
        if _is_restore(code) and isinstance(out, dict) and out.get("ok"):
            super().witness(pid, self.defect)
        return out


CORRUPT_PARAM = program(f'''
import torch
with torch.no_grad():
    next(g["model"].parameters()).add_(1.0)
print({MARK!r} + "{{}}")
''')

DROP_REATTACHMENT = program(f'''
for grp in g["optimizer"].param_groups:
    grp["params"] = [p.detach().clone().requires_grad_(True) for p in grp["params"]]
print({MARK!r} + "{{}}")
''')


@pytest.mark.parametrize("defect,oracle", [(CORRUPT_PARAM, "parameters"), (DROP_REATTACHMENT, "optimizer")])
def test_validation_still_detects_a_real_defect(tmp_path, defect, oracle):
    api = _DefectApi(defect=defect, workdir=tmp_path / "kernels")
    try:
        src = _source(api)
        before = api.witness(src, SNAPSHOT)
        j = Journal(tmp_path / "j.sqlite")
        res = Controller(api, j, tmp_path / "blobs").migrate("defect", src, "cpu")
        assert res["phase"] == "ABORTED" and res["failed_at"] == "DEST_VALIDATED", res
        assert res["reason"] == "contract_check_failed" and oracle in res["detail"]
        assert j.get("defect")["authoritative"] == src and j.get("defect")["admission"] == "open"
        assert api.witness(src, SNAPSHOT) == before
    finally:
        api.close()


def _boundary_fp(api, pid):
    """The controller's fingerprint of a runtime, every compared or declared
    field (the timing aside)."""
    fp = api.witness(pid, ctl.fingerprint_program())
    return {k: v for k, v in fp.items() if k != "fingerprint_ms"}


def test_a_mutating_validation_is_caught_by_the_side_effect_check(api, tmp_path, monkeypatch):
    """Validation switched back to the in-place continuation step: every
    oracle that runs still passes, and the validation's own before/after
    fingerprint sees the optimizer and the parameters move, and aborts at
    DEST_VALIDATED, before anything is committed."""
    monkeypatch.setattr(ctl, "verify_program", functools.partial(ctl.verify_program, continuation="inplace"))
    src = _source(api)
    before = api.witness(src, SNAPSHOT)
    j = Journal(tmp_path / "j.sqlite")
    res = Controller(api, j, tmp_path / "blobs").migrate("mut", src, "cpu")
    assert res["timings"]["oracles"] == "11/11"
    assert res["phase"] == "ABORTED" and res["failed_at"] == "DEST_VALIDATED", res
    assert res["reason"].startswith("validation_side_effect:")
    assert {"modules", "optimizers"} <= set(res["reason"].split(":", 1)[1].split(","))
    row = j.get("mut")
    assert row["authoritative"] == src and row["admission"] == "open"
    assert json.loads(row["boundary"])["restored"]["equal"]                    # the restore itself was exact
    assert api.witness(src, SNAPSHOT) == before                                # and the source is untouched


def test_the_commit_boundary_is_an_independent_second_guard(api, tmp_path, monkeypatch):
    """The same mutating validation with the side-effect check switched off:
    the pre-commit fingerprint against the cut still catches it, so either
    guard alone refuses the commit."""
    monkeypatch.setattr(ctl, "verify_program", functools.partial(ctl.verify_program, continuation="inplace"))
    monkeypatch.setattr(ctl.Controller, "_check_validation_side_effects", lambda self, mig, v: None)
    src = _source(api)
    before = api.witness(src, SNAPSHOT)
    j = Journal(tmp_path / "j.sqlite")
    res = Controller(api, j, tmp_path / "blobs").migrate("mut2", src, "cpu")
    assert res["phase"] == "ABORTED" and res["failed_at"] == "COMMITTED", res
    assert res["reason"].startswith("commit_boundary_mismatch:")
    assert {"modules", "optimizers"} <= set(res["reason"].split(":", 1)[1].split(","))
    assert j.get("mut2")["authoritative"] == src and api.witness(src, SNAPSHOT) == before


# ---------------------------------------------------------------------------
# B2: the oracles themselves, in process
# ---------------------------------------------------------------------------

def _fixture_ns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from experiments.fixture import FixtureSpec, _digest_params, build_transported_namespace, compute_expectation
    ns = build_transported_namespace(FixtureSpec(seed=7, workspace_corpus_files=2, workspace_corpus_bytes=2048))
    exp = compute_expectation(ns).to_dict()
    exp["param_digest"] = _digest_params(ns["model"])       # the reference forward moved a buffer
    return ns, exp


def _observe(ns):
    import torch
    from experiments.fixture import _digest_params
    opt = ns["optimizer"]
    return {"digest": _digest_params(ns["model"]),
            "steps": [float(s["step"]) for s in opt.state.values()],
            "state_len": len(opt.state),
            "torch_rng": torch.get_rng_state().clone(),
            "python_rng": random.getstate(),
            "loader": ns["loader"].generator.get_state().clone(),
            "shared_keys": sorted(ns["shared_a"]),
            "training": ns["model"].training}


def _same(a, b):
    import torch
    return all((torch.equal(a[k], b[k]) if isinstance(a[k], torch.Tensor) else a[k] == b[k]) for k in a)


def test_oracles_copy_mode_writes_nothing_and_inplace_does(tmp_path, monkeypatch):
    from experiments.oracles import verify
    ns, exp = _fixture_ns(tmp_path, monkeypatch)
    before = _observe(ns)
    rows = verify(ns, exp, destination_device="cpu", source_device="cpu", continuation="copy")
    assert [r.name for r in rows if r.ok is not True] == [] and len(rows) == 12
    assert _same(before, _observe(ns))
    cont = next(r for r in rows if r.name == "continuation")
    assert "disposable copy" in cont.detail
    # The control: the default is unchanged for E13/E14/the driver, and it
    # does move the state.
    rows = verify(ns, exp, destination_device="cpu", source_device="cpu")
    assert [r.name for r in rows if not r.ok] == []
    after = _observe(ns)
    assert after["steps"] == [4.0] * 6 and after["digest"] != before["digest"]


def test_oracles_copy_mode_does_not_consume_process_rng(tmp_path, monkeypatch):
    """A model with dropout: the continuation step draws from torch's global
    stream. On the copy it must leave the process stream where it was."""
    import torch
    from experiments.fixture import _digest_params
    from experiments.oracles import verify
    torch.manual_seed(3)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.5), torch.nn.Linear(8, 1))
    ns = {"model": model, "optimizer": torch.optim.AdamW(model.parameters(), lr=1e-2),
          "loss_fn": torch.nn.MSELoss(), "train_x": torch.randn(16, 4), "train_y": torch.randn(16, 1)}
    exp = {"values": {}, "param_digest": _digest_params(model), "forward_output": None}
    state = torch.get_rng_state().clone()
    rows = verify(ns, exp, destination_device="cpu", source_device="cpu", continuation="copy")
    assert next(r for r in rows if r.name == "continuation").ok
    assert torch.equal(state, torch.get_rng_state())
    verify(ns, exp, destination_device="cpu", source_device="cpu", continuation="inplace")
    assert not torch.equal(state, torch.get_rng_state())                  # the control: in place, it draws


def test_oracles_skip_mode_runs_no_model_code(tmp_path, monkeypatch):
    from experiments.oracles import verify
    ns, exp = _fixture_ns(tmp_path, monkeypatch)
    before = _observe(ns)
    rows = verify(ns, exp, destination_device="cpu", source_device="cpu", continuation="skip")
    names = [r.name for r in rows]
    assert "forward output" not in names and "continuation" not in names and all(r.ok for r in rows)
    assert _same(before, _observe(ns))
    with pytest.raises(ValueError):
        verify(ns, exp, destination_device="cpu", source_device="cpu", continuation="sometimes")


def test_a_forward_declared_not_run_is_never_a_pass(tmp_path, monkeypatch):
    """An expectation that declares the forward not run (every expectation
    the controller's capture writes) yields a row that is NOT RUN: `ok` is
    None, the mode is "declared", the reason is the expectation's, and the
    totals count eleven checks run and one declared, never twelve passes."""
    from experiments.oracles import FORWARD_NOT_RUN_KEY, summarize, verify
    ns, exp = _fixture_ns(tmp_path, monkeypatch)
    exp = dict(exp, forward_output=None, **{FORWARD_NOT_RUN_KEY: ctl.FORWARD_NOT_RUN})
    rows = verify(ns, exp, destination_device="cpu", source_device="cpu", continuation="copy")
    fwd = next(r for r in rows if r.name == "forward output")
    assert (fwd.ok, fwd.mode) == (None, "declared") and fwd.detail == "NOT RUN: " + ctl.FORWARD_NOT_RUN
    assert len(rows) == 12 and [r.name for r in rows if r.ok is not True] == ["forward output"]
    ok, note = summarize(rows)
    assert ok and note.startswith("11 oracles pass") and note.endswith("1 not run (declared): forward output")
    # The fork-failure path keeps the declaration (it does not turn a row the
    # expectation declared into a failure).
    import experiments.oracles as oracles
    monkeypatch.setattr(oracles, "run_isolated", lambda fn, timeout=0: {"ok": False, "error": "child died"})
    rows = oracles.verify(ns, exp, destination_device="cpu", source_device="cpu", continuation="isolated")
    by = {r.name: r for r in rows}
    assert (by["forward output"].ok, by["forward output"].mode) == (None, "declared")
    assert by["continuation"].ok is False and "child died" in by["continuation"].detail


# ---------------------------------------------------------------------------
# The fingerprint, in process
# ---------------------------------------------------------------------------

def _fp_env():
    """The fingerprint source exec'd the way a kernel program does it."""
    import base64
    import hashlib
    import os as _os
    import time as _time

    import capsule.storage_sharing  # noqa: F401  (the program finds it in sys.modules)
    env = {"sys": sys, "os": _os, "_json": json, "_types": types, "_b64": base64, "_hl": hashlib,
           "_io": io, "_tar": tarfile, "_time": _time, "__name__": "__clusy_handoff__"}
    exec(compile(ctl._CAPTURE_FILTER + ctl._STATE_FP_SRC + ctl._RNG_SRC, "<fp>", "exec"), env)
    return env


def test_fingerprint_is_read_only_and_sensitive(tmp_path, monkeypatch):
    import torch
    ns, _ = _fixture_ns(tmp_path, monkeypatch)
    env = _fp_env()
    before = _observe(ns)
    a = env["_state_fingerprint"](ns, {})
    b = env["_state_fingerprint"](ns, {})
    assert a == b and _same(before, _observe(ns))
    # Sensitive to what it claims: one parameter element, an optimizer that
    # no longer points at the model, a Parameter that became a Tensor, a
    # workspace byte, a python RNG draw.
    moved = dict(ns, model=copy.deepcopy(ns["model"]))      # (the namespace holds a module: no deepcopy)
    with torch.no_grad():
        next(moved["model"].parameters()).view(-1)[0] += 1
    assert env["_state_fingerprint"](moved, {})["modules"] != a["modules"]
    unbound = dict(ns)
    unbound["optimizer"] = copy.deepcopy(ns["optimizer"])
    assert env["_state_fingerprint"](unbound, {})["optimizers"] != a["optimizers"]
    typed = dict(ns, top=torch.nn.Parameter(torch.ones(2)))
    plain = dict(ns, top=torch.ones(2))
    assert env["_state_fingerprint"](typed, {})["tensors"]["top"] != env["_state_fingerprint"](plain, {})["tensors"]["top"]
    (tmp_path / "data" / "extra.txt").write_text("x")
    assert env["_state_fingerprint"](ns, {})["workspace"] != a["workspace"]
    random.random()
    assert env["_state_fingerprint"](ns, {})["rng"]["python"] != a["rng"]["python"]


def test_rng_envelope_round_trip_and_a_declared_cuda_drop():
    import numpy as np
    import torch
    env = _fp_env()
    saved = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        envelope = env["_rng_capture"]()
        fp0 = env["_fp_rng"]({})
        random.random(); np.random.rand(); torch.rand(1)
        assert env["_fp_rng"]({}) != fp0
        # A capsule from a GPU source carries CUDA state; this host has none.
        envelope = json.loads(json.dumps(envelope))
        envelope["cuda"] = {"initialized": True, "states": [__import__("base64").b64encode(bytes(16)).decode()]}
        rep = env["_rng_apply"](envelope)
        assert rep == {"python": True, "numpy": True, "torch_cpu": True, "cuda": "declared_drop"}
        assert env["_fp_rng"]({}) == fp0
    finally:
        random.setstate(saved[0]); np.random.set_state(saved[1]); torch.set_rng_state(saved[2])


def _fp(**over):
    base = {f: {} for f in ctl.BOUNDARY_FIELDS}
    base["rng"] = {"python": {"sha256": "p"}, "numpy": {"loaded": True, "sha256": "n"},
                   "torch_cpu": {"loaded": True, "sha256": "t"}, "named": {}}
    base["rng_cuda"] = {"available": False, "initialized": False}
    base["devices"] = {"model.param.w": "cpu"}
    base["grads"] = {}
    base["modules"] = {"model": {"params": [["w", {"sha256": "abc"}]]}}
    base.update(over)
    return base


def test_compare_fingerprints_declared_and_undeclared_differences():
    cmp = ctl.compare_fingerprints
    assert cmp(_fp(), _fp()) == {"equal": True, "mismatched": [], "details": {}, "declared": {"rng_cuda": "absent"}}
    # A device move with identical bytes is declared, not a mismatch.
    v = cmp(_fp(devices={"model.param.w": "cuda:0"}), _fp())
    assert v["equal"] and v["declared"]["devices"] == {"model.param.w": ["cuda:0", "cpu"]}
    # Different bytes are a mismatch, named by field and key.
    v = cmp(_fp(), _fp(modules={"model": {"params": [["w", {"sha256": "abd"}]]}}))
    assert not v["equal"] and v["mismatched"] == ["modules"] and v["details"]["modules"] == ["model"]
    # CUDA RNG: carried from a GPU source to a CPU destination is a declared
    # drop; a CPU source to a GPU destination is not carried; carried and
    # restored must match.
    gpu = {"available": True, "initialized": True, "devices": [{"index": 0, "sha256": "c", "draws": ["1"]}]}
    assert cmp(_fp(rng_cuda=gpu), _fp())["declared"]["rng_cuda"] == "declared_drop"
    assert cmp(_fp(), _fp(rng_cuda={"available": True, "initialized": False}))["declared"]["rng_cuda"] == "not_carried"
    assert cmp(_fp(rng_cuda=gpu), _fp(rng_cuda=gpu))["declared"]["rng_cuda"] == "equal"
    other = {**gpu, "devices": [{"index": 0, "sha256": "d", "draws": ["2"]}]}
    v = cmp(_fp(rng_cuda=gpu), _fp(rng_cuda=other))
    assert not v["equal"] and "rng_cuda" in v["mismatched"]
    # A stream the source never loaded is not compared.
    v = cmp(_fp(rng={**_fp()["rng"], "numpy": {"loaded": False}}), _fp())
    assert v["equal"] and v["declared"]["rng_numpy"] == "not_loaded_at_source"
    # Gradients are carried in the capsule, so a lost or changed one is a
    # MISMATCH (the previous version declared it and committed without them).
    v = cmp(_fp(grads={"model.param.w": {"sha256": "g"}}), _fp(grads={"model.param.w": None}))
    assert not v["equal"] and v["mismatched"] == ["grads"] and "grads_not_carried" not in v["declared"]
    # A value neither side could digest is declared by name, from either side.
    v = cmp(_fp(unhashable=["x: meta tensor (no data)"]), _fp())
    assert v["equal"] and v["declared"]["unhashable"] == ["x: meta tensor (no data)"]


# ---------------------------------------------------------------------------
# B3: a view that could not be restored blocks the commit
# ---------------------------------------------------------------------------

def test_restore_program_refuses_when_a_view_record_fails(api, tmp_path):
    """Old code: `ok: true` with the failure inside the report."""
    src = api.create_project("clusy-exp-handoff-src-views", "cpu")
    api.witness(src, program(f'import torch\ng["t"] = torch.arange(6.)\ng["t2"] = torch.arange(4.)\n'
                             f'print({MARK!r} + "{{}}")'))
    cap = api.witness(src, ctl.capture_program())
    import base64
    import hashlib
    blob = base64.b64decode(cap["b64"])
    out = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tin, tarfile.open(fileobj=out, mode="w:gz") as tout:
        for m in tin.getmembers():
            data = tin.extractfile(m).read() if m.isfile() else None
            if m.name == "manifests.json":
                man = json.loads(data)
                man["views"]["views"].append({"path": ["t2"], "kind": "torch", "base_name": "__no_such_base__",
                                              "shape": [2], "stride": [1], "offset": 0, "dtype": "torch.float32"})
                data = json.dumps(man).encode()
                m.size = len(data)
            tout.addfile(m, io.BytesIO(data) if data is not None else None)
    blob = out.getvalue()
    dest = api.create_project("clusy-exp-handoff-views-dest", "cpu")
    Controller(api, Journal(tmp_path / "j.sqlite"), tmp_path / "blobs")._upload(dest, blob)
    res = api.witness(dest, ctl.restore_program(hashlib.sha256(blob).hexdigest()))
    assert res["ok"] is False and res["reason"] == "views_not_restored", res
    assert [f["path"] for f in res["views"]["failed"]] == [["t2"]]


#: Appended to the storage adapter's source as the kernels receive it: the
#: adapter now reports a failure for every restore.
_FAILING_ADAPTER = '''
_clusy_test_apply = apply_view_manifest
def apply_view_manifest(ns, manifest):
    r = dict(_clusy_test_apply(ns, manifest))
    r["failed"] = list(r.get("failed", [])) + [{"path": ["model"], "reason": "made to fail by the test"}]
    return r
'''


def test_a_failed_view_repair_aborts_before_dest_restored_is_journaled(api, tmp_path, monkeypatch):
    monkeypatch.setattr(ctl, "SHARING_SRC", ctl.SHARING_SRC + _FAILING_ADAPTER)
    src = _source(api)
    j = Journal(tmp_path / "j.sqlite")
    res = Controller(api, j, tmp_path / "blobs").migrate("vfail", src, "cpu")
    assert res["phase"] == "ABORTED" and res["failed_at"] == "DEST_RESTORED", res
    assert res["reason"] == "views_not_restored"
    assert all(e["phase"] != "DEST_RESTORED" for e in j.events("vfail"))
    assert j.get("vfail")["authoritative"] == src and j.get("vfail")["admission"] == "open"


def test_hidden_base_survives_the_controller_capture_and_restore(api, tmp_path):
    """controller_probes.py, scenario 3 (which the reviewer confirmed FIXED
    and asked not to regress): two overlapping slices whose base has no
    binding go through the real capture and restore programs, and a write
    through one is visible through the other. Also: the restore's own
    fingerprint says the views are intact."""
    src = _source(api, name="hidden")
    api.witness(src, program(f'import torch\nb = torch.arange(10, dtype=torch.float32)\n'
                             f'g["audit_a"], g["audit_b"] = b[1:5], b[3:7]\nprint({MARK!r} + "{{}}")'))
    cap = api.witness(src, ctl.capture_program())
    dest = api.create_project("clusy-exp-handoff-hidden-dest", "cpu")
    import base64
    Controller(api, Journal(tmp_path / "views.sqlite"), tmp_path / "views")._upload(dest, base64.b64decode(cap["b64"]))
    restore = api.witness(dest, ctl.restore_program(cap["sha256"]))
    assert restore["ok"] and restore["views"]["failed"] == [] and restore["views"]["restored"] == 2, restore
    check = api.witness(dest, program(f'g["audit_a"][2] = 99\nprint({MARK!r} + json.dumps({{"second_view": float(g["audit_b"][0])}}))'))
    assert check["second_view"] == 99
    views = restore["fingerprint"]["views"]
    if views.get("available"):                  # the adapter's non-mutating check, when present
        assert views["checked"] == views["intact"] == 2 and views["broken"] == []
        assert cap["fingerprint"]["views"] == views


# ---------------------------------------------------------------------------
# B3: the reviewer's storage cases at the commit boundary
# ---------------------------------------------------------------------------

#: The reviewer's three cases (semantic_probes.py), under names that do not
#: collide with the fixture: a repeated reference to a view, a Parameter view
#: bound in a module with an optimizer and an external alias, and a top-level
#: Parameter sharing a base.
STORAGE_CASES = '''
import torch
g["vbase"] = torch.arange(8, dtype=torch.float32)
g["vview"] = g["vbase"][1:4]
g["valias"] = g["vview"]
g["pbase"] = torch.arange(8, dtype=torch.float32)
g["vmodel"] = torch.nn.Linear(4, 1, bias=False)
g["vmodel"].weight = torch.nn.Parameter(g["pbase"][:4].reshape(1, 4))
g["vopt"] = torch.optim.SGD(g["vmodel"].parameters(), lr=0.1)
g["weight_alias"] = g["vmodel"].weight
g["tbase"] = torch.arange(8, dtype=torch.float32)
g["tparam"] = torch.nn.Parameter(g["tbase"][:4])
'''

STORAGE_PROBE = program(f'''
import torch
_r = {{"view_is_alias": g["vview"] is g["valias"],
      "opt_is_model": g["vopt"].param_groups[0]["params"][0] is g["vmodel"].weight,
      "alias_is_model": g["weight_alias"] is g["vmodel"].weight,
      "tparam_is_parameter": isinstance(g["tparam"], torch.nn.Parameter),
      "tparam_requires_grad": bool(g["tparam"].requires_grad)}}
g["vbase"][1] = 99
_r["view_follows_base"] = float(g["vview"][0]) == 99 and float(g["valias"][0]) == 99
_w = g["vmodel"].weight.detach().clone()
_p = g["pbase"].clone()
g["vopt"].zero_grad(); g["vmodel"](torch.ones(1, 4)).sum().backward(); g["vopt"].step()
_r["step_moves_model"] = not torch.equal(_w, g["vmodel"].weight.detach())
_r["step_moves_base"] = not torch.equal(_p, g["pbase"])
print({MARK!r} + json.dumps(_r))
''')

#: The first adapter's repair, reinstated for the test: a NEW view rebound at
#: one path, reported as a success. This is the reviewer's defect, whatever
#: version of the adapter is on disk.
_OLD_STYLE_REPAIR = '''
def apply_view_manifest(ns, manifest):
    import torch
    m = manifest if isinstance(manifest, ViewManifest) else ViewManifest.from_json(manifest)
    n = 0
    for v in m.views:
        base = ns.get(v.base_name)
        if base is None or v.kind != "torch":
            continue
        view = base.as_strided(tuple(v.shape), tuple(v.stride), v.offset)
        path = tuple(v.path)
        if len(path) == 1:
            ns[path[0]] = view
        elif path[1] == "param":
            mod = ns[path[0]]
            *parents, leaf = path[2].split(".")
            for p in parents:
                mod = getattr(mod, p)
            setattr(mod, leaf, torch.nn.Parameter(view, requires_grad=getattr(mod, leaf).requires_grad))
        n += 1
    return {"restored": n, "failed": [], "unsupported_at_capture": len(m.unsupported)}
'''


def test_the_controller_never_commits_a_broken_object_graph(api, tmp_path):
    """Whatever the adapter on disk does, a DONE migration has every one of
    the reviewer's relationships intact at the destination; anything else
    must abort. Old code (controller and adapter): DONE with all three
    broken."""
    src = _source(api, extra=STORAGE_CASES)
    res = Controller(api, Journal(tmp_path / "j.sqlite"), tmp_path / "blobs").migrate("graph", src, "cpu")
    if res["phase"] == "DONE":
        probe = api.witness(res["dest"], STORAGE_PROBE)
        assert all(probe.values()), probe
    else:
        assert res["phase"] == "ABORTED" and res["reason"].startswith(("commit_boundary_mismatch:",
                                                                       "views_not_restored")), res


def test_the_commit_boundary_catches_a_repair_that_reports_success(api, tmp_path, monkeypatch):
    """The adapter's report is not trusted: with the first adapter's repair
    reinstated (zero failures reported), the fingerprint still sees the
    broken identity, optimizer binding and Parameter class, and aborts."""
    monkeypatch.setattr(ctl, "SHARING_SRC", ctl.SHARING_SRC + _OLD_STYLE_REPAIR)
    src = _source(api, extra=STORAGE_CASES)
    j = Journal(tmp_path / "j.sqlite")
    res = Controller(api, j, tmp_path / "blobs").migrate("graph-old", src, "cpu")
    assert res["phase"] == "ABORTED" and res["failed_at"] == "DEST_RESTORED", res
    assert res["reason"].startswith("commit_boundary_mismatch:"), res
    fields = set(res["reason"].split(":", 1)[1].split(","))
    assert {"aliases", "optimizers"} <= fields, fields
    assert j.get("graph-old")["authoritative"] == src


# ---------------------------------------------------------------------------
# B3: a capsule from a CUDA source restores on a CPU destination
# ---------------------------------------------------------------------------

#: Registered in the SOURCE kernel for the duration of the capture only: every
#: CPU storage is written with the location tag `cuda:0`, exactly what a
#: capture on a GPU source produces. (The preflight's round trip loads
#: pickles in the source, so the tagger must not be active then.)
_TAG_CUDA = program(f'''
import torch
def _clusy_test_tag(obj):
    return "cuda:0" if getattr(getattr(obj, "device", None), "type", None) == "cpu" else None
torch.serialization.register_package(1, _clusy_test_tag, lambda o, l: None)
print({MARK!r} + "{{}}")
''')
_UNTAG = program(f'''
import torch
torch.serialization._package_registry[:] = [e for e in torch.serialization._package_registry if e[0] != 1]
print({MARK!r} + "{{}}")
''')


class _CudaTaggedSourceApi(localapi.LocalApi):
    def witness(self, pid, code, timeout_ms=600_000):
        if "_dump_filtered_session" in code:
            super().witness(pid, _TAG_CUDA)
            try:
                return super().witness(pid, code, timeout_ms)
            finally:
                super().witness(pid, _UNTAG)
        return super().witness(pid, code, timeout_ms)


def test_a_cuda_tagged_capsule_restores_on_a_cpu_destination(tmp_path, monkeypatch):
    """Old code: `checkpoint_restore_failed`, 'Attempting to deserialize
    object on a CUDA device but torch.cuda.is_available() is False'."""
    import torch
    if torch.cuda.is_available():
        pytest.skip("the remap is only exercised on a host without CUDA")
    api = _CudaTaggedSourceApi(workdir=tmp_path / "kernels")
    try:
        src = _source(api)
        before = api.witness(src, SNAPSHOT)
        res = Controller(api, Journal(tmp_path / "j.sqlite"), tmp_path / "blobs").migrate("remap", src, "cpu")
        assert res["phase"] == "DONE", res
        remap = res["restore"]["remap"]
        assert remap["registered"] and remap["remapped"] > 0 and set(remap["locations"]) == {"cuda:0"}
        assert api.witness(res["dest"], SNAPSHOT) == before
        # The control: the same capsule without the handler fails the way a
        # CPU destination fails today, so the tags are real.
        monkeypatch.setattr(ctl, "_REMAP_SRC", ctl._REMAP_SRC + "\ndef _remap_install(needed):\n"
                                                                  "    return {'registered': False}, None\n")
        src2 = _source(api, name="src2")
        res2 = Controller(api, Journal(tmp_path / "j2.sqlite"), tmp_path / "blobs2").migrate("noremap", src2, "cpu")
        assert res2["phase"] == "ABORTED" and res2["reason"] == "checkpoint_restore_failed", res2
        assert "CUDA" in res2["detail"]
    finally:
        api.close()


# ---------------------------------------------------------------------------
# Second review: model code must not run in the kernel it validates
# ---------------------------------------------------------------------------

#: The reviewer's adjacent.py case `hook_dict_of_tensors`: a forward hook that
#: stores activations in a TOP-LEVEL dict. A deepcopy copies the hook function
#: by reference, so a forward on the copy wrote into the real dict.
HOOK = '''
import torch
acts = {}
def save_act(mod, inp, out):
    acts["fc2"] = out.detach().clone()
    acts["calls"] = acts.get("calls", 0) + 1
hook_handle = model.fc2.register_forward_hook(save_act)
model.eval()
with torch.no_grad():
    model(train_x)
model.train()
'''

ACTS = program(f'''
import hashlib
a = g["acts"]
print({MARK!r} + json.dumps({{"calls": a["calls"], "fc2": hashlib.sha256(a["fc2"].numpy().tobytes()).hexdigest()}}))
''')


def _hooked_source(api, name="hooked", setup=HOOK):
    src = _source(api, name=name)
    st, _out, payload = api.execute(src, setup)          # a user cell, run in __main__ itself
    assert st == 200 and not payload.get("error"), payload
    return src


def test_a_hook_writing_the_namespace_is_committed_unchanged(api, tmp_path):
    """Old code (copy validation + deepcopy refresh): DONE, oracles 12/12,
    both boundaries 'equal', and the destination's `acts` had calls 1 -> 4 and
    different activations. Now the source runs no model code at all and
    every oracle runs in a forked child: the committed dict is the captured
    one, the validation's own before/after comparison saw nothing change,
    and the boundary's deep `objects` field shows it was compared."""
    src = _hooked_source(api)
    before, snap = api.witness(src, ACTS), api.witness(src, SNAPSHOT)
    res = Controller(api, Journal(tmp_path / "j.sqlite"), tmp_path / "blobs").migrate("hook", src, "cpu")
    assert res["phase"] == "DONE", res
    _declared_forward_only(res)
    assert res["validation"]["isolated"]
    assert res["capture"]["expectation_method"]["model_code_run"] is False
    assert api.witness(res["dest"], ACTS) == before == {"calls": 1, "fc2": before["fc2"]}
    assert api.witness(res["dest"], SNAPSHOT) == snap
    cut = res["commit_boundary"]["source"]
    assert "acts" in cut["objects"] and cut["objects"] == res["commit_boundary"]["pre_commit"]["objects"]


def test_an_aborted_switch_leaves_the_hooked_source_unchanged(api, tmp_path):
    """refresh_mutates_source.py. Old code: ABORTED (bad_capsule) with the
    source's hook dict at calls 1 -> 2, and the capture reporting the source
    untouched. Now the source's state is compared, not only its bindings."""
    src = _hooked_source(api)
    before, snap = api.witness(src, ACTS), api.witness(src, SNAPSHOT)
    res = Controller(api, Journal(tmp_path / "j.sqlite"), tmp_path / "blobs").migrate(
        "hook-abort", src, "cpu", fault="bad_capsule")
    assert res["phase"] == "ABORTED" and res["reason"] == "checkpoint_transfer_failed", res
    assert res["capture"]["source_state_unchanged"] is True and res["capture"]["source_bindings_identical"]
    assert api.witness(src, ACTS) == before and api.witness(src, SNAPSHOT) == snap


def test_the_deep_fingerprint_catches_an_escaped_hook_write(api, tmp_path, monkeypatch):
    """The control for the two tests above: validation switched back to the
    deepcopy ("copy") mode. The continuation step's forward runs the hook
    against the destination's REAL dict (a deepcopy shares the hook's
    closure); every check that runs still passes, and the validation's own
    before/after fingerprint sees `acts` change and aborts at DEST_VALIDATED.
    With that check switched off, the pre-commit fingerprint's `objects`
    field is the second, independent guard. The source is untouched either
    way."""
    monkeypatch.setattr(ctl, "VALIDATION_CONTINUATION", "copy")
    src = _hooked_source(api)
    before, fp0 = api.witness(src, ACTS), _boundary_fp(api, src)
    j = Journal(tmp_path / "j.sqlite")
    res = Controller(api, j, tmp_path / "blobs").migrate("hook-copy", src, "cpu")
    assert res["timings"]["oracles"] == "11/11"
    assert res["phase"] == "ABORTED" and res["failed_at"] == "DEST_VALIDATED", res
    assert res["reason"] == "validation_side_effect:objects", res
    assert res["validation"]["side_effects"]["details"] == {"objects": ["acts"]}
    assert j.get("hook-copy")["authoritative"] == src and j.get("hook-copy")["admission"] == "open"
    assert api.witness(src, ACTS) == before and _boundary_fp(api, src) == fp0
    monkeypatch.setattr(ctl.Controller, "_check_validation_side_effects", lambda self, mig, v: None)
    j2 = Journal(tmp_path / "j2.sqlite")
    res = Controller(api, j2, tmp_path / "blobs2").migrate("hook-copy2", src, "cpu")
    assert res["phase"] == "ABORTED" and res["failed_at"] == "COMMITTED", res
    assert res["reason"] == "commit_boundary_mismatch:objects", res
    assert json.loads(j2.get("hook-copy2")["boundary"])["pre_commit"]["details"]["objects"] == ["acts"]
    assert j2.get("hook-copy2")["authoritative"] == src
    assert api.witness(src, ACTS) == before and _boundary_fp(api, src) == fp0


#: Appended to the oracles module as the kernels receive it: no fork here, as
#: on a kernel whose CUDA is initialized.
_NO_FORK = '\n\ndef fork_unsafe_reason():\n    return "test: fork disabled"\n'


def test_where_no_fork_can_isolate_the_step_runs_on_a_copy_and_fails_closed(api, tmp_path, monkeypatch):
    """With the fork unavailable (the CUDA-initialized case) the continuation
    step runs IN the destination kernel on one disposable copy, bracketed by
    the before/after fingerprint. (The first version declared both model rows
    not run there, so the step went untested exactly where the hardware
    changed.) A plain workload commits exactly its captured state, with user
    training before the switch so the expectation must be the cut's. A
    forward hook that writes a global escapes the copy (deepcopy shares the
    hook's closure): the validation's own comparison sees `acts` change and
    the switch fails closed, with the source untouched."""
    monkeypatch.setattr(ctl, "ORACLES_SRC", ctl.ORACLES_SRC + _NO_FORK)
    src = _source(api, name="nofork")
    assert api.witness(src, TRAIN)["step_count"] == 5
    snap = api.witness(src, SNAPSHOT)
    res = Controller(api, Journal(tmp_path / "j.sqlite"), tmp_path / "blobs").migrate("nofork", src, "cpu")
    assert res["phase"] == "DONE", res
    _declared_forward_only(res)
    assert res["validation"]["isolated"] is False and "fork disabled" in res["validation"]["not_isolated_because"]
    rows = api.witness(res["dest"], ctl.recovery_check_program())["rows"]
    cont = next(r for r in rows if r["name"] == "continuation")
    assert cont["ok"] is True and "on a disposable copy" in cont["detail"] and "fork disabled" in cont["detail"]
    assert res["capture"]["expectation_source"] == "cut"
    exp = api.witness(res["dest"], EXP_PROG)
    assert exp["step_count"] == 5 and exp["forward_output_not_run"] == ctl.FORWARD_NOT_RUN
    assert api.witness(res["dest"], SNAPSHOT) == snap

    hooked = _hooked_source(api, name="nofork-hook", setup=HOOK)
    before, snap, fp0 = api.witness(hooked, ACTS), api.witness(hooked, SNAPSHOT), _boundary_fp(api, hooked)
    j = Journal(tmp_path / "j2.sqlite")
    res = Controller(api, j, tmp_path / "blobs2").migrate("nofork-hook", hooked, "cpu")
    assert res["phase"] == "ABORTED" and res["failed_at"] == "DEST_VALIDATED", res
    assert res["reason"] == "validation_side_effect:objects", res
    assert res["validation"]["side_effects"]["details"] == {"objects": ["acts"]}
    assert j.get("nofork-hook")["authoritative"] == hooked and j.get("nofork-hook")["admission"] == "open"
    assert api.witness(hooked, ACTS) == before and api.witness(hooked, SNAPSHOT) == snap
    assert _boundary_fp(api, hooked) == fp0


#: The same hook, logging to a WORKSPACE file on every forward: what a fork
#: does not isolate. The refresh's forward (in the child) appends a line.
HOOK_TO_FILE = HOOK.replace('acts["calls"] = acts.get("calls", 0) + 1',
                            'acts["calls"] = acts.get("calls", 0) + 1\n'
                            '    open("data/acts.log", "a").write("forward\\n")')


#: sha256 of every file under the runtime's workspace, read independently of
#: the controller's fingerprint.
WORKSPACE = program(f'''
import hashlib
out = {{}}
for d, _dirs, fs in os.walk("data"):
    for f in fs:
        p = os.path.join(d, f)
        out[p] = hashlib.sha256(open(p, "rb").read()).hexdigest()
print({MARK!r} + json.dumps(out, sort_keys=True))
''')


def test_a_hook_that_writes_the_workspace_is_caught_not_committed(api, tmp_path):
    """The stated limit of the fork: it contains memory, not the filesystem.
    The first version ran the refresh's forward on the SOURCE (in a forked
    child), which appended a line to the source's workspace log; the switch
    aborted, and the source kept the line. Now the source runs no model code,
    and the destination's validation (whose continuation step runs the hook
    in a forked child) is bracketed by a before/after fingerprint that
    includes the workspace: the line it appended on the DESTINATION aborts
    the switch (`validation_side_effect:workspace`) before anything is
    committed. The source is unchanged, workspace hashes and fingerprint
    both, and so it is after an aborted migration (`bad_capsule`), the
    reviewer-adjacent case."""
    src = _hooked_source(api, setup=HOOK_TO_FILE)
    log = program(f'print({MARK!r} + json.dumps(open("data/acts.log").read().count("forward")))')
    assert api.witness(src, log) == 1
    before, ws0, fp0 = api.witness(src, ACTS), api.witness(src, WORKSPACE), _boundary_fp(api, src)
    assert "data/acts.log" in ws0
    j = Journal(tmp_path / "j.sqlite")
    res = Controller(api, j, tmp_path / "blobs").migrate("fs", src, "cpu")
    assert res["phase"] == "ABORTED" and res["failed_at"] == "DEST_VALIDATED", res
    assert res["reason"] == "validation_side_effect:workspace", res
    assert res["validation"]["side_effects"]["details"] == {"workspace": ["acts.log"]}
    assert res["capture"]["source_state_unchanged"] is True
    assert j.get("fs")["authoritative"] == src and j.get("fs")["admission"] == "open"
    assert api.witness(src, log) == 1
    assert api.witness(src, ACTS) == before and api.witness(src, WORKSPACE) == ws0 and _boundary_fp(api, src) == fp0
    # The aborted migration: the capsule is corrupted in transit, so the
    # switch aborts at DEST_RESTORED and validation never runs. The source's
    # capture ran no forward, so nothing reached its workspace.
    res = Controller(api, Journal(tmp_path / "j2.sqlite"), tmp_path / "blobs2").migrate(
        "fs-abort", src, "cpu", fault="bad_capsule")
    assert res["phase"] == "ABORTED" and res["reason"] == "checkpoint_transfer_failed", res
    assert res["capture"]["source_state_unchanged"] is True
    assert api.witness(src, log) == 1
    assert api.witness(src, ACTS) == before and api.witness(src, WORKSPACE) == ws0 and _boundary_fp(api, src) == fp0


# ---------------------------------------------------------------------------
# Second review: gradients, object arrays, sparse and quantized tensors
# ---------------------------------------------------------------------------

PENDING_GRADS = program(f'''
g["optimizer"].zero_grad()
g["loss_fn"](g["model"](g["train_x"]), g["train_y"]).backward()
g["grad_alias"] = g["model"].fc1.weight.grad
print({MARK!r} + "{{}}")
''')

GRADS = program(f'''
import hashlib
m = g["model"]
print({MARK!r} + json.dumps({{
    "grads": [None if p.grad is None else hashlib.sha256(p.grad.numpy().tobytes()).hexdigest() for p in m.parameters()],
    "alias_is_grad": g.get("grad_alias") is m.fc1.weight.grad}}))
''')


def test_pending_gradients_are_carried_and_compared(api, tmp_path, monkeypatch):
    """adjacent.py `pending_grads` (backward done, step not taken). Old code:
    DONE with every `.grad` gone and both boundaries 'equal' (declared
    `grads_not_carried`). Now the capsule carries them, identity included, and
    a restore that drops them is a mismatch."""
    src = _source(api)
    api.witness(src, PENDING_GRADS)
    before = api.witness(src, GRADS)
    assert all(before["grads"]) and before["alias_is_grad"]
    res = Controller(api, Journal(tmp_path / "j.sqlite"), tmp_path / "blobs").migrate("grads", src, "cpu")
    assert res["phase"] == "DONE", res
    assert api.witness(res["dest"], GRADS) == before
    assert res["restore"]["grads"]["carried"] == res["restore"]["grads"]["reattached"] == 6
    assert "grads_not_carried" not in res["commit_boundary"]["verdicts"]["pre_commit"]["declared"]
    # The control: the reattachment switched off. The fingerprint aborts.
    monkeypatch.setattr(ctl, "_STATE_FP_SRC", ctl._STATE_FP_SRC + "\ndef _grads_apply(g, carried):\n"
                        "    return {'carried': 0, 'reattached': 0, 'failed': []}\n")
    src2 = _source(api, name="grads2")
    api.witness(src2, PENDING_GRADS)
    res2 = Controller(api, Journal(tmp_path / "j2.sqlite"), tmp_path / "blobs2").migrate("grads-off", src2, "cpu")
    assert res2["phase"] == "ABORTED" and res2["failed_at"] == "DEST_RESTORED", res2
    assert "grads" in res2["reason"].split(":", 1)[1].split(","), res2["reason"]


EXOTIC = '''
import numpy as np, torch
labels = np.array(["cat", None, 3, "dog"], dtype=object)
records = np.array([(1, "a"), (2, None)], dtype=[("n", "i4"), ("o", "O")])
sp = torch.sparse_coo_tensor(torch.tensor([[0, 1], [1, 0]]), torch.tensor([1.0, 2.0]), (2, 2))
csr = torch.tensor([[0.0, 3.0], [4.0, 0.0]]).to_sparse_csr()
qt = torch.quantize_per_tensor(torch.tensor([0.5, -1.0, 2.0]), 0.1, 3, torch.quint8)
'''

EXOTIC_PROBE = program(f'''
print({MARK!r} + json.dumps({{"labels": [repr(x) for x in g["labels"]], "records": repr(g["records"].tolist()),
                             "sp": g["sp"].to_dense().tolist(), "csr": g["csr"].to_dense().tolist(),
                             "qt": g["qt"].dequantize().tolist()}}))
''')


def test_object_arrays_sparse_and_quantized_tensors_migrate(api, tmp_path):
    """adjacent.py `object_ndarray` and `sparse_tensor`. Old code: the object
    array ALWAYS aborted (`commit_boundary_mismatch:tensors`: its raw bytes
    are pointers) and the sparse tensor crashed the capture (`no_witness`).
    Now both, plus CSR and quantized tensors, migrate with every value
    compared (nothing left undigested)."""
    src = _source(api, name="exotic")
    st, _o, p = api.execute(src, EXOTIC)
    assert st == 200 and not p.get("error"), p
    before = api.witness(src, EXOTIC_PROBE)
    res = Controller(api, Journal(tmp_path / "j.sqlite"), tmp_path / "blobs").migrate("exotic", src, "cpu")
    assert res["phase"] == "DONE", res
    assert api.witness(res["dest"], EXOTIC_PROBE) == before
    cut = res["commit_boundary"]["source"]
    assert cut["unhashable"] == []
    assert all(not str(cut["tensors"][k]["sha256"]).startswith("unhashable") for k in ("labels", "records", "sp", "csr", "qt"))


def test_fingerprint_digests_exotic_values_and_names_what_it_cannot(tmp_path, monkeypatch):
    import pickle
    import numpy as np
    import torch
    env = _fp_env()
    a = np.array(["cat", None, 3, {"k": [1, 2]}], dtype=object)
    b = pickle.loads(pickle.dumps(a))
    assert np.ascontiguousarray(a).tobytes() != np.ascontiguousarray(b).tobytes()     # the old digest's input
    assert env["_fp_array"](a) == env["_fp_array"](b)
    c = b.copy(); c[3] = {"k": [1, 3]}
    assert env["_fp_array"](c) != env["_fp_array"](a)
    sp = torch.sparse_coo_tensor(torch.tensor([[0, 1], [1, 0]]), torch.tensor([1.0, 2.0]), (2, 2))
    sp2 = torch.sparse_coo_tensor(torch.tensor([[0, 1], [1, 0]]), torch.tensor([1.0, 2.5]), (2, 2))
    uncoalesced = torch.sparse_coo_tensor(torch.tensor([[1, 0], [0, 1]]), torch.tensor([2.0, 1.0]), (2, 2))
    d = env["_fp_tensor_digest"]
    assert d(sp) != d(sp2) and d(sp) == d(uncoalesced) and not d(sp).startswith("unhashable")
    q = torch.quantize_per_tensor(torch.tensor([0.5, 1.0]), 0.1, 3, torch.quint8)
    assert d(q) != d(torch.quantize_per_tensor(torch.tensor([0.5, 1.2]), 0.1, 3, torch.quint8))
    assert d(q) != d(torch.quantize_per_tensor(torch.tensor([0.5, 1.0]), 0.2, 3, torch.quint8))   # the scale counts
    assert d(torch.tensor([1.0, 2.0]).conj()) == d(torch.tensor([1.0, 2.0]))
    ns, _ = _fixture_ns(tmp_path, monkeypatch)
    ns = dict(ns, ghost=torch.empty(3, device="meta"))
    fp = env["_state_fingerprint"](ns, {})
    assert fp["unhashable"] == ["ghost: meta tensor (no data)"]
    assert fp["tensors"]["ghost"]["sha256"] == "unhashable:meta tensor (no data)"


def test_the_deep_digest_is_process_independent():
    """`objects` must compare equal across two processes: set iteration order
    follows per-process string hashing, and ids differ. Two interpreters with
    different hash seeds digest the same value identically."""
    import subprocess
    code = (
        "import sys, json, base64, hashlib, io, os, tarfile, time, types\n"
        f"sys.path[:0] = [{str(ROOT / 'src')!r}, {str(ROOT / 'experiments')!r}]\n"
        "from handoff import controller as ctl\n"
        "import capsule.storage_sharing\n"
        "env = {'sys': sys, 'os': os, '_json': json, '_types': types, '_b64': base64, '_hl': hashlib, '_io': io,"
        " '_tar': tarfile, '_time': time, '__name__': '__clusy_handoff__'}\n"
        "exec(compile(ctl._CAPTURE_FILTER + ctl._STATE_FP_SRC, '<fp>', 'exec'), env)\n"
        "import torch\n"
        "shared = [1, 2]\n"
        "class Box:\n    def __init__(self):\n        self.tags = {'alpha', 'beta', 'gamma', 'delta'}\n"
        "        self.t = torch.arange(3.)\n        self.pair = (shared, shared)\n"
        "print(env['_fp_deep']({'box': Box(), 'fs': frozenset({'x', 'y', 'z'}), 'q': {('a', 1), ('b', 2)}}, {}))\n")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**__import__("os").environ, "PYTHONHASHSEED": seed}).stdout
            for seed in ("1", "2", "3")}
    assert len(outs) == 1, outs
    assert "<ref:" in outs.pop()                              # the shared list is a back-reference


# ---------------------------------------------------------------------------
# The oracles' isolated mode, in process
# ---------------------------------------------------------------------------

def test_oracles_isolated_mode_contains_hooks_and_copy_mode_does_not(tmp_path, monkeypatch):
    """In this (pytest) process the fixture build has already run backward
    passes, so PyTorch refuses autograd in a forked child: the continuation
    row is DECLARED not run, never failed and never run here. The forward
    still runs in the child, and its hook's write stays there."""
    from experiments.oracles import verify
    ns, exp = _fixture_ns(tmp_path, monkeypatch)
    acts = {"calls": 0}

    def hook(mod, inp, out):
        acts["calls"] += 1
    ns["model"].fc2.register_forward_hook(hook)
    before = _observe(ns)
    rows = verify(ns, exp, destination_device="cpu", source_device="cpu", continuation="isolated")
    by = {r.name: r for r in rows}
    assert [r.name for r in rows if r.ok is False] == [] and len(rows) == 12
    assert by["forward output"].mode != "declared" and "forked child" in by["forward output"].detail
    assert by["continuation"].mode == "declared" and "Autograd and Fork" in by["continuation"].detail
    assert by["continuation"].ok is None                  # not run: never a pass
    assert acts["calls"] == 0 and _same(before, _observe(ns))
    ok, note = __import__("experiments.oracles", fromlist=["summarize"]).summarize(rows)
    assert ok and "11 oracles pass" in note and "1 not run (declared): continuation" in note
    verify(ns, exp, destination_device="cpu", source_device="cpu", continuation="copy")
    assert acts["calls"] > 0                          # the control: a deepcopy shares the hook's closure


def test_a_runtime_that_trained_declares_its_continuation_check(api, tmp_path):
    """The handoff validates a FRESH destination, which has never run a
    backward pass: every check with a reference runs, the continuation step
    included (eleven; the forward output is declared, see
    `_declared_forward_only`). A runtime that trains afterwards and is
    re-checked (a recovery check after user work) cannot run the step in a
    forked child; that row is declared too, with its reason, and the other
    ten still run. Nothing is run in the kernel, and the check measures it."""
    src = _source(api)
    res = Controller(api, Journal(tmp_path / "j.sqlite"), tmp_path / "blobs").migrate("trained-after", src, "cpu")
    assert res["phase"] == "DONE", res
    _declared_forward_only(res)
    fresh = api.witness(res["dest"], ctl.recovery_check_program())
    assert (fresh["passed"], fresh["total"], [d["name"] for d in fresh["declared"]]) == (11, 11, ["forward output"])
    api.witness(res["dest"], TRAIN)
    snap = api.witness(res["dest"], SNAPSHOT)
    v = api.witness(res["dest"], ctl.recovery_check_program())
    # Ten run; `parameters` and `scheduler` now fail against the cut's
    # expectation, correctly: the user's training moved them after the cut.
    assert (v["total"], [d["name"] for d in v["declared"]]) == (10, ["forward output", "continuation"]), v
    assert v["summary"].startswith("10 checks run, ") and v["summary"].endswith(
        " 2 declared not run (forward output, continuation)")
    assert "Autograd and Fork" in v["declared"][1]["detail"]
    assert v["side_effects"]["checked"] and v["side_effects"]["changed"] == []
    assert api.witness(res["dest"], SNAPSHOT) == snap


def test_run_isolated_fails_closed():
    import os
    import time as _t
    from experiments.oracles import run_isolated
    assert run_isolated(lambda: {"x": 1}) == {"ok": True, "value": {"x": 1}}
    assert "ValueError: boom" in run_isolated(lambda: (_ for _ in ()).throw(ValueError("boom")))["error"]
    assert "died" in run_isolated(lambda: os._exit(3))["error"]
    t0 = _t.monotonic()
    assert "timed out" in run_isolated(lambda: _t.sleep(30), timeout=0.5)["error"]
    assert _t.monotonic() - t0 < 10


def test_a_failed_isolated_run_fails_the_model_rows(tmp_path, monkeypatch):
    import experiments.oracles as oracles
    ns, exp = _fixture_ns(tmp_path, monkeypatch)
    monkeypatch.setattr(oracles, "run_isolated", lambda fn, timeout=0: {"ok": False, "error": "child died"})
    rows = oracles.verify(ns, exp, destination_device="cpu", source_device="cpu", continuation="isolated")
    failed = {r.name: r.detail for r in rows if not r.ok}
    assert set(failed) == {"forward output", "continuation"} and all("child died" in d for d in failed.values())
    assert [r.name for r in rows].index("forward output") == [r.name for r in rows].index("destination device") + 1
    ok, note = oracles.summarize(rows)
    assert not ok


# ---------------------------------------------------------------------------
# Minor: per-migration temporary files
# ---------------------------------------------------------------------------

def test_capsule_files_are_per_migration():
    cap = ctl._capture_src()
    assert "/tmp/session.pkl" not in cap and "mkstemp" in cap
    assert ctl.blob_path("m1", "a" * 64) != ctl.blob_path("m2", "a" * 64) != ctl.blob_path("m1", "b" * 64)
    assert ctl.blob_path("e11/../x", "c" * 64).startswith("/tmp/handoff-e11_.._x-")
    assert ctl.DEFAULT_BLOB_PATH in ctl.upload_chunk_program("", True)        # the default-path contract
    assert ctl.blob_path("m1", "a" * 64) in ctl.upload_chunk_program("", True, path=ctl.blob_path("m1", "a" * 64))


def test_two_migrations_on_one_host_at_once(tmp_path):
    """Every local kernel shares one /tmp. Two switches at once used to share
    `/tmp/session.pkl` and `/tmp/handoff.blob`; now each has its own."""
    import threading
    from test_handoff_admission import SerialApi
    api = SerialApi(workdir=tmp_path / "kernels")
    try:
        srcs = [_source(api, name=f"par{i}") for i in range(2)]
        snaps = [api.witness(s, SNAPSHOT) for s in srcs]
        out = {}

        def go(i):
            out[i] = Controller(api, Journal(tmp_path / f"j{i}.sqlite"), tmp_path / f"b{i}").migrate(
                f"par{i}", srcs[i], "cpu")
        ts = [threading.Thread(target=go, args=(i,)) for i in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert [out[i]["phase"] for i in range(2)] == ["DONE", "DONE"], out
        assert [api.witness(out[i]["dest"], SNAPSHOT) for i in range(2)] == snaps
    finally:
        api.close()
