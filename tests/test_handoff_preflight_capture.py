"""The controller's capture leaves the source as it found it, and the
create-response window is recoverable.

These run the real kernel programs in real child interpreters through
`experiments/localapi.py`, the same test double the E11/E12 dry runs use. Each
check is also run against a deliberately broken variant, so a test that passes
here says the evidence can tell the difference, not merely that it was green.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from handoff import controller as ctl  # noqa: E402
from handoff.controller import (  # noqa: E402
    DEST_CREATED, MARK, Controller, ControllerKilled, CrashPoint, Journal, _wrap, seed_program,
)

localapi = pytest.importorskip("localapi")

POISON = _wrap(f'''
import sys, json
_fh = open("poison.txt", "w+")
_fh.write("x"); _fh.flush()
sys.modules["__main__"].__dict__["poison_fh"] = _fh
print({MARK!r} + json.dumps({{"ok": True}}))
''')

PROBE = _wrap(f'''
import sys, json
_g = sys.modules["__main__"].__dict__
_fh = _g.get("poison_fh")
print({MARK!r} + json.dumps({{"bound": _fh is not None, "open": bool(_fh is not None and not _fh.closed),
                              "id": id(_fh) if _fh is not None else None}}))
''')


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    a = localapi.LocalApi(workdir=tmp_path_factory.mktemp("kernels"))
    yield a
    a.close()


@pytest.fixture
def seeded_source(api):
    pid = api.create_project("clusy-exp-handoff-src-test", "cpu")
    api.witness(pid, seed_program(seed=7, files=2, corpus_bytes=2048))
    api.witness(pid, POISON)
    yield pid
    api.delete_project(pid)


def test_capture_excludes_inside_the_stash_and_the_source_keeps_the_name(api, seeded_source):
    before = api.witness(seeded_source, PROBE)
    cap = api.witness(seeded_source, ctl.capture_program(exclude=["poison_fh"]))
    after = api.witness(seeded_source, PROBE)
    assert cap["excluded"] == ["poison_fh"]
    assert cap["excluded_still_present"] == ["poison_fh"]
    assert cap["source_names_equal"] and cap["source_bindings_identical"]
    assert cap["source_names_before"] == cap["source_names_after"]
    assert "poison_fh" not in cap["dumped_names"]           # the capsule does not carry it
    assert "model" in cap["dumped_names"]
    # And from outside the capture: the same open handle, still bound.
    assert after == before and after["bound"] and after["open"]


def test_capture_evidence_detects_a_capture_that_mutates_the_source(api, seeded_source):
    # The first version: pop the rejected names from the live namespace,
    # outside any stash/finally. The evidence must say so.
    marker = "_before = {k: id(v) for k, v in _g.items()}\n"
    src = ctl._capture_src(exclude=["poison_fh"])
    assert marker in src
    broken = _wrap(src.replace(marker, marker + "for _k in _EXCLUDE: _g.pop(_k, None)\n"))
    cap = api.witness(seeded_source, broken)
    assert cap["excluded_still_present"] == []
    assert not cap["source_names_equal"] and not cap["source_bindings_identical"]
    assert api.witness(seeded_source, PROBE)["bound"] is False


def test_harness_programs_leave_nothing_in_main(api, seeded_source):
    listing = _wrap(f'import sys, json\nprint({MARK!r} + json.dumps(sorted(sys.modules["__main__"].__dict__)))')
    names0 = set(api.witness(seeded_source, listing))
    api.witness(seeded_source, ctl.touch_program())
    api.witness(seeded_source, ctl.describe_program())
    desc = {"has_cuda": False, "python_minor": "%d.%d" % sys.version_info[:2],
            "dill_version": __import__("dill").__version__, "packages": {}}
    pf = api.witness(seeded_source, ctl.preflight_program(desc))
    assert pf["rejected"] == ["poison_fh"]
    # The programs this round added: the barrier, the state fingerprint, the
    # non-mutating validation and recovery check, and the capture with its
    # cut fingerprint and RNG envelope.
    assert api.witness(seeded_source, ctl.barrier_program()) == {"barrier": True}
    api.witness(seeded_source, ctl.fingerprint_program())
    for prog in (ctl.verify_program(), ctl.recovery_check_program()):
        v = api.witness(seeded_source, prog)
        # The seed trained this kernel (backward passes), so PyTorch refuses
        # autograd in a forked child: the continuation row is declared, the
        # other eleven run in the child.
        assert v["continuation"] == "isolated" and v["isolated"] and v["total"] == 11
        assert [d["name"] for d in v["declared"]] == ["continuation"]
    api.witness(seeded_source, ctl.capture_program(exclude=["poison_fh"]))
    assert set(api.witness(seeded_source, listing)) == names0


def test_capture_and_validation_leave_the_source_state_and_rng_unchanged(api, seeded_source):
    """The cut is taken inside the capture, and the dump may run arbitrary
    reducers; the capture puts every RNG stream back to the cut afterwards.
    Validation runs in a forked child. Neither may move the state the fingerprint
    describes."""
    fp = lambda: {k: v for k, v in api.witness(seeded_source, ctl.fingerprint_program()).items()  # noqa: E731
                  if k in ctl.BOUNDARY_FIELDS + ("rng_cuda", "devices")}
    before = fp()
    cap = api.witness(seeded_source, ctl.capture_program(exclude=["poison_fh"]))
    # The cut describes what is DUMPED, so the excluded handle is not in it;
    # everything else is the live state.
    cut = {k: v for k, v in cap["fingerprint"].items() if k in before}
    assert "poison_fh" in before["names"] and "poison_fh" not in cut["names"]
    assert "poison_fh" in before["objects"] and "poison_fh" not in cut["objects"]
    assert cut == {**before, "names": {k: v for k, v in before["names"].items() if k != "poison_fh"},
                   "objects": {k: v for k, v in before["objects"].items() if k != "poison_fh"}}
    # And the capture's own second fingerprint, after the dump, equals the cut.
    assert ctl.compare_fingerprints(cap["fingerprint"], cap["fingerprint_after"])["equal"]
    assert fp() == before
    api.witness(seeded_source, ctl.verify_program())
    assert fp() == before
    assert cap["rng_carried"] == {"numpy": True, "torch_cpu": True, "cuda": False}


def test_program_sizes_stay_under_the_execute_cap():
    """The execute route caps `code` at 200,000 characters. The programs the
    handoff ships must stay under it with room for the adapter to grow."""
    desc = {"has_cuda": False, "python_minor": "3.11", "dill_version": "0", "packages": {}}
    sizes = {"seed": len(seed_program(1, 1, 1)), "describe": len(ctl.describe_program()),
             "preflight": len(ctl.preflight_program(desc)), "capture": len(ctl.capture_program()),
             "restore": len(ctl.restore_program("0" * 64)), "verify": len(ctl.verify_program()),
             "fingerprint": len(ctl.fingerprint_program())}
    assert max(sizes.values()) < 180_000, sizes


def test_controller_excludes_poison_and_completes(api, seeded_source, tmp_path):
    j = Journal(tmp_path / "j.sqlite")
    c = Controller(api, j, tmp_path / "blobs")
    res = c.migrate("t-poison", seeded_source, "cpu")
    try:
        assert res["phase"] == "DONE", res
        assert res["preflight"]["excluded"] == ["poison_fh"]
        assert res["capture"]["excluded_still_present"] == ["poison_fh"]
        row = j.get("t-poison")
        assert row["preflight_sha"] and Path(row["preflight_path"]).is_file()
        report = json.loads(Path(row["preflight_path"]).read_text())
        assert report["policy"]["decision"] == "proceed" and report["policy"]["excluded"] == ["poison_fh"]
        assert report["destination"]["packages"], "the destination described itself"
    finally:
        api.delete_project(res["dest"])


def test_strict_unknown_blocks_at_preflight(api, seeded_source, tmp_path):
    j = Journal(tmp_path / "j.sqlite")
    res = Controller(api, j, tmp_path / "blobs", strict_unknown=True).migrate("t-strict", seeded_source, "cpu")
    assert res["phase"] == "ABORTED" and res["failed_at"] == "PREFLIGHT"
    assert res["reason"].startswith("preflight_unknown:")
    assert j.get("t-strict")["capsule_path"] is None                      # nothing captured
    assert api.witness(seeded_source, PROBE)["bound"]                     # source untouched


def test_crash_in_the_create_response_window_is_reclaimed(api, tmp_path):
    src = api.create_project("clusy-exp-handoff-src-t-window", "cpu")
    try:
        api.witness(src, seed_program(seed=7, files=2, corpus_bytes=2048))
        j = Journal(tmp_path / "j.sqlite")

        def kill():
            raise ControllerKilled()
        with pytest.raises(ControllerKilled):
            Controller(api, Journal(j.path), tmp_path / "blobs", crash=CrashPoint(DEST_CREATED, after_journal=False),
                       kill=kill).migrate("t-window", src, "cpu")
        row = j.get("t-window")
        named = lambda: [p["id"] for p in api.list_projects() if p["name"] == "clusy-exp-handoff-t-window"]  # noqa: E731
        assert row["phase"] == "REQUESTED" and row["pending_dest_pid"] is None
        first = named()
        assert len(first) == 1                                           # created, and the journal does not know
        c = Controller(api, j, tmp_path / "blobs")
        res = c.migrate("t-window", src, "cpu", takeover=True)
        assert res["phase"] == "DONE" and res["dest"] != first[0]         # recovery made a second destination
        assert c.reclaim("t-window")["reclaimed"] == first                 # and reclaim finds the first by name
        assert named() == [res["dest"]]
    finally:
        for p in [p["id"] for p in api.list_projects() if "t-window" in p["name"]]:
            api.delete_project(p)


# -- the preflight never damages the source it judges ---------------------------

def test_an_aborted_switch_leaves_a_nested_open_log_byte_identical(api, tmp_path):
    """The reviewer's end-to-end case: a session object holding an open 'w+'
    log, a switch that aborts after capture. The first validator truncated
    the log during PREFLIGHT; the bindings check could not see it, because
    the object and its handle are still bound. So the check is on bytes."""
    import e11_transactional as e11
    src = api.create_project("clusy-exp-handoff-src-t-log", "cpu")
    try:
        api.witness(src, seed_program(seed=7, files=2, corpus_bytes=2048))
        api.witness(src, e11.nested_log_program())
        before = api.witness(src, e11.log_probe_program())
        res = Controller(api, Journal(tmp_path / "j.sqlite"), tmp_path / "blobs").migrate(
            "t-log", src, "cpu", fault="bad_capsule")
        after = api.witness(src, e11.log_probe_program())
        assert res["phase"] == "ABORTED" and res["failed_at"] == "DEST_RESTORED", res
        assert "run_log" in res["preflight"]["excluded"]
        assert res["capture"]["excluded_still_present"] == ["run_log"]
        assert before["bytes"] > 0 and (after["bytes"], after["sha256"]) == (before["bytes"], before["sha256"])
        assert after["handle_open"] and after["handle_tell"] == before["handle_tell"]
    finally:
        for p in [p["id"] for p in api.list_projects() if "t-log" in p["name"]]:
            api.delete_project(p)


def test_strict_mode_blocks_when_process_effects_were_never_measured(api, tmp_path):
    # No seed program, so no `__handoff_baseline__`: the effects are unknown,
    # and a strict gate must not read "unknown" as "none".
    src = api.create_project("clusy-exp-handoff-src-t-nobase", "cpu")
    try:
        api.witness(src, _wrap(f'import sys, json\nsys.modules["__main__"].__dict__["x"] = 1\n'
                               f'print({MARK!r} + json.dumps({{}}))'))
        j = Journal(tmp_path / "j.sqlite")
        res = Controller(api, j, tmp_path / "blobs", strict_unknown=True).migrate("t-nobase", src, "cpu")
        assert res["phase"] == "ABORTED" and res["failed_at"] == "PREFLIGHT", res
        assert res["reason"] == "preflight_process_effects_unavailable"
        assert res["preflight"]["effects_available"] is False
    finally:
        for p in [p["id"] for p in api.list_projects() if "t-nobase" in p["name"]]:
            api.delete_project(p)


# -- the E11 harness survives a trial that raises --------------------------------

def test_an_e11_trial_that_raises_is_recorded_and_its_projects_deleted(api, tmp_path, monkeypatch):
    import e11_transactional as e11

    def boom(self, mig, source_pid, dest_profile, **kw):
        # A provider failure after the destination exists and before anyone
        # was told its id: only the name can find it.
        self.api.create_project(f"clusy-exp-handoff-{mig}", dest_profile)
        raise RuntimeError("provider went away")
    monkeypatch.setattr(e11.Controller, "migrate", boom)
    row = e11.run_one(api, Journal(tmp_path / "j.sqlite"), tmp_path / "blobs", None, lambda *a: None)
    assert row["phase"] == "HARNESS_ERROR" and row["as_expected"] is False
    assert row["failed_checks"] == ["harness_error"] and "provider went away" in row["detail"]
    assert [p for p in api.list_projects() if row["migration"] in p["name"]] == []


class _SkewedDestinationApi(localapi.LocalApi):
    """A LocalApi whose destination describes itself with a different Python
    minor or dill version. Only the destination's self-description changes;
    the controller, the manifest and every kernel program run as they do live.
    The local kernels all share one interpreter, so this is the only way to
    make the version gates see a mismatch without a second toolchain."""

    def __init__(self, *, skew: dict, **kw):
        super().__init__(**kw)
        self.skew = skew

    def witness(self, pid, code, timeout_ms=600_000):
        out = super().witness(pid, code, timeout_ms)
        if code == ctl.describe_program():
            out = {**out, **self.skew}
        return out


@pytest.mark.parametrize("gate,skew", [
    ("python_minor", {"python_minor": "3.10"}),
    ("dill_exact", {"dill_version": "0.0.0-skewed"}),
])
def test_version_gates_refuse_at_preflight_and_leave_the_source(tmp_path, gate, skew):
    # The review's preflight point: the destination's own description, not the
    # source's, decides the gates. A skewed description must refuse the switch
    # before anything is captured, with the gate named in the reason, and the
    # source must still hold its state.
    api = _SkewedDestinationApi(skew=skew, workdir=tmp_path / "kernels")
    try:
        src = api.create_project("clusy-exp-handoff-src-gate", "cpu")
        api.witness(src, seed_program(seed=7, files=2, corpus_bytes=2048))
        api.witness(src, POISON)
        j = Journal(tmp_path / "j.sqlite")
        res = Controller(api, j, tmp_path / "blobs").migrate(f"t-gate-{gate}", src, "cpu")
        assert res["phase"] == "ABORTED" and res["failed_at"] == "PREFLIGHT", res
        assert res["reason"] == f"preflight_gate_failed:{gate}", res
        assert j.get(f"t-gate-{gate}")["capsule_path"] is None           # nothing captured
        assert api.witness(src, PROBE)["bound"]                          # source untouched
        report = json.loads(Path(j.get(f"t-gate-{gate}")["preflight_path"]).read_text())
        assert report["policy"]["decision"] != "proceed"
        # The destination project was released by the abort path.
        assert all(p["name"] != f"clusy-exp-handoff-t-gate-{gate}" for p in api.list_projects())
    finally:
        api.close()


def test_unskewed_description_passes_the_same_gates(tmp_path):
    # The control for the test above: the same wrapper with no skew proceeds.
    api = _SkewedDestinationApi(skew={}, workdir=tmp_path / "kernels")
    try:
        src = api.create_project("clusy-exp-handoff-src-gate", "cpu")
        api.witness(src, seed_program(seed=7, files=2, corpus_bytes=2048))
        res = Controller(api, Journal(tmp_path / "j.sqlite"), tmp_path / "blobs").migrate("t-gate-none", src, "cpu")
        assert res["phase"] == "DONE", res
    finally:
        api.close()
