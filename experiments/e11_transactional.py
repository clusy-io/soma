"""E11 — validate the destination before destroying the source.

Runs of the transactional protocol against real E2B sandboxes: one clean, three
with a fault injected at the destination (a missing package, an out-of-memory
during reconstruction, a corrupted capsule), and two that exercise the
PREFLIGHT the controller now runs before it captures anything:

  preflight_missing_distribution
      the source imports a distribution the destination does not have (a
      local dummy distribution: a package directory plus a `*.dist-info`
      in a temp dir on the source's sys.path) and binds an object from it.
      Expected: ABORTED at PREFLIGHT with the typed reason
      `preflight_missing_distribution:clusy-e11-probe-dist`, nothing
      captured, the destination deleted, and the source still accepting work
      with its counter intact and its bindings unchanged.
  poison_excluded
      the source binds an open file handle, which pickles and then poisons
      the all-or-nothing load. Expected: the preflight reports it rejected,
      the capture leaves it out while the source still holds it
      (`excluded_still_present`), and the migration completes with every
      oracle passing on the destination, which does not have the handle.
  nested_open_log
      the source binds an instance of a session class that keeps its log
      file OPEN ('w+', lines written), and the switch is then driven to an
      abort (the capsule is corrupted in transit). dill rebuilds a file handle
      by re-opening its path with its mode, so any load of this object in the
      source truncates the log; the first preflight validator did exactly
      that. Expected: the preflight rejects `run_log` WITHOUT loading it, the
      capture excludes it, the switch aborts at DEST_RESTORED, and afterwards
      the source's log file has the same bytes (size and sha256) and its
      handle is still open at the same position.

For every faulted run the assertion that matters is not that the switch failed
but that the SOURCE IS STILL USABLE afterwards: it accepts an execution and its
state is intact. That is the property the platform's serialize -> destroy ->
reboot -> replay order cannot offer, because the source is gone before the
destination is exercised.

Every trial also records the ADMISSION GATE and the COMMIT BOUNDARY. A write is
routed through the controller before the migration (acknowledged by the
source) and another after it; the checks are that the gate closed during any
migration that reached the capture, that it is open again afterwards, that the
later write ran on the runtime that is authoritative, and that both writes are
there. For a completed run the state fingerprint taken at the cut must equal
the destination's after the restore and again immediately before COMMITTED;
all three fingerprints and both verdicts are in the record. For every run that
reaches the capture, the capture's own second fingerprint (after the dump) must
equal the cut: the source's STATE was left as found, not only its bindings.

What the admission columns do NOT show: the writes here are sequential, one
before the switch and one after, so no write ever meets a closed gate and the
drain never has anything to wait for (refused 0, drain waited 0 in every run).
They show that the gate closes and reopens on the right runtime. Refusal of a
write that arrives while the gate is closed, and the drain of one admitted
before the close, are shown by E12 (a write while the controller is down) and
by tests/test_handoff_admission.py (a concurrent writer, chained hops, the
drain and its control).

Also two concurrent PATCH requests on the platform API, to confirm the
platform's own switch serialises them.

    python experiments/e11_transactional.py --cohort e11-rerun --outdir /tmp/soma-rerun/e11   # live
    python experiments/e11_transactional.py --local --outdir /tmp/e11                         # dry run

`--local` runs everything against `experiments/localapi.py` (all profiles are
the local CPU) and writes to `--outdir`, never to `results/`. `--sabotage`
(local only) breaks one thing on purpose so the expectation checks can be seen
to FAIL: `stale_inventory` makes the destination claim the dummy distribution
is installed; `unsafe_exclusion` makes the capture delete excluded names from
the live source the way the first version did; `unsafe_validator` ships the
manifest with its pre-load scan switched off, so the preflight loads every
pickle in the source the way the first validator did; `inplace_validation`
validates with the in-place continuation step the first controller used, which
still passes every oracle and must be caught by the commit-boundary check.

A trial that raises (a provider error, a missing witness) is recorded as a
`HARNESS_ERROR` row that fails its expectation, its projects are deleted, and
the cohort continues with the next fault.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import runmeta  # noqa: E402
from handoff import controller as _ctl  # noqa: E402
from handoff.controller import (  # noqa: E402
    MARK, Api, Controller, Journal, _wrap, seed_program, touch_program,
)

FAULTS = [None, "missing_package", "insufficient_memory", "bad_capsule",
          "preflight_missing_distribution", "poison_excluded", "nested_open_log"]
#: Faults injected by the controller at the destination's restore.
RESTORE_FAULTS = {"missing_package", "insufficient_memory", "bad_capsule"}
#: The controller fault each E11 fault runs with (a source-side setup may
#: also need the switch driven to an abort).
CONTROLLER_FAULT = {**{f: f for f in RESTORE_FAULTS}, "nested_open_log": "bad_capsule"}
PROBE_DIST, PROBE_MODULE = "clusy-e11-probe-dist", "clusy_e11_probe_dist"


def dummy_distribution_program() -> str:
    """Create, import and use a distribution only the SOURCE has.

    A real distribution as `importlib.metadata` sees one: a package directory
    and a `*.dist-info` with METADATA and top_level.txt, in a fresh temp dir
    put on the source's sys.path. The object bound in `__main__` belongs to a
    class of that package, so dill records it by reference and a destination
    without the package cannot load it. The in-process round trip cannot see
    that (the module is importable here); the package section must.
    """
    return _wrap(f'''
import sys, json
def _setup():
    import os, tempfile, importlib
    root = tempfile.mkdtemp(prefix="clusy-e11-dist-")
    pkg = os.path.join(root, {PROBE_MODULE!r})
    os.makedirs(pkg)
    with open(os.path.join(pkg, "__init__.py"), "w") as f:
        f.write("class Marker:\\n    def __init__(self, n):\\n        self.n = n\\n    def twice(self):\\n        return 2 * self.n\\n")
    info = os.path.join(root, {PROBE_MODULE!r} + "-0.1.0.dist-info")
    os.makedirs(info)
    with open(os.path.join(info, "METADATA"), "w") as f:
        f.write("Metadata-Version: 2.1\\nName: {PROBE_DIST}\\nVersion: 0.1.0\\n")
    with open(os.path.join(info, "top_level.txt"), "w") as f:
        f.write({PROBE_MODULE!r} + "\\n")
    sys.path.insert(0, root)
    importlib.invalidate_caches()
    mod = importlib.import_module({PROBE_MODULE!r})
    sys.modules["__main__"].__dict__["probe_marker"] = mod.Marker(21)
    return {{"root": root, "distribution": {PROBE_DIST!r}, "bound": "probe_marker"}}
print({MARK!r} + json.dumps(_setup()))
''')


def poison_program() -> str:
    """Bind an open file handle in the source's namespace."""
    return _wrap(f'''
import sys, json, tempfile, os
_p = os.path.join(tempfile.gettempdir(), "clusy_e11_poison_%d.txt" % os.getpid())
_fh = open(_p, "w+")
_fh.write("poison"); _fh.flush()
sys.modules["__main__"].__dict__["poison_fh"] = _fh
print({MARK!r} + json.dumps({{"bound": "poison_fh", "path": _p}}))
''')


#: Defined in the session's REAL `__main__` (so dill copies the class by
#: value, as it would a user's), holding its log open for the whole run.
RUN_LOG_SRC = '''
class RunLog:
    """Keeps its log file open for the whole run, as training loops do."""
    def __init__(self, path):
        self.path = path
        self.fh = open(path, "w+")
    def log(self, line):
        self.fh.write(line + "\\n")
        self.fh.flush()
'''


def nested_log_program() -> str:
    """Bind `run_log`, a session object with an open 'w+' log inside it."""
    return _wrap(f'''
import sys, json, os
_g = sys.modules["__main__"].__dict__
exec(compile({RUN_LOG_SRC!r}, "<session>", "exec"), _g)
_p = os.path.abspath("clusy_e11_train.log")
_g["run_log"] = _g["RunLog"](_p)
_g["run_log"].log("epoch 1 loss 0.9")
_g["run_log"].log("epoch 2 loss 0.7")
print({MARK!r} + json.dumps({{"bound": "run_log", "path": _p, "bytes": os.path.getsize(_p)}}))
''')


def log_probe_program(name: str = "run_log") -> str:
    """The on-disk bytes of `name.fh`'s file and the handle's own state."""
    return _wrap(f'''
import sys, json, os, hashlib
_o = sys.modules["__main__"].__dict__.get({name!r})
_fh = getattr(_o, "fh", None)
_p = getattr(_fh, "name", None)
_out = {{"bound": _o is not None, "path": _p}}
if isinstance(_p, str) and os.path.exists(_p):
    with open(_p, "rb") as _f:
        _b = _f.read()
    _out.update(bytes=len(_b), sha256=hashlib.sha256(_b).hexdigest())
if _fh is not None:
    _out["handle_open"] = not _fh.closed
    _out["handle_tell"] = None if _fh.closed else _fh.tell()
print({MARK!r} + json.dumps(_out))
''')


def routed_write_program() -> str:
    """A user write sent through the controller's route: increments a counter
    in the session namespace and acknowledges the new value."""
    return _wrap(f'''
import sys, json
_g = sys.modules["__main__"].__dict__
_g["e11_routed_writes"] = _g.get("e11_routed_writes", 0) + 1
print({MARK!r} + json.dumps({{"counter": _g["e11_routed_writes"]}}))
''')


def routed_count_program() -> str:
    return _wrap(f'''
import sys, json
print({MARK!r} + json.dumps({{"counter": sys.modules["__main__"].__dict__.get("e11_routed_writes")}}))
''')


SOURCE_SETUPS = {"preflight_missing_distribution": dummy_distribution_program,
                 "poison_excluded": poison_program,
                 "nested_open_log": nested_log_program}


def bindings_program(names: list[str] | None = None) -> str:
    """The source's bindings (name -> object id) and, for `names`, whether
    each is bound. Ids are stable while the objects live, so equal snapshots
    before and after an aborted switch mean nothing was rebound or removed."""
    return _wrap(f'''
import sys, json
_g = sys.modules["__main__"].__dict__
print({MARK!r} + json.dumps({{"ids": {{k: id(v) for k, v in _g.items() if not k.startswith("_i")}},
                             "has": {{n: n in _g for n in {list(names or [])!r}}}}}))
''')


def expected(fault: str | None) -> dict:
    if fault is None:
        return {"phase": "DONE"}
    if fault in RESTORE_FAULTS:
        reason = {"missing_package": "RESTORE_MISSING_MODULE", "insufficient_memory": "insufficient_memory",
                  "bad_capsule": "checkpoint_transfer_failed"}[fault]
        return {"phase": "ABORTED", "failed_at": "DEST_RESTORED", "reason": reason}
    if fault == "preflight_missing_distribution":
        return {"phase": "ABORTED", "failed_at": "PREFLIGHT",
                "reason": f"preflight_missing_distribution:{PROBE_DIST}"}
    if fault == "poison_excluded":
        return {"phase": "DONE", "excluded": ["poison_fh"]}
    if fault == "nested_open_log":
        return {"phase": "ABORTED", "failed_at": "DEST_RESTORED", "reason": "checkpoint_transfer_failed",
                "excluded": ["run_log"], "source_file_intact": True}
    raise ValueError(fault)


def judge(r: dict) -> tuple[bool, list[str]]:
    """Every expectation for this fault, as named checks. A run passes when
    all hold; the failed names are recorded so a FAIL says why."""
    e, fails = r["expect"], []
    def need(name, ok):
        if not ok:
            fails.append(name)
    need("phase", r["phase"] == e["phase"])
    if e["phase"] == "ABORTED":
        need("failed_at", r["failed_at"] == e["failed_at"])
        need("reason", r["reason"] == e["reason"])
        need("source_accepts_after", r["source_accepts_after"] is True)
        need("source_counter_intact", r["source_counter_intact"] is True)
        need("source_bindings_unchanged", r["source_bindings_unchanged"] is True)
        need("authority_is_source", r["authoritative_is"] == "source")
        need("destination_deleted", r["dest_alive_after"] is False)
        if e["failed_at"] == "PREFLIGHT":
            need("nothing_captured", r["capsule_bytes"] is None and not r["capsule_file_exists"])
            need("preflight_report_journaled", bool(r["preflight_report_journaled"]))
    else:
        need("dest_accepts_after", r["dest_accepts_after"] is True)
        need("authority_is_dest", r["authoritative_is"] == "dest")
        need("oracles_all_pass", bool(r.get("oracles")) and r["oracles"].split("/")[0] == r["oracles"].split("/")[1])
    # The admission gate: closed for every migration that reached the capture
    # cut, open again afterwards, and the routed writes on the right runtime.
    adm = r.get("admission") or {}
    reached_cut = e["phase"] == "DONE" or e.get("failed_at") not in (None, "PREFLIGHT")
    need("admission_closed_during_migration", adm.get("closed_during_migration") is reached_cut)
    need("admission_reopened", adm.get("state") == "open")
    need("routed_write_before_acknowledged", (r.get("routed_before") or {}).get("runtime") == "source")
    need("routed_write_after_on_authority", (r.get("routed_after") or {}).get("runtime")
         == ("dest" if e["phase"] == "DONE" else "source"))
    need("acknowledged_writes_preserved", r.get("routed_count_after") == 2)
    if e["phase"] == "DONE":
        cb = (r.get("commit_boundary") or {}).get("verdicts") or {}
        need("commit_boundary_restored_equal", (cb.get("restored") or {}).get("equal") is True)
        need("commit_boundary_pre_commit_equal", (cb.get("pre_commit") or {}).get("equal") is True)
    cap = r.get("capture") or {}
    if e["phase"] == "DONE" or e.get("failed_at") not in (None, "PREFLIGHT"):
        # Every run expected to reach the capture: the capture left the
        # source as it found it, bindings AND state.
        need("source_names_equal", cap.get("source_names_equal") is True)
        need("source_bindings_identical", cap.get("source_bindings_identical") is True)
        need("source_state_unchanged", cap.get("source_state_unchanged") is True)
    if e.get("excluded"):
        pf = r.get("preflight") or {}
        need("preflight_rejected", set(e["excluded"]) <= {n["name"] for n in pf.get("rejected", [])})
        need("capture_excluded", set(e["excluded"]) <= set(cap.get("excluded") or []))
        need("excluded_still_present_on_source", set(e["excluded"]) <= set(cap.get("excluded_still_present") or []))
        if e["phase"] == "DONE":
            need("destination_lacks_excluded", not any((r.get("dest_has") or {}).get(n) for n in e["excluded"]))
    if e.get("source_file_intact"):
        # The bytes, not the binding: the binding survives a truncation, the
        # user's data does not.
        lb, la = r.get("source_log_before") or {}, r.get("source_log_after") or {}
        need("source_file_intact", bool(lb.get("sha256")) and lb.get("bytes", 0) > 0
             and (la.get("bytes"), la.get("sha256")) == (lb.get("bytes"), lb.get("sha256")))
        need("source_handle_intact", la.get("handle_open") is True and la.get("handle_tell") == lb.get("handle_tell"))
    return not fails, fails


def _cleanup(api, mig: str, pids: set, log, search: bool) -> None:
    """Delete this trial's projects. On the error path the destination may
    exist without anyone having been told its id, so it is also looked up by
    the name the controller gives it. Never raises: a cleanup failure is
    logged and must not end the cohort."""
    done = set()
    for pid in pids - {None}:
        try:
            api.delete_project(pid); done.add(pid)
        except Exception as exc:  # noqa: BLE001
            log(f"[{mig}] cleanup: delete {pid} failed: {type(exc).__name__}: {exc}")
    if search:
        try:
            for it in api.list_projects():
                if str(it.get("name", "")) in (f"clusy-exp-handoff-{mig}", f"clusy-exp-handoff-src-{mig}") \
                        and it["id"] not in done:
                    api.delete_project(it["id"])
                    log(f"[{mig}] cleanup: deleted {it['id']} found by name")
        except Exception as exc:  # noqa: BLE001
            log(f"[{mig}] cleanup: listing projects failed: {type(exc).__name__}: {exc}")


def run_one(api, journal: Journal, blobs: Path, fault: str | None, log, strict_unknown: bool = False) -> dict:
    """One trial. Whatever happens inside, the trial's projects are deleted
    and a record comes back: a raised error becomes a `HARNESS_ERROR` row
    that fails its expectation instead of ending the cohort."""
    mig = f"e11-{fault or 'clean'}-{uuid.uuid4().hex[:6]}"
    t_trial = time.perf_counter()
    state: dict = {"src": None, "res": {}}
    ok = False
    try:
        out = _run_one(api, journal, blobs, fault, log, strict_unknown, mig, state)
        ok = True
        return out
    except Exception as exc:  # noqa: BLE001
        log(f"[{mig}] harness error: {type(exc).__name__}: {exc}")
        row = journal.get(mig) or {}
        return {"migration": mig, "fault": fault, "expect": expected(fault), "phase": "HARNESS_ERROR",
                "failed_at": None, "reason": type(exc).__name__, "detail": str(exc)[:500],
                "wall_s": round(time.perf_counter() - t_trial, 1), "timings": None, "oracles": None,
                "journal_phase": row.get("phase"), "source_accepts_after": None, "source_counter_after": None,
                "source_fenced_by": None, "dest_accepts_after": None, "dest_alive_after": None,
                "authoritative_is": "dest" if row and row.get("authoritative") == row.get("dest_pid") else "source",
                "preflight": (state["res"] or {}).get("preflight"), "capture": None,
                "admission": None, "commit_boundary": None, "routed_before": None, "routed_after": None,
                "routed_count_after": None,
                "events": journal.events(mig), "as_expected": False, "failed_checks": ["harness_error"]}
    finally:
        _cleanup(api, mig, {state["src"], (state["res"] or {}).get("dest")}, log, search=not ok)


def _run_one(api, journal: Journal, blobs: Path, fault: str | None, log, strict_unknown: bool,
             mig: str, state: dict) -> dict:
    src = state["src"] = api.create_project(f"clusy-exp-handoff-src-{mig}", "cpu")
    log(f"[{mig}] source {src}: seeding")
    seeded = api.witness(src, seed_program(seed=20260925, files=4, corpus_bytes=16 * 1024))
    setup = api.witness(src, SOURCE_SETUPS[fault]()) if fault in SOURCE_SETUPS else None
    c = Controller(api, journal, blobs, log=log, strict_unknown=strict_unknown)
    # A write routed through the controller before the migration starts (the
    # row exists from the request on): acknowledged by the source, so it must
    # be on whichever runtime is authoritative afterwards. Made before the
    # bindings snapshot, which must only see what the switch itself does.
    journal.create(mig, src, "cpu")
    rb = c.route(mig, routed_write_program())
    routed_before = {"runtime": "source" if rb["runtime"] == src else rb["runtime"], "counter": rb["result"]["counter"]}
    before = api.witness(src, touch_program())          # counter -> 1
    watch = list(expected(fault).get("excluded") or [])
    snap_before = api.witness(src, bindings_program(watch))
    probe_log = bool(expected(fault).get("source_file_intact"))
    log_before = api.witness(src, log_probe_program()) if probe_log else None
    t0 = time.perf_counter()
    res = state["res"] = c.migrate(mig, src, "cpu", fault=CONTROLLER_FAULT.get(fault))
    wall = time.perf_counter() - t0
    row = journal.get(mig)
    # The invariant: after a failed validation, the source still accepts work
    # and its state is intact; after a successful one, the source is fenced and
    # the destination accepts work. The bindings snapshot is taken BEFORE the
    # touch below, which is the only thing expected to change the source.
    snap_after = api.witness(src, bindings_program(watch)) if res["phase"] != "DONE" else None
    log_after = api.witness(src, log_probe_program()) if probe_log and res["phase"] != "DONE" else None
    src_after = c.runtime_accepts(src)
    dest_has = api.witness(res["dest"], bindings_program(watch))["has"] if res.get("dest") and res["phase"] == "DONE" else None
    dest_after = c.runtime_accepts(res["dest"]) if res.get("dest") else None
    alive = {it["id"] for it in api.list_projects()}
    # After the switch: a routed write lands on the authority (the destination
    # after a commit, the source after an abort), and both routed writes are
    # there. Last, so no check above sees its effect.
    ra = c.route(mig, routed_write_program())
    authority = row["dest_pid"] if res["phase"] == "DONE" else src
    routed_after = {"runtime": "dest" if ra["runtime"] == row.get("dest_pid") else
                    ("source" if ra["runtime"] == src else ra["runtime"]), "counter": ra["result"]["counter"]}
    routed_count_after = api.witness(authority, routed_count_program())["counter"]
    capsule_file = blobs / f"{mig}.capsule.tgz"
    out = {
        "migration": mig, "fault": fault, "expect": expected(fault),
        "phase": res["phase"], "failed_at": res.get("failed_at"),
        "reason": res.get("reason"), "detail": res.get("detail"), "wall_s": round(wall, 1),
        "timings": res.get("timings"), "oracles": (res.get("timings") or {}).get("oracles"),
        "seeded_names": seeded["names"], "source_setup": setup,
        "source_counter_before": before.get("counter"),
        "source_accepts_after": src_after.get("accepted"), "source_counter_after": src_after.get("counter"),
        "source_counter_intact": (src_after.get("counter") == before.get("counter", 0) + 1) if src_after.get("accepted") else None,
        "source_bindings_unchanged": (snap_after["ids"] == snap_before["ids"]) if snap_after else None,
        "source_fenced_by": src_after.get("fenced_by"),
        "dest_accepts_after": (dest_after or {}).get("accepted"),
        "dest_alive_after": (row.get("dest_pid") in alive) if row.get("dest_pid") else False,
        "dest_has": dest_has,
        "authoritative": row["authoritative"], "authoritative_is": "dest" if row["authoritative"] == row.get("dest_pid") else "source",
        "capsule_bytes": row.get("capsule_bytes"), "capsule_file_exists": capsule_file.exists(),
        "preflight_report_journaled": row.get("preflight_sha"),
        "preflight": res.get("preflight"),
        "source_log_before": log_before, "source_log_after": log_after,
        # The before/after name lists are in the CAPTURED journal event, which
        # `events` below carries; the record keeps the verdict on them.
        "capture": {k: v for k, v in (res.get("capture") or {}).items()
                    if k not in ("source_names_before", "source_names_after", "dumped_names")} or None,
        "restore": res.get("restore"), "validation": res.get("validation"),
        "routed_before": routed_before, "routed_after": routed_after, "routed_count_after": routed_count_after,
        "admission": c.admission_evidence(mig),
        # All three fingerprints (the cut, after the restore, before COMMITTED)
        # and both verdicts, from the journal.
        "commit_boundary": c._boundary_report(mig),
        "events": journal.events(mig),
    }
    out["as_expected"], out["failed_checks"] = judge(out)
    return out


def concurrent_patch_probe(api: Api, log) -> dict:
    """E9-live: two PATCHes race on one project. Expect one 200 and one typed conflict."""
    pid = api.create_project(f"clusy-exp-e9live-{uuid.uuid4().hex[:6]}", "cpu")
    api.witness(pid, "print('__CLUSY_HANDOFF__' + '{\"warm\": true}')")
    results = {}

    def patch(name, profile):
        req = urllib.request.Request(f"{api.base}/projects/{pid}", method="PATCH",
                                     data=json.dumps({"runtimeProfile": profile}).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {api.key}"})
        t = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                results[name] = {"status": r.status, "s": round(time.perf_counter() - t, 1)}
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            try:
                code = json.loads(body).get("error", {}).get("code")
            except Exception:
                code = None
            results[name] = {"status": e.code, "code": code, "s": round(time.perf_counter() - t, 1)}

    ts = [threading.Thread(target=patch, args=("A", "gpu_t4")), threading.Thread(target=patch, args=("B", "gpu_t4"))]
    for t in ts: t.start()
    for t in ts: t.join()
    log(f"  concurrent PATCH: {results}")
    api.delete_project(pid)
    ok = sorted(r["status"] for r in results.values())
    return {"results": results, "exactly_one_200": ok.count(200) == 1}


def _sabotage(api, kind: str | None):
    """Local-only deliberate breakage, so a check can be seen to fail."""
    if kind == "stale_inventory":
        # The destination LIES about its inventory: it claims the dummy
        # distribution. The preflight then admits the switch, the capsule is
        # captured and uploaded, and the load fails at DEST_RESTORED instead.
        describe = _ctl.describe_program()
        inner = api.witness

        def witness(pid, code, timeout_ms=600_000):
            out = inner(pid, code, timeout_ms)
            if code == describe:
                out["packages"][PROBE_DIST] = "0.1.0"
            return out
        api.witness = witness
    elif kind == "unsafe_exclusion":
        # The first version's capture: pop the rejected names from the live
        # namespace BEFORE the dump, outside any stash/finally.
        marker = '_before = {k: id(v) for k, v in _g.items()}\n'

        def broken(workspace_root="data", exclude=None, classified=None):
            src = _ctl._capture_src(workspace_root, exclude, classified)
            assert marker in src
            return _wrap(src.replace(marker, marker + "for _k in _EXCLUDE: _g.pop(_k, None)\n"))
        _ctl.capture_program = broken
    elif kind == "unsafe_validator":
        # The first validator: no pre-load scan and no file-handle rule, so
        # every pickle no other rule covers is LOADED in the source. The
        # kernels receive the manifest as source text inside each program's
        # prelude, so the switch-off is appended to that text.
        _ctl.MANIFEST_SRC += ("\n\ndef _load_hazard(sc):\n    return None\n"
                              "\n\ndef _file_handle_rule(sc):\n    return None\n")
    elif kind == "inplace_validation":
        # The first controller's validation: a real optimizer step on the
        # committed state. Every oracle still passes; the clean run must now
        # ABORT at the commit boundary, so its expectation FAILS.
        orig = _ctl.verify_program
        _ctl.verify_program = lambda *a, **kw: orig(*a, **{**kw, "continuation": "inplace"})
    elif kind:
        raise SystemExit(f"unknown --sabotage {kind}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default=os.environ.get("CLUSY_API_URL", "http://localhost:8010"))
    ap.add_argument("--faults", default=",".join(f or "clean" for f in FAULTS))
    ap.add_argument("--skip-concurrent", action="store_true")
    ap.add_argument("--cohort", default=None, help="written into every record (default: the run id)")
    ap.add_argument("--strict-unknown", action="store_true",
                    help="block on unknown names and process effects (or effects that could not be measured) "
                         "instead of reporting them; the fixture is NOT expected to pass this, see the "
                         "controller's preflight policy, so keep it out of a live cohort")
    ap.add_argument("--local", action="store_true", help="dry run against LocalApi; writes to --outdir")
    ap.add_argument("--outdir", default=None, help="output directory (default results/e11; a temp dir with --local)")
    ap.add_argument("--sabotage", choices=["stale_inventory", "unsafe_exclusion", "unsafe_validator",
                                           "inplace_validation"], default=None,
                    help="local only: break one thing on purpose to show the checks fail")
    args = ap.parse_args()
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    faults = [None if f == "clean" else f for f in args.faults.split(",")]

    if args.local:
        from localapi import LocalApi
        if "insufficient_memory" in faults:
            # The fault allocates and FILLS 128 GiB. An E2B sandbox refuses the
            # allocation; a laptop with memory overcommit (macOS) grants it and
            # then pages it in, so it is never run locally.
            log("  --local: skipping insufficient_memory (it would fill 128 GiB of overcommitted host memory)")
            faults = [f for f in faults if f != "insufficient_memory"]
        outdir = Path(args.outdir or tempfile.mkdtemp(prefix="e11-local-"))
        if (ROOT / "results").resolve() in [outdir.resolve(), *outdir.resolve().parents]:
            # A dry run is not a result; it must never land beside live records.
            print("--local refuses to write under results/; pass a scratch --outdir", file=sys.stderr)
            return 2
        api = LocalApi(workdir=outdir / "kernels")
        api_url = None
        args.skip_concurrent = True
        _sabotage(api, args.sabotage)
    else:
        if args.sabotage:
            print("--sabotage is local only", file=sys.stderr); return 2
        key = os.environ.get("CLUSY_HARNESS_API_KEY")
        if not key:
            print("export CLUSY_HARNESS_API_KEY", file=sys.stderr); return 1
        api = Api(args.api_url, key)
        api_url = args.api_url
        outdir = Path(args.outdir) if args.outdir else ROOT / "results" / "e11"
        why = runmeta.shipped_record_conflict(outdir / "e11_runs.jsonl", args.cohort)
        if why:
            print(why, file=sys.stderr); return 2
    outdir.mkdir(parents=True, exist_ok=True)

    meta = runmeta.start_run("e11", api_url=api_url, args=vars(args), cohort=args.cohort,
                             extra_files=[ROOT / "experiments" / "localapi.py"] if args.local else None)
    meta["cohort"] = args.cohort or meta["run_id"]
    meta["local"] = bool(args.local)
    journal = Journal(outdir / "journal.sqlite")
    log(f"E11 run {meta['run_id']} cohort {meta['cohort']} -> {outdir}")

    rows = []
    try:
        with (outdir / "e11_runs.jsonl").open("a") as f:
            for fault in faults:
                r = run_one(api, journal, outdir / "blobs", fault, log, strict_unknown=args.strict_unknown)
                r["cohort"], r["local"] = meta["cohort"], bool(args.local)
                runmeta.stamp(r, meta)
                rows.append(r); f.write(json.dumps(r, default=str) + "\n"); f.flush()
                log(f"  -> {r['phase']:<8} failed_at={r['failed_at']} reason={r['reason']} | source accepts after: {r['source_accepts_after']} "
                    f"(counter {r['source_counter_after']}, fenced_by {r['source_fenced_by']}) | dest accepts: {r['dest_accepts_after']} | "
                    f"authoritative={r['authoritative_is']} | {r['wall_s']}s | {'PASS' if r['as_expected'] else 'FAIL ' + str(r['failed_checks'])}")
            if not args.skip_concurrent:
                cp = concurrent_patch_probe(api, log)
                f.write(json.dumps(runmeta.stamp({"probe": "concurrent_patch", "cohort": meta["cohort"], **cp}, meta)) + "\n")
    finally:
        runmeta.finish_run(meta, api_url=api_url)
        meta["results"] = {"trials": len(rows), "as_expected": sum(1 for r in rows if r["as_expected"])}
        runmeta.write_meta(meta, outdir / "runs_meta.jsonl")
        if args.local:
            api.close()
    print("\nE11 summary")
    print(f"{'fault':<31} {'phase':<9} {'failed at':<14} {'source usable after':<20} {'dest accepts':<13} {'authority':<9} {'wall s':>7}  verdict")
    for r in rows:
        print(f"{str(r['fault']):<31} {r['phase']:<9} {str(r['failed_at']):<14} {str(r['source_accepts_after']):<20} "
              f"{str(r['dest_accepts_after']):<13} {r['authoritative_is']:<9} {r['wall_s']:>7}  "
              f"{'PASS' if r['as_expected'] else 'FAIL ' + ','.join(r['failed_checks'])}")
    print(f"\n{'fault':<31} {'gate closed':<12} {'gate after':<11} {'refused':>7} {'drain waited':>12} "
          f"{'routed after on':<16} {'writes kept':<12} commit boundary (restored / pre-commit)")
    for r in rows:
        adm = r.get("admission") or {}
        cb = ((r.get("commit_boundary") or {}).get("verdicts")) or {}
        eq = lambda v: "-" if not v else ("equal" if v.get("equal") else "MISMATCH " + ",".join(v.get("mismatched", [])))  # noqa: E731
        drains = adm.get("drains") or []
        print(f"{str(r['fault']):<31} {str(adm.get('closed_during_migration')):<12} {str(adm.get('state')):<11} "
              f"{str(adm.get('refused')):>7} {str(sum(d.get('waited_on', 0) for d in drains)):>12} "
              f"{str((r.get('routed_after') or {}).get('runtime')):<16} {str(r.get('routed_count_after') == 2):<12} "
              f"{eq(cb.get('restored'))} / {eq(cb.get('pre_commit'))}")
    for r in rows:
        pf = r.get("preflight") or {}
        if pf:
            print(f"  preflight[{r['fault'] or 'clean'}]: decision={pf.get('decision')} reason={pf.get('reason')} "
                  f"excluded={pf.get('excluded')} unknown={[u['name'] for u in pf.get('unknown', [])]} "
                  f"effects={[e['effect'] for e in (pf.get('effects') or [])]}")
    print("  (admission: these runs route one write before and one after each switch, so the gate is never met "
          "while closed; refusal and drain evidence is in E12 and tests/test_handoff_admission.py)")
    for r in rows:
        v = r.get("validation") or {}
        if v:
            print(f"  validation[{r['fault'] or 'clean'}]: mode={v.get('continuation')} isolated={v.get('isolated')} "
                  f"{v.get('passed')}/{v.get('total')} declared={[d['name'] for d in v.get('declared') or []]}")
    n_ok = sum(1 for r in rows if r["as_expected"])
    print(f"\nE11 {n_ok}/{len(rows)} runs as expected")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
