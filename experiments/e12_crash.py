"""E12 — crash-recoverable handoff: kill the controller at every phase.

For each phase boundary the controller is killed with SIGKILL semantics
(`os._exit`) either just AFTER the journal write (the classic crash) or just
BEFORE it (a lost acknowledgement: the side effect happened, the journal does
not know). A fresh controller is then started with the same migration id and
must finish the migration. Three invariants are checked after recovery, plus
one duplicate-request race:

  1. exactly one runtime accepts execution (source fenced XOR destination live)
  2. committed state is recoverable: all twelve oracles pass on the
     authoritative runtime, checked without writing to it (every check runs
     in a forked child of the kernel, which itself runs no model code)
  3. abandoned resources are reclaimed: no second destination survives
  4. no acknowledged routed write is lost: one write is routed through the
     controller before the crash and one while it is down. The one while it
     is down must be refused exactly when the journal's phase is inside the
     admission gate (ADMISSION_CLOSED .. DEST_VALIDATED) and acknowledged
     otherwise; after recovery every acknowledged write is on the
     authoritative runtime.

Recovery time is measured from the restart to DONE. Every record also carries
the three commit-boundary fingerprints (cut, after restore, before COMMITTED)
and their verdicts, from the journal.

WHAT THE COHORT COVERS, stated so a reader cannot over-read it. The crash
points are the controller's PHASE HOOKS (before and after every journal
write, now including PREFLIGHT) plus ONE point inside a phase: the
create-response-to-journal window (`DEST_CREATED`), where the provider has
created the destination and the controller dies before journaling its id. It
does not kill the controller at every instruction boundary; the other points
inside a phase are between two idempotent side effects that the phase hooks
already bracket. Every record carries `coverage` saying this.

For `DEST_CREATED` the expected outcome is specific: the restarted controller
knows nothing of the first destination, so it creates a second one and
completes (DONE); `reclaim()` must find the first by name and delete exactly
one orphan; no destination for the migration id may survive except the
committed one.

    python experiments/e12_crash.py --cohort e12-rerun --outdir /tmp/soma-rerun/e12   # live
    python experiments/e12_crash.py --local --outdir /tmp/e12                         # dry run

`--local` runs the crashing controller IN PROCESS against LocalApi (a crash
raises `ControllerKilled` instead of `os._exit`, which would also take down the
kernels the test double owns) and writes to `--outdir`. `--sabotage
reclaim_blind` (local only) gives the recovering controller a provider whose
project listing is empty, the failure mode that once made reclaim silently
succeed; the DEST_CREATED trial must then FAIL invariant 3.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import runmeta  # noqa: E402
from handoff.controller import (  # noqa: E402
    ADMISSION_CLOSED_PHASES, DEST_CREATED, MARK, PHASES, AdmissionClosed, Api, Controller, ControllerKilled,
    CrashPoint, Journal, _wrap, recovery_check_program, seed_program, touch_program,
)

#: Every phase whose journal write is a crash hook, in protocol order.
CRASH_PHASES = [p for p in PHASES if p not in ("REQUESTED", "DONE")]
COVERAGE = "phase hooks + create-response window"

CHILD = r'''
import sys, json
sys.path.insert(0, %(src)r)
from pathlib import Path
from handoff.controller import Api, Controller, CrashPoint, Journal
api = Api(%(url)r, %(key)r)
j = Journal(Path(%(journal)r))
c = Controller(api, j, Path(%(blobs)r), crash=CrashPoint(%(phase)r, after_journal=%(after)r), log=lambda *a: print(*a, flush=True))
res = c.migrate(%(mig)r, %(src_pid)r, "cpu")
print("__CHILD__" + json.dumps(res, default=str))
'''


def _raise_killed() -> None:
    raise ControllerKilled()


def routed_write_program() -> str:
    return _wrap(f'''
import sys, json
_g = sys.modules["__main__"].__dict__
_g["e12_routed_writes"] = _g.get("e12_routed_writes", 0) + 1
print({MARK!r} + json.dumps({{"counter": _g["e12_routed_writes"]}}))
''')


def routed_count_program() -> str:
    return _wrap(f'''
import sys, json
print({MARK!r} + json.dumps({{"counter": sys.modules["__main__"].__dict__.get("e12_routed_writes")}}))
''')


class _BlindList:
    """A provider view whose project listing is empty (sabotage)."""

    def __init__(self, api):
        self._api = api

    def __getattr__(self, name):
        return getattr(self._api, name)

    def list_projects(self, timeout: float = 300.0) -> list[dict]:
        return []


def _run_crashing_controller(api, journal, blobs, phase, after_journal, mig, src, local: bool) -> int:
    """Run a controller that dies at the crash point; return its exit code
    (137 when it died as planned)."""
    if not local:
        code = CHILD % {"src": str(ROOT / "src"), "url": api.base, "key": api.key, "journal": str(journal.path),
                        "blobs": str(blobs), "phase": phase, "after": after_journal, "mig": mig, "src_pid": src}
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=1500).returncode
    # In process: a separate journal connection, as a separate process would
    # have, and a kill that unwinds without running any cleanup.
    j = Journal(journal.path)
    c = Controller(api, j, blobs, crash=CrashPoint(phase, after_journal=after_journal),
                   log=lambda *a: print(*a, flush=True), kill=_raise_killed)
    try:
        c.migrate(mig, src, "cpu")
        return 0
    except ControllerKilled:
        return 137
    finally:
        j.db.close()


def expected(phase: str) -> dict:
    exp = {"final_phase": "DONE", "exactly_one_runtime_accepts": True, "committed_state_recoverable": True,
           "no_abandoned_destination": True, "crashed": True, "gate_matches_phase_while_down": True,
           "acknowledged_writes_preserved": True}
    if phase == DEST_CREATED:
        exp["reclaimed"] = 1
    return exp


def one_crash(api, journal, blobs, phase: str, after_journal: bool, log, *, local: bool = False,
              recover_api=None) -> dict:
    tag = "created" if phase == DEST_CREATED else phase.lower()
    mig = f"e12-{tag}-{'aj' if after_journal else 'bj'}-{uuid.uuid4().hex[:5]}"
    src = api.create_project(f"clusy-exp-handoff-src-{mig}", "cpu")
    api.witness(src, seed_program(seed=20260925, files=4, corpus_bytes=16 * 1024))
    api.witness(src, touch_program())
    # 0. a write routed through the controller before anything starts: the
    # source acknowledges it, so it must survive whatever happens next.
    router = Controller(recover_api or api, journal, blobs)
    journal.create(mig, src, "cpu")
    acked = [router.route(mig, routed_write_program())["runtime"]]
    # 1. run the controller so that it kills itself at the crash point
    t0 = time.perf_counter()
    rc = _run_crashing_controller(api, journal, blobs, phase, after_journal, mig, src, local)
    crashed = rc == 137
    row_at_crash = journal.get(mig) or {}
    phase_at_crash = row_at_crash.get("phase")
    # 1b. a write while no controller runs: refused exactly when the phase the
    # journal holds is inside the gate. The expectation is derived from the
    # PHASE, the gate from the admission column: they must agree.
    expect_refused = phase_at_crash in ADMISSION_CLOSED_PHASES
    try:
        down = router.route(mig, routed_write_program())
        write_while_down = {"refused": False, "runtime": "source" if down["runtime"] == src else "dest"}
        acked.append(down["runtime"])
    except AdmissionClosed:
        write_while_down = {"refused": True, "runtime": None}
    write_while_down.update(admission_at_crash=row_at_crash.get("admission"), expected_refused=expect_refused)
    alive_at_crash = [it["id"] for it in api.list_projects() if str(it.get("name", "")) == f"clusy-exp-handoff-{mig}"]
    log(f"[{mig}] child exit {rc} ({'crashed as planned' if crashed else 'DID NOT CRASH'}) journal phase now {phase_at_crash}; "
        f"destinations alive {len(alive_at_crash)}")
    # 2. recover: a fresh controller, same migration id
    t1 = time.perf_counter()
    c = Controller(recover_api or api, journal, blobs, log=log)
    res = c.migrate(mig, src, "cpu", takeover=True)   # the supervisor knows the previous controller is dead
    recovery_s = time.perf_counter() - t1
    reclaimed = c.reclaim(mig)
    row = journal.get(mig)
    # 3. invariants
    src_acc = c.runtime_accepts(src)
    dest_acc = c.runtime_accepts(row["dest_pid"]) if row.get("dest_pid") else {"accepted": False}
    exactly_one = bool(src_acc.get("accepted")) != bool(dest_acc.get("accepted"))
    committed_ok = None
    validated = next((e for e in journal.events(mig) if e["phase"] == "DEST_VALIDATED"), None) is not None
    validation_mode, declared = None, []
    if row["phase"] == "DONE":
        try:
            v = api.witness(row["authoritative"], recovery_check_program())
            committed_ok = v["passed"] == v["total"]
            oracles = f"{v['passed']}/{v['total']}"
            # How the check ran, so a figure never describes an older mode:
            # "isolated" (forked child) since the second review. A declared
            # row did not run; it is recorded, and neither passes nor fails.
            validation_mode = v.get("continuation")
            declared = [d["name"] for d in v.get("declared") or []]
        except Exception as e:  # noqa: BLE001
            committed_ok, oracles = False, f"check failed: {e}"
    else:
        oracles = "-"
    # 3c. Invariant 3, MEASURED rather than inferred. The previous version of
    # this harness recorded only reclaim()'s return list, and the figure scored
    # the invariant as "reclaimed list empty OR run reached DONE", which passes
    # vacuously for every completed run. What the invariant actually claims is
    # that after recovery no destination for this migration id survives except
    # the committed one, so that is what is counted here, from a fresh listing.
    survivors = [it["id"] for it in api.list_projects()
                 if str(it.get("name", "")) == f"clusy-exp-handoff-{mig}"]
    orphans = [pid for pid in survivors if pid != row.get("dest_pid")]
    # 3d. Invariant 4: every acknowledged routed write is on the authority.
    try:
        routed_count = api.witness(row["authoritative"], routed_count_program())["counter"]
    except Exception as e:  # noqa: BLE001
        routed_count = f"read failed: {e}"
    boundary = c._boundary_report(mig)
    out = {
        "migration": mig, "crash_phase": phase, "after_journal": after_journal,
        "trial": "create_window" if phase == DEST_CREATED else "phase_hook", "coverage": COVERAGE,
        "crashed": crashed, "child_exit": rc,
        "journal_phase_at_crash": phase_at_crash, "destinations_alive_at_crash": len(alive_at_crash),
        "final_phase": row["phase"], "resumed": res.get("resumed"),
        "abort_reason": row.get("abort_reason"),
        "recovery_s": round(recovery_s, 1), "child_wall_s": round(t1 - t0, 1),
        "source_accepts": src_acc.get("accepted"), "source_fenced_by": src_acc.get("fenced_by"),
        "dest_accepts": dest_acc.get("accepted"),
        "exactly_one_runtime_accepts": exactly_one, "committed_state_recoverable": committed_ok, "oracles": oracles,
        "validation_mode": validation_mode, "oracles_declared": declared,
        "validated_12_of_12_before_commit": validated,
        "reclaimed": reclaimed["reclaimed"],
        "survivors_after_reclaim": survivors, "orphans_after_reclaim": orphans,
        "no_abandoned_destination": not orphans,
        "phases": list(PHASES),
        "write_while_down": write_while_down,
        "gate_matches_phase_while_down": (write_while_down["refused"] == expect_refused
                                          and (row_at_crash.get("admission") == "closed") == expect_refused),
        "routed_writes_acknowledged": len(acked), "routed_count_after": routed_count,
        "acknowledged_writes_preserved": routed_count == len(acked),
        "admission": c.admission_evidence(mig),
        "commit_boundary": boundary,
        "commit_boundary_equal": all((boundary["verdicts"].get(k) or {}).get("equal") is True
                                     for k in ("restored", "pre_commit")) if boundary["verdicts"] else None,
        "expect": expected(phase),
        "events": journal.events(mig),
    }
    fails = [k for k, want in out["expect"].items()
             if (len(out["reclaimed"]) if k == "reclaimed" else out[k]) != want]
    out["as_expected"], out["failed_checks"] = not fails, fails
    # 4. tear down what this trial owns, then verify from a fresh listing that
    # nothing carrying this migration id is left behind. A delete that returns
    # a success code but leaves the project alive is a known failure mode here,
    # so the teardown is checked rather than assumed.
    delete_codes = {}
    for pid in {src, row.get("dest_pid"), *survivors} - {None}:
        delete_codes[pid] = api.delete_project(pid)
    leftover = [it["id"] for it in api.list_projects() if mig in str(it.get("name", ""))]
    out["delete_codes"] = delete_codes
    out["leftover_after_teardown"] = leftover
    out["fully_cleaned"] = not leftover
    return out


def duplicate_request(api, journal, blobs, log) -> dict:
    """Two controllers issue the same migration id concurrently."""
    mig = f"e12-dup-{uuid.uuid4().hex[:5]}"
    src = api.create_project(f"clusy-exp-handoff-src-{mig}", "cpu")
    api.witness(src, seed_program(seed=20260925, files=4, corpus_bytes=16 * 1024))
    results = {}
    def go(name):
        c = Controller(api, Journal(journal.path), blobs, log=lambda *a: None)
        results[name] = c.migrate(mig, src, "cpu")
    ts = [threading.Thread(target=go, args=(n,)) for n in ("A", "B")]
    for t in ts: t.start()
    for t in ts: t.join()
    row = journal.get(mig)
    c = Controller(api, journal, blobs)
    reclaimed = c.reclaim(mig)
    # how many destinations were ever created for this id?
    dests = [it["id"] for it in api.list_projects() if str(it.get("name", "")) == f"clusy-exp-handoff-{mig}"]
    out = {"migration": mig, "final_phase": row["phase"], "A": results["A"].get("phase"), "B": results["B"].get("phase"),
           "one_joined": bool(results["A"].get("joined")) != bool(results["B"].get("joined")),
           "destinations_alive": len(dests), "reclaimed": reclaimed["reclaimed"]}
    log(f"[{mig}] duplicate request: A={out['A']} B={out['B']} destinations alive={out['destinations_alive']}")
    for pid in {src, row.get("dest_pid"), *dests} - {None}:
        api.delete_project(pid)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default=os.environ.get("CLUSY_API_URL", "http://localhost:8010"))
    ap.add_argument("--phases", default=",".join(CRASH_PHASES + [DEST_CREATED]))
    ap.add_argument("--modes", default="after,before")
    ap.add_argument("--skip-duplicate", action="store_true")
    ap.add_argument("--cohort", default=None, help="written into every record (default: the run id)")
    ap.add_argument("--local", action="store_true", help="dry run against LocalApi; writes to --outdir")
    ap.add_argument("--outdir", default=None, help="output directory (default results/e12; a temp dir with --local)")
    ap.add_argument("--sabotage", choices=["reclaim_blind"], default=None,
                    help="local only: break one thing on purpose to show the checks fail")
    args = ap.parse_args()
    log = lambda *a: print(*a, flush=True)  # noqa: E731

    if args.local:
        from localapi import LocalApi
        outdir = Path(args.outdir or tempfile.mkdtemp(prefix="e12-local-"))
        if (ROOT / "results").resolve() in [outdir.resolve(), *outdir.resolve().parents]:
            # A dry run is not a result; it must never land beside live records.
            print("--local refuses to write under results/; pass a scratch --outdir", file=sys.stderr)
            return 2
        api = LocalApi(workdir=outdir / "kernels")
        api_url = None
    else:
        if args.sabotage:
            print("--sabotage is local only", file=sys.stderr); return 2
        key = os.environ.get("CLUSY_HARNESS_API_KEY")
        if not key:
            print("export CLUSY_HARNESS_API_KEY", file=sys.stderr); return 1
        api = Api(args.api_url, key)
        api_url = args.api_url
        outdir = Path(args.outdir) if args.outdir else ROOT / "results" / "e12"
        why = runmeta.shipped_record_conflict(outdir / "e12_runs.jsonl", args.cohort)
        if why:
            print(why, file=sys.stderr); return 2
    outdir.mkdir(parents=True, exist_ok=True)
    recover_api = _BlindList(api) if args.sabotage == "reclaim_blind" else None

    # The trial list: every phase hook in both modes, and the create window,
    # which exists only BEFORE a journal write by definition.
    trials = []
    for phase in args.phases.split(","):
        for mode in args.modes.split(","):
            if phase == DEST_CREATED and mode == "after":
                continue
            trials.append((phase, mode == "after"))

    meta = runmeta.start_run("e12", api_url=api_url, args=vars(args), cohort=args.cohort,
                             extra_files=[ROOT / "experiments" / "localapi.py"] if args.local else None)
    meta["cohort"] = args.cohort or meta["run_id"]
    meta["local"] = bool(args.local)
    meta["coverage"] = {"description": COVERAGE, "phases": list(PHASES),
                        "trials": [{"crash_phase": p, "after_journal": a} for p, a in trials]}
    journal = Journal(outdir / "journal.sqlite")
    log(f"E12 run {meta['run_id']} cohort {meta['cohort']} -> {outdir}  ({len(trials)} crash points: {COVERAGE})")
    rows = []
    try:
        with (outdir / "e12_runs.jsonl").open("a") as f:
            for phase, after in trials:
                r = one_crash(api, journal, outdir / "blobs", phase, after_journal=after, log=log,
                              local=args.local, recover_api=recover_api)
                r["cohort"], r["local"] = meta["cohort"], bool(args.local)
                runmeta.stamp(r, meta)
                rows.append(r); f.write(json.dumps(r, default=str) + "\n"); f.flush()
                log(f"  -> final {r['final_phase']} recovery {r['recovery_s']}s | one runtime accepts: {r['exactly_one_runtime_accepts']} "
                    f"| committed recoverable: {r['committed_state_recoverable']} ({r['oracles']}) "
                    f"| write while down {'refused' if r['write_while_down']['refused'] else 'acknowledged'} "
                    f"(gate matches phase: {r['gate_matches_phase_while_down']}) | writes kept {r['acknowledged_writes_preserved']} "
                    f"| reclaimed {len(r['reclaimed'])} orphans left {len(r['orphans_after_reclaim'])} "
                    f"| cleaned {r['fully_cleaned']} | {'PASS' if r['as_expected'] else 'FAIL ' + str(r['failed_checks'])}")
            if not args.skip_duplicate:
                d = duplicate_request(api, journal, outdir / "blobs", log)
                f.write(json.dumps(runmeta.stamp({"probe": "duplicate_request", "cohort": meta["cohort"],
                                                  "local": bool(args.local), **d}, meta)) + "\n")
    finally:
        runmeta.finish_run(meta, api_url=api_url)
        meta["results"] = {"trials": len(rows), "as_expected": sum(1 for r in rows if r["as_expected"])}
        runmeta.write_meta(meta, outdir / "runs_meta.jsonl")
        if args.local:
            api.close()
    print("\nE12 summary")
    print(f"{'crash at':<16} {'ack':<7} {'crashed':<8} {'final':<9} {'recovery s':>10} {'one accepts':<12} "
          f"{'state ok':<9} {'oracles':<8} {'reclaimed':>9} {'orphans':>8} {'cleaned':<8} verdict")
    for r in rows:
        print(f"{r['crash_phase']:<16} {('after' if r['after_journal'] else 'before'):<7} {str(r['crashed']):<8} {r['final_phase']:<9} "
              f"{r['recovery_s']:>10} {str(r['exactly_one_runtime_accepts']):<12} {str(r['committed_state_recoverable']):<9} {r['oracles']:<8} "
              f"{len(r['reclaimed']):>9} {len(r['orphans_after_reclaim']):>8} {str(r['fully_cleaned']):<8} "
              f"{'PASS' if r['as_expected'] else 'FAIL ' + ','.join(r['failed_checks'])}")
    n = len(rows)
    print(f"\nINVARIANTS over {n} crash points ({COVERAGE})")
    print(f"  exactly one runtime accepts      {sum(1 for r in rows if r['exactly_one_runtime_accepts'])}/{n}")
    print(f"  committed state recoverable      {sum(1 for r in rows if r['committed_state_recoverable'])}/{n}")
    print(f"  no abandoned destination         {sum(1 for r in rows if r['no_abandoned_destination'])}/{n}")
    print(f"  gate matches phase while down    {sum(1 for r in rows if r['gate_matches_phase_while_down'])}/{n}"
          f"  (refused while down: {sum(1 for r in rows if r['write_while_down']['refused'])})")
    print(f"  acknowledged writes preserved    {sum(1 for r in rows if r['acknowledged_writes_preserved'])}/{n}")
    print(f"  commit boundary equal            {sum(1 for r in rows if r['commit_boundary_equal'])}/{n}")
    print(f"  destinations reclaimed (count)   {sum(len(r['reclaimed']) for r in rows)}")
    print(f"  trials leaving nothing behind    {sum(1 for r in rows if r['fully_cleaned'])}/{n}")
    print(f"  as expected                      {sum(1 for r in rows if r['as_expected'])}/{n}")
    return 0 if all(r["as_expected"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
