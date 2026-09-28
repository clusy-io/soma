"""The admission gate: no acknowledged routed write is lost across a handoff.

A reviewer ran the controller against persistent local kernels, stopped it at
CAPTURED, sent a mutation through `execute_routed` (acknowledged, counter 1),
resumed, and the migration committed counter 0 (controller-probes.json in
output/paper-review-2026-09-27). These tests port that scenario and the
routing probe from semantic_probes.py, and then go further: a writer thread
issuing counter increments AND workspace file appends through the route for
the whole migration, with and without a controller crash inside the gate, an
aborted migration, the drain against an execution still in transit, and the
dead-owner and timeout paths of the drain.

The invariant every scenario checks is the one the paper needs: every
acknowledged routed write is present on the runtime that is authoritative
afterwards, every refused write is absent, and writes after the commit (or
abort) land on the new (or old) authority.

A second review found the gate was kept per migration ROW: a write routed
through the previous hop of a chain, or through a second id for the same
source, was admitted onto a gated runtime and lost (chain_bypass.py and
same_source_bypass.py). The last section ports both and pins the runtime-keyed
rule: the lineage resolution, the one-outgoing-migration start check, and a
writer routing through the first id across two hops.

Everything runs through `experiments/localapi.py`, the same test double the
E11/E12 dry runs use, in real child interpreters. LocalApi's pipe protocol is
not safe for two threads at once, so `SerialApi` below gives each kernel a
lock: a real kernel also runs one execution at a time and queues the rest.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from handoff import controller as ctl  # noqa: E402
from handoff.controller import MARK, Controller, ControllerKilled, CrashPoint, Journal, _wrap, seed_program  # noqa: E402

localapi = pytest.importorskip("localapi")

#: The refusal type. Looked up rather than imported so this file can also be
#: run against the controller as it was before the gate existed (the report's
#: old-code runs): there the placeholder is never raised, and every test that
#: expects a refusal FAILS with "DID NOT RAISE" instead of an import error.
AdmissionClosed = getattr(ctl, "AdmissionClosed", type("AdmissionClosedMissing", (Exception,), {}))


class SerialApi(localapi.LocalApi):
    """LocalApi with one lock per kernel, plus an optional hold for programs
    carrying a marker. The hold happens BEFORE the kernel lock is taken, so it
    models an execution that was admitted and is still in transit to the
    kernel: the kernel lock alone cannot hold the capture back for it, only
    the drain can. The hold lasts until `release_when()` is true (polled), so
    the tests order events by journal state rather than by sleeps."""

    def __init__(self, *, delay_marker: str | None = None, release_when=None, hold_timeout: float = 10.0, **kw):
        super().__init__(**kw)
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self.delay_marker, self.release_when, self.hold_timeout = delay_marker, release_when, hold_timeout

    def _lock(self, pid: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(pid, threading.Lock())

    def execute(self, pid, code, timeout_ms=600_000):
        if self.delay_marker and self.delay_marker in code and self.release_when is not None:
            deadline = time.time() + self.hold_timeout
            while not self.release_when() and time.time() < deadline:
                time.sleep(0.005)
        with self._lock(pid):
            return super().execute(pid, code, timeout_ms)

    def delete_project(self, pid):
        with self._lock(pid):
            return super().delete_project(pid)


def program(body: str) -> str:
    """A harness program in a private dict (see `_wrap`), with `g` bound to
    the kernel's real `__main__` for the writes it makes on purpose."""
    return _wrap('import sys, json, os\ng = sys.modules["__main__"].__dict__\n' + body)


def write_program(seq: int, marker: str = "") -> str:
    """One user write: increment a counter in the namespace AND append the
    sequence number to a workspace file (inside `data/`, which the capsule
    carries), then acknowledge with both."""
    return program(f'''
g["audit_counter"] = g.get("audit_counter", 0) + 1
os.makedirs("data", exist_ok=True)
with open(os.path.join("data", "audit.log"), "a") as f:
    f.write("{seq}\\n")
print({MARK!r} + json.dumps({{"counter": g["audit_counter"], "seq": {seq}}}))  # {marker}
''')


STATE = program(f'''
_p = os.path.join("data", "audit.log")
_log = [int(x) for x in open(_p).read().split()] if os.path.exists(_p) else []
print({MARK!r} + json.dumps({{"counter": g.get("audit_counter"), "log": _log}}))
''')


def _kill():
    raise ControllerKilled()


@pytest.fixture
def api(tmp_path):
    a = SerialApi(workdir=tmp_path / "kernels")
    yield a
    a.close()


def _source(api, name="src") -> str:
    pid = api.create_project(f"clusy-exp-handoff-src-{name}", "cpu")
    api.witness(pid, seed_program(seed=7, files=2, corpus_bytes=2048))
    api.witness(pid, program(f'g["audit_counter"] = 0\nprint({MARK!r} + "{{}}")'))
    return pid


class Writer(threading.Thread):
    """Issues routed writes until stopped, recording each outcome: ('ack',
    seq, runtime, counter) or ('refused', seq). Anything else is an error the
    test fails on."""

    def __init__(self, c: Controller, mig: str, start_seq: int = 1, pause: float = 0.004):
        super().__init__(daemon=True)
        self.c, self.mig, self.seq, self.pause = c, mig, start_seq, pause
        self.stop = threading.Event()
        self.log: list[tuple] = []

    def run(self):
        while not self.stop.is_set():
            seq = self.seq
            self.seq += 1
            try:
                r = self.c.route(self.mig, write_program(seq))
                self.log.append(("ack", seq, r["runtime"], r["result"]["counter"]))
            except AdmissionClosed:
                self.log.append(("refused", seq))
            except Exception as e:  # noqa: BLE001
                self.log.append(("error", seq, f"{type(e).__name__}: {e}"))
            time.sleep(self.pause)

    def finish(self, grace: float = 0.3):
        time.sleep(grace)
        self.stop.set()
        self.join(timeout=30)
        assert not self.is_alive()

    def acks(self):
        return [e for e in self.log if e[0] == "ack"]

    def refused(self):
        return [e[1] for e in self.log if e[0] == "refused"]

    def errors(self):
        return [e for e in self.log if e[0] == "error"]


def _assert_no_write_lost(final: dict, acked_seqs: list[int], refused: list[int]):
    """Every acknowledged write present exactly once, in acknowledgement
    order; every refused write absent; the counter agrees with the log."""
    assert final["log"] == acked_seqs, (final["log"][:20], acked_seqs[:20])
    assert not set(refused) & set(final["log"])
    assert final["counter"] == len(acked_seqs)


# ---------------------------------------------------------------------------
# The reviewer's scenario, and the routing probe
# ---------------------------------------------------------------------------

def test_reviewer_post_capture_write_is_refused_and_nothing_acknowledged_is_lost(api, tmp_path):
    """controller_probes.py, scenario 1. Old code: the write is ACKNOWLEDGED
    on the source after the capture (counter 1) and DONE commits counter 0."""
    src = _source(api)
    j = Journal(tmp_path / "journal.sqlite")
    with pytest.raises(ControllerKilled):
        Controller(api, Journal(j.path), tmp_path / "blobs", crash=CrashPoint("CAPTURED"), kill=_kill).migrate(
            "audit", src, "cpu")
    assert j.get("audit")["phase"] == "CAPTURED"
    c = Controller(api, j, tmp_path / "blobs")        # restarted, not yet resumed
    acknowledged = None
    with pytest.raises(AdmissionClosed):
        acknowledged = c.execute_routed("audit", write_program(1))
    source_after_write = api.witness(src, STATE)
    res = c.migrate("audit", src, "cpu", takeover=True)
    assert res["phase"] == "DONE", res
    committed = api.witness(res["dest"], STATE)
    lost = acknowledged is not None and committed["counter"] != acknowledged["counter"]
    assert not lost
    assert source_after_write["counter"] == 0 and committed["counter"] == 0 and committed["log"] == []
    # After DONE the route goes to the destination, and the gate is open.
    after = c.route("audit", write_program(2))
    assert after["runtime"] == res["dest"] and after["result"]["counter"] == 1
    row = j.get("audit")
    assert row["admission"] == "open" and row["authoritative"] == res["dest"]
    assert res["admission"]["refused"] == 1 and res["admission"]["closed_during_migration"]


class _RoutingProbe:
    """semantic_probes.py's provider double: records where work would run."""

    def __init__(self):
        self.calls = []

    def witness(self, pid, code):
        self.calls.append(pid)
        return {"accepted": True, "routed_to": pid}


def test_routing_probe_after_capture_is_refused(tmp_path):
    """semantic_probes.py `execution_after_capture`: a journal advanced to
    CAPTURED. Old code: {"accepted": true, "routed_to": "source"}."""
    j = Journal(tmp_path / "journal.sqlite")
    j.create("probe", "source", "cpu")
    j.advance("probe", "CAPTURED")
    probe = _RoutingProbe()
    c = Controller(probe, j, tmp_path / "blobs")
    with pytest.raises(AdmissionClosed):
        c.execute_routed("probe", "user mutation")
    assert probe.calls == []                       # nothing was sent anywhere
    j.advance("probe", "COMMITTED", authoritative="dest")
    assert c.execute_routed("probe", "user mutation") == {"accepted": True, "routed_to": "dest"}


def test_the_gate_is_derived_from_the_phase_in_the_same_write(tmp_path):
    j = Journal(tmp_path / "journal.sqlite")
    j.create("m", "s", "cpu")
    seen = {}
    for phase in ctl.PHASES[1:] + [ctl.ABORTED]:
        j.advance("m", phase)
        seen[phase] = j.get("m")["admission"]
    assert [p for p, a in seen.items() if a == "closed"] == ["ADMISSION_CLOSED", "CAPTURED", "DEST_RESTORED",
                                                             "DEST_VALIDATED"]


def test_route_and_close_are_serialised(tmp_path):
    """The routing read and the close are one transaction each, so an
    admission either committed before the close (and is visible to the drain)
    or was refused after it. Many threads race one close."""
    j = Journal(tmp_path / "journal.sqlite")
    j.create("race", "src", "cpu")
    results, go = [], threading.Event()

    def admitter(n):
        go.wait()                           # each thread uses its own connection (Journal.conn)
        for _ in range(20_000):
            try:
                a = j.admit("race", f"{socket.gethostname()}:{os.getpid()}:{n}", os.getpid())
                results.append(("ok", a["exec_id"]))
            except AdmissionClosed:
                results.append(("refused", None))
                return
    ts = [threading.Thread(target=admitter, args=(n,)) for n in range(6)]
    for t in ts:
        t.start()
    go.set()
    time.sleep(0.01)
    j.advance("race", "ADMISSION_CLOSED")
    close_at = next(e for e in reversed(j.events("race")) if e["phase"] == "ADMISSION_CLOSED")["at"]
    open_after_close = {r["exec_id"] for r in j.unfinished("race", "src")}
    for t in ts:
        t.join()
    admitted = {e for k, e in results if k == "ok"}
    assert admitted and any(k == "refused" for k, _ in results)
    # Every admission is visible to a drain that starts after the close ...
    assert admitted == open_after_close
    # ... and was registered before the close committed.
    started = dict(j.conn().execute("SELECT exec_id, started FROM inflight WHERE mig='race'").fetchall())
    assert all(started[e] <= close_at for e in admitted)


# ---------------------------------------------------------------------------
# A writer thread for the whole migration
# ---------------------------------------------------------------------------

def test_concurrent_writer_throughout_a_migration(api, tmp_path):
    src = _source(api)
    j = Journal(tmp_path / "journal.sqlite")
    c = Controller(api, j, tmp_path / "blobs")
    j.create("w", src, "cpu")
    assert c.route("w", write_program(0))["runtime"] == src      # acknowledged before any close
    w = Writer(c, "w")
    w.start()
    time.sleep(0.05)
    res = c.migrate("w", src, "cpu")
    w.finish()
    assert res["phase"] == "DONE", res
    assert not w.errors(), w.errors()[:3]
    acks = [(0, src, 1)] + [(seq, rt, n) for _, seq, rt, n in w.acks()]
    before = [a for a in acks if a[1] == src]
    after = [a for a in acks if a[1] == res["dest"]]
    assert len(before) >= 2 and after and w.refused(), (len(before), len(after), len(w.refused()))
    assert len(before) + len(after) == len(acks)                        # nothing ran anywhere else
    # Before the close the source counted them; after the commit the
    # destination continues from the captured count.
    assert [n for _, _, n in acks] == list(range(1, len(acks) + 1))
    final = api.witness(res["dest"], STATE)
    _assert_no_write_lost(final, [seq for seq, _, _ in acks], w.refused())
    assert res["commit_boundary"]["verdicts"]["pre_commit"]["equal"]
    assert res["admission"]["state"] == "open" and res["admission"]["refused"] == len(w.refused())


@pytest.mark.parametrize("phase,after_journal", [("ADMISSION_CLOSED", True), ("CAPTURED", True),
                                                  ("ADMISSION_CLOSED", False)])
def test_writer_across_a_controller_crash_and_restart(api, tmp_path, phase, after_journal):
    """The controller dies inside (or, for the before-journal kill, just
    outside) the gate while the writer keeps going. While it is down the gate
    holds exactly as the journal left it: closed after the ADMISSION_CLOSED or
    CAPTURED write (refused), open if the kill came before the close
    (accepted, and then drained into the capsule by the resumed controller)."""
    src = _source(api)
    j = Journal(tmp_path / "journal.sqlite")
    c = Controller(api, j, tmp_path / "blobs")
    j.create("wc", src, "cpu")
    w = Writer(c, "wc")
    w.start()
    time.sleep(0.05)
    with pytest.raises(ControllerKilled):
        Controller(api, Journal(j.path), tmp_path / "blobs", crash=CrashPoint(phase, after_journal=after_journal),
                   kill=_kill).migrate("wc", src, "cpu")
    gate = j.get("wc")["admission"]
    down = []
    for seq in range(100_000, 100_003):                  # explicit attempts while no controller runs
        try:
            down.append(("ack", seq, c.route("wc", write_program(seq))["runtime"]))
        except AdmissionClosed:
            down.append(("refused", seq))
    time.sleep(0.1)
    if after_journal:
        assert gate == "closed" and all(d[0] == "refused" for d in down)
    else:
        assert gate == "open" and all(d[0] == "ack" and d[2] == src for d in down)
    res = Controller(api, j, tmp_path / "blobs").migrate("wc", src, "cpu", takeover=True)
    w.finish()
    assert res["phase"] == "DONE", res
    assert not w.errors(), w.errors()[:3]
    acked = sorted([(seq, rt) for _, seq, rt, _ in w.acks()] + [(d[1], d[2]) for d in down if d[0] == "ack"],
                   key=lambda a: a[0])
    refused = w.refused() + [d[1] for d in down if d[0] == "refused"]
    assert {rt for _, rt in acked} <= {src, res["dest"]}
    final = api.witness(res["dest"], STATE)
    # The writer thread and the explicit attempts interleave; each write is
    # acknowledged only after it ran, so the file order is the ack order,
    # which for one writer plus sequential attempts is the sequence order
    # within each runtime. Compare as sets plus the counter.
    assert sorted(final["log"]) == sorted(seq for seq, _ in acked)
    assert not set(refused) & set(final["log"])
    assert final["counter"] == len(acked)


def test_aborted_migration_reopens_admission_on_the_source(api, tmp_path):
    src = _source(api)
    j = Journal(tmp_path / "journal.sqlite")
    c = Controller(api, j, tmp_path / "blobs")
    j.create("ab", src, "cpu")
    w = Writer(c, "ab")
    w.start()
    time.sleep(0.05)
    res = c.migrate("ab", src, "cpu", fault="bad_capsule")
    w.finish()
    assert res["phase"] == "ABORTED" and res["reason"] == "checkpoint_transfer_failed", res
    assert not w.errors(), w.errors()[:3]
    row = j.get("ab")
    assert row["admission"] == "open" and row["authoritative"] == src
    assert res["admission"]["closed_during_migration"] and w.refused()
    assert {rt for _, _, rt, _ in w.acks()} == {src}                    # before the close AND after the abort
    later = c.route("ab", write_program(10**6))
    assert later["runtime"] == src
    final = api.witness(src, STATE)
    _assert_no_write_lost(final, [seq for _, seq, _, _ in w.acks()] + [10**6], w.refused())
    assert all(p["id"] != row["dest_pid"] for p in api.list_projects())     # destination deleted


# ---------------------------------------------------------------------------
# The drain
# ---------------------------------------------------------------------------

def _slow_write_migration(tmp_path, monkeypatch=None, skip_drain=False):
    """One routed write admitted BEFORE the close and held in transit.

    With the drain it is released as soon as the gate closes: the capture
    must wait for it. The control removes the drain and holds the write until
    the capture has been journaled, which is what an unsynchronised route
    allows: the write is then acknowledged by the source after the cut."""
    j = Journal(tmp_path / "journal.sqlite")
    watch = Journal(j.path)

    def released():
        row = watch.get("slow") or {}
        if skip_drain:
            return ctl.PHASES.index(row.get("phase", "REQUESTED")) >= ctl.PHASES.index("CAPTURED") \
                if row.get("phase") in ctl.PHASES else True
        return row.get("admission") == "closed" or row.get("phase") not in ctl.PHASES
    api = SerialApi(workdir=tmp_path / "kernels", delay_marker="SLOW-IN-TRANSIT", release_when=released)
    try:
        src = _source(api)
        c = Controller(api, j, tmp_path / "blobs")
        j.create("slow", src, "cpu")
        if skip_drain:
            def barrier_only(self, mig, source_pid):
                self.api.witness(source_pid, ctl.barrier_program())
                return {"waited_on": 0}
            monkeypatch.setattr(Controller, "_drain", barrier_only)
        out = {}

        def slow():
            out["ack"] = c.route("slow", write_program(1, marker="SLOW-IN-TRANSIT"))
        t = threading.Thread(target=slow)
        t.start()
        deadline = time.time() + 5
        while not j.unfinished("slow", src) and time.time() < deadline:
            time.sleep(0.005)
        assert j.unfinished("slow", src), "the slow write never registered"
        res = c.migrate("slow", src, "cpu")
        t.join(timeout=30)
        final = api.witness(res["dest"], STATE) if res["phase"] == "DONE" else None
        return res, out.get("ack"), final, src
    finally:
        api.close()


def test_drain_waits_for_an_execution_admitted_before_the_close(tmp_path):
    res, ack, final, src = _slow_write_migration(tmp_path)
    assert res["phase"] == "DONE", res
    assert ack["runtime"] == src and ack["result"]["counter"] == 1          # acknowledged by the source ...
    assert final["counter"] == 1 and final["log"] == [1]                    # ... and carried to the destination
    drain = res["admission"]["drains"][0]
    assert drain["waited_on"] == 1


def test_without_the_drain_the_in_transit_write_is_lost(tmp_path, monkeypatch):
    """The control for the test above: with the drain removed, the same
    admitted write is acknowledged by the source after the cut and the
    destination commits without it. So the test above measures the drain."""
    res, ack, final, src = _slow_write_migration(tmp_path, monkeypatch, skip_drain=True)
    assert res["phase"] == "DONE", res
    assert ack["runtime"] == src and ack["result"]["counter"] == 1
    assert final["counter"] == 0 and final["log"] == []


def test_dead_owner_counts_as_finished_after_the_barrier(api, tmp_path):
    src = _source(api)
    j = Journal(tmp_path / "journal.sqlite")
    j.create("dead", src, "cpu")
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    ghost = uuid.uuid4().hex
    j.conn().execute("INSERT INTO inflight(exec_id, mig, runtime, owner, pid, started) VALUES (?,?,?,?,?,?)",
                     (ghost, "dead", src, f"{socket.gethostname()}:{p.pid}:0", p.pid, time.time()))
    res = Controller(api, j, tmp_path / "blobs", drain_timeout=5).migrate("dead", src, "cpu")
    assert res["phase"] == "DONE", res
    assert res["admission"]["drains"][0]["dead_owner_reaped"] == [ghost]
    assert j.conn().execute("SELECT finished FROM inflight WHERE exec_id=?", (ghost,)).fetchone()[0] is not None


def test_drain_timeout_aborts_and_reopens_on_the_source(api, tmp_path):
    src = _source(api)
    j = Journal(tmp_path / "journal.sqlite")
    j.create("stuck", src, "cpu")
    # A live owner (this process) whose execution never finishes.
    j.conn().execute("INSERT INTO inflight(exec_id, mig, runtime, owner, pid, started) VALUES (?,?,?,?,?,?)",
                     (uuid.uuid4().hex, "stuck", src, f"{socket.gethostname()}:{os.getpid()}:0", os.getpid(),
                      time.time()))
    c = Controller(api, j, tmp_path / "blobs", drain_timeout=0.5)
    res = c.migrate("stuck", src, "cpu")
    assert res["phase"] == "ABORTED" and res["reason"] == "drain_timeout" and res["failed_at"] == "ADMISSION_CLOSED"
    row = j.get("stuck")
    assert row["admission"] == "open" and row["authoritative"] == src and row["capsule_path"] is None
    assert c.route("stuck", write_program(5))["runtime"] == src


def test_a_duplicate_controller_does_not_reopen_the_gate(api, tmp_path):
    src = _source(api)
    j = Journal(tmp_path / "journal.sqlite")
    with pytest.raises(ControllerKilled):
        Controller(api, Journal(j.path), tmp_path / "blobs", crash=CrashPoint("CAPTURED"), kill=_kill).migrate(
            "dup", src, "cpu")
    dup = Controller(api, j, tmp_path / "blobs", join_timeout=0.4).migrate("dup", src, "cpu")   # lease held
    assert dup.get("joined") and dup.get("timeout")
    assert j.get("dup")["admission"] == "closed"
    with pytest.raises(AdmissionClosed):
        Controller(api, j, tmp_path / "blobs").execute_routed("dup", write_program(1))
    res = Controller(api, j, tmp_path / "blobs").migrate("dup", src, "cpu", takeover=True)
    assert res["phase"] == "DONE" and j.get("dup")["admission"] == "open"


def test_events_record_every_refusal(tmp_path):
    j = Journal(tmp_path / "journal.sqlite")
    j.create("ev", "s", "cpu")
    j.advance("ev", "ADMISSION_CLOSED")
    for _ in range(3):
        with pytest.raises(AdmissionClosed):
            j.admit("ev", "h:1:1", 1)
    assert j.inflight_summary("ev") == {"admitted": 0, "finished": 0, "refused": 3}
    notes = [e["note"] for e in j.events("ev") if e["note"].startswith("admission refused")]
    assert len(notes) == 3 and json.loads(j.events("ev")[1]["note"])["admission"] == "closed"


# ---------------------------------------------------------------------------
# The gate belongs to the runtime, not to the migration id (second review)
# ---------------------------------------------------------------------------

def test_a_route_through_an_earlier_hop_is_refused_while_the_next_hop_is_gated(api, tmp_path):
    """The reviewer's chain_bypass.py. hop1 src -> d1 is DONE; hop2 d1 -> d2
    is killed at CAPTURED. Old code: `route('hop2')` was refused but
    `route('hop1')` was ACCEPTED on d1 (acknowledged counter 2), and hop2
    committed counter 1: the acknowledged write was lost. Now the route
    through hop1 resolves d1, finds hop2's closed gate on it, and refuses;
    after hop2 commits it follows the chain to d2."""
    src = _source(api)
    j = Journal(tmp_path / "journal.sqlite")
    r1 = Controller(api, j, tmp_path / "blobs").migrate("hop1", src, "cpu")
    assert r1["phase"] == "DONE", r1
    d1 = r1["dest"]
    c = Controller(api, j, tmp_path / "blobs")
    first = c.route("hop1", write_program(1))
    assert first["runtime"] == d1 and first["result"]["counter"] == 1 and first["chain"] == ["hop1"]
    with pytest.raises(ControllerKilled):
        Controller(api, Journal(j.path), tmp_path / "blobs", crash=CrashPoint("CAPTURED"), kill=_kill).migrate(
            "hop2", d1, "cpu")
    for via in ("hop2", "hop1"):
        with pytest.raises(AdmissionClosed) as e:
            c.route(via, write_program(2))
        assert e.value.mig == "hop2" and e.value.via == via and e.value.phase == "CAPTURED"
    assert api.witness(d1, STATE)["counter"] == 1                     # nothing ran on d1
    assert j.authority("hop1") == {"runtime": d1, "chain": ["hop1"], "closed_by": "hop2", "phase": "DONE"}
    r2 = Controller(api, j, tmp_path / "blobs").migrate("hop2", d1, "cpu", takeover=True)
    assert r2["phase"] == "DONE", r2
    d2 = r2["dest"]
    assert api.witness(d2, STATE)["counter"] == 1                     # every acknowledged write committed
    after = c.route("hop1", write_program(3))                          # the earlier hop's id follows the chain
    assert after["runtime"] == d2 and after["chain"] == ["hop1", "hop2"] and after["result"]["counter"] == 2
    _assert_no_write_lost(api.witness(d2, STATE), [1, 3], [2])
    # The refusals are recorded against the gate that refused them.
    assert r2["admission"]["refused"] == 2
    notes = [e["note"] for e in j.events("hop2") if e["note"].startswith("admission refused")]
    assert any("via hop1" in n for n in notes)


def test_a_second_id_for_the_same_source_is_refused_while_the_gate_is_closed(api, tmp_path):
    """The reviewer's same_source_bypass.py. Migration A is killed at CAPTURED
    and a row B is created for the same source (a retry under a fresh id).
    Old code: `route('B')` was acknowledged on the source (counter 1) and A
    committed counter 0. Now B's route resolves the source, finds A's gate,
    and refuses; after A commits, B's route follows A to its destination."""
    src = _source(api)
    j = Journal(tmp_path / "journal.sqlite")
    with pytest.raises(ControllerKilled):
        Controller(api, Journal(j.path), tmp_path / "blobs", crash=CrashPoint("CAPTURED"), kill=_kill).migrate(
            "A", src, "cpu")
    j.create("B", src, "cpu")
    c = Controller(api, j, tmp_path / "blobs")
    with pytest.raises(AdmissionClosed) as e:
        c.route("B", write_program(1))
    assert e.value.mig == "A" and e.value.via == "B"
    assert api.witness(src, STATE)["counter"] == 0
    res = c.migrate("A", src, "cpu", takeover=True)
    assert res["phase"] == "DONE", res
    assert api.witness(res["dest"], STATE)["counter"] == 0
    later = c.route("B", write_program(2))
    assert later["runtime"] == res["dest"] and later["chain"] == ["B", "A"]
    _assert_no_write_lost(api.witness(res["dest"], STATE), [2], [1])


def test_one_outgoing_migration_per_runtime(api, tmp_path):
    """`Journal.start` refuses, in the lease transaction, a migration whose
    source is busy (another migration of it in progress), superseded (already
    migrated away) or not yet authoritative (another migration's uncommitted
    destination). Each refusal creates nothing and leaves authority where it
    was; the in-progress migration is unaffected."""
    src = _source(api)
    j = Journal(tmp_path / "journal.sqlite")
    with pytest.raises(ControllerKilled):
        Controller(api, Journal(j.path), tmp_path / "blobs", crash=CrashPoint("DEST_RESTORED"), kill=_kill).migrate(
            "A", src, "cpu")
    dest_a = j.get("A")["dest_pid"]
    projects = {p["id"] for p in api.list_projects()}
    c = Controller(api, j, tmp_path / "blobs")
    busy = c.migrate("B", src, "cpu")
    assert (busy["phase"], busy["reason"], busy["failed_at"], busy["conflict"]) == ("ABORTED", "source_busy",
                                                                                   "REQUESTED", "A")
    early = c.migrate("E", dest_a, "cpu")
    assert (early["phase"], early["reason"], early["conflict"]) == ("ABORTED", "source_not_authoritative", "A")
    assert {p["id"] for p in api.list_projects()} == projects            # nothing was created
    assert j.get("B")["authoritative"] == src and j.get("B")["admission"] == "open"
    assert j.get("A")["phase"] == "DEST_RESTORED" and j.get("A")["admission"] == "closed"
    with pytest.raises(AdmissionClosed):                                 # B's row routes into A's gate
        c.route("B", write_program(1))
    res = c.migrate("A", src, "cpu", takeover=True)
    assert res["phase"] == "DONE", res
    late = c.migrate("C", src, "cpu")
    assert (late["phase"], late["reason"], late["conflict"]) == ("ABORTED", "source_superseded", "A")
    # A resumed migration is never refused by its own row, and a duplicate
    # request for an id that is already terminal is a no-op.
    assert c.migrate("A", src, "cpu")["noop"]


def test_unclaimed_rows_do_not_block_and_two_starters_cannot_both_win(tmp_path):
    """Rows created only for routing (never claimed) are not competitors; of
    two controllers starting different ids for one source, exactly one gets
    the lease (the check and the claim are one transaction)."""
    j = Journal(tmp_path / "journal.sqlite")
    for m in ("r1", "r2", "r3"):
        j.create(m, "S", "cpu")
    assert j.start("r1", "owner-1") == (True, None)
    ok, conflict = j.start("r2", "owner-2")
    assert not ok and conflict["id"] == "r1" and conflict["reason"] == "source_busy"
    results, go = {}, threading.Event()

    def starter(m):
        jj = Journal(j.path)
        go.wait()
        results[m] = jj.start(m, "owner-" + m)
    for m in ("x", "y"):
        j.create(m, "T", "cpu")
    ts = [threading.Thread(target=starter, args=(m,)) for m in ("x", "y")]
    for t in ts:
        t.start()
    go.set()
    for t in ts:
        t.join()
    assert sorted(ok for ok, _ in results.values()) == [False, True]


def test_authority_resolution_follows_a_lineage(tmp_path):
    """The resolution itself, on a journal alone: a three-hop lineage
    a -> b -> c -> d, routed through every id. Aborted hops are ignored, an
    open hop in progress leaves authority where it is, a gated hop refuses."""
    j = Journal(tmp_path / "journal.sqlite")
    for mig, s, d in (("h1", "a", "b"), ("h2", "b", "c"), ("x", "c", "zz"), ("h3", "c", "d")):
        j.create(mig, s, "cpu")
        j.conn().execute("UPDATE migrations SET dest_pid=? WHERE id=?", (d, mig))
    for mig, d in (("h1", "b"), ("h2", "c")):
        j.advance(mig, "COMMITTED", authoritative=d)
        j.advance(mig, "DONE")
    j.advance("x", ctl.ABORTED)                                         # an aborted hop out of c
    assert {m: j.authority(m)["runtime"] for m in ("h1", "h2", "h3")} == {"h1": "c", "h2": "c", "h3": "c"}
    assert j.authority("h1")["chain"] == ["h1", "h2"]
    j.advance("h3", "PREFLIGHT")                                        # open: authority stays on c
    assert j.authority("h1")["runtime"] == "c" and j.authority("h1")["closed_by"] is None
    j.advance("h3", "ADMISSION_CLOSED")
    assert all(j.authority(m)["closed_by"] == "h3" for m in ("h1", "h2", "h3", "x"))
    j.advance("h3", "COMMITTED", authoritative="d")
    assert j.authority("h1") == {"runtime": "d", "chain": ["h1", "h2", "h3"], "closed_by": None, "phase": "DONE"}


def test_a_writer_through_the_first_id_across_two_hops(api, tmp_path):
    """The lifecycle WS-C runs (src -> d1 -> d2), with one writer routing
    through the FIRST hop's id for the whole time. Every acknowledged write is
    on d2 at the end, in order; every refused one is absent; each hop's gate
    refused while it was closed; writes after the second commit land on d2."""
    src = _source(api)
    j = Journal(tmp_path / "journal.sqlite")
    c = Controller(api, j, tmp_path / "blobs")
    j.create("L1", src, "cpu")
    w = Writer(c, "L1")
    w.start()
    time.sleep(0.05)
    r1 = c.migrate("L1", src, "cpu")
    assert r1["phase"] == "DONE", r1
    time.sleep(0.05)
    r2 = Controller(api, j, tmp_path / "blobs").migrate("L2", r1["dest"], "cpu")
    assert r2["phase"] == "DONE", r2
    w.finish()
    assert not w.errors(), w.errors()[:3]
    runtimes = [rt for _, _, rt, _ in w.acks()]
    assert set(runtimes) <= {src, r1["dest"], r2["dest"]} and runtimes[-1] == r2["dest"]
    assert runtimes == sorted(runtimes, key=[src, r1["dest"], r2["dest"]].index)   # never back to an old runtime
    assert r1["admission"]["refused"] > 0 and r2["admission"]["refused"] > 0
    assert [n for _, _, _, n in w.acks()] == list(range(1, len(w.acks()) + 1))
    _assert_no_write_lost(api.witness(r2["dest"], STATE), [seq for _, seq, _, _ in w.acks()], w.refused())
