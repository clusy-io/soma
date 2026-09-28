"""Transactional, crash-recoverable runtime handoff.

WHY A SEPARATE CONTROLLER. The platform's switch is serialize -> destroy ->
reboot -> replay: the source is destroyed before the destination has been
exercised, and a controller crash between "destination ready" and "switch
committed" has no journal to recover from. This controller keeps the
platform API as the provider (create project, execute, delete) and owns the
protocol above it. It is a Python process, which is what makes crash injection
tractable: `os._exit` at a chosen line is a faithful model of a dead
controller, and a restart with the same migration id is the recovery path.

THE PROTOCOL, as a phase machine persisted after every transition:

  REQUESTED         journal row exists; authoritative runtime = source
  DEST_PREPARED     destination project created and its sandbox warm
  PREFLIGHT         the destination has DESCRIBED ITSELF (python, dill, cuda,
                    installed distributions), the portability manifest has
                    been run on the source against that description, and the
                    policy admitted the switch; the report is journaled
  ADMISSION_CLOSED  the admission gate is closed in the same journal write:
                    routed work is refused from here until COMMITTED/ABORTED.
                    Before the capture, every execution admitted earlier is
                    drained and a barrier execution runs on the source
  CAPTURED          capsule bytes on disk (namespace dump with the gradients +
                    view manifest + RNG envelope + state fingerprint +
                    validation expectation + workspace tar), sha256
                    recorded; names the preflight rejected are left out, and
                    the source still holds them. The capture runs NO user
                    model code on the source (see VALIDATION below)
  DEST_RESTORED     capsule verified (size + sha256), loaded on the destination
                    (CUDA storages remapped where the destination has no such
                    device), every recorded view repaired, every gradient
                    reattached, the RNG streams set, and the destination's
                    state fingerprint equal to the cut's
  DEST_VALIDATED    the transition oracles pass on the destination, and the
                    destination's state and workspace are identical before
                    and after them (they run in a forked child of the
                    kernel, or on an in-process copy where no fork can
                    isolate them; any difference is `validation_side_effect`)
  COMMITTED         the destination's fingerprint is taken again and still
                    equals the cut's; then, in ONE journal write, authoritative
                    runtime = destination and admission reopens; source FENCED
  SOURCE_RELEASED   source project deleted (or paused if delete is refused)
  DONE
  ABORTED           any failure before COMMITTED: destination deleted, source
                    untouched, and, in ONE journal write, authoritative runtime
                    = source and admission reopened there

WHY THE DESTINATION IS PREPARED BEFORE THE CAPTURE. The first version captured
first and built the destination description from the SOURCE's own Python and
dill versions, so the "preflight" compared the source with itself and could
never fail; it also never consulted `admissible`, the package section or the
process effects. A destination can only describe itself once it exists, so it
is created first and asked. The price is a destination created for a switch
the preflight may then refuse; the abort path deletes it like any other.

THE PREFLIGHT POLICY:
  gate failed or imported distribution missing on the destination
      -> ABORTED at PREFLIGHT, reason `preflight_gate_failed:<gate>` or
         `preflight_missing_distribution:<dist>`. This controller has no
         repair path, so every missing distribution blocks.
  rejected names (would poison the all-or-nothing load)
      -> excluded from the capsule and reported; not blocking. The exclusion
         happens INSIDE the capture's stash/finally, so the source keeps them.
  a recorded view the storage adapter already predicts it cannot repair
      -> ABORTED at PREFLIGHT, reason `preflight_views_unsupported:<paths>`,
         before the gate closes and before anything is captured. The adapter
         reports these at capture (`predicted_failures`: dtype
         reinterpretation, an autograd view it cannot rebuild, an ndarray
         subclass, ...), and the restore would refuse them anyway with
         `views_not_restored`, but only after the whole capture and
         transfer. The preflight runs `collect_view_manifest` over a COPY of
         the names the capture would dump and removes every hidden base it
         injected in a `finally`, so judging the source leaves nothing in it.
         Views outside the contract (`unsupported`) are reported, not
         blocking.
  gradients the capsule does not carry
      -> reported as `grads_not_carried` and declared at the commit
         boundary; not blocking. Pickle drops every `.grad`. The capture
         carries the gradients of top-level tensors, parameters of top-level
         modules and optimizer slots, BY VALUE. A gradient on a tensor
         reachable only some other way (a module in a list, a tensor in an
         object's attributes) arrives as None, and a carried gradient that
         shares storage with another tensor (a view of a flat gradient
         buffer) arrives unshared; both are named, never compared as equal.
  unknown names and process effects
      -> reported; blocking only with `strict_unknown=True`, which also blocks
         when the effects could not be measured (no seed-time baseline).
         Strict mode is not expected to pass with the E11/E12 fixture: the
         fixture binds a list iterator (`it`, which the manifest cannot
         validate) and its own seeding and library lazy initialisation
         (torch adds to sys.path, numpy attaches `fft`) register as process
         effects. It exists to show the gate, not for a live cohort.

THE ADMISSION CONTRACT. A reviewer stopped the first version at CAPTURED, ran
a mutation through `execute_routed` (acknowledged, counter 1), resumed, and
the migration committed counter 0: an acknowledged write was lost, because the
route kept sending work to the source after the capture cut. The gate closes
that window, and its guarantee is stated here in full so it is not read as
more than it is:

  * The gate governs work submitted through this controller's route
    (`route` / `execute_routed`). In ONE `BEGIN IMMEDIATE` transaction the
    route reads the migration row, refuses with `AdmissionClosed` if the gate
    is closed, and otherwise registers the execution in the journal's
    `inflight` table against the runtime it read as authoritative. SQLite
    serialises that transaction against the close and against the authority
    flip, so every routed execution either registered before the close (and
    is drained before the capture) or sees the gate closed; after COMMITTED it
    is routed to the destination. An acknowledged routed write is therefore
    either in the capsule or on the runtime that is authoritative afterwards.
  * THE GATE BELONGS TO THE RUNTIME, NOT TO THE MIGRATION ID. A second
    review sent a write through a DIFFERENT migration id whose authority was
    the gated source: through the previous hop of a chain (hop1 src -> d1
    DONE, hop2 d1 -> d2 stopped at CAPTURED, a route via `hop1` ran on d1),
    and through a fresh id created for the same source. Both were admitted,
    acknowledged and then lost, because admission read only the row it was
    given. So the route resolves the runtime, in the same transaction, by
    walking the lineage: from the row's authoritative runtime R, look at
    every migration whose SOURCE is R. If any has its gate closed, refuse
    (the refusal names that migration). If one has COMMITTED, R is no longer
    authoritative: follow it to its destination and look again. Otherwise
    run on R. A route through any id of a chain therefore lands on the
    chain's current authority, and never on a runtime inside a gate.
  * One outgoing migration per runtime. `Journal.start` refuses, in the
    transaction that takes the lease, to START a migration whose source
    already has another migration in progress (`source_busy:<id>`), has
    already been migrated away from (`source_superseded:<id>`), or is the
    not-yet-committed destination of one (`source_not_authoritative:<id>`).
    The refused row is ABORTED with authority left on its source.
  * The drain waits (bounded by `drain_timeout`, else ABORTED with
    `drain_timeout` and the gate reopened on the source) until no execution
    admitted for this migration or runtime is unfinished; an execution whose
    owning process is dead counts as finished once the barrier has run. The
    barrier is one no-op execution on the source: a kernel runs executions one
    at a time, so its return proves everything submitted before it finished.
  * The gate is durable: it is a journal column, so a controller restarted
    while it is closed keeps refusing routed work until it resumes the
    migration to COMMITTED/DONE or ABORTED. A duplicate controller that finds
    the lease held only waits; it never reopens anything.
  * Direct API execution that bypasses the controller is OUTSIDE the
    contract, exactly as for any fencing-token scheme. After COMMITTED the
    kernel fence marker is a second guard, honoured by the programs that
    check it (`touch_program`); it does not instrument arbitrary code.

VALIDATION MUST NOT CHANGE THE COMMITTED WORKLOAD. The same review found the
validation's real optimizer step committed as if it were user work: 12/12 and
DONE, with every optimizer counter advanced 3 -> 4, the parameter bytes
changed and the application's own step counter still 3. A deepcopy did not
settle it: a second review put a forward hook on the model that stores
activations in a top-level dict, and the "copy" validation's forwards ran
the hook against the REAL dict (deepcopy copies functions by reference), so
the migration committed a changed dict with the boundary reporting equal.
The principle now: a transparent handoff runs NO user model code on the
SOURCE, and whatever validation runs user code on the DESTINATION is contained
or fails closed.

  * The source. The capture refreshes the validation expectation at the cut
    (after user training the seed's is stale) from STRUCTURE only: the
    parameter and buffer digest (a digest function defined in this program,
    the same algorithm the destination's `parameters` oracle uses; the first
    version called the fixture's helpers through `__main__`, which no capsule
    carries, so a runtime this controller had restored and the user had
    trained failed its next hop with `NameError: _digest_params`), the
    optimizer's step count, the scheduler's lr and last_epoch, the next
    loader indices (from a CLONE of the loader's generator), the plain
    values, and the digest of the commit-boundary fingerprint. No forward,
    hook, backward or step runs there, forked or not: a fork contains memory,
    not the workspace (a hook that appended to a file changed the source of
    an ABORTED switch), and none of those fields needs model code. So the
    forward output has no source reference, and the destination's forward
    row is DECLARED NOT RUN with the reason recorded in the expectation
    (`FORWARD_NOT_RUN`): recomputing a forward pass is an evaluation oracle,
    and the bytes are established by the parameter digest and the
    commit-boundary fingerprint. The validation record says so in its
    totals (for example 11 checks run, 1 declared not run), never 12/12.
  * The destination. Validation runs `oracles.verify(...,
    continuation="isolated")`: every oracle runs in a FORKED CHILD of the
    kernel, in place on the child's copy-on-write memory, and only the rows
    come back. Where no fork can isolate (CUDA already initialized in the
    kernel), the continuation step runs on ONE disposable in-process copy
    instead. Either way the validation program fingerprints the whole state
    AND the workspace immediately before and after the oracles, in the same
    execution, and any difference aborts the switch with
    `validation_side_effect:<fields>` (a fork does not contain the
    filesystem, and a copy does not contain code), with the source untouched
    and authority left there. PyTorch also refuses a backward pass in a child
    forked from a process that has already run one; a freshly restored
    destination never has, so the handoff's continuation step always runs,
    while a runtime re-validated after training (a recovery check after user
    work) gets its continuation row declared with that reason. The recovery
    check is the same non-mutating program, which is why it runs every check
    after a restart instead of filtering to a seven-check subset it believed
    idempotent.

THE COMMIT-BOUNDARY EQUIVALENCE. Non-mutating validation is a claim; the
fingerprint checks it. `_STATE_FP_SRC` computes, without writing anything:
parameter and buffer bytes, optimizer state and hyperparameters and which
parameter every slot and state key is bound to, scheduler and scaler state,
loader generators, the full python/numpy/torch RNG states (hash plus draws from
clones) and every named generator, CUDA RNG when CUDA is already initialized,
a digest of every plain value, a DEEP digest of every other captured name
(`objects`: containers of tensors and arrays, the instance state of objects
pickle rebuilds from their `__dict__`/slots, and the defaults and closure cells
of functions defined in `__main__`), identity groups, tensor
bytes/dtype/shape/class (sparse and quantized tensors by their components),
the gradients of top-level tensors, module parameters and optimizer slots,
`views_intact` from the storage adapter, and the workspace file hashes. The
capture records it at the cut (after the drain, inside the stash, immediately
before the dump) and AGAIN after the dump: a capture whose own work changed the
source (a reducer, a sampler) is `capture_mutated_source`. The destination
recomputes it after the restore AND again after validation, immediately before
COMMITTED. Any difference aborts with `commit_boundary_mismatch:<fields>`.
Four things are declared, recorded and never a reason to abort: CUDA RNG when
either end has no CUDA (`declared_drop` / `not_carried`), tensor DEVICE (the
bytes must still match), a value the fingerprint could not digest
(`unhashable`: the marker is compared, the value is not; the list names each
one), and gradients the capsule does not carry (`grads_not_carried`, below).
Pickle does not carry `.grad`. The capture carries, BY VALUE, the gradients
of top-level tensors, parameters of top-level modules and optimizer slots,
the restore reattaches them, and the `grads` field compares them, so a lost
or changed one of those aborts. Every other non-None gradient (on a tensor
reachable only through a container or an object's attributes) arrives as
None, and a carried gradient that shares storage with another tensor arrives
with its value but unshared; the fingerprint names both under
`grads_not_carried`, the preflight reports the same list, and no field
compares them as equal.

Every step is idempotent against the journal: a restarted controller reads the
phase and continues; a duplicate request for the same migration id joins the
existing row instead of starting a second migration.

Every program shipped into a kernel runs inside a private globals dict
(`_wrap`), so nothing of the harness is left in `__main__` for the next dump to
find, and the capture can show the source's bindings are unchanged by it.

WHAT IS PROTOTYPED AND WHAT IS NOT. The capsule travels through the API's
execute channel as base64 rather than through the vault; that is adequate for
the fixture sizes here and irrelevant to the protocol. The workspace is carried
as a tar inside the capsule.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
import socket
import sqlite3
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MARK = "__CLUSY_HANDOFF__"

PHASES = ["REQUESTED", "DEST_PREPARED", "PREFLIGHT", "ADMISSION_CLOSED", "CAPTURED", "DEST_RESTORED",
          "DEST_VALIDATED", "COMMITTED", "SOURCE_RELEASED", "DONE"]
ABORTED = "ABORTED"
#: A crash point that is not a phase: immediately after the provider returns
#: the new destination's id and BEFORE the controller journals it. This is the
#: one window the journal cannot cover; `reclaim()` covers it by name.
DEST_CREATED = "DEST_CREATED"
#: The phases during which the admission gate is closed. The gate is derived
#: from the phase in the SAME journal write that records it (see
#: `Journal.advance`), so a phase and its gate state can never disagree, and
#: the route also refuses on the phase itself in case a row was written by
#: something that set one without the other.
ADMISSION_CLOSED_PHASES = frozenset({"ADMISSION_CLOSED", "CAPTURED", "DEST_RESTORED", "DEST_VALIDATED"})
#: Phases after which the migration's SOURCE is no longer authoritative: the
#: route follows such a migration to its destination (see THE ADMISSION
#: CONTRACT).
AUTHORITY_MOVED_PHASES = frozenset({"COMMITTED", "SOURCE_RELEASED", "DONE"})


def admission_for(phase: str) -> str:
    """The gate state a phase implies: closed from the close to the commit
    (or abort), open otherwise."""
    return "closed" if phase in ADMISSION_CLOSED_PHASES else "open"


class HandoffError(Exception):
    def __init__(self, reason: str, phase: str, detail: str = ""):
        super().__init__(f"{phase}: {reason} {detail}".strip())
        self.reason, self.phase, self.detail = reason, phase, detail


class AdmissionClosed(HandoffError):
    """A routed execution refused because the migration's admission gate is
    closed (between ADMISSION_CLOSED and COMMITTED/ABORTED). Nothing ran: the
    refusal is decided before any execution is sent, so the caller may retry
    once the migration has committed (the retry then runs on the destination)
    or aborted (on the source)."""

    def __init__(self, mig: str, phase: str, via: str | None = None):
        route = f" (routed through {via})" if via and via != mig else ""
        super().__init__("admission_closed", phase,
                         f"migration {mig} is between the capture cut and its commit{route}")
        #: The migration whose gate refused, and the id the route was sent
        #: through (they differ when the gate belongs to a later hop of the
        #: lineage, or to another migration of the same source).
        self.mig, self.via = mig, via or mig


class FencedError(Exception):
    pass


class ControllerKilled(BaseException):
    """Raised by a crash point when the controller runs IN PROCESS (the local
    dry run), where `os._exit` would also kill the kernels the test double
    owns. A BaseException, so the migration's `except HandoffError` abort path
    does not see it: like a real kill, nothing is cleaned up, no lease is
    released, and the journal holds whatever was last written."""


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------

class Journal:
    """SQLite, WAL mode, one row per migration plus an event log and the
    admission gate's in-flight table. Every phase transition is a committed
    write BEFORE the next side effect, except where a test deliberately kills
    the controller between the side effect and the write to model a lost
    acknowledgement.

    WHY A CONNECTION PER THREAD. The route is called from user threads while
    the controller runs in its own, and the route's check-and-register must be
    one `BEGIN IMMEDIATE` transaction. Two threads sharing one connection would
    interleave their statements into each other's transactions, which is
    exactly the race the transaction exists to prevent. `self.db` stays the
    creating thread's connection (callers close it); any other thread gets its
    own from `conn()`."""

    def __init__(self, path: Path):
        self.path = path
        self._owner_thread = threading.get_ident()
        self._tls = threading.local()
        self.db = self._connect()
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS migrations(
            id TEXT PRIMARY KEY, phase TEXT NOT NULL, authoritative TEXT NOT NULL,
            source_pid TEXT NOT NULL, dest_pid TEXT, dest_profile TEXT NOT NULL,
            capsule_path TEXT, capsule_sha TEXT, capsule_bytes INTEGER,
            fence_token TEXT, abort_reason TEXT, pending_dest_pid TEXT,
            lease_owner TEXT, lease_at REAL, preflight_path TEXT, preflight_sha TEXT,
            admission TEXT NOT NULL DEFAULT 'open',
            fp_source TEXT, fp_restored TEXT, fp_commit TEXT, boundary TEXT,
            created_at REAL NOT NULL, updated_at REAL NOT NULL)""")
        for col in ("pending_dest_pid TEXT", "lease_owner TEXT", "lease_at REAL",
                    "preflight_path TEXT", "preflight_sha TEXT", "admission TEXT NOT NULL DEFAULT 'open'",
                    "fp_source TEXT", "fp_restored TEXT", "fp_commit TEXT", "boundary TEXT"):
            try:
                self.db.execute(f"ALTER TABLE migrations ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass
        self.db.execute("""CREATE TABLE IF NOT EXISTS events(
            seq INTEGER PRIMARY KEY AUTOINCREMENT, mig TEXT, at REAL, phase TEXT, note TEXT)""")
        # One row per routed execution: registered before it is sent, marked
        # finished after it returns (or raises). The drain reads this table.
        self.db.execute("""CREATE TABLE IF NOT EXISTS inflight(
            exec_id TEXT PRIMARY KEY, mig TEXT NOT NULL, runtime TEXT NOT NULL, owner TEXT NOT NULL,
            pid INTEGER, started REAL NOT NULL, finished REAL)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS inflight_open ON inflight(mig, runtime, finished)")

    def _connect(self) -> sqlite3.Connection:
        # A generous busy timeout: the transactions here are a handful of
        # statements, so waiting is always better than a spurious "locked".
        c = sqlite3.connect(str(self.path), isolation_level=None, timeout=60.0)
        c.execute("PRAGMA busy_timeout=60000")
        return c

    def conn(self) -> sqlite3.Connection:
        """This thread's connection (see the class doc)."""
        if threading.get_ident() == self._owner_thread:
            return self.db
        c = getattr(self._tls, "db", None)
        if c is None:
            c = self._tls.db = self._connect()
        return c

    @contextlib.contextmanager
    def tx(self):
        """One `BEGIN IMMEDIATE` transaction: the write lock is taken at BEGIN,
        so everything read inside is read under it."""
        c = self.conn()
        c.execute("BEGIN IMMEDIATE")
        try:
            yield c
        except BaseException:
            c.execute("ROLLBACK")
            raise
        else:
            c.execute("COMMIT")

    @staticmethod
    def _row(cur) -> dict | None:
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip([c[0] for c in cur.description], row))

    def get(self, mig: str) -> dict | None:
        return self._row(self.conn().execute("SELECT * FROM migrations WHERE id=?", (mig,)))

    def create(self, mig: str, source_pid: str, dest_profile: str) -> dict:
        now = time.time()
        # INSERT OR IGNORE makes a duplicate request join the existing row.
        self.conn().execute("""INSERT OR IGNORE INTO migrations
            (id, phase, authoritative, source_pid, dest_profile, fence_token, admission, created_at, updated_at)
            VALUES (?, 'REQUESTED', ?, ?, ?, ?, 'open', ?, ?)""",
            (mig, source_pid, source_pid, dest_profile, f"fence-{mig}", now, now))
        self.event(mig, "REQUESTED", "created or joined")
        return self.get(mig)

    def claim(self, mig: str, owner: str, takeover: bool = False) -> bool:
        """Compare-and-set a lease. A duplicate request finds the lease held and
        must JOIN rather than act; a restarted controller takes over, which is
        the supervisor's fencing decision and is passed in explicitly."""
        if takeover:
            self.conn().execute("UPDATE migrations SET lease_owner=?, lease_at=? WHERE id=?", (owner, time.time(), mig))
            return True
        cur = self.conn().execute(
            "UPDATE migrations SET lease_owner=?, lease_at=? WHERE id=? AND (lease_owner IS NULL OR lease_owner=?)",
            (owner, time.time(), mig, owner))
        return cur.rowcount == 1

    def start(self, mig: str, owner: str, takeover: bool = False) -> tuple[bool, dict | None]:
        """Take the lease (as `claim`), refusing to START a migration whose
        source runtime is not free, in ONE `BEGIN IMMEDIATE` transaction so
        two controllers starting different ids for one source cannot both
        pass. Returns (claimed, conflict): `conflict` is the other row and a
        reason when the start is refused, else None.

        Only a row still at REQUESTED is checked: a row past it already
        passed this check when it started (and a row created by `create` for
        routing, which no controller has claimed, is not a competitor). The
        source must not have another migration in progress, must not have
        been migrated away from, and must not be the uncommitted destination
        of another migration. One outgoing migration per runtime is what lets
        the route resolve authority by runtime (see THE ADMISSION CONTRACT)."""
        with self.tx() as c:
            row = self._row(c.execute("SELECT * FROM migrations WHERE id=?", (mig,)))
            if row is None:
                raise HandoffError("unknown_migration", "start", mig)
            if row["phase"] == "REQUESTED":
                src = row["source_pid"]
                cur = c.execute(
                    "SELECT id, phase, source_pid, dest_pid, authoritative, lease_owner FROM migrations "
                    "WHERE id != ? AND phase != ? AND (source_pid = ? OR dest_pid = ?) ORDER BY created_at",
                    (mig, ABORTED, src, src))
                for other in (dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()):
                    moved = other["phase"] in AUTHORITY_MOVED_PHASES
                    if other["source_pid"] == src:
                        if moved:
                            return False, {**other, "reason": "source_superseded"}
                        if other["phase"] != "REQUESTED" or other["lease_owner"] is not None:
                            return False, {**other, "reason": "source_busy"}
                        # An unclaimed REQUESTED row (created for routing,
                        # never started) is not a competitor.
                    elif not moved:
                        # `src` is another migration's destination that has
                        # not committed: it is not authoritative yet.
                        return False, {**other, "reason": "source_not_authoritative"}
            if takeover:
                c.execute("UPDATE migrations SET lease_owner=?, lease_at=? WHERE id=?", (owner, time.time(), mig))
                return True, None
            cur = c.execute(
                "UPDATE migrations SET lease_owner=?, lease_at=? WHERE id=? AND (lease_owner IS NULL OR lease_owner=?)",
                (owner, time.time(), mig, owner))
            return cur.rowcount == 1, None

    def release(self, mig: str, owner: str) -> None:
        self.conn().execute("UPDATE migrations SET lease_owner=NULL WHERE id=? AND lease_owner=?", (mig, owner))

    def advance(self, mig: str, phase: str, **cols) -> None:
        """Record a phase, and with it the admission gate that phase implies,
        in ONE transaction. Closing the gate IS the ADMISSION_CLOSED write;
        reopening it IS the COMMITTED (authority = destination) or ABORTED
        (authority = source) write. There is no moment at which the journal
        says one thing about the phase and another about the gate."""
        cols = {**cols, "admission": admission_for(phase)}
        sets = ", ".join(f"{k}=?" for k in cols)
        note = json.dumps({k: (v if not isinstance(v, str) or len(v) < 80 else v[:77] + '...') for k, v in cols.items()})
        with self.tx() as c:
            c.execute(f"UPDATE migrations SET phase=?, updated_at=?, {sets} WHERE id=?",
                      [phase, time.time(), *cols.values(), mig])
            c.execute("INSERT INTO events(mig, at, phase, note) VALUES (?,?,?,?)", (mig, time.time(), phase, note))

    def set_columns(self, mig: str, **cols) -> None:
        """Write evidence columns without changing the phase or the gate."""
        sets = ", ".join(f"{k}=?" for k in cols)
        self.conn().execute(f"UPDATE migrations SET {sets} WHERE id=?", [*cols.values(), mig])

    def event(self, mig: str, phase: str, note: str) -> None:
        self.conn().execute("INSERT INTO events(mig, at, phase, note) VALUES (?,?,?,?)", (mig, time.time(), phase, note))

    def events(self, mig: str) -> list[dict]:
        cur = self.conn().execute("SELECT seq, at, phase, note FROM events WHERE mig=? ORDER BY seq", (mig,))
        return [dict(zip(("seq", "at", "phase", "note"), r)) for r in cur.fetchall()]

    # -- the admission gate ------------------------------------------------------
    @staticmethod
    def _gate_closed(row: dict) -> bool:
        return row.get("admission") == "closed" or row["phase"] in ADMISSION_CLOSED_PHASES

    def _resolve(self, c: sqlite3.Connection, mig: str) -> tuple[dict, str, list[str], dict | None]:
        """Inside the caller's transaction: the runtime a route through `mig`
        must run on, the chain of migration ids followed to reach it, and the
        migration whose closed gate refuses it (None when admitted).

        Keyed on the RUNTIME (see THE ADMISSION CONTRACT): starting from the
        row's authoritative runtime R, every migration whose source is R is
        consulted. A closed gate refuses; a committed one means R is no longer
        authoritative, so the walk moves to its destination. Aborted
        migrations left authority on their source and are ignored."""
        row = self._row(c.execute("SELECT * FROM migrations WHERE id=?", (mig,)))
        if row is None:
            raise HandoffError("unknown_migration", "route", mig)
        if self._gate_closed(row):
            return row, row["authoritative"], [mig], row
        runtime, chain, seen = row["authoritative"], [mig], {row["authoritative"]}
        while True:
            cur = c.execute("SELECT * FROM migrations WHERE source_pid=? AND phase != ? ORDER BY created_at",
                            (runtime, ABORTED))
            out = [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]
            gate = next((o for o in out if self._gate_closed(o)), None)
            if gate is not None:
                return row, runtime, chain, gate
            moved = [o for o in out if o["phase"] in AUTHORITY_MOVED_PHASES and o["authoritative"] != runtime]
            if not moved:
                return row, runtime, chain, None
            if len(moved) > 1:
                raise HandoffError("authority_ambiguous", "route",
                                   f"runtime {runtime} was migrated away by {[o['id'] for o in moved]}")
            runtime = moved[0]["authoritative"]
            chain.append(moved[0]["id"])
            if runtime in seen:
                raise HandoffError("authority_cycle", "route", f"lineage {chain} returns to {runtime}")
            seen.add(runtime)

    def authority(self, mig: str) -> dict:
        """Where a route through `mig` would run now, read-only: the runtime,
        the chain followed, and the gating migration if the gate is closed."""
        with self.tx() as c:
            row, runtime, chain, gate = self._resolve(c, mig)
        return {"runtime": runtime, "chain": chain, "closed_by": None if gate is None else gate["id"],
                "phase": row["phase"]}

    def admit(self, mig: str, owner: str, pid: int) -> dict:
        """The route's check-and-register, in ONE `BEGIN IMMEDIATE`
        transaction: resolve the runtime by lineage (`_resolve`); if a gate
        is closed on it, record the refusal (against the migration that owns
        that gate) and raise `AdmissionClosed`; otherwise register the
        execution against the resolved runtime. Because the close and the
        authority flip are also single transactions, this read can never see
        a gate that is about to close with work it has not registered, and
        the drain, which keys on the runtime, sees every registration."""
        exec_id = uuid.uuid4().hex
        with self.tx() as c:
            row, runtime, chain, gate = self._resolve(c, mig)
            if gate is not None:
                via = "" if gate["id"] == mig else f" via {' -> '.join(chain)}"
                c.execute("INSERT INTO events(mig, at, phase, note) VALUES (?,?,?,?)",
                          (gate["id"], time.time(), gate["phase"],
                           f"admission refused: exec {exec_id} owner {owner}{via}"))
            else:
                c.execute("INSERT INTO inflight(exec_id, mig, runtime, owner, pid, started) VALUES (?,?,?,?,?,?)",
                          (exec_id, mig, runtime, owner, pid, time.time()))
        if gate is not None:
            raise AdmissionClosed(gate["id"], gate["phase"], via=mig)
        return {"exec_id": exec_id, "runtime": runtime, "phase": row["phase"], "chain": chain}

    def finish(self, exec_id: str) -> None:
        self.conn().execute("UPDATE inflight SET finished=? WHERE exec_id=? AND finished IS NULL",
                            (time.time(), exec_id))

    def unfinished(self, mig: str, runtime: str) -> list[dict]:
        """Executions admitted for this migration, or against this runtime,
        that have not finished. Keyed on the runtime too, so work routed
        against the source under another migration id is drained as well."""
        cur = self.conn().execute(
            "SELECT exec_id, mig, runtime, owner, pid, started FROM inflight "
            "WHERE (mig=? OR runtime=?) AND finished IS NULL ORDER BY started", (mig, runtime))
        return [dict(zip(("exec_id", "mig", "runtime", "owner", "pid", "started"), r)) for r in cur.fetchall()]

    def reap(self, exec_ids: list[str]) -> None:
        """Mark executions of dead owners finished, AFTER the barrier proved
        the kernel had run everything submitted before it."""
        c = self.conn()
        for e in exec_ids:
            c.execute("UPDATE inflight SET finished=? WHERE exec_id=? AND finished IS NULL", (time.time(), e))

    def inflight_summary(self, mig: str) -> dict:
        cur = self.conn().execute(
            "SELECT count(*), sum(finished IS NOT NULL) FROM inflight WHERE mig=?", (mig,))
        total, done = cur.fetchone()
        refused = self.conn().execute(
            "SELECT count(*) FROM events WHERE mig=? AND note LIKE 'admission refused:%'", (mig,)).fetchone()[0]
        return {"admitted": int(total or 0), "finished": int(done or 0), "refused": int(refused or 0)}


def _owner_dead(owner: str, pid: int | None) -> bool:
    """True only when the owner is a process on THIS host that no longer
    exists. An owner on another host, or one this process cannot signal, is
    treated as alive: the drain then waits for it (bounded by its timeout)
    rather than guessing."""
    host = (owner or "").split(":", 1)[0]
    if host != socket.gethostname() or not pid or pid == os.getpid():
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


# ---------------------------------------------------------------------------
# API client (the provider)
# ---------------------------------------------------------------------------

class Api:
    def __init__(self, base: str, key: str):
        self.base, self.key = base.rstrip("/"), key

    def _req(self, method: str, path: str, body: dict | None = None,
             timeout: float = 120.0) -> tuple[int, dict | list]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw, st = r.read().decode("utf-8", "replace"), r.status
        except urllib.error.HTTPError as e:
            raw, st = e.read().decode("utf-8", "replace"), e.code
        try:
            payload = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return st, {"_raw": raw[:2000]}
        # `GET /projects` answers with a JSON ARRAY. Wrapping it in a dict made
        # every listing look empty, which silently disabled `reclaim`: it saw
        # no projects, found nothing to delete, and reported success. A list is
        # returned as a list.
        if isinstance(payload, (dict, list)):
            return st, payload
        return st, {"_raw": payload}

    def create_project(self, name: str, profile: str) -> str:
        st, p = self._req("POST", "/projects", {"name": name, "sandboxType": "ml", "runtimeProfile": profile})
        if st != 201:
            raise HandoffError("provider_error", "create_project", f"HTTP {st} {json.dumps(p)[:200]}")
        return p["id"]

    def list_projects(self, timeout: float = 300.0) -> list[dict]:
        """Every project this credential owns. The route answers with an array."""
        st, p = self._req("GET", "/projects", timeout=timeout)
        if st != 200:
            raise HandoffError("provider_error", "list_projects", f"HTTP {st}")
        if isinstance(p, list):
            return p
        return p.get("projects") or p.get("data") or []

    def delete_project(self, pid: str) -> int:
        st, _ = self._req("DELETE", f"/projects/{pid}", timeout=300)
        return st

    def patch_profile(self, pid: str, profile: str, timeout: float = 1500.0) -> tuple[int, dict]:
        """The platform's in-place switch: PATCH runtimeProfile. Returns status
        and the response payload."""
        return self._req("PATCH", f"/projects/{pid}", {"runtimeProfile": profile}, timeout=timeout)

    def pause(self, pid: str) -> int:
        st, _ = self._req("POST", f"/projects/{pid}/sandbox/pause", {}, timeout=120)
        return st

    def execute(self, pid: str, code: str, timeout_ms: int = 600_000) -> tuple[int, str, dict]:
        st, p = self._req("POST", f"/projects/{pid}/sandbox/execute",
                          {"code": code, "timeout_ms": timeout_ms, "classify_intent": False},
                          timeout=timeout_ms / 1000 + 120)
        return st, (p.get("stdout") or ""), p

    def witness(self, pid: str, code: str, timeout_ms: int = 600_000) -> dict:
        """Execute and parse the single marker line the program prints."""
        st, out, p = self.execute(pid, code, timeout_ms)
        if st != 200:
            raise HandoffError("execute_failed", "execute", f"HTTP {st} {json.dumps(p)[:300]}")
        line = next((l for l in out.splitlines() if l.startswith(MARK)), None)
        if line is None:
            err = (p.get("error") or p.get("stderr") or out or "")[-600:]
            raise HandoffError("no_witness", "execute", f"kernel output: {err}")
        return json.loads(line[len(MARK):])


# ---------------------------------------------------------------------------
# Programs that run inside the sandboxes
# ---------------------------------------------------------------------------

FIXTURE_SRC = (ROOT / "experiments" / "fixture.py").read_text()
ORACLES_SRC = (ROOT / "experiments" / "oracles.py").read_text()
SHARING_SRC = (ROOT / "src" / "capsule" / "storage_sharing.py").read_text()
MANIFEST_SRC = (ROOT / "src" / "capsule" / "manifest.py").read_text()
REATTACH_SRC = (ROOT / "src" / "capsule" / "optimizer_reattach.py").read_text()

def _prelude(*, manifest: bool = True, fixture: bool = True, oracles: bool = True, isolation: bool = False) -> str:
    """The helper modules a kernel program needs, shipped as source text.

    WHY SELECTIVE. The execute route caps `code` at 200,000 characters and the
    full prelude is already about 137,000, most of it the portability manifest
    (only the seed, describe and preflight programs use it). The capture,
    restore and fingerprint programs load neither the manifest nor the fixture
    or oracles, which keeps them far from the cap as the storage adapter grows.
    `isolation` loads the oracles module WITHOUT the fixture, for its fork
    helpers only (`run_isolated`, `fork_unsafe_reason`; the fixture is
    imported only inside `verify`). The capture no longer uses it: it runs no
    model code on the source, forked or not. The default loads everything,
    as before."""
    if isolation and not oracles:
        return _prelude(manifest=manifest, fixture=fixture, oracles=False) + (
            ('_load_mod("experiments", "")\n' if not fixture else "")
            + f'_oracles = _load_mod("experiments.oracles", {ORACLES_SRC!r})\n')
    fixture = fixture or oracles       # the oracles import the fixture's digest
    parts = [f'''
import sys, os, json as _json, types as _types, base64 as _b64, hashlib as _hl, io as _io, tarfile as _tar, time as _time
def _load_mod(name, src):
    m = _types.ModuleType(name); m.__file__ = name + ".py"
    if "." in name:
        pkg = name.rsplit(".", 1)[0]
        if pkg not in sys.modules:
            p = _types.ModuleType(pkg); p.__path__ = []; sys.modules[pkg] = p
    sys.modules[name] = m
    exec(compile(src, name + ".py", "exec"), m.__dict__)
    return m
_load_mod("capsule", "")
_load_mod("capsule.storage_sharing", {SHARING_SRC!r})
''']
    if manifest:
        parts.append(f'_load_mod("capsule.manifest", {MANIFEST_SRC!r})\n')
    parts.append(f'_load_mod("capsule.optimizer_reattach", {REATTACH_SRC!r})\n')
    if fixture:
        parts.append(f'_load_mod("experiments", "")\n_fixture = _load_mod("experiments.fixture", {FIXTURE_SRC!r})\n')
    if oracles:
        parts.append(f'_oracles = _load_mod("experiments.oracles", {ORACLES_SRC!r})\n')
    return "".join(parts)

def _wrap(src: str) -> str:
    """Run `src` in a PRIVATE globals dict instead of the kernel's `__main__`.

    The next switch dumps `__main__`, so anything a harness program binds there
    either rides along in the capsule (and may be recorded by reference against
    a synthetic module the destination does not have) or has to be filtered
    out by name. Executing the program in its own dict leaves `__main__`
    exactly as the program found it, apart from what the program writes to it
    ON PURPOSE through `sys.modules["__main__"].__dict__`. That is also what
    lets the capture prove the source is untouched: its before/after binding
    snapshots are taken from inside this dict and compare equal.

    The cell itself is a bare expression statement returning None, so an
    IPython kernel binds no `_`/`Out` entry for it either.

    HEARTBEAT FIRST. The API interrupts an execution that emits no kernel
    event within 30 s of starting (a first-event timeout in the platform),
    and the interrupt leaves the kernel channel closed for the next request.
    Several handoff programs (the preflight's view prediction, the
    commit-boundary fingerprint, isolated validation) can run silently for
    longer than that on a loaded sandbox; the first live runs
    of the repaired controller failed exactly so ("Kernel channel closed",
    "ExecutionOutcomeUnknown", "not responding and has been recycled"). So
    every wrapped program prints one heartbeat line before it does anything.
    `Api.witness` reads only the line that starts with the result marker, so
    the heartbeat is ignored everywhere else.
    """
    return ('print("__clusy_heartbeat__", flush=True)\n'
            f'exec(compile({src!r}, "<clusy-handoff>", "exec"), {{"__name__": "__clusy_handoff__"}})\n')


def seed_program(seed: int, files: int, corpus_bytes: int) -> str:
    """Build the fixture in the kernel's REAL `__main__`, not as a module.

    This is the difference between a capsule that restores and one that does
    not. `dill` records a class defined in `__main__` BY VALUE, so the class
    body travels inside the capsule; a class belonging to any other module is
    recorded by reference, as an import the destination must satisfy. Loading
    the fixture as `experiments.fixture` therefore produced capsules whose
    restore died with "missing module 'experiments' - install it manually",
    because no sandbox image has this repository on its path. The platform
    driver execs the fixture into `globals()` for exactly this reason.

    The process baseline is recorded FIRST, before the fixture runs, under a
    dunder name the capture filter excludes: the preflight diffs the process
    against it to report what the session changed outside its namespace
    (global RNG streams, sys.path, environment, threads, library modules).
    """
    return _wrap(_prelude(fixture=False, oracles=False) + f'''
_g = sys.modules["__main__"].__dict__
from capsule.manifest import process_baseline
_base = process_baseline()
exec(compile({FIXTURE_SRC!r}, "clusy_fixture.py", "exec"), _g)
_spec = _g["FixtureSpec"](seed={seed}, workspace_corpus_files={files}, workspace_corpus_bytes={corpus_bytes},
                          include_workload=True, workload="tinyblock")
_ns = _g["build_transported_namespace"](_spec)
_g.update(_ns)
_exp = _g["compute_expectation"](_ns).__dict__
# Measuring the source changes the source: the fixture's reference forward
# increments a registered buffer AFTER the digest is taken. Re-digest so the
# expectation describes the bytes that actually leave (as the driver does).
_exp["param_digest"] = _g["_digest_params"](_ns["model"])
_g["__handoff_expectation__"] = _exp
_g["__handoff_counter__"] = 0
_g["__handoff_baseline__"] = _base
print({MARK!r} + _json.dumps({{"names": len(_ns), "expectation": _exp}}, default=str))
''')


def touch_program() -> str:
    """A side-effecting execution used to prove a runtime accepts work."""
    return _wrap(f'''
import sys as _s, json as _j
_g = _s.modules["__main__"].__dict__
if _g.get("__clusy_fence__"):
    print({MARK!r} + _j.dumps({{"accepted": False, "fenced_by": _g["__clusy_fence__"]}}))
else:
    _g["__handoff_counter__"] = _g.get("__handoff_counter__", 0) + 1
    print({MARK!r} + _j.dumps({{"accepted": True, "counter": _g["__handoff_counter__"]}}))
''')

def fence_program(token: str) -> str:
    return _wrap(f'''
import sys as _s, json as _j
_s.modules["__main__"].__dict__["__clusy_fence__"] = {token!r}
print({MARK!r} + _j.dumps({{"fenced": True}}))
''')


def describe_program() -> str:
    """Run ON THE DESTINATION: what it can load. This is the description the
    preflight judges the source against, so it must come from the destination
    itself, never from the source's own versions."""
    return _wrap(_prelude(fixture=False, oracles=False) + f'''
import dill
from capsule.manifest import distribution_inventory
try:
    import torch
    _cuda, _torch = bool(torch.cuda.is_available()), torch.__version__
except Exception:  # noqa: BLE001
    _cuda, _torch = False, None
print({MARK!r} + _json.dumps({{"python_minor": "%d.%d" % sys.version_info[:2], "python": sys.version.split()[0],
                             "dill_version": dill.__version__, "has_cuda": _cuda, "torch": _torch,
                             "packages": distribution_inventory()}}))
''')


# The names a capture dumps, and therefore the names a preflight must judge.
# Kept as one source string so the two programs cannot drift apart: a name the
# preflight never saw would ride in a capsule nobody classified.
_CAPTURE_FILTER = '''
_AMBIENT = {"In", "Out", "get_ipython", "exit", "quit", "open", "display", "_", "__", "___",
            "_ih", "_oh", "_dh", "_sh", "_exit_code", "_i", "_ii", "_iii"}
_KEEP_DUNDER = {"__handoff_expectation__", "__handoff_counter__"}
_ESSENTIAL_DUNDER = {"__name__", "__builtins__", "__doc__", "__package__", "__loader__", "__spec__"}
def _captured(_k, _v, _ambient=_AMBIENT, _mt=_types.ModuleType):
    """True for a user name the dump will carry."""
    return not (_k in _ambient or _k.startswith("_") or isinstance(_v, _mt))
'''


# Which gradients the capsule carries and which it does not, as source text.
# The fingerprint includes it (see `_STATE_FP_SRC`) and so does the preflight,
# which needs nothing else of the fingerprint: the list the preflight reports
# and the one the commit boundary declares are computed by the same code.
_GRADS_SRC = r"""
def _fp_rebuilt_from_state(obj):
    # True when pickle rebuilds `obj` from its __dict__/__slots__ alone (the
    # default protocol), so those are exactly what arrives at the far end.
    import copyreg
    t = type(obj)
    default_getstate = getattr(object, "__getstate__", None)
    return (t.__reduce_ex__ is object.__reduce_ex__ and t.__reduce__ is object.__reduce__
            and getattr(t, "__getstate__", None) in (None, default_getstate)
            and getattr(t, "__setstate__", None) is None
            and getattr(t, "__getnewargs_ex__", None) is None and getattr(t, "__getnewargs__", None) is None
            and t not in copyreg.dispatch_table)

def _grad_targets(names):
    # The tensors whose `.grad` the capsule carries and the fingerprint
    # compares, each ONCE (by identity, first path wins), in a deterministic
    # order: top-level tensors, every parameter of a module bound at top
    # level, every optimizer slot. (path string, path spec, tensor).
    torch = sys.modules.get("torch")
    if torch is None:
        return []
    out, seen = [], set()
    def add(path, spec, t):
        if isinstance(t, torch.Tensor) and id(t) not in seen:
            seen.add(id(t))
            out.append((path, spec, t))
    for k in sorted(names):
        v = names[k]
        if isinstance(v, torch.Tensor):
            add(k, ["name", k], v)
        elif isinstance(v, torch.nn.Module):
            for pn, pv in v.named_parameters():
                add(k + ".param." + pn, ["param", k, pn], pv)
        elif isinstance(v, torch.optim.Optimizer):
            for gi, grp in enumerate(v.param_groups):
                for pi, p in enumerate(grp["params"]):
                    add("%s.slot.%d.%d" % (k, gi, pi), ["slot", k, gi, pi], p)
    return out

#: Bounds on the read-only walk below: how deep it follows containers and
#: object state, and how many objects it visits. A walk that reaches either
#: bound SAYS so (a `<walk bound>` entry) rather than implying that nothing
#: lies deeper.
_GR_MAX_DEPTH, _GR_MAX_NODES = 8, 200000

#: Why a gradient outside `_grad_targets` is not in the capsule.
_GR_NOT_CARRIED = ("not carried: pickle drops .grad, and the capsule carries the gradients of top-level tensors, "
                   "parameters of top-level modules and optimizer slots only")

def _grads_reachable(names):
    # (path, tensor) for every tensor reachable from the captured names
    # through containers (list, tuple, dict, set), SimpleNamespace, the
    # __dict__ of objects pickle rebuilds from their state, and the
    # parameters, buffers and public attributes of any module met on the
    # way; each tensor ONCE (first path, breadth first from the sorted
    # names), plus whether a bound was hit. Read-only: nothing is written and
    # no user method runs beyond `named_parameters`/`named_buffers`.
    torch = sys.modules.get("torch")
    if torch is None:
        return [], False
    import collections
    np = sys.modules.get("numpy")
    atoms = (bool, int, float, complex, str, bytes, bytearray, type(None), type)
    if np is not None:
        atoms = atoms + (np.ndarray, np.generic)
    Module = torch.nn.Module
    out, seen, capped = [], set(), False
    todo = collections.deque((k, names[k], 0) for k in sorted(names))
    while todo:
        path, obj, depth = todo.popleft()
        if isinstance(obj, atoms) or id(obj) in seen:
            continue
        seen.add(id(obj))
        if len(seen) > _GR_MAX_NODES:
            capped = True
            break
        if isinstance(obj, torch.Tensor):
            out.append((path, obj))
            continue
        kids = []
        if isinstance(obj, Module):
            kids = [(path + ".param." + n, v) for n, v in obj.named_parameters()]
            kids += [(path + ".buffer." + n, v) for n, v in obj.named_buffers()]
            kids += [(path + "." + n, v) for n, v in list(vars(obj).items()) if not n.startswith("_")]
        elif isinstance(obj, (list, tuple)):
            kids = [("%s[%d]" % (path, i), v) for i, v in enumerate(obj)]
        elif isinstance(obj, dict):
            kids = [("%s[%s]" % (path, repr(k) if isinstance(k, (str, int)) else "<%s>" % type(k).__name__), v)
                    for k, v in list(obj.items())]
        elif isinstance(obj, (set, frozenset)):
            kids = [(path + "{}", v) for v in obj]
        elif isinstance(obj, _types.SimpleNamespace) or _fp_rebuilt_from_state(obj):
            d = getattr(obj, "__dict__", None)
            if isinstance(d, dict):
                kids = [(path + "." + str(k), v) for k, v in list(d.items())]
        kids = [(p, v) for p, v in kids if not isinstance(v, atoms)]
        if kids and depth >= _GR_MAX_DEPTH:
            capped = True
            continue
        todo.extend((p, v, depth + 1) for p, v in kids)
    return out, capped

def _grads_uncarried(names):
    # {path: reason} for every gradient the capsule does NOT carry as it is:
    # a non-None `.grad` on a tensor outside `_grad_targets`, which arrives as
    # None, and a carried gradient that shares storage with another tensor
    # (a view of a flat gradient buffer, a bucket), which arrives with its
    # value but unshared. Read-only. Empty when every gradient is carried.
    torch = sys.modules.get("torch")
    if torch is None:
        return {}
    import warnings
    targets = _grad_targets(names)
    carried = {id(t) for _p, _s, t in targets}
    reach, capped = _grads_reachable(names)
    out = {}
    with warnings.catch_warnings():
        # Reading `.grad` of a non-leaf warns; the read is still correct.
        warnings.simplefilter("ignore")
        for path, t in reach:
            if id(t) not in carried and t.grad is not None:
                out[path + ".grad"] = _GR_NOT_CARRIED
        grads = [(p + ".grad", t.grad) for p, _s, t in targets if t.grad is not None]

    def key(t):
        try:
            if t.layout != torch.strided or t.numel() == 0 or t.device.type == "meta":
                return None
            return (t.untyped_storage().data_ptr(), str(t.device))
        except Exception:  # noqa: BLE001
            return None
    owners = {}
    for p, t in list(reach) + grads:
        k = key(t)
        if k is not None:
            owners.setdefault(k, []).append((p, t))
    for p, gr in grads:
        # Identity is not sharing: a gradient also bound under another name
        # is one object, and the pickle memo keeps it one.
        others = sorted({q for q, o in owners.get(key(gr), []) if o is not gr})
        if others:
            out[p] = "carried by value; its storage sharing with %s is not" % ", ".join(others[:3])
    if capped:
        out["<walk bound>"] = ("the walk stopped at depth %d or %d objects: gradients beyond that were not "
                               "examined" % (_GR_MAX_DEPTH, _GR_MAX_NODES))
    return dict(sorted(out.items()))
"""


# The state fingerprint, as source text shipped into the kernels. Every
# helper is defined in the program's PRIVATE globals dict (see `_wrap`), never
# in `__main__`, and nothing here writes to the state it reads: RNG streams are
# read by cloning their state into a fresh generator, optimizer state is
# iterated (never indexed: `opt.state` is a defaultdict, and indexing a missing
# key would insert one), tensors are read through `detach()`, and CUDA is
# never initialized by the fingerprint. It includes `_GRADS_SRC` (below).
_STATE_FP_BODY = r"""
_FP_PLAIN_SCALARS = (bool, int, float, complex, str, bytes, type(None))
_FP_EXTRA_NAMES = ("__handoff_counter__",)

def _fp_sha(data):
    if isinstance(data, str):
        data = data.encode("utf-8", "surrogatepass")
    return _hl.sha256(data).hexdigest()

def _fp_hex(values):
    # float.hex is exact, so equal strings mean equal bits.
    return [float(v).hex() for v in values]

def _fp_qual(obj):
    t = type(obj)
    return t.__module__ + "." + t.__qualname__

#: Values the fingerprint could not digest, as "<where>: <why>" strings. Reset
#: by `_state_fingerprint` and reported as its `unhashable` field, so a value
#: that was not compared is NAMED, never silently equal.
_FP_UNHASHABLE = []

def _fp_mark(where, why):
    _FP_UNHASHABLE.append("%s: %s" % (where, why))
    return "unhashable:" + why

def _fp_dense_bytes(d):
    # A strided CPU tensor's element bytes (conjugate and negative views
    # resolved first: their bits are lazy flags, not data).
    torch = sys.modules["torch"]
    d = d.resolve_conj().resolve_neg().contiguous().reshape(-1)
    return d.view(torch.uint8).numpy().tobytes() if d.numel() else b""

def _fp_tensor_digest(t, where="tensor"):
    # sha256 of a tensor's VALUE, whatever its layout; an explicit
    # "unhashable:<why>" marker (recorded in `_FP_UNHASHABLE`) otherwise.
    torch = sys.modules["torch"]
    try:
        d = t.detach()
        if d.device.type == "meta":
            return _fp_mark(where, "meta tensor (no data)")
        if d.device.type != "cpu":
            d = d.to("cpu", copy=True)
        lay = d.layout
        if d.is_quantized:
            qs = d.qscheme()
            if qs in (torch.per_tensor_affine, torch.per_tensor_symmetric):
                params = "%s|%s|%d" % (qs, float(d.q_scale()).hex(), int(d.q_zero_point()))
            else:
                params = "%s|%s|%s|%d" % (qs, _fp_sha(_fp_dense_bytes(d.q_per_channel_scales())),
                                          _fp_sha(_fp_dense_bytes(d.q_per_channel_zero_points())),
                                          int(d.q_per_channel_axis()))
            return _fp_sha(("quantized|" + params + "|").encode() + _fp_dense_bytes(d.int_repr()))
        if lay == torch.strided:
            return _fp_sha(_fp_dense_bytes(d))
        if lay == torch.sparse_coo:
            c = d.coalesce()                       # a coalesced COPY unless already coalesced: read-only
            return _fp_sha(b"sparse_coo|" + _fp_dense_bytes(c.indices()) + b"|" + _fp_dense_bytes(c.values()))
        if lay in (torch.sparse_csr, torch.sparse_bsr):
            return _fp_sha(str(lay).encode() + b"|" + _fp_dense_bytes(d.crow_indices()) + b"|"
                           + _fp_dense_bytes(d.col_indices()) + b"|" + _fp_dense_bytes(d.values()))
        if lay in (torch.sparse_csc, torch.sparse_bsc):
            return _fp_sha(str(lay).encode() + b"|" + _fp_dense_bytes(d.ccol_indices()) + b"|"
                           + _fp_dense_bytes(d.row_indices()) + b"|" + _fp_dense_bytes(d.values()))
        if lay == getattr(torch, "_mkldnn", None):
            return _fp_sha(b"mkldnn|" + _fp_dense_bytes(d.to_dense()))
        return _fp_mark(where, "layout %s" % lay)
    except Exception as e:  # noqa: BLE001
        return _fp_mark(where, "%s: %s" % (type(e).__name__, str(e)[:80]))

def _fp_tensor(t, where="tensor"):
    torch = sys.modules["torch"]
    try:
        stride = list(t.stride()) if t.layout == torch.strided else None
    except Exception:  # noqa: BLE001
        stride = None
    return {"class": _fp_qual(t), "dtype": str(t.dtype), "shape": list(t.shape), "stride": stride,
            "layout": str(t.layout), "requires_grad": bool(t.requires_grad),
            "sha256": _fp_tensor_digest(t, where)}

def _fp_array(a, where="ndarray"):
    np = sys.modules["numpy"]
    # Bytes, dtype and shape, NOT strides: pickle rebuilds a strided array
    # contiguously, so its strides change although its values do not. The
    # view relationships have their own field (`views`). An array whose
    # dtype holds Python objects is digested by VALUE: its raw bytes are
    # object pointers, which differ in every process.
    try:
        if a.dtype.hasobject:
            memo = {}
            raw = "objects[" + ",".join(_fp_deep(x, memo) for x in a.ravel(order="C")) + "]"
        else:
            raw = np.ascontiguousarray(a).tobytes()
        digest = _fp_sha(raw)
    except Exception as e:  # noqa: BLE001
        digest = _fp_mark(where, "%s: %s" % (type(e).__name__, str(e)[:80]))
    return {"class": _fp_qual(a), "dtype": a.dtype.str, "shape": list(a.shape), "sha256": digest}

def _fp_safe_repr_types():
    # Types whose repr is their value and carries no address.
    import datetime, decimal, fractions, uuid, pathlib, enum
    return (datetime.date, datetime.time, datetime.timedelta, datetime.tzinfo, decimal.Decimal,
            fractions.Fraction, uuid.UUID, pathlib.PurePath, range, slice, enum.Enum, complex)

def _fp_deep(obj, memo):
    # Canonical text of a value, for the `objects` field: containers are
    # walked; tensors and arrays are digested; objects pickle rebuilds from
    # their state are walked through that state; functions defined in
    # __main__ (which dill carries by value) through their defaults, closure
    # cells and attributes; anything else is its type. `memo` makes a second
    # reference to an object a back-reference by visit order (identity inside
    # a value is state too) and keeps every visited object alive, so an id
    # is never reused by a temporary during the walk.
    if isinstance(obj, float):
        return "f:" + float(obj).hex()
    if isinstance(obj, _FP_PLAIN_SCALARS):
        return type(obj).__name__ + ":" + repr(obj)
    np, torch = sys.modules.get("numpy"), sys.modules.get("torch")
    if np is not None and isinstance(obj, np.generic):
        return "np:" + obj.dtype.str + ":" + _fp_deep(obj.item(), memo)
    seen = memo.get(id(obj))
    if seen is not None:
        return "<ref:%d>" % seen[0]
    memo[id(obj)] = (len(memo), obj)
    q = _fp_qual(obj)
    if isinstance(obj, (list, tuple)):
        return q + "[" + ",".join(_fp_deep(v, memo) for v in obj) + "]"
    if isinstance(obj, (set, frozenset)):
        # Visit in a canonical order: iteration order follows string hashing,
        # which is randomised per process.
        items = sorted(obj, key=lambda v: _fp_deep(v, {}))
        return q + "{" + ",".join(_fp_deep(v, memo) for v in items) + "}"
    if isinstance(obj, dict):
        return q + "{" + ",".join(_fp_deep(k, memo) + "=" + _fp_deep(v, memo) for k, v in obj.items()) + "}"
    if isinstance(obj, bytearray):
        return "bytearray:" + _fp_sha(bytes(obj))
    if torch is not None and isinstance(obj, torch.Tensor):
        return "tensor:" + _json.dumps(_fp_tensor(obj, "nested " + q), sort_keys=True)
    if np is not None and isinstance(obj, np.ndarray):
        return "ndarray:" + _json.dumps(_fp_array(obj, "nested " + q), sort_keys=True)
    if isinstance(obj, _fp_safe_repr_types()):
        return "repr:" + q + ":" + repr(obj)
    if isinstance(obj, type):
        return "class:" + obj.__module__ + "." + obj.__qualname__
    if isinstance(obj, _types.FunctionType):
        name = "function:" + str(obj.__module__) + "." + obj.__qualname__
        if obj.__module__ != "__main__":
            return name
        cells = []
        for c in obj.__closure__ or ():
            try:
                cells.append(_fp_deep(c.cell_contents, memo))
            except ValueError:
                cells.append("<empty cell>")
        return (name + "(defaults=" + _fp_deep(obj.__defaults__, memo) + ",kwdefaults="
                + _fp_deep(obj.__kwdefaults__, memo) + ",closure=[" + ",".join(cells) + "],dict="
                + _fp_deep(dict(obj.__dict__), memo) + ")")
    if isinstance(obj, _types.MethodType):
        return "method:" + _fp_deep(obj.__func__, memo) + "@" + _fp_deep(obj.__self__, memo)
    if isinstance(obj, _types.SimpleNamespace):
        return q + "(" + _fp_deep(vars(obj), memo) + ")"
    import collections, functools
    if isinstance(obj, collections.deque):
        return q + "(maxlen=%r)[" % (obj.maxlen,) + ",".join(_fp_deep(v, memo) for v in obj) + "]"
    if isinstance(obj, functools.partial):
        return (q + "(" + _fp_deep(obj.func, memo) + "," + _fp_deep(obj.args, memo) + ","
                + _fp_deep(obj.keywords, memo) + ")")
    if _fp_rebuilt_from_state(obj):
        parts = []
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            parts.append(_fp_deep(d, memo))
        for cls in type(obj).__mro__:
            slots = cls.__dict__.get("__slots__", ())
            for s in ((slots,) if isinstance(slots, str) else slots):
                if s in ("__dict__", "__weakref__") or not hasattr(obj, s):
                    continue
                parts.append(s + "=" + _fp_deep(getattr(obj, s), memo))
        return "object:" + q + "(" + ",".join(parts) + ")"
    return "object:" + q

def _fp_canon(obj, strict, _stack=()):
    # Canonical text of a value. `strict`: plain values only, None for
    # anything else. Otherwise tensors and arrays are encoded by digest and
    # any other object by its type alone (never its repr, which may carry an
    # address). Sets are sorted: their order follows string hashing, which is
    # randomised per process. A cycle becomes a back-reference by depth.
    for depth, o in enumerate(_stack):
        if o is obj:
            return "<cycle:%d>" % depth
    if isinstance(obj, float):
        return "f:" + float(obj).hex()
    if isinstance(obj, _FP_PLAIN_SCALARS):
        return type(obj).__name__ + ":" + repr(obj)
    st = _stack + (obj,)
    if isinstance(obj, (list, tuple, set, frozenset)):
        parts = [_fp_canon(v, strict, st) for v in obj]
        if strict and any(p is None for p in parts):
            return None
        if isinstance(obj, (set, frozenset)):
            parts = sorted(parts)
        return type(obj).__name__ + "[" + ",".join(parts) + "]"
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            pk, pv = _fp_canon(k, strict, st), _fp_canon(v, strict, st)
            if strict and (pk is None or pv is None):
                return None
            parts.append(pk + "=" + pv)
        return "dict{" + ",".join(parts) + "}"
    if strict:
        return None
    torch, np = sys.modules.get("torch"), sys.modules.get("numpy")
    if torch is not None and isinstance(obj, torch.Tensor):
        return "tensor:" + _json.dumps(_fp_tensor(obj), sort_keys=True)
    if np is not None and isinstance(obj, np.ndarray):
        return "ndarray:" + _json.dumps(_fp_array(obj), sort_keys=True)
    return "object:" + _fp_qual(obj)

def _fp_generator(v):
    # Full-state hash plus the next draws of a CLONE (the E13 approach).
    import random
    np, torch = sys.modules.get("numpy"), sys.modules.get("torch")
    try:
        if torch is not None and isinstance(v, torch.Generator):
            s = v.get_state()
            c = torch.Generator(device=v.device)
            c.set_state(s)
            return {"kind": "torch.Generator", "device": str(v.device), "sha256": _fp_sha(s.cpu().numpy().tobytes()),
                    "draws": _fp_hex(torch.rand(4, generator=c, device=v.device).tolist())}
        if np is not None and isinstance(v, np.random.Generator):
            bg = v.bit_generator
            s = bg.state
            c = np.random.Generator(type(bg)())
            c.bit_generator.state = s
            return {"kind": "numpy.Generator", "sha256": _fp_sha(_fp_canon(s, False)),
                    "draws": _fp_hex(c.random(4)), "libm_draws": _fp_hex(c.standard_normal(2))}
        if np is not None and isinstance(v, np.random.RandomState):
            s = v.get_state()
            c = np.random.RandomState()
            c.set_state(s)
            return {"kind": "numpy.RandomState", "sha256": _fp_sha(_fp_canon(s, False)),
                    "draws": _fp_hex(c.random_sample(4)), "libm_draws": _fp_hex(c.standard_normal(2))}
        if isinstance(v, random.Random):
            s = v.getstate()
            c = random.Random()
            c.setstate(s)
            return {"kind": "random.Random", "sha256": _fp_sha(repr(s)),
                    "draws": _fp_hex(c.random() for _ in range(4)) + ["%016x" % c.getrandbits(64)]}
    except Exception as e:  # noqa: BLE001
        return {"kind": _fp_qual(v), "error": "%s: %s" % (type(e).__name__, e)}
    return None

def _fp_rng(names):
    import random
    fp = {}
    st = random.getstate()
    r = random.Random()
    r.setstate(st)
    fp["python"] = {"sha256": _fp_sha(repr(st)),
                    "draws": _fp_hex(r.random() for _ in range(4)) + ["%016x" % r.getrandbits(64)]}
    np = sys.modules.get("numpy")
    if np is None:
        fp["numpy"] = {"loaded": False}
    else:
        name, key, pos, has_gauss, cached = np.random.get_state()
        rs = np.random.RandomState()
        rs.set_state((name, key, pos, has_gauss, cached))
        fp["numpy"] = {"loaded": True, "sha256": _fp_sha(name.encode() + np.ascontiguousarray(key, dtype="<u4").tobytes()
                                                         + ("|%d|%d|%s" % (int(pos), int(has_gauss), float(cached).hex())).encode()),
                       "draws": _fp_hex(rs.random_sample(4)), "libm_draws": _fp_hex(rs.standard_normal(2))}
    torch = sys.modules.get("torch")
    if torch is None:
        fp["torch_cpu"] = {"loaded": False}
    else:
        st = torch.get_rng_state()
        c = torch.Generator()
        c.set_state(st)
        fp["torch_cpu"] = {"loaded": True, "sha256": _fp_sha(st.numpy().tobytes()),
                           "draws": _fp_hex(torch.rand(4, generator=c).tolist())}
    named = {}
    for k in sorted(names):
        e = _fp_generator(names[k])
        if e is not None:
            named[k] = e
    fp["named"] = named
    return fp

def _fp_rng_cuda():
    # Never initializes CUDA: a stream that was never initialized is recorded
    # as such, and the capture carries CUDA state only when it is initialized.
    torch = sys.modules.get("torch")
    if torch is None:
        return {"available": False, "initialized": False}
    cu = {"available": bool(torch.cuda.is_available()), "initialized": False}
    if cu["available"]:
        cu["device_count"] = int(torch.cuda.device_count())
        cu["initialized"] = bool(torch.cuda.is_initialized())
    if cu["initialized"]:
        devs = []
        for i in range(cu["device_count"]):
            s = torch.cuda.get_rng_state(i)
            g = torch.Generator(device="cuda:%d" % i)
            g.set_state(s)
            devs.append({"index": i, "sha256": _fp_sha(s.cpu().numpy().tobytes()),
                         "draws": _fp_hex(torch.rand(4, device="cuda:%d" % i, generator=g).tolist())})
        cu["devices"] = devs
    return cu

def _fp_workspace(root):
    import stat as _st
    if not os.path.isdir(root):
        return {"present": False}
    files = {}
    for d, dirs, fs in os.walk(root):
        dirs.sort()
        for f in sorted(fs):
            p = os.path.join(d, f)
            rel = os.path.relpath(p, root)
            st = os.lstat(p)
            if not _st.S_ISREG(st.st_mode):
                files[rel] = ["not a regular file", oct(st.st_mode)]
                continue
            h = _hl.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            files[rel] = [h.hexdigest(), st.st_size, oct(st.st_mode & 0o777)]
    return {"present": True, "files": files}

def _fp_views(g, views_json):
    ss = sys.modules.get("capsule.storage_sharing")
    fn = getattr(ss, "views_intact", None)
    if fn is None:
        return {"available": False}
    try:
        r = fn(g, ss.ViewManifest.from_json(views_json or {}))
    except Exception as e:  # noqa: BLE001
        return {"available": True, "error": ("%s: %s" % (type(e).__name__, e))[:300]}
    broken = sorted(_json.dumps(b.get("path", b) if isinstance(b, dict) else b, default=str)
                    + " " + str(b.get("reason", "") if isinstance(b, dict) else "")
                    for b in (r.get("broken") or []))
    return {"available": True, "checked": int(r.get("checked", 0)), "intact": int(r.get("intact", 0)),
            "broken": broken}

def _state_fingerprint(g, views_json, workspace_root="data"):
    torch, np = sys.modules.get("torch"), sys.modules.get("numpy")
    del _FP_UNHASHABLE[:]
    names = {k: v for k, v in g.items() if _captured(k, v) or k in _FP_EXTRA_NAMES}
    Module = torch.nn.Module if torch is not None else ()
    Optim = torch.optim.Optimizer if torch is not None else ()
    Sched = getattr(torch.optim.lr_scheduler, "LRScheduler", ()) if torch is not None else ()
    Loader = torch.utils.data.DataLoader if torch is not None else ()
    Scaler = getattr(getattr(torch, "amp", None), "GradScaler", ()) if torch is not None else ()
    Tensor = torch.Tensor if torch is not None else ()
    Array = np.ndarray if np is not None else ()
    fp = {"names": {k: _fp_qual(v) for k, v in sorted(names.items())}}
    plain = {}
    for k, v in names.items():
        c = _fp_canon(v, True)
        if c is not None:
            plain[k] = _fp_sha(c)
    fp["plain"] = dict(sorted(plain.items()))
    # Reference paths: top-level names, and every parameter and buffer of a
    # module bound at top level. `path_of` names an object by its first path.
    refs = []
    for k in sorted(names):
        v = names[k]
        if not isinstance(v, _FP_PLAIN_SCALARS):
            refs.append((k, v))
        if Module and isinstance(v, Module):
            for pn, pv in v.named_parameters(remove_duplicate=False):
                refs.append((k + ".param." + pn, pv))
            for bn, bv in v.named_buffers(remove_duplicate=False):
                refs.append((k + ".buffer." + bn, bv))
    groups, path_of = {}, {}
    for path, obj in refs:
        groups.setdefault(id(obj), []).append(path)
        path_of.setdefault(id(obj), path)
    fp["aliases"] = sorted(sorted(ps) for ps in groups.values() if len(ps) > 1)
    tensors, modules, optims, scheds, stateful, loaders, devices, grads = {}, {}, {}, {}, {}, {}, {}, {}
    objects = {}
    for k in sorted(names):
        v = names[k]
        if Tensor and isinstance(v, Tensor):
            tensors[k] = _fp_tensor(v, k)
            devices[k] = str(v.device)
        elif Array and isinstance(v, Array):
            tensors[k] = _fp_array(v, k)
        elif Module and isinstance(v, Module):
            ps, bs = [], []
            for pn, pv in v.named_parameters():
                ps.append([pn, _fp_tensor(pv, k + ".param." + pn)])
                devices[k + ".param." + pn] = str(pv.device)
            for bn, bv in v.named_buffers():
                bs.append([bn, _fp_tensor(bv, k + ".buffer." + bn)])
                devices[k + ".buffer." + bn] = str(bv.device)
            modules[k] = {"class": _fp_qual(v), "params": ps, "buffers": bs,
                          "training": [bool(m.training) for m in v.modules()]}
        elif Optim and isinstance(v, Optim):
            flat = [p for grp in v.param_groups for p in grp["params"]]
            index = {id(p): i for i, p in enumerate(flat)}
            state = []
            for p, st in v.state.items():             # iterate: never index a defaultdict
                entries = []
                for sk in sorted(st, key=str):
                    sv = st[sk]
                    if Tensor and isinstance(sv, Tensor):
                        entries.append([str(sk), _fp_tensor(sv)])
                        devices[k + ".state.%s.%s" % (index.get(id(p), -1), sk)] = str(sv.device)
                    else:
                        entries.append([str(sk), _fp_canon(sv, False)])
                state.append([index.get(id(p), -1), path_of.get(id(p), "<unbound>"), entries])
            state.sort(key=lambda e: (e[0], e[1]))
            optims[k] = {"class": _fp_qual(v),
                         "binding": [path_of.get(id(p), "<unbound>") for p in flat],
                         "groups": [_fp_sha(_fp_canon({gk: gv for gk, gv in grp.items() if gk != "params"}, False))
                                    for grp in v.param_groups],
                         "state": state,
                         "state_keys_bound": all(index.get(id(p), -1) >= 0 for p in v.state)}
        elif Sched and isinstance(v, Sched):
            scheds[k] = {"class": _fp_qual(v), "optimizer": path_of.get(id(getattr(v, "optimizer", None)), "<unbound>"),
                         "state": _fp_sha(_fp_canon(v.state_dict(), False))}
        elif Scaler and isinstance(v, Scaler):
            stateful[k] = {"class": _fp_qual(v), "state": _fp_sha(_fp_canon(v.state_dict(), False))}
        elif Loader and isinstance(v, Loader):
            lg = getattr(v, "generator", None)
            sg = getattr(getattr(v, "sampler", None), "generator", None)
            loaders[k] = {"generator": None if lg is None else _fp_generator(lg),
                          "sampler_generator": ("same" if sg is lg else (None if sg is None else _fp_generator(sg)))}
        elif k not in plain:
            # Everything the fields above do not cover: containers holding
            # tensors, arrays or objects, instances, functions. Digested
            # DEEPLY, so a write into such a value (a hook filling a dict of
            # activations) is a difference, not an equal fingerprint.
            try:
                objects[k] = _fp_sha(_fp_deep(v, {}))
            except RecursionError:
                objects[k] = _fp_mark(k, "nested too deeply to walk")
            except Exception as e:  # noqa: BLE001
                objects[k] = _fp_mark(k, "%s: %s" % (type(e).__name__, str(e)[:80]))
    fp.update(tensors=tensors, modules=modules, optimizers=optims, schedulers=scheds, stateful=stateful,
              loaders=loaders, objects=objects)
    # Gradients: the capsule carries them (`_grads_collect`), so they are
    # compared by value, not declared.
    for path, _spec, t in _grad_targets(names):
        gr = t.grad
        grads[path] = None if gr is None else _fp_tensor(gr, path + ".grad")
        if gr is not None:
            devices[path + ".grad"] = str(gr.device)
    fp["rng"] = _fp_rng(names)
    fp["rng_cuda"] = _fp_rng_cuda()
    fp["views"] = _fp_views(g, views_json)
    fp["workspace"] = _fp_workspace(workspace_root)
    fp["devices"] = devices
    fp["grads"] = grads
    # What `grads` does NOT cover, by name and reason: declared at the commit
    # boundary, never compared as equal (see `compare_fingerprints`).
    fp["grads_not_carried"] = _grads_uncarried(names)
    fp["unhashable"] = sorted(set(_FP_UNHASHABLE))
    return fp

def _grads_collect(names):
    # [(path spec, grad)] for every covered tensor that has a gradient. The
    # capture dumps this list under an injected dunder name, in the SAME
    # pickle as the tensors, so a gradient also bound elsewhere keeps its
    # identity.
    return [(spec, t.grad) for _path, spec, t in _grad_targets(names) if t.grad is not None]

def _grads_apply(g, carried):
    # Reattach carried gradients by path, AFTER the view repair (which may
    # rebind a parameter). Assignment checks shape, dtype and device.
    rep = {"carried": len(carried or []), "reattached": 0, "failed": []}
    for spec, gr in carried or []:
        try:
            kind = spec[0]
            if kind == "name":
                t = g[spec[1]]
            elif kind == "param":
                t = g[spec[1]].get_parameter(spec[2])
            elif kind == "slot":
                t = g[spec[1]].param_groups[spec[2]]["params"][spec[3]]
            else:
                raise KeyError("unknown path kind %r" % (kind,))
            t.grad = gr
            rep["reattached"] += 1
        except Exception as e:  # noqa: BLE001
            rep["failed"].append({"path": list(spec), "reason": ("%s: %s" % (type(e).__name__, e))[:200]})
    return rep
"""
_STATE_FP_SRC = _GRADS_SRC + _STATE_FP_BODY

# The process-global RNG streams, carried in the capsule manifest. dill dumps
# `__main__`; the random, numpy and torch modules are recorded by reference,
# so their mutable stream state does not travel with it (the platform
# checkpoint carries an equivalent envelope). CUDA is captured only when
# already initialized, and restored only on a destination that has at least as
# many devices; otherwise it is a DECLARED drop, reported, never a silent one.
_RNG_SRC = r"""
def _rng_capture():
    import random
    st = random.getstate()
    env = {"python": [st[0], list(st[1]), st[2]], "numpy": None, "torch_cpu": None,
           "cuda": {"initialized": False, "states": []}}
    np = sys.modules.get("numpy")
    if np is not None:
        name, key, pos, has_gauss, cached = np.random.get_state()
        env["numpy"] = {"name": name, "key": _b64.b64encode(np.ascontiguousarray(key, dtype="<u4").tobytes()).decode(),
                        "pos": int(pos), "has_gauss": int(has_gauss), "cached": float(cached)}
    torch = sys.modules.get("torch")
    if torch is not None:
        env["torch_cpu"] = _b64.b64encode(torch.get_rng_state().numpy().tobytes()).decode()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            env["cuda"] = {"initialized": True,
                           "states": [_b64.b64encode(s.cpu().numpy().tobytes()).decode()
                                      for s in torch.cuda.get_rng_state_all()]}
    return env

def _rng_apply(env):
    import random
    rep = {"python": False, "numpy": False, "torch_cpu": False, "cuda": "not_carried"}
    py = env["python"]
    random.setstate((py[0], tuple(py[1]), py[2]))
    rep["python"] = True
    if env.get("numpy") is not None:
        import numpy as np
        n = env["numpy"]
        key = np.frombuffer(_b64.b64decode(n["key"]), dtype="<u4").astype(np.uint32)
        np.random.set_state((n["name"], key, n["pos"], n["has_gauss"], n["cached"]))
        rep["numpy"] = True
    if env.get("torch_cpu") is not None:
        import torch
        torch.set_rng_state(torch.frombuffer(bytearray(_b64.b64decode(env["torch_cpu"])), dtype=torch.uint8).clone())
        rep["torch_cpu"] = True
    cu = env.get("cuda") or {}
    if cu.get("initialized"):
        import torch
        states = [torch.frombuffer(bytearray(_b64.b64decode(s)), dtype=torch.uint8).clone() for s in cu["states"]]
        # With fewer local devices than captured states, the restore would
        # not fail here but at CUDA's next initialization, so the counts are
        # compared before anything is applied.
        if torch.cuda.is_available() and torch.cuda.device_count() >= len(states):
            torch.cuda.set_rng_state_all(states)
            rep["cuda"] = "restored"
        else:
            rep["cuda"] = "declared_drop"
    return rep
"""


#: Why the forward-output oracle has no source reference in a handoff (see
#: VALIDATION in the module doc): recorded in every expectation the capture
#: writes (`forward_output_not_run`) and reported verbatim by the
#: destination's declared forward row.
FORWARD_NOT_RUN = ("recompute is an evaluation oracle; the parameters digest and the commit-boundary "
                   "fingerprint establish the bytes")

# The capture's validation expectation, from STRUCTURE only, as source text
# (see VALIDATION in the module doc). Nothing here runs a forward, a hook, a
# backward or an optimizer step, and every helper is defined in the program's
# PRIVATE dict: the first version called the fixture's `compute_expectation`
# and `_digest_params` through `__main__`, which no capsule carries
# (underscore names and modules are filtered out), so a runtime this
# controller had itself restored, once the user trained on it, raised
# `NameError: _digest_params` at its next capture and failed validation. What
# this reads from the namespace are the workload NAMES the oracles check
# (`model`, `optimizer`, `scheduler`, `loader`) and the value keys the
# recorded expectation already lists; an absent name leaves its field None and
# the matching oracle then fails, which is the right outcome. It runs inside
# the capture's stash, on exactly the dumped names, between the RNG envelope
# and its re-application.
_EXPECT_SRC = r"""
_CUT_PLAIN = (bool, int, float, complex, str, bytes, type(None))

def _cut_plain(v, depth=0):
    # Plain data by EXACT type, so repr() below runs no user `__repr__` (a
    # str subclass passes isinstance and may define one).
    t = type(v)
    if t in _CUT_PLAIN:
        return True
    if depth > 20:
        return False
    if t in (list, tuple, set, frozenset):
        return all(_cut_plain(x, depth + 1) for x in v)
    if t is dict:
        return all(_cut_plain(k, depth + 1) and _cut_plain(x, depth + 1) for k, x in v.items())
    return False

def _cut_param_digest(model):
    # The oracles' parameter digest: the SAME algorithm as the fixture's
    # `_digest_params`, which the destination's `parameters` row computes
    # (a test pins that the two agree). Parameters then buffers, sorted by
    # name, cpu bytes.
    h = _hl.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().to("cpu", copy=True).contiguous().numpy().tobytes())
    for name, b in sorted(model.named_buffers()):
        h.update(name.encode())
        t = b.detach().to("cpu", copy=True).contiguous()
        h.update(t.numpy().tobytes() if t.numel() else b"empty")
    return h.hexdigest()[:32]

def _cut_value(ns, key):
    # One entry of `values`, read exactly as the destination's `values` row
    # reads it (experiments/oracles.py), from plain data only.
    np = sys.modules.get("numpy")
    if key == "arr_sum":
        if "arr" not in ns:
            return "<missing>"
        a = ns["arr"]
        if np is None or type(a) is not np.ndarray or a.dtype.hasobject:
            raise TypeError("arr is not a plain ndarray")
        return repr(float(a.sum()))
    if key == "co_dict":
        if "co_dict" not in ns:
            return "<missing>"
        d = ns["co_dict"]
        if type(d) is not dict or not _cut_plain(d):
            raise TypeError("co_dict is not plain data")
        return repr(sorted(d.items(), key=repr))
    if key == "sc_str":
        v = ns.get("sc_str", "<missing>")
        if type(v) is not str:
            raise TypeError("sc_str is not a str")
        return v
    v = ns.get(key, "<missing>")
    if not _cut_plain(v):
        raise TypeError("%s is not plain data" % key)
    return repr(v)

def _cut_step(ns, model):
    # (count, where it came from). The count the destination's `continuation`
    # row compares with: the optimizer state's `step` for the model's FIRST
    # parameter, the slot that row reads on its copy after one more step.
    # `opt.state` is a defaultdict, so it is iterated, never indexed. No state
    # for that parameter yet means the next step is the first (0). Without an
    # optimizer, the application's own `step_count`.
    torch = sys.modules.get("torch")
    opt = ns.get("optimizer")
    first = next(iter(model.parameters()), None)
    if torch is not None and isinstance(opt, torch.optim.Optimizer) and first is not None:
        for p, st in opt.state.items():
            if p is first:
                s = st.get("step") if isinstance(st, dict) else None
                if s is None:
                    break
                s = float(s.item() if isinstance(s, torch.Tensor) else s)
                return (int(s) if s.is_integer() else s), "optimizer state step of the first parameter"
        else:
            return 0, "no optimizer state yet for the first parameter"
    sc = ns.get("step_count")
    return (sc if type(sc) is int else None), "the application's step_count"

def _cut_loader_indices(loader):
    # The next indices the loader's sampler yields, read from a SHALLOW COPY
    # of the sampler drawing from a CLONE of its generator: the real
    # generator is never advanced, not even temporarily. The sampler's
    # iteration (and the dataset `__len__` it calls) is the one piece of
    # user-reachable code this runs; it is not model code, and the capture's
    # second fingerprint sees any state it changed. The caller re-applies the
    # RNG envelope afterwards (a sampler without a generator seeds from the
    # global stream).
    import copy as _copy
    torch = sys.modules["torch"]
    sampler = _copy.copy(loader.sampler)
    sg = getattr(loader.sampler, "generator", None)
    if sg is not None:
        clone = torch.Generator(device=sg.device)
        clone.set_state(sg.get_state())
        sampler.generator = clone
    return [int(i) for i in list(iter(sampler))[:8]]

def _cut_expectation(ns, fp):
    # (expectation, source, method) for the namespace at the cut and its
    # fingerprint. A field that cannot be read is None and named in
    # `not_refreshed`, and the source says the refresh was incomplete: the
    # matching oracle then fails with the cause on record, never a stale
    # value passing as the cut's.
    method = {"model_code_run": False, "forward_output": "declared not run: " + _FORWARD_NOT_RUN}
    old = ns.get("__handoff_expectation__")
    if not isinstance(old, dict):
        return old, "recorded", method
    torch = sys.modules.get("torch")
    exp, failed = {}, {}

    def field(name, fn):
        try:
            exp[name] = fn()
        except Exception as e:  # noqa: BLE001
            exp[name] = None
            failed[name] = ("%s: %s" % (type(e).__name__, e))[:160]

    values = {}
    for key in sorted(old.get("values") or {}):
        try:
            values[key] = _cut_value(ns, key)
        except Exception as e:  # noqa: BLE001
            failed["values." + key] = ("%s: %s" % (type(e).__name__, e))[:160]
    exp["values"] = values
    model = ns.get("model")
    if torch is not None and isinstance(model, torch.nn.Module):
        field("param_digest", lambda: _cut_param_digest(model))
        field("device_type", lambda: next(model.parameters()).device.type)
        field("step_count", lambda: _cut_step(ns, model))
        exp["step_count"], exp["step_count_from"] = exp["step_count"] or (None, None)
    else:
        exp.update(param_digest=None, device_type=None, step_count=None, step_count_from=None)
    sched, opt = ns.get("scheduler"), ns.get("optimizer")
    if sched is not None and opt is not None:
        field("scheduler_lr", lambda: float(opt.param_groups[0]["lr"]))
        field("scheduler_epoch", lambda: int(sched.last_epoch))
    else:
        exp.update(scheduler_lr=None, scheduler_epoch=None)
    loader = ns.get("loader")
    if loader is not None:
        field("next_loader_indices", lambda: _cut_loader_indices(loader))
    else:
        exp["next_loader_indices"] = None
    exp["forward_output"] = None
    exp["forward_output_not_run"] = _FORWARD_NOT_RUN
    exp["cut_fingerprint_sha256"] = _hl.sha256(_json.dumps(fp, sort_keys=True, default=str).encode()).hexdigest()
    exp["not_refreshed"] = failed
    return exp, ("cut" if not failed else "cut (incomplete: %s)" % ", ".join(sorted(failed))), method
"""


def barrier_program() -> str:
    """A no-op execution. A kernel runs executions one at a time, so its
    return proves every execution submitted to the kernel before it has
    finished, including one whose client gave up waiting (a timed-out
    request, or a caller whose process died)."""
    return _wrap(f'import json as _j\nprint({MARK!r} + _j.dumps({{"barrier": True}}))')


def fingerprint_program(views: dict[str, Any] | None = None, workspace_root: str = "data") -> str:
    """The state fingerprint of the runtime's `__main__`, read-only. `views` is
    the capsule's view manifest, so `views_intact` checks the relationships
    the capture recorded."""
    return _wrap(_prelude(manifest=False, fixture=False, oracles=False) + _CAPTURE_FILTER + _STATE_FP_SRC + f'''
_g = sys.modules["__main__"].__dict__
_t0 = _time.perf_counter()
_fp = _state_fingerprint(_g, _json.loads({json.dumps(views or {})!r}), workspace_root={workspace_root!r})
_fp["fingerprint_ms"] = (_time.perf_counter() - _t0) * 1000
print({MARK!r} + _json.dumps(_fp, default=str))
''')


def preflight_program(dest_desc: dict[str, Any]) -> str:
    """Run ON THE SOURCE: the portability manifest against the destination's
    own description. Reports; the controller decides.

    Process effects are diffed BEFORE the manifest runs, because the
    manifest's round-trip validator loads pickles, which can import modules,
    and those imports would otherwise be reported as the session's effects.
    (It loads only pickles its scan clears: one that would re-open a file,
    touch another process resource or leave a finalizer to run here is judged
    without being loaded, so the preflight cannot damage the source it is
    judging. See `capsule.manifest._load_hazard`.)
    The manifest runs under the same submodule re-attachment the capture
    uses, so its pickle probe sees the dump the capture will actually make.

    The storage adapter's capture-side prediction is read here too
    (`views`): `collect_view_manifest` runs over a COPY of exactly the names
    the capture will dump (the captured names minus the rejected ones), so
    the hidden bases it injects for unbound view bases land in that copy,
    never in `__main__`, and a `finally` removes them from the copy as well.
    The names of `__main__` that carry the hidden-base prefix are listed
    before and after, so a leak would be seen (`hidden_left_in_main`). And
    the gradients the capsule will not carry are listed (`grads`), by the
    same code the fingerprint declares them with.
    """
    return _wrap(_prelude(fixture=False, oracles=False) + _CAPTURE_FILTER + _GRADS_SRC + f'''
_g = sys.modules["__main__"].__dict__
_t0 = _time.perf_counter()
from capsule.manifest import build_manifest, Destination, Adapters, imported_distributions, process_effects
from capsule.optimizer_reattach import reattach, detach
from capsule.storage_sharing import collect_view_manifest, HIDDEN_BASE_PREFIX
_desc = _json.loads({json.dumps(dest_desc)!r})
_base = _g.get("__handoff_baseline__")
_effects = process_effects(_base) if _base is not None else None
_names = {{k: v for k, v in _g.items() if _captured(k, v)}}
_modules = sorted(k for k, v in _g.items() if isinstance(v, _types.ModuleType) and not k.startswith("_"))
# unrepairable=("",): this controller has no repair path, so every imported
# distribution the destination lacks is unrepairable.
_dest = Destination(has_cuda=bool(_desc["has_cuda"]), python_minor=_desc["python_minor"],
                    dill_version=_desc["dill_version"], packages=_desc["packages"], unrepairable=("",))
_imported = imported_distributions()
_pairs = reattach()
try:
    _pm = build_manifest(_names, _dest, Adapters(device_remap=True, optimizer_reattach=True, storage_sharing=True),
                         source_imported=_imported)
finally:
    detach(_pairs)
_rejected = [nv.name for nv in _pm.rejected]
_after = _pm.without(_rejected)
# What the capture will dump: the captured names minus the rejected ones. A
# value that IS the module dict is skipped, as the capture's walk skips it.
_dumps = {{k: v for k, v in _names.items() if k not in _rejected and v is not _g}}
_hidden_before = sorted(k for k in _g if k.startswith(HIDDEN_BASE_PREFIX))
_views = {{"records": None, "predicted_failures": [], "unsupported": [], "injected_bases": [], "error": None}}
_vns = dict(_dumps)
try:
    _vm = collect_view_manifest(_vns)
    _views.update(records=len(_vm.views), predicted_failures=list(_vm.predicted_failures),
                  unsupported=list(_vm.unsupported), injected_bases=list(_vm.injected_bases))
except Exception as _e:  # noqa: BLE001
    _views["error"] = ("%s: %s" % (type(_e).__name__, _e))[:300]
finally:
    # Undo the injection: every hidden base the collection added, whether it
    # finished or not.
    for _k in [k for k in _vns if k.startswith(HIDDEN_BASE_PREFIX) and k not in _dumps]:
        _vns.pop(_k, None)
    del _vns
_views["hidden_left_in_main"] = sorted(set(k for k in _g if k.startswith(HIDDEN_BASE_PREFIX)) - set(_hidden_before))
_carried = [p for p, _s, t in _grad_targets(_dumps) if t.grad is not None]
_grads = {{"carried": _carried, "not_carried": _grads_uncarried(_dumps)}}
del _dumps
print({MARK!r} + _json.dumps({{
    "manifest": _pm.to_json(), "rejected": _rejected,
    "unknown": [{{"name": nv.name, "reason": nv.reason, "type": nv.type_name}} for nv in _pm.unknown],
    "admissible": _pm.admissible, "admissible_after_exclusion": _after.admissible,
    "fully_validated_after_exclusion": _after.fully_validated, "blocking": _pm.blocking,
    "effects": _effects, "effects_available": _base is not None,
    "modules_not_captured": _modules, "views": _views, "grads": _grads,
    "source": {{"python_minor": "%d.%d" % sys.version_info[:2], "dill_version": __import__("dill").__version__}},
    "preflight_ms": (_time.perf_counter() - _t0) * 1000}}, default=str))
''')


def capture_program(workspace_root: str = "data", exclude: list[str] | None = None,
                    classified: list[str] | None = None) -> str:
    """Dump the source's `__main__` into a capsule.

    `exclude` is the preflight's list of rejected names. They are removed
    INSIDE `_dump_filtered_session`'s stash, and put back by its `finally`, so
    the capsule lacks them and the source still has them. The first version
    popped them from the live namespace before the dump, outside any
    restoration, so a refused-then-aborted switch had already deleted user
    state from the runtime it promised to leave untouched.

    `classified` is every name the preflight judged. A name present now that
    the preflight never saw is reported (`names_not_in_preflight`), since it
    rides in a capsule nobody classified.
    """
    return _wrap(_capture_src(workspace_root, exclude, classified))


def _capture_src(workspace_root: str = "data", exclude: list[str] | None = None,
                 classified: list[str] | None = None) -> str:
    """The capture program's source before `_wrap`. Separate so a test can
    build a deliberately broken variant and show the evidence catches it."""
    return (_prelude(manifest=False, fixture=False, oracles=False) + _CAPTURE_FILTER
            + _STATE_FP_SRC + _RNG_SRC + f"_FORWARD_NOT_RUN = {FORWARD_NOT_RUN!r}\n" + _EXPECT_SRC + f'''
_g = sys.modules["__main__"].__dict__
_t0 = _time.perf_counter()
import tempfile as _tf
from capsule.storage_sharing import collect_view_manifest
from capsule.optimizer_reattach import reattach, detach
import dill
_EXCLUDE = frozenset({list(exclude or [])!r})
_CLASSIFIED = {None if classified is None else list(classified)!r}
_before = {{k: id(v) for k, v in _g.items()}}
# Dump the REAL __main__ with the unpicklable ambient names temporarily
# removed. A synthetic filtered COPY of the module does not work, and the
# reason is worth stating: a function defined in the kernel carries
# __globals__ that points at the real __main__ dict. dill follows that
# reference no matter what the copy contains, walks back into the live
# module, and reaches IPython's history manager, which holds an
# sqlite3.Connection and does not pickle. The only filter that holds is one
# applied to the dict dill actually walks. This is also what the platform
# capture does: it dumps __main__ under an exclusion list.
#
# Everything that changes the dict happens between the pops and the finally:
# the preflight's exclusions, and the view manifest, which may INJECT a hidden
# base name for a view whose base is not otherwise bound. Collected here, the
# manifest describes exactly the dumped namespace, the injected base rides in
# the capsule (a dunder name added after the filter ran), and the finally
# removes it again so the source is left as it was.
#
# THE CUT is inside the same stash, immediately before the dump: the state
# fingerprint (over exactly the names that are dumped) and the RNG envelope.
# The controller runs this only after closing admission, draining and a
# barrier, so no routed work can land between the fingerprint and the bytes.
# Picklers and reducers are arbitrary code and may draw random numbers, so
# every stream is put back to the envelope after the dump: the source stays
# at the state the fingerprint describes (the platform does the same). The
# fingerprint is then taken AGAIN (`fingerprint_after`, still inside the
# stash): if anything this program ran changed the source, the controller
# sees it (`capture_mutated_source`) instead of a report that the source was
# left untouched because its bindings kept their ids.
#
# The gradients (`.grad` does not pickle) are dumped in the same session under
# an injected dunder name, after the cut and removed again after the dump.
#
# The validation's EXPECTATION is refreshed at the cut too. It used to be the
# one the seed recorded, which is only right when nothing ran between the seed
# and the switch; after user training every parameter, step and scheduler
# expectation is stale and a correct restore fails validation. It is computed
# from STRUCTURE only (`_cut_expectation`, see `_EXPECT_SRC`): no forward, no
# hook, no backward, no step runs on the source, forked or not, and no helper
# is looked up in `__main__`. The forward output is declared not run
# (`forward_output_not_run`), so the destination's forward row is declared
# with that reason instead of compared with a reference the source would have
# had to run the model to take. A namespace without a recorded expectation
# keeps none (`expectation_source` "recorded").
def _dump_filtered_session(_path, _exclude=_EXCLUDE, _captured=_captured, _keep=_KEEP_DUNDER,
                           _essential=_ESSENTIAL_DUNDER, _dill=dill, _collect=collect_view_manifest,
                           _main=sys.modules["__main__"], _fingerprint=_state_fingerprint,
                           _rng_capture=_rng_capture, _rng_apply=_rng_apply, _expect=_cut_expectation,
                           _grads=_grads_collect):
    _gd = _main.__dict__
    _stash = {{}}
    _injected = []
    _env = None
    try:
        for _k in list(_gd):
            _v = _gd[_k]
            if _k in _keep or _k in _essential:
                continue
            if _k in _exclude or not _captured(_k, _v):
                _stash[_k] = _gd.pop(_k)    # excluded, ambient, modules, dunders and helpers
        _dumped = sorted(k for k in _gd if k not in _keep and k not in _essential)
        _vm = _collect(_gd)
        _injected = list(_vm.injected_bases)
        _vj = _vm.to_json()
        _fp = _fingerprint(_gd, _vj, workspace_root={workspace_root!r})          # THE CUT
        _env = _rng_capture()
        try:
            _exp = _expect(_gd, _fp)
        finally:
            _rng_apply(_env)      # a sampler without its own generator draws from the global stream
        _carried = _grads({{k: v for k, v in _gd.items() if _captured(k, v)}})
        try:
            if _carried:
                _gd["__handoff_grads__"] = _carried
            _dill.dump_session(_path, main=_main, byref=True)
        finally:
            _gd.pop("__handoff_grads__", None)
            _rng_apply(_env)
        _fp_after = _fingerprint(_gd, _vj, workspace_root={workspace_root!r})
    finally:
        for _k in _injected:
            _gd.pop(_k, None)
        _gd.update(_stash)
        if _env is not None:
            _rng_apply(_env)
    return _vm, _dumped, _fp, _fp_after, _env, _exp, len(_carried)

_fd, _session_path = _tf.mkstemp(prefix="handoff-session-", suffix=".pkl")
os.close(_fd)
try:
    _pairs = reattach()
    try:
        _vm, _dumped, _fp, _fp_after, _env, (_expectation, _expectation_source, _expectation_method), _n_grads = \\
            _dump_filtered_session(_session_path)
    finally:
        detach(_pairs)
    _after = {{k: id(v) for k, v in _g.items()}}
    _excluded = sorted(k for k in _EXCLUDE if k in _before)
    _buf = _io.BytesIO()
    with _tar.open(fileobj=_buf, mode="w:gz") as _t:
        _t.add(_session_path, arcname="session.pkl")
        _man = _json.dumps({{"views": _vm.to_json(), "portability": {{"excluded": _excluded}},
                             "expectation": _expectation, "expectation_source": _expectation_source,
                             "expectation_method": _expectation_method,
                             "fingerprint": _fp, "rng_envelope": _env, "grads_carried": _n_grads}},
                           default=str).encode()
        _ti = _tar.TarInfo("manifests.json"); _ti.size = len(_man); _t.addfile(_ti, _io.BytesIO(_man))
        if os.path.isdir({workspace_root!r}):
            _t.add({workspace_root!r}, arcname="workspace")
finally:
    try:
        os.remove(_session_path)
    except OSError:
        pass
_blob = _buf.getvalue()
print({MARK!r} + _json.dumps({{"b64": _b64.b64encode(_blob).decode(), "sha256": _hl.sha256(_blob).hexdigest(),
                             "bytes": len(_blob), "excluded": _excluded,
                             "excluded_still_present": sorted(k for k in _excluded if _after.get(k) == _before[k]),
                             "source_names_before": sorted(_before), "source_names_after": sorted(_after),
                             "source_names_equal": set(_before) == set(_after),
                             "source_bindings_identical": _before == _after,
                             "dumped_names": _dumped,
                             "names_not_in_preflight": (None if _CLASSIFIED is None
                                                        else sorted(set(_dumped) - set(_CLASSIFIED))),
                             "expectation_source": _expectation_source, "expectation_method": _expectation_method,
                             "fingerprint": _fp, "fingerprint_after": _fp_after, "grads_carried": _n_grads,
                             "rng_carried": {{"numpy": _env["numpy"] is not None,
                                             "torch_cpu": _env["torch_cpu"] is not None,
                                             "cuda": bool(_env["cuda"]["initialized"])}},
                             "capture_ms": (_time.perf_counter() - _t0) * 1000}}, default=str))
''')


#: Where `upload_chunk_program` writes when no path is given. Kept for an
#: earlier harness that loads that fixed path itself; the controller always
#: passes a per-migration path (`blob_path`).
DEFAULT_BLOB_PATH = "/tmp/handoff.blob"


def blob_path(mig: str, sha: str) -> str:
    """The destination-side file a migration's capsule is uploaded to: one per
    migration and capsule, so two migrations whose kernels share a host (the
    local test double) never append to, or restore, each other's bytes."""
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in mig)[:80]
    return f"/tmp/handoff-{safe}-{sha[:16]}.blob"


def upload_chunk_program(b64_chunk: str, first: bool, path: str = DEFAULT_BLOB_PATH) -> str:
    mode = "wb" if first else "ab"
    return _wrap(f'''
import base64 as _b, json as _j
with open({path!r}, {mode!r}) as _f:
    _f.write(_b.b64decode({b64_chunk!r}))
print({MARK!r} + _j.dumps({{"ok": True}}))
''')


# The destination-aware storage handler, as source text (see restore_program).
_REMAP_SRC = r"""
def _remap_install(needed):
    rep = {"registered": False, "remapped": 0, "locations": {}, "cuda_available": None}
    if not needed:
        rep["reason"] = "the session pickle references no torch storage"
        return rep, None
    try:
        import torch
        import torch.serialization as ser
    except Exception as e:  # noqa: BLE001
        rep["reason"] = "torch unavailable here: %s" % type(e).__name__
        return rep, None
    rep["cuda_available"] = bool(torch.cuda.is_available())

    def _tagger(obj):
        return None

    def _deserializer(obj, location):
        if not isinstance(location, str) or not location.startswith("cuda"):
            return None
        try:
            index = int(location.split(":", 1)[1]) if ":" in location else 0
        except Exception:  # noqa: BLE001
            index = 0
        try:
            if torch.cuda.is_available() and index < torch.cuda.device_count():
                return None      # the tagged device exists here: torch keeps the placement
        except Exception:  # noqa: BLE001
            pass
        rep["remapped"] += 1
        rep["locations"][location] = rep["locations"].get(location, 0) + 1
        to_cpu = getattr(obj, "cpu", None)
        return to_cpu() if to_cpu is not None else obj

    snapshot = list(ser._package_registry)
    ser.register_package(15, _tagger, _deserializer)
    rep["registered"] = True
    return rep, (ser, snapshot)

def _remap_uninstall(handle):
    # register_package has no unregister API; the registry is restored exactly.
    if handle is not None:
        handle[0]._package_registry[:] = handle[1]
"""


def restore_program(sha: str, fault: str | None = None, blob: str = DEFAULT_BLOB_PATH) -> str:
    """Load the capsule on the destination (uploaded to `blob`).

    Order, and why:
      1. sha256 of the uploaded bytes, before dill executes anything. The
         uploaded file is removed once read.
      2. The workspace is extracted from a FRESH temporary directory (a fixed
         path shared by every local kernel let one run's files leak into the
         next).
      3. A destination-aware storage handler is registered before
         `dill.load_session`: a capsule captured on a CUDA source tags every
         storage `cuda:N`, and torch's own handler raises on a host without
         that device. Storages whose device exists here keep their placement;
         the others are remapped to cpu and COUNTED (`remap.remapped`). The
         registry is restored exactly afterwards.
      4. Every recorded view is repaired. A record the adapter reports as
         failed means a relationship the capsule promised is not there, so
         the restore returns `ok: False`, reason `views_not_restored`, with
         the failures. (The first version returned `ok: true` with the
         failures in the report, and the controller committed.)
      5. The carried gradients are reattached, by path, AFTER the view
         repair (which may rebind a parameter). One that cannot be is
         `grads_not_restored`.
      6. The RNG envelope is applied (CUDA only where the devices exist,
         otherwise a declared drop).
      7. The state fingerprint is taken, for the controller to compare with
         the one recorded at the cut.
    """
    return _wrap(_prelude(manifest=False, fixture=False, oracles=False) + _CAPTURE_FILTER + _STATE_FP_SRC
                 + _RNG_SRC + _REMAP_SRC + f"""
_g = sys.modules["__main__"].__dict__
_t0 = _time.perf_counter()
with open({blob!r}, "rb") as _bf:
    _blob = _bf.read()
try:
    os.remove({blob!r})
except OSError:
    pass
_got = _hl.sha256(_blob).hexdigest()
if _got != {sha!r}:
    print({MARK!r} + _json.dumps({{"ok": False, "reason": "checkpoint_transfer_failed", "detail": "sha256 mismatch before dill executed"}})); raise SystemExit(0)
if {fault!r} == "insufficient_memory":
    import numpy as _np
    try:
        _hog = _np.ones((1 << 34,), dtype=_np.float64)   # 128 GiB, must fail on any sandbox here
        _hog[:] = 1.0
    except MemoryError as _e:
        print({MARK!r} + _json.dumps({{"ok": False, "reason": "insufficient_memory", "detail": "MemoryError while materialising state"}})); raise SystemExit(0)
if {fault!r} == "missing_package":
    try:
        import clusy_dependency_that_does_not_exist  # noqa
    except ModuleNotFoundError as _e:
        print({MARK!r} + _json.dumps({{"ok": False, "reason": "RESTORE_MISSING_MODULE", "detail": str(_e)}})); raise SystemExit(0)
import tempfile as _tf, shutil as _sh
_in = _tf.mkdtemp(prefix="handoff_in_")
try:
    with _tar.open(fileobj=_io.BytesIO(_blob), mode="r:gz") as _t:
        _t.extractall(_in)
    _m = _json.load(open(os.path.join(_in, "manifests.json")))
    if os.path.isdir(os.path.join(_in, "workspace")):
        if os.path.isdir("data"): _sh.rmtree("data")
        _sh.copytree(os.path.join(_in, "workspace"), "data")
    import dill
    # Load into the kernel's real __main__, which is what the platform restore
    # does. A scratch ModuleType cannot be the target: the capsule is dumped from
    # the source kernel's real __main__, dill classifies any module reachable
    # through sys.modules as "imported", and its session loader resolves an
    # imported main by NAME on the destination. It updates sys.modules["__main__"]
    # and then asserts that is the module it loaded, so a scratch target fails on
    # a bare AssertionError.
    #
    # This does not clobber the destination kernel. dill applies the saved state
    # as a dict update, never a replace, and it drops __builtins__, __loader__ and
    # IPython's singletons from that state before the pickle is written. The
    # source-side filter already removed ambient names, modules and helpers, so
    # what lands here is the captured program state and nothing else.
    _session = os.path.join(_in, "session.pkl")
    with open(_session, "rb") as _f:
        _needs_torch = b"torch" in _f.read()
    _remap, _remap_handle = _remap_install(_needs_torch)
    try:
        dill.load_session(_session, main=sys.modules["__main__"])
    except Exception as _e:
        print({MARK!r} + _json.dumps({{"ok": False, "reason": "checkpoint_restore_failed", "detail": f"{{type(_e).__name__}}: {{str(_e)[:200]}}", "remap": _remap}})); raise SystemExit(0)
    finally:
        _remap_uninstall(_remap_handle)
finally:
    _sh.rmtree(_in, ignore_errors=True)
# The gradients ride under an injected name; take them out of __main__ at once
# so nothing else (the view repair, the fingerprint) sees that name.
_carried_grads = _g.pop("__handoff_grads__", None)
# Imported inside a function so the two names do not land in __main__. A
# top-level import here would leave them in the namespace, and the NEXT
# capture would record them by reference against a module the destination has
# no reason to have. That is the failure E13 hit at hop 2. (The whole program
# now runs in a private dict, which makes this doubly true.)
def _apply_views(_gd, _views):
    from capsule.storage_sharing import ViewManifest, apply_view_manifest
    return apply_view_manifest(_gd, ViewManifest.from_json(_views))
_vr = _apply_views(_g, _m["views"])
if _vr.get("failed"):
    print({MARK!r} + _json.dumps({{"ok": False, "reason": "views_not_restored",
                                  "detail": "; ".join(str(f.get("path")) + ": " + str(f.get("reason")) for f in _vr["failed"])[:300],
                                  "views": _vr, "remap": _remap}}, default=str)); raise SystemExit(0)
_grep = _grads_apply(_g, _carried_grads)
if _grep["failed"]:
    print({MARK!r} + _json.dumps({{"ok": False, "reason": "grads_not_restored",
                                  "detail": "; ".join(str(f["path"]) + ": " + f["reason"] for f in _grep["failed"])[:300],
                                  "views": _vr, "grads": _grep, "remap": _remap}}, default=str)); raise SystemExit(0)
_env = _m.get("rng_envelope")
_rng = _rng_apply(_env) if _env else {{"applied": False, "reason": "the capsule carries no RNG envelope"}}
_g["__handoff_expectation__"] = _m["expectation"]
_fp = _state_fingerprint(_g, _m["views"])
print({MARK!r} + _json.dumps({{"ok": True, "views": _vr, "grads": _grep, "remap": _remap, "rng": _rng,
                             "fingerprint": _fp, "restore_ms": (_time.perf_counter() - _t0) * 1000}}, default=str))
""")


#: How the controller validates a destination (see `oracles.verify`).
#: "isolated": every oracle runs in a forked child of the kernel (or, where no
#: fork can isolate model code, the model-executing checks run on an
#: in-process copy), and the validation program compares the state and the
#: workspace before and after (`side_effects`), so anything that escaped fails
#: the switch. A module-level name so a test can switch it to "inplace" or
#: "copy" and show the side-effect and commit-boundary checks catch what those
#: modes change.
VALIDATION_CONTINUATION = "isolated"


def recovery_check_program(source_device: str = "cpu", dest_device: str = "cpu",
                           continuation: str | None = None, views: dict[str, Any] | None = None,
                           workspace_root: str = "data") -> str:
    """After recovery, every oracle, non-mutating, with the same before/after
    side-effect check as the handoff's validation.

    The first version filtered to seven checks it believed invariant, because
    the validation's continuation step had moved the parameters and the data
    cursor, and it still ran the full mutating oracle before filtering, so the
    check itself advanced the committed state once more. Validation no longer
    writes (`continuation="isolated"`), so after a restart every check
    applies, and this one writes nothing either (its `side_effects` says so,
    measured)."""
    return verify_program(source_device, dest_device, continuation=continuation, views=views,
                          workspace_root=workspace_root)


def verify_program(source_device: str = "cpu", dest_device: str = "cpu",
                   continuation: str | None = None, views: dict[str, Any] | None = None,
                   workspace_root: str = "data") -> str:
    """The oracles on this runtime's `__main__`, bracketed by two state
    fingerprints taken in the SAME execution.

    `passed`/`total` count the checks that RAN; a check that did not run is
    listed under `declared` with its reason and is neither a pass nor a
    failure (`summary` says, e.g., "11 checks run, 11 passed, 1 declared not
    run"). `side_effects` compares the whole fingerprint (plain values, deep
    object digests, tensors, modules, optimizers, RNG streams, gradients,
    views, the workspace files) immediately before and after the oracles:
    `changed` names every field that differs. A forked child contains memory
    but not the filesystem, and an in-process copy (where no fork is
    possible) does not contain code, so this measured comparison, not the
    mode, is what shows validation left the runtime as it found it; the
    controller aborts on any difference (`validation_side_effect`), and on a
    comparison that could not be made. `views` is the capsule's view
    manifest, so `views_intact` is part of both fingerprints."""
    mode = continuation or VALIDATION_CONTINUATION
    return _wrap(_prelude(manifest=False) + _CAPTURE_FILTER + _STATE_FP_SRC + f'''
_g = sys.modules["__main__"].__dict__
_views = _json.loads({json.dumps(views or {})!r})
_iso = _oracles.fork_unsafe_reason() if {mode!r} == "isolated" else None

def _vfp():
    try:
        return _state_fingerprint(_g, _views, workspace_root={workspace_root!r}), None
    except Exception as _e:  # noqa: BLE001
        return None, ("%s: %s" % (type(_e).__name__, _e))[:300]

def _vdiff(a, b):
    # WHICH keys of a field changed, for the abort detail (files, for the
    # workspace).
    if isinstance(a, dict) and isinstance(b, dict):
        if "files" in a or "files" in b:
            a, b = a.get("files") or {{}}, b.get("files") or {{}}
        return sorted(str(k) for k in set(a) | set(b) if a.get(k) != b.get(k))[:12]
    return "value differs"

_fp0, _e0 = _vfp()
try:
    _rows = [r.to_dict() for r in _oracles.verify(_g, _g["__handoff_expectation__"],
             destination_device={dest_device!r}, source_device={source_device!r}, same_hardware=False,
             continuation={mode!r})]
finally:
    _fp1, _e1 = _vfp()
_se = {{"checked": _fp0 is not None and _fp1 is not None, "error": _e0 or _e1, "changed": None, "details": {{}}}}
if _se["checked"]:
    _se["changed"] = sorted(k for k in set(_fp0) | set(_fp1) if _fp0.get(k) != _fp1.get(k))
    _se["details"] = {{k: _vdiff(_fp0.get(k), _fp1.get(k)) for k in _se["changed"]}}
    _se["fields_compared"] = sorted(set(_fp0) | set(_fp1))
    _se["before_sha256"] = _hl.sha256(_json.dumps(_fp0, sort_keys=True, default=str).encode()).hexdigest()
    _se["after_sha256"] = _hl.sha256(_json.dumps(_fp1, sort_keys=True, default=str).encode()).hexdigest()
_ran = [r for r in _rows if r["mode"] != "declared"]
_decl = [{{"name": r["name"], "detail": r["detail"]}} for r in _rows if r["mode"] == "declared"]
_passed = sum(1 for r in _ran if r["ok"] is True)
print({MARK!r} + _json.dumps({{"passed": _passed, "total": len(_ran),
                             "failed": [r["name"] for r in _ran if r["ok"] is not True],
                             "declared": _decl,
                             "summary": "%d checks run, %d passed, %d declared not run%s" % (
                                 len(_ran), _passed, len(_decl),
                                 (" (" + ", ".join(d["name"] for d in _decl) + ")") if _decl else ""),
                             "continuation": {mode!r}, "isolated": {mode!r} == "isolated" and _iso is None,
                             "not_isolated_because": _iso, "side_effects": _se, "rows": _rows}}, default=str))
''')


# ---------------------------------------------------------------------------
# The commit-boundary comparison (host side)
# ---------------------------------------------------------------------------

#: Fingerprint fields that must be EQUAL at every commit boundary.
BOUNDARY_FIELDS = ("names", "plain", "objects", "aliases", "tensors", "modules", "optimizers", "schedulers",
                   "stateful", "loaders", "grads", "rng", "views", "workspace")


def _path_text(path: Any) -> str:
    """A storage-adapter path (`["model", "param", "fc1.weight"]`) as one
    dotted string for a reason code."""
    if isinstance(path, (list, tuple)):
        return ".".join(str(p) for p in path)
    return str(path)


def _diff(a: Any, b: Any, limit: int = 12) -> Any:
    """Which keys differ, for the abort detail: enough to say WHAT changed."""
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a) | set(b), key=str)
        return [str(k) for k in keys if a.get(k) != b.get(k)][:limit]
    return "value differs"


def _cuda_rng_verdict(src: dict | None, dst: dict | None) -> str:
    """CUDA RNG across a boundary: equal when carried and restored; declared
    otherwise, never silently equal.
      absent         neither end has CUDA state
      not_carried    the source never initialized CUDA (e.g. cpu -> T4): the
                     destination's streams are its own
      declared_drop  the source carried CUDA state and the destination cannot
                     hold it (T4 -> cpu)
      equal / mismatch  carried and restored: every device compared"""
    src, dst = src or {}, dst or {}
    src_devs = src.get("devices") if src.get("initialized") else None
    if not src_devs:
        return "not_carried" if dst.get("available") else "absent"
    dst_devs = dst.get("devices") if dst.get("initialized") else None
    if not dst.get("available") or not dst_devs or len(dst_devs) < len(src_devs):
        return "declared_drop"
    by_index = {d["index"]: d for d in dst_devs}
    same = all(by_index.get(d["index"], {}).get("sha256") == d["sha256"]
               and by_index.get(d["index"], {}).get("draws") == d["draws"] for d in src_devs)
    return "equal" if same else "mismatch"


def _split_libm_draws(a: dict[str, Any], b: dict[str, Any]) -> dict[str, str]:
    """Take the Gaussian draws out of both RNG fingerprints before they are compared.

    A Gaussian draw is the generator's next integers passed through the
    platform's libm (log, sqrt, exp), so macOS and glibc can differ in the
    last bit while the generator state is identical: the same distinction as
    TF32 recomputation on a GPU. The state hash and the uniform draws (pure
    integer arithmetic) stay in the exact comparison; the Gaussian draws are
    reported per stream as `equal` or `platform_arithmetic`, never a reason to
    abort and never counted as equal when they are not."""
    out: dict[str, str] = {}

    def take(name: str, ea: Any, eb: Any) -> None:
        if isinstance(ea, dict) and "libm_draws" in ea:
            da, db = ea.pop("libm_draws"), (eb or {}).pop("libm_draws", None) if isinstance(eb, dict) else None
            out[name] = "equal" if da == db else "platform_arithmetic"

    a["numpy"], b["numpy"] = dict(a.get("numpy") or {}), dict(b.get("numpy") or {})
    take("numpy", a["numpy"], b["numpy"])
    na, nb = {k: dict(v) for k, v in (a.get("named") or {}).items()}, {k: dict(v) for k, v in (b.get("named") or {}).items()}
    for k in na:
        take(f"named:{k}", na[k], nb.get(k))
    for k in nb:  # a destination-only entry still loses its libm draws before the (then unequal) comparison
        nb[k].pop("libm_draws", None)
    if "named" in a or "named" in b:
        a["named"], b["named"] = na, nb
    return out


def compare_fingerprints(src: dict[str, Any], dst: dict[str, Any]) -> dict[str, Any]:
    """Compare a destination fingerprint with the one taken at the cut.

    Every field in BOUNDARY_FIELDS must be equal (the carried gradients
    included), except that a numpy or torch stream the source never loaded
    is not compared (`not_loaded_at_source`). Declared differences are
    recorded, never a reason to abort, and are exactly four: CUDA RNG when
    either end has no CUDA, tensor DEVICE (the bytes are compared through the
    other fields, on cpu), values the fingerprint could not digest
    (`unhashable`, from either side: their markers are compared, their values
    are not), and the gradients the capsule does not carry
    (`grads_not_carried`, the SOURCE's list: they arrive as None or unshared,
    so they are named and never compared as equal). A gradient the
    destination lists that the cut did not is not a declared drop but a
    change, and a mismatch."""
    mismatched, details, declared = [], {}, {}
    for f in BOUNDARY_FIELDS:
        a, b = src.get(f), dst.get(f)
        if f == "rng":
            a, b = dict(a or {}), dict(b or {})
            for stream in ("numpy", "torch_cpu"):
                if not (a.get(stream) or {}).get("loaded"):
                    declared[f"rng_{stream}"] = "not_loaded_at_source"
                    a.pop(stream, None)
                    b.pop(stream, None)
            libm = _split_libm_draws(a, b)
            if libm:
                declared["rng_libm_draws"] = libm
        if a != b:
            mismatched.append(f)
            details[f] = _diff(a, b)
    cuda = _cuda_rng_verdict(src.get("rng_cuda"), dst.get("rng_cuda"))
    if cuda == "mismatch":
        mismatched.append("rng_cuda")
    declared["rng_cuda"] = cuda
    sd, dd = src.get("devices") or {}, dst.get("devices") or {}
    moved = {k: [sd.get(k), dd.get(k)] for k in sorted(set(sd) | set(dd)) if sd.get(k) != dd.get(k)}
    if moved:
        declared["devices"] = moved
    unhashable = sorted(set(src.get("unhashable") or []) | set(dst.get("unhashable") or []))
    if unhashable:
        declared["unhashable"] = unhashable
    gnc_src, gnc_dst = src.get("grads_not_carried") or {}, dst.get("grads_not_carried") or {}
    if gnc_src:
        declared["grads_not_carried"] = sorted(gnc_src)
    appeared = sorted(set(gnc_dst) - set(gnc_src))
    if appeared:
        mismatched.append("grads_not_carried")
        details["grads_not_carried"] = appeared[:12]
    return {"equal": not mismatched, "mismatched": mismatched, "details": details, "declared": declared}


# ---------------------------------------------------------------------------
# The controller
# ---------------------------------------------------------------------------

@dataclass
class CrashPoint:
    """Kill the controller at a phase boundary. `after_journal=False` models a
    lost acknowledgement: the side effect happened, the journal was not
    written, and the restarted controller must discover that.

    `CrashPoint(DEST_CREATED, after_journal=False)` is the one point that is
    not a phase: the destination exists at the provider and its id has not
    been journaled yet (the create-response-to-journal window)."""
    phase: str
    after_journal: bool = True


def _os_kill() -> None:
    os._exit(137)


class Controller:
    def __init__(self, api: Api, journal: Journal, blobs: Path, crash: CrashPoint | None = None,
                 log=None, *, strict_unknown: bool = False, kill=None, drain_timeout: float = 600.0,
                 drain_poll: float = 0.05, join_timeout: float = 900.0):
        self.api, self.j, self.blobs, self.crash = api, journal, blobs, crash
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.log = log or (lambda *a: None)
        self.timings: dict[str, float] = {}
        self.owner = f"ctl-{os.getpid()}-{time.time_ns() % 1_000_000}"
        #: Block on `unknown` names and process effects, not only report them.
        self.strict_unknown = strict_unknown
        #: How a crash point kills the controller. The default is a real
        #: SIGKILL-equivalent; an in-process harness passes a callable that
        #: raises `ControllerKilled`.
        self.kill = kill or _os_kill
        #: Evidence from this run's preflight and capture, returned by migrate.
        self.reports: dict[str, Any] = {}
        #: How long the drain waits for executions admitted before the close
        #: (then ABORTED, `drain_timeout`), and how often it looks.
        self.drain_timeout, self.drain_poll = drain_timeout, drain_poll
        #: How long a duplicate request waits for the lease holder to finish.
        self.join_timeout = join_timeout

    # -- crash injection -----------------------------------------------------
    def _maybe_crash(self, phase: str, after_journal: bool) -> None:
        if self.crash and self.crash.phase == phase and self.crash.after_journal == after_journal:
            self.log(f"  !! controller killed at {phase} ({'after' if after_journal else 'before'} journal write)")
            sys.stdout.flush(); sys.stderr.flush()
            self.kill()

    def _advance(self, mig: str, phase: str, **cols) -> None:
        self._maybe_crash(phase, after_journal=False)
        self.j.advance(mig, phase, **cols)
        self._maybe_crash(phase, after_journal=True)

    def _upload(self, pid: str, blob: bytes, chunk: int = 96_000, path: str = DEFAULT_BLOB_PATH) -> None:
        """The execute route caps `code` at 200k characters, so the capsule
        travels as base64 chunks appended to a file on the destination."""
        b64 = base64.b64encode(blob).decode()
        for i in range(0, len(b64), chunk):
            self.api.witness(pid, upload_chunk_program(b64[i:i + chunk], first=(i == 0), path=path))

    # -- routing / admission / fencing ---------------------------------------
    def route(self, mig: str, code: str) -> dict:
        """Run `code` on the authoritative runtime of the workload `mig`
        belongs to, through the admission gate (see the module doc, THE
        ADMISSION CONTRACT).

        The runtime is resolved by lineage, not by the row alone: a route
        through an earlier hop's id follows the chain to the current
        authority, and is refused if any migration out of the runtime it
        resolves to has its gate closed. The check and the registration are
        one journal transaction; the execution itself runs outside it, and is
        marked finished in a `finally`, so a raised or timed-out execution
        never holds up a drain forever (the barrier covers one the kernel is
        still running). Raises `AdmissionClosed` (nothing was run) while a
        gate is closed. Returns the runtime it ran on, the chain of ids
        followed, and the program's witness."""
        owner = f"{socket.gethostname()}:{os.getpid()}:{threading.get_ident()}"
        adm = self.j.admit(mig, owner, os.getpid())
        try:
            result = self.api.witness(adm["runtime"], code)
        finally:
            self.j.finish(adm["exec_id"])
        return {"runtime": adm["runtime"], "exec_id": adm["exec_id"], "phase": adm["phase"],
                "chain": adm["chain"], "result": result}

    def execute_routed(self, mig: str, code: str) -> dict:
        """Route by the journal's authoritative runtime, through the admission
        gate; returns the program's witness (see `route`)."""
        return self.route(mig, code)["result"]

    def _drain(self, mig: str, source_pid: str) -> dict[str, Any]:
        """After the close: wait until every execution admitted earlier has
        finished, then run the barrier on the source. An execution whose owning
        process is dead counts as finished once the barrier has returned: it
        either reached the kernel (and the barrier waited for it) or it never
        did. Bounded: past `drain_timeout` the migration aborts, which reopens
        the gate on the source."""
        t0 = time.perf_counter()
        deadline = time.monotonic() + self.drain_timeout
        waited: dict[str, dict] = {}
        dead: set[str] = set()
        while True:
            live = []
            for r in self.j.unfinished(mig, source_pid):
                waited[r["exec_id"]] = r
                if _owner_dead(r["owner"], r["pid"]):
                    dead.add(r["exec_id"])
                else:
                    live.append(r["exec_id"])
            if not live:
                break
            if time.monotonic() > deadline:
                raise HandoffError("drain_timeout", "ADMISSION_CLOSED",
                                   f"{len(live)} execution(s) admitted before the close still unfinished after "
                                   f"{self.drain_timeout:.0f} s: {live[:5]}")
            time.sleep(self.drain_poll)
        drain_s = time.perf_counter() - t0
        t = time.perf_counter()
        self.api.witness(source_pid, barrier_program())
        barrier_ms = (time.perf_counter() - t) * 1000
        if dead:
            self.j.reap(sorted(dead))
        ev = {"waited_on": len(waited), "dead_owner_reaped": sorted(dead), "drain_s": round(drain_s, 4),
              "barrier_ms": round(barrier_ms, 3)}
        self.j.event(mig, "ADMISSION_CLOSED", json.dumps({"drain": ev}))
        self.reports.setdefault("admission", {})["drain"] = ev
        return ev

    def admission_evidence(self, mig: str) -> dict[str, Any]:
        """What the gate did for this migration: its state now, the executions
        it admitted and refused, and the drain (from the journal, so a resumed
        controller reports a drain its predecessor ran)."""
        row = self.j.get(mig) or {}
        drains = [json.loads(e["note"])["drain"] for e in self.j.events(mig)
                  if e["phase"] == "ADMISSION_CLOSED" and e["note"].startswith('{"drain"')]
        closed = any(e["phase"] == "ADMISSION_CLOSED" and '"admission": "closed"' in e["note"]
                     for e in self.j.events(mig))
        return {"state": row.get("admission"), "phase": row.get("phase"), "closed_during_migration": closed,
                "drains": drains, **self.j.inflight_summary(mig)}

    def runtime_accepts(self, pid: str) -> dict:
        try:
            return self.api.witness(pid, touch_program())
        except HandoffError as e:
            return {"accepted": False, "error": e.reason, "detail": e.detail[:120]}

    # -- preflight ---------------------------------------------------------------
    def _preflight(self, mig: str, source_pid: str, dest: str) -> dict[str, Any]:
        """Describe the destination, judge the source against it, decide, and
        journal the report whatever the decision (an abort's evidence matters
        as much as a success's)."""
        desc = self.api.witness(dest, describe_program())
        pf = self.api.witness(source_pid, preflight_program(desc))
        man = pf["manifest"]
        # Policy. Rejected names are excluded, so what must be admissible is
        # the capsule that will actually be written: the manifest WITHOUT
        # them. Its remaining blockers are gates and missing distributions,
        # which no exclusion fixes.
        blocking = [] if pf["admissible_after_exclusion"] else list(pf["blocking"])
        reason, detail = None, []
        # The storage adapter's capture-side prediction (see THE PREFLIGHT
        # POLICY). A program that reported nothing about views is treated as
        # an error, never as "no views": it would mean the check did not run.
        views = pf.get("views") or {"error": "the preflight reported no view prediction"}
        predicted = views.get("predicted_failures") or []
        grads = pf.get("grads") or {}
        if blocking:
            reason = "preflight_" + blocking[0]          # preflight_gate_failed:<gate> | preflight_missing_distribution:<dist>
            src = pf["source"]
            seen = {"python_minor": f"source {src['python_minor']}, destination {desc['python_minor']}",
                    "dill_exact": f"source {src['dill_version']}, destination {desc['dill_version']}"}
            detail = [f"gate {g} failed ({seen.get(g, '')})" for g, ok in man["gates"].items() if not ok]
            detail += [f"{p['distribution']} {p['source_version']} imported on the source, absent on the destination"
                       for p in man["packages_missing"] if p["unrepairable"]]
        elif views.get("error") or views.get("hidden_left_in_main"):
            # Fail closed: a prediction that could not be made, or a
            # preflight that left a hidden base in the source (which must
            # never happen: it runs on a copy), is not a pass.
            reason = "preflight_views_error"
            blocking = detail = [f"view prediction failed: {views.get('error')}" if views.get("error")
                                 else f"hidden bases left in __main__: {views['hidden_left_in_main']}"]
        elif predicted:
            paths = [_path_text(p.get("path")) for p in predicted]
            reason = "preflight_views_unsupported:" + ",".join(paths[:8]) + (f",+{len(paths) - 8}" if len(paths) > 8 else "")
            blocking = [f"views_unsupported:{p}" for p in paths]
            detail = [f"{_path_text(p.get('path'))}: {p.get('reason')} ({p.get('detail', '')})" for p in predicted]
        elif self.strict_unknown and pf["unknown"]:
            blocking = detail = [f"unknown:{u['name']}" for u in pf["unknown"]]
            reason = f"preflight_unknown:{pf['unknown'][0]['name']}"
        elif self.strict_unknown and pf["effects"] is None:
            # No baseline, so nothing is known about what the session did
            # outside its namespace. A strict gate treats "not measured" as
            # blocking, never as "nothing to report".
            blocking = detail = ["process_effects_unavailable"]
            reason = "preflight_process_effects_unavailable"
        elif self.strict_unknown and pf["effects"]:
            blocking = detail = [f"process_effect:{e['effect']}" for e in pf["effects"]]
            reason = f"preflight_process_effect:{pf['effects'][0]['effect']}"
        policy = {
            "decision": "abort" if reason else "proceed",
            "reason": reason,
            "blocking": blocking,
            "blocking_detail": detail,
            "excluded": pf["rejected"],
            "unknown_reported": [u["name"] for u in pf["unknown"]],
            "effects_reported": [e["effect"] for e in (pf["effects"] or [])],
            "views_unsupported_reported": [_path_text(u.get("path")) for u in views.get("unsupported") or []],
            "grads_not_carried": sorted(grads.get("not_carried") or {}),
            "strict_unknown": self.strict_unknown,
            "repair_path": False,
        }
        report = {
            "migration": mig, "at": time.time(), "source_pid": source_pid, "dest_pid": dest,
            "destination": desc, "source": pf["source"],
            "manifest": man, "rejected": pf["rejected"], "unknown": pf["unknown"],
            "admissible": pf["admissible"], "admissible_after_exclusion": pf["admissible_after_exclusion"],
            "fully_validated_after_exclusion": pf["fully_validated_after_exclusion"],
            "effects": pf["effects"], "effects_available": pf["effects_available"],
            "modules_not_captured": pf["modules_not_captured"],
            "views": views, "grads": grads, "grads_not_carried": grads.get("not_carried") or {},
            "preflight_ms": pf["preflight_ms"], "policy": policy,
        }
        raw = json.dumps(report, default=str, sort_keys=True).encode()
        path = self.blobs / f"{mig}.preflight.json"
        path.write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
        # Journaled before the decision is acted on, so an ABORTED row points
        # at the report that explains it.
        self.j.set_columns(mig, preflight_path=str(path), preflight_sha=sha)
        self.j.event(mig, "PREFLIGHT", json.dumps({"decision": policy["decision"], "reason": reason,
                                                   "excluded": policy["excluded"],
                                                   "unknown": policy["unknown_reported"],
                                                   "effects": policy["effects_reported"],
                                                   "views_predicted_failures": [_path_text(p.get("path"))
                                                                                for p in predicted],
                                                   "views_unsupported": policy["views_unsupported_reported"],
                                                   "grads_not_carried": policy["grads_not_carried"],
                                                   "sha256": sha}))
        self.reports["preflight"] = {
            "decision": policy["decision"], "reason": reason, "blocking": blocking,
            "admissible": pf["admissible"], "admissible_after_exclusion": pf["admissible_after_exclusion"],
            "fully_validated_after_exclusion": pf["fully_validated_after_exclusion"],
            "counts": man["counts"], "gates": man["gates"], "excluded": policy["excluded"],
            "rejected": [n for n in man["names"] if n["verdict"] == "rejected"],
            "unknown": pf["unknown"], "semantics_change": [n for n in man["names"] if n["verdict"] == "semantics_change"],
            "effects": pf["effects"], "effects_available": pf["effects_available"],
            "packages_missing": man["packages_missing"], "package_basis": man["package_basis"],
            "views_predicted_failures": predicted, "views_unsupported": views.get("unsupported") or [],
            "views_injected_bases": views.get("injected_bases") or [],
            "views_hidden_left_in_main": views.get("hidden_left_in_main"),
            "grads_carried": len(grads.get("carried") or []), "grads_not_carried": grads.get("not_carried") or {},
            "imported_distributions": sorted((man.get("imported_distributions") or {}).keys()),
            "destination": {k: desc.get(k) for k in ("python_minor", "dill_version", "has_cuda", "torch")}
                           | {"packages": len(desc.get("packages") or {})},
            "report_path": str(path), "report_sha256": sha, "preflight_ms": pf["preflight_ms"],
        }
        return {**report, "path": str(path), "sha256": sha}

    def _load_preflight(self, row: dict) -> dict[str, Any]:
        """The journaled preflight report, verified against its journaled
        sha256. The capture's exclusion list comes from here, so a resumed
        controller excludes exactly what the preflight decided."""
        p = row.get("preflight_path")
        if not p or not Path(p).is_file():
            raise HandoffError("preflight_report_unavailable", "CAPTURED", str(p))
        raw = Path(p).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row.get("preflight_sha"):
            raise HandoffError("preflight_report_corrupt", "CAPTURED", "sha256 mismatch")
        return json.loads(raw)

    # -- the capsule and the fingerprints ----------------------------------------
    @staticmethod
    def _capsule_manifest(row: dict) -> dict[str, Any]:
        """The capsule's `manifests.json`, read from the capsule on disk after
        checking its journaled sha256 (a resumed controller trusts nothing it
        cannot verify)."""
        blob = Path(row["capsule_path"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != row["capsule_sha"]:
            raise HandoffError("capsule_corrupt_on_controller", "CAPTURED", "sha256 mismatch")
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as t:
            return json.load(t.extractfile("manifests.json"))

    def _boundary(self, mig: str, row: dict, stage: str, fp: dict[str, Any], src_fp: dict[str, Any]) -> dict:
        """Compare a destination fingerprint with the cut's, journal both the
        fingerprint and the verdict, and abort on any undeclared difference."""
        verdict = compare_fingerprints(src_fp, fp)
        boundary = json.loads(self.j.get(mig).get("boundary") or "{}")
        boundary[stage] = verdict
        col = {"restored": "fp_restored", "pre_commit": "fp_commit"}[stage]
        self.j.set_columns(mig, **{col: json.dumps(fp, default=str), "boundary": json.dumps(boundary, default=str)})
        self.j.event(mig, "COMMITTED" if stage == "pre_commit" else "DEST_RESTORED",
                     json.dumps({"commit_boundary": stage, "equal": verdict["equal"],
                                 "mismatched": verdict["mismatched"], "declared": sorted(verdict["declared"])}))
        self.reports.setdefault("commit_boundary", {})[stage] = verdict
        if not verdict["equal"]:
            raise HandoffError("commit_boundary_mismatch:" + ",".join(verdict["mismatched"]),
                               "COMMITTED" if stage == "pre_commit" else "DEST_RESTORED",
                               json.dumps(verdict["details"], default=str)[:300])
        return verdict

    def _boundary_report(self, mig: str) -> dict[str, Any]:
        row = self.j.get(mig) or {}
        load = lambda c: json.loads(row[c]) if row.get(c) else None  # noqa: E731
        return {"source": load("fp_source"), "restored": load("fp_restored"), "pre_commit": load("fp_commit"),
                "verdicts": load("boundary") or {}}

    def _check_validation_side_effects(self, mig: str, v: dict[str, Any]) -> None:
        """Fail closed on what validation did to the destination.

        The validation program fingerprints the destination's whole state and
        workspace immediately before and after the oracles, in one execution
        (see `verify_program`). A forked child contains memory, not the
        filesystem, and an in-process copy does not contain code, so a hook
        that wrote a file or a global would otherwise ride into the commit.
        Any changed field aborts with `validation_side_effect:<fields>`; a
        comparison that could not be made aborts too
        (`validation_side_effect:unverified`), since "not measured" is not
        "nothing changed". Journaled either way. The pre-commit fingerprint
        against the cut remains a second, independent guard."""
        se = v.get("side_effects") or {}
        self.j.event(mig, "DEST_VALIDATED", json.dumps({"validation_side_effects": {
            k: se.get(k) for k in ("checked", "changed", "error", "before_sha256", "after_sha256")}}))
        if not se.get("checked"):
            raise HandoffError("validation_side_effect:unverified", "DEST_VALIDATED",
                               f"the state could not be compared before and after validation: {se.get('error')}")
        if se.get("changed"):
            raise HandoffError("validation_side_effect:" + ",".join(se["changed"]), "DEST_VALIDATED",
                               json.dumps(se.get("details"), default=str)[:300])

    @staticmethod
    def _devices(src_fp: dict[str, Any], dest_desc: dict[str, Any]) -> tuple[str, str]:
        """The devices the oracles should expect: the model's device at the
        cut, and on the destination that device unless it is cuda landing on a
        host without CUDA (the remap puts it on cpu)."""
        devs = [d for k, d in (src_fp.get("devices") or {}).items() if ".param." in k]
        source = "cuda" if any(str(d).startswith("cuda") for d in devs) else "cpu"
        dest = source if source == "cpu" or dest_desc.get("has_cuda") else "cpu"
        return source, dest

    # -- the migration ---------------------------------------------------------
    def migrate(self, mig: str, source_pid: str, dest_profile: str, *, fault: str | None = None,
                takeover: bool = False) -> dict:
        t_start = time.perf_counter()
        row = self.j.create(mig, source_pid, dest_profile)
        if row["phase"] in (ABORTED, "DONE"):
            return {"migration": mig, "phase": row["phase"], "resumed": True, "noop": True}
        claimed, conflict = self.j.start(mig, self.owner, takeover=takeover)
        if conflict is not None:
            # One outgoing migration per runtime (see THE ADMISSION CONTRACT).
            # Nothing was created: the row is closed as ABORTED with
            # authority left on its source, which is where the route finds it.
            reason = conflict["reason"]
            detail = (f"migration {conflict['id']} (phase {conflict['phase']}) "
                      + {"source_busy": "is already moving this runtime",
                         "source_superseded": "already moved this runtime; its authority is elsewhere",
                         "source_not_authoritative": "has this runtime as its uncommitted destination"}[reason])
            self.log(f"  migrate {mig}: refused, {reason}: {detail}")
            self.j.advance(mig, ABORTED, abort_reason=f"REQUESTED:{reason}:{conflict['id']}",
                           authoritative=row["source_pid"])
            return {"migration": mig, "phase": ABORTED, "resumed": False, "reason": reason,
                    "failed_at": "REQUESTED", "detail": detail, "conflict": conflict["id"],
                    "timings": self.timings, "admission": self.admission_evidence(mig),
                    "commit_boundary": self._boundary_report(mig)}
        if not claimed:
            # Duplicate request: another controller holds the lease. Join by
            # waiting for a terminal phase rather than acting. It never
            # touches the gate: only the lease holder opens or closes it.
            self.log(f"  migrate {mig}: lease held by {self.j.get(mig)['lease_owner']}; joining")
            deadline = time.time() + self.join_timeout
            while time.time() < deadline:
                r = self.j.get(mig)
                if r["phase"] in (ABORTED, "DONE"):
                    return {"migration": mig, "phase": r["phase"], "resumed": True, "joined": True, "dest": r.get("dest_pid")}
                time.sleep(min(1.0, self.join_timeout / 4))
            return {"migration": mig, "phase": self.j.get(mig)["phase"], "joined": True, "timeout": True}
        resumed = row["phase"] != "REQUESTED"
        self.log(f"  migrate {mig}: starting at phase {row['phase']}{' (resumed)' if resumed else ''}")

        def phase_at_least(p: str) -> bool:
            return PHASES.index(self.j.get(mig)["phase"]) >= PHASES.index(p)

        try:
            # ---- DEST_PREPARED ---------------------------------------------
            # First, because the preflight needs a destination that can
            # describe itself (see the module doc).
            if not phase_at_least("DEST_PREPARED"):
                t = time.perf_counter()
                # Idempotent: a crash after creating the project but before the
                # journal write leaves an orphan; reclaim() finds it by name.
                pending = row.get("pending_dest_pid")
                if pending:
                    dest = pending   # created before a crash; adopt it
                    self.j.event(mig, "DEST_PREPARED", f"resume: adopting pending destination {dest}")
                else:
                    dest = self.api.create_project(f"clusy-exp-handoff-{mig}", dest_profile)
                    # The create-response-to-journal window: the destination
                    # exists at the provider and nothing durable says so. A
                    # kill here is the one orphan the journal cannot see.
                    self._maybe_crash(DEST_CREATED, after_journal=False)
                    # Journal the id IMMEDIATELY: the only unrecoverable-by-journal
                    # window is now between the create response and this write,
                    # and reclaim() covers it by name.
                    self.j.set_columns(mig, pending_dest_pid=dest)
                self.api.witness(dest, f"print({MARK!r} + '{{\"warm\": true}}')")   # provisions the sandbox
                self.timings["prepare_s"] = time.perf_counter() - t
                self._advance(mig, "DEST_PREPARED", dest_pid=dest)
            row = self.j.get(mig)
            dest = row["dest_pid"]

            # ---- PREFLIGHT -------------------------------------------------
            if not phase_at_least("PREFLIGHT"):
                t = time.perf_counter()
                report = self._preflight(mig, source_pid, dest)
                self.timings["preflight_s"] = time.perf_counter() - t
                if report["policy"]["decision"] != "proceed":
                    raise HandoffError(report["policy"]["reason"], "PREFLIGHT",
                                       "; ".join(report["policy"]["blocking_detail"]))
                self._advance(mig, "PREFLIGHT", preflight_path=report["path"], preflight_sha=report["sha256"])
            row = self.j.get(mig)

            # ---- ADMISSION_CLOSED ------------------------------------------
            # The journal write IS the close (Journal.advance derives the gate
            # from the phase in the same transaction). From here the route
            # refuses; what it admitted before is drained below.
            if not phase_at_least("ADMISSION_CLOSED"):
                self._advance(mig, "ADMISSION_CLOSED")

            # ---- CAPTURED --------------------------------------------------
            if not phase_at_least("CAPTURED"):
                t = time.perf_counter()
                # Drain and barrier before EVERY capture attempt, including a
                # resumed one: a controller that died after the close never
                # drained, and the work it admitted must be in the capsule.
                self._drain(mig, source_pid)
                self.timings["drain_s"] = time.perf_counter() - t
                t = time.perf_counter()
                pf = self._load_preflight(row)
                excluded = pf["policy"]["excluded"]
                classified = [n["name"] for n in pf["manifest"]["names"]]
                cap = self.api.witness(source_pid, capture_program(exclude=excluded, classified=classified))
                evidence = {k: cap.get(k) for k in ("excluded", "excluded_still_present", "source_names_equal",
                                                    "source_bindings_identical", "names_not_in_preflight")}
                evidence["source_names"] = len(cap.get("source_names_before") or [])
                # The source's STATE after the capture, not only its bindings:
                # the fingerprint taken again after the dump must equal the cut
                # (a reducer, or the refresh through the filesystem, could
                # have changed it while every binding kept its id).
                state_after = compare_fingerprints(cap["fingerprint"], cap.get("fingerprint_after") or {})
                evidence["source_state_unchanged"] = state_after["equal"]
                evidence["source_state_mismatched"] = state_after["mismatched"]
                self.reports["capture"] = {**evidence, "source_names_before": cap.get("source_names_before"),
                                           "source_names_after": cap.get("source_names_after"),
                                           "dumped_names": cap.get("dumped_names"), "bytes": cap.get("bytes"),
                                           "rng_carried": cap.get("rng_carried"), "grads_carried": cap.get("grads_carried"),
                                           "expectation_source": cap.get("expectation_source"),
                                           "expectation_method": cap.get("expectation_method")}
                # The name lists themselves are journaled, not only the verdict
                # on them, so a reader can audit exactly which names were
                # compared (and, if they differ, which ones).
                self.j.event(mig, "CAPTURED", json.dumps({"source_untouched": evidence,
                                                          "source_names_before": cap.get("source_names_before"),
                                                          "source_names_after": cap.get("source_names_after")}))
                lost = sorted(set(cap["excluded"]) - set(cap["excluded_still_present"]))
                if not cap["source_names_equal"] or not cap["source_bindings_identical"] or lost \
                        or not state_after["equal"]:
                    # The capture promised to leave the source as it found it.
                    # If it did not, nothing about this switch can be trusted.
                    raise HandoffError("capture_mutated_source", "CAPTURED",
                                       f"names equal={cap['source_names_equal']} bindings identical="
                                       f"{cap['source_bindings_identical']} excluded lost={lost} "
                                       f"state changed={state_after['mismatched']}")
                blob = base64.b64decode(cap["b64"])
                path = self.blobs / f"{mig}.capsule.tgz"
                path.write_bytes(blob)
                self.timings["capture_s"] = time.perf_counter() - t
                self._advance(mig, "CAPTURED", capsule_path=str(path), capsule_sha=cap["sha256"],
                              capsule_bytes=len(blob), fp_source=json.dumps(cap["fingerprint"], default=str))
            row = self.j.get(mig)
            manifest = self._capsule_manifest(row)
            src_fp = json.loads(row["fp_source"]) if row.get("fp_source") else manifest["fingerprint"]
            views = manifest.get("views") or {}

            # ---- DEST_RESTORED ---------------------------------------------
            # On resume, a destination journaled as RESTORED but not VALIDATED
            # is restored again from the capsule before it is validated. With
            # validation non-mutating this is no longer needed for correctness
            # (the pre-commit fingerprint would catch any drift), but restoring
            # is idempotent and it keeps the recovery path the one the first
            # restore took.
            if resumed and self.j.get(mig)["phase"] == "DEST_RESTORED":
                self.j.advance(mig, "CAPTURED")
                self.j.event(mig, "CAPTURED", "resume: re-restoring from the capsule (idempotent) before validating")
            if not phase_at_least("DEST_RESTORED"):
                t = time.perf_counter()
                blob = Path(row["capsule_path"]).read_bytes()
                if fault == "bad_capsule":
                    b = bytearray(blob); b[len(b) // 2] ^= 0xFF; blob = bytes(b)
                upload_to = blob_path(mig, row["capsule_sha"])
                self._upload(dest, blob, path=upload_to)
                res = self.api.witness(dest, restore_program(row["capsule_sha"], fault, blob=upload_to))
                self.timings["restore_s"] = time.perf_counter() - t
                self.reports["restore"] = {k: res.get(k) for k in ("ok", "reason", "views", "grads", "remap", "rng")}
                if not res.get("ok"):
                    raise HandoffError(res.get("reason", "restore_failed"), "DEST_RESTORED", res.get("detail", ""))
                self._boundary(mig, row, "restored", res["fingerprint"], src_fp)
                self._advance(mig, "DEST_RESTORED")

            # ---- DEST_VALIDATED --------------------------------------------
            if not phase_at_least("DEST_VALIDATED"):
                t = time.perf_counter()
                pf = self._load_preflight(self.j.get(mig))
                source_device, dest_device = self._devices(src_fp, pf.get("destination") or {})
                v = self.api.witness(dest, verify_program(source_device, dest_device, views=views))
                self.timings["validate_s"] = time.perf_counter() - t
                # `oracles` counts the checks that RAN ("11/11"); the rows
                # declared not run are listed beside it and in `summary`
                # ("11 checks run, 11 passed, 1 declared not run (forward
                # output)"), never folded into the pass count.
                self.timings["oracles"] = f"{v['passed']}/{v['total']}"
                declared = v.get("declared") or []
                self.reports["validation"] = {"continuation": v.get("continuation"), "passed": v["passed"],
                                              "total": v["total"], "failed": v["failed"],
                                              "declared": declared, "run": v["total"],
                                              "declared_not_run": len(declared),
                                              "summary": v.get("summary") or (
                                                  f"{v['total']} checks run, {v['passed']} passed, "
                                                  f"{len(declared)} declared not run"),
                                              "isolated": v.get("isolated"),
                                              "not_isolated_because": v.get("not_isolated_because"),
                                              "side_effects": v.get("side_effects")}
                if declared:
                    self.timings["oracles_declared"] = [d["name"] for d in declared]
                self._check_validation_side_effects(mig, v)
                if v["passed"] != v["total"]:
                    # Name WHY each check failed, in the journal and the abort
                    # detail: an aborted hop is otherwise unexplainable after
                    # the destination is gone.
                    why = {r["name"]: str(r.get("detail"))[:400] for r in (v.get("rows") or [])
                           if r.get("name") in v["failed"]}
                    self.reports["validation"]["failed_detail"] = why
                    self.j.event(mig, "DEST_VALIDATED", json.dumps({"failed_checks": why}))
                    raise HandoffError("contract_check_failed", "DEST_VALIDATED",
                                       f"failed oracles: {v['failed']} {json.dumps(why)[:900]}")
                self._advance(mig, "DEST_VALIDATED")

            # ---- COMMITTED: re-check the boundary, flip authority ------------
            if not phase_at_least("COMMITTED"):
                t = time.perf_counter()
                # The destination must STILL be the state captured at the cut,
                # after validation: this is what makes "validation does not
                # change the workload" checked rather than assumed.
                fp = self.api.witness(dest, fingerprint_program(views))
                self._boundary(mig, row, "pre_commit", fp, src_fp)
                self.timings["boundary_s"] = time.perf_counter() - t
                t = time.perf_counter()
                # Order matters: the journal flips FIRST (authority AND the
                # reopened gate, in one write) so a crash after this line still
                # routes to the destination; the kernel fence is a second,
                # independent guard applied next and re-applied on resume if
                # the crash landed between them.
                self._advance(mig, "COMMITTED", authoritative=dest)
                self.timings["commit_s"] = time.perf_counter() - t
            if self.j.get(mig)["phase"] == "COMMITTED":
                try:
                    self.api.witness(source_pid, fence_program(row["fence_token"]))
                except HandoffError:
                    pass  # source may already be gone; routing already excludes it

            # ---- SOURCE_RELEASED ------------------------------------------
            if not phase_at_least("SOURCE_RELEASED"):
                t = time.perf_counter()
                st = self.api.delete_project(source_pid)
                if st not in (200, 204, 404):
                    # The deployment has a known deadlock deleting stopped fenced
                    # runtimes; pause is the fallback release. Recorded, not hidden.
                    st2 = self.api.pause(source_pid)
                    self.j.event(mig, "SOURCE_RELEASED", f"delete HTTP {st}; pause HTTP {st2}")
                self.timings["release_s"] = time.perf_counter() - t
                self._advance(mig, "SOURCE_RELEASED")
            self._advance(mig, "DONE")
            self.j.release(mig, self.owner)
            self.timings["total_s"] = time.perf_counter() - t_start
            return {"migration": mig, "phase": "DONE", "resumed": resumed, "dest": dest, "timings": self.timings,
                    "preflight": self.reports.get("preflight"), "capture": self.reports.get("capture"),
                    "restore": self.reports.get("restore"), "validation": self.reports.get("validation"),
                    "admission": self.admission_evidence(mig), "commit_boundary": self._boundary_report(mig)}

        except HandoffError as e:
            row = self.j.get(mig)
            if row["phase"] in ("COMMITTED", "SOURCE_RELEASED", "DONE"):
                # Past the commit point the destination is authoritative and
                # the source may already be released: there is nothing to roll
                # back to, and reopening the source would create a second
                # authority. Record the error; keep authority where it is.
                self.j.event(mig, row["phase"], f"post-commit error, authority unchanged: {e}")
                self.j.release(mig, self.owner)
                return {"migration": mig, "phase": row["phase"], "resumed": resumed, "error": e.reason,
                        "failed_at": e.phase, "detail": e.detail[:300], "timings": self.timings,
                        "admission": self.admission_evidence(mig), "commit_boundary": self._boundary_report(mig)}
            # Abort: destination deleted, source untouched. Authority on the
            # source AND the gate reopened there, in the SAME journal write.
            if row.get("dest_pid"):
                self.api.delete_project(row["dest_pid"])
            self.j.advance(mig, ABORTED, abort_reason=f"{e.phase}:{e.reason}:{e.detail[:200]}",
                           authoritative=row["source_pid"])
            self.j.release(mig, self.owner)
            self.timings["total_s"] = time.perf_counter() - t_start
            return {"migration": mig, "phase": ABORTED, "resumed": resumed, "reason": e.reason,
                    "failed_at": e.phase, "detail": e.detail[:300], "timings": self.timings,
                    "preflight": self.reports.get("preflight"), "capture": self.reports.get("capture"),
                    "restore": self.reports.get("restore"), "validation": self.reports.get("validation"),
                    "admission": self.admission_evidence(mig), "commit_boundary": self._boundary_report(mig)}

    def reclaim(self, mig: str) -> dict:
        """Find and delete resources a crashed run may have left: a destination
        project created but not journaled, or a destination journaled but
        belonging to an aborted migration."""
        row = self.j.get(mig)
        reclaimed = []
        keep_ids = {row.get("dest_pid")} if row and row["phase"] in ("COMMITTED", "SOURCE_RELEASED", "DONE") else set()
        items = []
        for attempt in range(4):
            items = self.api.list_projects()
            if any(str(it.get("name", "")) == f"clusy-exp-handoff-{mig}" for it in items) or attempt == 3:
                break
            time.sleep(2.0)
        for it in items:
            if str(it.get("name", "")) == f"clusy-exp-handoff-{mig}":
                if it["id"] not in keep_ids:
                    self.api.delete_project(it["id"]); reclaimed.append(it["id"])
        return {"reclaimed": reclaimed}
