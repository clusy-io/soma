"""The experiment record: one row per transition, and the runner that emits it.

Every experiment in the program writes this shape, so results from different
experiments concatenate without a merge step and a later analysis never has to
know which script produced a row.

Two conventions carry most of the weight.

**A timing that was not measured is ``None``, never ``0``.** A zero is a
measurement; a null is an admission. Several of the requested phase timings are
only visible inside the server and cannot be observed by an external harness, so
they will be null for externally-driven transitions and populated for in-process
ones. Confusing "we did not look" with "it took no time" would silently
manufacture a fast result, which is the exact failure this program exists to
avoid.

**Teardown is unconditional.** ``ExperimentRun`` is a context manager that emits
its record and runs every registered cleanup in a ``finally``, so a crashed or
interrupted experiment still releases its resources and still leaves a row
saying what happened. An experiment that dies without a record is
indistinguishable from one that was never run.
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

RECORD_VERSION = 1

#: Terminal outcomes. `error` is reserved for the harness failing, as distinct
#: from the system under test failing, which is `failure`.
RESULTS = ("success", "failure", "skipped", "error")

#: Failure taxonomy. Deliberately closed: an unclassified failure is a gap in
#: this list, and forcing it to `unclassified` makes that gap visible in the
#: results rather than hiding it behind free text.
FAILURE_CLASSES = (
    "none",
    "capture_failed",
    "checksum_mismatch",
    "provider_preflight_failed",
    "version_gate_failed",
    "package_missing",
    "device_incompatible",
    "restore_killed_kernel",
    "target_start_timeout",
    "state_diverged",
    "harness_error",
    "unclassified",
)


# -- server identity -----------------------------------------------------------
#
# `src/runmeta.py` owns the question "which build is answering at this URL":
# GET /version first (the server's own statement), else an inference from the
# process listening on the port. It lives beside this package rather than in
# it, and this package can also be shipped alone into a sandbox for a probe,
# so the import must neither assume `src/` is on sys.path nor fail when
# runmeta is absent. Absent means the identity is recorded as unknown, never
# guessed. The server identity never comes from a checkout on disk: a
# directory says nothing about which process served a request.

_RUNMETA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runmeta.py")


def _load_runmeta() -> Any:
    mod = sys.modules.get("runmeta")
    if mod is not None:
        return mod
    if os.path.isfile(_RUNMETA_PATH):
        spec = importlib.util.spec_from_file_location("runmeta", _RUNMETA_PATH)
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["runmeta"] = mod
            try:
                spec.loader.exec_module(mod)
                return mod
            except Exception:
                sys.modules.pop("runmeta", None)
                return None
    try:
        import runmeta  # type: ignore[import-not-found]

        return runmeta
    except ImportError:
        return None


def _git_out(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _newest_mtime(root: Path, suffix: str) -> float | None:
    newest = None
    if root.is_dir():
        for p in root.rglob("*"):
            if p.suffix == suffix and p.is_file():
                m = p.stat().st_mtime
                newest = m if newest is None or m > newest else newest
    return newest


def inferred_identity_checks(identity: dict[str, Any]) -> dict[str, Any] | None:
    """Can an INFERRED identity be trusted to describe the running process?

    Inference reads the git state of the listening process's working directory
    NOW, which is a statement about that directory, and the directory can move
    on after the process started while the old code keeps serving. It can
    also have moved on before the process started, if nobody rebuilt: the
    process then runs an older compiled tree than the checkout it sits in.
    The ways that happens, each checkable from timestamps:

      * HEAD was committed after the process started: the process cannot have
        been built from it (blocking);
      * a source file was modified after the process started: the dirty flag,
        diff hash and source hash describe a later tree (blocking);
      * a source file is newer than every compiled file: the source moved on
        after the last build, so the compiled tree the process runs is not the
        checkout's tree, whatever HEAD says (blocking). This is the case of a
        commit made after a build and a restart without a rebuild, where HEAD
        and the start time alone would read as a clean match;
      * no compiled tree, or no source tree, under the working directory: the
        process was not started from a built checkout there (for example
        `node /elsewhere/dist/index.js` run from another directory), so the
        directory's commit says nothing about it (blocking);
      * the compiled tree is older than HEAD's commit although no source file
        is newer than it: the build came from a working tree that was then
        committed unchanged. Only recompiling can prove the two identical;
        inference cannot, so it is recorded, not blocking.

    `dist_modified_after_process_start`, which runmeta already computes, is
    blocking as well. Limits: a source file DELETED after the build leaves no
    newer mtime, and inputs outside `src/` (tsconfig, node_modules) are not
    examined. Returns None for an identity that was not inferred.
    """
    if identity.get("source") != "inferred_from_listening_process" or not identity.get("cwd"):
        return None
    # runmeta records the directory relative to the repository or as `~/...`.
    cwd = Path(os.path.expanduser(identity["cwd"]))
    if not cwd.is_absolute():
        cwd = Path(__file__).resolve().parents[2] / cwd
    started = None
    raw = identity.get("process_started")
    if raw:
        try:
            # `ps -o lstart=` prints local time; a naive datetime's timestamp()
            # is interpreted in local time too, which is what makes these
            # comparable with file mtimes and commit epochs.
            started = _dt.datetime.strptime(raw, "%a %b %d %H:%M:%S %Y").timestamp()
        except ValueError:
            started = None
    head_ct_raw = _git_out(cwd, "log", "-1", "--format=%ct", "HEAD")
    head_ct = float(head_ct_raw) if head_ct_raw and head_ct_raw.isdigit() else None
    src_newest = _newest_mtime(cwd / "src", ".ts")
    dist_newest = _newest_mtime(cwd / "dist", ".js")

    def after(a: float | None, b: float | None) -> bool | None:
        return None if a is None or b is None else a > b

    checks: dict[str, Any] = {
        "process_started_epoch": started,
        "head_commit_epoch": head_ct,
        "src_newest_mtime": src_newest,
        "dist_newest_mtime": dist_newest,
        "head_committed_after_process_start": after(head_ct, started),
        "src_modified_after_process_start": after(src_newest, started),
        "dist_modified_after_process_start": identity.get("dist_modified_after_process_start"),
        "src_newer_than_dist": after(src_newest, dist_newest),
        # Informational: see the docstring. Blocking only through
        # src_newer_than_dist, which is what makes it unsafe.
        "dist_predates_head_commit": after(head_ct, dist_newest),
        "no_compiled_tree": identity.get("dist_sha256") is None or dist_newest is None,
        "no_source_tree": not identity.get("src_files") or src_newest is None,
    }
    blocking = [k for k in ("head_committed_after_process_start", "src_modified_after_process_start",
                            "dist_modified_after_process_start", "src_newer_than_dist",
                            "no_compiled_tree", "no_source_tree") if checks[k] is True]
    checks["invalidated_by"] = blocking
    checks["valid"] = started is not None and not blocking
    if started is None:
        checks["invalidated_by"] = blocking + ["process_start_unknown"]
    return checks


def server_identity(api_url: str | None) -> dict[str, Any]:
    """`runmeta.server_identity(api_url)`, plus the inference checks when the
    identity was inferred. Never raises: a provenance failure must not cost
    the record it was meant to label."""
    rm = _load_runmeta()
    if rm is None:
        return {"source": "unknown", "reason": "runmeta unavailable (clusy_boundary shipped without src/runmeta.py)",
                "url": api_url}
    try:
        ident = dict(rm.server_identity(api_url))
    except Exception as exc:  # noqa: BLE001
        return {"source": "unknown", "reason": f"server_identity raised {type(exc).__name__}: {exc}"[:500],
                "url": api_url}
    try:
        checks = inferred_identity_checks(ident)
    except Exception as exc:  # noqa: BLE001
        checks = {"valid": False, "invalidated_by": [f"checks raised {type(exc).__name__}: {exc}"[:300]]}
    if checks is not None:
        ident["inference_checks"] = checks
    return ident


def server_commit_label(identity: dict[str, Any] | None) -> tuple[str, str | None]:
    """The short label a record carries as `git_commit`, and why it is
    'unknown' when it is. Short sha plus '-dirty' when the build was dirty.

    'unknown' whenever the evidence does not tie a commit to the running code:
    no identity, no commit, a reported build-info that does not match (or
    could not be checked against) the tree the process started from, a
    reported compiled tree rewritten after the process started, or an
    inference the checks above invalidate.
    """
    if not identity or identity.get("source") in (None, "unknown"):
        return "unknown", (identity or {}).get("reason") or "server identity unknown"
    sha = identity.get("git_commit")
    if not sha:
        return "unknown", f"{identity.get('source')} identity names no commit"
    if identity.get("source") == "server_reported":
        matches = identity.get("build_info_matches_dist")
        if matches is False:
            return "unknown", "reported build-info does not describe the dist the process started from"
        if matches is not True:
            # Null: the process is not running from dist/, or one of the two
            # hashes is missing. The commit is then an unchecked claim.
            return "unknown", "reported build-info could not be checked against the running dist"
        if identity.get("dist_modified_after_start") is True:
            # Modules the server imports lazily are read from disk at first
            # use, so a rebuild under a running process can change the code
            # that serves a request while the startup hash stays the same.
            return "unknown", "the server's dist/ was modified after the process started"
    checks = identity.get("inference_checks")
    if identity.get("source") == "inferred_from_listening_process":
        if not checks or not checks.get("valid"):
            return "unknown", "inferred identity invalidated: " + ", ".join((checks or {}).get("invalidated_by") or ["no checks"])
    return str(sha)[:8] + ("-dirty" if identity.get("git_dirty") else ""), None


#: What identifies the running service across two reads. The first six are the
#: keys `runmeta.finish_run` compares; the last two are the server's and the
#: inference's own "compiled tree rewritten under the process" flags, which
#: can flip without any of the others changing.
_IDENTITY_KEYS = ("git_commit", "git_dirty", "dist_sha256", "started_at", "process_started", "pid",
                  "dist_modified_after_start", "dist_modified_after_process_start")


def _identity_changes(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Which identifying keys differ between two identity reads (empty: the
    same service). When one read was reported and the other inferred (a
    /version timeout under load falls back to inference), only the pid is
    comparable across the two shapes."""
    if a.get("source") != b.get("source"):
        if a.get("pid") is not None and b.get("pid") is not None and a.get("pid") != b.get("pid"):
            return ["pid"]
        return []
    return [k for k in _IDENTITY_KEYS if a.get(k) != b.get(k)]


# -- expected build ------------------------------------------------------------
#
# A harness that knows which build it means to measure (an
# `--expect-server-commit` option) verifies the live identity against it
# before it starts.
# The same test is applied to every record at emit, from that record's OWN
# label, so a row whose server changed underneath it cannot carry a passing
# verdict borrowed from the start of the run.


def parse_server_expectation(expect: str) -> tuple[str, bool]:
    """`<hex prefix>` expects a clean build at that commit, `<hex prefix>-dirty`
    a dirty one. At least 7 hex digits, so a short typo cannot match by luck."""
    want_dirty = expect.endswith("-dirty")
    prefix = (expect[: -len("-dirty")] if want_dirty else expect).lower()
    if not re.fullmatch(r"[0-9a-f]{7,40}", prefix):
        raise ValueError(f"expected server commit {expect!r}: want 7-40 hex digits, optionally followed by -dirty")
    return prefix, want_dirty


def parse_server_diff_expectation(expect_diff: str) -> str:
    """A prefix of the dirty build's `git_diff_sha256` (at least 12 hex digits)."""
    d = expect_diff.lower()
    if not re.fullmatch(r"[0-9a-f]{12,64}", d):
        raise ValueError(f"expected server diff {expect_diff!r}: want 12-64 hex digits of git_diff_sha256")
    return d


def verify_server_expectation(identity: dict[str, Any], expect: str, *, expect_diff: str | None = None,
                              label: str | None = None, note: str | None = None) -> dict[str, Any]:
    """Does this identity establish the expected build? Reported and inferred
    identities are both accepted, but only when `server_commit_label` can name
    a commit from them: an unknown identity, a reported build-info that does
    not describe the running dist, or an inference the timestamp checks
    invalidate all fail here, whatever commit they mention.

    `label`/`note` override the label derived from `identity`: a record passes
    its own final label, which is 'unknown' when the server changed during its
    transition even though the identity read at its start was fine.

    `-dirty` alone accepts any uncommitted tree on top of the commit; with
    `expect_diff` the build's diff hash must match too.
    """
    prefix, want_dirty = parse_server_expectation(expect)
    diff_prefix = parse_server_diff_expectation(expect_diff) if expect_diff else None
    if label is None:
        label, note = server_commit_label(identity)
    v: dict[str, Any] = {
        "expected": expect,
        "checked_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "identity_source": identity.get("source"),
        "observed_commit": identity.get("git_commit"),
        "observed_dirty": identity.get("git_dirty"),
        "observed_label": label,
        "pid": identity.get("pid"),
        "dist_sha256": identity.get("dist_sha256"),
        "ok": False,
    }
    if expect_diff:
        v["expected_diff"] = expect_diff
        v["observed_diff_sha256"] = identity.get("git_diff_sha256")
    if label == "unknown":
        v["reason"] = f"the server identity does not establish a build: {note}"
    elif not str(identity.get("git_commit")).lower().startswith(prefix):
        v["reason"] = f"server is {label}, expected {expect}"
    elif bool(identity.get("git_dirty")) != want_dirty:
        v["reason"] = f"server is {label}, expected a {'dirty' if want_dirty else 'clean'} build of {prefix}"
    elif diff_prefix and not str(identity.get("git_diff_sha256") or "").lower().startswith(diff_prefix):
        v["reason"] = (f"server is {label} with diff {str(identity.get('git_diff_sha256'))[:16]}, "
                       f"expected diff {diff_prefix[:16]}")
    else:
        v["ok"] = True
        v["reason"] = "match"
        if want_dirty and not diff_prefix:
            v["diff_checked"] = False
    return v


def git_commit(repo: str | None = None, *, _explicit: bool = False) -> str:
    """The commit a repository is at, or a marker saying we could not tell.

    When ``_explicit`` is set the caller is asking about ONE named repository
    and a missing checkout must not be answered with a different repository's
    commit. That silent substitution is precisely the defect this function was
    changed to close: a lifted-ceiling row stamped with the harness commit is
    indistinguishable from a stock row, and the whole point of the stamp is to
    tell those two apart.
    """
    if _explicit and repo is None:
        return "unknown-repo-not-found"
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo or os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            sha = out.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo or os.path.dirname(os.path.abspath(__file__)),
                capture_output=True, text=True, timeout=10,
            )
            return sha + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        pass
    return "unknown"


@dataclass
class Endpoint:
    """One side of a transition."""

    provider: str | None = None       # e2b | modal | local | docker
    profile: str | None = None        # cpu | gpu_t4 | ... | image tag
    python: str | None = None
    torch: str | None = None
    cuda: str | None = None           # device name, or None when absent
    arch: str | None = None
    libc: str | None = None
    hostname: str | None = None

    @classmethod
    def local(cls, provider: str = "local", profile: str | None = None) -> "Endpoint":
        torch_v, cuda = None, None
        try:
            import torch

            torch_v = torch.__version__
            if torch.cuda.is_available():
                cuda = torch.cuda.get_device_name(0)
        except Exception:
            pass
        return cls(
            provider=provider,
            profile=profile or ("gpu" if cuda else "cpu"),
            python=sys.version.split()[0],
            torch=torch_v,
            cuda=cuda,
            arch=platform.machine(),
            libc=(platform.libc_ver()[0] or None),
            hostname=socket.gethostname(),
        )


#: Phases an external harness provably cannot time against the deployed API.
#: Each has no span, no project-scoped metric, and no duration in any log the
#: harness can read; several are visible only as an aggregate OTel histogram or
#: as a delta between two ISO-stamped server stdout lines.
#:
#: They are kept as fields rather than dropped, because a schema that omits them
#: silently loses the fact that Equation 1 in the paper cannot currently be
#: measured end to end. A reader of the records should be able to see the gap.
#: What must never happen is treating the null as a zero and summing the column.
EXTERNALLY_UNOBSERVABLE = frozenset({
    "capture_ms",            # visible only as an aggregate server-side histogram
    "workspace_flush_ms",    # the workspace flush has no histogram, span, or log
    "upload_ms",             # the offload runs inside the capture step, untimed
    "source_teardown_ms",    # runtime stop is not timed on the switch path
    "workspace_restore_ms",  # derivable only from server stdout timestamps
    "package_restore_ms",    # derivable only from server stdout timestamps
})


@dataclass
class Timings:
    """Per-phase durations in milliseconds. ``None`` means not measured.

    The distinction between null and zero is load-bearing here. Six of these
    phases cannot be observed by an external harness at all (see
    ``EXTERNALLY_UNOBSERVABLE``), so they will be null on every externally
    driven transition. Filling them with zero, or omitting them, would turn a
    known measurement gap into an apparently instantaneous phase.
    """

    capture_ms: float | None = None
    workspace_flush_ms: float | None = None
    upload_ms: float | None = None
    source_teardown_ms: float | None = None
    target_start_ms: float | None = None
    workspace_restore_ms: float | None = None
    package_restore_ms: float | None = None
    namespace_restore_ms: float | None = None
    validation_ms: float | None = None
    first_execute_ms: float | None = None
    end_to_end_ms: float | None = None

    def measured(self) -> dict[str, float]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def unmeasured(self) -> list[str]:
        return sorted(k for k, v in asdict(self).items() if v is None)


@dataclass
class Sizes:
    namespace_bytes: int | None = None
    workspace_bytes: int | None = None
    workspace_file_count: int | None = None
    package_count: int | None = None


@dataclass
class ExperimentRecord:
    version: int
    experiment_id: str
    experiment: str
    git_commit: str
    timestamp: str

    source: Endpoint
    destination: Endpoint
    sizes: Sizes
    timings: Timings

    result: str
    failure_class: str

    seed: int | None = None
    workload: str | None = None
    transition_index: int | None = None

    #: Provenance of the SYSTEM UNDER TEST and of the instrument measuring it.
    #: `git_commit` above is the server's, because that is what a reader means
    #: by "which build produced this", and it is derived from
    #: `build["server_identity"]`: what the running service said about itself
    #: (GET /version) or, failing that, what could be inferred from the process
    #: on the port. The harness commit and any deliberate build overrides live
    #: here too, so a row measured on a raised-limit build can never be
    #: mistaken for a stock-configuration row.
    build: dict[str, Any] = field(default_factory=dict)

    #: Per-property boundary verdicts, when a witness comparison ran.
    boundary: dict[str, Any] | None = None
    #: Free-form, for anything a specific experiment needs to carry.
    extra: dict[str, Any] = field(default_factory=dict)
    #: Populated when result is 'error'.
    error: str | None = None
    #: The invocation this record belongs to (`runmeta.stamp`), whose full
    #: metadata is one line of the experiment's runs_meta.jsonl, and the cohort
    #: it is analysed in. Both None when the caller supplied no run metadata.
    run: dict[str, Any] | None = None
    cohort: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Make the honesty convention machine-visible, not just documented.
        d["timings_unmeasured"] = self.timings.unmeasured()
        return d

    def append_to(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(self.to_dict(), sort_keys=True, default=str) + "\n")


class Phase:
    """Time one phase and write it onto a ``Timings``.

    Used as ``with run.phase('capture_ms'): ...``. A phase that raises still
    records its elapsed time, because how long a failure took is data.
    """

    def __init__(self, timings: Timings, name: str) -> None:
        self._timings = timings
        self._name = name
        self._start = 0.0

    def __enter__(self) -> "Phase":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        elapsed = (time.perf_counter() - self._start) * 1000.0
        setattr(self._timings, self._name, elapsed)
        return None


class ExperimentRun:
    """A single transition, guaranteed to emit a record and clean up.

    The guarantee is the point. Experiments in this program provision GPUs and
    sandboxes, so a run that dies between provisioning and teardown costs real
    money and leaves a resource alive. Cleanups registered here run in a
    ``finally``, in reverse order, and each is isolated so one failing cleanup
    cannot prevent the others.
    """

    def __init__(
        self,
        experiment: str,
        *,
        record_path: str,
        source: Endpoint | None = None,
        destination: Endpoint | None = None,
        seed: int | None = None,
        workload: str | None = None,
        transition_index: int | None = None,
        repo: str | None = None,
        now: str | None = None,
        build_overrides: dict[str, Any] | None = None,
        api_url: str | None = None,
        run_meta: dict[str, Any] | None = None,
        cohort: str | None = None,
    ) -> None:
        self.experiment = experiment
        self.record_path = record_path
        self.experiment_id = uuid.uuid4().hex[:16]
        self.source = source or Endpoint()
        self.destination = destination or Endpoint()
        self.sizes = Sizes()
        self.timings = Timings()
        self.result = "error"
        self.failure_class = "harness_error"
        self.boundary: dict[str, Any] | None = None
        self.extra: dict[str, Any] = {}
        self.error: str | None = None
        self.seed = seed
        self.workload = workload
        self.transition_index = transition_index
        self._repo = repo
        self._now = now
        # Deliberate deviations from the shipped configuration, e.g. a raised
        # checkpoint cap or transfer ceiling for the scale probe. Recorded so a
        # reader can separate those rows without consulting anyone's memory.
        self._build_overrides = dict(build_overrides or {})
        # The API under test. Its identity is read when the context opens
        # (before the transition's first request) and again at emit, so a
        # restart or rebuild in the middle of one transition is visible on the
        # row. Without a URL the identity is 'unknown', never a guess.
        self._api_url = api_url
        self._server_identity: dict[str, Any] | None = None
        self._run_meta = run_meta
        self.cohort = cohort if cohort is not None else (run_meta or {}).get("cohort")
        self._cleanups: list[tuple[str, Callable[[], Any]]] = []
        self._t0 = 0.0

    # -- resource lifecycle -------------------------------------------------

    def on_cleanup(self, label: str, fn: Callable[[], Any]) -> None:
        """Register a teardown. Runs in reverse order, in a finally, isolated."""
        self._cleanups.append((label, fn))

    def _run_cleanups(self) -> None:
        for label, fn in reversed(self._cleanups):
            try:
                fn()
            except Exception as exc:
                # A cleanup that fails is itself a finding: it usually means a
                # resource is still alive and still costing money.
                self.extra.setdefault("cleanup_failures", []).append(
                    f"{label}: {type(exc).__name__}: {exc}"
                )
        self._cleanups.clear()

    # -- phases -------------------------------------------------------------

    def phase(self, name: str) -> Phase:
        if not hasattr(self.timings, name):
            raise KeyError(f"{name!r} is not a phase in the record schema")
        return Phase(self.timings, name)

    # -- outcome ------------------------------------------------------------

    def succeed(self) -> None:
        self.result = "success"
        self.failure_class = "none"

    def fail(self, failure_class: str, detail: str | None = None) -> None:
        if failure_class not in FAILURE_CLASSES:
            raise ValueError(
                f"{failure_class!r} is not in the closed failure taxonomy; "
                f"add it to FAILURE_CLASSES rather than passing free text"
            )
        self.result = "failure"
        self.failure_class = failure_class
        if detail:
            self.extra["failure_detail"] = detail

    def skip(self, reason: str) -> None:
        self.result = "skipped"
        self.failure_class = "none"
        self.extra["skip_reason"] = reason

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "ExperimentRun":
        self._server_identity = server_identity(self._api_url)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if exc is not None:
                self.result = "error"
                self.failure_class = "harness_error"
                self.error = "".join(
                    traceback.format_exception(exc_type, exc, tb)
                )[-2000:]
            if self.timings.end_to_end_ms is None:
                self.timings.end_to_end_ms = (time.perf_counter() - self._t0) * 1000.0
        finally:
            self._run_cleanups()
            self._emit()
        return False  # never swallow: a harness bug must be visible

    def _build(self) -> tuple[str, dict[str, Any]]:
        start = self._server_identity or server_identity(self._api_url)
        end = server_identity(self._api_url) if self._api_url else start
        # An unreachable server at emit (it crashed, or the port is closed) is
        # recorded but does not relabel the row: the build that served the
        # transition is still the one identified when it began.
        end_known = end.get("source") not in (None, "unknown")
        changes = _identity_changes(start, end) if end_known else []
        changed = bool(changes)
        label, note = server_commit_label(start)
        if changed:
            # The label names the build that was up when the transition began;
            # a row whose server changed underneath it is not attributable to
            # one build and says so.
            label, note = "unknown", "server identity changed during the transition: " + ", ".join(changes)
        build: dict[str, Any] = {
            "server_commit": label,
            "server_identity": start,
            "server_identity_source": start.get("source"),
            "server_changed_during_run": changed,
            "harness_commit": git_commit(self._repo),
            "overrides": self._build_overrides,
        }
        if note:
            build["server_commit_note"] = note
        if changed or (self._api_url and not end_known):
            build["server_identity_at_emit"] = end
        run_start = (self._run_meta or {}).get("server_verification")
        if run_start is not None:
            # The harness's verdict on the identity read BEFORE its first
            # trial. It says nothing about this row, so it is kept under a name
            # that says when it was taken.
            build["run_start_server_verification"] = run_start
            if run_start.get("expected"):
                # This row's own verdict: the same test, applied to the label
                # this row carries (so 'unknown' after a mid-transition change
                # fails here even though the run-start verdict passed).
                try:
                    build["server_verification"] = verify_server_expectation(
                        start, run_start["expected"], expect_diff=run_start.get("expected_diff"),
                        label=label, note=note)
                except Exception as exc:  # noqa: BLE001 (a bad expectation must not cost the record)
                    build["server_verification"] = {
                        "expected": run_start.get("expected"), "ok": False,
                        "reason": f"verification raised {type(exc).__name__}: {exc}"[:300]}
        return label, build

    def _run_stamp(self) -> dict[str, Any] | None:
        if self._run_meta is None:
            return None
        rm = _load_runmeta()
        if rm is not None:
            stamp = rm.stamp({}, self._run_meta)["run"]
        else:
            stamp = {"run_id": self._run_meta.get("run_id"), "cohort": self._run_meta.get("cohort")}
        # The record's own cohort wins, so a caller overriding it per record
        # does not leave the stamp and the field disagreeing.
        stamp["cohort"] = self.cohort
        # `server_commit` in the stamp is runmeta's raw identity field, which
        # for an inferred identity is the checkout's state now. The harness
        # that started the run may have judged it (server_commit_label); carry
        # that verdict so the stamp cannot read as more certain than the row.
        if "server_commit_label" in self._run_meta:
            stamp["server_commit_label"] = self._run_meta["server_commit_label"]
        return stamp

    def _emit(self) -> None:
        label, build = self._build()
        rec = ExperimentRecord(
            version=RECORD_VERSION,
            experiment_id=self.experiment_id,
            experiment=self.experiment,
            git_commit=label,
            timestamp=self._now or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            source=self.source,
            destination=self.destination,
            sizes=self.sizes,
            timings=self.timings,
            result=self.result,
            failure_class=self.failure_class,
            seed=self.seed,
            workload=self.workload,
            transition_index=self.transition_index,
            boundary=self.boundary,
            extra=self.extra,
            error=self.error,
            build=build,
            run=self._run_stamp(),
            cohort=self.cohort,
        )
        rec.append_to(self.record_path)
        self.record = rec


def load_records(*paths: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out
