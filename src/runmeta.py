"""Run metadata: which build of what produced a record.

Server identity comes from the running service (GET /version), never from a
checkout that merely sits next to the harness. A label read from a checkout is
a claim about a directory; what a reader needs is a claim about the process
that served the requests.

So the server identity here comes from the running service first:

  1. `GET /version` on the API under test. The research build reports the git
     commit it was built from, whether that tree was dirty, a hash of the
     compiled `dist/` tree taken when the process started, and the process
     start time. This is the server's own statement about itself.
  2. If the route is absent (an older build), the identity is INFERRED from the
     process listening on the API port: its working directory, that
     directory's git HEAD and dirty state, a hash of its `dist/` tree, and
     whether `dist/` was modified after the process started (in which case the
     hash describes files the process may not have loaded). The record says
     which of the two it is, so an inferred identity is never read as a
     reported one.

The harness identity is the research repo's commit, a dirty flag, a hash of
the uncommitted diff, and a SHA-256 of every harness file the run executed,
so a record from a dirty tree can still be matched to the exact code. Local
paths are recorded relative to the repository (or to the home directory as
`~/...`), and the host name only as a short hash, so a record carries no user
name or machine name.

Call `start_run(...)` before the first live request and `finish_run(meta)`
after the last; `finish_run` re-reads the server identity and flags a restart
or rebuild that happened mid-run.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

SCHEMA = "runmeta/1"
ROOT = Path(__file__).resolve().parents[1]

#: Harness modules every live experiment executes, directly or by shipping
#: their source into a sandbox kernel. Hashed into every record.
_CORE_FILES = (
    "src/runmeta.py",
    "src/handoff/controller.py",
    "src/capsule/manifest.py",
    "src/capsule/storage_sharing.py",
    "src/capsule/optimizer_reattach.py",
    "experiments/fixture.py",
    "experiments/oracles.py",
)


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _display_path(path: str | Path) -> str:
    """A path as a record may show it: relative to the repository root when it
    is inside it, `~/...` when it is under the home directory, else as is.
    Symlinks are tried unresolved first, so a virtualenv interpreter is shown
    as itself rather than as the base interpreter it links to."""
    raw = Path(os.path.abspath(path))
    try:
        resolved = raw.resolve()
    except OSError:
        resolved = raw
    for cand in (raw, resolved):
        try:
            return cand.relative_to(ROOT).as_posix()
        except ValueError:
            pass
    try:
        homes = (Path.home(), Path.home().resolve())
    except (OSError, RuntimeError):
        homes = ()
    for cand in (raw, resolved):
        for home in homes:
            try:
                return "~/" + cand.relative_to(home).as_posix()
            except ValueError:
                pass
    return str(raw)


def _host_id() -> str:
    """A short, stable hash of the host name: enough to tell two machines
    apart across records without naming either."""
    return hashlib.sha256(socket.gethostname().encode()).hexdigest()[:12]


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def _repo_identity(repo: Path) -> dict[str, Any]:
    head = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain", "--untracked-files=no")
    diff = _git(repo, "diff", "HEAD")
    return {
        "path": _display_path(repo),
        "commit": head.strip() if head else None,
        "branch": (_git(repo, "rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
        "dirty": bool(status and status.strip()),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
    }


def _tree_hash(root: Path, suffixes: tuple[str, ...]) -> tuple[str | None, int, float | None]:
    """SHA-256 over (relative path, content) of every file with a suffix, in
    sorted order; also the file count and the newest mtime."""
    if not root.is_dir():
        return None, 0, None
    h = hashlib.sha256()
    n, newest = 0, None
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in suffixes:
            rel = p.relative_to(root).as_posix().encode()
            h.update(len(rel).to_bytes(4, "big") + rel)
            data = p.read_bytes()
            h.update(len(data).to_bytes(8, "big") + data)
            n += 1
            m = p.stat().st_mtime
            newest = m if newest is None or m > newest else newest
    return (h.hexdigest() if n else None), n, newest


def _listening_pid(port: int) -> int | None:
    try:
        out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    pids = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    return pids[0] if pids else None


def _process_cwd(pid: int) -> Path | None:
    try:
        out = subprocess.run(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in out.stdout.splitlines():
        if line.startswith("n/"):
            return Path(line[1:])
    return None


def _process_started(pid: int) -> tuple[str | None, float | None]:
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True, timeout=10)
        raw = out.stdout.strip()
        if not raw:
            return None, None
        t = _dt.datetime.strptime(raw, "%a %b %d %H:%M:%S %Y")
        return raw, t.timestamp()
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None, None


def server_identity(api_url: str | None, timeout: float = 10.0) -> dict[str, Any]:
    """The identity of the service answering at `api_url`. See module doc."""
    if not api_url:
        return {"source": "unknown", "reason": "no api url"}
    base = api_url.rstrip("/")
    try:
        with urllib.request.urlopen(base + "/version", timeout=timeout) as r:
            payload = json.loads(r.read().decode("utf-8", "replace"))
        if isinstance(payload, dict) and payload.get("git_commit"):
            return {"source": "server_reported", "url": base, **payload}
    except (urllib.error.URLError, OSError, ValueError):
        pass

    port = urllib.parse.urlparse(base).port
    ident: dict[str, Any] = {"source": "inferred_from_listening_process", "url": base,
                             "note": "the API did not answer GET /version; identity is inferred from the process on the port"}
    pid = _listening_pid(port) if port else None
    ident["pid"] = pid
    if pid is None:
        ident["source"] = "unknown"
        return ident
    started_raw, started_ts = _process_started(pid)
    ident["process_started"] = started_raw
    cwd = _process_cwd(pid)
    ident["cwd"] = _display_path(cwd) if cwd else None
    if cwd is None:
        return ident
    repo = _repo_identity(cwd)
    ident.update({"git_commit": repo["commit"], "git_branch": repo["branch"], "git_dirty": repo["dirty"],
                  "git_diff_sha256": repo["diff_sha256"]})
    dist_hash, n, newest = _tree_hash(cwd / "dist", (".js",))
    ident.update({"dist_sha256": dist_hash, "dist_files": n})
    if newest is not None and started_ts is not None:
        ident["dist_modified_after_process_start"] = newest > started_ts
    src_hash, n_src, _ = _tree_hash(cwd / "src", (".ts",))
    ident.update({"src_sha256": src_hash, "src_files": n_src})
    return ident


def harness_identity(extra_files: list[str | Path] | None = None) -> dict[str, Any]:
    ident = _repo_identity(ROOT)
    files: dict[str, str | None] = {}
    main = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
    wanted = [ROOT / f for f in _CORE_FILES] + [Path(p) for p in (extra_files or [])]
    if main is not None and main.is_file():
        wanted.append(main)
    for p in wanted:
        p = p if p.is_absolute() else (ROOT / p)
        files[_display_path(p)] = _sha256_file(p)
    ident["files_sha256"] = files
    return ident


def host_identity() -> dict[str, Any]:
    out: dict[str, Any] = {
        "host_id": _host_id(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": _display_path(sys.executable),
    }
    for mod in ("dill", "numpy", "torch"):
        m = sys.modules.get(mod)
        if m is not None:
            out[mod] = getattr(m, "__version__", None)
    return out


def start_run(experiment: str, *, api_url: str | None = None, args: dict[str, Any] | None = None,
              cohort: str | None = None, extra_files: list[str | Path] | None = None) -> dict[str, Any]:
    """Metadata for one invocation of an experiment. Every record the run
    writes should carry `run_id` and `cohort`, and the full dict should be
    written once to the run's metadata file."""
    started = _utc()
    return {
        "schema": SCHEMA,
        "experiment": experiment,
        "run_id": f"{experiment}-{started.replace(':', '').replace('-', '')[:15]}-{uuid.uuid4().hex[:6]}",
        "cohort": cohort,
        "started_at_utc": started,
        "argv": list(sys.argv),
        "args": args,
        "harness": harness_identity(extra_files),
        "server": server_identity(api_url),
        "host": host_identity(),
    }


def finish_run(meta: dict[str, Any], *, api_url: str | None = None) -> dict[str, Any]:
    """Stamp the end of the run and detect a server change during it."""
    meta["finished_at_utc"] = _utc()
    end = server_identity(api_url)
    meta["server_at_finish"] = end
    keys = ("git_commit", "git_dirty", "dist_sha256", "started_at", "process_started", "pid",
            "dist_modified_after_start", "dist_modified_after_process_start")
    meta["server_changed_during_run"] = any(meta["server"].get(k) != end.get(k) for k in keys)
    return meta


def stamp(record: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """Attach the compact identity every record needs to be traced to its run."""
    srv = meta.get("server") or {}
    # The label record.py would give this identity: the short commit only when
    # the evidence ties it to the running code (reported build-info matching
    # the dist the process started from and unmodified since, or an inference
    # that passes its checks), else "unknown". Imported lazily because
    # record.py imports this module lazily too.
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from clusy_boundary.record import server_commit_label
        label, why = server_commit_label(srv)
    except Exception as exc:  # noqa: BLE001
        label, why = "unknown", f"label unavailable: {type(exc).__name__}"
    record["run"] = {
        "run_id": meta.get("run_id"),
        "cohort": meta.get("cohort"),
        "harness_commit": (meta.get("harness") or {}).get("commit"),
        "harness_dirty": (meta.get("harness") or {}).get("dirty"),
        "server_commit": srv.get("git_commit"),
        "server_dirty": srv.get("git_dirty"),
        "server_dist_sha256": srv.get("dist_sha256"),
        "server_identity_source": srv.get("source"),
        "server_commit_label": label,
        "server_commit_label_note": why,
        "server_changed_during_run": meta.get("server_changed_during_run"),
    }
    return record


RESULTS = ROOT / "results"


def _cohorts_in(path: Path) -> set[str | None]:
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
        rows = rows if isinstance(rows, list) else [rows]
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r.get("cohort") for r in rows if isinstance(r, dict)}


def shipped_record_conflict(record: str | Path, cohort: str | None) -> str | None:
    """Why a live run must not write `record` with `cohort`, or None if it may.

    The files under results/ are the paper's records, and the analysers select
    them by cohort name, so a rerun that adds rows under a cohort already
    recorded there (in the record itself or in the `runs_meta.jsonl` beside
    it) would replace the paper's rows with the rerun's, even when the rerun
    fails. A record file without cohort labels (the escalation files) is
    refused whenever it exists. `cohort=None` means the run labels itself with
    its fresh run id, which cannot collide. Paths outside results/ are never
    refused here."""
    path = Path(os.path.abspath(record)).resolve()
    results = RESULTS.resolve()
    try:
        shown = f"results/{path.relative_to(results).as_posix()}"
    except ValueError:
        return None
    hint = "pass a new --cohort and a scratch output path"
    if path.exists() and not path.is_dir():
        try:
            cohorts = _cohorts_in(path)
        except (OSError, ValueError) as exc:
            return f"{shown} is a shipped record and could not be read ({type(exc).__name__}); {hint}"
        if cohorts == {None}:
            return f"{shown} is a shipped record; pass a scratch output path"
        if cohort is not None and cohort in cohorts:
            return f"cohort {cohort!r} is already recorded in {shown}; {hint}"
    meta = path.parent / "runs_meta.jsonl"
    if cohort is not None and meta.exists() and cohort in _cohorts_in(meta):
        return f"cohort {cohort!r} is already recorded in results/{meta.relative_to(results).as_posix()}; {hint}"
    return None


def write_meta(meta: dict[str, Any], path: Path) -> None:
    """Append the run's metadata as one JSON line (one line per run)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(meta, default=str) + "\n")


if __name__ == "__main__":  # quick look at what a run would record
    url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CLUSY_API_URL", "http://localhost:8010")
    print(json.dumps(start_run("probe", api_url=url), indent=1, default=str))
