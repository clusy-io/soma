"""Minimal API client for the resource-escalation experiments.

It talks to the research build of the platform API (`CLUSY_API_URL`, default
http://localhost:8002) and authenticates with a user bearer token taken from
`CLUSY_CAMPAIGN_JWT` (or the file named by `CLUSY_CAMPAIGN_JWT_FILE`), which
is what project creation requires. The API is not part of this artifact; see
docs/EXPERIMENTS.md for the calls it must answer.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class Resp:
    def __init__(self, status: int, payload, elapsed_ms: float) -> None:
        self.status, self.payload, self.elapsed_ms = status, payload, elapsed_ms

    @property
    def error_code(self):
        e = self.payload.get("error") if isinstance(self.payload, dict) else None
        return (e or {}).get("code")

    @property
    def error_message(self):
        e = self.payload.get("error") if isinstance(self.payload, dict) else None
        return (e or {}).get("message")


def env_token() -> str:
    """The bearer token for the API under test.

    Read from the file named by `CLUSY_CAMPAIGN_JWT_FILE` when that is set,
    otherwise from `CLUSY_CAMPAIGN_JWT`. User tokens expire (after an hour on
    the platform used), which is shorter than a single escalation sweep, so
    the client calls this again once on a 401: with the file form, an external
    refresher that rewrites the file keeps a long sweep alive. A stale token
    otherwise turns every later create into a 401, which is a harness
    condition, not a result.
    """
    path = os.environ.get("CLUSY_CAMPAIGN_JWT_FILE", "").strip()
    if path:
        with open(path) as fh:
            tok = fh.read().strip()
    else:
        tok = os.environ.get("CLUSY_CAMPAIGN_JWT", "").strip()
    if not tok:
        raise RuntimeError("set CLUSY_CAMPAIGN_JWT (or CLUSY_CAMPAIGN_JWT_FILE) to a bearer "
                           "token for the API under test")
    return tok


class Client:
    def __init__(self, base: str | None = None, jwt: str | None = None) -> None:
        base = base or os.environ.get("CLUSY_API_URL", "http://localhost:8002")
        if urllib.parse.urlparse(base).hostname not in LOCAL_HOSTS:
            raise RuntimeError(
                f"{base} is not a local stack. This creates projects, provisions "
                f"GPUs and deletes rows; it must run against a local API.")
        self.base = base.rstrip("/")
        self._jwt = jwt or env_token()

    def request(self, method: str, path: str, body=None, *, timeout_s: float = 60.0,
                _retried: bool = False) -> Resp:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._jwt}"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                raw, status = r.read().decode("utf-8", "replace"), r.status
        except urllib.error.HTTPError as exc:
            raw, status = exc.read().decode("utf-8", "replace"), exc.code
            # An expired token is a harness condition, not a result. Re-read the
            # token once and retry, so a sweep that outlives its token does not
            # report a column of 401s as if they were system failures.
            if status == 401 and not _retried:
                self._jwt = env_token()
                return self.request(method, path, body, timeout_s=timeout_s,
                                    _retried=True)
        except Exception as exc:
            return Resp(0, {"error": {"code": "TRANSPORT",
                                      "message": f"{type(exc).__name__}: {exc}"}},
                        (time.perf_counter() - t0) * 1000.0)
        try:
            payload = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            payload = {"_raw": raw[:4000]}
        # A list response (GET /projects) must stay a list: wrapping it in a
        # dict silently hides every project from any caller that iterates it,
        # which is how a leaked GPU goes unnoticed.
        if not isinstance(payload, (dict, list)):
            payload = {"_raw": payload}
        return Resp(status, payload, (time.perf_counter() - t0) * 1000.0)

    def list_projects(self) -> list:
        r = self.request("GET", "/projects", timeout_s=60)
        p = r.payload
        if isinstance(p, list):
            return p
        return p.get("projects") or p.get("items") or []

    # -- lifecycle ---------------------------------------------------------

    def create_project(self, label: str, profile: str = "gpu_t4") -> str:
        # The clusy-exp- prefix is what the cleanup sweeper and
        # `teardown.py --all` select on. A name formatted by hand that misses it
        # is invisible to them, which is the difference between a leaked GPU
        # being reaped and one running to the provider cap.
        name = f"clusy-exp-s3-esc-{label}-{uuid.uuid4().hex[:8]}"
        r = self.request("POST", "/projects",
                         {"name": name, "runtimeProfile": profile, "sandboxType": "ml"})
        if r.status != 201:
            raise RuntimeError(f"project create {r.status}: {r.payload}")
        got = r.payload.get("runtimeProfile") or (r.payload.get("settings") or {}).get("runtimeProfile")
        if got != profile:
            raise RuntimeError(
                f"asked for {profile!r}, project came back as {got!r}; creation "
                f"clamps rather than rejects, so this run would name a tier that "
                f"never existed")
        return str(r.payload["id"])

    def delete_project(self, pid: str) -> dict:
        r = self.request("DELETE", f"/projects/{pid}", timeout_s=180.0)
        return {"status": r.status, "message": r.error_message}

    def start(self, pid: str, timeout_s: float = 900.0) -> Resp:
        return self.request("POST", f"/projects/{pid}/sandbox/resume", {}, timeout_s=timeout_s)

    def execute(self, pid: str, code: str, timeout_s: float = 1800.0,
                cell_timeout_ms: int | None = None) -> Resp:
        """Execute a cell.

        `timeout_ms` must be sent explicitly: the route defaults a cell to
        **120 s** and interrupts the kernel past it, which silently truncates any
        training phase longer than two minutes. The HTTP read timeout is a
        separate budget and does not extend the cell's. Schema cap is 6 h.
        """
        body = {"code": code}
        if cell_timeout_ms:
            body["timeout_ms"] = min(int(cell_timeout_ms), 21_600_000)
        return self.request("POST", f"/projects/{pid}/sandbox/execute",
                            body, timeout_s=timeout_s)

    def switch(self, pid: str, profile: str, timeout_s: float = 2400.0) -> Resp:
        """The transition itself. One PATCH, no retries.

        A 409 here is a RESULT, never something to retry: retrying would
        overwrite a finding with a second sample of the same finding.
        """
        return self.request("PATCH", f"/projects/{pid}",
                            {"runtimeProfile": profile, "stateTransfer": "transfer"},
                            timeout_s=timeout_s)

    def profile_of(self, pid: str):
        p = self.request("GET", f"/projects/{pid}").payload
        return p.get("runtimeProfile") or (p.get("settings") or {}).get("runtimeProfile")


def emit(resp: Resp) -> dict:
    """Pull the structured markers a workload cell prints back out of stdout."""
    out = {}
    for line in (resp.payload.get("stdout") or "").splitlines():
        if line.startswith("__ESC__:"):
            try:
                out.update(json.loads(line[len("__ESC__:"):]))
            except json.JSONDecodeError:
                pass
    return out
