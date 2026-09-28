"""A local stand-in for the Clusy API, for dry-running the live harnesses.

Every live experiment (E11-E14) drives the API through a handful of calls:
create a project, execute code in its kernel, PATCH its runtime profile,
delete it. `LocalApi` implements those calls against persistent local Python
processes, so a harness can be run end to end on a laptop before any sandbox
or GPU time is spent. It is a test double for the HARNESS, not a model of the
system: nothing measured through it is a result.

A "kernel" is a child interpreter that executes code blocks in its real
`__main__`, the way a Jupyter kernel does, and returns captured stdout plus
any exception. `patch_profile` simulates the production switch shape: dump
`__main__` with dill (ambient and underscore names filtered, as the production
capture does), capture the four process-global RNG streams in an envelope,
kill the kernel, start a fresh one, load the session and apply the envelope.
Two switches can be broken on purpose, which is how a harness proves that its
checks can fail:

  LocalApi(rng_restore=False)   the envelope is not applied on the destination
  LocalApi(rng_perturb=True)    the envelope is applied, then the torch CPU
                                stream is advanced by one draw

There is no CUDA here; every profile runs on the local CPU.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

_KERNEL = r'''
import sys, io, json, struct, traceback, contextlib
_stdin, _stdout = sys.stdin.buffer, sys.stdout.buffer
def __clusy_local_loop():
    g = sys.modules["__main__"].__dict__
    while True:
        hdr = _stdin.read(4)
        if len(hdr) < 4:
            return
        (n,) = struct.unpack(">I", hdr)
        code = _stdin.read(n).decode("utf-8")
        buf, err = io.StringIO(), None
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<cell>", "exec"), g)
        except SystemExit:
            pass
        except BaseException as e:  # noqa: BLE001
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-4000:]
        out = json.dumps({"stdout": buf.getvalue(), "error": err}).encode("utf-8")
        _stdout.write(struct.pack(">I", len(out)) + out); _stdout.flush()
__clusy_local_loop()
'''

# The simulated production capture: filter the real __main__ for the duration
# of the dump, restore it afterwards, and record the RNG envelope.
_CAPTURE = r'''
def __clusy_local_capture(_path):
    import sys, random, pickle, types
    import dill
    g = sys.modules["__main__"].__dict__
    keep = {"__name__", "__builtins__", "__doc__", "__package__", "__loader__", "__spec__"}
    stash = {}
    try:
        for k in list(g):
            if k in keep:
                continue
            if k.startswith("_") or isinstance(g[k], types.ModuleType):
                stash[k] = g.pop(k)
        dill.dump_session(_path + ".session", main=sys.modules["__main__"], byref=True)
    finally:
        g.update(stash)
    env = {"random": random.getstate()}
    np = sys.modules.get("numpy")
    if np is not None:
        env["numpy"] = np.random.get_state()
    torch = sys.modules.get("torch")
    if torch is not None:
        env["torch"] = torch.get_rng_state()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            env["cuda"] = torch.cuda.get_rng_state_all()
    with open(_path + ".rng", "wb") as f:
        pickle.dump(env, f)
__clusy_local_capture(%(path)r)
del __clusy_local_capture
'''

_RESTORE = r'''
def __clusy_local_restore(_path, _apply, _perturb):
    import sys, random, pickle
    import dill
    dill.load_session(_path + ".session", main=sys.modules["__main__"])
    env = pickle.load(open(_path + ".rng", "rb"))
    if _apply:
        random.setstate(env["random"])
        if "numpy" in env:
            import numpy as np
            np.random.set_state(env["numpy"])
        if "torch" in env:
            import torch
            torch.set_rng_state(env["torch"])
            if "cuda" in env and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(env["cuda"])
    if _perturb:
        import torch
        torch.rand(1)
__clusy_local_restore(%(path)r, %(apply)r, %(perturb)r)
del __clusy_local_restore
'''


class _Kernel:
    def __init__(self, cwd: Path, python: str):
        self.proc = subprocess.Popen([python, "-u", "-c", _KERNEL], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(cwd))

    def run(self, code: str) -> dict[str, Any]:
        data = code.encode("utf-8")
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(struct.pack(">I", len(data)) + data); self.proc.stdin.flush()
        hdr = self.proc.stdout.read(4)
        if len(hdr) < 4:
            err = self.proc.stderr.read().decode("utf-8", "replace") if self.proc.stderr else ""
            return {"stdout": "", "error": f"kernel died: {err[-2000:]}"}
        (n,) = struct.unpack(">I", hdr)
        return json.loads(self.proc.stdout.read(n).decode("utf-8"))

    def kill(self) -> None:
        try:
            self.proc.kill(); self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass


class _DockerKernel(_Kernel):
    """The same kernel loop inside a fresh, network-less container.

    The container is a different isolation substrate (and, with a Linux image
    on a macOS host, a different OS; with `platform=linux/amd64` on an arm64
    host, a different ISA under emulation). Nothing is shared with the host:
    code and capsule bytes travel only through the execute channel, exactly
    as they do to a remote sandbox.
    """

    def __init__(self, image: str, platform: str | None = None):
        self.name = f"clusy-localapi-{uuid.uuid4().hex[:12]}"
        cmd = ["docker", "run", "-i", "--rm", "--network", "none", "--name", self.name]
        if platform:
            cmd += ["--platform", platform]
        cmd += ["--entrypoint", "sh", image, "-c",
                'mkdir -p /tmp/work && cd /tmp/work && exec python3 -u -c "$0"', _KERNEL]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def kill(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
        super().kill()


def parse_substrates(spec: str | None) -> dict[str, str]:
    """`cpu=python:/path/to/python,gpu_t4=docker:IMAGE[@linux/amd64]` -> {profile: target}."""
    out: dict[str, str] = {}
    for part in (spec or "").split(","):
        if part.strip():
            prof, target = part.split("=", 1)
            out[prof.strip()] = target.strip()
    return out


def substrate_identity(target: str) -> dict[str, Any]:
    """What a substrate is, recorded with the run (OS, ISA, Python, versions)."""
    probe = ("import sys, platform, json\n"
             "v = {}\n"
             "for m in ('torch', 'numpy', 'dill'):\n"
             "    try:\n"
             "        v[m] = __import__(m).__version__\n"
             "    except Exception:\n"
             "        v[m] = None\n"
             "print(json.dumps({'os': platform.system(), 'machine': platform.machine(),"
             " 'python': sys.version.split()[0], **v}))")
    kind, _, rest = target.partition(":")
    if kind == "docker":
        image, _, plat = rest.partition("@")
        cmd = ["docker", "run", "--rm", "--network", "none"] + (["--platform", plat] if plat else []) \
            + ["--entrypoint", "python3", image, "-c", probe]
    else:
        cmd = [rest or sys.executable, "-c", probe]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    ident = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else {"error": p.stderr[-400:]}
    ident["substrate"] = "container" if kind == "docker" else "host process"
    ident["target"] = target
    return ident


class LocalApi:
    """Drop-in for `handoff.controller.Api` in dry runs. See module doc.

    `substrates` maps a profile to where its kernels run: `python:<path>` (a
    host process with that interpreter) or `docker:<image>[@<platform>]` (a
    container). Unmapped profiles use `python`.
    """

    def __init__(self, *, rng_restore: bool = True, rng_perturb: bool = False,
                 python: str | None = None, workdir: Path | None = None,
                 substrates: dict[str, str] | None = None):
        self.rng_restore, self.rng_perturb = rng_restore, rng_perturb
        self.python = python or sys.executable
        self.substrates = dict(substrates or {})
        self.root = Path(workdir or tempfile.mkdtemp(prefix="clusy-localapi-"))
        self.projects: dict[str, dict[str, Any]] = {}
        self.base, self.key = "local://", "local"
        self.calls: list[tuple[str, str]] = []

    def _new_kernel(self, profile: str, cwd: Path) -> _Kernel:
        kind, _, rest = self.substrates.get(profile, "").partition(":")
        if kind == "docker":
            image, _, plat = rest.partition("@")
            return _DockerKernel(image, plat or None)
        return _Kernel(cwd, rest if kind == "python" and rest else self.python)

    # -- projects -------------------------------------------------------
    def create_project(self, name: str, profile: str) -> str:
        pid = str(uuid.uuid4())
        cwd = self.root / pid
        cwd.mkdir(parents=True)
        self.projects[pid] = {"id": pid, "name": name, "profile": profile, "cwd": cwd,
                              "kernel": self._new_kernel(profile, cwd), "memory": None}
        self.calls.append(("create", pid))
        return pid

    def list_projects(self, timeout: float = 300.0) -> list[dict]:
        return [{"id": p["id"], "name": p["name"], "runtimeProfile": p["profile"]} for p in self.projects.values()]

    def delete_project(self, pid: str) -> int:
        p = self.projects.pop(pid, None)
        if p is None:
            return 404
        p["kernel"].kill()
        self.calls.append(("delete", pid))
        return 204

    def pause(self, pid: str) -> int:
        return 200 if pid in self.projects else 404

    # -- execution -----------------------------------------------------
    def execute(self, pid: str, code: str, timeout_ms: int = 600_000) -> tuple[int, str, dict]:
        p = self.projects.get(pid)
        if p is None:
            return 404, "", {"error": {"code": "NOT_FOUND"}}
        res = p["kernel"].run(code)
        payload = {"stdout": res["stdout"], "error": res["error"]}
        return 200, res["stdout"], payload

    def witness(self, pid: str, code: str, timeout_ms: int = 600_000) -> dict:
        from handoff.controller import MARK, HandoffError
        st, out, p = self.execute(pid, code, timeout_ms)
        if st != 200:
            raise HandoffError("execute_failed", "execute", f"HTTP {st}")
        line = next((l for l in out.splitlines() if l.startswith(MARK)), None)
        if line is None:
            raise HandoffError("no_witness", "execute", f"kernel output: {(p.get('error') or out or '')[-600:]}")
        return json.loads(line[len(MARK):])

    # -- the simulated switch -----------------------------------------------
    def _switch(self, pid: str, *, profile: str | None = None, memory: int | None = None) -> tuple[int, dict]:
        p = self.projects.get(pid)
        if p is None:
            return 404, {"error": {"code": "NOT_FOUND"}}
        t0 = time.perf_counter()
        path = str(p["cwd"] / f".switch-{uuid.uuid4().hex[:8]}")
        cap = p["kernel"].run(_CAPTURE % {"path": path})
        if cap["error"]:
            return 409, {"error": {"code": "RUNTIME_STATE_CAPTURE_FAILED", "retryable": False, "detail": cap["error"][-400:]}}
        t_cap = time.perf_counter()
        p["kernel"].kill()
        p["kernel"] = self._new_kernel(profile or p["profile"], p["cwd"])
        res = p["kernel"].run(_RESTORE % {"path": path, "apply": self.rng_restore, "perturb": self.rng_perturb})
        t_res = time.perf_counter()
        if res["error"]:
            return 500, {"error": {"code": "LOCAL_RESTORE_FAILED", "detail": res["error"][-400:]}}
        if profile is not None:
            p["profile"] = profile
        if memory is not None:
            p["memory"] = memory
        self.calls.append(("switch", pid))
        ms = lambda a, b: round((b - a) * 1000, 1)  # noqa: E731
        return 200, {"id": pid, "runtimeProfile": p["profile"],
                     "switchPhases": {"capture": ms(t0, t_cap), "destroy": 0, "offload": 0, "flush": 0},
                     "switchTimeline": {"total_ms": ms(t0, t_res), "steps": [
                         {"step": "ckpt_dump_execute_returned", "ms": ms(t0, t_cap)},
                         {"step": "destroy_returned", "ms": 0.0},
                         {"step": "local_restore_returned", "ms": ms(t_cap, t_res)}]},
                     "local_simulation": True}

    def patch_profile(self, pid: str, profile: str, timeout: float = 1500.0) -> tuple[int, dict]:
        return self._switch(pid, profile=profile)

    def _req(self, method: str, path: str, body: dict | None = None, timeout: float = 120.0) -> tuple[int, Any]:
        """The subset of raw requests the harnesses make."""
        parts = path.strip("/").split("/")
        if method == "PATCH" and len(parts) == 2 and parts[0] == "projects":
            body = body or {}
            if "runtimeProfile" in body:
                return self._switch(parts[1], profile=body["runtimeProfile"])
            if "runtimeMemoryMiB" in body:
                return self._switch(parts[1], memory=body["runtimeMemoryMiB"])
        if method == "GET" and path.rstrip("/") == "/projects":
            return 200, self.list_projects()
        if method == "DELETE" and len(parts) == 2 and parts[0] == "projects":
            return self.delete_project(parts[1]), {}
        return 501, {"error": {"code": "NOT_SIMULATED", "detail": f"{method} {path}"}}

    def close(self) -> None:
        for pid in list(self.projects):
            self.delete_project(pid)


if __name__ == "__main__":  # self-test: a switch preserves RNG, and a broken one is detected
    sys.path.insert(0, str(ROOT / "src"))
    from handoff.controller import MARK
    probe = ("import random, json, torch, numpy as np\n"
             "print(%r + json.dumps({'py': random.random(), 'np': float(np.random.rand()), 't': float(torch.rand(1))}))" % MARK)
    results = {}
    for label, kw in (("restored", {}), ("not_restored", {"rng_restore": False}), ("perturbed", {"rng_perturb": True})):
        api = LocalApi(**kw)
        pid = api.create_project("selftest", "cpu")
        api.witness(pid, "import random, numpy as np, torch\nrandom.seed(1); np.random.seed(1); torch.manual_seed(1)\n"
                         "x = [random.random() for _ in range(3)]\nprint(%r + '{}')" % MARK)
        # clone reference: what the source would draw next
        ref = api.witness(pid, "import random, json, torch, numpy as np, copy\n"
                               "_r = random.Random(); _r.setstate(random.getstate())\n"
                               "_n = np.random.RandomState(); _n.set_state(np.random.get_state())\n"
                               "_t = torch.Generator(); _t.set_state(torch.get_rng_state())\n"
                               "print(%r + json.dumps({'py': _r.random(), 'np': float(_n.rand()), 't': float(torch.rand(1, generator=_t))}))\n"
                               "del _r, _n, _t" % MARK)
        st, _ = api.patch_profile(pid, "cpu")
        got = api.witness(pid, probe)
        results[label] = {k: got[k] == ref[k] for k in ref}
        api.close()
    print(json.dumps(results))
    ok = all(results["restored"].values()) and not all(results["not_restored"].values()) \
        and results["perturbed"]["t"] is False and results["perturbed"]["py"] is True
    print("selftest", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
