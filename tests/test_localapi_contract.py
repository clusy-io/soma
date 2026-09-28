"""`experiments/localapi.py` as the reference implementation of the runtime contract.

A cloud rerun needs an API that can create a runtime, execute code in it,
list runtimes and delete them. `LocalApi` is the implementation that ships,
and every offline run goes through it, so these tests pin the semantics the
controller relies on. They use only local Python processes (the substrate
`python:<this interpreter>`), never Docker:

* `parse_substrates` and `substrate_identity`, which map profiles to where
  their kernels run and record what that place is;
* a runtime's namespace persists across execute calls, and runtimes are
  isolated from each other;
* execute returns what the code printed, which is where the witness line
  travels;
* a user exception is reported in the payload and the runtime survives it;
* delete removes the runtime and its kernel process, so a later execute is a
  404, and the listing shows exactly the live runtimes by name (which is how
  `reclaim` finds a destination a crashed run left behind).
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from handoff.controller import MARK, HandoffError  # noqa: E402
from localapi import LocalApi, parse_substrates, substrate_identity  # noqa: E402

PY = f"python:{sys.executable}"


@pytest.fixture
def api(tmp_path):
    a = LocalApi(workdir=tmp_path / "kernels", substrates={"cpu": PY, "gpu_t4": PY})
    yield a
    a.close()


def test_parse_substrates_maps_profiles_to_targets_and_rejects_a_bare_profile():
    spec = f" cpu = {PY} ,, gpu_t4=docker:clusy/port:1@linux/amd64, "
    assert parse_substrates(spec) == {"cpu": PY, "gpu_t4": "docker:clusy/port:1@linux/amd64"}
    # Only the first "=" separates profile from target; a path may contain one.
    assert parse_substrates("cpu=python:/opt/py=3.11/bin/python") == {"cpu": "python:/opt/py=3.11/bin/python"}
    assert parse_substrates(None) == {} and parse_substrates(" , ") == {}
    with pytest.raises(ValueError):
        parse_substrates(f"cpu={PY},gpu_t4")


def test_substrate_identity_describes_this_interpreter_and_reports_a_broken_one(tmp_path):
    import dill
    import numpy

    ident = substrate_identity(PY)
    assert "error" not in ident, ident
    assert ident["substrate"] == "host process" and ident["target"] == PY
    assert ident["python"] == platform.python_version()
    assert (ident["os"], ident["machine"]) == (platform.system(), platform.machine())
    assert ident["dill"] == dill.__version__ and ident["numpy"] == numpy.__version__

    broken = tmp_path / "python3"
    broken.write_text("#!/bin/sh\necho 'interpreter refused to start' >&2\nexit 3\n")
    broken.chmod(0o755)
    bad = substrate_identity(f"python:{broken}")
    assert "interpreter refused to start" in bad["error"]
    assert bad["substrate"] == "host process" and bad["target"] == f"python:{broken}"
    assert "python" not in bad


def test_the_namespace_persists_across_execute_calls_and_stdout_is_returned(api):
    pid = api.create_project("persist", "cpu")
    st, out, payload = api.execute(pid, "import sys\nx = 41\n")
    assert (st, out, payload["error"]) == (200, "", None)

    st, out, payload = api.execute(pid, "x += 1\nprint('x is', x)\nprint(sys.executable)")
    assert st == 200 and payload["error"] is None, payload
    # The kernel runs on the mapped substrate's interpreter.
    assert out.splitlines() == ["x is 42", sys.executable]
    assert payload["stdout"] == out

    # The controller's witness: the one marker line among other output.
    got = api.witness(pid, "import json\nprint('noise')\nprint(%r + json.dumps({'x': x}))\nprint('tail')" % MARK)
    assert got == {"x": 42}


def test_separate_runtimes_are_isolated(api):
    a = api.create_project("a", "cpu")
    b = api.create_project("b", "gpu_t4")
    api.execute(a, "import os\nsecret = 'only in a'")

    st, out, payload = api.execute(b, "print(secret)")
    assert st == 200 and out == ""
    assert "NameError" in payload["error"]

    api.execute(b, "import os\nsecret = 'b wrote this'")
    _, out_a, _ = api.execute(a, "print(secret)\nprint(os.getpid())\nprint(os.getcwd())")
    _, out_b, _ = api.execute(b, "print(os.getpid())\nprint(os.getcwd())")
    secret_a, pid_a, cwd_a = out_a.splitlines()
    pid_b, cwd_b = out_b.splitlines()
    assert secret_a == "only in a"
    assert pid_a != pid_b
    assert Path(cwd_a).resolve() == (api.root / a).resolve()
    assert Path(cwd_b).resolve() == (api.root / b).resolve()


def test_a_user_exception_is_reported_and_the_runtime_survives_it(api):
    pid = api.create_project("raises", "cpu")
    _, before, _ = api.execute(pid, "import os\nprint(os.getpid())")

    st, out, payload = api.execute(pid, "kept = 7\nprint('ran up to here')\nraise ValueError('user bug')\nnever = 1")
    assert st == 200
    assert out == "ran up to here\n"
    assert "Traceback" in payload["error"] and "ValueError: user bug" in payload["error"]

    # Same process, state before the raise kept, nothing after it ran.
    st, out, payload = api.execute(pid, "print(os.getpid(), kept, 'never' in globals())")
    assert st == 200 and payload["error"] is None, payload
    assert out.split() == [before.strip(), "7", "False"]

    # SystemExit from user code ends the cell, not the runtime.
    st, out, payload = api.execute(pid, "print('bye')\nraise SystemExit(3)")
    assert (st, out, payload["error"]) == (200, "bye\n", None)
    assert api.execute(pid, "print(os.getpid())")[1] == before

    # A witness program that raises before its marker surfaces the error.
    with pytest.raises(HandoffError) as e:
        api.witness(pid, "raise KeyError('missing_column')")
    assert e.value.reason == "no_witness" and "KeyError" in e.value.detail


def test_delete_removes_the_runtime_so_a_later_execute_fails(api):
    doomed = api.create_project("doomed", "cpu")
    kept = api.create_project("kept", "cpu")
    api.execute(kept, "n = 5")
    kernel_pid = int(api.execute(doomed, "import os\nprint(os.getpid())")[1])

    assert api.delete_project(doomed) == 204
    # The kernel process is gone, not merely forgotten.
    with pytest.raises(ProcessLookupError):
        os.kill(kernel_pid, 0)
    assert api.execute(doomed, "print(1)") == (404, "", {"error": {"code": "NOT_FOUND"}})
    with pytest.raises(HandoffError) as e:
        api.witness(doomed, "print(1)")
    assert e.value.reason == "execute_failed"
    assert api.delete_project(doomed) == 404
    assert api.pause(doomed) == 404

    # The other runtime is untouched.
    assert api.pause(kept) == 200
    assert api.execute(kept, "print(n)")[:2] == (200, "5\n")
    assert api.calls == [("create", doomed), ("create", kept), ("delete", doomed)]


def test_the_listing_shows_the_live_runtimes_by_name(api):
    assert api.list_projects() == []
    src = api.create_project("clusy-exp-handoff-m1-src", "cpu")
    dst = api.create_project("clusy-exp-handoff-m1", "gpu_t4")
    assert src != dst

    listed = api.list_projects()
    assert {p["id"]: p for p in listed} == {
        src: {"id": src, "name": "clusy-exp-handoff-m1-src", "runtimeProfile": "cpu"},
        dst: {"id": dst, "name": "clusy-exp-handoff-m1", "runtimeProfile": "gpu_t4"},
    }
    # The raw routes the harnesses call answer the same way.
    assert api._req("GET", "/projects") == (200, listed)
    assert api._req("DELETE", f"/projects/{dst}") == (204, {})
    assert [p["name"] for p in api.list_projects()] == ["clusy-exp-handoff-m1-src"]
