"""What moving across operating systems and instruction sets needs from the checks.

Found by the offline cross-substrate runs (`experiments/xsub_isa.py`, and
E15 `--local --substrates` between a macOS host process and a Linux
container):

* A Gaussian draw is the generator's next integers passed through the
  platform's libm, so macOS and glibc can differ in the last bit while the
  generator state is identical. The commit boundary must compare the state
  and the integer-derived draws exactly and DECLARE the Gaussian draws, not
  abort on them, and never call them equal when they are not.
* A session with no benchmark fixture (any Python workload) must validate on
  the generic fingerprint, not crash in fixture-specific checks.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from handoff.controller import compare_fingerprints  # noqa: E402


def _fp(normal=("0x1.1a7db68570ab6p+0",), state="aa", uniform=("0x1.4fa29c854965ap-1",)):
    rng = {"python": {"sha256": "p", "draws": ["1"]},
           "numpy": {"loaded": True, "sha256": state, "draws": list(uniform), "libm_draws": list(normal)},
           "torch_cpu": {"loaded": False},
           "named": {"np_gen": {"kind": "numpy.Generator", "sha256": "g", "draws": ["u"], "libm_draws": list(normal)}}}
    return {"rng": rng}


def test_gaussian_draws_that_differ_by_libm_are_declared_not_a_mismatch():
    src = _fp()
    dst = _fp(normal=("0x1.1a7db68570ab5p+0",))  # one ulp: glibc vs Apple libm, measured
    v = compare_fingerprints(src, dst)
    assert v["equal"], v
    assert v["declared"]["rng_libm_draws"] == {"numpy": "platform_arithmetic", "named:np_gen": "platform_arithmetic"}


def test_equal_gaussian_draws_are_reported_equal():
    v = compare_fingerprints(_fp(), _fp())
    assert v["equal"]
    assert v["declared"]["rng_libm_draws"] == {"numpy": "equal", "named:np_gen": "equal"}


def test_a_different_generator_state_is_still_a_mismatch():
    v = compare_fingerprints(_fp(), _fp(state="bb"))
    assert not v["equal"] and "rng" in v["mismatched"]


def test_a_different_uniform_draw_is_still_a_mismatch():
    v = compare_fingerprints(_fp(), _fp(uniform=("0x1.0p-1",)))
    assert not v["equal"] and "rng" in v["mismatched"]


def test_the_comparison_does_not_mutate_its_inputs():
    src, dst = _fp(), _fp(normal=("0x1.0p+0",))
    a, b = copy.deepcopy(src), copy.deepcopy(dst)
    compare_fingerprints(src, dst)
    assert src == a and dst == b


def test_a_session_without_the_fixture_is_declared_not_crashed():
    import numpy as np

    import oracles

    ns = {"W": np.zeros(3), "opt": {"lr": 0.1}}
    rows = oracles.verify(ns, None, destination_device="cpu", source_device="cpu", continuation="isolated")
    assert [(r.name, r.ok, r.mode) for r in rows] == [("workload checks", None, "declared")]


def test_a_loaded_cuda_driver_rules_out_fork_isolation(tmp_path):
    """Live E15 hop 1 on a Modal T4: `torch.cuda.is_available()` had loaded the
    driver without setting torch's initialized flag, so the fork was judged
    safe and the child's optimizer step failed with "CUDA error:
    initialization error". The driver being mapped must rule the fork out."""
    import oracles

    maps = tmp_path / "maps"
    maps.write_text("7f00-7f01 r-xp 0 0:0 0 /usr/lib/x86_64-linux-gnu/libcuda.so.535.104.05\n")
    assert "CUDA driver is loaded" in (oracles.fork_unsafe_reason(maps_path=str(maps), device_nodes=()) or "")


def test_a_cuda_build_on_a_gpu_host_rules_out_fork_isolation(tmp_path, monkeypatch):
    import torch

    import oracles

    node = tmp_path / "nvidiactl"
    node.write_text("")
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising=False)
    reason = oracles.fork_unsafe_reason(maps_path=str(tmp_path / "absent"), device_nodes=(str(node),))
    assert reason and "GPU host" in reason


def test_a_cpu_host_still_forks(tmp_path):
    import oracles

    maps = tmp_path / "maps"
    maps.write_text("7f00-7f01 r-xp 0 0:0 0 /usr/lib/libc.so.6\n")
    assert oracles.fork_unsafe_reason(maps_path=str(maps), device_nodes=(str(tmp_path / "none"),)) is None
