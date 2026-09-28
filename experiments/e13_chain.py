"""E13: continued training across repeated migrations.

One training job, switched repeatedly with a block of training steps between
switches, using the platform's in-place switch (PATCH runtimeProfile or a RAM-tier
PATCH). After every hop the twelve transition oracles run against an
expectation recorded on the source just before the hop, plus chain-specific
checks: tied weights are still one object, a tensor view still shares storage,
the loss on a fixed batch is continuous, the optimizer step count is intact,
and every RNG stream equals its pre-hop state.

Runs (default `--chains control_t4_a,control_t4_b,same,control_cpu,hetero`):

  control_t4_a   gpu_t4, one uninterrupted process: seed, then per block the
  control_t4_b   same train block and the same verify block the `same` chain
                 runs, minus the PATCH. Two identical runs measure the
                 run-to-run noise floor on one SKU.
  same           gpu_t4 alternating RAM tiers (16 GiB <-> 32 GiB): a memory-only
                 shape change the API treats as a real switch (destroy and
                 reprovision). Compared bitwise with both T4 controls.
  control_cpu    cpu, nine blocks: an observational reference for the hetero
                 chain, which spans devices and so cannot be matched bitwise
                 by any single-device control.
  hetero         T4 -> CPU -> T4 -> A100 -> T4 -> CPU -> T4 -> A100 -> CPU -> T4.

WHAT CHANGED, AND WHY (the critique this version answers):

  RNG equality used to compare PREFIXES (the first words of each state). The
  torch prefix is the initial seed, which never changes; the Python and NumPy
  prefixes change only when the 624-word state regenerates. None of them saw
  the stream POSITION, CUDA was not looked at, and the snapshot was taken on
  the destination AFTER the oracles ran. Now every program fingerprints the
  FULL state of every stream (Python, NumPy legacy, torch CPU, each CUDA
  device, and every generator bound in the namespace including
  `loader.generator`) and records the next draws from CLONED generators, so
  the fingerprint never advances a live stream. The source fingerprint is the
  last thing a train block does; the destination fingerprint is the first
  thing a verify block does.

  The control used to run only the training block, while the chain also ran
  the verify block at every hop (whose continuation oracle takes an extra
  optimizer step). A control now performs exactly the chain's per-hop
  sequence minus the PATCH, so "same vs control" is a bitwise test.

  TF32 was blamed for cross-SKU drift by association only. Both train and
  verify now recompute the two drift-sensitive quantities (fixed-batch loss,
  forward-output sum) under the process's default flags AND with TF32
  disabled, restoring the flags afterwards: a controlled, within-hop ablation
  on identical bytes.

Every program shipped into a kernel runs inside a function that is deleted on
exit, and the pre-hop expectation travels to the destination through the
HARNESS (embedded in the verify program), not through `__main__`. The next
switch dumps `__main__`, so anything the harness leaves there becomes part of
what is being measured; and an expectation carried by the system under test
is not an independent reference.

Local dry run (no sandbox, no GPU; see experiments/localapi.py):

  python experiments/e13_chain.py --local --steps 3 --out <scratch dir>
  python experiments/e13_chain.py --local --steps 3 --local-break rng_restore --chains same,control_t4_a
  python experiments/e13_chain.py --local --steps 3 --local-break rng_perturb --chains same,control_t4_a
  python experiments/e13_chain.py --selftest

`--local` maps every profile to `cpu`, writes to a scratch directory, and ends
with PASS/FAIL checks whose expectation follows `--local-break` (override with
`--local-expect` to watch a check fail on a broken switch).
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import math
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from handoff.controller import Api, MARK, _prelude  # noqa: E402
import runmeta  # noqa: E402
from e13_analyse import GLOBAL_STREAMS, compare_runs, describe_divergence, rng_verdicts, summarise  # noqa: E402


def _indent(block: str, pad: str = "    ") -> str:
    return "\n".join(pad + line if line.strip() else line for line in block.splitlines())


HETERO = ["gpu_t4", "cpu", "gpu_t4", "gpu_a100_40", "gpu_t4", "cpu", "gpu_t4", "gpu_a100_40", "cpu", "gpu_t4"]
DEVICE = lambda p: "cuda" if p.startswith("gpu") else "cpu"  # noqa: E731
DEFAULT_CHAINS = "control_t4_a,control_t4_b,same,control_cpu,hetero"


# ---------------------------------------------------------------------------
# Helpers shipped INTO the kernels (source text, never a harness import)
# ---------------------------------------------------------------------------
#
# These are defined inside each program's wrapper function, so they are locals
# of a function that is deleted when the program ends and nothing of them
# reaches `__main__`. `--selftest` execs the same text in this process.

_HELPERS_SRC = r'''
def _e13_sha(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()

def _e13_hex(values):
    # float.hex is exact for every float64 and for every float32 widened to
    # float64, so equal strings mean equal bits.
    return [float(v).hex() for v in values]

def _e13_canon(obj, h):
    # A lossless encoding of a nested state object. repr() is not usable for
    # ndarrays: numpy elides long arrays with "...".
    import sys
    np = sys.modules.get("numpy")
    if isinstance(obj, dict):
        h.update(b"{")
        for k in sorted(obj, key=repr):
            h.update(repr(k).encode() + b":")
            _e13_canon(obj[k], h)
            h.update(b",")
        h.update(b"}")
    elif isinstance(obj, (list, tuple)):
        h.update(b"(" if isinstance(obj, tuple) else b"[")
        for v in obj:
            _e13_canon(v, h)
            h.update(b",")
        h.update(b")")
    elif np is not None and isinstance(obj, np.ndarray):
        h.update(("nd:%s:%r:" % (obj.dtype.str, obj.shape)).encode())
        h.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, float):
        h.update(b"f:" + float(obj).hex().encode())
    else:
        h.update(("%s:%r" % (type(obj).__name__, obj)).encode())

def _e13_generator_fp(v):
    # Fingerprint of one generator OBJECT: full-state hash plus the next draws
    # of a clone. Returns None when v is not a generator this knows.
    import sys, random, hashlib
    np, torch = sys.modules.get("numpy"), sys.modules.get("torch")
    try:
        if torch is not None and isinstance(v, torch.Generator):
            s = v.get_state()
            c = torch.Generator(device=v.device)
            c.set_state(s)
            return {"kind": "torch.Generator", "device": str(v.device), "sha256": _e13_sha(s.cpu().numpy().tobytes()),
                    "draws": _e13_hex(torch.rand(4, generator=c, device=v.device).tolist())}
        if np is not None and isinstance(v, np.random.Generator):
            bg = v.bit_generator
            s = bg.state
            h = hashlib.sha256()
            _e13_canon(s, h)
            c = np.random.Generator(type(bg)())
            c.bit_generator.state = s
            return {"kind": "numpy.Generator", "bit_generator": type(bg).__name__, "sha256": h.hexdigest(),
                    "draws": _e13_hex(c.random(4)) + _e13_hex(c.standard_normal(2))}
        if np is not None and isinstance(v, np.random.RandomState):
            s = v.get_state()
            h = hashlib.sha256()
            _e13_canon(s, h)
            c = np.random.RandomState()
            c.set_state(s)
            return {"kind": "numpy.RandomState", "sha256": h.hexdigest(),
                    "draws": _e13_hex(c.random_sample(4)) + _e13_hex(c.standard_normal(2))}
        if isinstance(v, random.Random):
            s = v.getstate()
            c = random.Random()
            c.setstate(s)
            return {"kind": "random.Random", "sha256": _e13_sha(repr(s).encode()),
                    "draws": _e13_hex(c.random() for _ in range(4)) + ["%016x" % c.getrandbits(64)]}
    except Exception as e:  # noqa: BLE001
        return {"kind": type(v).__module__ + "." + type(v).__name__, "error": "%s: %s" % (type(e).__name__, e)}
    return None

def _e13_rng_fingerprint(g, init_cuda):
    """Full-state fingerprint of every RNG stream, taken WITHOUT advancing any
    live stream: every draw below is made from a clone of the captured state.

    `init_cuda` is False on the SOURCE: initializing CUDA there would change
    what the platform capture sees (it captures CUDA RNG only when CUDA is
    initialized), so an uninitialized source is recorded as such. On the
    destination it is True, so a stream that arrived (or did not) can be
    hashed.
    """
    import sys, random, hashlib
    fp = {}
    st = random.getstate()
    r = random.Random()
    r.setstate(st)
    fp["python"] = {"sha256": _e13_sha(repr(st).encode()), "index": st[1][-1],
                    "gauss_next": None if st[2] is None else float(st[2]).hex(),
                    "draws": _e13_hex(r.random() for _ in range(4)) + ["%016x" % r.getrandbits(64)]}
    np = sys.modules.get("numpy")
    if np is None:
        fp["numpy"] = {"loaded": False}
    else:
        name, key, pos, has_gauss, cached = np.random.get_state()
        h = hashlib.sha256()
        h.update(name.encode())
        h.update(np.ascontiguousarray(key, dtype="<u4").tobytes())
        h.update(("|%d|%d|%s" % (int(pos), int(has_gauss), float(cached).hex())).encode())
        rs = np.random.RandomState()
        rs.set_state((name, key, pos, has_gauss, cached))
        fp["numpy"] = {"loaded": True, "sha256": h.hexdigest(), "pos": int(pos), "has_gauss": int(has_gauss),
                       "draws": _e13_hex(rs.random_sample(4)) + _e13_hex(rs.standard_normal(2))}
    torch = sys.modules.get("torch")
    if torch is None:
        fp["torch_cpu"] = {"loaded": False}
        fp["cuda"] = {"available": False, "fingerprinted": False, "reason": "torch not loaded"}
    else:
        st = torch.get_rng_state()
        c = torch.Generator()
        c.set_state(st)
        fp["torch_cpu"] = {"loaded": True, "sha256": _e13_sha(st.numpy().tobytes()), "initial_seed": int(torch.initial_seed()),
                           "draws": _e13_hex(torch.rand(4, generator=c).tolist())}
        cu = {"available": bool(torch.cuda.is_available()), "initialized_before": bool(torch.cuda.is_initialized())}
        if cu["available"]:
            cu["device_count"] = int(torch.cuda.device_count())
        if cu["available"] and (cu["initialized_before"] or init_cuda):
            devs = []
            for i in range(cu["device_count"]):
                s = torch.cuda.get_rng_state(i)
                cg = torch.Generator(device="cuda:%d" % i)
                cg.set_state(s)
                devs.append({"index": i, "name": torch.cuda.get_device_name(i), "sha256": _e13_sha(s.cpu().numpy().tobytes()),
                             "draws": _e13_hex(torch.rand(4, device="cuda:%d" % i, generator=cg).tolist())})
            cu["fingerprinted"], cu["devices"] = True, devs
        else:
            cu["fingerprinted"] = False
            cu["reason"] = ("no CUDA in this process" if not cu["available"] else
                            "CUDA available but not initialized; left uninitialized so the fingerprint "
                            "does not change what the switch captures")
        fp["cuda"] = cu
    # Every generator bound under a public name in the namespace, plus the
    # generator inside every DataLoader (the data cursor's stream). Names
    # beginning with "_" are skipped: the switch does not carry them by design.
    named = {}
    dl = getattr(getattr(getattr(torch, "utils", None), "data", None), "DataLoader", None) if torch is not None else None
    for k in sorted(g):
        if k.startswith("_"):
            continue
        v = g[k]
        e = _e13_generator_fp(v)
        if e is not None:
            named[k] = e
        if dl is not None and isinstance(v, dl):
            lg = getattr(v, "generator", None)
            sg = getattr(getattr(v, "sampler", None), "generator", None)
            if lg is not None:
                e = _e13_generator_fp(lg) or {"kind": type(lg).__name__, "error": "unsupported generator type"}
                e["shared_with_sampler"] = sg is lg
                named[k + ".generator"] = e
            if sg is not None and sg is not lg:
                named[k + ".sampler.generator"] = _e13_generator_fp(sg) or {"kind": type(sg).__name__, "error": "unsupported generator type"}
    fp["named"] = named
    return fp

def _e13_fp_same(a, b):
    """Idempotency compares the STREAMS, not the process.

    `cuda.initialized_before` describes the process: on a destination where
    CUDA is available but not initialized (a CPU capsule restored on a GPU
    host: the envelope says torch_cuda_initialized=False and the platform
    restore never touches CUDA), the first destination fingerprint has to
    initialize CUDA to hash its stream, so the second call sees
    initialized_before=True although no stream moved. Every stream field
    (hashes, draws, positions, device lists) is still compared."""
    def view(fp):
        v = dict(fp)
        cu = dict(v.get("cuda") or {})
        cu.pop("initialized_before", None)
        v["cuda"] = cu
        return v
    return view(a) == view(b)

def _e13_fingerprint_twice(g, init_cuda):
    """The RNG fingerprint, taken twice in a row to show it moved no stream,
    with its own wall time and whether it initialized CUDA. On a CPU -> GPU
    hop the destination fingerprint is what first initializes CUDA, so its
    time includes CUDA context creation; recording it separately keeps the
    hop's first-execute time comparable with runs that did not fingerprint."""
    import sys, time
    torch = sys.modules.get("torch")
    def cuda_init():
        return bool(torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized())
    before = cuda_init()
    t0 = time.perf_counter()
    a = _e13_rng_fingerprint(g, init_cuda)
    t1 = time.perf_counter()
    b = _e13_rng_fingerprint(g, init_cuda)
    t2 = time.perf_counter()
    return a, {"rng_fingerprint_idempotent": _e13_fp_same(a, b),
               "rng_fingerprint_s": round(t1 - t0, 4), "rng_fingerprint_again_s": round(t2 - t1, 4),
               "cuda_initialized_by_fingerprint": cuda_init() and not before}

def _e13_determinism():
    # cuDNN algorithm choice is PROCESS state: a capsule does not carry it and
    # a restored kernel starts from torch's defaults. So every program sets it.
    import os, torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {"cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_enabled": bool(torch.backends.cudnn.enabled),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_num_threads": int(torch.get_num_threads())}

def _e13_tf32_ablation(model, loss_fn, x, y, force=False):
    """The fixed-batch loss and the forward-output sum (the quantities the
    loss-continuity check and the forward-output oracle compare), under (a)
    the process's default flags and (b) TF32 disabled, flags restored after.
    Eval-mode, no-grad forwards: no parameter, buffer or RNG changes.

    The ablation keys on where the MODEL computes, not on whether the host
    has CUDA. The TF32 flags govern CUDA matmul and cuDNN only, so a model on
    the CPU computes the same bits under both settings: (b) is (a) by
    definition and is recorded once. This matters on a CPU -> GPU hop: the
    restore preserves placement, so the destination model is still on the
    CPU when verify runs (the next training block moves it), and both ends of
    that hop compute on CPU hosts. Recording a "TF32-off" value there, next to
    the host GPU's name, would attribute CPU-vs-CPU drift to the GPU. `force`
    runs (b) anyway; only the harness self-test uses it, to exercise the flag
    save/restore on a CPU box.
    """
    import torch
    def measure():
        was = model.training
        model.eval()
        try:
            with torch.no_grad():
                out = model(x)
                return {"fixed_batch_loss": float(loss_fn(out, y)), "forward_sum": float(out.sum())}
        finally:
            model.train(was)
    def flags():
        return {"matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "float32_matmul_precision": torch.get_float32_matmul_precision()}
    dev = next(model.parameters()).device
    rec = {"cuda_available": bool(torch.cuda.is_available()), "compute_device": str(dev),
           "compute_on_cuda": dev.type == "cuda", "default_flags": flags()}
    if rec["compute_on_cuda"]:
        # The GPU that did the arithmetic (not merely a GPU on the host).
        rec["device_name"] = torch.cuda.get_device_name(dev)
        rec["capability"] = list(torch.cuda.get_device_capability(dev))
    rec["default"] = measure()
    if rec["compute_on_cuda"] or force:
        saved = rec["default_flags"]
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            rec["tf32_off_flags"] = flags()
            rec["tf32_off"] = measure()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = saved["matmul_allow_tf32"]
            torch.backends.cudnn.allow_tf32 = saved["cudnn_allow_tf32"]
            torch.set_float32_matmul_precision(saved["float32_matmul_precision"])
        rec["flags_restored"] = flags() == saved
    else:
        rec["tf32_off"] = None
        rec["tf32_off_note"] = (("model on CPU although this host has CUDA" if rec["cuda_available"] else "no CUDA in this process")
                                + ": TF32 flags govern CUDA matmul and cuDNN only, so TF32-off equals the default")
    return rec

def _e13_optimizer_digest(opt):
    # sha256 over every state tensor's bytes (params in index order, keys
    # sorted), the non-tensor state values, and every param_group
    # hyperparameter. Device-independent: tensors are copied to CPU first.
    import hashlib, torch
    h = hashlib.sha256()
    def feed(v):
        if isinstance(v, torch.Tensor):
            t = v.detach().to("cpu", copy=True).contiguous()
            h.update(("t:%s:%r:" % (t.dtype, tuple(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"empty")
        elif isinstance(v, (list, tuple)):
            h.update(b"(")
            for x in v:
                feed(x)
            h.update(b")")
        elif isinstance(v, float):
            h.update(b"f:" + float(v).hex().encode())
        else:
            h.update(("%s:%r" % (type(v).__name__, v)).encode())
    sd = opt.state_dict()
    for idx in sorted(sd["state"]):
        h.update(("|state %r|" % (idx,)).encode())
        for k in sorted(sd["state"][idx]):
            h.update(k.encode() + b"=")
            feed(sd["state"][idx][k])
    for gi, grp in enumerate(sd["param_groups"]):
        h.update(("|group %d|" % gi).encode())
        for k in sorted(grp):
            h.update(k.encode() + b"=")
            feed(grp[k])
    steps = sorted({float(s["step"]) for s in sd["state"].values() if "step" in s})
    return h.hexdigest()[:32], steps

def _e13_generator_digest(gen):
    return None if gen is None else _e13_sha(gen.get_state().cpu().numpy().tobytes())[:32]
'''


def _fixture_loader_src() -> str:
    """Load the fixture's functions as a module (`experiments.fixture`) rather
    than calling the copies bound in `__main__`: a switch that filters
    underscore names (as `LocalApi` does) drops `_digest_params`, and the
    harness must not depend on what the system under test chose to carry."""
    from handoff.controller import FIXTURE_SRC
    return f'''
def _e13_load_fixture():
    import sys, types
    if "experiments" not in sys.modules:
        p = types.ModuleType("experiments"); p.__path__ = []; sys.modules["experiments"] = p
    m = types.ModuleType("experiments.fixture"); m.__file__ = "experiments.fixture.py"
    sys.modules["experiments.fixture"] = m
    exec(compile({FIXTURE_SRC!r}, "experiments.fixture.py", "exec"), m.__dict__)
    return m
'''


def _wrap(fn: str, body: str) -> str:
    """Run `body` as the function `fn` and delete it, even on failure, so no
    name of the harness is left in `__main__`."""
    return f'''
def {fn}():
{_indent(body)}
try:
    {fn}()
finally:
    del {fn}
'''


# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------

def seed_program(device: str, seed: int) -> str:
    """Build the fixture in the kernel's REAL `__main__`, not as a module, so
    `dill` records its classes by value. See `handoff.controller.seed_program`.
    The fixture definitions stay in `__main__` (that is the workload); the
    build itself runs in a deleted function."""
    from handoff.controller import FIXTURE_SRC
    body = f'''
import sys, json
{_HELPERS_SRC}
flags = _e13_determinism()
import torch
g = sys.modules["__main__"].__dict__
spec = FixtureSpec(seed={seed}, workload="resnet18", device={device!r},
                   workspace_corpus_files=4, workspace_corpus_bytes=16 * 1024)
ns = build_transported_namespace(spec)
g.update(ns)
# chain-specific state
g["tied_a"] = g["model"].fc.weight
g["tied_b"] = g["tied_a"]                       # identity aliasing: two names, one object
g["view_base"] = torch.arange(16, dtype=torch.float32, device={device!r})
g["view_part"] = g["view_base"][4:8]             # storage aliasing: two objects, one buffer
g["loss_trace"] = []
g["hop_log"] = []
print({MARK!r} + json.dumps({{"names": len(ns), "device": str(next(g["model"].parameters()).device),
                             "data_source": g.get("data_source"), "workload_kind": g.get("workload_kind"),
                             "flags": flags}}))
'''
    return f"exec(compile({FIXTURE_SRC!r}, 'clusy_fixture.py', 'exec'), globals())\n" + _wrap("__clusy_e13_seed", body)


def train_program(steps: int) -> str:
    """Train `steps` steps, then record the pre-hop expectation, the per-block
    fingerprints and, LAST, the RNG fingerprint."""
    body = f'''
import sys, json
{_HELPERS_SRC}
{_fixture_loader_src()}
flags = _e13_determinism()
import torch
fx = _e13_load_fixture()
g = sys.modules["__main__"].__dict__
model, opt, sched, loss_fn = g["model"], g["optimizer"], g["scheduler"], g["loss_fn"]
# What a user does after a promotion: use the accelerator they asked for.
# Placement is preserved by the switch; relocation is the program's decision.
dev0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(dev0)
for st in opt.state.values():
    for k, v in st.items():
        if isinstance(v, torch.Tensor): st[k] = v.to(dev0)
g["train_x"], g["train_y"] = g["train_x"].to(dev0), g["train_y"].to(dev0)
g["view_base"] = g["view_base"].to(dev0); g["view_part"] = g["view_base"][4:8]
# A fresh iterator per block, as before. It is a LOCAL now: the old program
# bound it in __main__ as `it`, which overwrote the fixture's own `it`
# (the "iterator over a container" construct) with a DataLoader iterator.
loader = g["loader"]; it = iter(loader)
model.train()
losses = []
for _ in range({steps}):
    try:
        xb, yb = next(it)
    except StopIteration:
        it = iter(loader); xb, yb = next(it)
    dev = next(model.parameters()).device
    xb, yb = xb.to(dev), yb.to(dev)
    opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
    losses.append(float(loss.detach()))   # same value; float() on a grad tensor warns into __main__.__warningregistry__
    g["loss_trace"].append(losses[-1])
# StepLR is a PER-EPOCH schedule, so it is stepped once per block, not once
# per batch. Stepping it per batch decayed lr by 0.5 every two steps: by hop 3
# it was ~1e-17, a real optimizer step no longer moved a float32 parameter,
# and the continuation oracle correctly reported that nothing moved.
sched.step()
# Read the step count back OUT of the optimizer rather than keeping a
# parallel counter: the continuation oracle takes a real training step of its
# own on every hop, which advances opt.state[p]["step"] but not a counter the
# harness maintains.
first = next(iter(opt.state)) if opt.state else None
g["step_count"] = int(opt.state[first]["step"]) if first is not None else g.get("step_count", 0)
model.eval()
# The fixed-batch loss and forward sum, under default flags and TF32 off.
tf32 = _e13_tf32_ablation(model, loss_fn, g["train_x"], g["train_y"])
exp = fx.compute_expectation(g).to_dict()
exp["param_digest"] = fx._digest_params(model)   # re-digest after the probe forward (see driver.py)
opt_digest, opt_steps = _e13_optimizer_digest(opt)
block = {{"param_digest": exp["param_digest"], "optimizer_digest": opt_digest, "optimizer_steps": opt_steps,
          "scheduler": {{"last_epoch": int(sched.last_epoch), "lr": [float(x) for x in sched.get_last_lr()]}},
          "loader_generator_sha256": _e13_generator_digest(getattr(loader, "generator", None)),
          "step_count": g["step_count"], "losses": losses}}
# LAST: the source RNG fingerprint, twice, to show it did not move a stream.
rng, fpm = _e13_fingerprint_twice(g, False)
print({MARK!r} + json.dumps({{
    "step_count": g["step_count"], "fixed_batch_loss": tf32["default"]["fixed_batch_loss"],
    "last_losses": losses, "lr": sched.get_last_lr()[0], "device": str(next(model.parameters()).device),
    "data_source": g.get("data_source"), "flags": flags, "tf32": tf32,
    "forward_sum_matches_expectation": tf32["default"]["forward_sum"] == exp["forward_output"],
    "expectation": exp, "block": block, "rng": rng, **fpm}}))
'''
    return _wrap("__clusy_e13_train", body)


def _carried(pre: dict) -> dict:
    """What the verify program needs from the source, carried by the HARNESS.
    JSON floats round-trip exactly, so nothing is lost in transit."""
    t = pre["tf32"]
    return {"expectation": pre["expectation"], "step_count": pre["step_count"],
            "tf32": {"default": t["default"], "tf32_off": t.get("tf32_off")}}


def verify_program(source_device: str, dest_device: str, same_hardware: bool, rtol: float, pre: dict) -> str:
    """Score the hop, leaving NOTHING of the harness in `__main__`.

    This matters because the next hop dumps `__main__` with the platform
    checkpoint. The first version bound `storage_shared` and the oracle module
    at top level; their `__module__` is a synthetic `capsule.*` /
    `experiments.*` package that exists only in this kernel, so `dill` recorded
    them by reference and hop 2's restore died with "missing module 'capsule'".
    Everything therefore runs inside a function and is discarded on return;
    only a compact dict of results is appended to the namespace's `hop_log`.

    Order is load-bearing: the RNG fingerprint is taken FIRST (before the
    fixed-batch loss, the prelude, or the oracles), then the fixed-batch
    loss and forward sum, then the oracles. The `continuation` oracle takes a
    real optimizer step, so anything measured after it measures that step.
    """
    carried = json.dumps(_carried(pre))
    body = f'''
import sys, json
{_HELPERS_SRC}
g = sys.modules["__main__"].__dict__
# FIRST: the destination RNG fingerprint, twice.
rng, fpm = _e13_fingerprint_twice(g, True)
flags = _e13_determinism()
{_prelude()}
import torch
from capsule.storage_sharing import storage_shared
pre = json.loads({carried!r})
model, loss_fn = g["model"], g["loss_fn"]
model.eval()
tf32 = _e13_tf32_ablation(model, loss_fn, g["train_x"], g["train_y"])
def rel(a, b):
    return abs(b - a) / max(abs(a), 1e-12)
pd, qd = pre["tf32"]["default"], tf32["default"]
po = pre["tf32"]["tf32_off"] or pd
qo = tf32["tf32_off"] or qd
loss_rel = rel(pd["fixed_batch_loss"], qd["fixed_batch_loss"])
rows = [r.to_dict() for r in _oracles.verify(g, pre["expectation"], destination_device={dest_device!r},
                                              source_device={source_device!r}, same_hardware={same_hardware!r})]
checks = {{
    "oracles_passed": sum(1 for r in rows if r["ok"]), "oracles_total": len(rows),
    "oracles_failed": [r["name"] for r in rows if not r["ok"]], "oracle_rows": rows,
    "tied_identity": g["tied_a"] is g["tied_b"],
    "view_shares_storage": storage_shared(g["view_base"], g["view_part"]),
    "loss_continuity_rel": loss_rel,
    "loss_continuity_rel_tf32_off": rel(po["fixed_batch_loss"], qo["fixed_batch_loss"]),
    "forward_sum_rel": rel(pd["forward_sum"], qd["forward_sum"]),
    "forward_sum_rel_tf32_off": rel(po["forward_sum"], qo["forward_sum"]),
    "loss_continuous": loss_rel <= {rtol!r}, "rtol": {rtol!r},
    "step_count": g["step_count"], "step_count_matches": g["step_count"] == pre["step_count"],
    "device": str(next(model.parameters()).device), "data_source": g.get("data_source"),
    "flags": flags, "tf32": tf32, "rng": rng, **fpm,
}}
g["hop_log"].append({{k: checks[k] for k in ("oracles_passed", "oracles_total", "tied_identity", "view_shares_storage",
                                             "loss_continuity_rel", "step_count", "device")}})
print({MARK!r} + json.dumps(checks, default=str))
'''
    return _wrap("__clusy_e13_verify", body)


def final_program() -> str:
    body = f'''
import sys, json
{_HELPERS_SRC}
{_fixture_loader_src()}
fx = _e13_load_fixture()
g = sys.modules["__main__"].__dict__
opt = g["optimizer"]
first = next(iter(opt.state)) if opt.state else None
opt_digest, opt_steps = _e13_optimizer_digest(opt)
print({MARK!r} + json.dumps({{
    "loss_trace": g["loss_trace"], "step_count": g["step_count"],
    "optimizer_step": int(opt.state[first]["step"]) if first is not None else None, "optimizer_steps": opt_steps,
    "param_digest": fx._digest_params(g["model"]), "optimizer_digest": opt_digest,
    "loader_generator_sha256": _e13_generator_digest(getattr(g["loader"], "generator", None)),
    "hop_log_len": len(g.get("hop_log") or []), "data_source": g.get("data_source"),
    "rng": _e13_rng_fingerprint(g, False)}}))
'''
    return _wrap("__clusy_e13_final", body)


# ---------------------------------------------------------------------------
# Chains and controls
# ---------------------------------------------------------------------------

def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _delete(api: Api, pid: str):
    """Release a project without letting a teardown error replace the
    record of what the chain measured."""
    try:
        return api.delete_project(pid)
    except Exception as exc:  # noqa: BLE001
        return f"delete failed: {type(exc).__name__}: {exc}"[:300]


def _create(api: Api, name: str, profile: str, record: dict, log) -> str | None:
    """Create the project, or record why it could not be created. A
    provisioning failure is a result of this run, not a reason to skip the
    remaining chains: `main` writes the record and moves on."""
    try:
        pid = api.create_project(f"clusy-exp-e13-{name}-{uuid.uuid4().hex[:6]}", profile)
    except Exception as exc:  # noqa: BLE001
        record["project"] = None
        record["error"] = f"create_project failed: {type(exc).__name__}: {exc}"[:2000]
        record["delete_status"] = "not created"
        record["finished_at_utc"] = _utc()
        log(f"[{name}] ERROR {record['error'][:300]}")
        return None
    record["project"] = pid
    return pid


def _excl_fingerprint(elapsed_s: float, post: dict) -> float | None:
    """Wall time of a verify witness minus the destination RNG fingerprint
    (both calls). On a CPU -> GPU hop that fingerprint is what initializes
    CUDA, a cost that without it would land in the next training block."""
    fs = [post.get("rng_fingerprint_s"), post.get("rng_fingerprint_again_s")]
    if any(f is None for f in fs):
        return None
    return round(elapsed_s - sum(fs), 1)


def _verdict_line(v: dict) -> str:
    named = ",".join(f"{k}={x['verdict']}" for k, x in v["named"].items())
    return (f"rng py={v['python']['verdict']} np={v['numpy']['verdict']} torch={v['torch_cpu']['verdict']} "
            f"cuda={v['cuda']['verdict']} [{named}]")


def run_chain(api: Api, name: str, hops: list, steps: int, seed: int, memory_hops: list | None, log,
              on_hop=None, route_planned: list | None = None) -> dict:
    """Seed, then per hop: train block, PATCH, verify block. The control is
    this sequence minus the PATCH (`run_control`)."""
    first = hops[0]
    record = {"chain": name, "kind": "chain", "route": hops, "route_planned": route_planned or hops,
              "memory_route": memory_hops, "hops": [], "steps_per_hop": steps, "seed": seed, "started_at_utc": _utc()}
    pid = _create(api, name, first, record, log)
    if pid is None:
        return record
    log(f"[{name}] project {pid} on {first}")
    try:
        seed_out = api.witness(pid, seed_program(DEVICE(first), seed), timeout_ms=1_800_000)
        record["seed_output"] = seed_out
        record["data_source"] = seed_out.get("data_source")
        cur = first
        for i, nxt in enumerate(hops[1:], start=1):
            pre = api.witness(pid, train_program(steps), timeout_ms=1_800_000)
            t0 = time.perf_counter()
            if memory_hops is not None:
                st, payload = api._req("PATCH", f"/projects/{pid}", {"runtimeMemoryMiB": memory_hops[i]}, timeout=1500)
            else:
                st, payload = api.patch_profile(pid, nxt)
            switch_s = time.perf_counter() - t0
            retried = False
            if st == 409 and (payload.get("error") or {}).get("retryable"):
                # The capture barrier refused the switch and kept the source,
                # which is the safety property working. The response says
                # retryable, so the chain retries ONCE and records that it did;
                # a second refusal ends the chain as a result, not an error.
                log(f"  hop {i} {cur}->{nxt}: 409 {(payload.get('error') or {}).get('code')}, source intact; retrying once")
                time.sleep(10); retried = True
                t0 = time.perf_counter()
                if memory_hops is not None:
                    st, payload = api._req("PATCH", f"/projects/{pid}", {"runtimeMemoryMiB": memory_hops[i]}, timeout=1500)
                else:
                    st, payload = api.patch_profile(pid, nxt)
                switch_s = time.perf_counter() - t0
            if st != 200:
                log(f"  hop {i} {cur}->{nxt}: PATCH {st} {json.dumps(payload)[:200]}")
                record["hops"].append({"hop": i, "from": cur, "to": nxt, "patch_status": st, "retried": retried,
                                       "pre": pre, "error": payload}); break
            if payload.get("switchPhases") is None:
                log(f"  hop {i} {cur}->{nxt}: PATCH 200 but no switchPhases: the API did not treat this as a shape change (no switch happened)")
                record["hops"].append({"hop": i, "from": cur, "to": nxt, "patch_status": st, "real_switch": False, "pre": pre}); break
            same = DEVICE(cur) == DEVICE(nxt) and (memory_hops is not None or cur == nxt)
            t1 = time.perf_counter()
            dest = DEVICE(nxt) if DEVICE(nxt) == "cpu" or DEVICE(cur) == "cuda" else DEVICE(cur)
            post = api.witness(pid, verify_program(DEVICE(cur), dest, same_hardware=same,
                                                   rtol=(1e-4 if DEVICE(cur) == DEVICE(nxt) else 1e-3), pre=pre),
                               timeout_ms=1_500_000)
            first_exec_s = time.perf_counter() - t1
            verdicts = rng_verdicts(pre.get("rng"), post.get("rng"))
            # first_execute_s is the whole verify witness, as in earlier runs;
            # the _excl_fingerprint figure removes the destination fingerprint
            # (and the CUDA initialization it may cause) so hop types and runs
            # can be compared on the same footing.
            hop = {"hop": i, "from": cur, "to": nxt, "patch_status": st, "real_switch": True, "retried": retried,
                   "switch_s": round(switch_s, 1), "first_execute_s": round(first_exec_s, 1),
                   "first_execute_excl_fingerprint_s": _excl_fingerprint(first_exec_s, post),
                   "cuda_initialized_by_fingerprint": post.get("cuda_initialized_by_fingerprint"),
                   "pre": pre, "post": post, "rng_verdicts": verdicts,
                   "switch_phases": payload.get("switchPhases"), "switch_timeline": payload.get("switchTimeline")}
            if memory_hops is not None:
                hop["memory_from"], hop["memory_to"] = memory_hops[i - 1], memory_hops[i]
            record["hops"].append(hop)
            cuda_note = ", CUDA init by fingerprint" if post.get("cuda_initialized_by_fingerprint") else ""
            log(f"  hop {i:>2} {cur:>11} -> {nxt:<11} switch {switch_s:6.1f}s first-exec {first_exec_s:6.1f}s "
                f"(excl. fingerprint {hop['first_execute_excl_fingerprint_s']}s{cuda_note}) | "
                f"oracles {post['oracles_passed']}/{post['oracles_total']} tied={post['tied_identity']} "
                f"view={post['view_shares_storage']} loss_rel={post['loss_continuity_rel']:.1e} "
                f"(tf32 off {post['loss_continuity_rel_tf32_off']:.1e}) steps={post['step_count']} | {_verdict_line(verdicts)}")
            if on_hop is not None:
                on_hop(pid)
            cur = nxt
        record["final"] = api.witness(pid, final_program())
        record["loss_trace"] = record["final"]["loss_trace"]
        record["final_optimizer_step"] = record["final"]["optimizer_step"]
    except Exception as exc:  # noqa: BLE001
        # A failed chain is still a result: keep what ran, name the failure,
        # and release the project instead of leaving a sandbox running.
        record["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        log(f"  [{name}] ERROR {record['error'][:300]}")
    finally:
        record["delete_status"] = _delete(api, pid)
        record["finished_at_utc"] = _utc()
    return record


def run_control(api: Api, profile: str, blocks: int, steps: int, seed: int, name: str, log=print) -> dict:
    """The chain minus the switches: seed, then per block exactly the
    chain's per-hop sequence (train block, then verify block with
    source = destination = this device and `same_hardware=True`) with no PATCH
    between them. Same train block, same verify block (so the continuation
    oracle's extra optimizer step happens here too), same data-cursor
    progression and the same once-per-block LR schedule."""
    dev = DEVICE(profile)
    record = {"chain": name, "kind": "control", "profile": profile, "blocks": [], "n_blocks": blocks,
              "steps_per_hop": steps, "seed": seed, "started_at_utc": _utc()}
    pid = _create(api, name, profile, record, log)
    if pid is None:
        return record
    log(f"[{name}] project {pid} on {profile}, {blocks} blocks x {steps} steps, no switches")
    try:
        seed_out = api.witness(pid, seed_program(dev, seed), timeout_ms=1_800_000)
        record["seed_output"] = seed_out
        record["data_source"] = seed_out.get("data_source")
        for b in range(1, blocks + 1):
            pre = api.witness(pid, train_program(steps), timeout_ms=1_800_000)
            t1 = time.perf_counter()
            post = api.witness(pid, verify_program(dev, dev, same_hardware=True, rtol=1e-4, pre=pre), timeout_ms=1_500_000)
            verify_s = time.perf_counter() - t1
            verdicts = rng_verdicts(pre.get("rng"), post.get("rng"))
            record["blocks"].append({"block": b, "profile": profile, "verify_s": round(verify_s, 1),
                                     "verify_excl_fingerprint_s": _excl_fingerprint(verify_s, post),
                                     "cuda_initialized_by_fingerprint": post.get("cuda_initialized_by_fingerprint"),
                                     "pre": pre, "post": post, "rng_verdicts": verdicts})
            log(f"  block {b:>2} {profile:>11} | oracles {post['oracles_passed']}/{post['oracles_total']} "
                f"loss_rel={post['loss_continuity_rel']:.1e} steps={post['step_count']} | {_verdict_line(verdicts)}")
        record["final"] = api.witness(pid, final_program())
        record["loss_trace"] = record["final"]["loss_trace"]
        record["final_optimizer_step"] = record["final"]["optimizer_step"]
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        log(f"  [{name}] ERROR {record['error'][:300]}")
    finally:
        record["delete_status"] = _delete(api, pid)
        record["finished_at_utc"] = _utc()
    return record


# ---------------------------------------------------------------------------
# Local dry-run adapter and checks
# ---------------------------------------------------------------------------

# The platform capture puts back submodule attributes a package deleted
# (torch.optim does `del adamw` after `from .adamw import AdamW`) for the
# duration of the dump, so dill records `torch.optim.adamw.AdamW` by reference.
# LocalApi's simulated capture does not, so dill pickles AdamW BY VALUE and the
# restored optimizer is an instance of a copied class: the `optimizer` oracle's
# isinstance check then fails after every local switch, a gap in the
# simulation rather than in the system or the harness. The adapter below runs
# the same step, `capsule.optimizer_reattach.reattach`, before each simulated
# capture. The source kernel is discarded by the switch, so nothing needs to be
# detached afterwards.
def _local_reattach_program() -> str:
    import inspect

    from capsule.optimizer_reattach import reattach
    return _wrap("__clusy_e13_reattach", "import sys, types\n" + inspect.getsource(reattach) + "reattach()\n")


def _local_api(**kw):
    from localapi import LocalApi

    class E13LocalApi(LocalApi):
        """LocalApi plus the platform capture's submodule reattach before capture."""

        reattach_program = _local_reattach_program()

        def _switch(self, pid, *, profile=None, memory=None):
            p = self.projects.get(pid)
            if p is not None:
                res = p["kernel"].run(self.reattach_program)
                if res.get("error"):
                    return 409, {"error": {"code": "LOCAL_REATTACH_FAILED", "retryable": False,
                                           "detail": res["error"][-400:]}}
            return super()._switch(pid, profile=profile, memory=memory)

    return E13LocalApi(**kw)


#: What a correct harness must see on chain hops, per LocalApi switch mode.
LOCAL_EXPECT = {
    "none": {"python": "equal", "numpy": "equal", "torch_cpu": "equal"},
    "rng_restore": {"python": "state_differs", "numpy": "state_differs", "torch_cpu": "state_differs"},
    "rng_perturb": {"python": "equal", "numpy": "equal", "torch_cpu": "state_differs"},
}


def local_checks(chains: dict[str, dict], expect: str) -> tuple[bool, list[str]]:
    """PASS/FAIL checks for a `--local` run. Every chain and control ran on
    the local CPU, so every comparison is expected to be bitwise equal and
    every recomputed value to be exactly continuous."""
    lines, ok_all = [], True

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal ok_all
        ok_all &= bool(ok)
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

    want = LOCAL_EXPECT[expect]
    chain_recs = {k: v for k, v in chains.items() if v.get("kind") == "chain"}
    ctrl_recs = {k: v for k, v in chains.items() if v.get("kind") == "control"}
    for name, rec in chains.items():
        check(f"{name}: completed without error", not rec.get("error"), str(rec.get("error") or "")[:200])

    hops = [(n, h) for n, r in chain_recs.items() for h in r.get("hops") or [] if h.get("post")]
    blocks = [(n, b) for n, r in ctrl_recs.items() for b in r.get("blocks") or [] if b.get("post")]
    if hops:
        bad = [f"{n} hop {h['hop']} {s}={h['rng_verdicts'][s]['verdict']}" for n, h in hops for s in GLOBAL_STREAMS
               if h["rng_verdicts"][s]["verdict"] != want[s]]
        check(f"chain hops: global RNG verdicts are {want} (expectation '{expect}')", not bad,
              f"{len(hops)} hops" if not bad else "; ".join(bad[:6]))
        bad = [f"{n} hop {h['hop']} {k}={v['verdict']}" for n, h in hops for k, v in h["rng_verdicts"]["named"].items()
               if v["verdict"] != "equal"]
        has_loader = all("loader.generator" in h["rng_verdicts"]["named"] for _, h in hops)
        check("chain hops: namespace generators (loader.generator, np_gen) equal (carried by the capsule, not the envelope)",
              not bad and has_loader, "; ".join(bad[:4]) if bad else f"names {sorted(hops[0][1]['rng_verdicts']['named'])}")
        bad = [f"{n} hop {h['hop']}" for n, h in hops if h["rng_verdicts"]["cuda"]["verdict"] != "not_applicable"]
        check("chain hops: CUDA stream not_applicable (no CUDA locally)", not bad, "; ".join(bad[:4]))
    if blocks:
        bad = [f"{n} block {b['block']} {s}" for n, b in blocks
               for s in (*GLOBAL_STREAMS, *b["rng_verdicts"]["named"])
               if (b["rng_verdicts"].get(s) or b["rng_verdicts"]["named"].get(s))["verdict"] != "equal"]
        check("controls: every stream equal between train end and verify start (no switch in between)", not bad,
              "; ".join(bad[:4]) if bad else f"{len(blocks)} blocks")
    posts = [(n, e) for n, e in hops + blocks]
    bad = [f"{n} {e.get('hop', e.get('block'))}" for n, e in posts
           if not (e["pre"].get("rng_fingerprint_idempotent") and e["post"].get("rng_fingerprint_idempotent"))]
    check("RNG fingerprint idempotent (computed twice in a row, identical) in every train and verify program", not bad,
          "; ".join(bad[:4]) if bad else f"{2 * len(posts)} programs")
    bad = [f"{n} {e.get('hop', e.get('block'))}: {e['post']['oracles_failed']}" for n, e in posts if e["post"]["oracles_failed"]]
    check("oracles all pass on every hop and block", not bad, "; ".join(bad[:4]))
    bad = [f"{n} {e.get('hop', e.get('block'))}" for n, e in posts
           if not (e["post"]["tied_identity"] and e["post"]["step_count_matches"])]
    check("tied weights one object and step count intact on every hop and block", not bad, "; ".join(bad[:4]))
    bad = [f"{n} {e.get('hop', e.get('block'))}: {e['post']['loss_continuity_rel']}" for n, e in posts
           if e["post"]["loss_continuity_rel"] != 0.0 or e["post"]["loss_continuity_rel_tf32_off"] != 0.0
           or e["post"]["forward_sum_rel"] != 0.0]
    check("fixed-batch loss and forward sum exactly continuous (default flags and TF32-off)", not bad, "; ".join(bad[:4]))
    bad = [f"{n} {e.get('hop', e.get('block'))}" for n, e in posts
           if e["pre"]["tf32"].get("tf32_off") is not None or e["post"]["tf32"].get("tf32_off") is not None]
    check("TF32: model on CPU, so TF32-off is recorded once (as the default)", not bad, "; ".join(bad[:4]))
    bad = [f"{n} {e.get('hop', e.get('block'))}" for n, e in posts
           if e["post"]["tf32"].get("compute_on_cuda") is not False or e["pre"]["tf32"].get("compute_on_cuda") is not False]
    check("TF32: compute device recorded at both ends (cpu locally)", not bad, "; ".join(bad[:4]))
    bad = [f"{n} {e.get('hop', e.get('block'))}" for n, e in posts
           if any(not isinstance(x.get("rng_fingerprint_s"), float) or x.get("cuda_initialized_by_fingerprint") is not False
                  for x in (e["pre"], e["post"]))]
    timed = [h.get("first_execute_excl_fingerprint_s") for _, h in hops]
    check("fingerprint timed in every train and verify program, never initializing CUDA locally; "
          "first-exec recorded with and without it", not bad and all(t is not None for t in timed),
          "; ".join(bad[:4]) if bad else f"{len(timed)} hops")
    bad = [f"{n} {e.get('hop', e.get('block'))}" for n, e in posts
           if not (e["pre"]["flags"]["cudnn_deterministic"] and not e["pre"]["flags"]["cudnn_benchmark"]
                   and e["post"]["flags"]["cudnn_deterministic"] and not e["post"]["flags"]["cudnn_benchmark"])]
    check("determinism flags set and recorded in every train and verify program", not bad, "; ".join(bad[:4]))
    bad = [f"{n} {e.get('hop', e.get('block'))}" for n, e in posts if not e["pre"].get("forward_sum_matches_expectation")]
    check("TF32 default-flag forward sum equals the oracle expectation's forward output", not bad, "; ".join(bad[:4]))

    s = summarise(chains, cohort="local")
    # The exact test includes the full RNG stream content, so a switch that
    # loses a global stream must make every comparison that involves a chain
    # UNEQUAL, first at block 1 in the verify-start fingerprint, naming exactly
    # the streams the break touched; control vs control stays equal. With an
    # intact switch every comparison is equal.
    broken = [st for st in GLOBAL_STREAMS if want[st] != "equal"]
    for key, c in s["comparisons"].items():
        fd = c["first_divergence"] or {}
        has_chain = any(chains.get(c[side], {}).get("kind") == "chain" for side in ("a", "b"))
        losses_eq = all(r["losses_equal"] for r in c["per_block"])
        detail = (f"blocks {c['blocks_equal']}/{c['blocks_compared']}, final {c['final'].get('bitwise_equal')}"
                  + (f", first divergence {describe_divergence(fd)}" if fd else "") + f"; all losses equal: {losses_eq}")
        if not broken or not has_chain:
            check(f"bitwise {key}: equal ({c['role'].split(':')[0]})", c["bitwise_equal"], detail)
        else:
            check(f"bitwise {key}: UNEQUAL, first in rng_verify_start at block 1, streams {broken}",
                  not c["bitwise_equal"] and fd.get("block") == 1 and fd.get("fields") == ["rng_verify_start"]
                  and (fd.get("rng_streams") or {}).get("rng_verify_start") == broken, detail)
    for key, v in s["step_counts"].items():
        check(f"final optimizer step equal: {key}", v["equal"], json.dumps(v))

    # Negative controls for the comparison itself: a one-ulp change in one
    # loss, and a changed parameter digest, must each be reported with the
    # exact place they occur. A comparison that cannot fail proves nothing.
    base = chains.get("same") or next(iter(chain_recs.values()), None)
    if base is not None and len(base.get("hops") or []) >= 2:
        mut = copy.deepcopy(base)
        x = mut["hops"][1]["pre"]["block"]["losses"][0]
        mut["hops"][1]["pre"]["block"]["losses"][0] = math.nextafter(x, math.inf)
        c = compare_runs(base, mut)
        fd = c["first_divergence"] or {}
        check("negative control: a one-ulp change in one loss is detected at block 2 step 1",
              not c["bitwise_equal"] and fd.get("block") == 2 and fd.get("step_in_block") == 1, json.dumps(fd))
        mut = copy.deepcopy(base)
        mut["hops"][-1]["pre"]["block"]["param_digest"] = "0" * 32
        c = compare_runs(base, mut)
        fd = c["first_divergence"] or {}
        check("negative control: a changed per-block param digest is detected with losses still equal",
              not c["bitwise_equal"] and fd.get("fields") == ["param_digest"], json.dumps(fd))
    return ok_all, lines


# ---------------------------------------------------------------------------
# In-process self-test of the shipped helpers (no API, no kernels)
# ---------------------------------------------------------------------------

def selftest() -> int:
    import pickle
    import random
    import numpy as np
    import torch
    ns: dict = {}
    exec(compile(_HELPERS_SRC, "e13_helpers", "exec"), ns)
    fp = ns["_e13_rng_fingerprint"]
    lines, ok_all = [], True

    def check(name, ok, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

    random.seed(1); np.random.seed(2); torch.manual_seed(3)
    # One draw each, so every stream is mid-block: a freshly seeded Mersenne
    # Twister regenerates its whole state on the first draw, which would let
    # even a prefix comparison see that one.
    random.random(); np.random.random_sample(); torch.rand(1)
    ds = torch.utils.data.TensorDataset(torch.arange(64))
    gen = torch.Generator(); gen.manual_seed(99)
    g = {"loader": torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, generator=gen),
         "np_gen": np.random.default_rng(5), "rs": np.random.RandomState(6), "rr": random.Random(7),
         "_hidden": random.Random(8)}
    live = lambda: (random.getstate(), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone(),  # noqa: E731
                    gen.get_state().clone(), pickle.dumps(g["np_gen"].bit_generator.state),
                    pickle.dumps(g["rs"].get_state()), g["rr"].getstate())
    before = live()
    a, b = fp(g, False), fp(g, False)
    after = live()
    check("fingerprint twice in a row is identical", a == b)
    check("fingerprint leaves every live stream where it was",
          before[0] == after[0] and before[1] == after[1] and torch.equal(before[2], after[2])
          and torch.equal(before[3], after[3]) and before[4:] == after[4:])
    check("namespace scan finds loader.generator, np_gen, rs, rr and skips _hidden",
          sorted(a["named"]) == ["loader.generator", "np_gen", "rr", "rs"], str(sorted(a["named"])))

    # Sensitivity: one draw on ONE stream flips exactly that stream, and the
    # old prefix comparison would not have seen it.
    old_prefix = lambda: (random.getstate()[1][:5], np.random.get_state()[1][:5].tolist(),  # noqa: E731
                          torch.get_rng_state()[:8].tolist())
    advances = [("python", "python", lambda: random.random()),
                ("numpy", "numpy", lambda: np.random.random_sample()),
                ("numpy (Gaussian cache)", "numpy", lambda: np.random.standard_normal()),
                ("torch_cpu", "torch_cpu", lambda: torch.rand(1)),
                ("loader.generator", "loader.generator", lambda: torch.rand(1, generator=gen)),
                ("np_gen", "np_gen", lambda: g["np_gen"].random()),
                ("rs", "rs", lambda: g["rs"].random_sample()),
                ("rr", "rr", lambda: g["rr"].random())]
    for label, key, adv in advances:
        f0, p0 = fp(g, False), old_prefix()
        adv()
        f1, p1 = fp(g, False), old_prefix()
        v = rng_verdicts(f0, f1)
        flat = {s: v[s]["verdict"] for s in GLOBAL_STREAMS} | {k: x["verdict"] for k, x in v["named"].items()}
        differs = sorted(k for k, x in flat.items() if x != "equal")
        blind = "" if key not in GLOBAL_STREAMS else f"; old prefix check blind: {p0 == p1}"
        check(f"one draw on {label} flips exactly that stream", differs == [key], f"{differs}{blind}")

    # CUDA verdicts, on synthetic fingerprints (no CUDA here).
    src_gpu = {"available": True, "fingerprinted": True, "device_count": 1,
               "devices": [{"index": 0, "name": "T4", "sha256": "aa", "draws": ["0x1p-1"]}]}
    no_cuda = {"available": False, "fingerprinted": False}
    dst_gpu = {"available": True, "fingerprinted": True, "device_count": 1,
               "devices": [{"index": 0, "name": "A100", "sha256": "bb", "draws": ["0x1p-2"]}]}
    v = rng_verdicts({"cuda": src_gpu}, {"cuda": no_cuda})["cuda"]
    check("CUDA -> CPU: not_applicable with the declared drop recorded",
          v["verdict"] == "not_applicable" and v.get("declared_drop") and v.get("source_sha256") == ["aa"], json.dumps(v))
    v = rng_verdicts({"cuda": no_cuda}, {"cuda": dst_gpu})["cuda"]
    check("CPU -> CUDA: not_applicable, destination stream not carried, its hash recorded",
          v["verdict"] == "not_applicable" and v.get("destination_stream_carried") is False
          and v.get("destination_sha256") == ["bb"], json.dumps(v))
    check("CUDA -> CUDA: equal only when state and draws agree",
          rng_verdicts({"cuda": src_gpu}, {"cuda": src_gpu})["cuda"]["verdict"] == "equal"
          and rng_verdicts({"cuda": src_gpu}, {"cuda": dst_gpu})["cuda"]["verdict"] == "state_differs")
    v = rng_verdicts({"cuda": {"available": True, "initialized_before": False, "fingerprinted": False}}, {"cuda": dst_gpu})["cuda"]
    check("uninitialized source CUDA: not_applicable (source had no initialized CUDA state)",
          v["verdict"] == "not_applicable" and "no initialized CUDA state" in v["reason"], v["reason"])

    # The idempotency comparison ignores process descriptors but still sees
    # every stream: one Python draw between two fingerprints must register.
    f0 = fp(g, False)
    random.random()
    check("stream-only idempotency comparison still detects a moved stream", not ns["_e13_fp_same"](f0, fp(g, False)))

    # Mocked CUDA, to run the CUDA branch of the shipped helpers on a CPU box.
    # FakeCuda mirrors the one behaviour that matters: get_rng_state and
    # get_device_name call torch.cuda._lazy_init(), which initializes CUDA.
    import types

    class FakeCuda:
        def __init__(self, init):
            self.init = init
        def is_available(self):
            return True
        def is_initialized(self):
            return self.init
        def device_count(self):
            return 1
        def get_rng_state(self, i=0):
            self.init = True
            return torch.get_rng_state()
        def get_device_name(self, i=None):
            self.init = True
            return "FakeT4"
        def get_device_capability(self, i=None):
            self.init = True
            return (7, 5)

    class GenMeta(type):
        def __instancecheck__(cls, obj):
            return isinstance(obj, torch.Generator)

    class FakeGenerator(metaclass=GenMeta):
        def __new__(cls, device="cpu"):
            return torch.Generator()

    class Proxy(types.ModuleType):
        def __getattr__(self, k):
            return getattr(torch, k)

    def with_fake_cuda(initialized, fn):
        p = Proxy("torch")
        p.cuda, p.Generator = FakeCuda(initialized), FakeGenerator
        p.rand = lambda *a, device=None, generator=None, **kw: torch.rand(*a, generator=generator, **kw)
        real = sys.modules["torch"]
        sys.modules["torch"] = p
        try:
            mns: dict = {}
            exec(compile(_HELPERS_SRC, "e13_helpers_fake_cuda", "exec"), mns)
            return fn(mns, p.cuda)
        finally:
            sys.modules["torch"] = real

    def dest_uninit(mns, cu):
        a, meta = mns["_e13_fingerprint_twice"](g, True)
        b = mns["_e13_rng_fingerprint"](g, True)
        return a, meta, a == b, mns["_e13_fp_same"](a, b), cu.init
    a, meta, raw_eq, same, init_after = with_fake_cuda(False, dest_uninit)
    check("fake CUDA, CPU -> GPU destination (available, uninitialized): fingerprint idempotent",
          meta["rng_fingerprint_idempotent"] and a["cuda"]["fingerprinted"] and a["cuda"]["initialized_before"] is False,
          f"raw dict equality (the check before this fix) would say {raw_eq}; stream-only comparison says {same}")
    check("fake CUDA: the destination fingerprint reports that it initialized CUDA",
          meta["cuda_initialized_by_fingerprint"] is True and init_after is True
          and isinstance(meta["rng_fingerprint_s"], float), json.dumps(meta))
    _, meta, _, _, _ = with_fake_cuda(True, dest_uninit)
    check("fake CUDA already initialized: idempotent, not initialized by the fingerprint",
          meta["rng_fingerprint_idempotent"] and meta["cuda_initialized_by_fingerprint"] is False, json.dumps(meta))
    a, meta = with_fake_cuda(False, lambda mns, cu: mns["_e13_fingerprint_twice"](g, False))
    check("fake CUDA, source side: CUDA left uninitialized and not fingerprinted",
          a["cuda"]["fingerprinted"] is False and meta["cuda_initialized_by_fingerprint"] is False
          and meta["rng_fingerprint_idempotent"], a["cuda"].get("reason", ""))
    model_cpu = torch.nn.Sequential(torch.nn.Linear(8, 4))
    xc, yc = torch.randn(16, 8), torch.randint(0, 4, (16,))
    t, init_after = with_fake_cuda(False, lambda mns, cu: (mns["_e13_tf32_ablation"](model_cpu, torch.nn.CrossEntropyLoss(), xc, yc),
                                                          cu.init))
    check("fake CUDA host, model on CPU: TF32-off recorded once, no GPU named, CUDA not touched",
          t["cuda_available"] and t["compute_on_cuda"] is False and t["tf32_off"] is None and "device_name" not in t
          and init_after is False, t.get("tf32_off_note", ""))

    # TF32 flag save/restore, exercised on CPU with force=True.
    model = torch.nn.Sequential(torch.nn.Linear(8, 4))
    x, y = torch.randn(16, 8), torch.randint(0, 4, (16,))
    flags0 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    t = ns["_e13_tf32_ablation"](model, torch.nn.CrossEntropyLoss(), x, y, force=True)
    flags1 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    check("TF32 ablation restores the process flags", t["flags_restored"] and flags0 == flags1, f"{flags0} -> {flags1}")
    check("TF32 ablation sets both TF32 flags off for (b)",
          t["tf32_off_flags"]["matmul_allow_tf32"] is False and t["tf32_off_flags"]["cudnn_allow_tf32"] is False)
    check("TF32 ablation on CPU: (b) equals (a)", t["tf32_off"] == t["default"])
    t = ns["_e13_tf32_ablation"](model, torch.nn.CrossEntropyLoss(), x, y)
    check("TF32 ablation without CUDA records TF32-off once (None)", t["tf32_off"] is None)

    # Optimizer digest: stable across a pickle round trip, sensitive to a step.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        opt.zero_grad(); torch.nn.CrossEntropyLoss()(model(x), y).backward(); opt.step()
    d0, s0 = ns["_e13_optimizer_digest"](opt)
    model2 = pickle.loads(pickle.dumps(model))
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    opt2.load_state_dict(pickle.loads(pickle.dumps(opt.state_dict())))
    d1, _ = ns["_e13_optimizer_digest"](opt2)
    opt.zero_grad(); torch.nn.CrossEntropyLoss()(model(x), y).backward(); opt.step()
    d2, s2 = ns["_e13_optimizer_digest"](opt)
    check("optimizer digest equal across a state round trip", d0 == d1)
    check("optimizer digest changes after one step", d0 != d2, f"steps {s0} -> {s2}")

    print("E13 helper self-test")
    print("\n".join(lines))
    print("selftest", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="E13: continued training across repeated migrations")
    ap.add_argument("--api-url", default=os.environ.get("CLUSY_API_URL", "http://localhost:8010"))
    ap.add_argument("--steps", type=int, default=None, help="training steps per block (default 30; 3 with --local)")
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--chains", default=DEFAULT_CHAINS)
    ap.add_argument("--same-hops", type=int, default=5, help="hops of the same chain; also the T4 controls' block count")
    ap.add_argument("--cohort", default=None, help="written into every record (default: this run's run_id)")
    ap.add_argument("--out", default=None, help="output directory (default results/e13; a temp dir with --local)")
    ap.add_argument("--local", action="store_true", help="dry run against experiments/localapi.py; every profile becomes cpu")
    ap.add_argument("--local-break", choices=sorted(LOCAL_EXPECT), default="none",
                    help="break the simulated switch: rng_restore (envelope not applied) or rng_perturb (torch CPU advanced once)")
    ap.add_argument("--local-expect", choices=sorted(LOCAL_EXPECT), default=None,
                    help="the RNG expectation the local checks apply (default: the --local-break mode)")
    ap.add_argument("--keep-local-workdir", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="in-process test of the shipped helpers, then exit")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    steps = args.steps if args.steps is not None else (3 if args.local else 30)
    log = lambda *a: print(*a, flush=True)  # noqa: E731

    on_hop = None
    if args.local:
        workdir = Path(tempfile.mkdtemp(prefix="clusy-e13-localapi-"))
        api = _local_api(rng_restore=args.local_break != "rng_restore", rng_perturb=args.local_break == "rng_perturb",
                         workdir=workdir)
        out = Path(args.out) if args.out else Path(tempfile.gettempdir()) / "clusy-e13-local"
        if (ROOT / "results").resolve() in [out.resolve(), *out.resolve().parents]:
            # A dry run is not a result; it must never land beside live records.
            print(f"--local refuses to write under {ROOT / 'results'}; pass a scratch --out", file=sys.stderr)
            shutil.rmtree(workdir, ignore_errors=True)
            return 2
        prof = lambda p: "cpu"  # noqa: E731
        api_url = None

        def on_hop(pid: str) -> None:
            # Each simulated switch leaves a full dill session on disk
            # (~130 MB for ResNet-18 + AdamW). Consumed ones are removed.
            p = api.projects.get(pid)
            for f in (p["cwd"].glob(".switch-*") if p else []):
                f.unlink(missing_ok=True)
    else:
        key = os.environ.get("CLUSY_HARNESS_API_KEY")
        if not key:
            print("export CLUSY_HARNESS_API_KEY", file=sys.stderr); return 1
        api = Api(args.api_url, key)
        out = Path(args.out) if args.out else ROOT / "results" / "e13"
        why = runmeta.shipped_record_conflict(out / "e13_chains.jsonl", args.cohort)
        if why:
            print(why, file=sys.stderr); return 2
        prof = lambda p: p  # noqa: E731
        api_url = args.api_url
    out.mkdir(parents=True, exist_ok=True)

    meta = runmeta.start_run("e13", api_url=api_url, cohort=args.cohort,
                             args={**vars(args), "steps": steps},
                             extra_files=["experiments/e13_analyse.py"] + (["experiments/localapi.py"] if args.local else []))
    if meta.get("cohort") is None:
        meta["cohort"] = meta["run_id"]
    meta["local_simulation"] = bool(args.local)
    if args.local:
        meta["local_break"] = args.local_break
        meta["local_adapter"] = "E13LocalApi: LocalApi plus the platform capture's submodule reattach before each simulated capture"
    log(f"E13 run {meta['run_id']} cohort {meta['cohort']} -> {out}" + (f" (LOCAL, break={args.local_break})" if args.local else ""))

    t4 = prof("gpu_t4")
    records: dict[str, dict] = {}
    try:
        with (out / "e13_chains.jsonl").open("a") as f:
            for chain in [c.strip() for c in args.chains.split(",") if c.strip()]:
                if chain in ("control_t4_a", "control_t4_b"):
                    r = run_control(api, t4, args.same_hops, steps, args.seed, chain, log)
                elif chain == "control_cpu":
                    r = run_control(api, prof("cpu"), len(HETERO) - 1, steps, args.seed, chain, log)
                elif chain == "same":
                    # A same-device test needs a REAL switch whose arithmetic
                    # is unchanged. `cpu` has no RAM tiers on the platform,
                    # so a memory PATCH there is
                    # silently a no-op and no switch happens. gpu_t4 has tiers,
                    # so the chain alternates RAM on one SKU: same device, same
                    # kernels, a genuine destroy and reprovision between hops.
                    hops = [t4] * (args.same_hops + 1)
                    mem = [16384 if i % 2 == 0 else 32768 for i in range(len(hops))]
                    r = run_chain(api, "same", hops, steps, args.seed, mem, log, on_hop=on_hop,
                                  route_planned=["gpu_t4"] * (args.same_hops + 1))
                elif chain == "hetero":
                    r = run_chain(api, "hetero", [prof(p) for p in HETERO], steps, args.seed, None, log,
                                  on_hop=on_hop, route_planned=HETERO)
                else:
                    log(f"unknown chain {chain!r}; skipped")
                    continue
                r["cohort"] = meta["cohort"]
                r["local_simulation"] = bool(args.local)
                if args.local:
                    r["local_break"] = args.local_break
                runmeta.stamp(r, meta)
                records[chain] = r
                f.write(json.dumps(r, default=str) + "\n"); f.flush()
    finally:
        meta = runmeta.finish_run(meta, api_url=api_url)
        meta["chains_written"] = sorted(records)
        runmeta.write_meta(meta, out / "runs_meta.jsonl")
        if args.local:
            api.close()
            if not args.keep_local_workdir:
                shutil.rmtree(workdir, ignore_errors=True)

    # Step-count equality, chain vs its control, recorded in the log for live
    # runs as well (the analysis script records it in the summary).
    s = summarise(records, cohort=meta["cohort"])
    for key, v in s["step_counts"].items():
        log(f"final optimizer step {key}: {v}")
    for key, c in s["comparisons"].items():
        log(f"bitwise {key}: equal={c['bitwise_equal']} ({c['blocks_equal']}/{c['blocks_compared']} blocks) "
            f"first divergence {c['first_divergence']}")
    if not args.local:
        return 0
    expect = args.local_expect or args.local_break
    ok, lines = local_checks(records, expect)
    print(f"E13 local checks (switch break={args.local_break}, expectation={expect})")
    print("\n".join(lines))
    print("local", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
