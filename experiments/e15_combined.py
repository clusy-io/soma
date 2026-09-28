"""E15: one workload through the repaired research controller, CPU -> T4 -> CPU.

WHY THIS EXISTS. The readiness review of 27 September found that the evidence
for the mechanism came in separate pieces: the storage adapter was tested on
its own (E6, local), the transactional controller on CPU-to-CPU E2B hops
(E11, E12), and the heterogeneous path only through the platform's in-place switch
(E13, E14). Nothing ran the repaired controller and the repaired adapter
TOGETHER, across devices, on a state that has every relationship the reviewer
broke. This is that run: one lifecycle, and at every commit boundary the
controller's own equivalence check plus checks the harness makes itself.

THE WORKLOAD is the controller's tinyblock fixture (so its twelve oracles and
their expectation apply) with the model, optimizer and scheduler replaced by
state that carries the reviewer's three counterexamples and the rest of the
combined fixture:

  model          `E15Net`, an MLP with DROPOUT (so training consumes the
                 device's RNG stream), defined in the session's `__main__`
                 (dill carries it by value, as it would a user's class). Its
                 two weights are Parameter VIEWS of one flat base tensor,
                 `e15_flat`: the reviewer's second case.
  optimizer      AdamW with non-empty state, bound to those Parameters
  scheduler      StepLR, stepped once per training block
  loader         the fixture's DataLoader with its own generator (the data
                 cursor), indexing `e15_data_x` / `e15_data_y`
  e15_w_alias    an external alias of `model.fc1.weight`
  e15_tail       a view of the base, bound again as `e15_tail_again`: the
                 reviewer's first case (a repeated reference to a view)
  e15_top        a top-level Parameter over the base: the reviewer's third
                 case (it must stay a Parameter that requires grad)
  e15_arr ...    a NumPy base with a view (also held in a list) and a
                 transposed view
  RNG            python, numpy and torch CPU streams advanced by user code in
                 every block (an input jitter); dropout advances the CUDA
                 stream on the T4
  e15_counter    the user counter the routed writes increment, and
  e15_acks       the list of write sequence numbers, which must equal the
                 workspace log `data/e15_writes.log` line for line
  data/          the fixture's corpus, `data/e15/notes.txt` (appended by the
                 user in every block, so between hops), and the write log

THE LIFECYCLE (every hop through `Controller.migrate`, never the platform's
PATCH):

  setup     cpu source seeded, E15 state built, block 0 trained on CPU
  hop 1     cpu -> gpu_t4, with a writer thread routing counter increments
            and workspace appends through the controller's route for the
            whole migration and a few writes past DONE
  block 1   on the T4: the user moves the base to CUDA and re-points every
            view at it (`set_`, so no object is replaced: the model's
            Parameters are CUDA views of a CUDA base), moves the optimizer
            state, and trains with dropout on CUDA
  hop 2a    gpu_t4 -> cpu with a fault (`bad_capsule`): must ABORT, leave
            authority and the reopened gate on the T4, leave its state
            unchanged (controller fingerprint AND the harness's own digest),
            and take the next routed write
  hop 2b    gpu_t4 -> cpu with the controller KILLED after the `--crash-phase`
            journal write (in process: `ControllerKilled`), routed writes
            refused while it is down, a fresh controller resuming with
            takeover to DONE; the CUDA storages are remapped to cpu on restore
  block 2   on the CPU destination: continue training

CHECKS. At every commit the controller's commit-boundary comparison must be
equal (declared differences only: CUDA RNG `not_carried` on cpu -> T4 and
`declared_drop` on T4 -> cpu, tensor devices on T4 -> cpu). Independently, a
harness probe (`probe_program`) reads the committed state WITHOUT writing it:
optimizer param and state-key identity, the alias, the repeated references,
Parameter classes and requires_grad, every view's storage pointer and offset
against the base, a write through a DEEPCOPIED base seen through the copied
views (torch's deepcopy memoizes storages, so the copy shares only if the
original does), a real optimizer step on a deepcopy that must move the copy's
model, and a digest of the core state in 13 fields (parameters and buffers,
gradients, optimizer, scheduler, the loader's and the NumPy generators, the
three global RNG streams, the E15 tensors, the E15 arrays, the plain values,
the workspace minus the write log) that must be equal before and after each
hop. The counter and the write list are left out of that digest because the
writer changes them during a hop; the write checks cover them instead. The
controller's own fingerprint is taken before and after every probe to show
the probe changed nothing. Every acknowledged routed write must be in the
destination's namespace and log exactly once, every refused one absent. The
probe also reads which gradients share storage with another tensor: the
storage adapter records no `.grad` views, so E15 claims gradients by value
only, and the analysis says so (and FAILS if a gradient that shared storage
before a hop does not after it). The capsule's refreshed validation
expectation is read back from the capsule on the controller's disk and
compared with the probe of the source (parameter digest, step count, forward
reference), so a refresh that silently lost a field is a FAIL of its own row
rather than an unexplained validation failure. `e15_analyse.py` turns the
records into a PASS/FAIL table per commit boundary and check, plus the
trajectory.

A CONTROLLER GAP THIS RUN FOUND, AND THE IMPORT CELL. The capture refreshes
the validation expectation at the cut with the fixture's
`compute_expectation`, which calls the fixture's `_digest_params` (and
`hashlib`) through `__main__`. The capture carries neither: underscore names
and modules are left out of every capsule. So on a runtime the controller
itself restored, the forked refresh raises `NameError: _digest_params`, the
stale expectation is used, and the second hop after user work aborts with
`contract_check_failed` (parameters, forward output, continuation,
scheduler). Where the refresh cannot fork (a live T4 source, CUDA
initialized) it does not raise: it records `param_digest: None` under the
source "cut", and validation fails on the parameters. The fix belongs in the
controller (the refresh should use the fixture module its prelude ships, not
the copy bound in `__main__`).

The default, `--reimport-cell none`, is therefore the controller alone: until
that fix lands, the default run stops at hop 2b, which is the true result.
`--reimport-cell fixture` is a DIAGNOSTIC: it re-runs the fixture's
definitions cell on every restored runtime before user work (as a user
resuming a session re-runs an imports cell), records which names were
missing, and so shows what the rest of the lifecycle does once the helpers
are reachable. Because the harness then did part of the controller's job,
such a run can never be judged OK: the analysis FAILS its "controller ran
unaided" row.

    python experiments/e15_combined.py --cohort e15-rerun --outdir /tmp/soma-rerun/e15   # live
    python experiments/e15_combined.py --local --outdir /tmp/e15                         # dry run

`--local` runs everything against `experiments/localapi.py`: every profile is
the laptop CPU, so the CUDA steps are SKIPPED and recorded as such (the T4
block trains on cpu with a note, and the CUDA checks are SKIP in the
analysis). One CUDA path IS exercised locally: captures taken on the local
stand-in for the T4 have every storage tagged `cuda:0` (a location tag, as
on a real GPU source), so the CPU destination's restore must remap them
(`--no-local-cuda-tags` turns it off). `--sabotage` (local only) breaks one
thing so the checks can be seen to FAIL: `inplace_validation` (validation
takes a real optimizer step on the committed state), `ungated_route` (the
route ignores the admission gate, the first controller's behaviour),
`unshare_views` (a defect copies `model.fc1.weight` off the base after hop
1). The earlier `no_import_cell` sabotage is now the default.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import runmeta  # noqa: E402
from handoff import controller as _ctl  # noqa: E402
from handoff.controller import (  # noqa: E402
    FIXTURE_SRC, MARK, PHASES, AdmissionClosed, Api, Controller, ControllerKilled, CrashPoint,
    Journal, _wrap, compare_fingerprints, fingerprint_program, seed_program,
)
import e15_analyse  # noqa: E402

SCHEMA = "e15/1"
SOURCE_PROFILE, DEST_PROFILE = "cpu", "cpu"
#: The workspace file every routed write appends its sequence number to,
#: relative to the workspace root `data/`.
WRITES_LOG = "e15_writes.log"
#: The view records the capture must make: four torch views of `e15_flat`
#: and two NumPy views of `e15_arr` (the list slot is a second path to one of
#: them, not a record of its own).
EXPECTED_VIEW_RECORDS = 6

# ---------------------------------------------------------------------------
# User code shipped into the kernels
# ---------------------------------------------------------------------------

#: Executed INTO the session's `__main__`, so the class belongs to `__main__`
#: and dill carries it by value (a class of any other module would be an
#: import the destination must satisfy). Defined inside a function, as the
#: fixture defines TinyBlock: `torch` is then a closure cell, not a global of
#: `__main__`, because the capture carries no modules and a method that looked
#: `torch` up in `__main__` would raise on the destination. `forward` uses only
#: submodules for the same reason.
E15_SESSION_SRC = '''
def _e15_define():
    import torch

    class E15Net(torch.nn.Module):
        """An MLP with dropout whose two weights are Parameter views of one base."""

        def __init__(self, base):
            super().__init__()
            self.fc1 = torch.nn.Linear(16, 32)
            self.act = torch.nn.ReLU()
            self.drop = torch.nn.Dropout(0.25)
            self.fc2 = torch.nn.Linear(32, 4)
            self.fc1.weight = torch.nn.Parameter(base[0:512].view(32, 16))
            self.fc2.weight = torch.nn.Parameter(base[512:640].view(4, 32))

        def forward(self, x):
            return self.fc2(self.drop(self.act(self.fc1(x))))

    return E15Net

E15Net = _e15_define()
del _e15_define
'''

#: Helpers every E15 program defines in its PRIVATE dict (see `_wrap`), never
#: in `__main__`. Nothing here writes to the state it reads.
_COMMON_SRC = r'''
import sys, os, json, random, hashlib, copy
import numpy as np
import torch
g = sys.modules["__main__"].__dict__

def _sha(b):
    if isinstance(b, str):
        b = b.encode("utf-8", "surrogatepass")
    return hashlib.sha256(b).hexdigest()

def _tbytes(t):
    # A tensor's element bytes on cpu, whatever its device: device moves
    # compare equal, changed values do not.
    t = t.detach().to("cpu", copy=True).contiguous().reshape(-1)
    return t.view(torch.uint8).numpy().tobytes() if t.numel() else b""

def _cuda_rng():
    # Per-device CUDA RNG digests, or None when CUDA is not initialized.
    # Never initializes CUDA itself.
    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        return None
    return [_sha(torch.cuda.get_rng_state(i).cpu().numpy().tobytes()) for i in range(torch.cuda.device_count())]

def _global_rng():
    name, key, pos, has_gauss, cached = np.random.get_state()
    return {"python": _sha(repr(random.getstate())),
            "numpy": _sha(name.encode() + np.ascontiguousarray(key, dtype="<u4").tobytes()
                          + ("|%d|%d|%s" % (int(pos), int(has_gauss), float(cached).hex())).encode()),
            "torch_cpu": _sha(torch.get_rng_state().numpy().tobytes())}

def _fixed_loss(model, loss_fn, x, y):
    # Eval mode, no grad, on the model's own device: no dropout draw, no write.
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            d = next(model.parameters()).device
            return float(loss_fn(model(x.to(d)), y.to(d)))
    finally:
        model.train(was)

#: (name, element offset into e15_flat) of every torch view of the base.
VIEWS = (("model.fc1.weight", 0), ("model.fc2.weight", 512), ("e15_tail", 680), ("e15_top", 688))

def _view_objs():
    m = g["model"]
    return {"model.fc1.weight": m.fc1.weight, "model.fc2.weight": m.fc2.weight,
            "e15_tail": g["e15_tail"], "e15_top": g["e15_top"]}
'''


def state_program(seed: int) -> str:
    """Replace the fixture's model, optimizer and scheduler with the E15
    state and add the rest of it. Initial values come from a PRIVATE
    generator, so the global streams are advanced only by user training."""
    return _wrap(f"MARK = {MARK!r}\nSEED = {int(seed)!r}\nSESSION_SRC = {E15_SESSION_SRC!r}\n" + _COMMON_SRC + r'''
exec(compile(SESSION_SRC, "<e15-session>", "exec"), g)
gen = torch.Generator().manual_seed(SEED)
flat = torch.randn(704, generator=gen) * 0.2
model = g["E15Net"](flat)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
g.update(model=model, optimizer=opt, scheduler=sched, e15_flat=flat, e15_w_alias=model.fc1.weight,
         e15_tail=flat[680:688], e15_top=torch.nn.Parameter(flat[688:696]))
g["e15_tail_again"] = g["e15_tail"]
arr = np.arange(48, dtype=np.float64)
g.update(e15_arr=arr, e15_arr_view=arr[8:20], e15_arr_T=arr.reshape(6, 8).T)
g["e15_arr_list"] = [g["e15_arr_view"]]
g["e15_data_x"] = torch.randn(128, 16, generator=gen)
g["e15_data_y"] = torch.randn(128, 4, generator=gen)
g.update(e15_counter=0, e15_acks=[], e15_blocks=[], loss_trace=[], step_count=0)
os.makedirs(os.path.join("data", "e15"), exist_ok=True)
with open(os.path.join("data", "e15", "notes.txt"), "w") as f:
    f.write("e15 session notes\n")
print(MARK + json.dumps({"names": sorted(k for k in g if k.startswith("e15_")),
                         "params": [type(p).__name__ for p in model.parameters()],
                         "shares_base": [p.untyped_storage().data_ptr() == flat.untyped_storage().data_ptr()
                                         for p in (model.fc1.weight, model.fc2.weight)]}))
''')


def import_cell_program() -> str:
    """The user's imports-and-definitions cell, re-run on a restored runtime
    by the `--reimport-cell fixture` DIAGNOSTIC only (see the module doc):
    re-executes the fixture source the seed ran, and reports which of the
    names the expectation refresh needs were missing."""
    return _wrap(f"MARK = {MARK!r}\nFIXTURE = {FIXTURE_SRC!r}\n" + r'''
import sys, json
g = sys.modules["__main__"].__dict__
missing = [k for k in ("_digest_params", "hashlib") if k not in g]
exec(compile(FIXTURE, "clusy_fixture.py", "exec"), g)
print(MARK + json.dumps({"missing_before": missing, "present_after": [k for k in ("_digest_params", "hashlib") if k in g]}))
''')


def train_program(steps: int, block: int, want: str, force_move: bool = False) -> str:
    """One block of user work on this runtime: move to the `want` device
    (keeping every relationship), train `steps` steps, step the scheduler,
    write through the NumPy base and append to the user's notes file.

    `force_move` runs the move even when nothing has to change device, onto a
    fresh buffer on the same device. A dry run has no CUDA, so without it the
    T4 block's move (new base, `model.to`, every view re-pointed with `set_`,
    optimizer state moved) would never execute before a live run; with it, the
    same code runs with `dev = cpu` and the next hop must carry the re-homed
    base."""
    return _wrap(f"MARK = {MARK!r}\nSTEPS = {int(steps)!r}\nBLOCK = {int(block)!r}\nWANT = {want!r}\n"
                 f"FORCE_MOVE = {bool(force_move)!r}\n" + _COMMON_SRC + r'''
m, opt, sched, loss_fn, loader = g["model"], g["optimizer"], g["scheduler"], g["loss_fn"], g["loader"]
notes = []
cuda_ok = bool(torch.cuda.is_available())
dev = torch.device("cuda") if (WANT == "cuda" and cuda_ok) else torch.device("cpu")
if WANT == "cuda" and not cuda_ok:
    notes.append("CUDA requested but this kernel has no CUDA: the block trains on cpu, every CUDA step skipped")
rng_start = _global_rng()
start = {"step_count": int(g["step_count"]), "device": str(next(m.parameters()).device),
         "fixed_batch_loss": _fixed_loss(m, loss_fn, g["train_x"], g["train_y"]),
         "lr": float(opt.param_groups[0]["lr"]), "cuda_rng": _cuda_rng()}

# The user's move to the device, keeping every relationship. `model.to` gives
# each Parameter new storage (Parameter identity kept, sharing lost), so the
# base goes over separately and every view is re-pointed at it with `set_`:
# no object is replaced, so the optimizer, the alias and the repeated
# reference still reach them. `step` stays on cpu, as torch's own
# optimizer state loading places it.
flat = g["e15_flat"]
objs = _view_objs()
move = {"needed": bool(flat.device != dev or any(p.device != dev for p in m.parameters())),
        "forced": bool(FORCE_MOVE)}
if FORCE_MOVE:
    notes.append("the move ran anyway (forced), re-homing the base onto a fresh %s buffer" % dev)
if move["needed"] or FORCE_MOVE:
    old_ptr = flat.untyped_storage().data_ptr()
    geo = {k: (int(t.storage_offset()), tuple(t.shape), tuple(t.stride())) for k, t in objs.items()}
    opt.zero_grad(set_to_none=True)
    new = flat.detach().to(dev, copy=True)
    m.to(dev)
    with torch.no_grad():
        for k, t in objs.items():
            if t.device != dev:
                t.data = t.data.to(dev)
            off, shape, stride = geo[k]
            t.set_(new.untyped_storage(), off, shape, stride)
    g["e15_flat"] = new
    for st in opt.state.values():
        for sk, sv in list(st.items()):
            if isinstance(sv, torch.Tensor) and sk != "step":
                st[sk] = sv.to(dev)
    g["train_x"], g["train_y"] = g["train_x"].to(dev), g["train_y"].to(dev)
    move["rehomed"] = g["e15_flat"].untyped_storage().data_ptr() != old_ptr
flat = g["e15_flat"]
fptr = flat.untyped_storage().data_ptr()
move["views"] = {k: {"device": str(t.device), "shares_base": t.untyped_storage().data_ptr() == fptr,
                     "offset": int(t.storage_offset()), "class": type(t).__name__, "requires_grad": bool(t.requires_grad)}
                 for k, t in _view_objs().items()}
move["base_device"] = str(flat.device)
move["optimizer_bound"] = all(a is b for a, b in zip([p for grp in opt.param_groups for p in grp["params"]],
                                                     m.parameters()))
move["state_devices"] = sorted({str(v.device) for st in opt.state.values() for v in st.values()
                                if isinstance(v, torch.Tensor)})
cuda_after_move = _cuda_rng()

# Training. Dropout draws from the device's stream (CUDA on the T4); the input
# jitter draws from python, numpy and torch CPU in every step.
m.train()
it = iter(loader)
losses = []
for _ in range(STEPS):
    try:
        idx = next(it)
    except StopIteration:
        it = iter(loader)
        idx = next(it)
    xb, yb = g["e15_data_x"][idx], g["e15_data_y"][idx]
    scale = 0.01 * random.random() + 0.01 * float(np.random.rand())
    xb = (xb + scale * torch.randn(xb.shape)).to(dev)
    yb = yb.to(dev)
    opt.zero_grad()
    loss = loss_fn(m(xb), yb)
    loss.backward()
    opt.step()
    losses.append(float(loss.detach()))
sched.step()
first = next(iter(opt.state))
g["step_count"] = int(opt.state[first]["step"])
g["loss_trace"].extend(losses)
g["e15_arr"][8 + BLOCK] += 1.0            # a write through the NumPy base, seen through the view
g["e15_blocks"].append({"block": BLOCK, "device": str(dev), "steps": STEPS, "step_count": g["step_count"],
                        "losses": [float(x).hex() for x in losses]})
os.makedirs(os.path.join("data", "e15"), exist_ok=True)
with open(os.path.join("data", "e15", "notes.txt"), "a") as f:
    f.write("block %d on %s: %d steps, step_count %d, last loss %s\n"
            % (BLOCK, dev, STEPS, g["step_count"], float(losses[-1]).hex()))
end = {"step_count": int(g["step_count"]), "device": str(next(m.parameters()).device),
       "fixed_batch_loss": _fixed_loss(m, loss_fn, g["train_x"], g["train_y"]),
       "lr": float(opt.param_groups[0]["lr"]), "cuda_rng": _cuda_rng()}
rng_end = _global_rng()
print(MARK + json.dumps({
    "block": BLOCK, "want": WANT, "device": str(dev), "cuda_available": cuda_ok,
    "cuda_device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
    "notes": notes, "start": start, "end": end, "move": move, "losses": losses,
    "global_rng_advanced": {k: rng_start[k] != rng_end[k] for k in rng_start},
    "cuda_rng_after_move": cuda_after_move,
    "cuda_rng_advanced_by_training": (None if cuda_after_move is None else end["cuda_rng"] != cuda_after_move),
    "optimizer_steps": sorted({float(s["step"]) for s in opt.state.values() if "step" in s}),
}, default=str))
''')


def write_program(seq: int) -> str:
    """One routed user write: increment the counter, record the sequence
    number in the namespace AND append it to a workspace file, acknowledge."""
    return _wrap(f"MARK = {MARK!r}\nSEQ = {int(seq)!r}\nLOG = {WRITES_LOG!r}\n" + r'''
import sys, os, json
g = sys.modules["__main__"].__dict__
g["e15_counter"] = g.get("e15_counter", 0) + 1
g.setdefault("e15_acks", []).append(SEQ)
os.makedirs("data", exist_ok=True)
with open(os.path.join("data", LOG), "a") as f:
    f.write("%d\n" % SEQ)
print(MARK + json.dumps({"counter": g["e15_counter"], "seq": SEQ}))
''')


def probe_program() -> str:
    """The harness's own read of the state (see the module doc, CHECKS).
    Writes nothing: the behavioural checks run on deep copies under forked
    RNG streams, and `core_unchanged_by_probe` digests the state again after
    them."""
    return _wrap(f"MARK = {MARK!r}\nWRITES_LOG = {WRITES_LOG!r}\n" + _COMMON_SRC + r'''
def _optimizer_digest(opt):
    h = hashlib.sha256()
    def feed(v):
        if isinstance(v, torch.Tensor):
            h.update(("t:%s:%r:" % (v.dtype, tuple(v.shape))).encode())
            h.update(_tbytes(v))
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
    return h.hexdigest()

def _workspace():
    files = {}
    for d, dirs, fs in os.walk("data"):
        dirs.sort()
        for fn in sorted(fs):
            p = os.path.join(d, fn)
            with open(p, "rb") as fh:
                files[os.path.relpath(p, "data")] = [_sha(fh.read()), os.path.getsize(p),
                                                     oct(os.stat(p).st_mode & 0o777)]
    return files

def _core():
    m, opt, sched = g["model"], g["optimizer"], g["scheduler"]
    core = {}
    hp = hashlib.sha256()
    for name, p in sorted(m.named_parameters()):
        hp.update(("%s|%s|%r|" % (name, p.dtype, tuple(p.shape))).encode())
        hp.update(_tbytes(p))
    for name, b in sorted(m.named_buffers()):
        hp.update(("%s|%s|%r|" % (name, b.dtype, tuple(b.shape))).encode())
        hp.update(_tbytes(b))
    core["model_params_and_buffers"] = hp.hexdigest()
    core["model_grads"] = _sha("|".join("%s:%s" % (n, "none" if p.grad is None else _sha(_tbytes(p.grad)))
                                        for n, p in sorted(m.named_parameters())))
    core["optimizer"] = _optimizer_digest(opt)
    core["scheduler"] = _sha(json.dumps(sched.state_dict(), sort_keys=True, default=repr))
    core["loader_generator"] = _sha(g["loader"].generator.get_state().numpy().tobytes())
    core["np_gen"] = _sha(repr(g["np_gen"].bit_generator.state))
    rng = _global_rng()
    core.update(rng_python=rng["python"], rng_numpy=rng["numpy"], rng_torch_cpu=rng["torch_cpu"])
    core["e15_tensors"] = _sha(b"|".join(_tbytes(g[k]) for k in (
        "e15_flat", "e15_tail", "e15_top", "e15_data_x", "e15_data_y", "train_x", "train_y", "t_cpu")))
    core["e15_numpy"] = _sha(b"|".join(np.ascontiguousarray(g[k]).tobytes()
                                       for k in ("e15_arr", "e15_arr_view", "e15_arr_T", "arr")))
    plain = [repr(g["sc_int"]), float(g["sc_float"]).hex(), g["sc_str"], repr(g["sc_bool"]), repr(g["co_list"]),
             repr(sorted(g["co_dict"].items(), key=repr)), repr(sorted(g["co_set"])), repr(g["co_tuple"]),
             repr(g["step_count"]), repr([float(x).hex() for x in g["loss_trace"]]), repr(g["e15_blocks"])]
    core["plain"] = _sha("\n".join(plain))
    ws = _workspace()
    core["workspace"] = _sha(json.dumps({k: v for k, v in ws.items() if k != WRITES_LOG}, sort_keys=True))
    return core, ws

def _fixture_param_digest(model):
    # The fixture's `_digest_params` algorithm, computed here on the live
    # model, so the digest in the capsule's refreshed expectation can be
    # compared with the source it claims to describe (a refresh that lost the
    # helper records None; a stale one records an older digest).
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().to("cpu", copy=True).contiguous().numpy().tobytes())
    for name, b in sorted(model.named_buffers()):
        h.update(name.encode())
        t = b.detach().to("cpu", copy=True).contiguous()
        h.update(t.numpy().tobytes() if t.numel() else b"empty")
    return h.hexdigest()[:32]

def _grad_sharing():
    # For every gradient, the other tensors of the state that share its
    # storage. The storage adapter records no `.grad` views (it carries
    # gradients by value), so a relationship here would not survive a hop
    # and nothing would report it: the analysis turns this into a check
    # when one exists and a scope note when none does.
    m = g["model"]
    pool = {"e15_flat": g["e15_flat"], "e15_tail": g["e15_tail"], "e15_top": g["e15_top"]}
    pool.update({"param:" + n: p for n, p in m.named_parameters()})
    grads = {n: p.grad for n, p in m.named_parameters() if p.grad is not None}
    if g["e15_top"].grad is not None:
        grads["e15_top"] = g["e15_top"].grad
    pool.update({"grad:" + n: t for n, t in grads.items()})
    out = {}
    for n, t in grads.items():
        ptr = t.untyped_storage().data_ptr()
        out[n] = sorted(k for k, o in pool.items() if k != "grad:" + n and o.device == t.device
                        and o.untyped_storage().data_ptr() == ptr)
    return out

m, opt = g["model"], g["optimizer"]
params = list(m.parameters())
slots = [p for grp in opt.param_groups for p in grp["params"]]
core_before, ws = _core()
out = {"device": str(next(m.parameters()).device), "step_count": g.get("step_count"),
       "optimizer_steps": sorted({float(s["step"]) for s in opt.state.values() if "step" in s}),
       "fixture_param_digest": _fixture_param_digest(m), "grad_sharing": _grad_sharing()}
objs = _view_objs()
out["identity"] = {
    "optimizer_params_are_model_params": len(slots) == len(params) and all(a is b for a, b in zip(slots, params)),
    "optimizer_state_nonempty": len(opt.state) > 0,
    "state_keys_are_model_params": len(opt.state) > 0 and all(any(k is p for p in params) for k in opt.state),
    "alias_is_model_weight": g["e15_w_alias"] is m.fc1.weight,
    "repeated_view_is_one_object": g["e15_tail_again"] is g["e15_tail"],
    "numpy_repeated_view_is_one_object": g["e15_arr_list"][0] is g["e15_arr_view"],
    "scheduler_bound_to_optimizer": g["scheduler"].optimizer is opt,
}
out["classes"] = {k: {"class": type(t).__name__, "requires_grad": bool(t.requires_grad), "is_leaf": bool(t.is_leaf)}
                  for k, t in objs.items()}
out["model_params_are_parameters"] = all(type(p) is torch.nn.Parameter and p.requires_grad for p in params)
flat = g["e15_flat"]
fptr = flat.untyped_storage().data_ptr()
out["torch_views"] = {}
for k, off in VIEWS:
    t = objs[k]
    out["torch_views"][k] = {"device": str(t.device), "base_device": str(flat.device),
                             "same_storage": t.device == flat.device and t.untyped_storage().data_ptr() == fptr,
                             "offset": int(t.storage_offset()), "expected_offset": off}
arr = g["e15_arr"]
def _nptr(a):
    return a.__array_interface__["data"][0]
out["numpy_views"] = {}
for k, a, off in (("e15_arr_view", g["e15_arr_view"], 64), ("e15_arr_T", g["e15_arr_T"], 0),
                  ("e15_arr_list[0]", g["e15_arr_list"][0], 64)):
    out["numpy_views"][k] = {"shares": bool(np.shares_memory(arr, a)), "byte_offset": _nptr(a) - _nptr(arr),
                             "expected_offset": off}

# Behavioural checks on deep copies, with every global stream forked.
py_state, np_state = random.getstate(), np.random.get_state()
devs = list(range(torch.cuda.device_count())) if (torch.cuda.is_available() and torch.cuda.is_initialized()) else []
beh = {}
with torch.random.fork_rng(devices=devs):
    try:
        # Reviewer case 1: torch's deepcopy memoizes storages, so the copied
        # views share the copied base exactly when the originals share theirs.
        c = copy.deepcopy({"flat": g["e15_flat"], "tail": g["e15_tail"], "tail_again": g["e15_tail_again"]})
        with torch.no_grad():
            c["flat"][680] = 99.0
        beh["repeated_view_write_through_copied_base"] = {
            "one_object": c["tail"] is c["tail_again"], "tail": float(c["tail"][0]),
            "tail_again": float(c["tail_again"][0])}
        # Reviewer case 2: the model, optimizer, loss and batch copied in ONE
        # call, so the copy's optimizer references the copy's parameters;
        # a real step must move the copy's model.
        w = copy.deepcopy({"model": g["model"], "optimizer": g["optimizer"], "loss_fn": g["loss_fn"],
                           "x": g["train_x"], "y": g["train_y"]})
        cm, co = w["model"], w["optimizer"]
        cparams = list(cm.parameters())
        cslots = [p for grp in co.param_groups for p in grp["params"]]
        beh["copy_optimizer_bound_to_copy_model"] = len(cslots) == len(cparams) and all(
            a is b for a, b in zip(cslots, cparams))
        beh["copy_fixed_batch_loss"] = _fixed_loss(cm, w["loss_fn"], w["x"], w["y"])
        before = [p.detach().clone() for p in cparams]
        cm.train()
        co.zero_grad()
        w["loss_fn"](cm(w["x"]), w["y"]).backward()
        co.step()
        beh["copy_step_moved_every_copy_param"] = all(not torch.equal(a, b) for a, b in zip(before, cparams))
        beh["copy_step_counts"] = sorted({float(s["step"]) for s in co.state.values() if "step" in s})
        del c, w, cm, co, cparams, cslots, before
    except Exception as e:  # noqa: BLE001
        beh["error"] = ("%s: %s" % (type(e).__name__, e))[:300]
random.setstate(py_state)
np.random.set_state(np_state)
core_after, _ = _core()
out["behavioural"] = beh
out["core"] = core_before
out["core_unchanged_by_probe"] = core_before == core_after
out["cuda_rng"] = _cuda_rng()
out["workspace_files"] = ws
_n = os.path.join("data", "e15", "notes.txt")
out["notes_lines"] = len(open(_n).read().splitlines()) if os.path.exists(_n) else None
_p = os.path.join("data", WRITES_LOG)
out["writes"] = {"counter": g.get("e15_counter"), "acks": list(g.get("e15_acks") or []),
                 "log": [int(x) for x in open(_p).read().split()] if os.path.exists(_p) else []}
print(MARK + json.dumps(out, default=str))
''')


# ---------------------------------------------------------------------------
# The local stand-in (dry run only)
# ---------------------------------------------------------------------------

#: Registered for the duration of ONE capture on the local stand-in for the
#: T4: every CPU storage is written with the location tag `cuda:0`, which is
#: what a capture on a GPU host produces, so the CPU destination's restore
#: must go through the remap handler. The preflight's round trip loads
#: pickles in the source, so the tagger is active for the capture only.
_TAG_CUDA = _wrap(f'''
import torch, json
def _e15_tag(obj):
    return "cuda:0" if getattr(getattr(obj, "device", None), "type", None) == "cpu" else None
torch.serialization.register_package(1, _e15_tag, lambda o, l: None)
print({MARK!r} + json.dumps({{"tagged": True}}))
''')
_UNTAG = _wrap(f'''
import torch, json
torch.serialization._package_registry[:] = [e for e in torch.serialization._package_registry if e[0] != 1]
print({MARK!r} + json.dumps({{"tagged": False}}))
''')


def _local_api_class():
    from localapi import LocalApi

    class E15LocalApi(LocalApi):
        """LocalApi with one lock per kernel (its pipe protocol is not safe
        for two threads, and a real kernel also runs one execution at a time)
        and, optionally, `cuda:0` location tags on captures taken on
        projects of the given profiles."""

        def __init__(self, *, cuda_tag_profiles: tuple[str, ...] = (), **kw):
            super().__init__(**kw)
            self._locks: dict[str, threading.Lock] = {}
            self._guard = threading.Lock()
            self.cuda_tag_profiles = set(cuda_tag_profiles)
            self.tagged_captures = 0

        def _lock(self, pid: str) -> threading.Lock:
            with self._guard:
                return self._locks.setdefault(pid, threading.Lock())

        def execute(self, pid, code, timeout_ms=600_000):
            with self._lock(pid):
                p = self.projects.get(pid)
                tag = bool(p is not None and p["profile"] in self.cuda_tag_profiles
                           and "_dump_filtered_session" in code)
                if tag:
                    super().execute(pid, _TAG_CUDA, timeout_ms)
                    self.tagged_captures += 1
                try:
                    return super().execute(pid, code, timeout_ms)
                finally:
                    if tag:
                        super().execute(pid, _UNTAG, timeout_ms)

        def delete_project(self, pid):
            with self._lock(pid):
                return super().delete_project(pid)

    return E15LocalApi


class SerializedApi(Api):
    """The live client with one lock per project: at most one execute request
    outstanding per kernel, as a notebook client sends them.

    WHY. The first live E15 run (cohort e15-live-1, 27 September) stopped in
    hop 1 when the sandbox bridge answered one of two concurrent execute
    requests on the same kernel with `ExecutionOutcomeUnknown` ("the kernel
    is no longer executing, but the execution outcome could not be
    verified"), and the kernel then failed its teardown readback. That is a
    bridge behaviour under concurrent requests to one kernel, recorded as a
    platform finding, not something the controller does. A kernel runs one
    execution at a time anyway, so serialising requests per project changes
    nothing the experiment measures: the writer still races the controller
    where admission is decided (the journal's gate, the drain of in-flight
    routed work and the authority flip), because a routed write registers in
    the journal BEFORE it waits for this lock, and the drain waits for it.
    """

    def __init__(self, base: str, key: str):
        super().__init__(base, key)
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _lock(self, pid: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(pid, threading.Lock())

    def execute(self, pid, code, timeout_ms=600_000):
        with self._lock(pid):
            return super().execute(pid, code, timeout_ms)

    def delete_project(self, pid):
        with self._lock(pid):
            return super().delete_project(pid)


# ---------------------------------------------------------------------------
# Routed writes
# ---------------------------------------------------------------------------

class Writes:
    """Every routed write of the lifecycle, in the order the harness learned
    its outcome: ack (with the runtime it ran on), refused (with the gate
    that refused it) or error (indeterminate: it may or may not have run)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self.log: list[dict[str, Any]] = []

    def next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def add(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self.log.append(entry)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(e) for e in self.log]


def routed_write(client: Controller, root: str, writes: Writes, hop: str, origin: str) -> dict[str, Any]:
    """One write through the controller's route (the path `execute_routed`
    takes; `route` also names the runtime it ran on)."""
    seq = writes.next_seq()
    entry: dict[str, Any] = {"seq": seq, "hop": hop, "origin": origin, "t_sent": time.time()}
    try:
        r = client.route(root, write_program(seq))
        entry.update(outcome="ack", runtime=r["runtime"], chain=r["chain"], counter=r["result"].get("counter"),
                     seq_echo=r["result"].get("seq"))
    except AdmissionClosed as e:
        entry.update(outcome="refused", gate=e.mig, gate_phase=e.phase)
    except Exception as e:  # noqa: BLE001
        entry.update(outcome="error", error=f"{type(e).__name__}: {e}"[:300])
    entry["t_done"] = time.time()
    writes.add(entry)
    return entry


class Writer(threading.Thread):
    """Routes writes through `root` until stopped, `pause` seconds apart."""

    def __init__(self, client: Controller, root: str, writes: Writes, hop: str, pause: float):
        super().__init__(daemon=True, name=f"e15-writer-{hop}")
        self.client, self.root, self.writes, self.hop, self.pause = client, root, writes, hop, pause
        self.stop_evt = threading.Event()

    def run(self) -> None:
        while not self.stop_evt.is_set():
            routed_write(self.client, self.root, self.writes, self.hop, "writer")
            self.stop_evt.wait(self.pause)

    def mine(self) -> list[dict[str, Any]]:
        return [e for e in self.writes.snapshot() if e["hop"] == self.hop and e["origin"] == "writer"]

    def wait_for(self, pred, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if pred(self.mine()):
                return True
            time.sleep(min(0.05, max(self.pause, 0.01)))
        return pred(self.mine())

    def finish(self) -> None:
        self.stop_evt.set()
        self.join(timeout=max(60.0, 10 * self.pause))


def _raise_killed() -> None:
    raise ControllerKilled()


# ---------------------------------------------------------------------------
# The lifecycle
# ---------------------------------------------------------------------------

class Lifecycle:
    def __init__(self, api, journal: Journal, outdir: Path, args, meta: dict, log):
        self.api, self.j, self.outdir, self.args, self.meta, self.log = api, journal, outdir, args, meta, log
        self.blobs = outdir / "blobs"
        self.tag = uuid.uuid4().hex[:6]
        self.root = f"e15-{self.tag}-h1"
        self.writes = Writes()
        #: Every project this run created or was handed, for teardown.
        self.pids: set[str] = set()
        self.roles: dict[str, str] = {}
        #: Every writer started, so teardown can stop one a failed hop left
        #: running (it would otherwise route writes into deleted projects).
        self.writers: list[Writer] = []
        self.client = Controller(api, journal, self.blobs, log=log)
        self.records_path = outdir / "e15_records.jsonl"
        self.records: list[dict[str, Any]] = []
        self.t0 = time.perf_counter()

    # -- bookkeeping ---------------------------------------------------------
    def emit(self, stage: str, name: str, **body) -> dict[str, Any]:
        rec = {"schema": SCHEMA, "stage": stage, "name": name, "tag": self.tag, "root": self.root,
               "local": bool(self.args.local), "cohort": self.meta["cohort"],
               "at_s": round(time.perf_counter() - self.t0, 3), **body}
        runmeta.stamp(rec, self.meta)
        self.records.append(rec)
        with self.records_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def role(self, pid: str | None) -> str | None:
        return None if pid is None else self.roles.get(pid, pid)

    def writes_view(self, hop: str | None = None) -> dict[str, Any]:
        log = self.writes.snapshot()
        for e in log:
            e["runtime_role"] = self.role(e.get("runtime"))
        mine = [e for e in log if hop is None or e["hop"] == hop]
        return {"entries": mine,
                "acked_so_far": [e["seq"] for e in log if e["outcome"] == "ack"],
                "refused_so_far": [e["seq"] for e in log if e["outcome"] == "refused"],
                "errors_so_far": [e["seq"] for e in log if e["outcome"] == "error"]}

    def observe(self, pid: str) -> dict[str, Any]:
        """The harness probe, bracketed by the controller's own fingerprint:
        equal fingerprints before and after show the probe wrote nothing."""
        fp0 = self.api.witness(pid, fingerprint_program())
        probe = self.api.witness(pid, probe_program())
        fp1 = self.api.witness(pid, fingerprint_program())
        v = compare_fingerprints(fp0, fp1)
        probe["fingerprint_unchanged_by_probe"] = {"equal": v["equal"], "mismatched": v["mismatched"],
                                                   "rng_cuda": v["declared"].get("rng_cuda")}
        probe["_fingerprint"] = fp0
        return probe

    @staticmethod
    def _strip(probe: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in probe.items() if k != "_fingerprint"}

    def capsule_expectation(self, mig: str) -> dict[str, Any] | None:
        """The validation expectation the capture wrote into the capsule,
        read back from the controller's copy on disk (sha-checked against the
        journal, as the controller itself reads it). The capture report says
        only WHERE the expectation came from (`expectation_source`); this is
        WHAT it says, so the analysis can compare it with the source."""
        row = self.j.get(mig) or {}
        path = row.get("capsule_path")
        if not path:
            return None
        try:
            blob = Path(path).read_bytes()
            if hashlib.sha256(blob).hexdigest() != row.get("capsule_sha"):
                return {"error": "capsule sha256 does not match the journal"}
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as t:
                man = json.load(t.extractfile("manifests.json"))
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"[:300]}
        exp = man.get("expectation") or {}
        return {"source": man.get("expectation_source"), "isolation": man.get("expectation_isolation"),
                "method": man.get("expectation_method"),
                **{k: exp.get(k) for k in ("param_digest", "step_count", "forward_output", "forward_output_not_run",
                                           "not_refreshed", "forward_output_gap",
                                           "device_type", "scheduler_lr", "scheduler_epoch")},
                "next_loader_indices": len(exp.get("next_loader_indices") or [])}

    def committed_at(self, mig: str) -> float | None:
        for e in self.j.events(mig):
            if e["phase"] == "COMMITTED" and '"authoritative"' in (e["note"] or ""):
                return e["at"]
        return None

    def alive(self) -> set[str]:
        try:
            return {it["id"] for it in self.api.list_projects()}
        except Exception as exc:  # noqa: BLE001
            self.log(f"  list_projects failed: {type(exc).__name__}: {exc}")
            return set()

    def import_cell(self, pid: str) -> dict[str, Any] | None:
        if self.args.reimport_cell != "fixture":
            return None
        return self.api.witness(pid, import_cell_program())

    def sabotage_after_commit(self, dest: str) -> dict[str, Any] | None:
        """`--sabotage unshare_views`: a defect on the destination after the
        commit, before the harness looks: `model.fc1.weight` gets its own
        copy of its bytes (Parameter identity and values kept, the storage
        relationship to the base gone)."""
        if self.args.sabotage != "unshare_views":
            return None
        prog = _wrap(f'''
import sys, json
g = sys.modules["__main__"].__dict__
w = g["model"].fc1.weight
w.data = w.data.clone()
print({MARK!r} + json.dumps({{"unshared": "model.fc1.weight"}}))
''')
        return self.api.witness(dest, prog)

    # -- stages ----------------------------------------------------------------
    def setup(self) -> str:
        a = self.args
        t = time.perf_counter()
        src = self.api.create_project(f"clusy-exp-handoff-src-{self.root}", SOURCE_PROFILE)
        self.pids.add(src)
        self.roles[src] = "cpu-source"
        self.log(f"[setup] source {src}: seeding the fixture and building the E15 state")
        seeded = self.api.witness(src, seed_program(seed=a.seed, files=4, corpus_bytes=16 * 1024))
        state = self.api.witness(src, state_program(a.seed))
        train = self.api.witness(src, train_program(a.steps, 0, "cpu"))
        self.emit("setup", "setup", source_pid=src, source_profile=SOURCE_PROFILE, gpu_profile=a.gpu_profile,
                  seeded_names=seeded.get("names"), state=state, args=vars(a), phases=PHASES,
                  expected_view_records=EXPECTED_VIEW_RECORDS, writes_log=WRITES_LOG,
                  wall_s=round(time.perf_counter() - t, 2))
        self.emit("train", "block0", block=0, runtime=src, runtime_role="cpu-source", profile=SOURCE_PROFILE,
                  import_cell=None, report=train)
        self.log(f"[block0] cpu: step_count {train['end']['step_count']}, last loss {train['losses'][-1]:.4f}")
        return src

    def hop_commit(self, src: str) -> str | None:
        """Hop 1: cpu -> GPU profile with a writer throughout."""
        a, mig = self.args, self.root
        self.log(f"[hop1] {mig}: {SOURCE_PROFILE} -> {a.gpu_profile}, writer routing through {self.root}")
        pre = self.observe(src)
        self.j.create(mig, src, a.gpu_profile)
        writer = Writer(self.client, self.root, self.writes, "hop1", a.writer_pause)
        self.writers.append(writer)
        writer.start()
        writer.wait_for(lambda es: any(e["outcome"] == "ack" for e in es), timeout=a.phase_timeout)
        c = Controller(self.api, self.j, self.blobs, log=self.log)
        t = time.perf_counter()
        res = c.migrate(mig, src, a.gpu_profile)
        wall = time.perf_counter() - t
        dest = res.get("dest") or (self.j.get(mig) or {}).get("dest_pid")
        if dest:
            self.pids.add(dest)
            self.roles[dest] = "gpu-runtime"
        if res["phase"] == "DONE":
            writer.wait_for(lambda es: sum(1 for e in es if e["outcome"] == "ack" and e.get("runtime") == dest)
                            >= a.post_commit_writes, timeout=a.phase_timeout)
        writer.finish()
        sab = self.sabotage_after_commit(dest) if res["phase"] == "DONE" else None
        post = self.observe(dest) if res["phase"] == "DONE" else None
        alive = self.alive()
        self.emit("hop", "hop1", kind="commit", mig=mig, source_pid=src, dest_pid=dest,
                  source_role="cpu-source", dest_role="gpu-runtime",
                  source_profile=SOURCE_PROFILE, dest_profile=a.gpu_profile, fault=None, crash_phase=None,
                  result=res, events=self.j.events(mig), journal_row=self._row(mig),
                  expectation=self.capsule_expectation(mig),
                  authority_after=self.j.authority(self.root), committed_at=self.committed_at(mig),
                  writes=self.writes_view("hop1"), probe_pre=self._strip(pre),
                  probe_post=self._strip(post) if post else None, sabotage=sab,
                  source_alive_after=src in alive, dest_alive_after=(dest in alive) if dest else None,
                  wall_s=round(wall, 2))
        self.log(f"[hop1] -> {res['phase']} {res.get('reason') or ''} oracles "
                 f"{(res.get('timings') or {}).get('oracles')} | writes acked "
                 f"{sum(1 for e in writer.mine() if e['outcome'] == 'ack')}, refused "
                 f"{sum(1 for e in writer.mine() if e['outcome'] == 'refused')}")
        return dest if res["phase"] == "DONE" else None

    def train_block(self, pid: str, block: int, want: str, role: str, profile: str) -> dict[str, Any]:
        cell = self.import_cell(pid)
        # A dry run has no CUDA: the GPU block still runs its move, onto a
        # fresh cpu buffer (see `train_program`).
        force = bool(self.args.local and want == "cuda")
        rep = self.api.witness(pid, train_program(self.args.steps, block, want, force_move=force))
        self.emit("train", f"block{block}", block=block, runtime=pid, runtime_role=role, profile=profile,
                  import_cell=cell, report=rep)
        self.log(f"[block{block}] {role} on {rep['device']}: step_count {rep['end']['step_count']}, "
                 f"last loss {rep['losses'][-1]:.4f}{' | ' + '; '.join(rep['notes']) if rep['notes'] else ''}")
        return rep

    def hop_abort(self, gpu: str) -> dict[str, Any]:
        """Hop 2a: GPU -> cpu with a fault; must abort and leave the GPU
        runtime exactly as it was."""
        a = self.args
        mig = f"e15-{self.tag}-h2a"
        self.log(f"[hop2a] {mig}: {a.gpu_profile} -> {DEST_PROFILE} with fault {a.abort_fault}")
        pre = self.observe(gpu)
        t = time.perf_counter()
        res = Controller(self.api, self.j, self.blobs, log=self.log).migrate(mig, gpu, DEST_PROFILE,
                                                                             fault=a.abort_fault)
        wall = time.perf_counter() - t
        dest = (self.j.get(mig) or {}).get("dest_pid")
        if dest:
            self.pids.add(dest)
            self.roles[dest] = "aborted-destination"
        post = self.observe(gpu)
        fp_verdict = compare_fingerprints(pre["_fingerprint"], post["_fingerprint"])
        alive = self.alive()
        after = routed_write(self.client, self.root, self.writes, "hop2a", "after-abort")
        self.emit("hop", "hop2a", kind="abort", mig=mig, source_pid=gpu, dest_pid=dest,
                  source_role="gpu-runtime", dest_role="aborted-destination",
                  source_profile=a.gpu_profile, dest_profile=DEST_PROFILE, fault=a.abort_fault, crash_phase=None,
                  result=res, events=self.j.events(mig), journal_row=self._row(mig),
                  expectation=self.capsule_expectation(mig),
                  authority_after=self.j.authority(self.root),
                  source_fingerprint_verdict=fp_verdict,
                  source_rng_cuda=[pre["_fingerprint"].get("rng_cuda"), post["_fingerprint"].get("rng_cuda")],
                  probe_pre=self._strip(pre), probe_post=self._strip(post),
                  write_after_abort={**after, "runtime_role": self.role(after.get("runtime"))},
                  writes=self.writes_view("hop2a"),
                  source_alive_after=gpu in alive, dest_alive_after=(dest in alive) if dest else False,
                  wall_s=round(wall, 2))
        self.log(f"[hop2a] -> {res['phase']} at {res.get('failed_at')} ({res.get('reason')}); source state "
                 f"{'unchanged' if fp_verdict['equal'] else 'CHANGED ' + str(fp_verdict['mismatched'])}; "
                 f"next write on {self.role(after.get('runtime'))}")
        return res

    def hop_recovery(self, gpu: str) -> str | None:
        """Hop 2b: GPU -> cpu, controller killed inside the gate, writes
        refused while it is down, a fresh controller resumes to DONE."""
        a = self.args
        mig = f"e15-{self.tag}-h2b"
        self.log(f"[hop2b] {mig}: {a.gpu_profile} -> {DEST_PROFILE}, controller killed after the "
                 f"{a.crash_phase} journal write")
        pre = self.observe(gpu)
        writer = Writer(self.client, self.root, self.writes, "hop2b", a.writer_pause)
        self.writers.append(writer)
        writer.start()
        writer.wait_for(lambda es: any(e["outcome"] == "ack" for e in es), timeout=a.phase_timeout)
        dead_j = Journal(self.j.path)
        dead = Controller(self.api, dead_j, self.blobs, log=self.log, crash=CrashPoint(a.crash_phase),
                          kill=_raise_killed)
        killed, early = False, None
        t = time.perf_counter()
        try:
            early = dead.migrate(mig, gpu, DEST_PROFILE)
        except ControllerKilled:
            killed = True
        finally:
            # A dead process's connection is closed by the OS; the lease it
            # held stays in the journal, which is what takeover is for.
            dead_j.db.close()
        row_down = self._row(mig)
        down = {"killed": killed, "early_result": early, "phase_at_kill": row_down.get("phase"),
                "admission_at_kill": row_down.get("admission"), "authority": self.j.authority(self.root),
                "dead_controller_reports": {k: dead.reports.get(k) for k in ("preflight", "capture", "admission")}}
        down_writes = [routed_write(self.client, self.root, self.writes, "hop2b", "while-down")
                       for _ in range(a.down_writes)]
        # Give the writer thread time to meet the closed gate as well.
        time.sleep(a.down_seconds)
        down["authority_before_resume"] = self.j.authority(self.root)
        resumed = Controller(self.api, self.j, self.blobs, log=self.log)
        t2 = time.perf_counter()
        res = resumed.migrate(mig, gpu, DEST_PROFILE, takeover=True) if killed else early
        recovery_s = time.perf_counter() - t2
        wall = time.perf_counter() - t
        dest = (res or {}).get("dest") or (self.j.get(mig) or {}).get("dest_pid")
        if dest:
            self.pids.add(dest)
            self.roles[dest] = "cpu-destination"
        done = bool(res) and res["phase"] == "DONE"
        if done:
            writer.wait_for(lambda es: sum(1 for e in es if e["outcome"] == "ack" and e.get("runtime") == dest)
                            >= a.post_commit_writes, timeout=a.phase_timeout)
        writer.finish()
        post = self.observe(dest) if done else None
        alive = self.alive()
        self.emit("hop", "hop2b", kind="recovery", mig=mig, source_pid=gpu, dest_pid=dest,
                  source_role="gpu-runtime", dest_role="cpu-destination",
                  source_profile=a.gpu_profile, dest_profile=DEST_PROFILE, fault=None, crash_phase=a.crash_phase,
                  result=res, down=down, down_writes=down_writes,
                  capture=(res or {}).get("capture") or dead.reports.get("capture"),
                  preflight=(res or {}).get("preflight") or dead.reports.get("preflight"),
                  events=self.j.events(mig), journal_row=self._row(mig),
                  expectation=self.capsule_expectation(mig),
                  authority_after=self.j.authority(self.root), committed_at=self.committed_at(mig),
                  writes=self.writes_view("hop2b"), probe_pre=self._strip(pre),
                  probe_post=self._strip(post) if post else None,
                  source_alive_after=gpu in alive, dest_alive_after=(dest in alive) if dest else None,
                  recovery_s=round(recovery_s, 2), wall_s=round(wall, 2))
        self.log(f"[hop2b] killed={killed} at {down['phase_at_kill']} (gate {down['admission_at_kill']}); "
                 f"{sum(1 for w in down_writes if w['outcome'] == 'refused')}/{len(down_writes)} writes refused "
                 f"while down; resumed -> {(res or {}).get('phase')} in {recovery_s:.1f} s, remapped "
                 f"{(((res or {}).get('restore') or {}).get('remap') or {}).get('remapped')}")
        return dest if done else None

    def final(self, pid: str | None) -> None:
        probe = self.observe(pid) if pid else None
        self.emit("final", "final", runtime=pid, runtime_role=self.role(pid),
                  probe=self._strip(probe) if probe else None, writes=self.writes_view(None),
                  tagged_captures=getattr(self.api, "tagged_captures", None),
                  wall_s=round(time.perf_counter() - self.t0, 2))

    def _row(self, mig: str) -> dict[str, Any]:
        r = self.j.get(mig) or {}
        return {k: r.get(k) for k in ("phase", "authoritative", "admission", "source_pid", "dest_pid",
                                      "abort_reason", "lease_owner")}

    # -- the whole run ---------------------------------------------------------
    def run(self) -> None:
        a = self.args
        stopped = None
        try:
            src = self.setup()
            gpu = self.hop_commit(src)
            if gpu is None:
                stopped = "hop1 did not reach DONE"
                return
            self.train_block(gpu, 1, "cuda", "gpu-runtime", a.gpu_profile)
            self.hop_abort(gpu)
            cpu = self.hop_recovery(gpu)
            if cpu is None:
                stopped = "hop2b did not reach DONE"
                self.final(gpu)
                return
            self.train_block(cpu, 2, "cpu", "cpu-destination", DEST_PROFILE)
            self.final(cpu)
        except Exception as exc:  # noqa: BLE001
            stopped = f"harness error: {type(exc).__name__}: {exc}"
            self.log(f"  !! {stopped}")
            raise
        finally:
            if stopped:
                self.emit("stopped", "stopped", reason=stopped[:500], writes=self.writes_view(None))
            self.teardown()

    def teardown(self) -> None:
        for w in self.writers:
            if w.is_alive():
                w.finish()
        done = set()
        for pid in sorted(self.pids):
            try:
                self.api.delete_project(pid)
                done.add(pid)
            except Exception as exc:  # noqa: BLE001
                self.log(f"  teardown: delete {pid} failed: {type(exc).__name__}: {exc}")
        try:
            for it in self.api.list_projects():
                name = str(it.get("name", ""))
                if (name.startswith(f"clusy-exp-handoff-e15-{self.tag}")
                        or name == f"clusy-exp-handoff-src-{self.root}") and it["id"] not in done:
                    self.api.delete_project(it["id"])
                    self.log(f"  teardown: deleted {it['id']} ({name}) found by name")
        except Exception as exc:  # noqa: BLE001
            self.log(f"  teardown: listing projects failed: {type(exc).__name__}: {exc}")


def _apply_sabotage(kind: str | None, args) -> None:
    """Local-only deliberate breakage (see the module doc)."""
    if kind is None or kind == "unshare_views":
        return
    if kind == "inplace_validation":
        _ctl.VALIDATION_CONTINUATION = "inplace"
    elif kind == "ungated_route":
        _ctl.Journal._gate_closed = staticmethod(lambda row: False)
    else:
        raise SystemExit(f"unknown --sabotage {kind}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--api-url", default=os.environ.get("CLUSY_API_URL", "http://localhost:8010"))
    ap.add_argument("--local", action="store_true", help="dry run against LocalApi; writes to --outdir")
    ap.add_argument("--outdir", default=None, help="default results/e15 (a temp dir with --local)")
    ap.add_argument("--cohort", default=None, help="written into every record (default: the run id)")
    ap.add_argument("--gpu-profile", default="gpu_t4")
    ap.add_argument("--steps", type=int, default=6, help="training steps per block")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--crash-phase", choices=["ADMISSION_CLOSED", "CAPTURED"], default="CAPTURED",
                    help="hop 2b: the journal write after which the controller is killed")
    ap.add_argument("--abort-fault", choices=["bad_capsule", "missing_package"], default="bad_capsule")
    ap.add_argument("--writer-pause", type=float, default=None,
                    help="seconds between routed writes (default 0.02 locally, 1.0 live)")
    ap.add_argument("--post-commit-writes", type=int, default=3,
                    help="acknowledged writes on the new destination before a writer stops")
    ap.add_argument("--down-writes", type=int, default=3, help="hop 2b: writes routed while the controller is down")
    ap.add_argument("--down-seconds", type=float, default=None,
                    help="hop 2b: how long the controller stays down (default 0.3 locally, 10 live)")
    ap.add_argument("--phase-timeout", type=float, default=None,
                    help="bound on each wait for writer progress (default 20 s locally, 300 s live)")
    ap.add_argument("--reimport-cell", choices=["none", "fixture"], default="none",
                    help="'fixture' is a DIAGNOSTIC that re-runs the fixture definitions cell on each restored "
                         "runtime; such a run is never judged OK (see the module doc)")
    ap.add_argument("--no-local-cuda-tags", action="store_true",
                    help="--local: do not tag the T4 stand-in's captures cuda:0")
    ap.add_argument("--substrates", default=None,
                    help="with --local: where each profile's kernels run, e.g. "
                         "'cpu=python:/path/to/python,gpu_t4=docker:IMAGE' (see localapi.parse_substrates)")
    ap.add_argument("--sabotage", choices=["inplace_validation", "ungated_route", "unshare_views"],
                    default=None, help="local only: break one thing on purpose to show the checks fail")
    args = ap.parse_args()
    log = lambda *x: print(*x, flush=True)  # noqa: E731
    if args.writer_pause is None:
        args.writer_pause = 0.02 if args.local else 1.0
    if args.down_seconds is None:
        args.down_seconds = 0.3 if args.local else 10.0
    if args.phase_timeout is None:
        args.phase_timeout = 20.0 if args.local else 300.0

    if args.local:
        LocalCls = _local_api_class()
        outdir = Path(args.outdir or tempfile.mkdtemp(prefix="e15-local-"))
        if (ROOT / "results").resolve() in [outdir.resolve(), *outdir.resolve().parents]:
            # A dry run is not a result; it must never land beside live records.
            print("--local refuses to write under results/; pass a scratch --outdir", file=sys.stderr)
            return 2
        from localapi import parse_substrates
        api = LocalCls(workdir=outdir / "kernels", substrates=parse_substrates(args.substrates),
                       cuda_tag_profiles=() if args.no_local_cuda_tags else (args.gpu_profile,))
        api_url = None
        _apply_sabotage(args.sabotage, args)
    else:
        if args.sabotage:
            print("--sabotage is local only", file=sys.stderr)
            return 2
        key = os.environ.get("CLUSY_HARNESS_API_KEY")
        if not key:
            print("export CLUSY_HARNESS_API_KEY", file=sys.stderr)
            return 1
        api = SerializedApi(args.api_url, key)
        api_url = args.api_url
        outdir = Path(args.outdir) if args.outdir else ROOT / "results" / "e15"
        why = runmeta.shipped_record_conflict(outdir / "e15_records.jsonl", args.cohort)
        if why:
            print(why, file=sys.stderr)
            return 2
    outdir.mkdir(parents=True, exist_ok=True)

    extra = [ROOT / "experiments" / "e15_analyse.py"] + ([ROOT / "experiments" / "localapi.py"] if args.local else [])
    meta = runmeta.start_run("e15", api_url=api_url, args=vars(args), cohort=args.cohort, extra_files=extra)
    meta["cohort"] = args.cohort or meta["run_id"]
    meta["local"] = bool(args.local)
    if args.local and args.substrates:
        from localapi import parse_substrates, substrate_identity
        meta["substrates"] = {prof: substrate_identity(t) for prof, t in parse_substrates(args.substrates).items()}
    journal = Journal(outdir / "journal.sqlite")
    log(f"E15 run {meta['run_id']} cohort {meta['cohort']} -> {outdir}"
        + (f"  (LOCAL DRY RUN{', sabotage ' + args.sabotage if args.sabotage else ''})" if args.local else ""))
    life = Lifecycle(api, journal, outdir, args, meta, log)
    harness_error = None
    try:
        life.run()
    except Exception as exc:  # noqa: BLE001  (recorded and judged, never swallowed)
        harness_error = f"{type(exc).__name__}: {exc}"
    finally:
        runmeta.finish_run(meta, api_url=api_url)
        if args.local:
            api.close()
    result = e15_analyse.judge(life.records)
    meta["results"] = {"checks": len(result["rows"]), "failed": result["failed"], "skipped": result["skipped"],
                       "harness_error": harness_error, "records": str(life.records_path)}
    runmeta.write_meta(meta, outdir / "runs_meta.jsonl")
    print()
    print(e15_analyse.render(result))
    if harness_error:
        print(f"\nHARNESS ERROR: {harness_error}")
    print(f"\nrecords: {life.records_path}")
    return 0 if (result["ok"] and not harness_error) else 1


if __name__ == "__main__":
    raise SystemExit(main())
