"""Verification for a transported namespace, aware of what a boundary may change.

The forecast identified this file as the most likely source of failures that
*look* like system failures and are not. Two rules avoid them.

**Values are held to bitwise identity; recomputation is not.** Moving a tensor
between devices copies bits without changing them, so a parameter that differs
after a transition was lost, not rounded — exact is the right standard and any
tolerance there would hide real corruption. But *recomputing* a forward pass on
a different device is a different arithmetic, and demanding equality would fail
every heterogeneous transition for a reason that is physics rather than a
defect. So the parameter digest is exact everywhere, and the forward output is
exact within a device class and tolerance-bounded across one.

**The device is expected to change.** The oracle asserts tensors are on the
*destination's* device, not on the source's. Checking that the device did not
change would fail every row in Table 1 by construction.

Every tolerance-bounded pass is labelled and carries its measured deviation, so
a weaker claim is never rendered as a stronger one.

**Where the checks that run model code execute (`continuation`).** Two of the
checks EXECUTE the model: the forward output and the continuation step (a real
backward plus `optimizer.step()`). Run on the namespace's own objects they
change it: the step advances every optimizer state counter and moves the
parameters, and the tinyblock fixture's forward increments a registered
buffer. That is what an evaluation that compares against controls wants (E13
and E14 match those calls in their controls), and it is the default,
`"inplace"`. It is wrong for a runtime handoff, which must commit exactly the
state it captured: a reviewer's probe found a clean migration reporting 12/12
while every optimizer counter had advanced 3 -> 4 and the parameter bytes had
changed. `"copy"` runs both checks on ONE `copy.deepcopy` of the model,
optimizer, scheduler, loss function and train batch taken together (one call,
so the copy's optimizer references the copy's parameters), under a forked
process RNG, and judges the step's success on the copy. The shared-reference
check becomes identity only and the data cursor is read from a clone of the
loader's generator. `"skip"` is the same without the two model-executing rows.

**A deepcopy isolates objects, not code.** `copy.deepcopy` copies functions
by reference, so a forward hook, a closure or a `forward` that writes to a
module-level name writes into the REAL namespace even when it runs on the
copy. A second review found exactly that: a hook storing activations in a
top-level dict ran three extra times during a "copy" validation, and the
migration committed the changed dict. `"isolated"` is the mode for a runtime
handoff: the whole oracle set runs in a FORKED CHILD process (in place there,
on the child's own copy-on-write memory), and the rows come back over a pipe;
this process runs no model code at all. What a fork does not isolate is
stated, not hidden: the filesystem and anything outside the process (a hook
that appends to a file changes the real file). So a fork CONTAINS memory and
the caller must still check what it does not contain: the controller hashes
the state and the workspace before and after the call and aborts on any
difference (`validation_side_effect`). Where a fork is unsafe (no `os.fork`,
or CUDA already initialized, which a forked child cannot use) the
model-executing rows run IN THIS PROCESS on one disposable `"copy"` under a
forked RNG, and the same before/after check is what makes that safe: a hook
or closure that escapes the copy is caught and the switch fails closed. (An
earlier version declared those rows not run instead; with the before/after
check in place, running them on the copy is both stronger and safe.) PyTorch
adds one more limit: once a process has run a backward pass, it refuses
autograd in any child forked after that ("Autograd and Fork"). A freshly
restored destination has never run one, so a handoff's validation is
unaffected; a runtime re-validated after training gets its continuation row
declared (not run), with that reason, instead of a failure.

**The forward output has no reference in a handoff.** Taking one means running
the model on the SOURCE, and a transparent handoff runs no user model code on
the source (a forward hook there is user code with side effects). An
expectation that carries `forward_output_not_run` (the controller's) therefore
declares the forward row with that reason: recomputation is an evaluation
oracle, and the bytes are established by the parameter digest and the
controller's commit-boundary fingerprint. An expectation from an older capsule
that carries `forward_output_gap` declares it the same way. A declared row is
NOT RUN: its `ok` is None, never a pass and never a failure, and callers count
it separately (see `summarize` and the controller's `verify_program`).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

#: Where the model-executing checks run; see the module doc.
CONTINUATION_MODES = ("inplace", "copy", "skip", "isolated")

#: The rows that execute model code, in the order `verify` emits them, each
#: with the row it follows (so a declared or failed stand-in keeps its place).
MODEL_ROWS = (("forward output", "destination device"), ("continuation", "optimizer"))

#: How long an isolated oracle run may take before the child is killed and
#: the model-executing rows fail (fail closed, never fall back in process).
ISOLATION_TIMEOUT_S = 300.0

#: PyTorch's refusal of autograd in a child forked after the parent ran a
#: backward pass (torch/csrc/autograd/engine.cpp, "Autograd and Fork").
AUTOGRAD_FORK_REFUSAL = "Unable to handle autograd's threading in combination with fork-based multiprocessing"

#: The objects the disposable copy is taken over, in ONE deepcopy call. They
#: travel together because the relationships between them are what the
#: continuation step exercises: a separately copied optimizer would reference
#: the ORIGINAL parameters and its step would move the real model.
COPY_TOGETHER = ("model", "optimizer", "scheduler", "loss_fn", "train_x", "train_y")

#: The expectation key that declares the forward row not run, and says why
#: (see the module doc). Its value is the reason, reported verbatim.
FORWARD_NOT_RUN_KEY = "forward_output_not_run"

#: Relative tolerance for a recomputed forward pass across a device change.
#: fp32 accumulation order differs between CUDA and CPU reductions, and between
#: GPU generations; 1e-4 is loose enough for that and far tighter than any real
#: corruption, which shifts outputs by orders of magnitude.
CROSS_DEVICE_RTOL = 1e-4


@dataclass
class OracleResult:
    name: str
    #: True pass, False failure, None NOT RUN (only with mode "declared"): a
    #: check that did not run is never counted as a pass.
    ok: bool | None
    mode: str        # "exact" | "tolerance" | "structural" | "declared" (not run; see module doc)
    detail: str
    measured: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "ok": self.ok, "mode": self.mode,
            "detail": self.detail, "measured": self.measured,
        }


@contextlib.contextmanager
def _forked_rng(torch: Any):
    """Run a block without moving any process-global RNG stream.

    The disposable copy's forward and step may draw from the global streams
    (dropout draws from torch's; a sampler without its own generator draws a
    seed from it). The copy is disposable, the process streams are not: a
    validation that advanced them would hand the destination a different
    future than the source had. CUDA streams are forked only when CUDA is
    already initialized, so a check never initializes CUDA as a side effect.
    """
    import random
    import sys

    np = sys.modules.get("numpy")
    py_state = random.getstate()
    np_state = np.random.get_state() if np is not None else None
    devices = (list(range(torch.cuda.device_count()))
               if torch.cuda.is_available() and torch.cuda.is_initialized() else [])
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            random.setstate(py_state)
            if np_state is not None:
                np.random.set_state(np_state)


def fork_unsafe_reason(maps_path: str = "/proc/self/maps",
                       device_nodes: tuple[str, ...] = ("/dev/nvidiactl", "/dev/nvidia0")) -> str | None:
    """Why a forked child cannot isolate model code in this process, or None.

    CUDA is the case that matters: once a process has initialized CUDA, a
    forked child cannot use it (torch refuses, "Cannot re-initialize CUDA in
    forked subprocess"), so a CUDA model's forward would fail in the child
    for a reason that has nothing to do with the state. Checked with
    `is_initialized`, which never initializes CUDA itself.

    `is_initialized` is not enough on a GPU host. `torch.cuda.is_available()`
    and `device_count()` load and initialize the CUDA driver through
    `cudaGetDeviceCount` without setting torch's flag, and a child forked after
    that fails its first CUDA call, including the ones an optimizer step or a
    backward pass makes on CPU tensors, with "CUDA error: initialization
    error" (live E15 hop 1 on a Modal T4, 27 September). So the driver library
    being mapped into this process, or a CUDA build of torch on a host with
    NVIDIA device nodes, also rules the fork out; none of these checks
    initializes anything. The caller then runs the model checks on a
    disposable copy, bracketed by its whole-state fingerprint."""
    import os
    import sys

    if not hasattr(os, "fork"):
        return "os.fork is not available on this platform"
    try:
        with open(maps_path) as fh:
            if any("libcuda.so" in line for line in fh):
                return "the CUDA driver is loaded in this process and a forked child cannot use it"
    except OSError:
        pass
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_initialized():
                return "CUDA is initialized in this process and a forked child cannot use it"
        except Exception as exc:  # noqa: BLE001
            return f"CUDA state unreadable ({type(exc).__name__})"
        if getattr(getattr(torch, "version", None), "cuda", None) and any(
                os.path.exists(p) for p in device_nodes):
            return "a CUDA build of torch on a GPU host: the driver may already be initialized here"
    return None


def run_isolated(fn: Any, *, timeout: float = ISOLATION_TIMEOUT_S) -> dict[str, Any]:
    """Run `fn()` in a forked child and return {"ok": True, "value": ...} or
    {"ok": False, "error": ...}. `fn`'s result must be JSON-serialisable.

    The child works on its own copy-on-write image of this process, so
    nothing it does to memory (a hook writing a global, an optimizer step, a
    buffer update, an RNG draw) reaches the parent. It never returns to the
    caller: it writes its result to a pipe and leaves with `os._exit`, which
    skips atexit handlers and never flushes the parent's inherited buffers.
    Before running anything it points fds 0-2 at /dev/null and replaces
    `sys.stdout`/`sys.stderr`: a kernel's stdout may be the channel its host
    reads (the local test double) or a socket another thread owns (Jupyter),
    and a child writing there would corrupt the parent's stream. It limits
    torch to one intra-op thread, because OpenMP thread pools do not survive
    a fork (the DataLoader workers do the same).

    The parent reads until the child closes the pipe, bounded by `timeout`;
    past it the child is killed and the run fails. A child that dies without
    writing a result is a failure too. Nothing is retried in process."""
    import io
    import json
    import os
    import select
    import signal
    import sys
    import time
    import warnings

    r, w = os.pipe()
    with warnings.catch_warnings():
        # Python 3.12+ warns when a multi-threaded process forks; the child
        # below touches only its own thread, a pipe and /dev/null.
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:                                           # the child: never returns
        try:
            os.close(r)
            try:
                null = os.open(os.devnull, os.O_RDWR)
                for fd in (0, 1, 2):
                    os.dup2(null, fd)
                sys.stdout = sys.stderr = io.StringIO()
                torch = sys.modules.get("torch")
                if torch is not None:
                    torch.set_num_threads(1)
                payload = {"ok": True, "value": fn()}
            except BaseException as exc:  # noqa: BLE001
                payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
            try:
                data = json.dumps(payload, default=str).encode()
            except Exception as exc:  # noqa: BLE001
                data = json.dumps({"ok": False, "error": f"result not serialisable: {exc}"[:500]}).encode()
            view = memoryview(data)
            while view:
                view = view[os.write(w, view):]
        finally:
            os._exit(0)
    os.close(w)
    chunks: list[bytes] = []
    error = None
    deadline = time.monotonic() + timeout
    try:
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                error = f"isolated run timed out after {timeout:.0f} s"
                break
            ready, _, _ = select.select([r], [], [], min(left, 1.0))
            if ready:
                chunk = os.read(r, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
    finally:
        os.close(r)
        if error is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        _, status = os.waitpid(pid, 0)
    if error is not None:
        return {"ok": False, "error": error}
    try:
        out = json.loads(b"".join(chunks))
    except ValueError:
        how = (f"signal {os.WTERMSIG(status)}" if os.WIFSIGNALED(status)
               else f"exit status {os.WEXITSTATUS(status)}")
        return {"ok": False, "error": f"the isolated child died ({how}) without a result"}
    return out if isinstance(out, dict) else {"ok": False, "error": "malformed result from the isolated child"}


def _with_model_rows(rows: list[OracleResult], extra: dict[str, OracleResult]) -> list[OracleResult]:
    """Put stand-ins for the model-executing rows where `verify` emits them.
    Nothing is added when the anchor row is absent (no model, no optimizer):
    those rows already fail on their own."""
    out = list(rows)
    for name, after in MODEL_ROWS:
        if name not in extra:
            continue
        at = next((i for i, r in enumerate(out) if r.name == after), None)
        if at is not None:
            out.insert(at + 1, extra[name])
    return out


def verify(
    ns: dict[str, Any],
    expected: dict[str, Any],
    *,
    destination_device: str,
    source_device: str,
    rtol: float = CROSS_DEVICE_RTOL,
    same_hardware: bool = False,
    continuation: str = "inplace",
    isolation_timeout: float = ISOLATION_TIMEOUT_S,
) -> list[OracleResult]:
    """Check a restored namespace against what the source recorded.

    `continuation` is "inplace" (the default: E13, E14 and the driver compare
    against controls that make the same calls), "copy" (the model-executing
    checks run on one disposable deepcopy; see the module doc for what that
    does NOT isolate), "skip" (as "copy", without the two model-executing
    rows) or "isolated" (every check runs in a forked child, and where no fork
    can isolate them the model-executing checks run on an in-process copy; the
    mode for a runtime handoff, whose caller compares the state before and
    after the call).
    """
    if continuation not in CONTINUATION_MODES:
        raise ValueError(f"continuation must be one of {CONTINUATION_MODES}, not {continuation!r}")
    if ns.get("model") is None and not (expected or {}).get("param_digest"):
        # A session with no benchmark fixture (any Python workload): these
        # rows know the fixture's names and model, so none applies. The
        # generic check is the controller's exact commit-boundary fingerprint
        # (names, values, object digests, aliases, views, RNG, workspace),
        # compared after restore and again before commit, with validation
        # bracketed by the same fingerprint. Declared, never counted as a pass.
        return [OracleResult("workload checks", None, "declared",
                             "NOT RUN: no benchmark fixture in this session; stored state is checked "
                             "by the commit-boundary fingerprint")]
    if continuation == "isolated":
        kw = dict(destination_device=destination_device, source_device=source_device, rtol=rtol,
                  same_hardware=same_hardware)
        reason = fork_unsafe_reason()
        if reason is None:
            res = run_isolated(lambda: [r.to_dict() for r in verify(ns, expected, continuation="inplace", **kw)],
                               timeout=isolation_timeout)
            if res.get("ok"):
                rows = [OracleResult(**d) for d in res["value"]]
                for r in rows:
                    if r.name not in dict(MODEL_ROWS) or r.mode == "declared":
                        continue
                    if not r.ok and AUTOGRAD_FORK_REFUSAL in r.detail:
                        # Not a result about the state: torch refused to run
                        # the step at all (see the module doc).
                        r.ok, r.mode, r.measured = None, "declared", None
                        r.detail = ("NOT RUN: this process has already run a backward pass, and PyTorch refuses "
                                    "autograd in a child forked after that (Autograd and Fork); running the step "
                                    "in this process could change the state it is validating")
                    else:
                        r.detail += " (in a forked child; this process ran no model code)"
                return rows
            # Fail closed: the model-executing rows FAIL with the reason (the
            # forward row stays declared when the expectation declares it);
            # the others are re-read here, which runs no model code.
            why = f"isolated run failed: {res.get('error')}"
            extra = {n: OracleResult(n, False, "structural", why) for n, _ in MODEL_ROWS}
            not_run = expected.get(FORWARD_NOT_RUN_KEY)
            if not_run:
                extra["forward output"] = OracleResult("forward output", None, "declared", f"NOT RUN: {not_run}")
            return _with_model_rows(verify(ns, expected, continuation="skip", **kw), extra)
        # No fork can isolate model code here (CUDA initialized). The
        # model-executing rows run on ONE disposable in-process copy under a
        # forked RNG. A copy does not contain code (a hook's closure is shared
        # with the real objects), which is why this is safe only together
        # with the caller's before/after comparison of the whole state and
        # workspace: an escaped write is then a failed validation, not a
        # silent change. Declaring the rows instead would leave the step
        # untested exactly where the hardware changed.
        rows = verify(ns, expected, continuation="copy", **kw)
        for r in rows:
            if r.name in dict(MODEL_ROWS) and r.mode != "declared":
                r.detail += (f" (in this process on a disposable copy: no fork can isolate model code here, "
                             f"{reason}; the caller compares the state before and after)")
        return rows

    import torch

    from .fixture import _digest_params

    in_place = continuation == "inplace"
    out: list[OracleResult] = []
    # WHAT MAY BE HELD TO BITWISE EQUALITY, AND WHAT MAY NOT.
    #
    # STORED state is held to bitwise equality everywhere: parameter and buffer
    # bytes must be identical no matter which provider or SKU the capsule lands
    # on. That is the strong claim and it survives the whole matrix.
    #
    # RECOMPUTED state cannot be. The forward output is produced by running the
    # model again on the destination, and float reduction order depends on the
    # hardware: a different GPU architecture selects different kernels, and a
    # different CPU host vectorises differently. Measured: L4 -> H100 and
    # cpu -> (Modal CPU) both deviate, while cpu -> gpu_t4 happened to agree to
    # 0.00e+00 -- agreement that is luck, not a guarantee.
    #
    # `crossed` therefore keys on whether the ARITHMETIC SUBSTRATE changed, not
    # on whether the device class changed. Keying it on device class alone
    # demands bitwise equality of a recomputed value across two different pieces
    # of silicon, which fails for reasons of floating-point arithmetic rather
    # than of state preservation -- a manufactured failure in the one table that
    # is supposed to measure state preservation.
    crossed = not same_hardware

    def add(name, ok, mode, detail, measured=None):
        out.append(OracleResult(name, ok, mode, detail, measured))

    # -- plain values ----------------------------------------------------
    want = expected.get("values", {})
    bad = []
    for key, wv in want.items():
        if key == "arr_sum":
            got = repr(float(ns["arr"].sum())) if "arr" in ns else "<missing>"
        elif key == "co_dict":
            got = repr(sorted(ns["co_dict"].items(), key=repr)) if "co_dict" in ns else "<missing>"
        elif key == "sc_str":
            got = ns.get("sc_str", "<missing>")
        else:
            got = repr(ns.get(key, "<missing>"))
        if got != wv:
            bad.append(f"{key}: {got!r} != {wv!r}")
    add("values", not bad, "exact",
        f"{len(want)} scalar/container values exact" if not bad else "; ".join(bad[:4]))

    # -- shared reference, tested by mutation (in place) or identity ------
    # Two names over one dict: the object graph relationship is identity, and
    # identity is what the write probe observes. Without the write the check
    # is identity itself, which is exactly as strong for a plain dict and
    # leaves the object untouched.
    a, b = ns.get("shared_a"), ns.get("shared_b")
    if a is None or b is None:
        add("shared reference", False, "structural", "names missing")
    elif in_place:
        a["probe"] = 99
        add("shared reference", b.get("probe") == 99, "structural",
            "mutation through one name visible through the other"
            if b.get("probe") == 99 else "names no longer share an object")
        a.pop("probe", None)
    else:
        add("shared reference", a is b, "structural",
            "both names bound to one object (identity; nothing written)"
            if a is b else "names no longer share an object")

    # -- self-reference ---------------------------------------------------
    r = ns.get("recur")
    add("self-reference", r is not None and r.get("self") is r, "structural",
        "cycle preserved" if r is not None and r.get("self") is r else "cycle lost")

    model = ns.get("model")
    if model is None:
        add("model", False, "structural", "model missing from the namespace")
        return out

    # -- parameters: exact, always ---------------------------------------
    got_digest = _digest_params(model)
    want_digest = expected.get("param_digest")
    add("parameters", got_digest == want_digest, "exact",
        "parameter and buffer bytes identical" if got_digest == want_digest
        else f"digest {got_digest} != {want_digest}")

    # -- device: PRESERVED, except where the destination cannot host it ----
    # `destination_device` is the caller's expectation, not the destination's
    # hardware. The driver derives it as "the source device, unless cuda state
    # is landing on a CPU-only host, in which case cpu" -- see the long note at
    # its definition. Restoring a cpu namespace onto a GPU box leaves it on cpu,
    # and that is correct: relocation is the program's decision, not the
    # runtime's.
    actual = next(model.parameters()).device.type
    add("destination device", actual == destination_device, "structural",
        f"parameters are on {actual}, expected {destination_device}")

    # -- the objects the model-executing checks run on ---------------------
    # In place: the namespace's own objects. Copy: ONE deepcopy of all of
    # them together (see COPY_TOGETHER). If the copy cannot be made, the two
    # rows that need it fail with the reason rather than falling back to the
    # real objects, which would silently turn a non-mutating check into a
    # mutating one.
    work: dict[str, Any] | None = None
    work_error = None
    if in_place:
        work = ns
    elif continuation == "copy":
        import copy as _copy
        try:
            work = _copy.deepcopy({k: ns.get(k) for k in COPY_TOGETHER})
        except Exception as exc:  # noqa: BLE001
            work_error = f"disposable copy failed: {type(exc).__name__}: {exc}"
    rng_guard = contextlib.nullcontext if in_place else (lambda: _forked_rng(torch))

    # -- forward: exact within a device class, tolerance across -----------
    want_out = expected.get("forward_output")
    gap = expected.get("forward_output_gap")
    not_run = expected.get(FORWARD_NOT_RUN_KEY)
    if continuation == "skip":
        pass
    elif not_run:
        # Declared by the expectation (see the module doc): not a pass.
        add("forward output", None, "declared", f"NOT RUN: {not_run}")
    elif gap:
        add("forward output", None, "declared",
            f"NOT CHECKED: the source took no forward reference, because it could not run model code "
            f"outside its own process ({gap})")
    elif want_out is None or "train_x" not in ns:
        add("forward output", False, "structural", "no reference output recorded")
    elif work is None:
        add("forward output", False, "structural", work_error or "no model to run")
    else:
        fwd_model = work["model"]
        with rng_guard(), torch.no_grad():
            was = fwd_model.training
            fwd_model.eval()
            got_out = float(fwd_model(work["train_x"]).sum())
            fwd_model.train(was)
        delta = abs(got_out - want_out)
        rel = delta / max(abs(want_out), 1e-12)
        where = "" if in_place else ", on a disposable copy"
        if crossed:
            add("forward output", rel <= rtol, "tolerance",
                f"max relative deviation {rel:.2e} (limit {rtol:.0e}, "
                f"{source_device}->{destination_device}{where})", rel)
        else:
            add("forward output", delta == 0.0, "exact",
                ("bitwise identical" + where) if delta == 0.0
                else f"same device class but deviates by {delta:.3e}", rel)

    # -- optimizer: identity, references, and a real step -----------------
    opt = ns.get("optimizer")
    if opt is None:
        add("optimizer", False, "structural", "optimizer missing")
    else:
        # Identity only, on the REAL objects: a read, never a write.
        notes = []
        if not isinstance(opt, torch.optim.Optimizer):
            notes.append("not an instance of torch.optim.Optimizer")
        model_ids = {id(p) for p in model.parameters()}
        slot_ids = {id(p) for g in opt.param_groups for p in g["params"]}
        if not slot_ids or not slot_ids <= model_ids:
            notes.append("slots do not point at the live model parameters")
        add("optimizer", not notes, "structural",
            "isinstance holds and slots are bound to live parameters"
            if not notes else "; ".join(notes))

        # continuation: the step must actually advance state (on the copy,
        # when there is one: its success is judged there).
        if continuation == "skip":
            pass
        elif work is None:
            add("continuation", False, "structural", work_error or "no model to run")
        else:
            try:
                c_model, c_opt = work["model"], work["optimizer"]
                with rng_guard():
                    before = [p.detach().clone() for p in c_model.parameters()]
                    c_opt.zero_grad()
                    work["loss_fn"](c_model(work["train_x"]), work["train_y"]).backward()
                    c_opt.step()
                moved = any(not torch.equal(x, y) for x, y in zip(before, c_model.parameters()))
                key = list(c_model.parameters())[0]
                step_now = float(c_opt.state[key]["step"]) if key in c_opt.state else -1.0
                want_step = expected.get("step_count")
                ok = moved and (want_step is None or step_now == want_step + 1)
                where = "" if in_place else " (on a disposable copy; the namespace is untouched)"
                add("continuation", ok, "structural",
                    f"step advanced to {step_now} and parameters moved{where}" if ok
                    else f"moved={moved} step={step_now} expected={None if want_step is None else want_step + 1}{where}")
            except Exception as exc:
                add("continuation", False, "structural",
                    f"training step raised: {type(exc).__name__}: {exc}")

    # -- scheduler --------------------------------------------------------
    sched, want_lr = ns.get("scheduler"), expected.get("scheduler_lr")
    if sched is None or opt is None or want_lr is None:
        add("scheduler", False, "structural", "scheduler or reference missing")
    else:
        # The optimizer step above does not change lr; only scheduler.step() does.
        lr_ok = abs(float(opt.param_groups[0]["lr"]) - want_lr) < 1e-12
        ep_ok = int(sched.last_epoch) == expected.get("scheduler_epoch")
        add("scheduler", lr_ok and ep_ok, "exact",
            f"lr {want_lr} and last_epoch {sched.last_epoch} preserved"
            if lr_ok and ep_ok
            else f"lr={opt.param_groups[0]['lr']} epoch={sched.last_epoch}")

    # -- AMP scaler -------------------------------------------------------
    scaler = ns.get("scaler")
    add("AMP scaler", scaler is not None and hasattr(scaler, "state_dict"),
        "structural", "scaler present with readable state"
        if scaler is not None else "scaler missing")

    # -- data cursor: exact; the sampler generator is a CPU generator ------
    loader, want_idx = ns.get("loader"), expected.get("next_loader_indices")
    if loader is None or want_idx is None:
        add("data cursor", False, "structural", "loader or reference missing")
    else:
        try:
            if in_place:
                gen = loader.generator
                saved = gen.get_state().clone() if gen is not None else None
                got_idx = [int(i) for i in list(iter(loader.sampler))[:8]]
                if saved is not None:
                    gen.set_state(saved)
            else:
                # Read the next indices from a shallow copy of the sampler
                # that draws from a CLONE of its generator: the real
                # generator is never advanced, not even temporarily.
                import copy as _copy
                sampler = _copy.copy(loader.sampler)
                sg = getattr(loader.sampler, "generator", None)
                if sg is not None:
                    clone = torch.Generator(device=sg.device)
                    clone.set_state(sg.get_state())
                    sampler.generator = clone
                with _forked_rng(torch):
                    got_idx = [int(i) for i in list(iter(sampler))[:8]]
            add("data cursor", got_idx == want_idx, "exact",
                f"next indices match: {got_idx[:4]}..." if got_idx == want_idx
                else f"order changed: {got_idx[:4]} vs {want_idx[:4]}")
        except Exception as exc:
            add("data cursor", False, "structural",
                f"sampler raised: {type(exc).__name__}: {exc}")

    # -- workspace corpus: content and modes, re-read at the destination ---
    #
    # The namespace capsule and the workspace are two different transports:
    # the capsule goes through dill and the checkpoint store, the workspace
    # through the platform's workspace flush and its restore. A transition can carry one and
    # lose the other, and until this row existed the corpus was written,
    # transported and paid for on every trial without anything checking it —
    # so a total workspace loss would have been reported as a clean switch.
    #
    # Modes matter as much as bytes here. A restore that recreates every file
    # 0644 has lost information the source deliberately varied, and that is a
    # real portability limit rather than a cosmetic one.
    manifest = ns.get("workspace_manifest")
    if isinstance(manifest, dict) and manifest:
        import hashlib as _hl
        import os as _os

        root = ns.get("workspace_root")
        # The transported root is the SOURCE's absolute path. The destination
        # may mount the workspace at a different prefix, so fall back to the
        # same basename under the destination's cwd before concluding loss.
        candidates = [c for c in (root, _os.path.abspath(_os.path.basename(str(root or "data")))) if c]
        base = next((c for c in candidates if _os.path.isdir(c)), None)
        if base is None:
            add("workspace corpus", False, "structural",
                f"workspace root absent at destination (tried {candidates!r}): "
                f"{len(manifest)} files did not arrive")
        else:
            missing, wrong_bytes, wrong_mode = [], [], []
            for rel, meta in manifest.items():
                path = _os.path.join(base, rel)
                if not _os.path.isfile(path):
                    missing.append(rel)
                    continue
                with open(path, "rb") as fh:
                    blob = fh.read()
                if _hl.sha256(blob).hexdigest()[:32] != meta["sha256"] or len(blob) != meta["size"]:
                    wrong_bytes.append(rel)
                got_mode = oct(_os.stat(path).st_mode & 0o777)
                if got_mode != meta["mode"]:
                    wrong_mode.append(f"{rel}:{got_mode}!={meta['mode']}")
            ok = not (missing or wrong_bytes or wrong_mode)
            detail = (
                f"{len(manifest)} files, {sum(m['size'] for m in manifest.values()):,} bytes, "
                f"content and modes preserved"
                if ok else
                f"missing={len(missing)} corrupt={len(wrong_bytes)} "
                f"mode_changed={len(wrong_mode)}"
                + (f" e.g. {wrong_mode[0]}" if wrong_mode else "")
                + (f" e.g. {missing[0]}" if missing else "")
            )
            add("workspace corpus", ok, "exact", detail, measured=float(len(manifest)))

    return out


def summarize(results: list[OracleResult]) -> tuple[bool, str]:
    declared = [r for r in results if r.mode == "declared"]
    results = [r for r in results if r.mode != "declared"]
    failed = [r for r in results if not r.ok]
    tol = [r for r in results if r.ok and r.mode == "tolerance"]
    if failed:
        return False, "; ".join(f"{r.name}: {r.detail}" for r in failed[:4])
    note = f"{len(results)} oracles pass"
    if tol:
        note += f" ({len(tol)} within tolerance: " + \
                ", ".join(f"{r.name} {r.measured:.1e}" for r in tol) + ")"
    if declared:
        note += f"; {len(declared)} not run (declared): " + ", ".join(r.name for r in declared)
    return True, note
