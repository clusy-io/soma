"""Comparison of two witnesses into one row per correctness property.

Each function here answers one question about a transition and returns a single
row. The rules they share:

- A property that could not be examined returns NA, never PASS. The report
  counts NA rows separately, so a transition cannot look verified because a
  probe found nothing to compare.
- Values are held to bitwise identity wherever bitwise identity is physically
  achievable. Moving a tensor between devices copies bits; it does not change
  them, so a parameter that differs after a device move has been lost, not
  rounded, and the row says FAIL.
- Where bitwise identity is not achievable, the row is judged against a
  declared tolerance and is labelled ``tolerance`` in the report, so a weaker
  pass is never displayed as a stronger one. Continuation across a device
  change is the only property in this file that gets that treatment, because
  a CUDA reduction and a CPU reduction genuinely disagree in the last bits.
"""

from __future__ import annotations

from typing import Any, Callable

from .aliasgraph import diff_partitions
from .status import CheckResult, Mode, Status
from .witness import Witness, alias_snapshot_from_witness

#: Relative deviation permitted on a continuation that crosses a device change.
DEFAULT_CROSS_DEVICE_RTOL = 2e-3


def _crossed_device(before: Witness, after: Witness) -> bool:
    return before.host.device_class != after.host.device_class


def _compare_records(
    before: dict[str, Any], after: dict[str, Any], label: str
) -> tuple[list[str], dict[str, Any]]:
    """Compare two name -> TensorRecord maps, returning offenders and counts."""
    offenders: list[str] = []
    missing = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    changed: list[str] = []

    for name in sorted(set(before) & set(after)):
        if not before[name].value_equal(after[name]):
            changed.append(name)

    offenders.extend(f"{n} (missing after)" for n in missing)
    offenders.extend(f"{n} (appeared after)" for n in added)
    offenders.extend(f"{n} (value changed)" for n in changed)

    metrics = {
        f"{label}_before": len(before),
        f"{label}_after": len(after),
        f"{label}_missing": len(missing),
        f"{label}_added": len(added),
        f"{label}_changed": len(changed),
    }
    return offenders, metrics


def _module_tensor_map(w: Witness, kind: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for var, cap in w.modules.items():
        source = cap.parameters if kind == "parameters" else cap.buffers
        for name, rec in source.items():
            out[f"{var}.{name}"] = rec
    return out


def check_parameters(before: Witness, after: Witness) -> CheckResult:
    b = _module_tensor_map(before, "parameters")
    a = _module_tensor_map(after, "parameters")
    if not b and not a:
        return CheckResult("parameters", Status.NA, detail="no torch modules in namespace")
    offenders, metrics = _compare_records(b, a, "parameters")
    status = Status.PASS if not offenders else Status.FAIL
    detail = f"{len(b)} parameters" if status is Status.PASS else f"{len(offenders)} differ"
    return CheckResult("parameters", status, Mode.EXACT, detail, metrics, offenders)


def check_buffers(before: Witness, after: Witness) -> CheckResult:
    b = _module_tensor_map(before, "buffers")
    a = _module_tensor_map(after, "buffers")
    if not b and not a:
        return CheckResult(
            "buffers", Status.NA,
            detail="no non-parameter buffers (no BatchNorm-style running stats)",
        )
    offenders, metrics = _compare_records(b, a, "buffers")
    status = Status.PASS if not offenders else Status.FAIL
    detail = f"{len(b)} buffers" if status is Status.PASS else f"{len(offenders)} differ"
    return CheckResult("buffers", status, Mode.EXACT, detail, metrics, offenders)


def check_optimizer_slots(before: Witness, after: Witness) -> CheckResult:
    """Momentum, second-moment and step counters carried by the optimizer.

    Losing these does not stop training; it silently restarts the adaptive
    estimates, which shows up as a transient loss spike that is easy to blame on
    the data.
    """
    if not before.optimizers and not after.optimizers:
        return CheckResult("optimizer slots", Status.NA, detail="no optimizer in namespace")

    b_tensors: dict[str, Any] = {}
    a_tensors: dict[str, Any] = {}
    b_scalars: dict[str, Any] = {}
    a_scalars: dict[str, Any] = {}
    for var, cap in before.optimizers.items():
        b_tensors.update({f"{var}:{k}": v for k, v in cap.slots.items()})
        b_scalars.update({f"{var}:{k}": v for k, v in cap.scalar_slots.items()})
    for var, cap in after.optimizers.items():
        a_tensors.update({f"{var}:{k}": v for k, v in cap.slots.items()})
        a_scalars.update({f"{var}:{k}": v for k, v in cap.scalar_slots.items()})

    offenders, metrics = _compare_records(b_tensors, a_tensors, "slots")

    for name in sorted(set(b_scalars) | set(a_scalars)):
        if name not in a_scalars:
            offenders.append(f"{name} (missing after)")
        elif name not in b_scalars:
            offenders.append(f"{name} (appeared after)")
        elif b_scalars[name] != a_scalars[name]:
            offenders.append(f"{name} ({b_scalars[name]!r} -> {a_scalars[name]!r})")
    metrics["scalar_slots"] = len(b_scalars)

    if not b_tensors and not b_scalars and not a_tensors and not a_scalars:
        return CheckResult(
            "optimizer slots", Status.NA,
            detail="optimizer present but holds no state yet (no step taken)",
            metrics=metrics,
        )

    status = Status.PASS if not offenders else Status.FAIL
    detail = (
        f"{len(b_tensors)} tensor slots, {len(b_scalars)} scalar"
        if status is Status.PASS else f"{len(offenders)} differ"
    )
    return CheckResult("optimizer slots", status, Mode.EXACT, detail, metrics, offenders)


def check_optimizer_refs(before: Witness, after: Witness) -> CheckResult:
    """Which model parameter each optimizer slot will write into.

    This is the row that catches the failure where every value compares equal
    and training still runs, but the optimizer updates tensors the model no
    longer uses, so the model never improves.
    """
    if not before.optimizers and not after.optimizers:
        return CheckResult("optimizer refs", Status.NA, detail="no optimizer in namespace")

    offenders: list[str] = []
    unbound_after = 0
    for var in sorted(set(before.optimizers) | set(after.optimizers)):
        b = before.optimizers.get(var)
        a = after.optimizers.get(var)
        if b is None:
            offenders.append(f"{var} (optimizer appeared after)")
            continue
        if a is None:
            offenders.append(f"{var} (optimizer missing after)")
            continue
        unbound_after += len(a.unbound)
        for key in sorted(set(b.param_refs) | set(a.param_refs)):
            bt, at = b.param_refs.get(key), a.param_refs.get(key)
            if bt != at:
                offenders.append(f"{var}:{key} points at {at!r}, was {bt!r}")
        if b.groups != a.groups:
            for gi, (bg, ag) in enumerate(zip(b.groups, a.groups)):
                for k in sorted(set(bg) | set(ag)):
                    if bg.get(k) != ag.get(k):
                        offenders.append(f"{var}:group{gi}.{k} {bg.get(k)!r} -> {ag.get(k)!r}")
            if len(b.groups) != len(a.groups):
                offenders.append(
                    f"{var}: {len(b.groups)} param groups before, {len(a.groups)} after"
                )

    metrics = {
        "tracked_slots": sum(len(c.param_refs) for c in before.optimizers.values()),
        "unbound_after": unbound_after,
    }
    if unbound_after:
        offenders.append(
            f"{unbound_after} optimizer slots point at tensors no live module owns"
        )
    status = Status.PASS if not offenders else Status.FAIL
    detail = (
        f"{metrics['tracked_slots']} slots bound to live parameters"
        if status is Status.PASS else f"{len(offenders)} broken"
    )
    return CheckResult("optimizer refs", status, Mode.STRUCTURAL, detail, metrics, offenders)


def _check_state_dicts(
    before: dict[str, Any], after: dict[str, Any], row: str, absent_detail: str
) -> CheckResult:
    if not before and not after:
        return CheckResult(row, Status.NA, detail=absent_detail)
    offenders: list[str] = []
    for var in sorted(set(before) | set(after)):
        b, a = before.get(var), after.get(var)
        if b is None:
            offenders.append(f"{var} (appeared after)")
            continue
        if a is None:
            offenders.append(f"{var} (missing after)")
            continue
        for k in sorted(set(b.entries) | set(a.entries)):
            if b.entries.get(k) != a.entries.get(k):
                offenders.append(f"{var}.{k}: {b.entries.get(k)!r} -> {a.entries.get(k)!r}")
    status = Status.PASS if not offenders else Status.FAIL
    kinds = ", ".join(sorted({c.kind for c in before.values()})) or "none"
    detail = kinds if status is Status.PASS else f"{len(offenders)} differ"
    return CheckResult(row, status, Mode.EXACT, detail, {"objects": len(before)}, offenders)


def check_scheduler(before: Witness, after: Witness) -> CheckResult:
    return _check_state_dicts(
        before.schedulers, after.schedulers, "scheduler",
        "no LR scheduler in namespace",
    )


def check_amp_scaler(before: Witness, after: Witness) -> CheckResult:
    return _check_state_dicts(
        before.scalers, after.scalers, "AMP scaler",
        "no GradScaler in namespace (run is not using AMP)",
    )


def _rng_row(
    row: str,
    before: Witness,
    after: Witness,
    pick: Callable[[Any], Any],
    *,
    carried_only_when_unavailable: bool = False,
) -> CheckResult:
    if before.rng is None or after.rng is None:
        return CheckResult(row, Status.NA, detail="no RNG capture in witness")
    b, a = pick(before.rng), pick(after.rng)

    if not b.available and not a.available:
        return CheckResult(row, Status.NA, detail=b.note or a.note or "stream unavailable")

    if b.available and not a.available:
        if carried_only_when_unavailable:
            # A CUDA generator cannot be installed on a host with no device.
            # Carrying the bytes intact is the strongest available guarantee,
            # and calling it anything else would overstate the result.
            if b.state_digest and b.state_digest == (a.state_digest or ""):
                return CheckResult(
                    row, Status.PASS, Mode.STRUCTURAL,
                    "carried, not installed (no device on destination)",
                    {"carried": True},
                )
            return CheckResult(
                row, Status.FAIL, Mode.STRUCTURAL,
                "dropped: destination has no device and did not carry the state",
                {"carried": False},
            )
        return CheckResult(
            row, Status.FAIL, Mode.EXACT,
            f"stream available before, unavailable after ({a.note})",
        )

    if a.available and not b.available:
        return CheckResult(
            row, Status.NA, Mode.NONE,
            f"stream unavailable before the boundary ({b.note})",
        )

    state_same = b.state_digest == a.state_digest
    draws_same = b.forward_draws == a.forward_draws
    metrics = {"state_digest_equal": state_same, "forward_draws_equal": draws_same}

    if state_same and draws_same:
        return CheckResult(row, Status.PASS, Mode.EXACT, "state and next draws identical", metrics)
    if draws_same and not state_same:
        # Equivalent position, different encoding. The observable behaviour is
        # what matters, so this passes, but the discrepancy is worth recording.
        return CheckResult(
            row, Status.PASS, Mode.EXACT,
            "next draws identical; state encoding differs", metrics,
        )
    if state_same and not draws_same:
        return CheckResult(
            row, Status.FAIL, Mode.EXACT,
            "state bytes match but next draws diverge", metrics,
        )
    return CheckResult(row, Status.FAIL, Mode.EXACT, "generator reset or advanced", metrics)


def check_python_rng(before: Witness, after: Witness) -> CheckResult:
    return _rng_row("Python RNG", before, after, lambda r: r.python)


def check_numpy_rng(before: Witness, after: Witness) -> CheckResult:
    row = _rng_row("NumPy RNG", before, after, lambda r: r.numpy_legacy)
    # The legacy global is often untouched while an explicit Generator does the
    # real work, so a pass on the global alone would be misleading.
    b_gens = before.rng.generators if before.rng else {}
    a_gens = after.rng.generators if after.rng else {}
    if b_gens or a_gens:
        offenders = [
            f"{n}: {b_gens.get(n)!r} -> {a_gens.get(n)!r}"
            for n in sorted(set(b_gens) | set(a_gens))
            if b_gens.get(n) != a_gens.get(n)
        ]
        row.metrics["named_generators"] = len(b_gens)
        if offenders:
            row.status = Status.FAIL
            row.offenders.extend(offenders)
            row.detail = f"{len(offenders)} named Generator(s) diverged"
        elif row.status is Status.NA:
            row.status = Status.PASS
            row.mode = Mode.EXACT
            row.detail = f"{len(b_gens)} named Generator(s) identical; global unused"
        else:
            row.detail += f"; {len(b_gens)} named Generator(s) identical"
    return row


def check_torch_cpu_rng(before: Witness, after: Witness) -> CheckResult:
    return _rng_row("Torch CPU RNG", before, after, lambda r: r.torch_cpu)


def check_torch_cuda_rng(before: Witness, after: Witness) -> CheckResult:
    return _rng_row(
        "Torch CUDA RNG", before, after, lambda r: r.torch_cuda,
        carried_only_when_unavailable=True,
    )


def check_data_cursor(before: Witness, after: Witness) -> CheckResult:
    """Where the loader is positioned and what order it will emit next."""
    if not before.loaders and not after.loaders:
        return CheckResult("data cursor", Status.NA, detail="no DataLoader in namespace")

    offenders: list[str] = []
    probed = 0
    unprobed: list[str] = []
    for var in sorted(set(before.loaders) | set(after.loaders)):
        b, a = before.loaders.get(var), after.loaders.get(var)
        if b is None:
            offenders.append(f"{var} (loader appeared after)")
            continue
        if a is None:
            offenders.append(f"{var} (loader missing after)")
            continue
        for field_name in ("batch_size", "drop_last", "dataset_len", "sampler_kind"):
            if getattr(b, field_name) != getattr(a, field_name):
                offenders.append(
                    f"{var}.{field_name}: {getattr(b, field_name)!r} -> "
                    f"{getattr(a, field_name)!r}"
                )
        if b.next_indices is None or a.next_indices is None:
            unprobed.append(var)
        else:
            probed += 1
            if b.next_indices != a.next_indices:
                offenders.append(
                    f"{var}: next indices {b.next_indices[:6]} -> {a.next_indices[:6]}"
                )
        if b.generator_digest != a.generator_digest:
            offenders.append(f"{var}: sampler generator state changed")

    metrics = {"loaders": len(before.loaders), "order_probed": probed}
    if offenders:
        return CheckResult(
            "data cursor", Status.FAIL, Mode.EXACT,
            f"{len(offenders)} differ", metrics, offenders,
        )
    if probed == 0:
        return CheckResult(
            "data cursor", Status.NA, Mode.NONE,
            f"loaders present but sampling order not probeable ({', '.join(unprobed)})",
            metrics,
        )
    detail = f"{probed} loader(s), next-index order identical"
    if unprobed:
        detail += f"; {len(unprobed)} not probeable"
    return CheckResult("data cursor", Status.PASS, Mode.EXACT, detail, metrics)


def check_alias_graph(before: Witness, after: Witness) -> CheckResult:
    """Whether variables that shared storage still share it."""
    b = alias_snapshot_from_witness(before)
    a = alias_snapshot_from_witness(after)
    if not b.partition and not a.partition:
        return CheckResult("alias graph", Status.NA, detail="empty namespace")

    splits, merges = diff_partitions(b, a)
    truncated = sorted(set(before.alias_truncated) | set(after.alias_truncated))
    metrics = {
        "components_before": len(b.partition),
        "components_after": len(a.partition),
        "splits": len(splits),
        "merges": len(merges),
        "truncated": len(truncated),
    }
    offenders = splits + merges

    if offenders:
        return CheckResult(
            "alias graph", Status.FAIL, Mode.STRUCTURAL,
            f"{len(splits)} split, {len(merges)} merged", metrics, offenders,
        )
    if truncated:
        # The partition matched, but part of it was derived from a bounded walk,
        # so it is not evidence of the full property.
        return CheckResult(
            "alias graph", Status.NA, Mode.STRUCTURAL,
            f"partition matched but {len(truncated)} name(s) hit the traversal "
            f"budget: {', '.join(truncated[:4])}",
            metrics, truncated,
        )
    return CheckResult(
        "alias graph", Status.PASS, Mode.STRUCTURAL,
        f"{len(b.partition)} components preserved", metrics,
    )


def check_workspace_hashes(before: Witness, after: Witness) -> CheckResult:
    if not before.workspace and not after.workspace:
        return CheckResult(
            "workspace hashes", Status.NA,
            detail="no workspace root given (pass --workspace to enable)",
        )
    offenders: list[str] = []
    missing = sorted(set(before.workspace) - set(after.workspace))
    added = sorted(set(after.workspace) - set(before.workspace))
    changed = sorted(
        p for p in set(before.workspace) & set(after.workspace)
        if before.workspace[p] != after.workspace[p]
    )
    offenders.extend(f"{p} (missing after)" for p in missing)
    offenders.extend(f"{p} (appeared after)" for p in added)
    offenders.extend(f"{p} (contents changed)" for p in changed)
    metrics = {
        "files_before": len(before.workspace),
        "files_after": len(after.workspace),
        "missing": len(missing),
        "added": len(added),
        "changed": len(changed),
    }
    status = Status.PASS if not offenders else Status.FAIL
    detail = (
        f"{len(before.workspace)} files identical"
        if status is Status.PASS else f"{len(offenders)} differ"
    )
    return CheckResult("workspace hashes", status, Mode.EXACT, detail, metrics, offenders)


def check_continuation(
    before: Witness, after: Witness, *, rtol: float = DEFAULT_CROSS_DEVICE_RTOL
) -> CheckResult:
    """Whether training continues along the same trajectory after the boundary.

    Every other row inspects state at rest. This one runs the system: a fixed
    number of further steps is taken on each side from the same point, and the
    loss sequences are compared. It is the only row that can catch a boundary
    that restored every value correctly and still changed what happens next.
    """
    b = before.continuation
    a = after.continuation
    if not b or not a:
        return CheckResult(
            "continuation", Status.NA,
            detail="no continuation recorded (needs a step function; see --continuation)",
        )
    if b.get("error") or a.get("error"):
        return CheckResult(
            "continuation", Status.ERROR,
            detail=f"before={b.get('error')!r} after={a.get('error')!r}",
        )

    bl = [float(x) for x in b.get("losses", [])]
    al = [float(x) for x in a.get("losses", [])]
    if not bl or not al:
        return CheckResult("continuation", Status.NA, detail="continuation recorded no losses")
    if len(bl) != len(al):
        return CheckResult(
            "continuation", Status.FAIL, Mode.NONE,
            f"{len(bl)} steps before, {len(al)} after",
            {"steps_before": len(bl), "steps_after": len(al)},
        )

    deltas = [abs(x - y) for x, y in zip(bl, al)]
    rels = [
        d / max(abs(x), 1e-12) for d, x in zip(deltas, bl)
    ]
    max_abs = max(deltas)
    max_rel = max(rels)
    metrics = {
        "steps": len(bl),
        "max_abs_delta": max_abs,
        "max_rel_delta": max_rel,
        "first_loss_before": bl[0],
        "first_loss_after": al[0],
        "last_loss_before": bl[-1],
        "last_loss_after": al[-1],
    }

    if max_abs == 0.0:
        return CheckResult(
            "continuation", Status.PASS, Mode.EXACT,
            f"{len(bl)} steps, losses bitwise identical", metrics,
        )

    if _crossed_device(before, after):
        # A CUDA reduction and a CPU reduction do not agree in the last bits,
        # so identity is not the right standard here. The measured deviation is
        # reported so the claim is checkable rather than asserted.
        if max_rel <= rtol:
            return CheckResult(
                "continuation", Status.PASS, Mode.TOLERANCE,
                f"{len(bl)} steps, max relative deviation {max_rel:.2e} "
                f"(limit {rtol:.0e}, device change {before.host.device_class}"
                f"->{after.host.device_class})",
                metrics,
            )
        return CheckResult(
            "continuation", Status.FAIL, Mode.TOLERANCE,
            f"max relative deviation {max_rel:.2e} exceeds {rtol:.0e}", metrics,
        )

    return CheckResult(
        "continuation", Status.FAIL, Mode.EXACT,
        f"same device class but losses diverge: max abs {max_abs:.3e}", metrics,
    )


def check_variables(before: Witness, after: Witness) -> CheckResult:
    """Top-level names present on each side.

    A dropped dataframe belongs to no category above, so without this row a
    namespace could lose it and still report a clean table.
    """
    missing = sorted(set(before.names) - set(after.names))
    added = sorted(set(after.names) - set(before.names))
    metrics = {
        "before": len(before.names),
        "after": len(after.names),
        "missing": len(missing),
        "added": len(added),
    }
    offenders = [f"{n} (missing after)" for n in missing]
    offenders += [f"{n} (appeared after)" for n in added]
    if not before.names and not after.names:
        return CheckResult("variables", Status.NA, detail="empty namespace")
    status = Status.PASS if not offenders else Status.FAIL
    detail = (
        f"{len(before.names)} names preserved"
        if status is Status.PASS else f"{len(missing)} missing, {len(added)} added"
    )
    return CheckResult("variables", status, Mode.STRUCTURAL, detail, metrics, offenders)


def check_class_identity(before: Witness, after: Witness) -> CheckResult:
    """Whether restored objects are still instances of the live classes.

    A serializer that writes a class by value rebuilds a duplicate class object
    on load. The restored object then has every correct value and every correct
    method and is not an instance of the class the rest of the process uses, so
    ``isinstance`` guards in framework code take the wrong branch while nothing
    raises. No value comparison can detect this, because the values are right.
    """
    b, a = before.class_identity, after.class_identity
    if not b and not a:
        return CheckResult("class identity", Status.NA, detail="no class identity captured")

    offenders: list[str] = []
    demoted = 0
    for name in sorted(set(b) & set(a)):
        be, ae = b[name], a[name]
        if be.get("type_path") != ae.get("type_path"):
            offenders.append(
                f"{name}: {be.get('type_path')} -> {ae.get('type_path')}"
            )
            continue
        if be.get("by_reference") and not ae.get("by_reference"):
            demoted += 1
            offenders.append(
                f"{name}: {be.get('type_path')} was restored by value; "
                f"it is no longer the live class object"
            )
        lost = set(be.get("isa", [])) - set(ae.get("isa", []))
        if lost:
            offenders.append(
                f"{name}: no longer an instance of {', '.join(sorted(lost))}"
            )

    metrics = {
        "names": len(b),
        "by_reference_before": sum(1 for e in b.values() if e.get("by_reference")),
        "by_reference_after": sum(1 for e in a.values() if e.get("by_reference")),
        "demoted_to_by_value": demoted,
    }
    status = Status.PASS if not offenders else Status.FAIL
    detail = (
        f"{metrics['by_reference_before']} of {len(b)} classes resolve by reference, unchanged"
        if status is Status.PASS else f"{len(offenders)} object(s) changed class identity"
    )
    return CheckResult("class identity", status, Mode.STRUCTURAL, detail, metrics, offenders)


def check_module_modes(before: Witness, after: Witness) -> CheckResult:
    """train() versus eval(), which silently changes dropout and BatchNorm."""
    if not before.modules and not after.modules:
        return CheckResult("module modes", Status.NA, detail="no torch modules in namespace")
    offenders = [
        f"{v}: {'train' if before.modules[v].training else 'eval'} -> "
        f"{'train' if after.modules[v].training else 'eval'}"
        for v in sorted(set(before.modules) & set(after.modules))
        if before.modules[v].training != after.modules[v].training
    ]
    status = Status.PASS if not offenders else Status.FAIL
    return CheckResult(
        "module modes", status, Mode.STRUCTURAL,
        f"{len(before.modules)} module(s)" if status is Status.PASS else f"{len(offenders)} changed",
        {"modules": len(before.modules)}, offenders,
    )


#: The canonical report, in order. Rows above the separator are the properties
#: the paper commits to; rows below are additional evidence.
CANONICAL_CHECKS: list[Callable[[Witness, Witness], CheckResult]] = [
    check_parameters,
    check_buffers,
    check_optimizer_slots,
    check_optimizer_refs,
    check_scheduler,
    check_amp_scaler,
    check_python_rng,
    check_numpy_rng,
    check_torch_cpu_rng,
    check_torch_cuda_rng,
    check_data_cursor,
    check_alias_graph,
    check_workspace_hashes,
    check_continuation,
]

SUPPLEMENTARY_CHECKS: list[Callable[[Witness, Witness], CheckResult]] = [
    check_variables,
    check_class_identity,
    check_module_modes,
]


def run_all(
    before: Witness, after: Witness, *, rtol: float = DEFAULT_CROSS_DEVICE_RTOL
) -> tuple[list[CheckResult], list[CheckResult]]:
    def run(fn: Callable[[Witness, Witness], CheckResult]) -> CheckResult:
        try:
            if fn is check_continuation:
                return check_continuation(before, after, rtol=rtol)
            return fn(before, after)
        except Exception as exc:  # a broken probe must not look like a pass
            name = fn.__name__.removeprefix("check_").replace("_", " ")
            return CheckResult(name, Status.ERROR, detail=f"{type(exc).__name__}: {exc}")

    return [run(f) for f in CANONICAL_CHECKS], [run(f) for f in SUPPLEMENTARY_CHECKS]
