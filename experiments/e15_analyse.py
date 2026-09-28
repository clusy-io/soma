"""E15 analysis: a PASS/FAIL table per commit boundary and per check, and the
lifecycle's trajectory.

Reads the records `e15_combined.py` writes (`e15_records.jsonl`, schema
`e15/1`, one line per stage: setup, block0, hop1, block1, hop2a, hop2b,
block2, final, and `stopped` if the lifecycle ended early). The harness calls
`judge` itself at the end of a run, so the table printed there and the one
printed here are the same function over the same records.

Verdicts are PASS, FAIL, SKIP (a check that needs CUDA, on a `--local` dry run,
where every profile is the laptop CPU; it is required on a live run) and NOTE
(a fact the reader needs that is not a pass or a failure, such as the scope of
the gradient claim). A boundary the lifecycle never reached FAILS its checks
rather than disappearing from the table. The run is judged OK only when
nothing FAILS, and a run in which the harness re-bound the fixture's helpers
on a restored runtime (`--reimport-cell fixture`) always FAILS "the controller
ran unaided": its result is a diagnostic of the rest of the lifecycle, never
evidence about the controller alone.

    python experiments/e15_analyse.py                                  # results/e15, latest cohort and run
    python experiments/e15_analyse.py --records /tmp/e15/e15_records.jsonl --cohort C --run-id R
    python experiments/e15_analyse.py --records X --selftest            # mutate the records, show checks fail

`--selftest` is how the checks are shown to be able to fail without spending
a sandbox: each mutation breaks one property in a copy of real records (an
acknowledged write missing from the log, a boundary verdict flipped, the remap
count zeroed, an alias broken, the abort that completed, ...) and the named
check must turn FAIL while the unmutated records stay OK. It needs the records
of a COMPLETE lifecycle. Until the controller's expectation refresh is fixed,
only the `--reimport-cell fixture` diagnostic completes, and its records FAIL
exactly one row ("the controller ran unaided"); the self-test accepts that one
known failure in the unmutated records and nothing else.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "e15/1"
DEFAULT_RECORDS = ROOT / "results" / "e15" / "e15_records.jsonl"
#: Relative tolerance for a recomputed fixed-batch loss across a hop: the
#: controller's own cross-device forward tolerance (oracles.CROSS_DEVICE_RTOL).
LOSS_RTOL = 1e-4
NO_CUDA = "local dry run: LocalApi has no CUDA, so this is exercised live only"
ABORT_REASON = {"bad_capsule": "checkpoint_transfer_failed", "missing_package": "RESTORE_MISSING_MODULE"}
SECTIONS = {
    "run": "run",
    "hop1": "hop 1   cpu -> gpu, writer throughout (commit)",
    "hop2a": "hop 2a  gpu -> cpu, injected fault (abort)",
    "hop2b": "hop 2b  gpu -> cpu, controller killed and resumed (commit)",
    "trajectory": "trajectory",
}
#: The run-level row that FAILS whenever the harness did part of the
#: controller's work (see `judge`). The self-test tolerates it, and only it, in
#: the unmutated records of a diagnostic run.
UNAIDED = "the controller ran unaided (the harness re-bound nothing on a restored runtime)"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(path: Path, cohort: str | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
    """The records of ONE run: `run_id`, else the latest run of `cohort`,
    else the latest run in the file."""
    with open(path) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    rows = [r for r in rows if r.get("schema") == SCHEMA]
    if cohort is not None:
        rows = [r for r in rows if r.get("cohort") == cohort]
    if not rows:
        return []
    rid = run_id or (rows[-1].get("run") or {}).get("run_id")
    return [r for r in rows if (r.get("run") or {}).get("run_id") == rid]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

class Table:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []

    def add(self, section: str, check: str, ok: bool | None, detail: str = "", *, skip: str | None = None,
            note: bool = False) -> None:
        verdict = "NOTE" if note else ("SKIP" if skip else ("PASS" if ok else "FAIL"))
        self.rows.append({"section": SECTIONS.get(section, section), "check": check, "verdict": verdict,
                          "detail": (skip if skip else detail)[:400]})

    def section(self, section: str):
        return lambda check, ok=None, detail="", **kw: self.add(section, check, ok, detail, **kw)


def _get(d: Any, *path, default=None):
    for p in path:
        if not isinstance(d, dict):
            return default
        d = d.get(p)
    return default if d is None else d


def _fp_views_ok(fp: dict[str, Any], expected: int) -> tuple[bool, str]:
    v = (fp or {}).get("views") or {}
    ok = (v.get("available") is True and int(v.get("checked", 0)) >= expected
          and v.get("intact") == v.get("checked") and not v.get("broken"))
    return ok, f"checked {v.get('checked')} intact {v.get('intact')} broken {v.get('broken')}"


def _write_checks(add, h: dict[str, Any], hop: str) -> None:
    """Every acknowledged write present once, every refused one absent, on the
    runtime the probe read after the hop (the destination after a commit)."""
    w = h.get("writes") or {}
    acked, refused, errors = w.get("acked_so_far") or [], w.get("refused_so_far") or [], w.get("errors_so_far") or []
    mine = w.get("entries") or []
    pw = _get(h, "probe_post", "writes", default={})
    log, ns = pw.get("log"), pw.get("acks")
    if log is None or ns is None:
        add("every acknowledged write present exactly once", False, "no post-hop probe of the writes")
        add("every refused write absent", False, "no post-hop probe of the writes")
    else:
        missing = sorted(set(acked) - set(log))
        extra = sorted(set(log) - set(acked))
        dup = len(log) != len(set(log))
        add("every acknowledged write present exactly once",
            not missing and not extra and not dup and ns == log and pw.get("counter") == len(acked),
            f"{len(acked)} acknowledged so far; log {len(log)} lines, namespace {len(ns)}, counter {pw.get('counter')}"
            + (f"; MISSING {missing[:8]}" if missing else "") + (f"; UNACKNOWLEDGED {extra[:8]}" if extra else "")
            + ("; duplicates" if dup else "") + ("" if ns == log else "; namespace list and log differ"))
        hit = sorted(set(refused) & (set(log) | set(ns)))
        add("every refused write absent", not hit, f"{len(refused)} refused so far" + (f"; PRESENT {hit[:8]}" if hit else ""))
    add("no indeterminate write (every write acknowledged or refused)", not errors,
        f"errors {errors[:5]}" if errors else f"{len(mine)} writes in this hop")
    src, dest = h.get("source_pid"), h.get("dest_pid")
    acks = [e for e in mine if e.get("outcome") == "ack"]
    n_ref = sum(1 for e in mine if e.get("outcome") == "refused")
    add("the writer met the closed gate (refusals exercised)", n_ref >= 1,
        f"{len(acks)} acknowledged, {n_ref} refused during the hop")
    off = [e["seq"] for e in acks if e.get("runtime") not in (src, dest)]
    add("every write of the hop ran on its source or its destination", not off, f"elsewhere: {off[:8]}" if off else "")
    at = h.get("committed_at")
    on_dest = [e for e in acks if e.get("runtime") == dest]
    early = [e["seq"] for e in on_dest if at is None or e["t_done"] < at]
    add("writes after the commit ran on the destination", len(on_dest) >= 1 and not early,
        f"{len(on_dest)} on the destination after COMMITTED" + (f"; before it: {early[:5]}" if early else ""))


def _expectation_check(add, h: dict[str, Any], cap: dict[str, Any]) -> None:
    """The expectation the destination is validated against must be the one
    refreshed AT THE CUT and must describe the source there. The capture
    report's `expectation_source` alone is not enough: where the refresh
    cannot fork it reports "cut" even when the digest helper was missing and
    `param_digest` came back None, and validation then fails on the
    parameters with the cause out of sight. So the capsule's own expectation
    is compared with the harness's probe of the source (same digest
    algorithm, same step count). It must carry NO forward reference: the
    controller runs no model code on the source, and says so."""
    e = h.get("expectation") or {}
    pre = h.get("probe_pre") or {}
    want = pre.get("fixture_param_digest")
    digest = e.get("param_digest")
    digest_txt = ("MISSING (None)" if not digest else "matches the source" if digest == want
                  else f"differs from the source ({str(digest)[:8]} vs {str(want)[:8]})")
    # The controller no longer runs the model on the source at the cut (a
    # forward there runs user hooks on the runtime the user still owns), so
    # the expectation carries NO forward reference and says why. A reference
    # present here means the source ran model code, which is the defect.
    fwd = e.get("forward_output")
    not_run = e.get("forward_output_not_run")
    method = e.get("method") or {}
    ran_model = fwd is not None or method.get("model_code_run") is True
    fwd_txt = ("PRESENT: the source ran the model" if ran_model
               else "declared not run" if not_run else "absent WITHOUT a declared reason")
    add("validation expectation refreshed at the cut and describing the source, no model code run there",
        cap.get("expectation_source") == "cut" and e.get("source") == "cut" and not e.get("error")
        and bool(digest) and digest == want and e.get("step_count") == pre.get("step_count")
        and not ran_model and bool(not_run),
        (f"error {e['error']}" if e.get("error") else
         f"report {cap.get('expectation_source')!r}, capsule {e.get('source')!r}; param_digest {digest_txt}; "
         f"step_count {e.get('step_count')} vs source {pre.get('step_count')}; forward {fwd_txt}"))


def _probe_checks(add, pre: dict[str, Any] | None, post: dict[str, Any] | None, *, where: str) -> None:
    """The harness's own read of the state on `where` (see the probe)."""
    if not post:
        add(f"harness probe ran on {where}", False, "no probe")
        return
    idt = post.get("identity") or {}
    add("optimizer params are the model's Parameters (identity)", idt.get("optimizer_params_are_model_params") is True)
    add("optimizer state is non-empty and keyed by those Parameters",
        idt.get("optimizer_state_nonempty") is True and idt.get("state_keys_are_model_params") is True)
    add("external alias is the model weight", idt.get("alias_is_model_weight") is True)
    add("repeated references stay one object (torch and numpy)",
        idt.get("repeated_view_is_one_object") is True and idt.get("numpy_repeated_view_is_one_object") is True)
    cls = post.get("classes") or {}
    want = {"model.fc1.weight": ["Parameter", True, True], "model.fc2.weight": ["Parameter", True, True],
            "e15_top": ["Parameter", True, True], "e15_tail": ["Tensor", False, True]}
    got = {k: [v.get("class"), v.get("requires_grad"), v.get("is_leaf")] for k, v in cls.items()}
    add("Parameter classes, requires_grad and leaf-ness kept",
        post.get("model_params_are_parameters") is True and got == want,
        "" if got == want else f"got {got}")
    tv, nv = post.get("torch_views") or {}, post.get("numpy_views") or {}
    bad = [k for k, v in tv.items() if not (v.get("same_storage") and v.get("offset") == v.get("expected_offset"))]
    bad += [k for k, v in nv.items() if not (v.get("shares") and v.get("byte_offset") == v.get("expected_offset"))]
    add("every view shares its base at its offset (pointers, read-only)", len(tv) == 4 and len(nv) == 3 and not bad,
        f"broken: {bad}" if bad else f"{len(tv)} torch views on {next(iter(tv.values()), {}).get('base_device')}, "
                                      f"{len(nv)} numpy views")
    beh = post.get("behavioural") or {}
    rv = beh.get("repeated_view_write_through_copied_base") or {}
    add("a write through the copied base reaches both names of the copied view",
        rv.get("one_object") is True and rv.get("tail") == 99.0 and rv.get("tail_again") == 99.0,
        beh.get("error") or json.dumps(rv))
    steps = post.get("optimizer_steps") or []
    want_steps = [steps[0] + 1] if len(steps) == 1 else None
    add("a real optimizer step on a deepcopy moves the copy's model",
        beh.get("copy_optimizer_bound_to_copy_model") is True and beh.get("copy_step_moved_every_copy_param") is True
        and want_steps is not None and beh.get("copy_step_counts") == want_steps,
        beh.get("error") or f"copy steps {beh.get('copy_step_counts')}, expected {want_steps}")
    fpu = post.get("fingerprint_unchanged_by_probe") or {}
    add("the probe changed nothing (controller fingerprint and its own digest)",
        post.get("core_unchanged_by_probe") is True and fpu.get("equal") is True,
        f"fingerprint mismatched {fpu.get('mismatched')}" if not fpu.get("equal") else "")
    if pre:
        a, b = pre.get("core") or {}, post.get("core") or {}
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        add("independent state digest equal before and after", bool(a) and not diff,
            f"differs: {diff}" if diff else f"{len(a)} core fields: parameters and buffers, gradients, optimizer, "
                                            f"scheduler, loader and NumPy generators, three RNG streams, E15 "
                                            f"tensors, E15 arrays, plain values, workspace (the write log, counter "
                                            f"and write list are the write rows' job)")
        add("workspace files equal (the write log aside)", bool(a) and a.get("workspace") == b.get("workspace"))
        rng = [k for k in ("rng_python", "rng_numpy", "rng_torch_cpu") if a.get(k) != b.get(k)]
        add("python, numpy and torch CPU RNG streams equal", bool(a) and not rng, f"differ: {rng}" if rng else "")
        _grad_sharing_check(add, pre.get("grad_sharing"), post.get("grad_sharing"))


def _grad_sharing_check(add, before: dict[str, list] | None, after: dict[str, list] | None) -> None:
    """Gradients are carried BY VALUE: the storage adapter records no `.grad`
    views, so a gradient that shares storage (a bucketed layout, a grad that
    is a view of a flat buffer) would come back unshared with no failure
    reported anywhere. E15's gradients share nothing, so its gradient rows are
    value claims only; this says so, and FAILS if a relationship that existed
    before a hop is gone after it (or one appears)."""
    if before is None or after is None:
        add("gradient storage relationships (scope)", False, "not probed")
        return
    shared = {k: v for k, v in before.items() if v}
    if not shared and not any(after.values()):
        add("gradient storage relationships not exercised (gradients claimed by value only)", note=True,
            detail=f"{len(before)} gradients before, {len(after)} after; none shares storage with another tensor. "
                   f"The storage adapter records no .grad views, so a gradient that is a view of a buffer is "
                   f"outside what E15 shows")
        return
    add("gradient storage relationships kept", before == after,
        f"before {shared}, after {({k: v for k, v in after.items() if v})}")


def _commit_checks(T: Table, name: str, h: dict[str, Any] | None, *, local: bool, tags: bool, expected_views: int,
                   crash_phase: str | None) -> None:
    add = T.section(name)
    first = name == "hop1"
    if h is None:
        add("boundary reached", False, "no record: the lifecycle stopped before this hop")
        return
    res = h.get("result") or {}
    add("controller reached DONE", res.get("phase") == "DONE",
        f"{res.get('phase')} {res.get('failed_at') or ''} {res.get('reason') or ''} {(res.get('detail') or '')[:160]}")
    if not first:
        down = h.get("down") or {}
        add(f"controller killed after the {crash_phase} journal write",
            down.get("killed") is True and down.get("phase_at_kill") == crash_phase,
            f"killed {down.get('killed')} at {down.get('phase_at_kill')}")
        add("gate closed while the controller was down",
            down.get("admission_at_kill") == "closed" and _get(down, "authority", "closed_by") == h.get("mig"),
            f"admission {down.get('admission_at_kill')}, route closed by {_get(down, 'authority', 'closed_by')}")
        dw = h.get("down_writes") or []
        add("writes routed while it was down were refused", bool(dw) and all(w.get("outcome") == "refused" for w in dw),
            f"{sum(1 for w in dw if w.get('outcome') == 'refused')}/{len(dw)} refused")
        add("a fresh controller resumed with takeover", res.get("resumed") is True and res.get("phase") == "DONE",
            f"resumed {res.get('resumed')} -> {res.get('phase')} in {h.get('recovery_s')} s")
    pf = h.get("preflight") or res.get("preflight") or {}
    add("preflight admitted the switch and excluded nothing", pf.get("decision") == "proceed" and not pf.get("excluded"),
        f"decision {pf.get('decision')}, excluded {pf.get('excluded')}, unknown "
        f"{[u.get('name') for u in pf.get('unknown') or []]}")
    cap = h.get("capture") or res.get("capture") or {}
    add("capture left the source unchanged (bindings and state)",
        cap.get("source_state_unchanged") is True and cap.get("source_bindings_identical") is True,
        f"state {cap.get('source_state_unchanged')} {cap.get('source_state_mismatched') or ''}, "
        f"bindings {cap.get('source_bindings_identical')}")
    _expectation_check(add, h, cap)
    rst = res.get("restore") or {}
    views, grads = rst.get("views") or {}, rst.get("grads") or {}
    add("every view record repaired on restore", not views.get("failed") and views.get("restored", 0) >= expected_views,
        f"restored {views.get('restored')}, failed {views.get('failed')}")
    add("gradients carried and reattached (by value)",
        grads.get("carried", 0) > 0 and grads.get("reattached") == grads.get("carried") and not grads.get("failed"),
        f"carried {grads.get('carried')}, reattached {grads.get('reattached')}, failed {grads.get('failed')}")
    cb = res.get("commit_boundary") or {}
    verdicts = cb.get("verdicts") or {}
    for stage in ("restored", "pre_commit"):
        v = verdicts.get(stage) or {}
        add(f"commit boundary equal ({'after restore' if stage == 'restored' else 'before COMMITTED'})",
            v.get("equal") is True, f"mismatched {v.get('mismatched')}" if v else "no verdict")
    fps = [cb.get(k) for k in ("source", "restored", "pre_commit")]
    oks = [_fp_views_ok(fp, expected_views) for fp in fps]
    add("views intact at the cut, after restore and before COMMITTED (views_intact)",
        all(fp for fp in fps) and all(ok for ok, _ in oks), " | ".join(d for _, d in oks))
    declared = (verdicts.get("pre_commit") or {}).get("declared") or {}
    add("no value left undigested by the fingerprint", not declared.get("unhashable"),
        f"unhashable {declared.get('unhashable')}" if declared.get("unhashable") else "")
    cuda_want = "not_carried" if first else "declared_drop"
    if local:
        add(f"CUDA RNG declared {cuda_want}", skip=f"{NO_CUDA} (verdict here: {declared.get('rng_cuda')})")
    else:
        add(f"CUDA RNG declared {cuda_want}", declared.get("rng_cuda") == cuda_want, str(declared.get("rng_cuda")))
    moved = declared.get("devices") or {}
    if first:
        add("placement preserved (no tensor changed device)", not moved, f"moved {moved}" if moved else "")
    elif local:
        add("every moved tensor went cuda -> cpu", skip=NO_CUDA)
    else:
        wrong = {k: v for k, v in moved.items() if not (str(v[0]).startswith("cuda") and str(v[1]) == "cpu")}
        add("every moved tensor went cuda -> cpu", bool(moved) and not wrong,
            f"{len(moved)} moved" + (f"; not cuda -> cpu: {list(wrong)[:5]}" if wrong else ""))
    if not first:
        remap = rst.get("remap") or {}
        locs = remap.get("locations") or {}
        label = "CUDA storages remapped to cpu on restore"
        if local and not tags:
            add(label, skip=NO_CUDA + " (location tags off)")
        else:
            add(label + (" (simulated cuda:0 location tags)" if local else ""),
                remap.get("remapped", 0) > 0 and bool(locs) and all(str(k).startswith("cuda") for k in locs),
                f"remapped {remap.get('remapped')} from {locs}")
    val = res.get("validation") or {}
    declared_rows = [d.get("name") for d in val.get("declared") or []]
    # Exactly one row may be declared not run: the forward output, which has
    # no source reference because the source runs no model code (above).
    # Every other check must have run and passed; a declared continuation, or
    # a forward row that "passed", is a defect.
    add("validation passed without writing (isolated), only the forward row declared not run",
        val.get("continuation") == "isolated" and val.get("passed") == val.get("total")
        and val.get("total", 0) + len(declared_rows) == 12 and declared_rows == ["forward output"],
        f"{val.get('passed')}/{val.get('total')} run ({val.get('continuation')}), declared {declared_rows}")
    adm = res.get("admission") or {}
    add("admission closed during the migration and open afterwards",
        adm.get("closed_during_migration") is True and adm.get("state") == "open",
        f"closed {adm.get('closed_during_migration')}, now {adm.get('state')}, admitted {adm.get('admitted')}, "
        f"refused {adm.get('refused')}, drains {[d.get('waited_on') for d in adm.get('drains') or []]}")
    # The controller deletes the source, and pauses it when the provider
    # refuses the delete (a known deployment deadlock); it journals the
    # fallback, so a paused source is a release, recorded as such.
    fallback = [e["note"] for e in h.get("events") or []
                if e.get("phase") == "SOURCE_RELEASED" and str(e.get("note", "")).startswith("delete HTTP")]
    add("source released after the commit (deleted, or paused on a refused delete)",
        (h.get("source_alive_after") is False or bool(fallback)) and h.get("dest_alive_after") is True,
        f"source alive {h.get('source_alive_after')}" + (f", {fallback[0]}" if fallback else "")
        + f", destination alive {h.get('dest_alive_after')}")
    _write_checks(add, h, name)
    _probe_checks(add, h.get("probe_pre"), h.get("probe_post"), where="the destination")


def _abort_checks(T: Table, h: dict[str, Any] | None, *, local: bool, fault: str) -> None:
    add = T.section("hop2a")
    if h is None:
        add("boundary reached", False, "no record: the lifecycle stopped before this hop")
        return
    res, row, auth = h.get("result") or {}, h.get("journal_row") or {}, h.get("authority_after") or {}
    add(f"ABORTED at DEST_RESTORED with the {fault} reason",
        res.get("phase") == "ABORTED" and res.get("failed_at") == "DEST_RESTORED"
        and res.get("reason") == ABORT_REASON.get(fault),
        f"{res.get('phase')} at {res.get('failed_at')}: {res.get('reason')}")
    src = h.get("source_pid")
    add("authority stays on the GPU runtime", row.get("authoritative") == src and auth.get("runtime") == src,
        f"journal {row.get('authoritative') == src}, route {auth.get('runtime') == src}")
    adm = res.get("admission") or {}
    add("the gate closed for the capture and reopened on the GPU runtime",
        adm.get("closed_during_migration") is True and row.get("admission") == "open" and auth.get("closed_by") is None,
        f"closed {adm.get('closed_during_migration')}, now {row.get('admission')}")
    add("the aborted destination was deleted", h.get("dest_alive_after") is False and h.get("source_alive_after") is True)
    cap = res.get("capture") or {}
    add("capture left the source unchanged (bindings and state)",
        cap.get("source_state_unchanged") is True and cap.get("source_bindings_identical") is True)
    v = h.get("source_fingerprint_verdict") or {}
    add("GPU runtime state unchanged (controller fingerprint, before vs after)", v.get("equal") is True,
        f"mismatched {v.get('mismatched')}")
    cu = _get(v, "declared", "rng_cuda")
    if local:
        add("CUDA RNG on the GPU runtime unchanged", skip=f"{NO_CUDA} (verdict here: {cu})")
    else:
        add("CUDA RNG on the GPU runtime unchanged", cu == "equal", str(cu))
    wa = h.get("write_after_abort") or {}
    add("the next routed write lands on the GPU runtime", wa.get("outcome") == "ack" and wa.get("runtime") == src,
        f"{wa.get('outcome')} on {wa.get('runtime_role')}")
    _probe_checks(add, h.get("probe_pre"), h.get("probe_post"), where="the GPU runtime")


def _trajectory_checks(T: Table, by: dict[str, dict[str, Any]], *, local: bool, steps: int | None) -> None:
    add = T.section("trajectory")
    blocks = [by.get(f"block{i}") for i in range(3)]
    for i, b in enumerate(blocks):
        if b is None:
            add(f"block {i} ran", False, "no record")
    present = [b for b in blocks if b is not None]
    for prev, cur in zip(present, present[1:]):
        p, c = prev["report"], cur["report"]
        add(f"block {cur['block']} starts where block {prev['block']} ended (step_count)",
            c["start"]["step_count"] == p["end"]["step_count"],
            f"{p['end']['step_count']} -> {c['start']['step_count']}")
        a, b2 = p["end"]["fixed_batch_loss"], c["start"]["fixed_batch_loss"]
        rel = abs(b2 - a) / max(abs(a), 1e-12)
        add(f"fixed-batch loss continuous into block {cur['block']} (rtol {LOSS_RTOL:.0e})", rel <= LOSS_RTOL,
            f"{a!r} ({p['end']['device']}) -> {b2!r} ({c['start']['device']}), relative {rel:.2e}")
    for b in present:
        r = b["report"]
        adv = r.get("global_rng_advanced") or {}
        add(f"block {b['block']}: optimizer advanced by the block's steps and user code drew every global stream",
            r["end"]["step_count"] - r["start"]["step_count"] == (steps or len(r["losses"]))
            and bool(adv) and all(adv.values()),
            f"step {r['start']['step_count']} -> {r['end']['step_count']}, streams advanced {adv}")
    b1 = by.get("block1")
    if b1 is not None:
        r = b1["report"]
        views = (r.get("move") or {}).get("views") or {}
        mv = r.get("move") or {}
        add("block 1 moved the base and re-pointed every view (identity kept)",
            (mv.get("needed") or mv.get("forced")) is True and mv.get("rehomed") is True
            and len(views) == 4 and all(v["shares_base"] for v in views.values())
            and all(v["class"] == ("Tensor" if k == "e15_tail" else "Parameter") for k, v in views.items())
            and mv.get("optimizer_bound") is True,
            f"forced {mv.get('forced')}, rehomed {mv.get('rehomed')}, base {mv.get('base_device')}, "
            f"views {[(k, v['device'], v['shares_base']) for k, v in views.items()]}")
        if local:
            add("block 1 trained on CUDA, the model's Parameters CUDA views of a CUDA base",
                skip=NO_CUDA + f" ({'; '.join(r.get('notes') or [])})")
            add("dropout consumed the CUDA stream in block 1", skip=NO_CUDA)
        else:
            on_cuda = (str(r.get("device")).startswith("cuda") and str((r.get("move") or {}).get("base_device")).startswith("cuda")
                       and len(views) == 4 and all(v["shares_base"] and v["device"].startswith("cuda") for v in views.values())
                       and (r.get("move") or {}).get("optimizer_bound") is True)
            add("block 1 trained on CUDA, the model's Parameters CUDA views of a CUDA base", on_cuda,
                f"{r.get('device')} ({r.get('cuda_device')}), base {(r.get('move') or {}).get('base_device')}, "
                f"views {[(k, v['device'], v['shares_base']) for k, v in views.items()]}")
            add("dropout consumed the CUDA stream in block 1", r.get("cuda_rng_advanced_by_training") is True,
                str(r.get("cuda_rng_advanced_by_training")))
    b2 = by.get("block2")
    if b2 is not None:
        add("block 2 trained on cpu after the return", b2["report"].get("device") == "cpu", b2["report"].get("device"))
    fin = by.get("final") or {}
    pw = _get(fin, "probe", "writes", default=None)
    w = fin.get("writes") or {}
    if pw is None:
        add("every acknowledged write of the lifecycle is on the final runtime", False, "no final probe")
    else:
        acked, refused = w.get("acked_so_far") or [], w.get("refused_so_far") or []
        add("every acknowledged write of the lifecycle is on the final runtime",
            sorted(pw.get("log") or []) == sorted(acked) and pw.get("acks") == pw.get("log")
            and pw.get("counter") == len(acked) and not (set(refused) & set(pw.get("log") or [])),
            f"{len(acked)} acknowledged, {len(refused)} refused, {len(w.get('errors_so_far') or [])} errors "
            f"over the lifecycle")
        notes = _get(fin, "probe", "notes_lines")
        add("the user's notes file grew by one line per block", notes == 1 + len(present),
            f"{notes} lines for {len(present)} blocks")
    cells = [(b["block"], b.get("runtime_role"), b.get("import_cell")) for b in present if b.get("import_cell")]
    for blk, role, cell in cells:
        add(f"import cell re-run on the restored {role} before block {blk} (diagnostic)", note=True,
            detail=f"missing before: {cell.get('missing_before')}. The capture carries no underscore names or "
                   f"modules, so the controller's expectation refresh (fixture compute_expectation -> "
                   f"_digest_params) needs them re-bound; see e15_combined.py, THE IMPORT CELL, and the run "
                   f"row '{UNAIDED}'")


def judge(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The verdict table for one run's records."""
    T = Table()
    by = {r.get("name"): r for r in records}
    setup = by.get("setup")
    args = (setup or {}).get("args") or {}
    local = any(r.get("local") for r in records)
    tags = local and not args.get("no_local_cuda_tags")
    expected_views = int((setup or {}).get("expected_view_records") or 6)
    add = T.section("run")
    add("records present", bool(records) and setup is not None, f"{len(records)} records")
    stopped = by.get("stopped")
    add("lifecycle completed", stopped is None and "final" in by, (stopped or {}).get("reason", ""))
    # Records from before the default changed carry no `reimport_cell` and
    # ran the cell; the cells themselves are the evidence either way.
    cells = [(r.get("name"), r.get("runtime_role"), (r.get("import_cell") or {}).get("missing_before"))
             for r in records if r.get("stage") == "train" and r.get("import_cell")]
    mode = args.get("reimport_cell", "fixture" if cells else "none")
    ran = ("the harness re-ran the fixture's definitions on "
           + ", ".join(f"{role} before {name} (missing {miss})" for name, role, miss in cells) if cells else
           "no restored runtime reached user work, so the cell never ran, but the run was configured to help")
    add(UNAIDED, mode == "none" and not cells,
        (f"--reimport-cell {mode}: {ran}. The controller's expectation refresh reaches those helpers through "
         f"__main__ and the capsule carries none of them, so this run is a diagnostic of the rest of the "
         f"lifecycle, not evidence about the controller alone") if (cells or mode != "none")
        else "--reimport-cell none")
    if args.get("sabotage"):
        add(f"sabotage: {args['sabotage']}", note=True, detail="a deliberately broken run: FAILs are expected")
    _commit_checks(T, "hop1", by.get("hop1"), local=local, tags=tags, expected_views=expected_views, crash_phase=None)
    _abort_checks(T, by.get("hop2a"), local=local, fault=args.get("abort_fault", "bad_capsule"))
    _commit_checks(T, "hop2b", by.get("hop2b"), local=local, tags=tags, expected_views=expected_views,
                   crash_phase=args.get("crash_phase", "CAPTURED"))
    _trajectory_checks(T, by, local=local, steps=args.get("steps"))
    rows = T.rows
    failed = sum(1 for r in rows if r["verdict"] == "FAIL")
    run = (records[0].get("run") or {}) if records else {}
    return {"rows": rows, "failed": failed, "passed": sum(1 for r in rows if r["verdict"] == "PASS"),
            "skipped": sum(1 for r in rows if r["verdict"] == "SKIP"), "ok": bool(records) and failed == 0,
            "local": local, "run_id": run.get("run_id"), "cohort": records[0].get("cohort") if records else None,
            "trajectory": trajectory(records), "writes": writes_summary(records)}


# ---------------------------------------------------------------------------
# Trajectory and writes
# ---------------------------------------------------------------------------

def trajectory(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per stage, in lifecycle order."""
    out = []
    for r in records:
        st, name = r.get("stage"), r.get("name")
        if st == "train":
            rep = r["report"]
            out.append({"stage": name, "runtime": r.get("runtime_role"), "profile": r.get("profile"),
                        "device": rep.get("device"), "steps": len(rep.get("losses") or []),
                        "step_count": f"{rep['start']['step_count']} -> {rep['end']['step_count']}",
                        "lr": rep["end"]["lr"], "last_loss": (rep.get("losses") or [None])[-1],
                        "fixed_batch_loss": rep["end"]["fixed_batch_loss"],
                        "note": "; ".join(rep.get("notes") or [])})
        elif st == "hop":
            res = r.get("result") or {}
            mine = (r.get("writes") or {}).get("entries") or []
            out.append({"stage": name, "runtime": f"{r.get('source_role')} -> {r.get('dest_role')}",
                        "profile": f"{r.get('source_profile')} -> {r.get('dest_profile')}",
                        "device": None, "steps": None,
                        "step_count": _get(r, "probe_post", "step_count"),
                        "lr": None, "last_loss": None, "fixed_batch_loss": None,
                        "note": (f"{res.get('phase')}"
                                 + (f" at {res.get('failed_at')} ({res.get('reason')})" if res.get("failed_at") else "")
                                 + (f", killed after {r.get('crash_phase')}" if r.get("crash_phase") else "")
                                 + f", writes {sum(1 for e in mine if e.get('outcome') == 'ack')} acked / "
                                   f"{sum(1 for e in mine if e.get('outcome') == 'refused')} refused"
                                 + f", {r.get('wall_s')} s")})
    return out


def writes_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    fin = next((r for r in records if r.get("name") == "final"), None) or (records[-1] if records else {})
    w = fin.get("writes") or {}
    entries = [e for r in records if r.get("stage") == "hop" for e in (r.get("writes") or {}).get("entries") or []]
    by_hop: dict[str, dict[str, int]] = {}
    for e in entries:
        d = by_hop.setdefault(e["hop"], {"ack": 0, "refused": 0, "error": 0})
        d[e["outcome"]] = d.get(e["outcome"], 0) + 1
    return {"acknowledged": len(w.get("acked_so_far") or []), "refused": len(w.get("refused_so_far") or []),
            "errors": len(w.get("errors_so_far") or []), "by_hop": by_hop}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(result: dict[str, Any]) -> str:
    lines = [f"E15 combined lifecycle, run {result.get('run_id')} cohort {result.get('cohort')}"
             + ("  (LOCAL DRY RUN: exercises the harness, not a result)" if result.get("local") else "")]
    section = None
    for r in result["rows"]:
        if r["section"] != section:
            section = r["section"]
            lines.append("")
            lines.append(section)
        lines.append(f"  {r['verdict']:<5} {r['check']}" + (f"  [{r['detail']}]" if r["detail"] else ""))
    lines.append("")
    lines.append("trajectory")
    lines.append(f"  {'stage':<8} {'runtime':<34} {'device':<7} {'step_count':<11} {'lr':<10} "
                 f"{'last loss':<10} {'fixed-batch loss':<17} note")
    for t in result["trajectory"]:
        fmt = lambda v, spec: "-" if v is None else format(v, spec)  # noqa: E731
        lines.append(f"  {t['stage']:<8} {str(t['runtime']):<34} {str(t['device'] or '-'):<7} "
                     f"{str(t['step_count'] if t['step_count'] is not None else '-'):<11} {fmt(t['lr'], '.3g'):<10} "
                     f"{fmt(t['last_loss'], '.5f'):<10} {fmt(t['fixed_batch_loss'], '.7f'):<17} {t['note']}")
    w = result["writes"]
    lines.append("")
    lines.append(f"routed writes: {w['acknowledged']} acknowledged, {w['refused']} refused, {w['errors']} errors; "
                 f"by hop {json.dumps(w['by_hop'])}")
    helped = any(r["check"] == UNAIDED and r["verdict"] == "FAIL" for r in result["rows"])
    lines.append(f"\nE15 {result['passed']} PASS, {result['failed']} FAIL, {result['skipped']} SKIP"
                 f"  ->  {'OK' if result['ok'] else 'NOT OK'}"
                 + ("  (the harness re-bound the fixture helpers: a diagnostic, not evidence about the "
                    "controller alone)" if helped else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test: the checks can fail
# ---------------------------------------------------------------------------

def _mutations(records: list[dict[str, Any]]) -> list[tuple[str, str, str, Any]]:
    """(description, section, check that must FAIL (a prefix), mutator) over
    real records. The section matters: one check runs in several sections, and
    a mutation of one hop must fail THAT hop's row."""
    def rec(name):
        return lambda rs: next(r for r in rs if r.get("name") == name)

    def drop_last_ack(rs):
        pw = rec("hop2b")(rs)["probe_post"]["writes"]
        pw["log"] = pw["log"][:-1]
        pw["acks"] = pw["acks"][:-1]

    def refused_present(rs):
        h = rec("hop1")(rs)
        seq = h["writes"]["refused_so_far"][0]
        h["probe_post"]["writes"]["log"].append(seq)
        h["probe_post"]["writes"]["acks"].append(seq)

    def flip_boundary(rs):
        rec("hop1")(rs)["result"]["commit_boundary"]["verdicts"]["pre_commit"]["equal"] = False

    def zero_remap(rs):
        rec("hop2b")(rs)["result"]["restore"]["remap"]["remapped"] = 0

    def break_alias(rs):
        rec("hop1")(rs)["probe_post"]["identity"]["alias_is_model_weight"] = False

    def abort_completed(rs):
        rec("hop2a")(rs)["result"]["phase"] = "DONE"

    def rng_moved(rs):
        rec("hop2b")(rs)["probe_post"]["core"]["rng_torch_cpu"] = "0" * 64

    def step_jump(rs):
        rec("block1")(rs)["report"]["start"]["step_count"] += 1

    def validation_declared(rs):
        v = rec("hop1")(rs)["result"]["validation"]
        v["declared"] = [{"name": "continuation", "detail": "x"}]
        v["total"] -= 1
        v["passed"] -= 1

    def down_write_acked(rs):
        rec("hop2b")(rs)["down_writes"][0]["outcome"] = "ack"

    def view_unshared(rs):
        rec("hop2b")(rs)["probe_post"]["torch_views"]["e15_top"]["same_storage"] = False

    def param_class(rs):
        rec("hop1")(rs)["probe_post"]["classes"]["e15_top"]["class"] = "Tensor"

    def no_refusal(rs):
        for e in rec("hop1")(rs)["writes"]["entries"]:
            if e["outcome"] == "refused":
                e["outcome"] = "ack"
                e["runtime"] = rec("hop1")(rs)["source_pid"]

    def fp_views_broken(rs):
        rec("hop2b")(rs)["result"]["commit_boundary"]["pre_commit"]["views"]["broken"] = ["x"]

    def probe_wrote(rs):
        rec("hop1")(rs)["probe_post"]["fingerprint_unchanged_by_probe"]["equal"] = False

    def abort_source_changed(rs):
        rec("hop2a")(rs)["source_fingerprint_verdict"]["equal"] = False

    def stopped(rs):
        rs[:] = [r for r in rs if r.get("name") not in ("block2", "final")]

    def harness_reimport(rs):
        rec("setup")(rs)["args"]["reimport_cell"] = "fixture"
        rec("block1")(rs)["import_cell"] = {"missing_before": ["_digest_params", "hashlib"]}

    def expectation_lost_digest(rs):
        # What the read-only refresh records when the helper is missing.
        rec("hop1")(rs)["expectation"]["param_digest"] = None

    def expectation_stale(rs):
        e = rec("hop2b")(rs)["expectation"]
        e["step_count"] = (e.get("step_count") or 0) - 6

    def grad_sharing_lost(rs):
        h = rec("hop2b")(rs)
        name = next(iter(h["probe_pre"]["grad_sharing"]))
        h["probe_pre"]["grad_sharing"][name] = ["e15_flat"]

    return [
        ("the harness re-ran the definitions cell on a restored runtime", "run", UNAIDED, harness_reimport),
        ("the refreshed expectation lost its parameter digest", "hop1",
         "validation expectation refreshed at the cut", expectation_lost_digest),
        ("the refreshed expectation describes an older step", "hop2b",
         "validation expectation refreshed at the cut", expectation_stale),
        ("a gradient that shared the base comes back unshared", "hop2b", "gradient storage relationships kept",
         grad_sharing_lost),
        ("an acknowledged write missing at the destination", "hop2b",
         "every acknowledged write present exactly once", drop_last_ack),
        ("a refused write present at the destination", "hop1", "every refused write absent", refused_present),
        ("the pre-commit boundary verdict unequal", "hop1", "commit boundary equal (before COMMITTED)", flip_boundary),
        ("no storage remapped on the T4 -> cpu restore", "hop2b", "CUDA storages remapped to cpu on restore",
         zero_remap),
        ("the alias no longer the model weight", "hop1", "external alias is the model weight", break_alias),
        ("the faulted hop completed", "hop2a", "ABORTED at DEST_RESTORED", abort_completed),
        ("the torch CPU stream moved across the hop", "hop2b", "python, numpy and torch CPU RNG streams equal",
         rng_moved),
        ("the step count jumped across hop 1", "trajectory", "block 1 starts where block 0 ended", step_jump),
        ("the continuation check declared on a fresh destination", "hop1",
         "validation passed without writing (isolated)", validation_declared),
        ("the source ran the model at the cut (a forward reference present)", "hop1",
         "validation expectation refreshed at the cut",
         lambda rs: rec("hop1")(rs)["expectation"].update(forward_output=1.0)),
        ("the forward row passed instead of being declared", "hop1",
         "validation passed without writing (isolated)",
         lambda rs: rec("hop1")(rs)["result"]["validation"].update(declared=[], total=12, passed=12)),
        ("a write acknowledged while the controller was down", "hop2b",
         "writes routed while it was down were refused", down_write_acked),
        ("the top-level Parameter off the base", "hop2b", "every view shares its base at its offset", view_unshared),
        ("the top-level Parameter now a plain Tensor", "hop1", "Parameter classes, requires_grad and leaf-ness kept",
         param_class),
        ("the writer never met the closed gate", "hop1", "the writer met the closed gate", no_refusal),
        ("a view broken in the pre-commit fingerprint", "hop2b", "views intact at the cut", fp_views_broken),
        ("the probe changed the fingerprint", "hop1", "the probe changed nothing", probe_wrote),
        ("the aborted hop changed the GPU runtime", "hop2a", "GPU runtime state unchanged", abort_source_changed),
        ("the lifecycle stopped before block 2", "run", "lifecycle completed", stopped),
    ]


def live_shaped(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A copy of dry-run records rewritten to what a CORRECT live run records
    where the two differ: CUDA RNG verdicts, tensors moved cuda -> cpu on the
    return hop, and a T4 block on CUDA. It lets the self-test run the live-only branches,
    which a dry run only ever SKIPs; it is not evidence about a live run."""
    rs = copy.deepcopy(records)
    by = {r.get("name"): r for r in rs}
    for r in rs:
        r["local"] = False
    for name, verdict in (("hop1", "not_carried"), ("hop2b", "declared_drop")):
        for stage in ("restored", "pre_commit"):
            by[name]["result"]["commit_boundary"]["verdicts"][stage]["declared"]["rng_cuda"] = verdict
    for stage in ("restored", "pre_commit"):
        by["hop2b"]["result"]["commit_boundary"]["verdicts"][stage]["declared"]["devices"] = {
            "model.param.fc1.weight": ["cuda:0", "cpu"], "e15_flat": ["cuda:0", "cpu"]}
    by["hop2a"]["source_fingerprint_verdict"]["declared"]["rng_cuda"] = "equal"
    rep = by["block1"]["report"]
    rep.update(device="cuda", cuda_device="Tesla T4", cuda_rng_advanced_by_training=True, notes=[])
    rep["move"].update(base_device="cuda:0", needed=True, forced=False)
    for view in rep["move"]["views"].values():
        view["device"] = "cuda:0"
    return rs


def _live_mutations() -> list[tuple[str, str, str, Any]]:
    def rec(rs, name):
        return next(r for r in rs if r.get("name") == name)

    def set_(name, path, value):
        def f(rs):
            d = rec(rs, name)
            for k in path[:-1]:
                d = d[k]
            d[path[-1]] = value
        return f

    return [
        ("CUDA RNG carried instead of not_carried on cpu -> T4", "hop1", "CUDA RNG declared not_carried",
         set_("hop1", ["result", "commit_boundary", "verdicts", "pre_commit", "declared", "rng_cuda"], "absent")),
        ("CUDA RNG not declared dropped on T4 -> cpu", "hop2b", "CUDA RNG declared declared_drop",
         set_("hop2b", ["result", "commit_boundary", "verdicts", "pre_commit", "declared", "rng_cuda"], "equal")),
        ("a tensor that moved cpu -> cuda on the return hop", "hop2b", "every moved tensor went cuda -> cpu",
         set_("hop2b", ["result", "commit_boundary", "verdicts", "pre_commit", "declared", "devices"],
              {"x": ["cpu", "cuda:0"]})),
        ("the abort advanced the T4's CUDA stream", "hop2a", "CUDA RNG on the GPU runtime unchanged",
         set_("hop2a", ["source_fingerprint_verdict", "declared", "rng_cuda"], "mismatch")),
        ("block 1 left on cpu", "trajectory", "block 1 trained on CUDA", set_("block1", ["report", "device"], "cpu")),
        ("dropout drew nothing from CUDA", "trajectory", "dropout consumed the CUDA stream",
         set_("block1", ["report", "cuda_rng_advanced_by_training"], False)),
        ("a second row declared on the return hop", "hop2b", "validation passed without writing (isolated)",
         set_("hop2b", ["result", "validation", "declared"],
              [{"name": "forward output", "detail": "x"}, {"name": "continuation", "detail": "x"}])),
        # The reviewer's case: on the T4 the read-only refresh finds no digest
        # helper, records None and still reports "cut".
        ("the T4's read-only refresh recorded no parameter digest", "hop2b",
         "validation expectation refreshed at the cut", set_("hop2b", ["expectation", "param_digest"], None)),
        ("the T4 source ran the model at the cut", "hop2b",
         "validation expectation refreshed at the cut",
         lambda rs: rec(rs, "hop2b")["expectation"].update(forward_output=1.0)),
    ]


def _failing(res: dict[str, Any]) -> set[tuple[str, str]]:
    return {(r["section"], r["check"]) for r in res["rows"] if r["verdict"] == "FAIL"}


def _known_only(res: dict[str, Any]) -> bool:
    """True when every FAIL is the one known, attributed failure: the harness
    helped the controller (a `--reimport-cell fixture` diagnostic run)."""
    return all(sec == SECTIONS["run"] and chk == UNAIDED for sec, chk in _failing(res))


def _run_mutations(records, mutations, ok_all: bool) -> bool:
    base_fail = _failing(judge(records))
    for desc, section, check, mutate in mutations:
        rs = copy.deepcopy(records)
        try:
            mutate(rs)
        except (KeyError, IndexError, StopIteration, TypeError, AttributeError) as exc:
            print(f"  FAIL  {desc}: mutation not applicable to these records ({type(exc).__name__}: {exc})")
            ok_all = False
            continue
        res = judge(rs)
        hit = [r for r in res["rows"] if r["section"] == SECTIONS[section] and r["check"].startswith(check)]
        caught = bool(hit) and all(r["verdict"] == "FAIL" for r in hit) and not res["ok"]
        already = bool(hit) and all((r["section"], r["check"]) in base_fail for r in hit)
        ok_all &= caught
        print(f"  {'PASS' if caught else 'FAIL'}  {desc}: '{check}' -> "
              f"{[r['verdict'] for r in hit] or 'no such check'}"
              + ("  (already FAIL in the unmutated records)" if caught and already else ""))
    return ok_all


def selftest(records: list[dict[str, Any]]) -> int:
    base = judge(records)
    ok_all = base["ok"] or _known_only(base)
    print(f"unmutated records: {'OK' if base['ok'] else 'NOT OK'} ({base['passed']} PASS, {base['failed']} FAIL, "
          f"{base['skipped']} SKIP)"
          + ("" if base["ok"] else "; the only FAIL is the known one, the harness helped the controller"
             if _known_only(base) else "; FAILS beyond the known one, so the self-test cannot attribute a mutation"))
    for r in base["rows"]:
        if r["verdict"] == "FAIL":
            print(f"      FAIL {r['section']}: {r['check']} [{r['detail'][:200]}]")
    ok_all = _run_mutations(records, _mutations(records), ok_all)
    if any(r.get("local") for r in records):
        live = live_shaped(records)
        lb = judge(live)
        ok_all &= (lb["ok"] or _known_only(lb)) and lb["skipped"] == 0
        print(f"live-shaped copy: {'OK' if lb['ok'] else 'NOT OK'} ({lb['passed']} PASS, {lb['failed']} FAIL, "
              f"{lb['skipped']} SKIP; every live-only check must run and pass)")
        for r in lb["rows"]:
            if r["verdict"] in ("FAIL", "SKIP"):
                print(f"      {r['verdict']} {r['section']}: {r['check']} [{r['detail'][:200]}]")
        ok_all = _run_mutations(live, _live_mutations(), ok_all)
    print("selftest", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", default=str(DEFAULT_RECORDS))
    ap.add_argument("--cohort", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--json", default=None, help="also write the verdict table as JSON here")
    ap.add_argument("--selftest", action="store_true", help="mutate the records and show each check can fail")
    args = ap.parse_args()
    if not Path(args.records).exists():
        print(f"no records at {args.records} (a live run writes results/e15/e15_records.jsonl; a dry run, its --outdir)")
        return 1
    records = load(Path(args.records), args.cohort, args.run_id)
    if not records:
        print(f"no {SCHEMA} records in {args.records}" + (f" for cohort {args.cohort}" if args.cohort else ""))
        return 1
    if args.selftest:
        return selftest(records)
    result = judge(records)
    print(render(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1, default=str) + "\n")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
