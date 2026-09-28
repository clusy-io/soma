"""E14 analysis: per-cohort summary of the lifecycle arms, and the checks.

Reads the append-only `e14_arms.jsonl` written by `e14_lifecycle.py` (schema
`e14/2`; older records without a schema are counted and ignored) and, per
cohort and per `prep_passes` value, reports:

  * wall time, runtime seconds per SKU, estimated runtime cost under every
    recorded rate table, and ratios against `always` and `uninterrupted`;
  * work recomputed and counted application LOC;
  * the fingerprint comparison, in two kinds. WITHIN an arm: its TRAIN input
    against the output of the PREP that fed it, and the prep artefacts at the
    top of ANALYSE against the digests PREP recorded. This is what carrying
    (or recomputing) the state did, independent of the host. ACROSS arms:
    which arms trained from identical inputs and produced identical outputs.
    Across arms the PREPs ran in different sandboxes, and the fixture's CPU
    arithmetic depends on the host's thread count and ISA, so on a live run a
    cross-arm difference can come from the host rather than from lost state;
    the PREP-output groups and each PREP's recorded host tell the two apart;
  * the analysis metrics, including the ones that consume the prep artefacts,
    and the dependencies each arm was missing;
  * an lr trajectory summary (per-epoch schedule or not);
  * a crossover table when a cohort holds `switch` and `always` at two or
    more `--prep-passes` values: one row per invocation (every attempt, not
    only the latest), with the measured sign of switch minus always. No
    break-even pass count is interpolated.

`checks(records, requested)` is the verification list the harness runs after
a `--local` dry run (and `--checks` runs on real data). A requested arm that
failed, or left no witness a check needs, FAILS that check; "n/a" is kept for
arms that were not requested. Within-arm gates hold everywhere. Cross-arm
equalities are gates on a local dry run (one CPU, one thread count) and
informational on a live run.

Usage:
  python experiments/e14_analyse.py                     # results/e14 -> results/e14/e14_summary.json
  python experiments/e14_analyse.py --in X.jsonl --out Y.json --cohort C --checks
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "e14/2"

INPUT_COMPONENTS = ("params", "optimizer", "optimizer_step", "scheduler", "loader_generator", "module_training",
                    "step_count", "rng.python", "rng.numpy", "rng.torch_cpu", "rng.cuda")
OUTPUT_COMPONENTS = ("params", "optimizer", "optimizer_step", "scheduler", "loader_generator", "step_count")
PREP_METRICS = ("tta_accuracy", "representation_shift", "input_standardisation")
# The application checkpoints that save no prep artefacts, no loader
# generator and no module modes, and what they DO save (none of which may
# differ from the arm's own PREP output at the top of TRAIN).
OMITS_PREP = ("appckpt_incomplete", "appckpt_naive")
INCOMPLETE_SAVED = ("params", "optimizer", "optimizer_step", "scheduler", "step_count",
                    "rng.python", "rng.numpy", "rng.torch_cpu")

# Components excluded from the equality gates, with the reason. The CUDA
# stream is excluded because on a GPU runtime it legitimately differs between
# arms without influencing this job (no random op runs on the GPU): the
# switch destination starts from the process default, while every arm that
# ran the fixture's torch.manual_seed on that runtime (always, restart, both
# application-checkpoint rebuilds) starts from the seeded stream. Within an
# arm it differs too: PREP on a CPU runtime has no CUDA stream at all.
NOTED_EXCEPTIONS = {
    "rng.cuda": ("the CUDA stream is not an input of this job's TRAIN (no random op runs on the GPU) and "
                 "differs by construction: a switch destination starts from the process default, while an arm "
                 "that ran the fixture's torch.manual_seed on the GPU runtime starts from the seeded stream, and "
                 "a PREP on a CPU runtime has none"),
}
GATED = tuple(c for c in INPUT_COMPONENTS if c not in NOTED_EXCEPTIONS)
CROSS_ARM_NOTE = ("cross-arm comparisons span different sandboxes: the fixture's three AdamW steps and the feature "
                  "passes run on each runtime's CPU, whose bits depend on the intra-op thread count and ISA, so on a "
                  "live run a difference here can come from the host rather than from lost state. Compare the arms' "
                  "PREP outputs (identical_prep_output_groups) and hosts before attributing one.")


def load(path) -> list[dict]:
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def component(fp: dict | None, name: str):
    if fp is None:
        return None
    if name.startswith("rng."):
        v = (fp.get("rng") or {}).get(name[4:])
        if name == "rng.cuda":
            if not v or not v.get("available"):
                return "unavailable"
            return tuple(v.get("states") or ())
        return v
    v = fp.get(name)
    if name in ("scheduler", "module_training") and isinstance(v, dict):
        return v.get("digest")
    return v


def differing(a: dict | None, b: dict | None, comps=INPUT_COMPONENTS) -> list[str]:
    return [c for c in comps if component(a, c) != component(b, c)]


def latest(records: list[dict]) -> dict[str, dict]:
    """Per arm, the last completed attempt in file order (the last attempt of
    any outcome when none completed)."""
    best: dict[str, dict] = {}
    for r in records:
        cur = best.get(r.get("arm"))
        if cur is None or r.get("outcome") == "completed" or cur.get("outcome") != "completed":
            best[r.get("arm")] = r
    return best


def _fp(r: dict | None, which: str) -> dict | None:
    return ((r or {}).get("train") or {}).get(which)


def prep_witness(r: dict | None, phase: str) -> dict | None:
    """The PREP witness whose state `phase` ('train' or 'analysis') consumed:
    the arm's own PREP, or for restart the PREP it re-ran on that runtime."""
    if not r:
        return None
    name = (r.get("prep_feeding") or {}).get(phase, "prep")
    return r.get("prep") if name == "prep" else (r.get("prep_again") or {}).get(name)


def _prep_fp(r: dict | None, phase: str = "train") -> dict | None:
    return (prep_witness(r, phase) or {}).get("prep_output")


def _host(w: dict | None) -> dict | None:
    h = (w or {}).get("host")
    if not h:
        return None
    return {k: h.get(k) for k in ("cpu_model", "cpu_count", "torch_threads", "cpu_capability", "cuda_device")}


def own_prep_diff(r: dict | None) -> dict | None:
    """The arm's TRAIN input against the output of the PREP that fed it."""
    fp_p, fi = _prep_fp(r, "train"), _fp(r, "train_input")
    if fp_p is None or fi is None:
        return None
    return {"differs": differing(fp_p, fi, GATED),
            "noted_exceptions_differing": [c for c in NOTED_EXCEPTIONS if component(fp_p, c) != component(fi, c)]}


def artefacts_vs_prep(r: dict | None) -> dict | None:
    """The prep artefacts at the top of ANALYSE against the digests the PREP
    that fed the analysis recorded when it made them."""
    made = (prep_witness(r, "analysis") or {}).get("artefacts")
    seen = ((r or {}).get("analysis") or {}).get("artefacts_at_entry")
    if made is None or seen is None:
        return None
    return {"compared": sorted(made),
            "missing_at_analysis": sorted(k for k in made if seen.get(k, "missing") == "missing"),
            "differ_from_prep": sorted(k for k in made if seen.get(k, "missing") not in ("missing", made[k]))}


def expected_failure(arm: str, r: dict | None, local: bool) -> bool:
    """The one planned failure: live, the naive checkpoint saved on the GPU
    runtime cannot be opened on the CPU runtime without map_location."""
    return bool(r) and arm == "appckpt_naive" and not local and r.get("outcome") == "failed" \
        and r.get("failed_phase") == "app_load_2" and "CUDA" in (r.get("error") or "")


def _why(r: dict | None) -> str:
    if r is None:
        return "no record"
    if r.get("outcome") != "completed":
        return f"{r.get('outcome')} at {r.get('failed_phase')}: {(r.get('error') or '')[:160]}"
    return "completed without this witness"


def _groups(recs: dict[str, dict], get, comps) -> list[list[str]]:
    keyed: dict[tuple, list[str]] = {}
    for arm, r in recs.items():
        fp = get(r)
        if fp is None:
            continue
        keyed.setdefault(tuple(repr(component(fp, c)) for c in comps), []).append(arm)
    return sorted(keyed.values(), key=lambda g: (-len(g), g))


def lr_summary(train: dict | None) -> dict | None:
    if not train or "lr_trajectory" not in train:
        return None
    lrs = train["lr_trajectory"]
    changes = [i for i in range(1, len(lrs)) if lrs[i] != lrs[i - 1]]
    bounds = train.get("epoch_boundaries") or []
    bpe = train.get("batches_per_epoch")
    # TRAIN starts a fresh epoch at its first step, so after `steps` steps
    # exactly steps // batches_per_epoch epochs are complete, and the
    # scheduler must have stepped once for each (including a last epoch that
    # ends on the final step).
    expected = len(lrs) // bpe if bpe else None
    sin = ((train.get("train_input") or {}).get("scheduler") or {}).get("last_epoch")
    sout = ((train.get("train_output") or {}).get("scheduler") or {}).get("last_epoch")
    advanced = None if sin is None or sout is None else sout - sin
    per_epoch = (set(changes) <= set(bounds)
                 and advanced == len(bounds) == train.get("epochs_completed")
                 and (expected is None or expected == len(bounds))
                 and (bpe is None or all(b % bpe == 0 for b in bounds)))
    return {"steps": len(lrs), "lr_first": lrs[0] if lrs else None, "lr_last": lrs[-1] if lrs else None,
            "lr_min": min(lrs) if lrs else None, "distinct_lrs": len(set(lrs)), "change_steps": changes,
            "epoch_boundaries": bounds, "batches_per_epoch": bpe, "epochs_completed": train.get("epochs_completed"),
            "epochs_expected": expected, "scheduler_epochs_advanced": advanced, "per_epoch": per_epoch,
            "collapsed": bool(lrs) and min(lrs) < 1e-6}


def _cost(r: dict, table: str):
    return (r.get("est_runtime_cost_usd") or {}).get(table)


def _ratio(a, b):
    return None if a is None or b in (None, 0) else a / b


def _cpu_phase_s(r: dict) -> float:
    ph = r.get("phases") or {}
    return float(ph.get("prep") or 0.0) + float(ph.get("analyse") or 0.0)


def arm_row(r: dict, ref_always: dict | None, ref_unint: dict | None, tables: list[str]) -> dict:
    a = r.get("analysis") or {}
    row = {
        "outcome": r.get("outcome"), "attempt": r.get("attempt"), "failed_phase": r.get("failed_phase"),
        "error": r.get("error"), "wall_s": r.get("wall_s"), "runtime_s": r.get("runtime_s"),
        "est_runtime_cost_usd": r.get("est_runtime_cost_usd"), "runtimes": r.get("runtimes"), "phases": r.get("phases"),
        "create_failures": r.get("create_failures") or [],
        "cpu_phase_s": _cpu_phase_s(r), "work_recomputed_s": r.get("work_recomputed_s"),
        "work_recomputed_steps": r.get("work_recomputed_steps"), "work_recomputed": r.get("work_recomputed"),
        "app_loc": r.get("app_loc"), "checkpoint_contents": r.get("checkpoint_contents"),
        "vs_always": {}, "vs_uninterrupted": {},
    }
    for ref, key in ((ref_always, "vs_always"), (ref_unint, "vs_uninterrupted")):
        if ref is None or ref.get("outcome") != "completed" or r.get("outcome") != "completed":
            row[key] = None
            continue
        row[key] = {"wall": _ratio(r.get("wall_s"), ref.get("wall_s")),
                    "cost": {t: _ratio(_cost(r, t), _cost(ref, t)) for t in tables}}
    if a:
        row["analysis"] = {"accuracy": a.get("accuracy"), "mean_loss": a.get("mean_loss"),
                           "tta_accuracy": (a.get("tta_accuracy") or {}).get("accuracy"),
                           "representation_mean_cosine": (a.get("representation_shift") or {}).get("mean_cosine"),
                           "standardisation_consistent": (a.get("input_standardisation") or {}).get("consistent"),
                           "missing_dependencies": a.get("missing_dependencies"),
                           "metric_errors": a.get("metric_errors"),
                           "final_params_digest": a.get("final_params_digest")}
    return row


def fingerprint_block(recs: dict[str, dict]) -> dict:
    ref_arm = "switch" if _fp(recs.get("switch"), "train_input") else (
        "uninterrupted" if _fp(recs.get("uninterrupted"), "train_input") else None)
    ref = recs.get(ref_arm) if ref_arm else None
    per_arm = {}
    for arm, r in recs.items():
        fi, fo = _fp(r, "train_input"), _fp(r, "train_output")
        own, arts = own_prep_diff(r), artefacts_vs_prep(r)
        if fi is None and own is None and arts is None:
            continue
        per_arm[arm] = {
            # within the arm: what carrying or recomputing the state did
            "train_input_vs_own_prep_output": own,
            "analysis_artefacts_vs_own_prep": arts,
            # across arms: a measurement (see CROSS_ARM_NOTE)
            "train_input_differs_from_reference": differing(fi, _fp(ref, "train_input")) if ref and fi else None,
            "train_output_differs_from_reference": (differing(fo, _fp(ref, "train_output"), OUTPUT_COMPONENTS)
                                                    if ref and fo else None),
            "prep_output_differs_from_reference": (differing(_prep_fp(r), _prep_fp(ref), GATED)
                                                   if ref and _prep_fp(r) and _prep_fp(ref) else None),
            "prep_host": _host(prep_witness(r, "train")),
            "train_host": _host(r.get("train")),
            "module_training": ((fi or {}).get("module_training") or {}),
            "rng_cuda": ((fi or {}).get("rng") or {}).get("cuda"),
        }
    final = {}
    for arm, r in recs.items():
        d = (r.get("analysis") or {}).get("final_params_digest")
        if d:
            final.setdefault(d, []).append(arm)
    return {"reference": ref_arm, "noted_exceptions": NOTED_EXCEPTIONS, "cross_arm_note": CROSS_ARM_NOTE,
            "per_arm": per_arm,
            "identical_prep_output_groups": _groups(recs, _prep_fp, GATED),
            "identical_train_input_groups": _groups(recs, lambda r: _fp(r, "train_input"), INPUT_COMPONENTS),
            "identical_train_input_groups_excluding_noted": _groups(recs, lambda r: _fp(r, "train_input"), GATED),
            "identical_train_output_groups": _groups(recs, lambda r: _fp(r, "train_output"), OUTPUT_COMPONENTS),
            "identical_final_params_groups": sorted(final.values(), key=lambda g: (-len(g), g))}


CROSSOVER_NOTE = (
    "One row per invocation (run_id), pairing switch, always and uninterrupted within that invocation, so a pass "
    "count run twice has two rows. No switch-cost-equals-always point is estimated: each pass count holds one or "
    "two single runs (no intervals), and a line drawn between two neighbouring pass counts would ignore every other "
    "row, including any later reversal of the sign. switch_minus_always lists the measured differences and where "
    "their sign changes; that is the result. cpu_phase_s is the switch arm's prep + analyse phase wall time.")


def _arm_cell(r: dict | None) -> dict | None:
    if r is None:
        return None
    return {"attempt": r.get("attempt"), "outcome": r.get("outcome"), "wall_s": r.get("wall_s"),
            "runtime_s": r.get("runtime_s"), "cost": r.get("est_runtime_cost_usd")}


def crossover(records: list[dict], tables: list[str]) -> dict | None:
    """Switch against always per invocation, at every pass count and attempt.

    Every invocation that ran switch or always at a pass count is a row (in
    file order within the pass count); ratios are filled only when both arms
    completed in that invocation. None unless completed pairs exist at two
    or more pass counts."""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        p = r.get("prep_passes")
        if p is None:
            continue
        groups.setdefault((p, (r.get("run") or {}).get("run_id")), []).append(r)
    order = {k: i for i, k in enumerate(groups)}
    rows = []
    for key in sorted(groups, key=lambda k: (k[0], order[k])):
        p, run_id = key
        inv = latest(groups[key])
        s, a, u = inv.get("switch"), inv.get("always"), inv.get("uninterrupted")
        if not (s or a):
            continue
        both = bool(s and a and s.get("outcome") == a.get("outcome") == "completed")
        row = {"prep_passes": p, "run_id": run_id,
               "cpu_phase_s": _cpu_phase_s(s) if s and s.get("outcome") == "completed" else None,
               "prep_in_kernel_s": ((s or {}).get("prep") or {}).get("prep_seconds"),
               "switch": _arm_cell(s), "always": _arm_cell(a),
               "wall_ratio_switch_over_always": _ratio(s.get("wall_s"), a.get("wall_s")) if both else None,
               "cost_ratio_switch_over_always": {t: _ratio(_cost(s, t), _cost(a, t)) for t in tables} if both else None}
        if u and u.get("outcome") == "completed":
            row["uninterrupted"] = _arm_cell(u)
        rows.append(row)
    paired = [r for r in rows if r["cost_ratio_switch_over_always"] is not None]
    if len({r["prep_passes"] for r in paired}) < 2:
        return None
    diffs = {}
    for t in tables:
        pts = []
        for r in paired:
            s_, a_ = (r["switch"]["cost"] or {}).get(t), (r["always"]["cost"] or {}).get(t)
            if s_ is not None and a_ is not None:
                pts.append({"prep_passes": r["prep_passes"], "run_id": r["run_id"],
                            "attempt": r["switch"]["attempt"], "usd": s_ - a_,
                            "switch": "cheaper" if s_ < a_ else ("dearer" if s_ > a_ else "equal")})
        signs: dict[int, list[str]] = {}
        for pt in pts:
            signs.setdefault(pt["prep_passes"], []).append(pt["switch"])
        ps = sorted(signs)
        diffs[t] = {"by_run": pts,
                    "switch_by_prep_passes": {str(p): signs[p] for p in ps},
                    "mixed_within_prep_passes": [p for p in ps if len(set(signs[p])) > 1],
                    "sign_changes_between_prep_passes": [[p0, p1] for p0, p1 in zip(ps, ps[1:])
                                                         if set(signs[p0]) != set(signs[p1])]}
    return {"rows": rows, "switch_minus_always": diffs, "switch_cost_equals_always": None, "note": CROSSOVER_NOTE}


def _block(records: list[dict], tables: list[str]) -> dict:
    recs = latest(records)
    attempts: dict[str, dict] = {}
    for r in records:
        a = attempts.setdefault(r.get("arm"), {"attempts": 0, "completed": 0, "failed": 0})
        a["attempts"] += 1
        a["completed" if r.get("outcome") == "completed" else "failed"] += 1
    return {
        "attempts": attempts,
        "arms": {arm: arm_row(r, recs.get("always"), recs.get("uninterrupted"), tables) for arm, r in recs.items()},
        "fingerprints": fingerprint_block(recs),
        "lr": {arm: lr_summary(r.get("train")) for arm, r in recs.items() if r.get("train")},
        "checks": checks(records),
    }


def summarise(records: list[dict], cohorts: list[str] | None = None) -> dict:
    ours = [r for r in records if r.get("schema") == SCHEMA]
    ignored = len(records) - len(ours)
    names = cohorts or sorted({r.get("cohort") for r in ours if r.get("cohort")})
    out = {"schema": "e14_summary/1", "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "ignored_records_without_schema": ignored, "cohorts": {}}
    for c in names:
        rs = [r for r in ours if r.get("cohort") == c]
        if not rs:
            continue
        tables = sorted({t for r in rs for t in (r.get("est_runtime_cost_usd") or {})})
        passes = sorted({r.get("prep_passes") for r in rs if r.get("prep_passes") is not None})
        runs = sorted({(r.get("run") or {}).get("run_id") for r in rs if (r.get("run") or {}).get("run_id")})
        out["cohorts"][c] = {
            "records": len(rs), "runs": runs, "local_simulation": sorted({bool(r.get("local_simulation")) for r in rs}),
            "data_source": sorted({str(r.get("data_source")) for r in rs}),
            "gpu_profiles": sorted({(r.get("profiles") or {}).get("gpu") for r in rs if r.get("profiles")}),
            "steps": sorted({r.get("steps") for r in rs}),
            "primary_rate_table": rs[-1].get("primary_rate_table"), "rate_tables": rs[-1].get("rates"),
            "cost_label": "estimated runtime cost", "clock_rule": rs[-1].get("clock_rule"),
            "prep_passes": passes,
            "by_prep_passes": {str(p): _block([r for r in rs if r.get("prep_passes") == p], tables) for p in passes},
            "crossover": crossover(rs, tables),
        }
    return out


# ---- verification checks ------------------------------------------------------

def _check(name: str, ok, detail, gate: bool = True) -> dict:
    return {"name": name, "ok": ok, "detail": detail, "gate": gate}


def checks(records: list[dict], requested: list[str] | None = None) -> list[dict]:
    """Checks over ONE prep_passes group of one cohort (callers group).

    `requested` is the arms the run asked for (the harness passes them; by
    default, every arm with a record). `ok` is True, False, or None. None
    means NOT APPLICABLE and is kept for an arm that was not requested; a
    requested arm that failed, or lacks the witness a check needs, makes that
    check False, so a switch that is refused or crashes cannot pass."""
    recs = latest([r for r in records if r.get("schema", SCHEMA) == SCHEMA])
    out = []
    local = all(r.get("local_simulation") for r in recs.values()) if recs else False
    wanted = list(dict.fromkeys(requested)) if requested else list(recs)
    exp_fail = sorted(a for a in wanted if expected_failure(a, recs.get(a), local))

    # 0. every requested arm ran to completion (bar the one planned failure)
    bad0 = {a: _why(recs.get(a)) for a in wanted
            if a not in exp_fail and (recs.get(a) or {}).get("outcome") != "completed"}
    out.append(_check("requested_arms_completed", not bad0 if wanted else None,
                      {"requested": wanted, "not_completed": bad0, "expected_failures": exp_fail}))

    # 1. application LOC is counted from the marked blocks
    app = [a for a in wanted if a.startswith("appckpt")]
    if app:
        bad, detail = [], {}
        for a in app:
            r = recs.get(a) or {}
            c = r.get("app_code") or {}
            lines = c.get("lines") or []
            detail[a] = r.get("app_loc")
            if not (r.get("app_loc") and r.get("app_loc") == c.get("total_loc") == len(lines)
                    == c.get("save_loc", 0) + c.get("rebuild_loc", 0) + c.get("load_loc", 0)):
                bad.append(a)
        if "appckpt_complete" in app and "appckpt_incomplete" in app:
            if not (recs.get("appckpt_complete") or {}).get("app_loc", 0) > (recs.get("appckpt_incomplete") or {}).get("app_loc", 0):
                bad.append("complete not larger than incomplete")
        out.append(_check("app_loc_counted_from_markers", not bad, {"loc": detail, "bad": bad}))
    else:
        out.append(_check("app_loc_counted_from_markers", None, "no application-checkpoint arm requested"))

    # 2. the scheduler steps once per completed epoch and lr does not collapse
    if wanted:
        bad, lrs = {}, {}
        for a in wanted:
            s = lr_summary((recs.get(a) or {}).get("train"))
            if s is None:
                bad[a] = f"no TRAIN witness ({_why(recs.get(a))})"
                continue
            lrs[a] = s
            if not s["per_epoch"] or s["collapsed"]:
                bad[a] = {k: s[k] for k in ("per_epoch", "collapsed", "epochs_completed", "epochs_expected",
                                            "scheduler_epochs_advanced")}
        ex = next(iter(lrs.values()), None)
        out.append(_check("lr_per_epoch", not bad,
                          {"bad": bad, "example": ex and {k: ex[k] for k in (
                              "lr_first", "lr_last", "change_steps", "epoch_boundaries", "epochs_completed",
                              "epochs_expected", "scheduler_epochs_advanced")}}))
    else:
        out.append(_check("lr_per_epoch", None, "no arm requested"))

    # 3. determinism flags and data provenance
    if wanted:
        bad = []
        for a in wanted:
            t = (recs.get(a) or {}).get("train") or {}
            d = t.get("determinism") or {}
            if not (d.get("cudnn_deterministic") is True and d.get("cudnn_benchmark") is False):
                bad.append(a)
        out.append(_check("determinism_flags_set", not bad, {"bad": bad}))
    missing_ds = [a for a in wanted if (recs.get(a) or {}).get("outcome") == "completed" and not recs[a].get("data_source")]
    out.append(_check("data_source_recorded", not missing_ds if wanted else None,
                      {"data_source": sorted({str(r.get("data_source")) for r in recs.values()}), "missing": missing_ds}))

    # 4. WITHIN each arm: the TRAIN input equals the output of the PREP that
    #    fed it. This is carry fidelity isolated from the host: both
    #    fingerprints describe the same state, before and after the move, so
    #    it holds on a live run whatever the hosts' arithmetic. It FAILS on a
    #    broken switch (its RNG streams differ) and on a refused or crashed one
    #    (no TRAIN witness).
    def own_equal(name, arms_):
        considered = [a for a in arms_ if a in wanted]
        if not considered:
            return _check(name, None, f"none of {list(arms_)} requested")
        per, bad = {}, []
        for a in considered:
            d = own_prep_diff(recs.get(a))
            if d is None:
                per[a] = {"no_witness": _why(recs.get(a))}
                bad.append(a)
                continue
            per[a] = d
            if d["differs"]:
                bad.append(a)
        return _check(name, not bad, {"bad": bad, "per_arm": per})

    out.append(own_equal("switch_train_input_equals_own_prep_output", ["switch"]))
    out.append(own_equal("complete_train_input_equals_own_prep_output", ["appckpt_complete"]))
    # Nothing between PREP and TRAIN may change the state when nothing moves
    # (and fingerprinting itself must draw nothing).
    out.append(own_equal("unmoved_train_input_equals_own_prep_output", ["uninterrupted", "always", "restart"]))

    # 5. the incomplete checkpoints lose the loader generator, and only what
    #    they omitted: nothing they saved may differ from their own PREP output.
    considered = [a for a in OMITS_PREP if a in wanted]
    if considered:
        per, bad = {}, []
        for a in considered:
            d = own_prep_diff(recs.get(a))
            if d is None:
                per[a] = {"no_witness": _why(recs.get(a))}
                bad.append(a)
                continue
            wrong = [c for c in d["differs"] if c in INCOMPLETE_SAVED]
            per[a] = {**d, "saved_items_that_differ": wrong}
            if "loader_generator" not in d["differs"] or wrong:
                bad.append(a)
        out.append(_check("incomplete_differs_from_own_prep_in_loader_generator", not bad, {"bad": bad, "per_arm": per}))
    else:
        out.append(_check("incomplete_differs_from_own_prep_in_loader_generator", None, "no incomplete checkpoint arm requested"))

    # 6. the prep artefacts reach ANALYSE unchanged: at the top of ANALYSE
    #    their digests equal the ones PREP recorded (presence alone would pass
    #    a checkpoint that restored stale or wrong artefacts of the right
    #    shape). The checkpoints that omit them must arrive with none.
    considered = [a for a in wanted if a not in exp_fail]
    if considered:
        per, bad = {}, []
        for a in considered:
            v = artefacts_vs_prep(recs.get(a))
            if v is None:
                per[a] = {"no_witness": _why(recs.get(a))}
                bad.append(a)
                continue
            per[a] = {"missing": v["missing_at_analysis"], "differ": v["differ_from_prep"]}
            ok = (set(v["missing_at_analysis"]) == set(v["compared"]) if a in OMITS_PREP
                  else not v["missing_at_analysis"] and not v["differ_from_prep"])
            if not ok:
                bad.append(a)
        out.append(_check("prep_artefacts_reach_analysis_unchanged", not bad, {"bad": bad, "per_arm": per}))
    else:
        out.append(_check("prep_artefacts_reach_analysis_unchanged", None, "no arm requested"))

    # 7. ANALYSE names the missing dependencies of the incomplete arms and no
    #    others, with no other metric error anywhere
    considered = [a for a in wanted if a not in exp_fail]
    if considered:
        bad, detail = [], {}
        for a in considered:
            w = (recs.get(a) or {}).get("analysis")
            if not w:
                detail[a] = {"no_witness": _why(recs.get(a))}
                bad.append(a)
                continue
            miss, errs = w.get("missing_dependencies") or {}, w.get("metric_errors") or {}
            detail[a] = {"missing": miss, "errors": errs}
            if a in OMITS_PREP:
                if set(miss) != set(PREP_METRICS) or not all(v.startswith("KeyError: ") for v in miss.values()) or errs:
                    bad.append(a)
            elif miss or errs:
                bad.append(a)
        out.append(_check("analysis_reports_missing_dependencies", not bad, {"bad": bad, "per_arm": detail}))
    else:
        out.append(_check("analysis_reports_missing_dependencies", None, "no arm requested"))

    # 8. the standardisation check agrees with the statistics wherever it ran
    ran = {a: ((recs.get(a) or {}).get("analysis") or {}).get("input_standardisation") for a in wanted}
    ran = {a: v for a, v in ran.items() if v}
    if ran:
        bad = [a for a, v in ran.items() if v.get("consistent") is not True]
        out.append(_check("input_standardisation_consistent", not bad,
                          {"bad": bad, "per_arm": {a: {k: v.get(k) for k in ("max_abs_channel_mean",
                                                                             "max_abs_channel_std_minus_1")}
                                                   for a, v in ran.items()}}))
    else:
        out.append(_check("input_standardisation_consistent", None, "no arm computed it"))

    # 9. ACROSS arms: equalities that need two sandboxes to build bit-identical
    #    state. Gates on a local dry run (one CPU, one thread count), where
    #    they must hold and FAIL on a broken switch; informational live, where
    #    the hosts' arithmetic can differ (CROSS_ARM_NOTE). Each carries both
    #    arms' PREP hosts and whether their PREP outputs were already equal.
    def cross(name, a, b, comps=GATED, which="train_input"):
        base = {"cross_arm": True, "note": None if local else CROSS_ARM_NOTE}
        if a not in wanted or b not in wanted:
            return _check(name, None, f"needs {a} and {b} (not both requested)", gate=local)
        fa, fb = _fp(recs.get(a), which), _fp(recs.get(b), which)
        if fa is None or fb is None:
            return _check(name, False, {**base, "no_witness": {x: _why(recs.get(x)) for x, f in ((a, fa), (b, fb))
                                                               if f is None}}, gate=local)
        d = differing(fa, fb, comps)
        noted = ([c for c in NOTED_EXCEPTIONS if component(fa, c) != component(fb, c)]
                 if which == "train_input" else [])
        pa, pb = _prep_fp(recs.get(a)), _prep_fp(recs.get(b))
        return _check(name, not d, {**base, "differs": d, "noted_exceptions_differing": noted,
                                    "prep_outputs_differ": differing(pa, pb, GATED) if pa and pb else None,
                                    "prep_hosts": {a: _host(prep_witness(recs.get(a), "train")),
                                                   b: _host(prep_witness(recs.get(b), "train"))}},
                      gate=local)

    out.append(cross("switch_train_input_equals_uninterrupted", "switch", "uninterrupted"))
    out.append(cross("restart_train_input_equals_uninterrupted", "restart", "uninterrupted"))
    out.append(cross("complete_train_input_equals_switch", "appckpt_complete", "switch"))
    out.append(cross("complete_train_output_equals_switch", "appckpt_complete", "switch",
                     OUTPUT_COMPONENTS, "train_output"))

    # 10. the complete checkpoint's analysis equals the switch's: final
    #     weights and every metric, including the artefact-dependent ones.
    if "appckpt_complete" in wanted and "switch" in wanted:
        fa = (recs.get("appckpt_complete") or {}).get("analysis")
        fb = (recs.get("switch") or {}).get("analysis")
        if not fa or not fb:
            out.append(_check("complete_analysis_equals_switch", False,
                              {"no_witness": {x: _why(recs.get(x)) for x, f in (("appckpt_complete", fa), ("switch", fb))
                                              if not f}}, gate=local))
        else:
            def vals(w):
                return {"final_params_digest": w.get("final_params_digest"), "accuracy": w.get("accuracy"),
                        "mean_loss": w.get("mean_loss"), "tta_accuracy": w.get("tta_accuracy"),
                        "representation_shift": w.get("representation_shift"),
                        "input_standardisation": w.get("input_standardisation")}
            va, vb = vals(fa), vals(fb)
            diff = [k for k in va if va[k] != vb[k]]
            out.append(_check("complete_analysis_equals_switch", not diff,
                              {"differs": diff, "final_params": [va["final_params_digest"], vb["final_params_digest"]],
                               "tta_accuracy": [(va["tta_accuracy"] or {}).get("accuracy"),
                                                (vb["tta_accuracy"] or {}).get("accuracy")],
                               "mean_cosine": [(va["representation_shift"] or {}).get("mean_cosine"),
                                               (vb["representation_shift"] or {}).get("mean_cosine")],
                               "note": None if local else CROSS_ARM_NOTE}, gate=local))
    else:
        out.append(_check("complete_analysis_equals_switch", None, "needs appckpt_complete and switch", gate=local))

    # 11. the clock: intervals well formed, per-SKU sums consistent, the switch
    #     arm's intervals contiguous, application-checkpoint moves overlapping,
    #     and no create request whose outcome could not be established.
    bad, detail = [], {}
    for a in wanted:
        r = recs.get(a)
        if r is None:
            continue
        ivs = r.get("runtimes") or []
        unresolved = [c["name"] for c in (r.get("create_failures") or []) if not c.get("resolved")]
        if not ivs:
            detail[a] = {"intervals": 0, "unresolved_creates": unresolved}
            if unresolved:
                bad.append(a)
            continue
        tot = sum(iv["seconds"] for iv in ivs)
        per = sum((r.get("runtime_s") or {}).values())
        span = max(iv["end_s"] for iv in ivs) - min(iv["start_s"] for iv in ivs)
        ok = all(iv["end_s"] >= iv["start_s"] and abs(iv["seconds"] - (iv["end_s"] - iv["start_s"])) < 1e-9 for iv in ivs)
        ok = ok and abs(tot - per) < 1e-6 and not unresolved
        if a == "switch" and r.get("outcome") == "completed":
            ok = ok and len(ivs) == 3 and all(abs(ivs[i]["end_s"] - ivs[i + 1]["start_s"]) < 1e-9 for i in range(2)) \
                and abs(tot - span) < 1e-6
        if a in ("uninterrupted", "always"):
            ok = ok and len(ivs) == 1
        if a.startswith("appckpt") and r.get("outcome") == "completed":
            ok = ok and tot > span        # the moves overlap two live runtimes, both charged
        detail[a] = {"intervals": len(ivs), "charged_s": round(tot, 3), "span_s": round(span, 3),
                     "create_failures": len(r.get("create_failures") or []), "unresolved_creates": unresolved}
        if not ok:
            bad.append(a)
    out.append(_check("clock_rule_consistent", not bad if detail else None, {"bad": bad, "per_arm": detail}))

    # 12. the naive checkpoint: live it must fail at the CPU load; locally
    #     there is no CUDA, so it cannot fail and the check is informational.
    n = recs.get("appckpt_naive")
    if n and "appckpt_naive" in wanted:
        if local:
            out.append(_check("naive_fails_on_cpu_load", None,
                              f"local dry run has no CUDA, so the naive load cannot fail (outcome {n.get('outcome')})",
                              gate=False))
        else:
            out.append(_check("naive_fails_on_cpu_load", expected_failure("appckpt_naive", n, local),
                              {"outcome": n.get("outcome"), "failed_phase": n.get("failed_phase"),
                               "error": (n.get("error") or "")[:300]}, gate=False))
    return out


def print_checks(results: list[dict]) -> None:
    for c in results:
        mark = "n/a " if c["ok"] is None else ("PASS" if c["ok"] else "FAIL")
        gate = "" if c.get("gate") else " (informational)"
        print(f"  {mark}  {c['name']}{gate}  {json.dumps(c['detail'], default=str)[:400]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(ROOT / "results" / "e14" / "e14_arms.jsonl"))
    ap.add_argument("--out", default=None, help="summary JSON (default: e14_summary.json next to --in)")
    ap.add_argument("--cohort", action="append", default=None, help="cohort(s) to summarise (default: all)")
    ap.add_argument("--checks", action="store_true", help="print the checks and exit non-zero if a gate fails")
    args = ap.parse_args()
    records = load(args.inp)
    summary = summarise(records, args.cohort)
    summary["input"] = str(args.inp)
    out = Path(args.out) if args.out else Path(args.inp).with_name("e14_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(f"wrote {out}  ({len(summary['cohorts'])} cohort(s); {summary['ignored_records_without_schema']} "
          f"older records without a schema ignored)")
    failed = False
    for c, cs in summary["cohorts"].items():
        print(f"\ncohort {c}  data_source={cs['data_source']}  primary rates={cs['primary_rate_table']}")
        table = cs["primary_rate_table"]
        for p, blk in cs["by_prep_passes"].items():
            print(f"  prep_passes={p}")
            print(f"    {'arm':<20} {'outcome':<10} {'wall s':>7} {'est $':>9} {'x always':>8} {'x unint':>8} "
                  f"{'recomp s':>8} {'LOC':>4} {'acc':>7} {'tta':>7}  TRAIN input vs own PREP output | vs reference")
            fps = blk["fingerprints"]["per_arm"]
            for arm, row in blk["arms"].items():
                ana = row.get("analysis") or {}
                cost = (row.get("est_runtime_cost_usd") or {}).get(table)
                va = ((row.get("vs_always") or {}).get("cost") or {}).get(table)
                vu = ((row.get("vs_uninterrupted") or {}).get("cost") or {}).get(table)
                f = lambda v, w=7, d=3: f"{v:>{w}.{d}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"  # noqa: E731
                fa = fps.get(arm) or {}
                own = (fa.get("train_input_vs_own_prep_output") or {}).get("differs")
                print(f"    {arm:<20} {str(row['outcome']):<10} {f(row.get('wall_s'), 7, 1)} {f(cost, 9, 5)} {f(va, 8, 2)} "
                      f"{f(vu, 8, 2)} {f(row.get('work_recomputed_s'), 8, 1)} {row.get('app_loc') or 0:>4} "
                      f"{f(ana.get('accuracy'), 7, 4)} {f(ana.get('tta_accuracy'), 7, 4)}  "
                      f"{own} | {fa.get('train_input_differs_from_reference')}")
            print(f"    identical PREP output (across arms): {blk['fingerprints']['identical_prep_output_groups']}")
            print(f"    identical TRAIN input (excluding noted): {blk['fingerprints']['identical_train_input_groups_excluding_noted']}")
            print(f"    identical TRAIN output: {blk['fingerprints']['identical_train_output_groups']}")
            if args.checks:
                print_checks(blk["checks"])
                failed |= any(ch["ok"] is False and ch.get("gate") for ch in blk["checks"])
        if cs.get("crossover"):
            print("  crossover (switch vs always, one row per invocation):")
            for r in cs["crossover"]["rows"]:
                cr = (r.get("cost_ratio_switch_over_always") or {}).get(table)
                wr = r.get("wall_ratio_switch_over_always")
                print(f"    passes {r['prep_passes']:>4}  attempt {(r.get('switch') or r.get('always'))['attempt']}  "
                      f"{r['run_id']}  cost ratio {cr if cr is None else f'{cr:.2f}'}  "
                      f"wall ratio {wr if wr is None else f'{wr:.2f}'}")
            sm = cs["crossover"]["switch_minus_always"].get(table) or {}
            print(f"    switch vs always by passes: {sm.get('switch_by_prep_passes')}; sign changes between "
                  f"{sm.get('sign_changes_between_prep_passes')}; no break-even is estimated")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
