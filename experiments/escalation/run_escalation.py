"""Run policy P3 — runtime switching — for both escalation experiments.

Only P3 is executed. The other policies (always large, reactive restart,
application checkpoint) were derived from the per-tier throughputs this script
records; the paper reports only the moves themselves, which
`experiments/rederive.py escalation` tallies from the records.

The recovery contract, and it is deliberate: the system never anticipates the
OOM. The workload genuinely attempts the higher-demand phase on the tier it no
longer fits, genuinely receives CUDA OOM, discards the failed attempt, switches
runtime, and retries. Nothing is pre-emptively migrated.

    last valid state S_t -> attempt phase -> CUDA OOM -> discard
                         -> switch runtime -> restore S_t -> retry

    python experiments/escalation/run_escalation.py --exp gpt2 --out /tmp/soma-rerun/escalation_runs.jsonl
    python experiments/escalation/run_escalation.py --exp vit --out /tmp/soma-rerun/escalation_runs.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# experiments/ on sys.path, so `escalation` imports as a package; src/ for runmeta.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from escalation.client import Client, emit  # noqa: E402
from escalation.teardown import teardown_one  # noqa: E402
from escalation import workloads as W  # noqa: E402
from runmeta import shipped_record_conflict  # noqa: E402

#: Repository root (this file is experiments/escalation/run_escalation.py).
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "escalation", "escalation_runs.jsonl")

#: Steps per execute cell. Keeps any single cell well inside its budget so one
#: overrun cannot discard a whole run, and yields per-chunk throughput samples.
CHUNK_STEPS = 150
#: Explicit cell budget. The route defaults to 120 s and INTERRUPTS the kernel
#: past it; the HTTP read timeout is a separate budget and does not extend it.
CELL_TIMEOUT_MS = 1_800_000

# Calibrated on the real hardware (results/escalation/escalation_calibration.jsonl).
# Each phase is a measured regime, not an assumption:
#   GPT-2  seq 256 fits T4 at 5.1 GB; seq 1024 needs 16.2 GB and OOMs T4's 14.56.
#   ViT    384 fits T4 at 12.1 GB; 448 needs 16.2 and OOMs T4; 512 needs 20.8
#          and OOMs L4 despite L4's nominal 22.03, because the allocator's
#          reserved overhead sits on top of peak allocated.
PLANS = {
    "gpt2": {
        "workload": "gpt2",
        "total_steps": 200,
        "tiers": ["gpu_t4", "gpu_a100_40"],
        "phases": [
            {"dim": 256, "batch": 8, "tier": "gpu_t4"},
            {"dim": 1024, "batch": 8, "tier": "gpu_a100_40"},
        ],
    },
    "vit": {
        "workload": "vit",
        "total_steps": 30,
        "tiers": ["gpu_t4", "gpu_l4", "gpu_a100_40"],
        "phases": [
            {"dim": 384, "batch": 32, "tier": "gpu_t4"},
            {"dim": 448, "batch": 32, "tier": "gpu_l4"},
            {"dim": 512, "batch": 32, "tier": "gpu_a100_40"},
        ],
    },
}


def build_src(workload: str, total_steps: int, initial_dim: int = 224) -> str:
    """Render the in-sandbox workload source.

    ViT's patch count is fixed by the model's `image_size`, so the model must be
    BUILT at phase 0's resolution. Building at 224 and then feeding 384 asserts
    inside torchvision rather than OOMing, which is a harness error wearing the
    costume of a result.
    """
    src = W.GPT2_BUILD if workload == "gpt2" else W.VIT_BUILD
    return (src.replace("TOTAL_STEPS_PLACEHOLDER", str(total_steps))
               .replace("INITIAL_RES_PLACEHOLDER", str(initial_dim)))


def escalate_cell(workload: str, dim: int) -> str:
    """Workload-level reshape, run BEFORE the failing attempt, on the old tier."""
    if workload != "vit":
        return ""
    return ('_r = escalate_resolution(%d)\n'
            '_emit(event="reshaped", **_r)\n' % dim)


def do_switch(c: Client, pid: str, tier: str, rec: dict) -> bool:
    t0 = time.perf_counter()
    sw = c.switch(pid, tier)
    wall = (time.perf_counter() - t0) * 1000.0
    rec["switches"].append({
        "to": tier, "status": sw.status, "code": sw.error_code,
        "patch_wall_ms": round(wall, 1),
        "source_phases_ms": sw.payload.get("switchPhases") or {},
    })
    if sw.status != 200:
        rec["failure"] = f"switch to {tier}: {sw.status} {sw.error_code} {sw.error_message}"
        return False
    return True


def verify_cell(c: Client, pid: str, rec: dict, attempts: int = 3) -> dict:
    """First execute after a switch. This is where the restore cost lands.

    Restoring a GPT-2-sized namespace takes minutes, and the destination is
    provisioned lazily, so the first execute can land while hydration is still
    running and come back `sandbox_unavailable`. That is a transient, not a
    result, so it is retried — but the reason is recorded on every attempt,
    because a bare null here is indistinguishable from a genuine loss of state
    and that is exactly the ambiguity that wasted the first sweep.
    """
    last = {}
    for attempt in range(1, attempts + 1):
        t0 = time.perf_counter()
        v = c.execute(pid, W.VERIFY_AFTER_SWITCH, timeout_s=2400,
                      cell_timeout_ms=CELL_TIMEOUT_MS)
        info = emit(v)
        kr = (v.payload or {}).get("kernel_restore") or {}
        p = v.payload or {}
        last = {
            "attempt": attempt,
            "first_execute_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "http_status": v.status,
            "failure_kind": p.get("failure_kind"),
            "cell_error": (str(p.get("error"))[:220] if p.get("error") else None),
            "restore_state": kr.get("state"),
            "restore_var_count": kr.get("var_count"),
            "destination_phases_ms": kr.get("phases") or {},
            "gpu": info.get("gpu"),
            "steps_done": info.get("steps_done"),
            "opt_is_optimizer": info.get("opt_is_optimizer"),
            "opt_step": info.get("opt_step"),
            "device": info.get("device"),
        }
        rec["restores"].append(last)
        if info.get("steps_done") is not None:
            return info
        if p.get("failure_kind") not in ("sandbox_unavailable", "timeout", None):
            break
        print(f"    verify attempt {attempt} inconclusive "
              f"({last['failure_kind']}: {str(last['cell_error'])[:90]}), retrying")
    return {}


def run_one(c: Client, exp: str, fraction: float, rep: int) -> dict:
    plan = PLANS[exp]
    n_total = plan["total_steps"]
    phases = plan["phases"]
    rec = {
        "experiment": exp, "policy": "P3_runtime_switch", "rep": rep,
        "oom_fraction": fraction, "total_steps": n_total,
        "tiers": plan["tiers"], "phases_plan": phases,
        "tier_time_s": {}, "tier_steps": {}, "per_step_s": {},
        "switches": [], "restores": [], "ooms": [], "reshapes": [],
        "failure": None, "wall_s": None,
    }
    t_run = time.perf_counter()
    pid = c.create_project(f"{exp}-f{int(fraction*100)}-r{rep}", profile=phases[0]["tier"])
    rec["project"] = pid
    try:
        if c.start(pid, timeout_s=1200).status != 200:
            rec["failure"] = "source runtime failed to start"
            return rec
        b = c.execute(pid, build_src(plan["workload"], n_total, phases[0]["dim"]), timeout_s=2400,
                      cell_timeout_ms=CELL_TIMEOUT_MS)
        built = emit(b)
        if built.get("event") != "built":
            rec["failure"] = f"build: {json.dumps(b.payload)[:300]}"
            return rec
        rec["build_s"] = built.get("build_s")
        rec["params"] = built.get("params")
        rec["gpu_start"] = built.get("gpu")

        # --- split the total work across phases ----------------------------
        if exp == "gpt2":
            splits = [int(round(n_total * fraction)), n_total - int(round(n_total * fraction))]
        else:
            per = n_total // len(phases)
            splits = [per] * len(phases)
            splits[-1] += n_total - sum(splits)

        for i, ph in enumerate(phases):
            n = splits[i]
            if n <= 0:
                continue
            if i > 0:
                # The workload escalates demand on the tier it is ALREADY on,
                # then genuinely attempts and genuinely fails there.
                esc = escalate_cell(plan["workload"], ph["dim"])
                if esc:
                    r = c.execute(pid, esc, timeout_s=1200, cell_timeout_ms=CELL_TIMEOUT_MS)
                    rec["reshapes"].append(emit(r))
                a = c.execute(pid, W.ATTEMPT_OOM.format(
                    n=1, dim=ph["dim"], batch=ph["batch"]), timeout_s=1800,
                    cell_timeout_ms=CELL_TIMEOUT_MS)
                oom = emit(a)
                rec["ooms"].append(oom)
                if not oom.get("oom"):
                    rec["failure"] = (
                        f"phase {i} did NOT OOM on {phases[i-1]['tier']} "
                        f"(dim={ph['dim']}); the escalation is not bracketed")
                    return rec
                if not do_switch(c, pid, ph["tier"], rec):
                    return rec
                v = verify_cell(c, pid, rec)
                if v.get("steps_done") is None:
                    last = (rec["restores"] or [{}])[-1]
                    rec["failure"] = (
                        f"post-switch verify inconclusive after {last.get('attempt')} "
                        f"attempts: failure_kind={last.get('failure_kind')} "
                        f"error={last.get('cell_error')}")
                    return rec

            # Chunk the phase. One cell per phase would be simpler, but a long
            # phase then depends entirely on the cell budget being right, and a
            # single overrun discards the whole run. Chunking also yields
            # intermediate throughput samples and bounds the blast radius of any
            # one interrupted cell.
            tier = ph["tier"]
            done, secs, last = 0, 0.0, None
            while done < n:
                take = min(CHUNK_STEPS, n - done)
                t0 = time.perf_counter()
                p = c.execute(pid, W.RUN_PHASE.format(
                    n=take, dim=ph["dim"], batch=ph["batch"]),
                    timeout_s=3600, cell_timeout_ms=CELL_TIMEOUT_MS)
                info = emit(p)
                if info.get("event") != "phase_ok":
                    rec["failure"] = (f"phase {i} chunk at step {done}/{n} failed: "
                                      f"{json.dumps(p.payload)[:260]}")
                    return rec
                done += take
                secs += info["seconds"]
                last = info
            rec["tier_time_s"][tier] = round(
                rec["tier_time_s"].get(tier, 0.0) + secs, 3)
            rec["tier_steps"][tier] = rec["tier_steps"].get(tier, 0) + n
            rec["per_step_s"][tier] = round(secs / max(1, n), 4)
            rec.setdefault("peak_gb", {})[tier] = last.get("peak_gb")
            rec.setdefault("loss", []).append(last.get("loss"))

        rec["completed"] = True
    finally:
        rec["wall_s"] = round(time.perf_counter() - t_run, 1)
        try:
            rec["teardown"] = teardown_one(c, pid, verbose=False)
        except Exception as exc:
            rec["teardown"] = {"error": f"{type(exc).__name__}: {exc}"}
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", choices=["gpt2", "vit"], required=True)
    ap.add_argument("--fractions", default="0.25,0.5,0.75,0.9")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--total-steps", type=int, default=None,
                    help="override the plan's job length. Switch overhead is a "
                         "FIXED cost while the work a restart redoes grows with "
                         "progress, so the job has to be long enough that the "
                         "crossover falls inside the swept range rather than "
                         "past its end.")
    ap.add_argument("--out", default=RESULTS,
                    help="JSONL file the run records are appended to "
                         "(default results/escalation/escalation_runs.jsonl, which is "
                         "shipped, so a rerun must name a new file)")
    args = ap.parse_args()
    # The shipped record is the paper's; a rerun appends to its own file.
    why = shipped_record_conflict(args.out, None)
    if why:
        print(why, file=sys.stderr)
        return 2
    if args.total_steps:
        PLANS[args.exp]["total_steps"] = args.total_steps

    fractions = [float(x) for x in args.fractions.split(",")] if args.exp == "gpt2" else [0.0]
    c = Client()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # Rep-major: repetition 1 of every point before any point sees a second.
    # An abort then leaves a balanced sweep rather than some points finished and
    # others empty.
    done = 0
    for rep in range(1, args.reps + 1):
        for f in fractions:
            label = f"{args.exp} f={f:.2f} rep={rep}"
            print(f"\n=== {label} ===", flush=True)
            try:
                rec = run_one(c, args.exp, f, rep)
            except Exception as exc:
                rec = {"experiment": args.exp, "oom_fraction": f, "rep": rep,
                       "failure": f"{type(exc).__name__}: {exc}", "completed": False}
            with open(out, "a") as fh:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
            done += 1
            if rec.get("failure"):
                print(f"  FAILED: {rec['failure'][:200]}")
            else:
                sw = ", ".join(f"{s['to']}:{s['patch_wall_ms']:.0f}ms" for s in rec["switches"])
                print(f"  ok  wall={rec['wall_s']}s  tiers={rec['tier_time_s']}  switches[{sw}]")
    print(f"\n{done} runs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
