"""Find the memory thresholds empirically, on the real hardware.

The escalation experiments need three measured regimes per workload:

    D1 <= M_small        phase 1 fits the cheap tier
    M_small < D2 <= M_mid    phase 2 does NOT fit the cheap tier
    M_mid   < D3 <= M_large  phase 3 does NOT fit the mid tier

Guessing these from parameter counts does not work: `transformers` selects a
memory-efficient attention kernel when one is available, which changes the
scaling of the dominant term. So the thresholds are measured, using the same
workload source the real runs use.

Runs one tier at a time and tears down between tiers, so a failed calibration
never leaves a large GPU running.

    python experiments/escalation/calibrate.py --workload gpt2 --profile gpu_t4 --out /tmp/soma-rerun/cal.jsonl
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
from escalation import workloads as W  # noqa: E402
from runmeta import shipped_record_conflict  # noqa: E402

#: Repository root (this file is experiments/escalation/calibrate.py).
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "escalation", "escalation_calibration.jsonl")

#: (dim, batch) ladders. `dim` is sequence length for GPT-2 and image
#: resolution for ViT. Ordered cheapest-first so the run stops at the first OOM
#: and does not pay for configurations beyond the threshold.
LADDERS = {
    "gpt2": [(256, 8), (512, 8), (1024, 8), (1024, 16), (1024, 24), (2048, 16)],
    # Batch held constant at 32 so RESOLUTION is the only independent variable:
    # a ladder that moved both would confound "the workload outgrew the tier"
    # with "the batch was raised", and progressive resizing is the realistic
    # driver. Measured on T4: 224 -> 4.8 GB, 384 -> 12.1 GB, and 384@64 OOMs.
    "vit": [(448, 32), (512, 32), (576, 32)],
}

PROBE = r'''
_reset_peak()
_ok, _err, _peak, _per_step = True, None, None, None
_resh = None
try:
    # ViT's patch count is fixed by the model's image_size, so raising the input
    # resolution is a WORKLOAD step (interpolate the positional embedding), not
    # something the probe can skip. Doing it here keeps calibration on the same
    # code path the measured runs use.
    if "cur_res" in dir() and {dim} != cur_res:
        _resh = escalate_resolution({dim})
    _d = train_steps({n}, {dim}, {batch})
    _per_step = round(_d / {n}, 4)
    _peak = _peak_gb()
except torch.cuda.OutOfMemoryError as e:
    _ok = False; _err = "cuda_oom"
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        _ok = False; _err = "cuda_oom"
    else:
        _ok = False; _err = type(e).__name__ + ": " + str(e)[:120]
if not _ok:
    _free_after_oom()
_emit(event="probe", dim={dim}, batch={batch}, fits=_ok, error=_err,
      peak_gb=_peak, per_step_s=_per_step, reshaped=_resh)
'''


def build_src(workload: str, total_steps: int = 100) -> str:
    src = W.GPT2_BUILD if workload == "gpt2" else W.VIT_BUILD
    return src.replace("TOTAL_STEPS_PLACEHOLDER", str(total_steps))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workload", choices=["gpt2", "vit"], required=True)
    ap.add_argument("--profile", default="gpu_t4")
    ap.add_argument("--steps", type=int, default=2, help="steps per probe point")
    ap.add_argument("--stop-after-oom", type=int, default=1,
                    help="stop this many OOMs in; 0 probes the whole ladder")
    ap.add_argument("--out", default=RESULTS,
                    help="JSONL file the probe rows are appended to "
                         "(default results/escalation/escalation_calibration.jsonl, which is "
                         "shipped, so a rerun must name a new file)")
    args = ap.parse_args()
    # The shipped record is the paper's; a rerun appends to its own file.
    why = shipped_record_conflict(args.out, None)
    if why:
        print(why, file=sys.stderr)
        return 2

    c = Client()
    pid = c.create_project(f"cal-{args.workload}", profile=args.profile)
    rows, t_start = [], time.perf_counter()
    print(f"project {pid[:8]} on {args.profile}")
    try:
        r = c.start(pid, timeout_s=1200)
        if r.status != 200:
            print(f"start failed: {r.status} {r.payload}")
            return 1
        b = c.execute(pid, build_src(args.workload), timeout_s=2400)
        built = emit(b)
        if not built.get("event") == "built":
            print("build failed:", json.dumps(b.payload)[:900])
            return 1
        print(f"built on {built['gpu']['name']} ({built['gpu']['vram_gb']} GB), "
              f"{built['params']:,} params, {built['build_s']}s")

        ooms = 0
        for dim, batch in LADDERS[args.workload]:
            e = c.execute(pid, PROBE.format(n=args.steps, dim=dim, batch=batch),
                          timeout_s=1800)
            p = emit(e)
            if not p:
                print(f"  ({dim},{batch}) NO MARKER: {json.dumps(e.payload)[:220]}")
                continue
            p.update(workload=args.workload, profile=args.profile,
                     gpu=built["gpu"], steps=args.steps)
            rows.append(p)
            flag = "fits" if p["fits"] else f"OOM ({p['error']})"
            peak = f"peak {p['peak_gb']} GB" if p["peak_gb"] else ""
            rate = f"{p['per_step_s']}s/step" if p["per_step_s"] else ""
            print(f"  dim={dim:<5} batch={batch:<3} {flag:<18} {peak:<16} {rate}")
            if not p["fits"]:
                ooms += 1
                if args.stop_after_oom and ooms >= args.stop_after_oom:
                    print("  (stopping at first OOM; the threshold is bracketed)")
                    break
    finally:
        # Clear the namespace before deleting: teardown is gated on the same
        # checkpoint barrier as a switch, and dumping a GPT-2-sized namespace
        # nobody wants pushes DELETE past 240 s while the GPU keeps billing.
        from escalation.teardown import teardown_one
        td = teardown_one(c, pid, verbose=False)
        print(f"teardown: {td['delete']} in {td['seconds']}s  "
              f"elapsed {time.perf_counter() - t_start:.0f}s")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"records -> {out}")

    fits = [r for r in rows if r["fits"]]
    oom = [r for r in rows if not r["fits"]]
    if fits and oom:
        print(f"\nTHRESHOLD on {args.profile}: largest fitting "
              f"{(fits[-1]['dim'], fits[-1]['batch'])} at {fits[-1]['peak_gb']} GB; "
              f"first OOM {(oom[0]['dim'], oom[0]['batch'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
