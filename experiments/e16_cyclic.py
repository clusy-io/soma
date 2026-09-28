"""E16: two cyclic sessions share one GPU by moving between runtimes.

Each workload alternates a CPU phase (pool scoring and data preparation with
NumPy) and a GPU phase (training a small ResNet-style CNN), the pattern of an
active-learning or data-centric training loop. The two are offset by half a
cycle: while A trains on the GPU, B prepares data on the CPU, and at every
phase boundary both sessions move through the deployed switch path (PATCH
runtimeProfile with state transfer): first the GPU holder moves to the CPU,
then the other session moves to the GPU. So at most one GPU runtime is alive at
a time. The two moves run one after the other, not concurrently: two
concurrent ~85 MB checkpoint read-backs over the operator's network timed out
(e16-micro attempt 1, both moves refused with the source kept). A move refused
that way is retried once after 15 s; every attempt is recorded.

The dedicated baseline (each workload holds its own GPU runtime for its whole
life) is not run: its cost is modelled from the measured phase durations,
charging both workloads the GPU rate for every phase and no moves, which is
the cheapest the baseline could be (it assumes CPU phases run as fast on the
GPU runtime's few cores as on the CPU runtime, which the lifecycle experiment
showed they do not).

Every move is checked: the workload's step counters and a digest of its model
parameters must be equal before and after.

    PYTHONPATH=src:. python experiments/e16_cyclic.py --phase-s 150 --cycles 2 --cohort e16-rerun --out /tmp/soma-rerun/e16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from handoff.controller import MARK, Api  # noqa: E402
from e14_lifecycle import RATE_TABLES, PRIMARY_RATE_TABLE, Runtimes  # noqa: E402
import runmeta  # noqa: E402

SETUP = r'''
import time, json, hashlib, numpy as np, torch, torch.nn as nn
torch.manual_seed(%(seed)d); np.random.seed(%(seed)d)
def _block(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
model = nn.Sequential(_block(3, 64), _block(64, 128), _block(128, 256), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 10))
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
X = torch.randn(1024, 3, 64, 64); Y = torch.randint(0, 10, (1024,))
pool = np.random.rand(20000, 256).astype(np.float32)
work = {"train_steps": 0, "cpu_batches": 0, "phases": 0, "last_loss": None}
def digest_state():
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().to("cpu").contiguous().numpy().tobytes())
    return h.hexdigest()[:16]
print(%(mark)r + json.dumps({"digest": digest_state(), **work}))
'''

CPU_PHASE = r'''
import time, json, hashlib, numpy as np, torch, torch.nn as nn
print("cpu phase start", flush=True)
_t0 = time.time(); _n = 0
while time.time() - _t0 < %(dur)f:
    _w = np.random.rand(256, 64).astype(np.float32)
    _s = pool @ _w
    _sel = np.argpartition(-_s.max(1), 512)[:512]
    pool[_sel] = pool[_sel] * 0.999 + 0.001
    _n += 1
    if _n %% 10 == 0:
        print(".", end="", flush=True)
work["cpu_batches"] += _n; work["phases"] += 1
print()
print(%(mark)r + json.dumps({"kind": "cpu", "seconds": time.time() - _t0, "batches": _n, "digest": digest_state(), **work}))
'''

GPU_PHASE = r'''
import time, json, hashlib, numpy as np, torch, torch.nn as nn
print("gpu phase start", flush=True)
_dev = "cuda" if torch.cuda.is_available() else "cpu"
model.to(_dev)
for _st in opt.state.values():
    for _k, _v in list(_st.items()):
        if torch.is_tensor(_v): _st[_k] = _v.to(_dev)
_t0 = time.time(); _steps = 0
while time.time() - _t0 < %(dur)f:
    _i = torch.randint(0, X.shape[0], (128,))
    _x, _y = X[_i].to(_dev), Y[_i].to(_dev)
    _loss = torch.nn.functional.cross_entropy(model(_x), _y)
    opt.zero_grad(); _loss.backward(); opt.step(); _steps += 1
    if _steps %% 10 == 0:
        print(".", end="", flush=True)
model.to("cpu")
for _st in opt.state.values():
    for _k, _v in list(_st.items()):
        if torch.is_tensor(_v): _st[_k] = _v.to("cpu")
work["train_steps"] += _steps; work["phases"] += 1; work["last_loss"] = float(_loss)
print()
print(%(mark)r + json.dumps({"kind": "gpu", "device": _dev, "gpu": torch.cuda.get_device_name(0) if _dev == "cuda" else None, "seconds": time.time() - _t0, "steps": _steps, "digest": digest_state(), **work}))
'''

PROBE = r'''
import time, json, hashlib, numpy as np, torch, torch.nn as nn
print(%(mark)r + json.dumps({"digest": digest_state(), **work}))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default=os.environ.get("CLUSY_API_URL", "http://localhost:8010"))
    ap.add_argument("--phase-s", type=float, default=150.0)
    ap.add_argument("--cycles", type=int, default=2, help="GPU phases per workload")
    ap.add_argument("--gpu", default="gpu_t4")
    ap.add_argument("--cohort", default="e16-micro")
    ap.add_argument("--out", default=None, help="output directory (default results/e16; a temp dir with --local)")
    ap.add_argument("--local", action="store_true", help="dry run against LocalApi (no GPU; every profile runs on the local CPU)")
    a = ap.parse_args()
    if a.local:
        out = Path(a.out) if a.out else Path(tempfile.gettempdir()) / "clusy-e16-local"
        if (ROOT / "results").resolve() in [out.resolve(), *out.resolve().parents]:
            # A dry run is not a result; it must never land beside live records.
            print("--local refuses to write under results/; pass a scratch --out", file=sys.stderr)
            return 2
    else:
        out = Path(a.out) if a.out else ROOT / "results" / "e16"
        # One file per cohort, written at the end even when the run fails: never
        # replace an existing record, least of all the shipped one.
        record = out / f"{a.cohort}.json"
        why = runmeta.shipped_record_conflict(record, a.cohort)
        if why is None and record.exists():
            why = f"{record} already exists; pass a new --cohort or another --out"
        if why:
            print(why, file=sys.stderr)
            return 2
    out.mkdir(parents=True, exist_ok=True)
    if a.local:
        from localapi import LocalApi
        api, a.api_url = LocalApi(), None
    else:
        api = Api(a.api_url, os.environ["CLUSY_HARNESS_API_KEY"])
    meta = runmeta.start_run("e16", api_url=a.api_url, args=vars(a), cohort=a.cohort)
    log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)  # noqa: E731
    rA, rB = Runtimes(api, "e16A"), Runtimes(api, "e16B")
    rB.t0 = rA.t0
    rec = {"schema": "e16/1", "cohort": a.cohort, "phase_s": a.phase_s, "cycles": a.cycles, "gpu": a.gpu,
           "phases": {"A": [], "B": []}, "moves": [], "checks": [], "error": None}
    names = {"A": rA, "B": rB}
    pids = {}
    try:
        # A starts on the GPU, B on the CPU: half a cycle apart.
        start_profile = {"A": a.gpu, "B": "cpu"}
        for w in "AB":
            pids[w] = names[w].create(start_profile[w])
            res = api.witness(pids[w], SETUP % {"seed": 7 if w == "A" else 11, "mark": MARK})
            log(f"{w} created on {start_profile[w]}, digest {res['digest']}")
        where = dict(start_profile)
        rounds = 2 * a.cycles
        for r in range(rounds):
            results = {}

            def phase(w: str) -> None:
                code = (GPU_PHASE if where[w] != "cpu" else CPU_PHASE) % {"dur": a.phase_s, "mark": MARK}
                t0 = names[w].now()
                res = api.witness(pids[w], code, timeout_ms=int((a.phase_s + 600) * 1000))
                res.update(start_s=t0, end_s=names[w].now(), runtime=where[w], round=r)
                results[w] = res

            ts = [threading.Thread(target=phase, args=(w,)) for w in "AB"]
            [t.start() for t in ts]; [t.join() for t in ts]
            for w in "AB":
                if w not in results:
                    raise RuntimeError(f"phase {r} of {w} did not return")
                rec["phases"][w].append(results[w])
                log(f"round {r} {w} {results[w]['kind']} on {where[w]}: {results[w].get('steps') or results[w].get('batches')} "
                    f"in {results[w]['seconds']:.0f}s")
            if r == rounds - 1:
                break
            # Swap: one move after the other.
            before = {w: api.witness(pids[w], PROBE % {"mark": MARK}) for w in "AB"}
            target = {w: ("cpu" if where[w] != "cpu" else a.gpu) for w in "AB"}
            # The GPU holder leaves first, so a second GPU runtime never overlaps it.
            for w in sorted("AB", key=lambda k: where[k] == "cpu"):
                for attempt in (1, 2):
                    st, payload, info = names[w].switch(pids[w], target[w])
                    info.update(workload=w, round=r, frm=where[w], attempt=attempt)
                    if st != 200:
                        info["refusal"] = json.dumps(payload, default=str)[:400]
                    rec["moves"].append(info)
                    if st == 200:
                        break
                    log(f"move {w} -> {target[w]} attempt {attempt} returned {st}: {info['refusal'][:200]}")
                    time.sleep(15)
                if st != 200:
                    raise RuntimeError(f"move of {w} to {target[w]} returned {st}")
                where[w] = target[w]
                after = api.witness(pids[w], PROBE % {"mark": MARK})
                ok = after == before[w]
                rec["checks"].append({"workload": w, "round": r, "ok": ok, "before": before[w], "after": after})
                log(f"move {w} -> {target[w]}: PATCH {info['patch_return_s'] - info['patch_start_s']:.0f}s, state equal {ok}")
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:500]
        log("ERROR", rec["error"])
    finally:
        for w in "AB":
            names[w].teardown("end of run")
        runmeta.finish_run(meta, api_url=a.api_url)
    sA, sB = rA.summary(), rB.summary()
    rec["runtimes"] = {"A": sA, "B": sB}
    rates = RATE_TABLES[PRIMARY_RATE_TABLE]["usd_per_s"]
    shared = sum((s["est_runtime_cost_usd"][PRIMARY_RATE_TABLE] or 0.0) for s in (sA, sB))
    # Dedicated baseline: every phase of both workloads billed at the GPU rate, no moves.
    ded_s = sum(p["end_s"] - p["start_s"] for w in "AB" for p in rec["phases"][w])
    dedicated = ded_s * rates[a.gpu]
    gpu_s = sum(s["runtime_s"].get(a.gpu, 0.0) for s in (sA, sB))
    rec["summary"] = {"shared_cost_usd": shared, "dedicated_cost_usd_modelled": dedicated,
                      "saving_pct": 100 * (1 - shared / dedicated) if dedicated else None,
                      "gpu_runtime_s_shared": gpu_s, "gpu_runtime_s_dedicated": ded_s,
                      "moves": len(rec["moves"]), "all_checks_ok": all(c["ok"] for c in rec["checks"]),
                      "error": rec["error"]}
    meta["results"] = rec["summary"]
    (out / f"{a.cohort}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    runmeta.write_meta(meta, out / "runs_meta.jsonl")
    print(json.dumps(rec["summary"], indent=1))
    return 0 if not rec["error"] and rec["summary"]["all_checks_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
