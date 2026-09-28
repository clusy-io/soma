"""A live session moved across instruction sets by the research controller, offline.

Three hops through the same controller the cloud experiments use
(`handoff.controller.Controller.migrate`), with kernels in local containers
and no network: arm64 container -> x86-64 container (emulated) -> arm64
container. The session is a NumPy SGD job with the relationships a restore
must keep: a view of the parameters, an optimizer-style dict that references
them, an alias, a reference cycle, two RNG streams (the legacy global one
and a `Generator`), and a workspace log file the job appends to.

Controls: the same job uninterrupted on arm64 and on x86-64. The switched
run's per-step losses are compared bitwise with both; where they differ the
record says at which step, because matrix products run different SIMD kernels
on each ISA (arithmetic, not state). Stored state is compared exactly at
every boundary by the controller itself.

    PYTHONPATH=src:. python experiments/xsub_isa.py --outdir DIR \
        --arm docker:IMAGE --x86 docker:IMAGE@linux/amd64
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from handoff.controller import MARK, Controller, Journal  # noqa: E402
from localapi import LocalApi, parse_substrates, substrate_identity  # noqa: E402

SETUP = r'''
import numpy as np, random, os
random.seed(7); np.random.seed(7)
gen = np.random.default_rng(11)
X = gen.standard_normal((256, 8)); y = X @ np.arange(1., 9.) + 0.1 * gen.standard_normal(256)
W = np.zeros(8)
W_view = W[2:6]
opt = {"lr": 0.02, "step": 0, "m": np.zeros(8), "params": W}
cfg = {"opt": opt}; cfg["self"] = cfg
alias = W
losses = []
os.makedirs("data", exist_ok=True)
open("data/train.log", "w").close()
def train(k):
    for _ in range(k):
        idx = gen.integers(0, 256, 32)
        noise = np.random.standard_normal(8) * 1e-3
        g = 2 * X[idx].T @ (X[idx] @ W - y[idx]) / 32 + noise
        opt["m"] *= 0.9; opt["m"] += g
        W[:] -= opt["lr"] * opt["m"]
        opt["step"] += 1
        loss = float(np.mean((X @ W - y) ** 2))
        losses.append(loss)
        with open("data/train.log", "a") as f:
            f.write("%d %s\n" % (opt["step"], loss.hex()))
'''

PROBE = r'''
import json, platform, numpy as np
_log = open("data/train.log").read().splitlines()
print(%(mark)r + json.dumps({
    "machine": platform.machine(), "system": platform.system(),
    "step": opt["step"], "losses": [float(v).hex() for v in losses],
    "view_shares_W": bool(np.shares_memory(W_view, W)) and W_view.base is W,
    "opt_params_is_W": opt["params"] is W, "alias_is_W": alias is W, "cycle": cfg["self"] is cfg,
    "log_lines": len(_log), "log_matches_losses": [l.split()[1] for l in _log] == [float(v).hex() for v in losses],
    "W": [float(v).hex() for v in W],
}))
'''


def probe(api: LocalApi, pid: str) -> dict:
    return api.witness(pid, PROBE % {"mark": MARK})


def run_uninterrupted(api: LocalApi, profile: str, steps: list[int]) -> dict:
    pid = api.create_project(f"control-{profile}", profile)
    api.witness(pid, SETUP + f"\nprint({MARK!r} + '{{}}')")
    for k in steps:
        api.witness(pid, f"train({k})\nprint({MARK!r} + '{{}}')")
    out = probe(api, pid)
    api.delete_project(pid)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--arm", required=True, help="substrate for the arm64 profile, e.g. docker:IMAGE")
    ap.add_argument("--x86", required=True, help="substrate for the x86-64 profile, e.g. docker:IMAGE@linux/amd64")
    ap.add_argument("--steps", type=int, default=40, help="training steps per block")
    a = ap.parse_args()
    out = Path(a.outdir)
    if (ROOT / "results").resolve() in [out.resolve(), *out.resolve().parents]:
        # The shipped record lives there; a rerun must not replace it.
        print("refusing to write under results/; pass a scratch --outdir", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    subs = parse_substrates(f"arm64={a.arm},x86_64={a.x86}")
    api = LocalApi(workdir=out / "kernels", substrates=subs)
    journal = Journal(out / "journal.sqlite")
    log = lambda *x: print(*x, flush=True)  # noqa: E731
    rec: dict = {"schema": "xsub-isa/1", "substrates": {k: substrate_identity(v) for k, v in subs.items()},
                 "steps_per_block": a.steps, "hops": [], "started": time.time()}
    route = ["arm64", "x86_64", "arm64"]
    try:
        pid = api.create_project("xsub", route[0])
        api.witness(pid, SETUP + f"\nprint({MARK!r} + '{{}}')")
        api.witness(pid, f"train({a.steps})\nprint({MARK!r} + '{{}}')")
        for i, dest in enumerate(route[1:], 1):
            before = probe(api, pid)
            mig = f"xsub-isa-h{i}"
            journal.create(mig, pid, dest)
            t = time.perf_counter()
            res = Controller(api, journal, out / "blobs", log=log).migrate(mig, pid, dest)
            wall = time.perf_counter() - t
            ok = res.get("phase") == "DONE"
            new = res.get("dest") if ok else None
            after = probe(api, new) if ok else None
            rec["hops"].append({"hop": i, "from": route[i - 1], "to": dest, "phase": res.get("phase"),
                                "reason": res.get("reason"), "timings": res.get("timings"),
                                "boundary": res.get("boundary"), "wall_s": round(wall, 2),
                                "before": before, "after": after,
                                "state_equal": bool(after) and all(before[k] == after[k] for k in
                                                                   ("step", "losses", "W", "log_lines"))})
            log(f"[hop{i}] {route[i-1]} -> {dest}: {res.get('phase')} {res.get('reason') or ''} "
                f"{wall:.1f}s on {after and after['machine']}")
            if not ok:
                break
            pid = new
            api.witness(pid, f"train({a.steps})\nprint({MARK!r} + '{{}}')")
        rec["final"] = probe(api, pid)
        api.delete_project(pid)
        blocks = [a.steps] * len(route)
        rec["controls"] = {prof: run_uninterrupted(api, prof, blocks) for prof in ("arm64", "x86_64")}
    finally:
        api.close()
    fin = rec.get("final") or {}
    rel = ("view_shares_W", "opt_params_is_W", "alias_is_W", "cycle", "log_matches_losses")
    first_diff = {}
    for prof, c in (rec.get("controls") or {}).items():
        la, lb = fin.get("losses") or [], c.get("losses") or []
        first_diff[prof] = next((j for j, (x, y) in enumerate(zip(la, lb)) if x != y), None if len(la) == len(lb) else min(len(la), len(lb)))
    rec["summary"] = {
        "hops_done": sum(1 for h in rec["hops"] if h["phase"] == "DONE"), "hops": len(route) - 1,
        "state_equal_at_every_boundary": all(h["state_equal"] for h in rec["hops"]),
        "relationships_kept": {k: fin.get(k) for k in rel},
        "machines": [h["after"]["machine"] for h in rec["hops"] if h.get("after")],
        "steps": fin.get("step"),
        "losses_equal_to_control": {p: d is None for p, d in first_diff.items()},
        "first_differing_step_vs_control": first_diff,
    }
    (out / "xsub_isa.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec["summary"], indent=1))
    s = rec["summary"]
    ok = s["hops_done"] == s["hops"] and s["state_equal_at_every_boundary"] and all(s["relationships_kept"].values())
    print("XSUB-ISA", "OK" if ok else "NOT OK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
