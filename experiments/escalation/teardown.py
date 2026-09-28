"""Tear down an escalation project without paying for a capture nobody wants.

Deletion is gated on the same durable-checkpoint barrier as a switch, which is
correct (refusing to drop rows whose state you cannot confirm is saved beats
dropping them), but it means teardown of a GPT-2-sized namespace has to dump and
upload ~1.5 GB before the project can go. Measured: DELETE and even pause both
exceed 240 s on a live GPT-2 run.

So clear the namespace first. By teardown the run's measurements are already
recorded, the state is deliberately being discarded, and a capture of an empty
namespace is trivial. This turns a multi-minute teardown into a fast one and,
more importantly, stops a failed run from holding a GPU while its teardown hangs.

    python experiments/escalation/teardown.py --all
    python experiments/escalation/teardown.py --project <uuid>
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# experiments/ on sys.path, so `escalation` imports as a package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from escalation.client import Client  # noqa: E402

CLEAR = r'''
import gc
try:
    import torch
except Exception:
    torch = None
_keep = {"gc", "torch", "json", "os", "sys", "time", "math"}
for _n in [k for k in list(globals()) if not k.startswith("_") and k not in _keep]:
    try:
        del globals()[_n]
    except Exception:
        pass
gc.collect()
if torch is not None and torch.cuda.is_available():
    torch.cuda.empty_cache()
print("__ESC__:{\"event\":\"cleared\"}")
'''


def pause_fallback(c: Client, pid: str) -> dict:
    """Stop the runtime when DELETE failed.

    A switch interrupted before the destination finished hydrating can leave a
    project whose execute is blocked and whose DELETE is refused, so the GPU
    keeps billing. Pause is the fallback the API offers. If it is refused too,
    the runtime has to be stopped from the provider's console. (The recorded
    runs used an internal stop endpoint of the platform here; its outcome is
    the `teardown.bridge_stop` field in the escalation records.)
    """
    r = c.request("POST", f"/projects/{pid}/sandbox/pause", {}, timeout_s=240)
    return {"ok": r.status in (200, 202, 204), "status": r.status, "code": r.error_code}


def teardown_one(c: Client, pid: str, verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    st = c.request("GET", f"/projects/{pid}/sandbox/status", timeout_s=60)
    running = (st.payload or {}).get("status") == "running"
    cleared = None
    if running:
        r = c.execute(pid, CLEAR, timeout_s=600)
        cleared = r.status == 200
        if verbose:
            print(f"  cleared namespace: {cleared}")
    d = c.delete_project(pid)
    out = {"project": pid, "was_running": running, "cleared": cleared,
           "delete": d, "seconds": round(time.perf_counter() - t0, 1)}
    # A failed delete may have left the sandbox billing. Stopping the burn
    # matters more than the orphaned row, so always fall through to pause.
    if d.get("status") not in (204, 200):
        out["pause"] = pause_fallback(c, pid)
        if verbose:
            print(f"  delete {d['status']} -> pause: {out['pause']}")
    elif verbose:
        print(f"  delete {d['status']} in {out['seconds']}s")
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project")
    ap.add_argument("--all", action="store_true",
                    help="every clusy-exp-s3-esc-* project")
    args = ap.parse_args()

    c = Client()
    targets = []
    if args.project:
        targets = [args.project]
    elif args.all:
        targets = [p["id"] for p in c.list_projects()
                   if "-esc-" in (p.get("name") or "")]
    if not targets:
        print("nothing to tear down")
        return 0

    print(f"{len(targets)} project(s)")
    for pid in targets:
        print(f"{pid[:8]}:")
        teardown_one(c, pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
