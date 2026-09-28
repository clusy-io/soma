"""Cross-interpreter behavioural tests for the storage-sharing adapter.

Every case crosses a REAL interpreter boundary: the namespace is dumped with
dill in this process and loaded in a fresh subprocess, exactly as a capsule
crosses a sandbox boundary. An in-process pickle round trip would not be
evidence, because `dill` memoises objects within one call.

Run directly to emit a JSON summary for the README:  python tests/test_storage_sharing.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from capsule.storage_sharing import (  # noqa: E402
    apply_view_manifest, collect_view_manifest, storage_shared,
)

LOADER = r'''
import sys, json, dill
sys.path.insert(0, %(src)r)
from capsule.storage_sharing import ViewManifest, apply_view_manifest, storage_shared
with open(%(path)r, "rb") as f:
    ns = dill.load(f)
manifest = ViewManifest.from_json(json.load(open(%(mpath)r)))
report = apply_view_manifest(ns, manifest) if %(adapt)r else {"restored": 0, "failed": [], "skipped": True}
pairs = json.load(open(%(pairs)r))
out = {}
for label, (base_path, view_path) in pairs.items():
    def get(p):
        if p[0] == "@base_of":
            rec = next(v for v in manifest.views if list(v.path) == [p[1]])
            return ns[rec.base_name]
        o = ns[p[0]]
        if len(p) == 1: return o
        if p[1] in ("param","buffer"):
            mod = o
            *parents, leaf = p[2].split(".")
            for q in parents: mod = getattr(mod, q)
            return getattr(mod, leaf)
        return o[p[1]]
    out[label] = storage_shared(get(base_path), get(view_path))
# The pointer-level, non-mutating check must agree with the write probe.
try:
    from capsule.storage_sharing import views_intact
    intact = views_intact(ns, manifest)
except ImportError:
    intact = None
print("__RESULT__" + json.dumps({"shared": out, "report": report, "intact": intact}))
'''


def _build_namespace():
    import numpy as np
    import torch
    torch.manual_seed(0)
    ns: dict = {}
    # torch: slice view, transpose view, narrow, expand, view-of-view
    ns["t_base"] = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    ns["t_slice"] = ns["t_base"][1:3]
    ns["t_T"] = ns["t_base"].t()
    ns["t_narrow"] = ns["t_base"].narrow(1, 2, 3)
    ns["t_chain"] = ns["t_slice"][:, ::2]           # view of a view
    ns["t_orphan_view"] = torch.zeros(10)[2:7]      # base NOT in namespace
    # inside a container
    ns["bag"] = {"base": torch.ones(8), }
    ns["bag"]["v"] = ns["bag"]["base"][4:]
    # numpy: slice, transpose, chain
    ns["a_base"] = np.arange(20, dtype=np.float64).reshape(4, 5)
    ns["a_slice"] = ns["a_base"][1:3]
    ns["a_T"] = ns["a_base"].T
    ns["a_chain"] = ns["a_slice"][:, 1:4]
    # unsupported: array over a bytes buffer (base is bytes, not ndarray)
    ns["a_over_bytes"] = np.frombuffer(b"\x00" * 64, dtype=np.float64)
    # a module whose parameter is a view of another tensor (tied slice)
    lin = torch.nn.Linear(6, 4, bias=False)
    ns["shared_weight_source"] = torch.randn(8, 6)
    lin.weight = torch.nn.Parameter(ns["shared_weight_source"][:4])
    ns["lin"] = lin
    return ns


PAIRS = {
    "torch slice":       (("t_base",), ("t_slice",)),
    "torch transpose":   (("t_base",), ("t_T",)),
    "torch narrow":      (("t_base",), ("t_narrow",)),
    "torch view chain":  (("t_base",), ("t_chain",)),
    "torch in dict":     (("bag", "base"), ("bag", "v")),
    "numpy slice":       (("a_base",), ("a_slice",)),
    "numpy transpose":   (("a_base",), ("a_T",)),
    "numpy view chain":  (("a_base",), ("a_chain",)),
    "module param view": (("shared_weight_source",), ("lin", "param", "weight")),
}


def run_case(adapt: bool) -> dict:
    import dill
    ns = _build_namespace()
    # ground truth in the SOURCE interpreter
    before = {}
    for label, (bp, vp) in PAIRS.items():
        if bp[0] == "@base_of":
            continue  # the orphan's base exists only after capture injects it
        def get(p):
            o = ns[p[0]]
            if len(p) == 1: return o
            if p[1] == "param":
                return getattr(ns[p[0]], p[2])
            return o[p[1]]
        before[label] = storage_shared(get(bp), get(vp))
    manifest = collect_view_manifest(ns)
    with tempfile.TemporaryDirectory() as d:
        path, mpath, ppath = f"{d}/ns.pkl", f"{d}/m.json", f"{d}/pairs.json"
        with open(path, "wb") as f:
            dill.dump(ns, f)
        json.dump(manifest.to_json(), open(mpath, "w"))
        json.dump({k: [list(a), list(b)] for k, (a, b) in PAIRS.items()}, open(ppath, "w"))
        code = LOADER % {"src": str(ROOT / "src"), "path": path, "mpath": mpath,
                         "adapt": adapt, "pairs": ppath}
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr[-2000:]
        line = [l for l in proc.stdout.splitlines() if l.startswith("__RESULT__")][-1]
        res = json.loads(line[len("__RESULT__"):])
    return {"before": before, "after": res["shared"], "report": res["report"], "intact": res["intact"],
            "manifest": {"preserved": manifest.preserved,
                         "injected_bases": manifest.injected_bases,
                         "unsupported": manifest.unsupported}}


def test_without_adapter_sharing_is_lost():
    r = run_case(adapt=False)
    assert all(r["before"].values()), "every pair must share storage at the source"
    assert not any(r["after"].values()), f"without the adapter no pair should share: {r['after']}"
    assert r["intact"]["intact"] == 0 and len(r["intact"]["broken"]) == r["intact"]["checked"] > 0


def test_with_adapter_sharing_is_restored():
    r = run_case(adapt=True)
    assert all(r["after"].values()), f"adapter failed to restore: {r['after']}"
    assert r["report"]["failed"] == []
    assert r["report"]["restored"] >= len(PAIRS)  # a reshaped base is itself a view of its temporary
    assert r["intact"] == {"checked": r["report"]["restored"], "intact": r["report"]["restored"], "broken": []}


def test_unsupported_is_reported_not_silent():
    ns = _build_namespace()
    m = collect_view_manifest(ns)
    reasons = [u["reason"] for u in m.unsupported]
    assert any("not ndarray" in x for x in reasons), reasons
    assert ["a_over_bytes"] in [u["path"] for u in m.unsupported]


def test_orphan_view_is_not_aliasing():
    """A view whose base is a temporary shares its buffer with nothing bound in
    the namespace, so there is no observable aliasing to preserve. It must be
    left alone: not recorded, not injected, not reported as unsupported."""
    ns = _build_namespace()
    m = collect_view_manifest(ns)
    assert not any(list(v.path) == ["t_orphan_view"] for v in m.views)
    assert not any(u["path"] == ["t_orphan_view"] for u in m.unsupported)


def test_no_views_is_a_noop():
    import torch
    ns = {"x": torch.zeros(3), "y": [1, 2, 3]}
    m = collect_view_manifest(ns)
    assert m.preserved == 0 and m.unsupported == [] and m.injected_bases == []
    assert apply_view_manifest(ns, m) == {"restored": 0, "failed": [], "unsupported_at_capture": 0}


if __name__ == "__main__":
    off, on = run_case(False), run_case(True)
    summary = {
        "pairs": list(PAIRS),
        "shared_at_source": off["before"],
        "shared_after_load_without_adapter": off["after"],
        "shared_after_load_with_adapter": on["after"],
        "adapter_report": on["report"],
        "manifest": on["manifest"],
    }
    print(json.dumps(summary, indent=1))
