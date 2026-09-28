"""The storage adapter must preserve the OBJECT GRAPH around a view, not only
the storage relationship.

The reviewer's audit (27 September) found that the first repair rebuilt each
view as a NEW object and rebound it at one path. Every E6 storage pair then
passed, while the graph around it broke with zero reported failures: two names
bound to one view became two objects, an optimizer stepped a Parameter the
model no longer held, and a top-level Parameter came back as a plain Tensor
without requires_grad.

Every case here crosses a REAL interpreter boundary, and not only for the
load: the namespace is BUILT and dumped in one fresh interpreter and loaded,
repaired and inspected in another. Building it in a child also means a class
defined by the case lives in that child's `__main__`, so dill carries it by
value, as it would carry a user's class out of a kernel.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

OUTSIDE = "reference outside the covered paths"

_COMMON = '''
import sys, json, hashlib, gc
import dill, numpy as np, torch
def digest(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def ptr(x):
    return x.untyped_storage().data_ptr() if isinstance(x, torch.Tensor) else x.__array_interface__["data"][0]
'''

_IMPORTS = '''
from capsule.storage_sharing import collect_view_manifest, apply_view_manifest, storage_shared, ViewManifest
try:
    from capsule.storage_sharing import views_intact
except ImportError:  # the pre-fix adapter has no non-mutating check
    views_intact = None
'''


def _run(code: str) -> dict:
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=240)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [l for l in proc.stdout.splitlines() if l.startswith("__RESULT__")][-1]
    return json.loads(line[len("__RESULT__"):])


def cross(build: str, inspect: str, *, after_dump: str = "", before_apply: str = "") -> tuple[dict, dict]:
    """Build `ns` in one interpreter, dump it, then load + apply + inspect it in another.

    `build` binds `ns`; it may record source facts in `src`. `after_dump`
    runs in the source after the dump, with the injected hidden bases already
    removed again (what the controller's capture does), e.g. to compute the
    continuation a destination must reproduce. `before_apply` runs in the
    destination between load and repair, to corrupt a case on purpose.
    `inspect` fills `result`.
    """
    with tempfile.TemporaryDirectory() as d:
        pkl, mpath, spath = f"{d}/ns.pkl", f"{d}/manifest.json", f"{d}/src.json"
        source = "\n".join([
            f"import sys\nsys.path.insert(0, {str(SRC)!r})", _COMMON, _IMPORTS,
            "src = {}", build,
            "manifest = collect_view_manifest(ns)",
            "src['records'] = len(manifest.views)",
            "src['record_paths'] = [[list(p) for p in r.all_paths()] if hasattr(r, 'all_paths') else [list(r.path)] for r in manifest.views]",
            "src['intact_in_window'] = views_intact(ns, manifest) if views_intact else None",
            # what the capture already knew (absent from the pre-fix manifests)
            "src['predicted'] = ([[f['path'], f['reason']] for f in manifest.predicted_failures]"
            " if hasattr(manifest, 'predicted_failures') else None)",
            "src['autograd'] = [getattr(r, 'autograd', None) for r in manifest.views]",
            "src['unsupported'] = manifest.unsupported",
            f"with open({pkl!r}, 'wb') as _f:\n    dill.dump(ns, _f)",
            "for _k in manifest.injected_bases:\n    ns.pop(_k, None)",
            "src['intact_after_window'] = views_intact(ns, manifest) if views_intact else None",
            f"json.dump(manifest.to_json(), open({mpath!r}, 'w'))",
            after_dump,
            f"json.dump(src, open({spath!r}, 'w'), default=str)",
            "print('__RESULT__' + json.dumps(src, default=str))",
        ])
        src = _run(source)
        dest = "\n".join([
            f"import sys\nsys.path.insert(0, {str(SRC)!r})", _COMMON, _IMPORTS,
            f"with open({pkl!r}, 'rb') as _f:\n    ns = dill.load(_f)",
            f"manifest = ViewManifest.from_json(json.load(open({mpath!r})))",
            f"src = json.load(open({spath!r}))",
            "result = {}",
            before_apply,
            "report = apply_view_manifest(ns, manifest)",
            "result['report'] = report",
            "result['intact'] = views_intact(ns, manifest) if views_intact else None",
            inspect,
            "print('__RESULT__' + json.dumps(result, default=str))",
        ])
        return src, _run(dest)


def _all_intact(summary: dict | None, n: int) -> bool:
    return summary is not None and summary["checked"] == n and summary["intact"] == n and summary["broken"] == []


# ---------------------------------------------------------------------------
# The reviewer's three counterexamples, as regressions
# ---------------------------------------------------------------------------


def test_repeated_reference_to_a_view_stays_one_object():
    """Reviewer case 1: `view` and `alias` are one object. After repair a
    write through the base read 99 through one name and 1 through the other."""
    src, res = cross(
        build='''
base = torch.arange(8, dtype=torch.float32)
view = base[1:4]
a = np.arange(10.0)
av = a[2:6]
ns = {"base": base, "view": view, "alias": view, "a": a, "av": av, "av_alias": av,
      "av_list": [av, 0], "av_dict": {"k": av}}
''',
        inspect='''
result["torch_identity"] = ns["view"] is ns["alias"]
ns["base"][1] = 99
result["view_value"], result["alias_value"] = float(ns["view"][0]), float(ns["alias"][0])
result["numpy_identity"] = all(x is ns["av"] for x in (ns["av_alias"], ns["av_list"][0], ns["av_dict"]["k"]))
ns["a"][2] = -7
result["numpy_values"] = [float(x[0]) for x in (ns["av"], ns["av_alias"], ns["av_list"][0], ns["av_dict"]["k"])]
''')
    assert res["torch_identity"] is True
    assert res["view_value"] == 99.0 and res["alias_value"] == 99.0
    assert res["numpy_identity"] is True
    assert res["numpy_values"] == [-7.0] * 4
    assert res["report"]["failed"] == []
    # one record per OBJECT, carrying every path that reached it
    assert src["records"] == 2
    assert sorted(map(len, src["record_paths"])) == [2, 4]
    assert _all_intact(res["intact"], 2)


def test_parameter_sharing_a_base_keeps_optimizer_alias_and_trains():
    """Reviewer case 2: the model's Parameter shares a base and is referenced
    by the optimizer and an external alias. A real backward + step must move
    the model AND the base."""
    _, res = cross(
        build='''
base = torch.arange(8, dtype=torch.float32)
model = torch.nn.Linear(4, 1, bias=False)
model.weight = torch.nn.Parameter(base[:4].reshape(1, 4))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
ns = {"base": base, "model": model, "optimizer": optimizer, "weight_alias": model.weight}
''',
        inspect='''
m, opt = ns["model"], ns["optimizer"]
result["optimizer_points_to_model"] = opt.param_groups[0]["params"][0] is m.weight
result["alias_points_to_model"] = ns["weight_alias"] is m.weight
before = m.weight.detach().clone()
opt.zero_grad(); m(torch.ones(1, 4)).sum().backward(); opt.step()
result["model_updated_by_optimizer_step"] = not torch.equal(before, m.weight)
result["base_updated_by_optimizer_step"] = torch.equal(ns["base"][:4], m.weight.detach().reshape(-1))
result["expected_value"] = torch.equal(m.weight.detach(), before - 0.1)
result["intact_after_step"] = views_intact(ns, manifest) if views_intact else None
''')
    assert res["optimizer_points_to_model"] is True
    assert res["alias_points_to_model"] is True
    assert res["model_updated_by_optimizer_step"] is True
    assert res["base_updated_by_optimizer_step"] is True
    assert res["expected_value"] is True
    assert res["report"]["failed"] == []
    assert _all_intact(res["intact_after_step"], 1)


def test_top_level_parameter_keeps_class_and_requires_grad():
    """Reviewer case 3: a top-level Parameter sharing a base came back as a
    plain Tensor with requires_grad=False."""
    _, res = cross(
        build='''
base = torch.arange(8, dtype=torch.float32)
ns = {"base": base, "parameter": torch.nn.Parameter(base[:4])}
''',
        inspect='''
p = ns["parameter"]
result["is_parameter"] = isinstance(p, torch.nn.Parameter)
result["requires_grad"] = p.requires_grad
result["is_leaf"] = p.is_leaf
result["shares"] = storage_shared(p, ns["base"])
''')
    assert res["is_parameter"] is True and res["requires_grad"] is True and res["is_leaf"] is True
    assert res["shares"] is True
    assert res["report"]["failed"] == []


# ---------------------------------------------------------------------------
# Combined optimizer-plus-views fixture
# ---------------------------------------------------------------------------

_COMBINED = '''
torch.manual_seed(0)
flat = torch.randn(40)
model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False), torch.nn.Tanh(), torch.nn.Linear(4, 2, bias=False))
model[0].weight = torch.nn.Parameter(flat[:16].view(4, 4))     # two parameters,
model[2].weight = torch.nn.Parameter(flat[16:24].view(2, 4))   # one base
opt = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.01)
x, y = torch.randn(8, 4), torch.randn(8, 2)
for _ in range(2):
    opt.zero_grad(); ((model(x) - y) ** 2).mean().backward(); opt.step()
tail = flat[30:34]
top = torch.nn.Parameter(flat[34:38])
arr = np.arange(24.0)
arr_v = arr[4:12]
arr_T = arr.reshape(4, 6).T
ns = {"flat": flat, "model": model, "opt": opt, "x": x, "y": y,
      "w_alias": model[0].weight, "tail": tail, "tail_again": tail, "top": top,
      "arr": arr, "arr_v": arr_v, "arr_T": arr_T, "arr_again": [arr_v]}
src["state_len"] = len(opt.state)
'''

_STEP = '''
m, o = ns["model"], ns["opt"]
o.zero_grad(); ((m(ns["x"]) - ns["y"]) ** 2).mean().backward(); o.step()
'''

_AFTER_STEP_FACTS = '''({"flat": digest(ns["flat"]), "w0": digest(ns["model"][0].weight), "w2": digest(ns["model"][2].weight),
 "steps": sorted(float(s["step"]) for s in ns["opt"].state.values()),
 "exp_avg": sorted(digest(s["exp_avg"]) for s in ns["opt"].state.values())})
'''


def test_combined_optimizer_and_views_fixture():
    src, res = cross(
        build=_COMBINED,
        after_dump=_STEP + "src['after_step'] = " + _AFTER_STEP_FACTS,
        inspect='''
m, o = ns["model"], ns["opt"]
params = list(m.parameters())
result["optimizer_param_identity"] = [a is b for a, b in zip(o.param_groups[0]["params"], params)]
result["state_len"] = len(o.state)
result["state_keys_are_params"] = all(any(k is p for p in params) for k in o.state)
result["alias"] = ns["w_alias"] is m[0].weight
result["tail_identity"] = ns["tail"] is ns["tail_again"]
result["numpy_identity"] = ns["arr_again"][0] is ns["arr_v"]
result["classes"] = [type(t).__name__ for t in (m[0].weight, m[2].weight, ns["top"])]
result["requires_grad"] = [t.requires_grad for t in (m[0].weight, m[2].weight, ns["top"])]
fp = ptr(ns["flat"])
result["one_storage"] = all(ptr(t) == fp for t in (m[0].weight, m[2].weight, ns["tail"], ns["top"]))
result["numpy_shares"] = all(np.shares_memory(ns["arr"], a) for a in (ns["arr_v"], ns["arr_T"], ns["arr_again"][0]))
''' + _STEP + '''
result["after_step"] = ''' + _AFTER_STEP_FACTS + '''
result["base_moved_with_model"] = (torch.equal(ns["flat"][:16].view(4, 4), m[0].weight.detach())
                                   and torch.equal(ns["flat"][16:24].view(2, 4), m[2].weight.detach()))
result["intact_after_step"] = views_intact(ns, manifest) if views_intact else None
''')
    assert res["optimizer_param_identity"] == [True, True]
    assert res["state_len"] == src["state_len"] == 2
    assert res["state_keys_are_params"] is True
    assert res["alias"] is True and res["tail_identity"] is True and res["numpy_identity"] is True
    assert res["classes"] == ["Parameter", "Parameter", "Parameter"]
    assert res["requires_grad"] == [True, True, True]
    assert res["one_storage"] is True and res["numpy_shares"] is True
    # A real step on the destination reproduces the source's own next step
    # bit for bit, and the shared base moved with the model.
    assert res["after_step"] == src["after_step"]
    assert res["base_moved_with_model"] is True
    assert res["report"]["failed"] == []
    n = src["records"]
    assert _all_intact(src["intact_in_window"], n) and _all_intact(res["intact"], n)
    assert _all_intact(res["intact_after_step"], n)


def test_parameters_over_a_hidden_base_train_together():
    """Two parameters viewing a base nothing else binds: the anchor is a
    synthetic hidden base, so the source-side check after the capture window
    has only the group itself to go on."""
    src, res = cross(
        build='''
torch.manual_seed(1)
buf = torch.randn(24)
model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False), torch.nn.Linear(4, 2, bias=False))
model[0].weight = torch.nn.Parameter(buf[:16].view(4, 4))
model[1].weight = torch.nn.Parameter(buf[16:24].view(2, 4))
del buf
opt = torch.optim.AdamW(model.parameters(), lr=0.05)
x, y = torch.randn(8, 4), torch.randn(8, 2)
opt.zero_grad(); ((model(x) - y) ** 2).mean().backward(); opt.step()
bag = [torch.arange(6, dtype=torch.float32)]
ns = {"model": model, "opt": opt, "x": x, "y": y, "bag": bag, "bag_view": bag[0][2:5]}
''',
        after_dump='''
m, o = ns["model"], ns["opt"]
o.zero_grad(); ((m(ns["x"]) - ns["y"]) ** 2).mean().backward(); o.step()
src["after"] = [digest(p) for p in m.parameters()]
''',
        inspect='''
m, o = ns["model"], ns["opt"]
result["shared"] = ptr(m[0].weight) == ptr(m[1].weight)
result["opt_identity"] = all(a is b for a, b in zip(o.param_groups[0]["params"], m.parameters()))
o.zero_grad(); ((m(ns["x"]) - ns["y"]) ** 2).mean().backward(); o.step()
result["after"] = [digest(p) for p in m.parameters()]
ns["bag"][0][3] = 42
result["bag_view"] = float(ns["bag_view"][1])
# Unbind both hidden bases. The synthetic anchor's group still vouches for
# itself; the base that had a covered path, and lost it, is reported missing.
for k in [k for k in ns if k.startswith("__clusy_view_base_")]:
    del ns[k]
ns["bag"] = []
result["unbound"] = views_intact(ns, manifest)
''')
    assert res["shared"] is True and res["opt_identity"] is True
    assert res["after"] == src["after"]
    assert res["bag_view"] == 42.0
    assert res["unbound"]["intact"] == 2
    assert [(b["path"], b["reason"]) for b in res["unbound"]["broken"]] == [
        (["bag_view"], "base '__clusy_view_base_1' missing")]
    assert res["report"]["failed"] == []
    n = src["records"]
    assert n == 3
    # identical at the cut (hidden bases bound), after it (removed again) and
    # at the destination: what a commit-boundary comparison needs
    assert src["intact_in_window"] == src["intact_after_window"] == res["intact"]
    assert _all_intact(res["intact"], n)


def test_hidden_base_overlapping_slices_still_propagate():
    """The fix the reviewer confirmed must not regress: two overlapping slices
    of an unbound base come back over one storage."""
    _, res = cross(
        build='''
b = torch.arange(10, dtype=torch.float32)
ns = {"audit_a": b[1:5], "audit_b": b[3:7]}
del b
''',
        inspect='''
ns["audit_a"][2] = 99
result["second_view"] = float(ns["audit_b"][0])
result["hidden"] = [k for k in ns if k.startswith("__clusy_view_base_")]
''')
    assert res["second_view"] == 99.0
    assert res["hidden"] == ["__clusy_view_base_0"]
    assert res["report"]["failed"] == []


# ---------------------------------------------------------------------------
# Autograd views: the gradient link is a relationship too
# ---------------------------------------------------------------------------

AUTOGRAD = "autograd view: gradient link not carried"

_NONLEAF_STEP = '''
W, Wv, Wm, opt = ns["W"], ns["Wv"], ns["Wm"], ns["opt"]
opt.zero_grad(); ((Wv ** 2).sum() + (Wm[1] * 3).sum()).backward(); opt.step()
{into}["W_next"] = W.detach().tolist()
{into}["W_grad"] = None if W.grad is None else W.grad.tolist()   # None: the step never reached W
'''


def test_autograd_view_of_a_parameter_keeps_its_gradient_link():
    """Review blocker, 27 September (second round). The optimizer owns W; the
    training loop computes its loss through non-leaf views of W held in the
    namespace. The serializer brings each view back as a LEAF, and a set_
    repair left it one: storage shared, gradient going to the view, W.grad
    None, the optimizer step a no-op, reported as restored with 0 failures.

    The destination's real step must reproduce the source's own next step bit
    for bit, every reference must still reach the one view object, and the
    full-storage non-leaf `Wm`, bound BEFORE W, must not displace W as anchor.
    """
    src, res = cross(
        build='''
torch.manual_seed(0)
W = torch.nn.Parameter(torch.randn(8))
Wv = W[:4]          # non-leaf, SliceBackward
Wm = W.view(2, 4)   # non-leaf, covers the whole storage
opt = torch.optim.SGD([W], lr=0.1)
ns = {"Wm": Wm, "W": W, "Wv": Wv, "Wv_again": Wv, "bag": [Wv, 1], "opt": opt}
src["is_leaf"] = [Wv.is_leaf, Wm.is_leaf]
''',
        after_dump=_NONLEAF_STEP.format(into="src"),
        inspect='''
W, Wv, Wm, opt = ns["W"], ns["Wv"], ns["Wm"], ns["opt"]
result["identity"] = [ns["Wv_again"] is Wv, ns["bag"][0] is Wv, opt.param_groups[0]["params"][0] is W]
result["is_leaf"] = [Wv.is_leaf, Wm.is_leaf]
result["autograd_base_is_W"] = [Wv._base is W, Wm._base is W]
''' + _NONLEAF_STEP.format(into="result") + '''
with torch.no_grad():
    W[0] = 123.0
result["write_visible"] = [float(Wv[0]), float(Wm[0, 0])]
result["intact_after_step"] = views_intact(ns, manifest) if views_intact else None
# The pre-fix repair's end state: a LEAF over W's bytes at every reference.
# Storage and geometry are right; only leaf-ness says the gradient link is
# gone, and views_intact (hence the commit boundary) must say so.
leaf = torch.zeros(4, requires_grad=True)
with torch.no_grad():
    leaf.set_(W.untyped_storage(), 0, (4,), (1,))
ns["Wv"] = ns["Wv_again"] = ns["bag"][0] = leaf
result["leaf_drift"] = ([[b["path"], b["reason"]] for b in views_intact(ns, manifest)["broken"]]
                        if views_intact else None)
''')
    assert src["is_leaf"] == [False, False]
    # the step through the views moves W exactly as it did at the source
    assert res["W_grad"] == src["W_grad"], (res["W_grad"], src["W_grad"])
    assert res["W_next"] == src["W_next"]
    assert res["report"]["failed"] == []
    assert res["identity"] == [True, True, True]
    assert res["is_leaf"] == [False, False]
    assert res["autograd_base_is_W"] == [True, True]
    assert res["write_visible"] == [123.0, 123.0]
    assert src["autograd"] == ["view_of_base", "view_of_base"]
    assert src["intact_in_window"] == src["intact_after_window"] == res["intact"]
    assert _all_intact(res["intact"], 2) and _all_intact(res["intact_after_step"], 2)
    assert res["leaf_drift"] == [[["Wv"], "is_leaf True, recorded a non-leaf"]]


def test_non_leaf_views_that_cannot_be_rebuilt_fail_explicitly():
    """A non-leaf whose gradient does not end at a leaf anchor cannot be
    rebuilt as a view of it. Each is a named failure, predicted at capture,
    left untouched at the destination, and seen as broken by views_intact:
    (a) views of a non-leaf `h = W1 * 2`, whose own link to W1 the serializer
    drops; (b) views of a Parameter only an optimizer holds, so the anchor is
    synthetic; (c) a non-leaf view registered as a module buffer, which a
    replacement cannot re-register. A leaf view in the same namespace is
    still repaired: failure is per record."""
    src, res = cross(
        build='''
torch.manual_seed(2)
W1 = torch.nn.Parameter(torch.randn(4))
h = W1 * 2
W2 = torch.nn.Parameter(torch.randn(6))
opt2 = torch.optim.SGD([W2], lr=0.1)
W3 = torch.nn.Parameter(torch.randn(5))
m = torch.nn.Module()
m.register_buffer("hb", W3[:2])
plain = torch.arange(6, dtype=torch.float32)
ns = {"h": h, "h_head": h[:2], "h_tail": h[2:], "opt2": opt2, "a2": W2[:3], "b2": W2[3:],
      "W3": W3, "m": m, "plain": plain, "plain_v": plain[1:3]}
''',
        before_apply='''
watch = ["h_head", "h_tail", "a2", "b2"]
before = {k: (id(ns[k]), ptr(ns[k])) for k in watch}
before["m"] = (id(ns["m"].hb), ptr(ns["m"].hb))
''',
        inspect='''
result["failed"] = sorted([f["path"][0], f["reason"]] for f in report["failed"])
now = {k: (id(ns[k]), ptr(ns[k])) for k in watch}
now["m"] = (id(ns["m"].hb), ptr(ns["m"].hb))
result["untouched"] = now == before
result["broken"] = sorted(b["path"][0] for b in result["intact"]["broken"]) if result["intact"] else None
ns["plain"][1] = 7
result["plain_repaired"] = float(ns["plain_v"][0]) == 7.0
''')
    expected = sorted([["a2", AUTOGRAD], ["b2", AUTOGRAD], ["h_head", AUTOGRAD], ["h_tail", AUTOGRAD],
                       ["m", "path not rebindable"]])
    assert res["failed"] == expected, res
    assert sorted([p[0], r] for p, r in src["predicted"]) == expected
    assert res["untouched"] is True
    assert res["broken"] == ["a2", "b2", "h_head", "h_tail", "m"]
    assert res["plain_repaired"] is True and res["report"]["restored"] == 1
    # every one was intact at the source cut: the check sees drift, not kinds
    assert _all_intact(src["intact_in_window"], 6)


# ---------------------------------------------------------------------------
# NumPy: all references or none
# ---------------------------------------------------------------------------

_HOLDERS = {
    # a user object's attribute: the instance is a collector-visible referrer
    "attribute": ("class Holder:\n    pass\nh = Holder()\nh.arr = v\nextra = {'h': h}",
                  "ns['h'].arr"),
    # a SimpleNamespace's __dict__ holds only the array, so that dict is
    # UNTRACKED and gc.get_referrers cannot see it; the liveness check must
    "untracked_dict": ("import types\nextra = {'sn': types.SimpleNamespace(arr=v)}", "ns['sn'].arr"),
    # two levels deep: past the covered paths
    "nested": ("extra = {'deep': {'inner': {'arr': v}}}", "ns['deep']['inner']['arr']"),
    # a closure cell
    "closure": ("def make(x):\n    return lambda: x\nextra = {'f': make(v)}", "ns['f']()"),
    # a dunder name the walk does not cover
    "dunder_name": ("extra = {'__kept_ref__': v}", "ns['__kept_ref__']"),
}


@pytest.mark.parametrize("holder", sorted(_HOLDERS))
def test_numpy_view_held_outside_covered_paths_fails_and_is_untouched(holder):
    setup, access = _HOLDERS[holder]
    _, res = cross(
        build=f'''
a = np.arange(10.0)
v = a[2:6]
{setup}
tb = torch.arange(6, dtype=torch.float32)
ns = {{"a": a, "v": v, "v_alias": v, "tb": tb, "tv": tb[1:3], **extra}}
''',
        inspect=f'''
held = {access}
result["untouched_identity"] = (ns["v"] is held) and (ns["v_alias"] is held)
result["still_private_copy"] = not np.shares_memory(ns["v"], ns["a"])
result["values"] = ns["v"].tolist()
result["failed"] = [(f["path"], f["reason"]) for f in report["failed"]]
result["detail"] = [f.get("detail") for f in report["failed"]]
result["torch_repaired"] = storage_shared(ns["tb"], ns["tv"])
''')
    # the record fails, with the named reason, and nothing moved
    assert res["failed"] == [[["v"], OUTSIDE]], res
    assert res["untouched_identity"] is True
    assert res["still_private_copy"] is True
    assert res["values"] == [2.0, 3.0, 4.0, 5.0]
    # the failure is per record: the unrelated torch view is still repaired
    assert res["torch_repaired"] is True
    assert res["report"]["restored"] == 1
    assert res["intact"]["intact"] == 1 and [b["path"] for b in res["intact"]["broken"]] == [["v"]]


def test_numpy_views_in_tuples():
    """A tuple is immutable, so the array inside is replaced by replacing the
    tuple: allowed when every holder of the tuple is a covered name (all of
    them then get ONE new tuple), refused when anything else holds it."""
    _, res = cross(
        build='''
a = np.arange(12.0)
class Holder:
    pass
h = Holder()
t3 = (a[8:10], 3)
h.t = t3
ns = {"a": a, "t1": (a[0:2], 1), "t2": (a[4:6], 2), "t3": t3, "h": h}
ns["t2b"] = ns["t2"]
''',
        inspect='''
ns["a"][:] = -1
result["t1"] = ns["t1"][0].tolist()
result["t2_same_tuple"] = ns["t2"] is ns["t2b"]
result["t2"] = ns["t2"][0].tolist()
result["t3_untouched"] = ns["t3"] is ns["h"].t and not np.shares_memory(ns["t3"][0], ns["a"])
result["failed"] = [(f["path"], f["reason"]) for f in report["failed"]]
''')
    assert res["t1"] == [-1.0, -1.0]
    assert res["t2_same_tuple"] is True and res["t2"] == [-1.0, -1.0]
    assert res["t3_untouched"] is True
    assert res["failed"] == [[["t3", 0], OUTSIDE]]


def test_refused_repair_inside_a_tuple_leaves_the_tuple_itself():
    """The tuple is held only by covered names, but the array inside it is
    also an attribute of a plain object. The refusal must leave the TUPLE the
    same object too: the pre-fix code rebound it first and, on undo, built a
    third tuple around the survivor."""
    _, res = cross(
        build='''
a = np.arange(8.0)
class Holder:
    pass
h = Holder()
t = (a[2:5], 1)
h.arr = t[0]
ns = {"a": a, "t": t, "t_again": t, "h": h}
''',
        before_apply='t_id = id(ns["t"])',
        inspect='''
result["failed"] = [(f["path"], f["reason"]) for f in report["failed"]]
result["same_tuple"] = id(ns["t"]) == t_id and ns["t"] is ns["t_again"]
result["same_array"] = ns["t"][0] is ns["h"].arr
''')
    assert res["failed"] == [[["t", 0], OUTSIDE]]
    assert res["same_tuple"] is True and res["same_array"] is True


def test_restore_cost_is_linear_not_a_heap_walk_per_record():
    """Review finding (minor): every NumPy record called gc.get_referrers (a
    walk of every tracked object) and up to two gc.collect(), so a restore of
    N views cost O(N x heap). Now: no walk at all for records that repair, at
    most one collection per call, and one walk to NAME the holders of all
    refused records together. Counted, not timed, so the test is exact."""
    _, res = cross(
        build='''
X = np.random.default_rng(0).random((300, 3))
Y = np.random.default_rng(1).random((200, 3))
class Dataset:
    pass
ds = Dataset()
ok_rows = [X[i] for i in range(300)]
held_rows = [Y[i] for i in range(200)]
ds.rows = list(held_rows)      # a second, uncovered list holding every Y row
ns = {"X": X, "ok_rows": ok_rows, "Y": Y, "held_rows": held_rows, "ds": ds}
''',
        before_apply='''
calls = {"collect": 0, "get_referrers": 0}
_real_collect, _real_referrers = gc.collect, gc.get_referrers
def _counting_collect(*a, **k):
    calls["collect"] += 1
    return _real_collect(*a, **k)
def _counting_referrers(*objs):
    calls["get_referrers"] += 1
    return _real_referrers(*objs)
gc.collect, gc.get_referrers = _counting_collect, _counting_referrers
''',
        inspect='''
gc.collect, gc.get_referrers = _real_collect, _real_referrers
result["calls"] = calls
result["failed_paths"] = sorted({f["path"][0] for f in report["failed"]})
result["reasons"] = sorted({f["reason"] for f in report["failed"]})
result["details_name_the_list"] = all("held by list" in f["detail"] for f in report["failed"])
ns["X"][:] = -1
result["ok_rows_repaired"] = all(float(r[0]) == -1.0 for r in ns["ok_rows"])
''')
    assert res["report"]["restored"] == 300 and len(res["report"]["failed"]) == 200
    assert res["failed_paths"] == ["held_rows"] and res["reasons"] == [OUTSIDE]
    assert res["ok_rows_repaired"] is True
    assert res["details_name_the_list"] is True
    assert res["calls"]["get_referrers"] <= 1, res["calls"]
    assert res["calls"]["collect"] <= 1, res["calls"]


# ---------------------------------------------------------------------------
# Explicit rejection: a record that cannot be repaired is a FAILURE
# ---------------------------------------------------------------------------


def test_torch_rejections_are_failures_and_leave_state_alone():
    src, res = cross(
        build='''
b = torch.arange(8, dtype=torch.float32)
b2 = torch.arange(8, dtype=torch.float32) * 10
b3 = torch.arange(8, dtype=torch.float32) * 100
ns = {"b": b, "view": b[1:3],
      "b2": b2, "iv": b2.view(torch.int32),     # reinterpreting view: dtype differs from its base
      "b3": b3, "v3": b3[2:4],
      "b4": torch.ones(6), "v4": None}
ns["v4"] = ns["b4"][1:4]
''',
        before_apply='''
ns["view"] = [1, 2, 3]                      # not a tensor at the path any more
ns["b4"] = ns["b4"].to("meta")              # base on another device
# point b3's record at the object that is b's base: "view is the base of another record"
rec = next(r for r in manifest.views if list(r.path) == ["v3"])
rec.path, rec.paths, rec.base_name = ("b",), [("b",)], "b3"
before = {"iv": ptr(ns["iv"]), "v4": ptr(ns["v4"]), "b": ptr(ns["b"]), "b_obj": id(ns["b"])}
''',
        inspect='''
result["reasons"] = sorted((f["path"][0], f["reason"]) for f in report["failed"])
result["iv_untouched"] = ptr(ns["iv"]) == before["iv"] and ns["iv"].dtype == torch.int32
result["v4_untouched"] = ptr(ns["v4"]) == before["v4"]
result["b_untouched"] = ptr(ns["b"]) == before["b"] and id(ns["b"]) == before["b_obj"] and ns["b"].shape == (8,)
''')
    assert res["reasons"] == [
        ["b", "view is the base of another record"],
        ["iv", "dtype mismatch with base"],
        ["v4", "device mismatch with base"],
        ["view", "not a tensor at the path"],
    ], res
    assert res["report"]["restored"] == 0
    assert res["iv_untouched"] and res["v4_untouched"] and res["b_untouched"]
    assert res["intact"]["intact"] == 0


def test_numpy_subclass_view_is_rejected_not_flattened():
    _, res = cross(
        build='''
class Tagged(np.ndarray):
    pass
a = np.arange(10.0)
ns = {"a": a, "tv": a.view(Tagged)[1:4]}
''',
        inspect='''
result["cls"] = type(ns["tv"]).__name__
result["reasons"] = [f["reason"] for f in report["failed"]]
''')
    assert res["cls"] == "Tagged"
    assert res["reasons"] == ["ndarray subclass"]


def test_capture_predicts_the_refusals_the_source_already_shows():
    """Review finding (minor): refusals appeared only after capture, transfer
    and restore. Whatever the SOURCE already shows is now in the manifest's
    `predicted_failures`, with the reason the restore reports, so a caller can
    refuse or exclude before the cut. The one refusal deliberately NOT
    predicted is a NumPy view held outside the covered paths: the source
    holds references the capsule does not carry, so it is decided at the
    destination (and still fails there, explicitly)."""
    src, res = cross(
        build='''
from collections import namedtuple
Pair = namedtuple("Pair", "x y")
class Tagged(np.ndarray):
    pass
class Dataset:
    def __init__(self, x):
        self.x = x
c = torch.randn(4, dtype=torch.cfloat)
f = torch.arange(8, dtype=torch.float32)
a = np.arange(20.0)
X = np.arange(40.0).reshape(10, 4)
X_train = X[:8]
ns = {"c": c, "as_real": torch.view_as_real(c),          # dtype reinterpretation
      "f": f, "as_int": f.view(torch.int32),             # dtype reinterpretation
      "a": a, "tagged": a.view(Tagged)[1:4],             # ndarray subclass
      "pair": Pair(a[5:7], 0),                           # tuple subclass holder
      "X": X, "X_train": X_train, "train_ds": Dataset(X_train)}
''',
        inspect='''
result["failed"] = sorted([f["path"], f["reason"]] for f in report["failed"])
''')
    predicted = sorted([p, r] for p, r in src["predicted"])
    assert predicted == [
        [["as_int"], "dtype mismatch with base"],
        [["as_real"], "dtype mismatch with base"],
        [["pair", 0], "container cannot be rebuilt"],
        [["tagged"], "ndarray subclass"],
    ], predicted
    # every prediction is a real refusal, and the one extra refusal is the
    # outside holder the source cannot judge
    assert [f for f in res["failed"] if f not in predicted] == [[["X_train"], OUTSIDE]]
    assert all(p in res["failed"] for p in predicted)


def test_capture_reports_cross_framework_sharing_and_unchecked_tensors():
    """Review finding (minor): `torch.from_numpy` sharing between two covered
    names was lost with no record and no report, and a lone sparse tensor was
    reported under the same reason as a refused view."""
    import numpy as np
    import torch

    sys.path.insert(0, str(SRC))
    from capsule.storage_sharing import collect_view_manifest

    a = np.arange(6.0)
    ns = {"a": a, "t": torch.from_numpy(a), "s": torch.eye(3).to_sparse(), "x": torch.ones(2)}
    m = collect_view_manifest(ns)
    by_path = {tuple(u["path"]): u for u in m.unsupported}
    assert set(by_path) == {("t",), ("s",)}
    assert by_path[("t",)]["reason"].startswith("shares memory with a NumPy array")
    assert by_path[("t",)]["with"] == ["a"]
    assert by_path[("s",)]["reason"].startswith("no untyped storage") and "not checked" in by_path[("s",)]["reason"]
    assert m.views == [] and m.injected_bases == []


# ---------------------------------------------------------------------------
# views_intact: non-mutating, and it can fail
# ---------------------------------------------------------------------------


def test_views_intact_writes_nothing_and_detects_breakage():
    src, res = cross(
        build=_COMBINED,
        inspect='''
from torch.overrides import TorchFunctionMode
class CallLog(TorchFunctionMode):
    """Every torch function or Tensor method called from Python, by name."""
    def __init__(self):
        super().__init__()
        self.names = []
    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.names.append(getattr(func, "__name__", str(func)))
        return func(*args, **(kwargs or {}))
def in_place(names):
    return sorted({n for n in names if n in ("__setitem__", "set_", "copy_")
                   or (n.endswith("_") and not n.startswith("__"))})
tensors = [ns["flat"], *ns["model"].parameters(), ns["tail"], ns["top"]]
versions = [t._version for t in tensors]
digests = [digest(t) for t in tensors] + [digest(ns["arr"])]
ns["arr"].flags.writeable = False          # a NumPy write through the base would now raise
with CallLog() as log:
    first = views_intact(ns, manifest)
result["intact_in_place_calls"] = in_place(log.names)
result["unchanged"] = ([t._version for t in tensors] == versions
                       and [digest(t) for t in tensors] + [digest(ns["arr"])] == digests)
result["repeat_equal"] = views_intact(ns, manifest.to_json()) == first
ns["arr"].flags.writeable = True
with CallLog() as log:                       # the write-probe oracle, by contrast, writes
    storage_shared(ns["flat"], ns["tail"])
result["probe_in_place_calls"] = in_place(log.names)
# break one relationship of each kind the check claims to see
ns["tail_again"] = ns["tail"].clone()                                        # identity split
ns["top"].requires_grad_(False)                                              # attribute drift
ns["arr_T"] = ns["arr_T"].copy()                                             # no longer a view
ns["model"][2]._parameters["weight"] = torch.nn.Parameter(ns["model"][2].weight.detach().clone())  # new storage
result["broken"] = sorted((tuple(b["path"])[0], b["reason"].split(":")[0]) for b in views_intact(ns, manifest)["broken"])
''')
    assert res["intact_in_place_calls"] == []
    assert "__setitem__" in res["probe_in_place_calls"]   # the log does see a write
    assert res["unchanged"] is True
    assert res["repeat_equal"] is True
    reasons = dict(res["broken"])
    assert set(reasons) == {"tail", "top", "arr_T", "model"}, res["broken"]
    assert reasons["tail"] == "paths no longer reference one object"
    assert reasons["top"].startswith("requires_grad")
    assert reasons["arr_T"] == "does not share the base's buffer"
    assert reasons["model"] == "does not share the base's storage"


def test_apply_is_idempotent():
    """Resuming a restore may re-apply the manifest; a second pass is a no-op."""
    _, res = cross(
        build=_COMBINED,
        inspect='''
objs = [ns["model"][0].weight, ns["top"], ns["arr_v"], ns["arr_T"]]
ids, ptrs = [id(o) for o in objs], [ptr(o) for o in objs]
again = apply_view_manifest(ns, manifest)
objs = [ns["model"][0].weight, ns["top"], ns["arr_v"], ns["arr_T"]]
result["again"] = again
result["same_objects"] = [id(o) for o in objs] == ids and [ptr(o) for o in objs] == ptrs
''')
    assert res["report"]["failed"] == [] and res["again"]["failed"] == []
    assert res["again"]["restored"] == res["report"]["restored"]
    assert res["same_objects"] is True
