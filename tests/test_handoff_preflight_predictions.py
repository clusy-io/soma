"""What the capsule will not carry is known before the cut, and said.

Two gaps the first round left silent until after the capture and transfer:

  * The storage adapter predicts, at capture, the view records its restore
    will refuse (`predicted_failures`: a dtype reinterpretation, an autograd
    view it cannot rebuild, an ndarray subclass, ...). Nothing read that
    prediction, so such a switch closed the gate, captured, transferred and
    only then aborted at DEST_RESTORED (`views_not_restored`). The preflight
    now runs `collect_view_manifest` over a COPY of the names the capture
    would dump and refuses before the gate closes
    (`preflight_views_unsupported:<paths>`); what the adapter leaves outside
    its contract (`unsupported`) is reported, not blocking. The hidden bases
    the collection injects land in that copy and are removed in a `finally`,
    so judging the source leaves nothing in it.
  * Pickle does not carry `.grad`. The capsule carries, by value, the
    gradients of top-level tensors, parameters of top-level modules and
    optimizer slots. Every other non-None gradient arrives as None, and a
    carried gradient that shares storage with another tensor arrives
    unshared. The preflight lists both under `grads_not_carried`, the
    commit-boundary fingerprint declares the same list (computed by the same
    code), and no compared field covers them, so nothing reports them equal.

Each check also runs against a deliberately broken variant where that is
meaningful, so a green test says the evidence can tell the difference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from handoff import controller as ctl  # noqa: E402
from handoff.controller import MARK, Controller, Journal, _wrap, seed_program  # noqa: E402

localapi = pytest.importorskip("localapi")


def program(body: str) -> str:
    return _wrap('import sys, json, os\ng = sys.modules["__main__"].__dict__\n' + body)


#: The names in the source's `__main__` that carry the storage adapter's
#: hidden-base prefix.
HIDDEN = program(f'print({MARK!r} + json.dumps(sorted(k for k in g if k.startswith("__clusy_view_base_"))))')

#: Two views of one base that no name binds: the collection must inject a
#: hidden base for them. `pf_bits` reinterprets the bytes as int32, which the
#: adapter predicts it cannot repair (a dtype mismatch with its base).
VIEWS_REFUSED = '''
import torch
_t = torch.arange(8.)
pf_head = _t[:4]
pf_bits = _t[4:].view(torch.int32)
del _t
'''

#: The same shape of state, repairable: two overlapping slices of an unbound
#: base. The collection injects a hidden base and predicts nothing.
VIEWS_REPAIRABLE = '''
import torch
_t = torch.arange(10.)
pf_a, pf_b = _t[1:5], _t[3:7]
del _t
'''

#: A CPU tensor over a NumPy buffer: outside the adapter's contract, reported
#: in `unsupported`, never repaired.
CROSS_FRAMEWORK = '''
import numpy as np, torch
xs_arr = np.arange(6.0)
xs_t = torch.from_numpy(xs_arr)
'''

#: Gradients the capsule cannot carry as they are: a module reachable only
#: through a list (its gradients arrive as None), and the model's gradients
#: re-pointed at views of one flat buffer (their values are carried, their
#: sharing is not).
GRADS_UNCARRIED = '''
import torch
nested = [torch.nn.Linear(3, 2)]
nested[0](torch.ones(1, 3)).sum().backward()
flat_grad = torch.zeros(sum(p.numel() for p in model.parameters()))
_off = 0
for _p in model.parameters():
    _p.grad = flat_grad[_off:_off + _p.numel()].view_as(_p)
    _off += _p.numel()
del _off, _p
'''

GRAD_PROBE = program(f'''
import hashlib
m = g["model"]
print({MARK!r} + json.dumps({{
    "model": [None if p.grad is None else hashlib.sha256(p.grad.numpy().tobytes()).hexdigest() for p in m.parameters()],
    "nested": [p.grad is None for p in g["nested"][0].parameters()],
    "shares_flat": [p.grad is not None and p.grad.untyped_storage().data_ptr() == g["flat_grad"].untyped_storage().data_ptr()
                    for p in m.parameters()]}}))
''')


@pytest.fixture
def api(tmp_path):
    a = localapi.LocalApi(workdir=tmp_path / "kernels")
    yield a
    a.close()


def _source(api, name: str, setup: str) -> str:
    pid = api.create_project(f"clusy-exp-handoff-src-{name}", "cpu")
    api.witness(pid, seed_program(seed=7, files=2, corpus_bytes=2048))
    st, _out, payload = api.execute(pid, setup)            # a user cell, run in __main__ itself
    assert st == 200 and not payload.get("error"), payload
    return pid


def _fp(api, pid):
    fp = api.witness(pid, ctl.fingerprint_program())
    fp.pop("fingerprint_ms", None)
    return fp


def _describe(api) -> dict:
    dest = api.create_project("clusy-exp-handoff-describe", "cpu")
    try:
        return api.witness(dest, ctl.describe_program())
    finally:
        api.delete_project(dest)


# ---------------------------------------------------------------------------
# Views: refused before the cut, and the source left as it was
# ---------------------------------------------------------------------------

def test_a_view_the_adapter_predicts_it_cannot_repair_is_refused_before_the_cut(api, tmp_path):
    """Old code: the preflight never read the prediction, so the switch
    closed the gate, captured and transferred, and aborted only at
    DEST_RESTORED (`views_not_restored`). Now: ABORTED at PREFLIGHT with the
    path in the reason, the gate never closed, nothing captured, and the
    source's `__main__` holds no hidden base although the collection had to
    inject one."""
    src = _source(api, "pf-views", VIEWS_REFUSED)
    fp0 = _fp(api, src)
    j = Journal(tmp_path / "j.sqlite")
    res = Controller(api, j, tmp_path / "blobs").migrate("pf-views", src, "cpu")
    assert res["phase"] == "ABORTED" and res["failed_at"] == "PREFLIGHT", res
    assert res["reason"] == "preflight_views_unsupported:pf_bits", res
    pf = res["preflight"]
    assert [(p["path"], p["reason"]) for p in pf["views_predicted_failures"]] == [(["pf_bits"], "dtype mismatch with base")]
    assert pf["blocking"] == ["views_unsupported:pf_bits"]
    assert pf["views_injected_bases"] == ["__clusy_view_base_0"]      # the collection did inject, into its copy
    assert pf["views_hidden_left_in_main"] == [] and api.witness(src, HIDDEN) == []
    row = j.get("pf-views")
    assert row["authoritative"] == src and row["admission"] == "open" and not row.get("capsule_path")
    assert all(e["phase"] not in ("ADMISSION_CLOSED", "CAPTURED") for e in j.events("pf-views"))
    ev = next(json.loads(e["note"]) for e in j.events("pf-views") if e["phase"] == "PREFLIGHT")
    assert ev["views_predicted_failures"] == ["pf_bits"]
    assert _fp(api, src) == fp0                                      # the source is exactly as it was


def test_the_preflight_leaves_no_hidden_base_and_a_leak_is_refused(api, tmp_path, monkeypatch):
    """The preflight program on its own, on a source whose views need a
    hidden base and are repairable: it injects one (into its copy), predicts
    nothing, and leaves `__main__` and the fingerprint as they were. The
    control: a variant that collects over the REAL `__main__` and never undoes
    the injection leaves the name behind, and the controller refuses that
    preflight (`preflight_views_error`) instead of reading it as a pass."""
    src = _source(api, "pf-hidden", VIEWS_REPAIRABLE)
    fp0, desc = _fp(api, src), _describe(api)
    pf = api.witness(src, ctl.preflight_program(desc))
    views = pf["views"]
    assert views["error"] is None and views["records"] == 2 and views["predicted_failures"] == []
    assert views["injected_bases"] == ["__clusy_view_base_0"] and views["hidden_left_in_main"] == []
    assert api.witness(src, HIDDEN) == [] and _fp(api, src) == fp0
    # The broken variant.
    real = ctl.preflight_program
    leaky = lambda d: real(d).replace("_vns = dict(_dumps)", "_vns = _g").replace(   # noqa: E731
        "for _k in [k for k in _vns if k.startswith(HIDDEN_BASE_PREFIX) and k not in _dumps]:", "for _k in []:")
    assert leaky(desc) != real(desc)
    monkeypatch.setattr(ctl, "preflight_program", leaky)
    other = _source(api, "pf-leak", VIEWS_REPAIRABLE)
    res = Controller(api, Journal(tmp_path / "j.sqlite"), tmp_path / "blobs").migrate("pf-leak", other, "cpu")
    assert res["phase"] == "ABORTED" and res["reason"] == "preflight_views_error", res
    assert res["preflight"]["views_hidden_left_in_main"] == ["__clusy_view_base_0"]
    assert api.witness(other, HIDDEN) == ["__clusy_view_base_0"]


def test_views_outside_the_contract_are_reported_not_blocking(api, tmp_path):
    """A tensor over a NumPy buffer is outside the adapter's contract: the
    preflight reports it (`views_unsupported`, with the path it shares with)
    and the switch proceeds. Old code: not reported anywhere before the
    capture."""
    src = _source(api, "pf-unsup", CROSS_FRAMEWORK)
    j = Journal(tmp_path / "j.sqlite")
    res = Controller(api, j, tmp_path / "blobs").migrate("pf-unsup", src, "cpu")
    assert res["phase"] == "DONE", res
    pf = res["preflight"]
    assert pf["decision"] == "proceed" and pf["views_predicted_failures"] == []
    assert [(u["path"], u["with"]) for u in pf["views_unsupported"]] == [(["xs_t"], ["xs_arr"])]
    assert "NumPy" in pf["views_unsupported"][0]["reason"]
    ev = next(json.loads(e["note"]) for e in j.events("pf-unsup") if e["phase"] == "PREFLIGHT")
    assert ev["views_unsupported"] == ["xs_t"]


# ---------------------------------------------------------------------------
# Gradients: what the capsule does not carry is declared end to end
# ---------------------------------------------------------------------------

#: The gradients GRADS_UNCARRIED leaves the capsule unable to carry as they are.
SHARED = [f"model.param.{n}.grad" for n in ("bn.bias", "bn.weight", "fc1.bias", "fc1.weight", "fc2.bias", "fc2.weight")]
NESTED = ["nested[0].param.bias.grad", "nested[0].param.weight.grad"]


def test_gradients_the_capsule_does_not_carry_are_declared_end_to_end(api, tmp_path):
    """Old code: the preflight said nothing about gradients, the fingerprint
    had no `grads_not_carried`, and the migration committed with the nested
    module's gradients gone and the model's no longer views of `flat_grad`,
    the boundary reporting equal with nothing declared. Now the preflight
    lists each one with its reason, the commit boundary declares exactly that
    list at both stages, the destination shows what the declaration says (the
    nested gradients None, the model's equal by value and unshared), and no
    compared field covers the dropped ones."""
    src = _source(api, "pf-grads", GRADS_UNCARRIED)
    before = api.witness(src, GRAD_PROBE)
    assert all(before["shares_flat"]) and before["nested"] == [False, False]
    j = Journal(tmp_path / "j.sqlite")
    res = Controller(api, j, tmp_path / "blobs").migrate("pf-grads", src, "cpu")
    assert res["phase"] == "DONE", res
    pf = res["preflight"]
    assert sorted(pf["grads_not_carried"]) == sorted(SHARED + NESTED)
    assert pf["grads_carried"] == 6
    assert all(pf["grads_not_carried"][p].startswith("carried by value; its storage sharing with flat_grad")
               for p in SHARED)
    assert all(pf["grads_not_carried"][p].startswith("not carried: pickle drops .grad") for p in NESTED)
    ev = next(json.loads(e["note"]) for e in j.events("pf-grads") if e["phase"] == "PREFLIGHT")
    assert ev["grads_not_carried"] == sorted(SHARED + NESTED)
    cb = res["commit_boundary"]
    assert cb["source"]["grads_not_carried"] == pf["grads_not_carried"]      # one list, computed by one code path
    for stage in ("restored", "pre_commit"):
        v = cb["verdicts"][stage]
        assert v["equal"] and v["declared"]["grads_not_carried"] == sorted(SHARED + NESTED), v
    # Nothing compares the dropped gradients as equal: no compared field has them.
    assert not set(NESTED) & {p + ".grad" for p in cb["source"]["grads"]}
    after = api.witness(res["dest"], GRAD_PROBE)
    assert after["model"] == before["model"]                 # carried by value
    assert after["shares_flat"] == [False] * 6               # the sharing is not, as declared
    assert after["nested"] == [True, True]                   # arrived as None, as declared
    assert cb["pre_commit"]["grads_not_carried"] == {}       # nothing left for the destination to declare


def test_a_gradient_dropped_without_declaration_is_a_mismatch():
    """The comparison declares the SOURCE's list only. A gradient the
    destination lists that the cut did not (one that arrived differently
    from how it left) is a change, not a declared drop."""
    base = {f: {} for f in ctl.BOUNDARY_FIELDS}
    base["rng"] = {"python": {"sha256": "p"}, "numpy": {"loaded": True}, "torch_cpu": {"loaded": True}, "named": {}}
    src = dict(base, grads_not_carried={"a.grad": "not carried: x"})
    v = ctl.compare_fingerprints(src, dict(base, grads_not_carried={"a.grad": "not carried: x"}))
    assert v["equal"] and v["declared"]["grads_not_carried"] == ["a.grad"]
    v = ctl.compare_fingerprints(dict(base), dict(base, grads_not_carried={"b.grad": "carried by value; ..."}))
    assert not v["equal"] and v["mismatched"] == ["grads_not_carried"] and v["details"]["grads_not_carried"] == ["b.grad"]
