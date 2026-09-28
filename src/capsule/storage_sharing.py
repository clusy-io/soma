"""Preserve storage sharing across a serialization boundary for common views.

THE GAP THIS CLOSES. `dill` (and `pickle`, and `torch.save` for NumPy) serialize
each tensor or array object independently. A tensor and a view of it are two
Python objects, so they are written as two independent storages and come back
as two independent storages. Every value is correct; the *relationship* is gone.
Measured in E1c: write -999.0 through the base, the view still reads 5.0.

Nothing raises, and the published `id()`-granular correctness predicate is
satisfied vacuously, because the two objects never shared an id. This adapter is
the constructive answer: a bounded manifest of view relationships captured
before the dump and re-applied after the load.

WHAT "BOUNDED" MEANS, PRECISELY.

  * Views are recognised through the framework's own pointers: the untyped
    storage for torch (so `nn.Parameter(t[:4])`, whose `_base` is None, is
    still seen) and `ndarray.base` for NumPy. A NumPy view whose base is not
    itself an array (e.g. an array over a bytes buffer) is reported, not
    adapted.
  * The base must be reachable. If the base is bound in the namespace under some
    name it is used; if it is not, it is carried across under a hidden name so it
    is serialized exactly once. Views are re-derived from that single base.
  * Paths covered: top-level names, one level into list/tuple/dict, and
    parameters/buffers of an `nn.Module` bound at top level (every name of a
    tied parameter, not only the first). An object reachable ONLY deeper than
    that is not seen at all: its sharing is neither preserved nor reported.
  * Geometry is (shape, stride, storage_offset) for torch and (shape, strides,
    byte offset) for NumPy. That covers slices, transposes, reshapes of
    contiguous tensors, `narrow`, `select`, `unfold`, and `expand`/broadcast
    views; anything `as_strided` can express.
  * Sharing ACROSS frameworks (`torch.from_numpy`, `Tensor.numpy()`) is
    reported in `unsupported`, never repaired: each side is serialized by its
    own framework and comes back over its own buffer.
  * The serializer carries no autograd graph: every non-leaf tensor comes back
    as a leaf. The one graph edge that IS a storage relationship, a non-leaf
    view of a leaf that requires grad (`Wv = W[:4]` of a Parameter `W`), is
    re-created (below). Any other non-leaf in a storage group is a failure. A
    non-leaf that shares storage with nothing (`h = W * 2`) is outside this
    adapter; its lost graph is the serializer's, and nothing here reports it.

REPAIR PRESERVES THE OBJECT GRAPH, NOT ONLY THE STORAGE. A view restored by the
serializer is one object that other objects already point at: a second name, a
module's `_parameters` slot, an optimizer's `param_groups` list and its `state`
keys, an external alias. The first version built a NEW view and rebound it at
one path, so every other reference kept the stale copy: two names bound to one
view became two objects, an optimizer stepped a Parameter the model no longer
held, and a top-level Parameter came back as a plain Tensor without
requires_grad. All of that reported zero failures. The rule now:

  * Torch leaves: repair IN PLACE. `Tensor.set_` re-points the restored object
    at the base's storage with the recorded geometry. No object is replaced,
    so its class, requires_grad, .grad, module registration, optimizer params
    and state keys, and every alias anywhere in the heap stay what the
    serializer made them (the pickle memo already preserved identity).
  * Torch non-leaf views of the anchor: `set_` is NOT enough. The serializer
    made the view a leaf, and a leaf re-pointed at W's storage still takes
    its own gradient: a loss computed through it leaves W.grad at None, so an
    optimizer on W silently does nothing. Such a view is rebuilt as
    `anchor.as_strided(...)` with grad enabled, which is an autograd view of
    the anchor again, and replaces the restored object at every reference
    exactly like a NumPy array (next point). It can be rebuilt only when the
    anchor was the view's autograd base and a leaf that requires grad; a
    non-leaf in any other position fails with `autograd view: gradient link
    not carried`.
  * NumPy: an ndarray's buffer cannot be re-pointed, so the repaired array must
    replace the old one at EVERY reference. Before touching anything, the
    adapter counts the old object's references and compares them with the
    covered slots that hold it. The reference count sees every holder, which
    the collector does not (a dict or tuple holding only arrays is untracked,
    so `gc.get_referrers` misses an attribute of an unpickled object). If
    anything else holds it, the record fails with `reference outside the
    covered paths` and nothing has moved. Otherwise every slot is rebound and
    a weakref proves the old object died, else every rebind is undone. All or
    none, per record. Garbage is collected at most once per call, and only if
    a count is off, so a restore of N views costs O(N) and not O(N x heap).
  * Anything the adapter cannot repair without breaking another relationship
    is a FAILURE in the report, never a silent success: a non-tensor or
    non-array at the path, a dtype or device that differs from the base, an
    ndarray subclass (re-deriving it would drop the subclass's own state), a
    view that is itself the base of another record, a tuple also held outside
    the namespace, a non-leaf view that cannot be rebuilt.

`views_intact` checks the relationships without writing anything, for a
controller's validation and commit-boundary checks: storage, geometry, class,
requires_grad, leaf-ness and, for a rebuilt autograd view, that its autograd
base is the recorded base. `storage_shared` remains the behavioural
write-probe oracle the tests use.

What the capture already knows is reported before any bytes move. Everything
the adapter does not handle is in the manifest's `unsupported` list, and every
recorded view whose repair the SOURCE state already shows will be refused
(dtype reinterpretation, a non-leaf that cannot be rebuilt, an ndarray
subclass, a tuple subclass, a replaced tensor at a module path) is in
`predicted_failures`, with the reason string the restore will report, so a
caller can refuse or exclude before the cut. One refusal is NOT predicted: a
NumPy view held outside the covered paths (an attribute of a dataset object,
say). The source holds references the capsule does not carry, IPython's
output history and every name the capture excludes among them, so a count
taken there would refuse switches that restore cleanly; the exact check runs
at the destination. Silence is the failure mode this exists to remove.
"""

from __future__ import annotations

import dataclasses
import gc
import sys
import types
import weakref
from dataclasses import dataclass, field
from typing import Any

HIDDEN_BASE_PREFIX = "__clusy_view_base_"

#: The failure reason for a view some holder outside the covered paths still
#: references. Named, because a controller keys on it.
REASON_OUTSIDE = "reference outside the covered paths"

#: The failure reason for a non-leaf tensor whose gradient path the repair
#: cannot re-create (its autograd base is not the anchor, or the anchor is not
#: a leaf that requires grad).
REASON_AUTOGRAD = "autograd view: gradient link not carried"

#: `ViewRecord.autograd` values. "leaf": repaired in place. "view_of_base": a
#: non-leaf view whose autograd base is the anchor, a leaf that requires grad;
#: rebuilt as an autograd view of it. "not_carried": any other non-leaf.
AUTOGRAD_LEAF, AUTOGRAD_VIEW, AUTOGRAD_NOT_CARRIED = "leaf", "view_of_base", "not_carried"

# Frames of this module are the walk's own and are excluded from the holder
# diagnosis. Compared by globals dict because the controller loads this file
# into a synthetic module inside the kernel, where `__name__` is not a
# reliable key.
_THIS_MODULE_GLOBALS = globals()


@dataclass
class ViewRecord:
    """One view OBJECT and how to rebuild it from its base."""

    path: tuple  # ("name",) | ("name", key) | ("name", "param", "layer.weight"); the first of `paths`
    kind: str  # "torch" | "numpy"
    base_name: str  # namespace name the base is (or will be) bound under
    shape: tuple
    stride: tuple  # torch: element strides. numpy: byte strides.
    offset: int  # torch: storage_offset (elements). numpy: byte offset from base data ptr.
    dtype: str
    #: every covered path that reached this object at capture, `path` first.
    #: The NumPy repair rebinds each of them, and `views_intact` checks they
    #: still reach ONE object. Empty in manifests written before this field.
    paths: list = field(default_factory=list)
    #: class name at capture ("Parameter", "Tensor", "ndarray")
    cls: str = ""
    #: torch only: requires_grad at capture
    requires_grad: bool | None = None
    #: numpy only: flags.writeable at capture (a broadcast view is read-only)
    writeable: bool | None = None
    #: torch only: AUTOGRAD_LEAF, AUTOGRAD_VIEW or AUTOGRAD_NOT_CARRIED. None in
    #: manifests written before this field, which are repaired in place as
    #: leaves and checked without leaf-ness, as they were.
    autograd: str | None = None

    def all_paths(self) -> list[tuple]:
        return [tuple(p) for p in self.paths] or [tuple(self.path)]

    def to_json(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["path"] = list(self.path)
        d["shape"] = list(self.shape)
        d["stride"] = list(self.stride)
        d["paths"] = [list(p) for p in self.all_paths()]
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "ViewRecord":
        return cls(
            path=tuple(d["path"]), kind=d["kind"], base_name=d["base_name"],
            shape=tuple(d["shape"]), stride=tuple(d["stride"]),
            offset=int(d["offset"]), dtype=str(d["dtype"]),
            paths=[tuple(p) for p in (d.get("paths") or [d["path"]])],
            cls=str(d.get("cls") or ""), requires_grad=d.get("requires_grad"),
            writeable=d.get("writeable"), autograd=d.get("autograd"),
        )


@dataclass
class ViewManifest:
    views: list[ViewRecord] = field(default_factory=list)
    #: bases injected under hidden names so they serialize once
    injected_bases: list[str] = field(default_factory=list)
    #: things that look like views but are outside the bounded contract
    unsupported: list[dict[str, Any]] = field(default_factory=list)
    #: for each hidden base, the covered paths that also reach it (empty for a
    #: synthetic full-storage anchor). Lets `views_intact` find the base after
    #: the capture has removed the hidden name from the source again.
    base_paths: dict[str, list] = field(default_factory=dict)
    #: recorded views whose repair the source state already shows will be
    #: refused, as {path, kind, reason, detail} with the reason the restore
    #: reports. Recorded all the same, so a restore that goes ahead anyway
    #: still fails explicitly rather than dropping the view.
    predicted_failures: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "views": [v.to_json() for v in self.views],
            "injected_bases": list(self.injected_bases),
            "unsupported": list(self.unsupported),
            "base_paths": {k: [list(p) for p in ps] for k, ps in self.base_paths.items()},
            "predicted_failures": list(self.predicted_failures),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "ViewManifest":
        return cls(
            views=[ViewRecord.from_json(v) for v in d.get("views", [])],
            injected_bases=list(d.get("injected_bases", [])),
            unsupported=list(d.get("unsupported", [])),
            base_paths={k: [tuple(p) for p in ps] for k, ps in (d.get("base_paths") or {}).items()},
            predicted_failures=list(d.get("predicted_failures") or []),
        )

    @property
    def preserved(self) -> int:
        return len(self.views)


def _as_manifest(manifest: Any) -> ViewManifest:
    return manifest if isinstance(manifest, ViewManifest) else ViewManifest.from_json(manifest)


# ---------------------------------------------------------------------------
# Walking the namespace
# ---------------------------------------------------------------------------


def _module_type():
    try:
        import torch
        return torch.nn.Module
    except Exception:  # pragma: no cover
        return ()


def _named(mod: Any, which: str):
    """Every (name, tensor) of a module, tied ones under each of their names.

    `remove_duplicate=False` matters for the reference list, not for grouping:
    a parameter tied under two names is one object, and both names are paths
    that must still reach it after restore.
    """
    fn = getattr(mod, f"named_{which}")
    try:
        return list(fn(recurse=True, remove_duplicate=False))
    except TypeError:  # pragma: no cover - torch before 2.0
        return list(fn(recurse=True))


def _iter_references(ns: dict[str, Any]):
    """Yield (path, object) for EVERY covered reference, repeats included.

    A container that is the namespace itself is skipped, and so are dunder
    names other than hidden bases (interpreter and harness state, never user
    data the capture carries).
    """
    Module = _module_type()
    for name, val in list(ns.items()):
        if name.startswith("__") and not name.startswith(HIDDEN_BASE_PREFIX):
            continue
        if val is ns:
            continue
        yield (name,), val
        if isinstance(val, (list, tuple)):
            for i, item in enumerate(val):
                yield (name, i), item
        elif isinstance(val, dict):
            for k, item in list(val.items()):
                if isinstance(k, (str, int)):
                    yield (name, k), item
        elif Module and isinstance(val, Module):
            for pname, p_ in _named(val, "parameters"):
                yield (name, "param", pname), p_
            for bname, b in _named(val, "buffers"):
                yield (name, "buffer", bname), b


def _candidates(ns: dict[str, Any]) -> list[tuple[list[tuple], Any]]:
    """(paths, object) once per OBJECT, in first-reached order.

    One object reachable by two names is identity aliasing, which the pickle
    memo already preserves; storage sharing is about DISTINCT objects over one
    buffer, and counting one object twice would manufacture a false group. The
    other paths are kept, though, because the repair must honour every one.
    """
    order: list[tuple[list[tuple], Any]] = []
    index: dict[int, int] = {}
    for path, obj in _iter_references(ns):
        i = index.get(id(obj))
        if i is None:
            index[id(obj)] = len(order)
            order.append(([path], obj))
        else:
            order[i][0].append(path)
    return order


def _resolve(ns: dict[str, Any], path: tuple) -> Any:
    """The object at `path`. Raises KeyError/IndexError/AttributeError if absent."""
    if len(path) == 1:
        return ns[path[0]]
    container = ns[path[0]]
    if len(path) == 3 and path[1] in ("param", "buffer"):
        mod = container
        *parents, leaf = str(path[2]).split(".")
        for p in parents:
            mod = getattr(mod, p)
        return getattr(mod, leaf)
    return container[path[1]]


def _resolve_all(ns: dict[str, Any], paths: list[tuple]):
    """(object, None) if every path reaches the same object, else (None, problem)."""
    obj, first = None, None
    for p in paths:
        try:
            cur = _resolve(ns, p)
        except Exception as exc:  # noqa: BLE001
            return None, ("path missing after load", f"{list(p)}: {type(exc).__name__}: {exc}")
        if first is None:
            obj, first = cur, p
        elif cur is not obj:
            return None, ("paths no longer reference one object",
                          f"{list(first)} and {list(p)} are different objects")
    return obj, None


def _is_torch(x: Any) -> bool:
    try:
        import torch
        return isinstance(x, torch.Tensor)
    except Exception:
        return False


def _is_numpy(x: Any) -> bool:
    try:
        import numpy as np
        return isinstance(x, np.ndarray)
    except Exception:
        return False


def _np_root(a: Any) -> Any:
    """The last ndarray on `a`'s base chain (NumPy collapses view chains to
    the owner, so this is normally one step)."""
    root = a
    while _is_numpy(getattr(root, "base", None)):
        root = root.base
    return root


def _np_ptr(a: Any) -> int:
    return int(a.__array_interface__["data"][0])


def _find_name(ns: dict[str, Any], obj: Any) -> str | None:
    """Top-level name bound to `obj`, or None. Only top-level names can serve
    as a base reference in the manifest; a base living inside a container is
    carried under a hidden name instead (dill's memo makes that free: the same
    object reached by two paths is serialized once and restored as one)."""
    for k, v in ns.items():
        if v is obj:
            return k
    return None


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def collect_view_manifest(ns: dict[str, Any]) -> ViewManifest:
    """Record every view the contract covers. MUTATES `ns` only to add hidden
    base names when a view's base is not otherwise reachable; nothing else.

    Torch views are found by STORAGE POINTER, not by `_base`. `_base` misses a
    real and common case: `nn.Parameter(t[:4])` shares t's storage and has
    `_base is None`, because Parameter construction goes through
    `_make_subclass`. Every tensor in a storage group is re-derived from one
    anchor covering the whole storage; if no namespace tensor covers it, a
    synthetic full-storage view is injected under a hidden name.

    Each OBJECT is recorded once, with every covered path that reaches it.
    A torch record also carries its autograd kind (which repair keeps its
    gradient path), and every record the source already shows will be
    refused is listed in `predicted_failures` as well as recorded.
    """
    m = ViewManifest()
    injected: dict[int, str] = {}  # id(base) -> hidden name

    def predict(rec: ViewRecord, reason: str, detail: str) -> None:
        m.predicted_failures.append({"path": list(rec.path), "kind": rec.kind,
                                     "reason": reason, "detail": detail})

    def base_name_for(base: Any, reached_at: list[tuple]) -> str:
        nm = _find_name(ns, base)
        if nm is not None:
            return nm
        if id(base) in injected:
            return injected[id(base)]
        nm = f"{HIDDEN_BASE_PREFIX}{len(injected)}"
        ns[nm] = base
        injected[id(base)] = nm
        m.injected_bases.append(nm)
        m.base_paths[nm] = [tuple(p) for p in reached_at]
        return nm

    candidates = _candidates(ns)
    paths_of = {id(obj): paths for paths, obj in candidates}

    # ---- torch: storage groups --------------------------------------------
    groups: dict[tuple, list[tuple[list[tuple], Any]]] = {}
    for paths, obj in candidates:
        if not _is_torch(obj):
            continue
        try:
            if obj.numel() == 0 or obj.device.type == "meta":
                continue  # no bytes behind it, so nothing to alias
            key = (obj.untyped_storage().data_ptr(), str(obj.device))
        except Exception:  # sparse / quantized / uninitialized tensors expose no storage
            # Its own reason: this is not a view the contract refused, it is a
            # tensor whose sharing (a sparse tensor over a dense values tensor,
            # say) could not even be looked for.
            m.unsupported.append({"path": list(paths[0]), "paths": [list(p) for p in paths],
                                  "kind": "torch", "reason": REASON_NO_STORAGE})
            continue
        groups.setdefault(key, []).append((paths, obj))
    for key, members in groups.items():
        if len(members) < 2:
            continue  # nothing to share; a lone reshape of a temporary is not aliasing
        # Anchor: a member covering the full storage from offset 0, else
        # synthetic. A LEAF is preferred among full-storage members: it is the
        # autograd base the group's non-leaf views hang from, so choosing a
        # full-storage non-leaf (`W.view(2, 4)`) would make every other view
        # of W unrebuildable for no reason.
        def covers_all(t):
            return t.storage_offset() == 0 and t.numel() * t.element_size() == t.untyped_storage().nbytes() \
                and t.is_contiguous()
        full = [t for _, t in members if covers_all(t)]
        anchor = next((t for t in full if t.is_leaf), full[0] if full else None)
        synthetic = anchor is None
        if synthetic:
            proto = members[0][1]
            n = proto.untyped_storage().nbytes() // proto.element_size()
            anchor = proto.as_strided((n,), (1,), 0)
        anchor_name = base_name_for(anchor, paths_of.get(id(anchor), []))
        for paths, t in members:
            if t is anchor:
                continue
            autograd = _autograd_kind(t, anchor, synthetic)
            rec = ViewRecord(
                path=paths[0], paths=list(paths), kind="torch", base_name=anchor_name,
                shape=tuple(t.shape), stride=tuple(t.stride()),
                offset=int(t.storage_offset()), dtype=str(t.dtype),
                cls=type(t).__name__, requires_grad=bool(t.requires_grad), autograd=autograd,
            )
            m.views.append(rec)
            # Same checks, in the same order, as `_repair_torch`, so the
            # predicted reason is the one the restore will report.
            if t.dtype != anchor.dtype:
                predict(rec, "dtype mismatch with base", f"view {t.dtype}, base {anchor.dtype}")
            elif autograd == AUTOGRAD_NOT_CARRIED:
                predict(rec, REASON_AUTOGRAD, _not_carried_detail(t, anchor, synthetic))
            elif autograd == AUTOGRAD_VIEW and any(len(p) == 3 for p in paths):
                predict(rec, "path not rebindable",
                        "a rebuilt view cannot be re-registered at a module path: "
                        + str([list(p) for p in paths if len(p) == 3]))

    # ---- numpy: base chain ---------------------------------------------------
    for paths, obj in candidates:
        if not _is_numpy(obj):
            continue
        base = obj.base
        if base is None:
            continue
        if not _is_numpy(base):
            m.unsupported.append({"path": list(paths[0]), "paths": [list(p) for p in paths],
                                  "kind": "numpy", "reason": f"base is {type(base).__name__}, not ndarray"})
            continue
        root = _np_root(obj)
        # Only aliasing that is OBSERVABLE from the namespace needs preserving:
        # a reshape of a temporary shares storage with nothing else bound.
        shares_with_other = any(
            (o2 is not obj) and _is_numpy(o2) and _np_shares(o2, obj) for _, o2 in candidates
        ) or _find_name(ns, root) is not None
        if not shares_with_other:
            continue
        rec = ViewRecord(
            path=paths[0], paths=list(paths), kind="numpy",
            base_name=base_name_for(root, paths_of.get(id(root), [])),
            shape=tuple(obj.shape), stride=tuple(obj.strides),
            offset=_np_ptr(obj) - _np_ptr(root), dtype=str(obj.dtype),
            cls=type(obj).__name__, writeable=bool(obj.flags.writeable),
        )
        m.views.append(rec)
        import numpy as np
        if type(obj) is not np.ndarray:  # same test, same order, as `_repair_numpy`
            predict(rec, "ndarray subclass", f"{type(obj).__name__}: re-deriving it would drop the subclass's own state")
            continue
        subclassed = sorted({str(p[0]) for p in paths if len(p) == 2
                             and isinstance(ns.get(p[0]), tuple) and type(ns.get(p[0])) is not tuple})
        if subclassed:
            predict(rec, "container cannot be rebuilt",
                    f"held in tuple subclass(es) at {subclassed}, which a plain tuple would not replace faithfully")

    _report_cross_framework(candidates, m)
    return m


#: The `unsupported` reason for a tensor with no untyped storage.
REASON_NO_STORAGE = "no untyped storage (sparse, quantized or opaque layout): sharing not checked"
#: The `unsupported` reason for a CPU tensor over a NumPy buffer.
REASON_CROSS_FRAMEWORK = "shares memory with a NumPy array (torch.from_numpy or Tensor.numpy()): not preserved"


def _autograd_kind(t: Any, anchor: Any, synthetic: bool) -> str:
    """Which repair keeps `t`'s gradient path (see AUTOGRAD_* above).

    A non-leaf can be rebuilt only as `anchor.as_strided(...)`, whose gradient
    reaches the anchor. That equals the source's path exactly when the
    anchor IS the view's autograd base (torch collapses view chains to it)
    and is a leaf that requires grad, and when no conj/neg bit separates the
    two (as_strided does not carry one).
    """
    if t.is_leaf:
        return AUTOGRAD_LEAF
    if (not synthetic and t._base is anchor and anchor.is_leaf and anchor.requires_grad
            and t.is_conj() == anchor.is_conj() and t.is_neg() == anchor.is_neg()):
        return AUTOGRAD_VIEW
    return AUTOGRAD_NOT_CARRIED


def _not_carried_detail(t: Any, anchor: Any, synthetic: bool) -> str:
    if synthetic:
        return "non-leaf at capture, and its autograd base is not bound on a covered path"
    if t._base is not anchor:
        return "non-leaf at capture whose autograd base is not the recorded base"
    if not (anchor.is_leaf and anchor.requires_grad):
        return "non-leaf view of a base that is not itself a leaf that requires grad"
    return "non-leaf view whose conjugate or negative bit differs from its base"


def _report_cross_framework(candidates: list[tuple[list[tuple], Any]], m: ViewManifest) -> None:
    """Report every CPU tensor whose storage overlaps a covered ndarray's
    buffer. Two live allocations cannot overlap, so an overlap IS sharing:
    `torch.from_numpy(a)` or `t.numpy()`. The pair is serialized by two
    frameworks into two buffers and nothing here re-creates it, so it is
    reported, never silently dropped. O((T + A) log A)."""
    spans = []
    for paths, obj in candidates:
        if _is_numpy(obj) and obj.size:
            try:
                lo, hi = _np_bounds(obj)
            except Exception:  # noqa: BLE001
                continue
            spans.append((lo, hi, paths[0]))
    if not spans:
        return
    spans.sort(key=lambda s: s[0])
    los = [s[0] for s in spans]
    reach = []  # reach[i]: the furthest end among spans[0..i]
    for s in spans:
        reach.append(max(reach[-1], s[1]) if reach else s[1])
    import bisect
    for paths, obj in candidates:
        if not _is_torch(obj):
            continue
        try:
            if obj.device.type != "cpu" or obj.numel() == 0:
                continue
            st = obj.untyped_storage()
            lo, hi = st.data_ptr(), st.data_ptr() + st.nbytes()
        except Exception:  # noqa: BLE001
            continue
        i = bisect.bisect_left(los, hi) - 1
        if i < 0 or reach[i] <= lo:
            continue
        while spans[i][1] <= lo:  # the overlapping span nearest to the left
            i -= 1
        m.unsupported.append({"path": list(paths[0]), "paths": [list(p) for p in paths], "kind": "torch",
                              "reason": REASON_CROSS_FRAMEWORK, "with": list(spans[i][2])})


def _np_bounds(a: Any) -> tuple[int, int]:
    try:
        from numpy.lib.array_utils import byte_bounds
    except ImportError:  # pragma: no cover - numpy before 2.0
        from numpy import byte_bounds
    lo, hi = byte_bounds(a)
    return int(lo), int(hi)


def _np_shares(a: Any, b: Any) -> bool:
    try:
        import numpy as np
        return bool(np.shares_memory(a, b))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Relationship checks (no writes). Shared by the repair and `views_intact`.
# ---------------------------------------------------------------------------


def _base_of(ns: dict[str, Any], manifest: ViewManifest, name: str) -> Any:
    base = ns.get(name)
    if base is not None:
        return base
    for p in manifest.base_paths.get(name, []):
        try:
            return _resolve(ns, tuple(p))
        except Exception:  # noqa: BLE001
            continue
    return None


def _torch_meta(t: Any, v: ViewRecord) -> str | None:
    """Everything about `t` the record pins that does not need the base:
    dtype, geometry, class, requires_grad, and leaf-ness. Leaf-ness is what
    separates a view whose gradient reaches its base from a detached leaf over
    the same bytes; a check without it passed the reviewer's broken case."""
    if str(t.dtype) != v.dtype:
        return f"dtype {t.dtype}, recorded {v.dtype}"
    if tuple(t.shape) != tuple(v.shape) or tuple(t.stride()) != tuple(v.stride) \
            or int(t.storage_offset()) != int(v.offset):
        return (f"geometry {tuple(t.shape)}/{tuple(t.stride())}/{t.storage_offset()}, "
                f"recorded {tuple(v.shape)}/{tuple(v.stride)}/{v.offset}")
    if v.cls and type(t).__name__ != v.cls:
        return f"class {type(t).__name__}, recorded {v.cls}"
    if v.requires_grad is not None and bool(t.requires_grad) != bool(v.requires_grad):
        return f"requires_grad {t.requires_grad}, recorded {v.requires_grad}"
    if v.autograd is not None and bool(t.is_leaf) != (v.autograd == AUTOGRAD_LEAF):
        return f"is_leaf {t.is_leaf}, recorded {'a leaf' if v.autograd == AUTOGRAD_LEAF else 'a non-leaf'}"
    return None


def _torch_check(t: Any, base: Any, v: ViewRecord) -> str | None:
    """None iff `t` is the recorded view of `base`'s storage. Reads pointers
    and metadata only."""
    if not _is_torch(t):
        return f"not a tensor ({type(t).__name__})"
    if not _is_torch(base):
        return f"base is not a tensor ({type(base).__name__})"
    if t.device != base.device:
        return f"device {t.device}, base on {base.device}"
    if t.untyped_storage().data_ptr() != base.untyped_storage().data_ptr():
        return "does not share the base's storage"
    reason = _torch_meta(t, v)
    if reason is None and v.autograd == AUTOGRAD_VIEW and t._base is not base:
        return "gradient link lost: the view's autograd base is not the recorded base"
    return reason


def _np_geometry(a: Any, v: ViewRecord) -> str | None:
    if str(a.dtype) != v.dtype:
        return f"dtype {a.dtype}, recorded {v.dtype}"
    if tuple(a.shape) != tuple(v.shape) or tuple(a.strides) != tuple(v.stride):
        return f"geometry {tuple(a.shape)}/{tuple(a.strides)}, recorded {tuple(v.shape)}/{tuple(v.stride)}"
    if v.cls and type(a).__name__ != v.cls:
        return f"class {type(a).__name__}, recorded {v.cls}"
    if v.writeable is not None and bool(a.flags.writeable) != bool(v.writeable):
        return f"writeable {a.flags.writeable}, recorded {v.writeable}"
    return None


def _np_check(a: Any, base: Any, v: ViewRecord) -> str | None:
    """None iff `a` is the recorded view of `base`'s buffer. Same owner (NumPy
    collapses view chains to it) plus the recorded byte offset from it: memory
    inside a live allocation can only be reached through a view of it."""
    if not _is_numpy(a):
        return f"not an ndarray ({type(a).__name__})"
    if not _is_numpy(base):
        return f"base is not an ndarray ({type(base).__name__})"
    if _np_root(a) is not _np_root(base):
        return "does not share the base's buffer"
    if _np_ptr(a) - _np_ptr(base) != int(v.offset):
        return f"byte offset {_np_ptr(a) - _np_ptr(base)}, recorded {v.offset}"
    return _np_geometry(a, v)


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


def _extent_bytes(shape: tuple, stride: tuple, offset: int, itemsize: int) -> int:
    """Bytes from the storage start to one past the view's last element."""
    if any(s == 0 for s in shape):
        return 0
    return (offset + sum((s - 1) * st for s, st in zip(shape, stride)) + 1) * itemsize


class _Apply:
    """State shared by the records of ONE `apply_view_manifest` call.

    A garbage cycle can hold a restored object for a while (a spent unpickler
    keeps its memo, which reaches every loaded object). One collection clears
    all of it, so the collector runs at most once per call, and only when a
    reference count comes out high. It used to run, together with a full
    `gc.get_referrers` heap walk, once per record: O(records x heap).
    """

    def __init__(self) -> None:
        self.collected = False

    def collect_once(self) -> bool:
        """Collect if this call has not yet; True iff it collected now."""
        if self.collected:
            return False
        self.collected = True
        gc.collect()
        return True


def _repair_torch(ns: dict[str, Any], manifest: ViewManifest, v: ViewRecord,
                  anchor_ids: set[int], ctx: _Apply) -> tuple | None:
    import torch
    t, problem = _resolve_all(ns, v.all_paths())
    if problem:
        return problem
    if not isinstance(t, torch.Tensor):
        return "not a tensor at the path", type(t).__name__
    base = _base_of(ns, manifest, v.base_name)
    if base is None:
        return "base missing after load", v.base_name
    if not isinstance(base, torch.Tensor):
        return "base is not a tensor", f"{v.base_name}: {type(base).__name__}"
    if t is base:
        return "view resolves to its own base", v.base_name
    if id(t) in anchor_ids:
        # a repair would move the storage the other record's views are re-derived from
        return "view is the base of another record", str(list(v.path))
    if _torch_check(t, base, v) is None:
        return None  # already the recorded view (an in-process call, or a repeat)
    # The recorded offset and strides are in elements of the view's dtype.
    # set_ would accept a different base dtype and reinterpret the bytes; the
    # contract refuses instead, so a mismatched base cannot pass as a repair.
    if str(t.dtype) != v.dtype or t.dtype != base.dtype:
        return "dtype mismatch with base", f"view {t.dtype} (recorded {v.dtype}), base {base.dtype}"
    if t.device != base.device:
        return "device mismatch with base", f"view on {t.device}, base on {base.device}"
    if tuple(t.shape) != tuple(v.shape):
        return "restored object differs from the record", f"shape {tuple(t.shape)}, recorded {tuple(v.shape)}"
    if v.offset < 0 or any(st < 0 for st in v.stride) or \
            _extent_bytes(v.shape, v.stride, v.offset, t.element_size()) > base.untyped_storage().nbytes():
        return "geometry outside the base's storage", f"{v.shape}/{v.stride}/{v.offset}"
    if v.autograd == AUTOGRAD_NOT_CARRIED:
        # set_ would make the storage right and leave a leaf that takes its
        # own gradient: exactly the silent success this refuses.
        return REASON_AUTOGRAD, "non-leaf at capture, and no repair re-creates its gradient path to the base"
    if v.autograd == AUTOGRAD_VIEW:
        box = [t]
        del t  # the box is now this frame's only reference (see `_replace_everywhere`)
        return _rebuild_autograd_view(ns, base, v, box, ctx)
    # IN PLACE. The object at the path keeps its identity, class, requires_grad
    # and .grad, so the module slot, the optimizer's params and state keys and
    # every alias still reach it. no_grad because set_ on a leaf that requires
    # grad is an in-place op autograd would otherwise refuse.
    with torch.no_grad():
        t.set_(base.untyped_storage(), int(v.offset), tuple(v.shape), tuple(v.stride))
    after = _torch_check(t, base, v)
    if after is not None:
        return "set_ did not produce the recorded view", after
    return None


def _rebuild_autograd_view(ns: dict[str, Any], base: Any, v: ViewRecord, box: list,
                           ctx: _Apply) -> tuple | None:
    """Replace the restored (leaf) object by `base.as_strided(...)`, an
    autograd view of the base, at every covered path or at none.

    The source's view took its gradient to the base; the restored leaf takes
    it to itself. Only a NEW tensor made from the base under grad mode can be
    a non-leaf view of it again, so this is a replacement, with the same
    all-or-none rules as a NumPy array. A non-leaf cannot be an optimizer
    param or a module Parameter, so the references to replace are names and
    container slots; a module buffer path cannot be re-registered and fails.
    """
    import torch
    paths = v.all_paths()
    if any(len(p) == 3 for p in paths):
        return ("path not rebindable", "a rebuilt view cannot be re-registered at a module path: "
                + str([list(p) for p in paths if len(p) == 3]))
    if not (base.is_leaf and base.requires_grad):
        return REASON_AUTOGRAD, f"base {v.base_name} is not a leaf that requires grad after load"
    with torch.enable_grad():  # a caller under no_grad would otherwise get a detached leaf
        new = base.as_strided(tuple(v.shape), tuple(v.stride), int(v.offset))
    if type(new) is not type(box[0]) or new.is_conj() != box[0].is_conj() or new.is_neg() != box[0].is_neg():
        return ("rebuilt view differs from the restored object",
                f"{type(new).__name__} conj={new.is_conj()} neg={new.is_neg()}, restored "
                f"{type(box[0]).__name__} conj={box[0].is_conj()} neg={box[0].is_neg()}")
    problem = _replace_everywhere(ns, paths, box, new, ctx)
    if problem:
        return problem
    after = _torch_check(new, base, v)
    return None if after is None else ("rebuilt view is not the recorded view", after)


def _refs_via(d: dict[str, Any], key: Any) -> int:
    return sys.getrefcount(d[key])


# The interpreter's own contribution to `_refs_via` (the argument on the
# stack), measured the same way on a value held by exactly one dict slot, so
# the tuple check below does not hard-code a CPython constant.
_REF_OVERHEAD = _refs_via({"k": (object(),)}, "k") - 1


def _box_refs(box: list) -> int:
    return sys.getrefcount(box[0])


# `_box_refs` of an object whose ONLY holder is the box: the box's own slot
# plus the interpreter's temporaries. Calibrated here rather than hard-coded,
# so a CPython that borrows stack references keeps the arithmetic right.
_BOX_ALONE = _box_refs([object()])


def _covered_name(name: Any) -> bool:
    return isinstance(name, str) and (not name.startswith("__") or name.startswith(HIDDEN_BASE_PREFIX))


def _tuple_problem(ns: dict[str, Any], name: str) -> tuple | None:
    """A tuple holding the old object must itself be replaced, which is only
    safe if every holder of the TUPLE is a covered namespace name: those all
    get the one replacement tuple. Anything else would keep the old tuple, and
    with it the old object, alive."""
    if type(ns[name]) is not tuple:
        return "container cannot be rebuilt", f"{name!r} is a {type(ns[name]).__name__}, not a plain tuple"
    names = [k for k in list(ns) if ns[k] is ns[name]]
    uncovered = [k for k in names if not _covered_name(k)]
    if uncovered:
        return REASON_OUTSIDE, f"tuple at {name!r} is also bound to uncovered names {uncovered}"
    if _refs_via(ns, name) - _REF_OVERHEAD != len(names):
        return REASON_OUTSIDE, f"tuple at {name!r} is also held outside the namespace"
    return None


def _slot_refs(ns: dict[str, Any], slots: list[tuple]) -> int:
    """How many strong references the covered `slots` hold: one per namespace
    name, and one per (container, key), so a list or tuple bound under two
    names is counted once."""
    held = set()
    for p in slots:
        held.add(("name", p[0]) if len(p) == 1 else (id(ns[p[0]]), p[1]))
    return len(held)


def _rebind(ns: dict[str, Any], slots: list[tuple], old: Any, new: Any) -> None:
    """Point every covered slot that holds `old` at `new`. Used forwards to
    repair and backwards to undo; a slot already moved is left alone, which is
    what makes two paths into one list (or one tuple under two names) safe."""
    for p in slots:
        name = p[0]
        if len(p) == 1:
            if ns.get(name) is old:
                ns[name] = new
            continue
        c = ns.get(name)
        if isinstance(c, tuple):
            if not any(x is old for x in c):
                continue
            nt = tuple(new if x is old else x for x in c)
            for k in [k for k in list(ns) if ns[k] is c]:
                ns[k] = nt
        elif isinstance(c, (list, dict)):
            if c[p[1]] is old:
                c[p[1]] = new


def _replace_everywhere(ns: dict[str, Any], paths: list[tuple], box: list, new: Any,
                        ctx: _Apply) -> tuple | None:
    """Put `new` at every covered path that holds `box[0]`, or at none.

    `box` must be the caller's ONLY reference to the old object (no local,
    list, tuple or closure of ours may hold it), because the count below and
    the liveness proof both read references to it. This function empties it.

    THE COUNT FIRST. The old object's reference count, less the box, must
    equal the references the covered slots hold. The count sees every strong
    holder, including the ones the collector cannot: an unpickled object's
    `__dict__` that holds only arrays is an untracked dict, invisible to
    `gc.get_referrers`, and it is exactly where a dataset keeps its split. A
    mismatch fails the record before anything moves, so a refused record,
    tuple included, is left as it was, object for object. A high count gets
    one collection first (a garbage holder is not a reference).

    THEN THE PROOF. After the rebind, the old object must die when the box is
    emptied. Given a correct count this always holds; it stays because the
    all-or-none promise should rest on behaviour, not on arithmetic. If the
    object survives, every slot gets it back. (A tuple then comes back as an
    equal tuple, not the same object: the original died with the rebind.)
    """
    slots = [p for p in paths if _resolve(ns, p) is box[0]]
    for name in dict.fromkeys(p[0] for p in slots if len(p) == 2 and isinstance(ns[p[0]], tuple)):
        problem = _tuple_problem(ns, name)
        if problem and problem[0] == REASON_OUTSIDE and ctx.collect_once():
            problem = _tuple_problem(ns, name)
        if problem:
            return problem
    extra = _box_refs(box) - _BOX_ALONE - _slot_refs(ns, slots)
    if extra > 0 and ctx.collect_once():
        extra = _box_refs(box) - _BOX_ALONE - _slot_refs(ns, slots)
    if extra > 0:
        return REASON_OUTSIDE, f"{extra} reference(s) besides the {len(slots)} covered slot(s)", "describe"
    try:
        _rebind(ns, slots, box[0], new)
    except Exception as exc:  # noqa: BLE001
        _rebind(ns, slots, new, box[0])
        return "rebind raised", f"{type(exc).__name__}: {exc}"
    ref = weakref.ref(box[0])
    box.clear()
    if ref() is not None:
        ctx.collect_once()
    survivor = ref()
    if survivor is None:
        return None
    _rebind(ns, slots, new, survivor)
    del survivor
    return (REASON_OUTSIDE, "still reachable after every covered slot was rebound, though the "
            "reference count matched the covered slots", "describe")


def _repair_numpy(ns: dict[str, Any], manifest: ViewManifest, v: ViewRecord,
                  anchor_ids: set[int], ctx: _Apply) -> tuple | None:
    """Replace the restored array by a view of the base at every reference,
    or at none (see `_replace_everywhere`)."""
    import numpy as np
    paths = v.all_paths()
    if any(len(p) == 3 for p in paths):
        return "path not rebindable for an ndarray", str([list(p) for p in paths if len(p) == 3])
    old, problem = _resolve_all(ns, paths)
    if problem:
        return problem
    if not isinstance(old, np.ndarray):
        return "not an ndarray at the path", type(old).__name__
    if type(old) is not np.ndarray:
        return "ndarray subclass", f"{type(old).__name__}: re-deriving it would drop the subclass's own state"
    base = _base_of(ns, manifest, v.base_name)
    if base is None:
        return "base missing after load", v.base_name
    if not isinstance(base, np.ndarray):
        return "base is not an ndarray", f"{v.base_name}: {type(base).__name__}"
    if old is base:
        return "view resolves to its own base", v.base_name
    if id(old) in anchor_ids:
        return "view is the base of another record", str(list(v.path))
    if _np_check(old, base, v) is None:
        return None  # already the recorded view
    if str(old.dtype) != v.dtype or tuple(old.shape) != tuple(v.shape):
        return "restored object differs from the record", f"{old.dtype}{tuple(old.shape)}, recorded {v.dtype}{tuple(v.shape)}"
    new = np.ndarray(tuple(v.shape), dtype=np.dtype(v.dtype), buffer=base,
                     offset=int(v.offset), strides=tuple(v.stride))
    if v.writeable is False:
        new.flags.writeable = False
    box = [old]
    del old  # the box is now this frame's only reference
    return _replace_everywhere(ns, paths, box, new, ctx)


def _describe_holders(ns: dict[str, Any], pending: list[tuple[dict[str, Any], list[tuple]]]) -> None:
    """Name what holds each object the count refused, for the report's
    `detail`. ONE heap walk for all of them, after every record is done, so a
    restore with many refusals stays O(heap), not O(refusals x heap). The
    walk can only name collector-visible holders; the count already decided."""
    objs = []
    for _, paths in pending:
        try:
            objs.append(_resolve(ns, paths[0]))
        except Exception:  # noqa: BLE001
            objs.append(None)
    index = {id(o): i for i, o in enumerate(objs) if o is not None}
    allowed = {id(ns), id(objs), id(index)}
    for _, paths in pending:
        for p in paths:
            if len(p) == 2:
                allowed.add(id(ns.get(p[0])))
    targets = tuple(o for o in objs if o is not None)
    allowed.add(id(targets))
    seen: list[set[str]] = [set() for _ in objs]
    for r in gc.get_referrers(*targets):
        if id(r) in allowed or (isinstance(r, types.FrameType) and r.f_globals is _THIS_MODULE_GLOBALS):
            continue
        for x in gc.get_referents(r):
            i = index.get(id(x))
            if i is not None and objs[i] is x:
                seen[i].add(type(r).__name__)
    for i, (entry, _) in enumerate(pending):
        o = objs[i]
        names = [k for k in list(ns) if o is not None and ns[k] is o and not _covered_name(k)]
        parts = [f"uncovered names {names}"] if names else []
        if seen[i]:
            parts.append("held by " + ", ".join(sorted(seen[i])))
        if not parts:
            parts.append("held by something the collector does not track (an untracked dict or "
                         "tuple, such as an object's __dict__, a frame, or a C extension)")
        entry["detail"] = f"{entry['detail']}; " + "; ".join(parts)


def apply_view_manifest(ns: dict[str, Any], manifest: ViewManifest | dict) -> dict[str, Any]:
    """Re-establish every recorded view over its (now single) base.

    Torch leaves are repaired in place; torch non-leaf views of their base,
    and NumPy arrays, are replaced at every covered reference or not at all
    (see the module docstring). Returns a small report: how many records are
    now the recorded view, and which could not be made so, each with a
    `reason` and a `detail`. Never raises for a single bad record; the report
    is the contract, and a caller that needs the views must refuse when
    `failed` is non-empty. Idempotent: a record that is already the recorded
    view counts as restored and is not touched.
    """
    manifest = _as_manifest(manifest)
    anchor_ids: set[int] = set()
    for v in manifest.views:
        b = _base_of(ns, manifest, v.base_name)
        if b is not None:
            anchor_ids.add(id(b))
        del b
    ctx = _Apply()
    restored, failed, describe = 0, [], []
    for v in manifest.views:
        try:
            if v.kind == "torch":
                problem = _repair_torch(ns, manifest, v, anchor_ids, ctx)
            elif v.kind == "numpy":
                problem = _repair_numpy(ns, manifest, v, anchor_ids, ctx)
            else:
                problem = ("unknown record kind", v.kind)
        except Exception as exc:  # noqa: BLE001
            problem = ("repair raised", f"{type(exc).__name__}: {exc}")
        if problem is None:
            restored += 1
            continue
        entry = {"path": list(v.path), "kind": v.kind, "reason": problem[0], "detail": problem[1]}
        failed.append(entry)
        if len(problem) > 2:
            describe.append((entry, v.all_paths()))
    if describe:
        try:
            _describe_holders(ns, describe)
        except Exception:  # noqa: BLE001  (a diagnosis must never turn a report into a crash)
            pass
    return {"restored": restored, "failed": failed,
            "unsupported_at_capture": len(manifest.unsupported)}


# ---------------------------------------------------------------------------
# Non-mutating verification
# ---------------------------------------------------------------------------


def views_intact(ns: dict[str, Any], manifest: ViewManifest | dict) -> dict[str, Any]:
    """Check, WITHOUT writing, that every recorded view is still its base's view.

    Per record: every recorded path still reaches one object; the object shares
    its base's storage (torch: untyped storage pointer; NumPy: same owner and
    the recorded byte offset from it); shape, stride, offset and dtype are the
    recorded ones; and the class, requires_grad and leaf-ness (torch) and
    writeable flag (NumPy) are what the capture saw. A non-leaf autograd view
    must also still have the recorded base as its autograd base (`_base`), the
    link its gradient takes. Unlike `storage_shared`, nothing is
    written, so a version counter or a read-only array is never touched and
    the check can run on state that is about to be committed.

    A hidden base that is not bound (the source after the capture removed it
    again) is found through the manifest's `base_paths`; a synthetic anchor
    that has no path is replaced by the group itself: every record of that
    base must share ONE storage (torch) or ONE owner (NumPy, whose owner IS
    the recorded base). The result is therefore the same at the capture cut
    and after restore when nothing changed.

    Returns {"checked": n, "intact": n, "broken": [{path, kind, reason}]}.
    """
    manifest = _as_manifest(manifest)
    broken: dict[int, dict[str, Any]] = {}
    orphans: dict[str, list[tuple[int, Any]]] = {}

    def mark(i: int, v: ViewRecord, reason: str) -> None:
        broken.setdefault(i, {"path": list(v.path), "kind": v.kind, "reason": reason})

    for i, v in enumerate(manifest.views):
        try:
            obj, problem = _resolve_all(ns, v.all_paths())
            if problem:
                mark(i, v, f"{problem[0]}: {problem[1]}")
                continue
            base = _base_of(ns, manifest, v.base_name)
            # Only a SYNTHETIC anchor (no covered path ever reached it) falls
            # back to the group; a base that had a path and lost it is missing.
            if base is None and v.base_name.startswith(HIDDEN_BASE_PREFIX) \
                    and not manifest.base_paths.get(v.base_name):
                if v.kind == "numpy":
                    reason = _np_check(obj, _np_root(obj), v) if _is_numpy(obj) else f"not an ndarray ({type(obj).__name__})"
                else:
                    reason = _torch_meta(obj, v) if _is_torch(obj) else f"not a tensor ({type(obj).__name__})"
                orphans.setdefault(v.base_name, []).append((i, obj))
            elif base is None:
                reason = f"base {v.base_name!r} missing"
            elif v.kind == "torch":
                reason = _torch_check(obj, base, v)
            elif v.kind == "numpy":
                reason = _np_check(obj, base, v)
            else:
                reason = f"unknown record kind {v.kind!r}"
            if reason:
                mark(i, v, reason)
        except Exception as exc:  # noqa: BLE001
            mark(i, v, f"check raised {type(exc).__name__}: {exc}")

    for name, members in orphans.items():
        keys = set()
        for i, obj in members:
            try:
                if _is_torch(obj):
                    keys.add((obj.untyped_storage().data_ptr(), str(obj.device)))
                elif _is_numpy(obj):
                    keys.add(id(_np_root(obj)))
            except Exception:  # noqa: BLE001
                keys.add(("unreadable", i))
        if len(keys) > 1:
            for i, _ in members:
                mark(i, manifest.views[i], f"members of hidden base {name} no longer share one storage")

    checked = len(manifest.views)
    return {"checked": checked, "intact": checked - len(broken),
            "broken": [broken[i] for i in sorted(broken)]}


# ---------------------------------------------------------------------------
# Behavioural check: the E1c oracle, as a reusable function
# ---------------------------------------------------------------------------


def _write_probe_torch(w: Any, r: Any) -> bool:
    """Write a sentinel through `w`'s first element, look for it through `r`."""
    import torch
    idx = (0,) * w.dim()
    wd, rd = w.data, r.data
    old = wd[idx].clone()
    sentinel = torch.tensor(-999.0, dtype=w.dtype, device=w.device)
    try:
        wd[idx] = sentinel
        return bool((rd == sentinel).any().item())
    finally:
        wd[idx] = old


def _write_probe_numpy(w: Any, r: Any) -> bool:
    idx = (0,) * w.ndim
    old = w[idx].copy()
    try:
        w[idx] = -999
        return bool((r == -999).any())
    finally:
        w[idx] = old


def storage_shared(a: Any, b: Any) -> bool:
    """True iff a write through one object is visible through the other.

    This is the behavioural definition the E1c oracle uses. It consults neither
    ids nor data pointers. A view covers a SUBSET of its base, so the probe is
    written through the smaller side first: a sentinel written through
    `base[0,0]` is invisible to `base[1:3]` even though they share storage.
    Both directions are tried, so argument order does not matter.

    It WRITES (and restores) a sentinel, which bumps torch version counters;
    use `views_intact` on state that must not change.
    """
    if _is_torch(a) and _is_torch(b):
        if a.numel() == 0 or b.numel() == 0 or a.device != b.device:
            return False
        small, big = (a, b) if a.numel() <= b.numel() else (b, a)
        return _write_probe_torch(small, big) or _write_probe_torch(big, small)
    if _is_numpy(a) and _is_numpy(b):
        if a.size == 0 or b.size == 0:
            return False
        small, big = (a, b) if a.size <= b.size else (b, a)
        return _write_probe_numpy(small, big) or _write_probe_numpy(big, small)
    return False
