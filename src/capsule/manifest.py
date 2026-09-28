"""A portability manifest: what a switch will do to each name, BEFORE the
destructive step.

THE QUESTION. Reconstruction is all-or-nothing and the source is destroyed
before the destination is exercised. So the useful thing to know is not "did it
load" afterwards but "will it load, and will it still mean the same thing"
beforehand. This module answers that per name, with one of six verdicts:

  preserved         will reconstruct and behave as it did
  remapped          will reconstruct under a declared adaptation (cuda -> cpu)
  semantics_change  will load and NOT mean the same thing (silent unless declared)
  omitted           will be excluded from the capsule and reported as skipped
  rejected          would fail at load or first use, taking the whole namespace
                    with it; the switch must exclude it or refuse
  unknown           picklable, but nothing here establishes what it will mean
                    on the other side; an abstention, not a prediction

plus a package section: which distributions the source has IMPORTED that the
destination image lacks, and whether the bounded repair path may install them.

WHY `unknown` EXISTS. The first version answered `preserved` for any picklable
object no rule recognised. That made the manifest unsafe as a gate: an object
with a custom `__reduce__` that restores as a different type pickles cleanly,
matched no rule, was predicted preserved, and arrived as something else. A
default of `preserved` is a claim the manifest never checked. So the default is
now `unknown`, and `preserved` has to be EARNED, by a rule or by a validator.

WHAT THE PREDICTION IS BUILT FROM, in order of authority:
  1. a pickle probe (the same per-global probe the production dump runs on its
     failure path, run here proactively);
  2. a small table of construct classes whose behaviour across the boundary is
     KNOWN from E1 and E5, keyed on type, not on name;
  3. a value rule: builtin atoms and builtin containers of them round-trip by
     construction of the pickle protocol;
  4. a READ of the probe's pickle, never a load, for what loading it would do
     (`_scan_pickle`): a file handle anywhere inside the object is rebuilt by
     re-opening its path, which is a rule of its own, and anything that would
     act on a process resource keeps the object away from step 5;
  5. an in-process round-trip validator for everything else (see
     `_round_trip_verdict` for exactly what it can and cannot establish);
  6. the destination's declared capabilities (cuda, python, dill, packages).

The manifest is an estimate. E7 measures how good an estimate: it runs this
over every E1 construct and scores the prediction against the measured class.
Note the rule table in (2) was BUILT from the E1/E5 observations, so scoring it
against E1 is an in-sample measurement; E7's adversarial set is the
out-of-sample check.
"""

from __future__ import annotations

import io
import re
import sys
import types
from dataclasses import dataclass, field
from typing import Any

PRESERVED, REMAPPED, SEMANTICS_CHANGE, OMITTED, REJECTED, UNKNOWN = (
    "preserved", "remapped", "semantics_change", "omitted", "rejected", "unknown")
VERDICTS = (PRESERVED, REMAPPED, SEMANTICS_CHANGE, OMITTED, REJECTED, UNKNOWN)

#: The in-process validator deserialises a copy of the object, so it is bounded
#: by the size of the pickle the probe already produced. Anything larger is an
#: abstention rather than a memory spike on the source.
VALIDATE_MAX_BYTES = 32 * 1024 * 1024
#: The pre-load scan walks every opcode of that pickle; past this many it
#: abstains rather than spend unbounded time on the source.
SCAN_MAX_OPS = 5_000_000


@dataclass
class Destination:
    has_cuda: bool
    python_minor: str            # "3.11"
    dill_version: str            # "0.3.9"
    #: distribution name -> version, or None if unknown (then no package verdicts)
    packages: dict[str, str] | None = None
    #: distribution-name prefixes the repair path is forbidden from installing.
    #: A caller with NO repair path passes ("",), which matches every name.
    unrepairable: tuple[str, ...] = ("torch", "torchvision", "torchaudio", "nvidia-", "cuda", "triton")


@dataclass
class Adapters:
    """Which adaptations are switched on. The verdict for views and optimizers
    depends on these, which is what makes the manifest an ablation instrument."""
    device_remap: bool = True
    optimizer_reattach: bool = True
    storage_sharing: bool = False


@dataclass
class NameVerdict:
    name: str
    verdict: str
    reason: str
    type_name: str


@dataclass
class Manifest:
    names: list[NameVerdict] = field(default_factory=list)
    packages_missing: list[dict[str, Any]] = field(default_factory=list)
    #: Imported modules whose distribution has no usable name in its metadata,
    #: so no destination inventory can be checked for it. Reported, not judged.
    packages_unattributed: list[str] = field(default_factory=list)
    #: Imported distributions present on both sides at different versions.
    #: Informational: a version skew is not a verdict this module can make.
    packages_version_skew: list[dict[str, Any]] = field(default_factory=list)
    gates: dict[str, Any] = field(default_factory=dict)
    #: What the package section compared: "imported" (distributions providing
    #: a module the source has imported), "freeze" (legacy: the whole source
    #: freeze, which over-reports), or None (no package verdicts).
    package_basis: str | None = None
    imported_distributions: dict[str, Any] | None = None

    @property
    def counts(self) -> dict[str, int]:
        c = {k: 0 for k in VERDICTS}
        for n in self.names:
            c[n.verdict] += 1
        return c

    @property
    def rejected(self) -> list[NameVerdict]:
        return [n for n in self.names if n.verdict == REJECTED]

    @property
    def unknown(self) -> list[NameVerdict]:
        return [n for n in self.names if n.verdict == UNKNOWN]

    @property
    def blocking(self) -> list[str]:
        """Typed reasons the switch cannot proceed that no exclusion can fix:
        a failed compatibility gate, or an imported distribution the
        destination lacks and may not install."""
        out = [f"gate_failed:{g}" for g, ok in self.gates.items() if not ok]
        out += [f"missing_distribution:{p['distribution']}" for p in self.packages_missing if p["unrepairable"]]
        return out

    @property
    def admissible(self) -> bool:
        """A switch is admissible only if nothing would take the namespace down
        and every compatibility gate is predicted to pass."""
        return not self.rejected and all(self.gates.values()) and not any(
            p["unrepairable"] for p in self.packages_missing)

    @property
    def fully_validated(self) -> bool:
        """Admissible AND every name has a determined verdict. A manifest with
        abstentions is admissible (nothing is known to fail) but it is not a
        complete prediction, and a caller that wants a safety gate rather than
        a report should key on this."""
        return self.admissible and not self.unknown

    def without(self, excluded: set[str] | list[str]) -> "Manifest":
        """The manifest of the capsule that will actually be written once
        `excluded` names are left out of it."""
        ex = set(excluded)
        return Manifest(names=[n for n in self.names if n.name not in ex],
                        packages_missing=list(self.packages_missing),
                        packages_unattributed=list(self.packages_unattributed),
                        packages_version_skew=list(self.packages_version_skew),
                        gates=dict(self.gates), package_basis=self.package_basis,
                        imported_distributions=self.imported_distributions)

    def to_json(self) -> dict[str, Any]:
        return {
            "names": [n.__dict__ for n in self.names],
            "counts": self.counts,
            "unknown": [n.name for n in self.unknown],
            "packages_missing": self.packages_missing,
            "packages_unattributed": self.packages_unattributed,
            "packages_version_skew": self.packages_version_skew,
            "package_basis": self.package_basis,
            "imported_distributions": self.imported_distributions,
            "gates": self.gates,
            "blocking": self.blocking,
            "admissible": self.admissible,
            "fully_validated": self.fully_validated,
        }


# ---------------------------------------------------------------------------
# Known construct classes (from E1/E5), keyed on TYPE
# ---------------------------------------------------------------------------

def _is_file_like_open(obj: Any) -> bool:
    """Open file handles pickle and then poison the load. `NamedTemporaryFile`
    returns a wrapper that is NOT an io.IOBase, so detect by protocol."""
    if isinstance(obj, io.IOBase):
        return not obj.closed
    inner = getattr(obj, "file", None)
    if isinstance(inner, io.IOBase):
        return not inner.closed
    return callable(getattr(obj, "fileno", None)) and callable(getattr(obj, "read", None)) \
        and not bool(getattr(obj, "closed", False))


def _classify_known(obj: Any, adapters: Adapters, dest: Destination,
                    shares_storage: bool) -> tuple[str, str] | None:
    """Return (verdict, reason) for types whose boundary behaviour is known.

    `shares_storage` is True when another namespace name aliases this object's
    buffer. A reshape of a temporary has a base but shares with nothing bound,
    so nothing observable is lost; the verdict must key on OBSERVABLE aliasing.
    """
    import threading
    import subprocess

    # --- semantics differ silently, measured in E1 ---------------------------
    if isinstance(obj, threading.Thread):
        return SEMANTICS_CHANGE, "thread: loads, reports alive in a process that never started it"
    if isinstance(obj, subprocess.Popen):
        return SEMANTICS_CHANGE, "subprocess handle: loads with a pid that is not a child here"
    if isinstance(obj, types.ModuleType):
        return PRESERVED, "module: re-imported by reference on the destination"
    lock_types = tuple(t for t in (type(threading.Lock()), type(threading.RLock())) if isinstance(t, type))
    if isinstance(obj, lock_types):
        return SEMANTICS_CHANGE, "lock: loads unlocked regardless of source state"
    import logging
    if isinstance(obj, logging.Logger):
        return SEMANTICS_CHANGE, "logger: handlers/level are process configuration, not object state"

    # --- torch ----------------------------------------------------------------
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None:
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda and not dest.has_cuda:
                if adapters.device_remap:
                    return REMAPPED, "cuda tensor -> cpu (destination has no cuda)"
                return REJECTED, "cuda tensor on a cpu destination without device remap: load raises"
            if shares_storage:
                if adapters.storage_sharing:
                    return PRESERVED, "tensor sharing storage with another name: re-derived by adapter"
                return SEMANTICS_CHANGE, "tensor sharing storage with another name: sharing lost"
            return PRESERVED, "tensor"
        if isinstance(obj, torch.optim.Optimizer):
            if adapters.optimizer_reattach:
                return PRESERVED, "optimizer: class identity kept by submodule re-attachment"
            return SEMANTICS_CHANGE, "optimizer: class serialized by value; isinstance(Optimizer) fails"
        if isinstance(obj, torch.Generator):
            if obj.device.type == "cuda" and not dest.has_cuda:
                return OMITTED, "cuda RNG: dropped on a cpu destination (declared rule)"
            return SEMANTICS_CHANGE, "torch RNG: stream position does not continue (measured, E5)"

    # --- numpy ----------------------------------------------------------------
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            if shares_storage:
                if adapters.storage_sharing and isinstance(obj.base, (np.ndarray, type(None))):
                    return PRESERVED, "ndarray sharing storage with another name: re-derived by adapter"
                return SEMANTICS_CHANGE, "ndarray sharing storage with another name: sharing lost"
            return PRESERVED, "ndarray"
        if isinstance(obj, np.random.Generator):
            return PRESERVED, "numpy Generator: bit generator state round-trips"
    except Exception:
        pass

    import random
    if isinstance(obj, random.Random):
        return PRESERVED, "python Random: state round-trips"
    return None


def _storage_groups(ns: dict[str, Any]) -> set[str]:
    """Names whose buffer is aliased by at least one other top-level name."""
    shared: set[str] = set()
    try:
        import torch
        ptr: dict[tuple, list[str]] = {}
        for k, v in ns.items():
            if isinstance(v, torch.Tensor) and v.numel() > 0:
                try:
                    ptr.setdefault((v.untyped_storage().data_ptr(), str(v.device)), []).append(k)
                except Exception:
                    pass
        for names in ptr.values():
            if len(names) > 1:
                shared.update(names)
    except Exception:
        pass
    try:
        import numpy as np
        arrs = [(k, v) for k, v in ns.items() if isinstance(v, np.ndarray) and v.size > 0]
        for i, (k1, a) in enumerate(arrs):
            for k2, b in arrs[i + 1:]:
                if np.shares_memory(a, b):
                    shared.update((k1, k2))
    except Exception:
        pass
    return shared


def _pickle_probe(name: str, obj: Any) -> tuple[bool, str, bytes | None]:
    """The production dump's per-global probe, run proactively. The bytes are
    returned so the round-trip validator does not pickle the object twice."""
    import dill
    try:
        return True, "", dill.dumps(obj, recurse=True)
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:80]}", None


# ---------------------------------------------------------------------------
# The value rule: builtin values round-trip by construction
# ---------------------------------------------------------------------------

_ATOMS = (type(None), bool, int, float, complex, str, bytes, bytearray)
_CONTAINERS = (list, tuple, dict, set, frozenset)
_VALUE_MAX_NODES = 200_000
_VALUE_MAX_DEPTH = 64


def _is_builtin_value(obj: Any) -> bool:
    """True when `obj` is a builtin atom, or a builtin container (EXACT types,
    not subclasses) whose every member recursively is one.

    Why this is enough to call it preserved: pickle reconstructs these types
    from their contents with no user code in the path, and the memo keeps
    shared and cyclic references. Subclasses are excluded on purpose, because a
    subclass can carry its own `__reduce__` and that is exactly the case the
    rule must not vouch for. Bounded in node count and depth so a huge or
    pathologically deep structure falls through to the validator instead of
    exhausting the stack; an object seen twice (sharing or a cycle) is checked
    once."""
    seen: set[int] = set()
    budget = [_VALUE_MAX_NODES]

    def walk(o: Any, depth: int) -> bool:
        budget[0] -= 1
        if budget[0] < 0:
            return False
        t = type(o)
        if t in _ATOMS:
            return True
        if t not in _CONTAINERS or depth > _VALUE_MAX_DEPTH:
            return False
        if id(o) in seen:
            return True
        seen.add(id(o))
        if t is dict:
            return all(walk(k, depth + 1) and walk(v, depth + 1) for k, v in o.items())
        return all(walk(x, depth + 1) for x in o)

    return walk(obj, 0)


# ---------------------------------------------------------------------------
# The round-trip validator
# ---------------------------------------------------------------------------

_HEAPTYPE = 1 << 9          # Py_TPFLAGS_HEAPTYPE: set on classes defined in Python
_MISSING = object()


def _tname(t: type) -> str:
    return f"{getattr(t, '__module__', '?')}.{getattr(t, '__qualname__', getattr(t, '__name__', '?'))}"


def _has_own_eq(t: type) -> bool:
    return any("__eq__" in vars(c) for c in t.__mro__ if c is not object)


def _c_layout_base(t: type) -> type | None:
    """The first class in the MRO, other than `object`, implemented in C. Such
    a base keeps state in C fields that `vars()` never shows, so comparing
    `vars()` alone would call two different objects equal."""
    for c in t.__mro__:
        if c is object:
            continue
        if not (getattr(c, "__flags__", 0) & _HEAPTYPE):
            return c
    return None


def _slot_names(t: type) -> list[str]:
    out: list[str] = []
    for c in t.__mro__:
        s = vars(c).get("__slots__", ())
        for n in ([s] if isinstance(s, str) else list(s)):
            if n not in ("__dict__", "__weakref__") and n not in out:
                out.append(n)
    return out


#: Types whose identity carries no meaning: immutable values that pickle may
#: legitimately split (ints and floats are never memoized) or merge (interned
#: strings), and objects Python itself hands out fresh or cached (bound
#: methods, weak references). The aliasing check skips them; everything else
#: is a node whose sharing the round trip must preserve.
_NO_IDENTITY = (type(None), bool, int, float, complex, str, bytes, tuple, frozenset, range, slice,
                type(Ellipsis), type(NotImplemented), types.CodeType, types.MethodType,
                types.BuiltinFunctionType, types.MethodWrapperType, types.ModuleType)


def _identity_free(o: Any) -> bool:
    import weakref
    if type(o) in _NO_IDENTITY or isinstance(o, weakref.ref):
        return True
    np = sys.modules.get("numpy")
    return np is not None and isinstance(o, (np.generic, np.dtype))


_MAX_OVERLAP_PAIRS = 50_000


def _mem_extent(x: Any) -> tuple[str, int, int] | None:
    """(device, first byte, one past the last byte) of the memory an ndarray
    or strided tensor addresses; None when it addresses none. Raises for a
    layout it does not model (sparse, quantized), which the caller turns into
    an abstention."""
    np = sys.modules.get("numpy")
    if np is not None and isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        bb = getattr(np, "byte_bounds", None)
        if bb is None:                         # numpy 2 moved it
            from numpy.lib.array_utils import byte_bounds as bb
        lo, hi = bb(x)
        return ("cpu", int(lo), int(hi))
    torch = sys.modules["torch"]
    if x.layout != torch.strided:
        raise TypeError(f"{x.layout} tensor")
    if x.numel() == 0 or x.device.type == "meta":
        return None
    lo = int(x.data_ptr())
    ext = sum((s - 1) * st for s, st in zip(x.shape, x.stride()))   # torch strides are never negative
    return (str(x.device), lo, lo + (ext + 1) * x.element_size())


def _overlapping_pairs(ext: list[tuple[str, int, int] | None]) -> set[tuple[int, int]] | None:
    """Index pairs (i < j) whose extents overlap, by a sweep over start
    addresses; None past `_MAX_OVERLAP_PAIRS`."""
    order = sorted((e[0], e[1], e[2], i) for i, e in enumerate(ext) if e is not None)
    pairs: set[tuple[int, int]] = set()
    active: list[tuple[str, int, int]] = []          # (device, end, index)
    for dev, lo, hi, i in order:
        active = [a for a in active if a[0] == dev and a[1] > lo]
        for _d, _h, j in active:
            pairs.add((min(i, j), max(i, j)))
        if len(pairs) > _MAX_OVERLAP_PAIRS:
            return None
        active.append((dev, hi, i))
    return pairs


def _shares(x: Any, y: Any, extents_overlap: bool, np: Any) -> bool | None:
    if not extents_overlap:
        return False
    if np is not None and isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        try:
            return bool(np.shares_memory(x, y, max_work=100_000))
        except Exception:  # noqa: BLE001  (TooHardError: the exact answer is too costly)
            return None
    return True


class _Compare:
    """Structural equality between an object and its round-trip copy.

    Returns True (equal), False (differs) or None (the validator cannot decide),
    and records where the first difference or undecidable node was, so the
    manifest's reason says what changed rather than only that something did.
    Only rules that are SUPPORTED are applied: builtin values, containers,
    numpy and torch arrays and generators, functions (by code, defaults and
    closure), classes (by their bodies), and plain Python instances (by their
    attributes). Anything else is undecidable, never assumed equal.

    EQUAL VALUES ARE NOT ENOUGH: SHARING IS PART OF THE STATE. Two fields that
    are one list on the source and two equal lists after the round trip
    compare equal node by node, and the object is still broken: an append
    through one no longer shows through the other. So the walk keeps a
    BIJECTION between source and restored nodes. A source node paired with a
    second restored node, or two source nodes paired with one restored node,
    is a difference ("aliasing differs"). Memory is the same question one
    level down. A tensor and a view of it are two Python objects over one
    buffer, and a pickle that copies each gives two buffers. So every array the
    walk reaches is recorded, and `finish()` requires the same overlap
    relation between their memory on both sides.
    """

    def __init__(self, dest: "Destination | None", max_nodes: int = 200_000, max_depth: int = 40):
        self.dest, self.nodes, self.max_depth = dest, max_nodes, max_depth
        self.seen: set[tuple[int, int]] = set()
        self.note = ""
        self.top_method = ""
        #: id(source node) -> id(restored node), its inverse, and the path each
        #: source node was first reached by (for the reason text).
        self.fwd: dict[int, int] = {}
        self.rev: dict[int, int] = {}
        self.where: dict[int, str] = {}
        #: Every paired node is kept alive until the walk ends, so an id is
        #: never reused by a temporary (a generator's `.state` dict is built
        #: fresh on each access) and mistaken for a node already seen.
        self.keep: list[Any] = []
        #: Arrays reached by the walk: (source, restored, path), one entry per
        #: source object, for the memory-overlap check in `finish()`.
        self.arrays: list[tuple[Any, Any, str]] = []
        self._array_ids: set[int] = set()

    def _fail(self, path: str, msg: str) -> bool:
        self.note = self.note or f"{path}: {msg}"
        return False

    def _undecided(self, path: str, msg: str) -> None:
        self.note = self.note or f"{path}: {msg}"
        return None

    def _all(self, pairs, path: str, depth: int) -> bool | None:
        """Conjunction over (a, b, subpath): False wins over None, None over True."""
        undecided = False
        for a, b, sub in pairs:
            r = self.eq(a, b, sub, depth + 1)
            if r is False:
                return False
            if r is None:
                undecided = True
        return None if undecided else True

    def eq(self, a: Any, b: Any, path: str, depth: int = 0) -> bool | None:
        if a is b:
            return True
        self.nodes -= 1
        if self.nodes < 0:
            return self._undecided(path, "comparison budget exhausted")
        if depth > self.max_depth:
            return self._undecided(path, "nested deeper than the validator follows")
        ta, tb = type(a), type(b)
        if _tname(ta) != _tname(tb):
            return self._fail(path, f"{_tname(ta)} restores as {_tname(tb)}")
        if ta in _ATOMS:
            if ta is float and a != a and b != b:
                return True
            return True if a == b else self._fail(path, f"value {a!r:.60} restores as {b!r:.60}")
        if not _identity_free(a):
            ia, ib = id(a), id(b)
            pa, pb = self.fwd.get(ia), self.rev.get(ib)
            if pa is not None and pa != ib:
                return self._fail(path, f"aliasing differs: one object on the source (also at {self.where[ia]}) "
                                        "restores as two separate objects")
            if pb is not None and pb != ia:
                return self._fail(path, f"aliasing differs: two separate objects on the source restore as one "
                                        f"(also at {self.where[pb]})")
            if pa is None:
                self.fwd[ia], self.rev[ib], self.where[ia] = ib, ia, path
                self.keep += (a, b)
        key = (id(a), id(b))
        if key in self.seen:           # a cycle or a shared node: already being compared
            return True
        self.seen.add(key)
        if ta is not tb:
            # Same name, different class object: the class itself was copied BY
            # VALUE, so its body and bases are part of what the round trip
            # changed and are compared too. This is not hypothetical: dill
            # 0.4.1 rebuilds a by-value subclass of a generic library class
            # (a user `torch.utils.data.Dataset`) from the INHERITED
            # `__orig_bases__`, so it comes back as a subclass of
            # `typing.Generic` only and `isinstance(ds, Dataset)` is False on
            # the destination. Comparing instance attributes alone misses it.
            r = self.eq(ta, tb, f"{path}.__class__", depth + 1)
            if r is not True:
                return r
        r = self._known(a, b, path, depth)
        if r is not _MISSING:
            return r
        if _has_own_eq(ta):
            try:
                res = a == b
                res = None if res is NotImplemented else bool(res)
            except Exception as exc:  # noqa: BLE001
                return self._undecided(path, f"the type's own __eq__ raised {type(exc).__name__}")
            if res is True:
                if depth == 0:
                    self.top_method = "own_eq"
                return True
            if res is False and ta is tb:
                return self._fail(path, "unequal under the type's own __eq__")
            # The class was copied BY VALUE (it is a different class object
            # with the same name), and many __eq__ implementations, dataclasses
            # among them, return NotImplemented or False for an instance of a
            # different class object. That says nothing about the state, so the
            # state is compared instead.
        return self._state(a, b, path, depth)

    # -- supported rules -----------------------------------------------------
    def _known(self, a: Any, b: Any, path: str, depth: int):  # noqa: C901
        if isinstance(a, dict):
            if len(a) != len(b):
                return self._fail(path, f"{len(a)} entries restore as {len(b)}")
            # Pickle rebuilds a dict in insertion order, so an ordered pairing
            # is exact and also covers keys that are not atoms (tensors key an
            # optimizer's state).
            pairs = []
            for i, ((ka, va), (kb, vb)) in enumerate(zip(a.items(), b.items())):
                pairs.append((ka, kb, f"{path}<key {i}>"))
                pairs.append((va, vb, f"{path}[{ka!r:.30}]"))
            if hasattr(a, "default_factory"):
                pairs.append((a.default_factory, b.default_factory, f"{path}.default_factory"))
            r = self._all(pairs, path, depth)
            return r if r is not True or not hasattr(a, "__dict__") else self._state(a, b, path, depth, content_done=True)
        if isinstance(a, (list, tuple)):
            if len(a) != len(b):
                return self._fail(path, f"length {len(a)} restores as {len(b)}")
            r = self._all(((x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b))), path, depth)
            return r if r is not True or not hasattr(a, "__dict__") else self._state(a, b, path, depth, content_done=True)
        if isinstance(a, (set, frozenset)):
            if all(type(x) in _ATOMS for x in a) and all(type(x) in _ATOMS for x in b):
                return True if set(a) == set(b) else self._fail(path, "set members differ")
            return self._undecided(path, "set of non-atomic members")
        for atom in _ATOMS:
            if isinstance(a, atom):
                # A strict subclass of a builtin atom (exact atoms were handled
                # in `eq`): compare the builtin content with the BASE type's
                # equality, since the subclass may override __eq__ or __str__,
                # then any instance attributes it adds.
                try:
                    same = atom.__eq__(a, b)
                except Exception:  # noqa: BLE001
                    same = NotImplemented
                if same is NotImplemented:
                    return self._undecided(path, f"subclass of {atom.__name__} not comparable")
                if not same and not (atom is float and a != a and b != b):
                    return self._fail(path, "value differs")
                return self._state(a, b, path, depth, content_done=True) if hasattr(a, "__dict__") else True
        if isinstance(a, types.ModuleType):
            return True if a.__name__ == b.__name__ else self._fail(path, "a different module")
        if isinstance(a, types.CodeType):
            return True if a == b else self._fail(path, "code object differs")
        if isinstance(a, type):
            return self._class(a, b, path, depth)
        if isinstance(a, types.FunctionType):
            pairs = [(a.__code__, b.__code__, f"{path}.__code__"),
                     (a.__defaults__, b.__defaults__, f"{path}.__defaults__"),
                     (a.__kwdefaults__, b.__kwdefaults__, f"{path}.__kwdefaults__"),
                     (a.__qualname__, b.__qualname__, f"{path}.__qualname__"),
                     (a.__dict__, b.__dict__, f"{path}.__dict__")]
            ca, cb = a.__closure__ or (), b.__closure__ or ()
            if len(ca) != len(cb):
                return self._fail(path, "closure size differs")
            for i, (x, y) in enumerate(zip(ca, cb)):
                try:
                    vx = x.cell_contents
                except ValueError:
                    vx = _MISSING
                try:
                    vy = y.cell_contents
                except ValueError:
                    vy = _MISSING
                pairs.append((vx, vy, f"{path}.<closure {a.__code__.co_freevars[i] if i < len(a.__code__.co_freevars) else i}>"))
            # A function's globals are not compared: they are the session's own
            # names and are validated as names in their own right.
            return self._all(pairs, path, depth)
        if isinstance(a, types.MethodType):
            return self._all([(a.__func__, b.__func__, f"{path}.__func__"),
                              (a.__self__, b.__self__, f"{path}.__self__")], path, depth)
        if isinstance(a, (types.BuiltinFunctionType, types.MethodWrapperType)):
            if getattr(a, "__qualname__", None) != getattr(b, "__qualname__", None):
                return self._fail(path, "a different builtin")
            sa, sb = getattr(a, "__self__", None), getattr(b, "__self__", None)
            return self.eq(sa, sb, f"{path}.__self__", depth + 1)
        import functools
        if isinstance(a, functools.partial):
            return self._all([(a.func, b.func, f"{path}.func"), (a.args, b.args, f"{path}.args"),
                              (a.keywords, b.keywords, f"{path}.keywords"),
                              (a.__dict__, b.__dict__, f"{path}.__dict__")], path, depth)
        import weakref
        if isinstance(a, weakref.ref):
            return self.eq(a(), b(), f"{path}()", depth + 1)
        import random
        if isinstance(a, random.Random):
            return True if a.getstate() == b.getstate() else self._fail(path, "Random state differs")
        np = sys.modules.get("numpy")
        if np is not None:
            if isinstance(a, np.ndarray):
                self._note_array(a, b, path)
                if a.shape != b.shape or a.dtype != b.dtype:
                    return self._fail(path, f"array {a.dtype}{a.shape} restores as {b.dtype}{b.shape}")
                if a.dtype.kind == "O":
                    return self._all(((x, y, f"{path}.flat[{i}]") for i, (x, y) in enumerate(zip(a.flat, b.flat))), path, depth)
                try:
                    same = bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
                except Exception as exc:  # noqa: BLE001
                    return self._undecided(path, f"array comparison raised {type(exc).__name__}")
                return True if same else self._fail(path, "array values differ")
            if isinstance(a, np.generic):
                if a.dtype != b.dtype:
                    return self._fail(path, "numpy scalar dtype differs")
                return True if (a == b or (a != a and b != b)) else self._fail(path, "numpy scalar differs")
            if isinstance(a, np.random.Generator):
                return self.eq(a.bit_generator.state, b.bit_generator.state, f"{path}.bit_generator.state", depth + 1)
            if isinstance(a, np.random.RandomState):
                return self.eq(a.get_state(legacy=False), b.get_state(legacy=False), f"{path}.state", depth + 1)
            if isinstance(a, np.random.BitGenerator):
                return self.eq(a.state, b.state, f"{path}.state", depth + 1)
            if isinstance(a, np.dtype):
                return True if a == b else self._fail(path, "dtype differs")
        torch = sys.modules.get("torch")
        if torch is not None:
            if isinstance(a, torch.Tensor):
                self._note_array(a, b, path)
                return self._tensor(a, b, path, depth, torch)
            if isinstance(a, torch.Generator):
                if a.device != b.device:
                    return self._fail(path, f"generator on {a.device} restores on {b.device}")
                return True if torch.equal(a.get_state(), b.get_state()) else self._fail(path, "generator state differs")
        return _MISSING

    def _tensor(self, a: Any, b: Any, path: str, depth: int, torch: Any) -> bool | None:
        if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
            return self._fail(path, f"tensor {a.dtype}{tuple(a.shape)} restores as {b.dtype}{tuple(b.shape)}")
        if a.device.type == "cuda" and self.dest is not None and not self.dest.has_cuda:
            return self._undecided(path, "nested cuda tensor on a destination without cuda")
        if a.device != b.device:
            return self._fail(path, f"tensor on {a.device} restores on {b.device}")
        if a.requires_grad != b.requires_grad:
            return self._fail(path, "requires_grad differs")
        try:
            x, y = a.detach(), b.detach()
            if x.is_floating_point() or x.is_complex():
                nx, ny = torch.isnan(x), torch.isnan(y)
                same = torch.equal(nx, ny) and torch.equal(x[~nx], y[~ny])
            else:
                same = torch.equal(x, y)
        except Exception as exc:  # noqa: BLE001
            return self._undecided(path, f"tensor comparison raised {type(exc).__name__}")
        if not same:
            return self._fail(path, "tensor values differ")
        da, db = getattr(a, "__dict__", {}) or {}, getattr(b, "__dict__", {}) or {}
        return self.eq(da, db, f"{path}.__dict__", depth + 1) if (da or db) else True

    # -- memory sharing between arrays ---------------------------------------
    def _note_array(self, a: Any, b: Any, path: str) -> None:
        if id(a) not in self._array_ids:
            self._array_ids.add(id(a))
            self.arrays.append((a, b, path))
            self.keep += (a, b)

    def finish(self) -> bool | None:
        """After an equal walk: do the arrays it reached overlap in memory on
        the restored side exactly where they overlapped on the source?

        Candidate pairs come from byte extents (a sweep, not all pairs). Two
        numpy arrays whose extents overlap are then asked exactly
        (`np.shares_memory`, bounded work), because interleaved views such as
        `a[::2]` and `a[1::2]` have overlapping extents and share no element.
        Any pair involving a tensor keeps the extent answer, which can only
        err towards "shared", so a miss it causes is conservative."""
        if len(self.arrays) < 2:
            return True
        try:
            src = [_mem_extent(a) for a, _b, _p in self.arrays]
            dst = [_mem_extent(b) for _a, b, _p in self.arrays]
        except Exception as exc:  # noqa: BLE001
            return self._undecided(self.arrays[0][2], f"array memory layout not understood ({type(exc).__name__})")
        ps, pd = _overlapping_pairs(src), _overlapping_pairs(dst)
        if ps is None or pd is None:
            return self._undecided(self.arrays[0][2], "too many overlapping arrays to compare their sharing")
        np = sys.modules.get("numpy")
        undecided = None
        for i, j in sorted(ps | pd):
            s = _shares(self.arrays[i][0], self.arrays[j][0], (i, j) in ps, np)
            d = _shares(self.arrays[i][1], self.arrays[j][1], (i, j) in pd, np)
            if s is None or d is None:
                undecided = undecided or (i, j)
                continue
            if s != d:
                what = "is no longer shared" if s else "becomes shared"
                return self._fail(self.arrays[j][2], f"aliasing differs: memory shared with {self.arrays[i][2]} "
                                                    f"on the source {what} after the round trip")
        if undecided is not None:
            return self._undecided(self.arrays[undecided[1]][2], "memory sharing with "
                                   f"{self.arrays[undecided[0]][2]} too costly to decide exactly")
        return True

    def _class(self, a: type, b: type, path: str, depth: int) -> bool | None:
        """A class copied by value: compare bases by name and the class body."""
        if _tname(a) != _tname(b):
            return self._fail(path, f"class {_tname(a)} restores as {_tname(b)}")
        ba, bb = [_tname(x) for x in a.__bases__], [_tname(x) for x in b.__bases__]
        if ba != bb:
            return self._fail(path, f"class bases {ba} restore as {bb}")
        # Derived or interpreter-managed entries: the ABC cache is rebuilt by the
        # metaclass, and __dict__/__weakref__ are slot descriptors of the class.
        skip = {"__dict__", "__weakref__", "__module__", "_abc_impl", "__parameters__"}
        va, vb = dict(vars(a)), dict(vars(b))
        ka, kb = set(va) - skip, set(vb) - skip
        if ka != kb:
            return self._fail(path, f"class body differs: {sorted(ka ^ kb)[:3]}")
        pairs = []
        for k in sorted(ka):
            x, y = va[k], vb[k]
            if isinstance(x, (staticmethod, classmethod)):
                x, y = x.__func__, getattr(y, "__func__", y)
            elif isinstance(x, property):
                pairs += [(x.fget, getattr(y, "fget", y), f"{path}.{k}.fget"),
                          (x.fset, getattr(y, "fset", y), f"{path}.{k}.fset"),
                          (x.fdel, getattr(y, "fdel", y), f"{path}.{k}.fdel")]
                continue
            elif type(x).__name__ in ("member_descriptor", "getset_descriptor"):
                if type(x) is not type(y) or x.__name__ != y.__name__:
                    return self._fail(f"{path}.{k}", "slot descriptor differs")
                continue
            pairs.append((x, y, f"{path}.{k}"))
        return self._all(pairs, path, depth)

    def _state(self, a: Any, b: Any, path: str, depth: int, content_done: bool = False) -> bool | None:
        """Instance state: attributes and slots. Only sound when no C base
        holds state the attributes do not show."""
        base = _c_layout_base(type(a))
        if base is not None and not content_done:
            return self._undecided(path, f"instance of a C type ({_tname(base)}) whose state vars() cannot see")
        da, db = getattr(a, "__dict__", _MISSING), getattr(b, "__dict__", _MISSING)
        pairs = []
        if (da is _MISSING) != (db is _MISSING):
            return self._fail(path, "instance __dict__ present on one side only")
        if da is not _MISSING:
            if not isinstance(da, dict) or not isinstance(db, dict):
                return self._undecided(path, "non-dict instance namespace")
            if set(da) != set(db):
                gone, new = sorted(set(da) - set(db)), sorted(set(db) - set(da))
                return self._fail(path, f"attributes lost {gone[:3]} gained {new[:3]}")
            pairs += [(da[k], db[k], f"{path}.{k}") for k in da]
        for s in _slot_names(type(a)):
            pairs.append((getattr(a, s, _MISSING), getattr(b, s, _MISSING), f"{path}.{s}"))
        return self._all(pairs, path, depth) if pairs else True


# ---------------------------------------------------------------------------
# What LOADING a pickle would do, established without loading it
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. The round-trip validator below runs `dill.loads` in the
# SOURCE process, which is the user's live runtime. Loading is not a pure
# function of the bytes: dill rebuilds a file handle by RE-OPENING ITS PATH
# with the mode it was opened with (`dill._dill._create_filehandle`), so a
# handle opened 'w' or 'w+' truncates the user's file, and it does that even
# for a handle that was already CLOSED (the reopen happens before the close).
# `with open("data/out.csv", "w") as f:` at the top of a cell leaves exactly
# such an `f` bound in `__main__`. The first validator loaded these, and an
# aborted switch, which promises to leave the source untouched, emptied the
# user's file. So a pickle is first READ, opcode by opcode, with a small
# stack machine that resolves every global it names and the arguments of
# every call it makes, and is loaded only when nothing it would call creates
# or touches a process resource.

_FILEHANDLE = ("dill._dill", "_create_filehandle")
_STD_STREAMS = ("<stdin>", "<stdout>", "<stderr>")

#: Modules whose objects ARE process resources or act on the filesystem,
#: processes, signals or the network. A pickle that references anything in
#: them is not loaded in the source, whatever it would do with it. Matched as
#: the module or a submodule.
_RESOURCE_MODULES = (
    "os", "posix", "nt", "subprocess", "_posixsubprocess", "socket", "_socket", "ssl", "_ssl",
    "mmap", "shutil", "tempfile", "multiprocessing", "_multiprocessing", "concurrent.futures",
    "signal", "_signal", "ctypes", "_ctypes", "atexit", "pty", "fcntl", "termios", "select",
    "selectors", "asyncio", "socketserver", "http.client", "http.server", "urllib.request",
    "ftplib", "smtplib", "webbrowser", "sqlite3", "_sqlite3", "_thread",
)
#: Callables that are harmless to REFERENCE and harmful to CALL at load time.
#: Kept separate because references are everywhere: dill ships `io.open` in
#: the globals of any by-value function whose body calls `open`, and blocking
#: on that would abstain on every session class with a `save` method.
_RESOURCE_CALLS = frozenset({
    ("builtins", "open"), ("io", "open"), ("_io", "open"), ("_pyio", "open"), ("io", "open_code"),
    ("_io", "open_code"), ("io", "FileIO"), ("_io", "FileIO"), ("_pyio", "FileIO"), ("codecs", "open"),
    ("builtins", "exec"), ("builtins", "eval"), ("builtins", "compile"), ("builtins", "__import__"),
    ("builtins", "breakpoint"), ("builtins", "input"), ("builtins", "exit"), ("builtins", "quit"),
    ("sys", "exit"), ("sys", "settrace"), ("sys", "setprofile"), ("sys", "setrecursionlimit"),
    ("sys", "setswitchinterval"), ("sys", "addaudithook"), ("importlib", "import_module"),
    ("importlib", "reload"), ("numpy", "load"), ("numpy", "save"), ("numpy", "savez"),
    ("numpy", "savez_compressed"), ("numpy", "savetxt"), ("numpy", "loadtxt"), ("numpy", "fromfile"),
    ("numpy", "memmap"), ("torch", "load"), ("torch", "save"), ("torch", "from_file"),
    ("logging", "basicConfig"), ("logging", "disable"), ("logging", "setLoggerClass"),
    ("logging", "captureWarnings"), ("gc", "disable"), ("gc", "collect"), ("gc", "freeze"),
})


class _G:
    """A global the pickle resolves by name."""
    __slots__ = ("mod", "name")

    def __init__(self, mod: str, name: str):
        self.mod, self.name = mod, name


class _PickleScan:
    def __init__(self) -> None:
        self.refs: set[tuple[str, str]] = set()
        #: (global, arguments) for every REDUCE / NEWOBJ / OBJ / INST whose
        #: callable is a global; arguments as far as the stack machine models
        #: them (strings, numbers, booleans, None, tuples; anything else opaque).
        self.calls: list[tuple[_G, Any]] = []
        #: A class rebuilt by value carries '__del__' in its body.
        self.has_del = False
        self.error: str | None = None
        #: Too large to scan; only a substring test was made.
        self.oversize = False
        self.oversize_filehandle = False


_SCAN_STRINGS = frozenset({"STRING", "BINSTRING", "SHORT_BINSTRING", "UNICODE", "BINUNICODE",
                           "SHORT_BINUNICODE", "BINUNICODE8"})
_SCAN_NUMBERS = frozenset({"INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4", "FLOAT", "BINFLOAT"})
_OPAQUE = object()


def _scan_pickle(payload: bytes, max_bytes: int = VALIDATE_MAX_BYTES, max_ops: int = SCAN_MAX_OPS) -> _PickleScan:
    """Walk `payload` with `pickletools.genops` and model the unpickler's
    stack using each opcode's declared stack effect, WITHOUT executing
    anything. Any inconsistency (an opcode this model does not follow, a
    memo miss, a global named by a non-string) is recorded as `error`, and
    the caller then refuses to load: an unscannable pickle is never trusted."""
    import pickletools
    sc = _PickleScan()
    if len(payload) > max_bytes:
        # A full scan copies every argument; past the validation limit only
        # the question that matters most is asked, by substring.
        sc.oversize = True
        sc.oversize_filehandle = _FILEHANDLE[1].encode() in payload
        return sc
    mark = pickletools.markobject
    stack: list[Any] = []
    marks: list[int] = []
    memo: dict[int, Any] = {}
    try:
        for n_ops, (op, arg, _pos) in enumerate(pickletools.genops(payload)):
            if n_ops > max_ops:
                sc.error = f"more than {max_ops:,} opcodes"
                return sc
            name = op.name
            if name in _SCAN_STRINGS:
                if arg == "__del__":
                    sc.has_del = True
                stack.append(arg)
                continue
            if name in _SCAN_NUMBERS:
                stack.append(arg)
                continue
            if name in ("NONE", "NEWTRUE", "NEWFALSE"):
                stack.append({"NONE": None, "NEWTRUE": True, "NEWFALSE": False}[name])
                continue
            if name == "MARK":
                marks.append(len(stack))
                continue
            if name == "MEMOIZE":
                memo[len(memo)] = stack[-1]
                continue
            if name in ("PUT", "BINPUT", "LONG_BINPUT"):
                memo[arg] = stack[-1]
                continue
            if name in ("GET", "BINGET", "LONG_BINGET"):
                stack.append(memo[arg])
                continue
            if name == "DUP":
                stack.append(stack[-1])
                continue
            g = None
            if name in ("GLOBAL", "INST"):
                mod, _, qn = arg.partition(" ")
                g = _G(mod, qn)
                sc.refs.add((mod, qn))
                if name == "GLOBAL":
                    stack.append(g)
                    continue
            before = op.stack_before
            if mark in before:
                if not marks:
                    raise ValueError(f"{name} without a MARK")
                k = marks.pop()
                sl = stack[k:]
                del stack[k:]
                pre = before.index(mark)
                if pre:
                    if len(stack) < pre:
                        raise ValueError(f"stack underflow at {name}")
                    del stack[len(stack) - pre:]
                if name == "TUPLE":
                    stack.append(tuple(sl))
                    continue
                if name == "OBJ" and sl and isinstance(sl[0], _G):
                    sc.calls.append((sl[0], tuple(sl[1:])))
                elif name == "INST":
                    sc.calls.append((g, tuple(sl)))
            else:
                n = len(before)
                if len(stack) < n:
                    raise ValueError(f"stack underflow at {name}")
                popped = stack[len(stack) - n:] if n else []
                if n:
                    del stack[len(stack) - n:]
                if name == "STACK_GLOBAL":
                    mod, qn = popped
                    if not (isinstance(mod, str) and isinstance(qn, str)):
                        raise ValueError("STACK_GLOBAL names a global the scan cannot resolve")
                    sc.refs.add((mod, qn))
                    stack.append(_G(mod, qn))
                    continue
                if name in ("TUPLE1", "TUPLE2", "TUPLE3", "EMPTY_TUPLE"):
                    stack.append(tuple(popped))
                    continue
                if name in ("REDUCE", "NEWOBJ", "NEWOBJ_EX") and isinstance(popped[0], _G):
                    sc.calls.append((popped[0], popped[1]))
            stack.extend(_OPAQUE for x in op.stack_after if x is not mark)
    except Exception as exc:  # noqa: BLE001
        sc.error = f"{type(exc).__name__}: {str(exc)[:60]}"
    return sc


def _file_handle_rule(sc: _PickleScan) -> tuple[str, str] | None:
    """The verdict for a pickle that rebuilds file handles, which dill does by
    re-opening each one BY PATH with its original mode.

      open handle, anywhere in the object          -> rejected (as a top-level
                                                      open handle is)
      closed handle opened for writing ('w', 'x')  -> rejected: the re-open
                                                      truncates the file (or
                                                      raises), on whichever
                                                      side loads it
      other closed handle                          -> unknown
      the process's own std streams                -> no verdict (dill hands
                                                      back this process's)

    Rejected names are excluded from the capsule, which is the point: a
    closed 'w' handle carried to the destination would truncate the file the
    workspace restore has just put back."""
    if sc.oversize:
        if sc.oversize_filehandle:
            return REJECTED, "holds a file handle and is too large to inspect which; excluded, never loaded in the source"
        return None
    rejected, unknown = [], []
    for g, args in sc.calls:
        if (g.mod, g.name) != _FILEHANDLE:
            continue
        a = args if isinstance(args, tuple) else ()
        path = a[0] if len(a) > 0 and a[0] is not _OPAQUE else None      # a str, or an fd number
        mode = a[1] if len(a) > 1 and isinstance(a[1], str) else None
        closed = a[3] if len(a) > 3 and isinstance(a[3], bool) else None
        if isinstance(path, str) and path in _STD_STREAMS:
            continue
        where = f"{str(path)[-80:]!r} mode {mode!r}"
        if closed is not True or mode is None:
            trunc = ", truncating the file" if mode is None or "w" in mode else ""
            rejected.append(f"open file handle inside the object ({where}): pickles, and dill re-opens it by "
                            f"path at load{trunc}; excluded like a top-level handle, never loaded in the source")
        elif "w" in mode or "x" in mode:
            rejected.append(f"closed file handle opened for writing ({where}): dill still re-opens it by path "
                            "at load, which truncates the file (or raises for 'x') on whichever side loads it; "
                            "excluded, never loaded in the source")
        else:
            unknown.append(f"closed file handle ({where}): restores by re-opening its path, which depends on "
                           "the destination's filesystem; not loaded in the source")
    if rejected:
        return REJECTED, rejected[0]
    if unknown:
        return UNKNOWN, unknown[0]
    return None


def _resolve_loaded(mod: str, qn: str) -> Any:
    """A global the pickle names, looked up among modules ALREADY imported
    here (never importing one); None if it is not."""
    o = sys.modules.get(mod)
    for part in qn.split(".") if o is not None else ():
        o = getattr(o, part, None)
        if o is None:
            break
    return o


def _python_finalizer(cls: type) -> type | None:
    for c in getattr(cls, "__mro__", ()):
        if c is not object and (getattr(c, "__flags__", 0) & _HEAPTYPE) and "__del__" in vars(c):
            return c
    return None


def _load_hazard(sc: _PickleScan) -> tuple[str, str] | None:
    """Why this pickle must not be loaded in the source process, or None.
    Checked in order: the scan itself, file handles (which have a verdict of
    their own), resource modules and calls, and finalizers: the validator
    discards its copy, so a `__del__` on it would run here."""
    if sc.error:
        return UNKNOWN, f"pickle not scannable for load-time effects ({sc.error}); not loaded in the source"
    fh = _file_handle_rule(sc)
    if fh is not None:
        return fh
    if sc.oversize:
        return None       # the size limit, checked by the caller, decides
    for mod, qn in sorted(sc.refs):
        if any(mod == p or mod.startswith(p + ".") for p in _RESOURCE_MODULES) or (mod == "pathlib" and "." in qn):
            return UNKNOWN, (f"the pickle references {mod}.{qn}, which creates or acts on a process resource; "
                             "not loaded in the source")
    for g, _args in sc.calls:
        if (g.mod, g.name) in _RESOURCE_CALLS:
            return UNKNOWN, f"loading calls {g.mod}.{g.name}; not loaded in the source"
    if sc.has_del:
        return UNKNOWN, ("a class rebuilt by value defines __del__, which would run in the source when the "
                         "validator discards its copy; not loaded in the source")
    for mod, qn in sorted(sc.refs):
        o = _resolve_loaded(mod, qn)
        c = _python_finalizer(o) if isinstance(o, type) else None
        if c is not None:
            return UNKNOWN, (f"{_tname(c)} defines __del__, which would run in the source when the validator "
                             "discards its copy; not loaded in the source")
    return None


def _round_trip_verdict(obj: Any, payload: bytes | None, dest: Destination | None,
                        max_bytes: int = VALIDATE_MAX_BYTES, name: str = "obj",
                        scan: _PickleScan | None = None) -> tuple[str, str]:
    """Establish a verdict for an object no rule recognises by loading the
    probe's own pickle back IN THIS PROCESS and comparing.

      loading would act on a process resource -> never loaded (see
                                                 `_load_hazard`); rejected for
                                                 a file handle, else unknown
      restored type differs                   -> semantics_change
      same type with its own __eq__           -> that __eq__ decides
      otherwise                               -> state compared structurally,
                                                 sharing included
      load raises                             -> unknown (it fails here; the
                                                 destination's load may too)
      anything the comparison cannot rule     -> unknown

    THE LIMITS, stated plainly.
    (1) This validator runs in the SOURCE process, the user's live runtime,
        so it must never do there what a load would do to the world outside
        the object. `_load_hazard` reads the pickle first and refuses to load
        one that would re-open a file, reference a process-resource module,
        call a resource constructor, or leave a copy whose `__del__` would
        run here. Those objects get their verdict from the rule or are
        `unknown`; they are never exercised.
    (2) What the scan cannot see is SESSION CODE that runs at load: a
        `__setstate__`, a callable a `__reduce__` returns, an `__init__` that
        reduce re-runs, a class-creation hook of a class rebuilt by value.
        Running those hooks is what the validator is for (it is how a reducer
        that restores the wrong thing is caught), so they do run here, and a
        hook with an external effect (a hook that writes a file, starts a
        thread, registers the copy in a global table) has that effect in the
        source. The destination's load runs the same hooks.
    (3) A copy of an object that stands for something outside the process (a
        socket, a descriptor number, a pid) can compare equal here and be dead
        on the destination. The known-type rules, the pickle probe, the
        package section and `process_effects` exist for those, and all of them
        run BEFORE this.
    (4) A type with its own `__eq__` is trusted when it says equal; sharing
        inside such an object is not examined.
    What the validator adds is the one thing only a round trip shows: whether
    the object's own serialisation hooks (`__reduce__`, `__getstate__`,
    `__setstate__`) give back the object that went in, sharing included.
    """
    import dill
    if payload is None:
        return UNKNOWN, "no pickle to validate"
    sc = scan if scan is not None else _scan_pickle(payload, max_bytes)
    hazard = _load_hazard(sc)
    if hazard is not None:
        return hazard
    if len(payload) > max_bytes:
        return UNKNOWN, f"too large to validate in process ({len(payload):,} bytes, limit {max_bytes:,})"
    try:
        back = dill.loads(payload)
    except Exception as exc:  # noqa: BLE001
        return UNKNOWN, (f"round trip raised on load in the source process ({type(exc).__name__}: "
                         f"{str(exc)[:60]}); the destination's load may fail too")
    if _tname(type(obj)) != _tname(type(back)):
        return SEMANTICS_CHANGE, f"round-trip restores as {_tname(type(back))}"
    cmp = _Compare(dest)
    try:
        r = cmp.eq(obj, back, name, 0)
        if r is True:
            r = cmp.finish()
    except RecursionError:
        return UNKNOWN, "validator recursion limit"
    except Exception as exc:  # noqa: BLE001
        return UNKNOWN, f"validator raised {type(exc).__name__}: {str(exc)[:60]}"
    if r is True:
        if cmp.top_method == "own_eq":
            return PRESERVED, "validated by the type's own __eq__"
        return PRESERVED, "round-trip structurally equal"
    if r is False:
        return SEMANTICS_CHANGE, f"round-trip changes the object: {cmp.note}"
    return UNKNOWN, f"validator cannot decide: {cmp.note}"


# ---------------------------------------------------------------------------
# Imported distributions
# ---------------------------------------------------------------------------

def _norm_dist(name: str) -> str:
    """PEP 503 normalisation, so `Foo_Bar` and `foo-bar` are one distribution."""
    return re.sub(r"[-_.]+", "-", name).lower()


def imported_distributions(module_names: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """The distributions that provide a module this process has IMPORTED.

    Comparing the whole `pip freeze` against the destination over-reports: a
    distribution installed on the source and never imported cannot be needed
    to load a capsule. What can be needed is a distribution whose module a
    captured object refers to, and that module is necessarily imported. So the
    set is built from the top-level names in `sys.modules`, mapped to
    distributions by `importlib.metadata.packages_distributions()`. Modules no
    distribution claims (the standard library, synthetic harness modules) are
    not in the result."""
    import importlib.metadata as md
    names = list(sys.modules) if module_names is None else module_names
    top = sorted({n.split(".", 1)[0] for n in names if n and not n.startswith("__")})
    mapping = md.packages_distributions()
    out: dict[str, dict[str, Any]] = {}
    for t in top:
        for dist in mapping.get(t, ()):
            if not dist:
                # A dist-info whose METADATA has no Name maps its modules to
                # None. Nothing can be compared for it, and a None key would
                # break every consumer that sorts or normalises names.
                continue
            if dist not in out:
                try:
                    ver = md.version(dist)
                except Exception:  # noqa: BLE001
                    ver = None
                out[dist] = {"version": ver, "modules": []}
            out[dist]["modules"].append(t)
    return out


def distribution_inventory() -> dict[str, str]:
    """Every installed distribution in this process's environment, name ->
    version. This is what a destination reports about itself."""
    import importlib.metadata as md
    inv: dict[str, str] = {}
    for d in md.distributions():
        try:
            name = d.metadata["Name"]
        except Exception:  # noqa: BLE001
            name = None
        if name:
            inv[name] = d.version
    return inv


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_manifest(
    ns: dict[str, Any],
    dest: Destination,
    adapters: Adapters | None = None,
    *,
    source_python_minor: str | None = None,
    source_dill_version: str | None = None,
    source_imported: dict[str, Any] | None = None,
    source_freeze: dict[str, str] | None = None,
    validate: bool = True,
    validate_max_bytes: int = VALIDATE_MAX_BYTES,
    skip_names: tuple[str, ...] = ("In", "Out", "exit", "quit", "get_ipython", "display"),
) -> Manifest:
    """Classify every name in `ns` for a switch to `dest`.

    `source_imported` (from `imported_distributions()`) drives the package
    section; each value is a version string or a dict with a "version" key.
    `source_freeze` is the older whole-freeze comparison, kept for callers that
    have nothing better, and labelled as such in the output. `validate=False`
    turns the round-trip validator off, leaving every object no rule covers at
    `unknown`."""
    adapters = adapters or Adapters()
    m = Manifest()

    # Compatibility gates: these are exact in the shipped restore program.
    sp = source_python_minor or ".".join(map(str, sys.version_info[:2]))
    try:
        import dill
        sd = source_dill_version or dill.__version__
    except Exception:
        sd = source_dill_version or "?"
    m.gates = {
        "python_minor": sp == dest.python_minor,
        "dill_exact": sd == dest.dill_version,
    }

    shared = _storage_groups(ns)
    for name, obj in ns.items():
        if name.startswith("__") or name in skip_names:
            continue
        tname = type(obj).__module__ + "." + type(obj).__qualname__
        # 1. Things that PICKLE and then poison the all-or-nothing load.
        if _is_file_like_open(obj):
            m.names.append(NameVerdict(name, REJECTED, "open file handle: pickles, then raises during load_session", tname))
            continue
        # 2. Things that do not pickle: the production dump probes and SKIPS them.
        ok, err, payload = _pickle_probe(name, obj)
        if not ok:
            m.names.append(NameVerdict(name, OMITTED, f"unpicklable, will be skipped and reported: {err}", tname))
            continue
        # 3. Types whose post-load behaviour is known.
        known = _classify_known(obj, adapters, dest, shares_storage=name in shared)
        if known is not None:
            m.names.append(NameVerdict(name, known[0], known[1], tname))
            continue
        # 4. Builtin values.
        if _is_builtin_value(obj):
            m.names.append(NameVerdict(name, PRESERVED, "builtin value", tname))
            continue
        # 5. What loading this pickle would do, read from the pickle without
        #    loading it. A file handle anywhere inside the object is a RULE
        #    (dill re-opens it by path at load), so it applies whether or not
        #    the validator runs.
        scan = _scan_pickle(payload, validate_max_bytes)
        fh = _file_handle_rule(scan)
        if fh is not None:
            m.names.append(NameVerdict(name, fh[0], fh[1], tname))
            continue
        # 6. Everything else has to earn a verdict; the default is an abstention.
        if validate:
            verdict, reason = _round_trip_verdict(obj, payload, dest, validate_max_bytes, name, scan=scan)
        else:
            verdict, reason = UNKNOWN, "picklable; no rule applies and validation is off"
        m.names.append(NameVerdict(name, verdict, reason, tname))

    # Package section: which of the source's distributions the destination lacks.
    if dest.packages is not None and (source_imported is not None or source_freeze):
        if source_imported is not None:
            m.package_basis = "imported"
            m.imported_distributions = source_imported
            wanted = {}
            for d, v in source_imported.items():
                if not isinstance(d, str) or not d:
                    # A distribution whose METADATA has no Name: there is
                    # nothing to look up in any inventory, so it is reported
                    # rather than crashing the comparison.
                    m.packages_unattributed += list(v.get("modules") or []) if isinstance(v, dict) else [str(d)]
                    continue
                wanted[d] = v.get("version") if isinstance(v, dict) else v
        else:
            m.package_basis = "freeze"
            wanted = {d: v for d, v in (source_freeze or {}).items() if isinstance(d, str) and d}
        have = {_norm_dist(d): v for d, v in dest.packages.items() if isinstance(d, str) and d}
        for dist, ver in sorted(wanted.items()):
            nd = _norm_dist(dist)
            if nd not in have:
                unrep = any(nd.startswith(_norm_dist(p)) for p in dest.unrepairable)
                m.packages_missing.append({"distribution": dist, "source_version": ver,
                                           "unrepairable": unrep})
            elif ver is not None and have[nd] is not None and have[nd] != ver:
                m.packages_version_skew.append({"distribution": dist, "source_version": ver,
                                                "destination_version": have[nd]})
    return m


# ---------------------------------------------------------------------------
# Process effects: state a namespace walk cannot see
# ---------------------------------------------------------------------------

def _rng_states() -> dict[str, Any]:
    out = {}
    import random
    out["python"] = random.getstate()
    try:
        import numpy as np
        out["numpy_legacy"] = np.random.get_state()[1].tobytes()
    except Exception:
        pass
    try:
        import torch
        out["torch_cpu"] = torch.get_rng_state().numpy().tobytes()
    except Exception:
        pass
    return out


def process_baseline() -> dict[str, Any]:
    """Snapshot the process state a session can mutate without binding a name.

    Heavy libraries are imported FIRST so their own import-time side effects
    (torch appends to sys.path) are not mistaken for the session's. The same
    goes for what the manifest itself will import later: reading distribution
    metadata lazily imports `email.parser` and friends, which adds attributes
    to the `email` package, and a baseline taken before that would report the
    harness's own imports as a mutated library module."""
    for lib in ("numpy", "torch", "dill", "tarfile", "importlib.metadata"):
        try:
            __import__(lib)
        except Exception:
            pass
    try:
        imported_distributions()
        distribution_inventory()
    except Exception:  # noqa: BLE001
        pass
    import logging
    import os
    import threading
    root = logging.getLogger()
    return {
        "sys_path": list(sys.path),
        "environ": dict(os.environ),
        "logging": {"level": root.level, "handlers": len(root.handlers)},
        "threads": {t.ident for t in threading.enumerate()},
        "cwd": os.getcwd(),
        "rng": _rng_states(),
        "module_attrs": {n: set(vars(mod).keys()) for n, mod in list(sys.modules.items())
                         if mod is not None and hasattr(mod, "__dict__")},
    }


def process_effects(baseline: dict[str, Any]) -> list[dict[str, Any]]:
    """Diff the current process against `baseline`. Every entry is a
    SEMANTICS_CHANGE prediction: the effect exists on the source, is not part of
    any capsule layer, and will silently not exist on the destination."""
    import logging
    import os
    import threading
    out = []
    now_path = list(sys.path)
    if now_path != baseline["sys_path"]:
        added = [p for p in now_path if p not in baseline["sys_path"]]
        out.append({"effect": "sys.path", "verdict": SEMANTICS_CHANGE,
                    "reason": f"sys.path mutated (+{len(added)}); not captured, imports may fail on the destination"})
    env_now = dict(os.environ)
    changed = {k for k in set(env_now) | set(baseline["environ"])
               if env_now.get(k) != baseline["environ"].get(k)}
    if changed:
        out.append({"effect": "os.environ", "verdict": SEMANTICS_CHANGE,
                    "reason": f"environment mutated ({len(changed)} keys); process state, not captured"})
    root = logging.getLogger()
    if (root.level, len(root.handlers)) != (baseline["logging"]["level"], baseline["logging"]["handlers"]):
        out.append({"effect": "logging", "verdict": SEMANTICS_CHANGE,
                    "reason": "root logger reconfigured; handlers/level are process configuration"})
    live = {t.ident for t in threading.enumerate()} - baseline["threads"]
    if live:
        out.append({"effect": "threads", "verdict": SEMANTICS_CHANGE,
                    "reason": f"{len(live)} thread(s) started; a restored Thread reports alive in a process that never started it"})
    if os.getcwd() != baseline["cwd"]:
        out.append({"effect": "cwd", "verdict": SEMANTICS_CHANGE,
                    "reason": "working directory changed; not captured"})
    now_rng = _rng_states()
    for k, label in (("python", "python random"), ("numpy_legacy", "legacy numpy global RNG"), ("torch_cpu", "torch CPU RNG")):
        if k in now_rng and k in baseline["rng"] and now_rng[k] != baseline["rng"][k]:
            out.append({"effect": f"rng:{k}", "verdict": SEMANTICS_CHANGE,
                        "reason": f"{label} advanced; global RNG stream position does not continue across the boundary (measured)"})
    for n, keys in baseline["module_attrs"].items():
        mod = sys.modules.get(n)
        if mod is None or not hasattr(mod, "__dict__"):
            continue
        added = set(vars(mod).keys()) - keys
        # __mp_main__ is an alias of __main__; the session's own globals are
        # namespace state, not a mutated library module.
        if added and n not in ("__main__", "__mp_main__") and mod is not sys.modules.get("__main__"):
            out.append({"effect": f"module:{n}", "verdict": SEMANTICS_CHANGE,
                        "reason": f"module {n} gained attributes {sorted(added)[:3]}; modules restore by reference, so the mutation is lost"})
            break
    return out
