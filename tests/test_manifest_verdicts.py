"""The manifest must EARN `preserved`.

Each test pins one behaviour of the verdict logic that the first version got
wrong or did not have: a picklable object no rule covered was predicted
preserved, which let a custom reducer that restores as a different type
through the gate. The classes live at module level on purpose, so dill pickles
them by reference and the in-process round trip returns the same class, as it
does for a class in a kernel's real `__main__`.
"""

from __future__ import annotations

import collections
import importlib
import sys

import pytest

from capsule.manifest import (
    PRESERVED,
    REJECTED,
    SEMANTICS_CHANGE,
    UNKNOWN,
    Destination,
    build_manifest,
    imported_distributions,
    _is_builtin_value,
)

DEST = Destination(has_cuda=False, python_minor="%d.%d" % sys.version_info[:2],
                   dill_version=importlib.import_module("dill").__version__)


def verdict(obj, **kw):
    m = build_manifest({"x": obj}, DEST, **kw)
    (nv,) = m.names
    return nv.verdict, nv.reason


# -- fixtures: user classes with and without their own serialisation hooks --

class Plain:
    def __init__(self, a, b):
        self.a, self.b = a, b


class ReducesToDict:
    """The critique's counterexample: pickles cleanly, restores as a dict."""
    def __init__(self, c):
        self.c = c

    def __reduce__(self):
        return (dict, ({"c": self.c},))


class ReducesToZero:
    """Same type back, different value."""
    def __init__(self, n):
        self.n = n

    def __reduce__(self):
        return (ReducesToZero, (0,))


class DropsField:
    def __init__(self):
        self.keep, self.hits = 1, 41

    def __getstate__(self):
        s = dict(self.__dict__)
        s.pop("hits")
        return s

    def __setstate__(self, s):
        self.__dict__.update(s)
        self.hits = 0


class EqByKey:
    """Own __eq__ that says equal after the round trip."""
    def __init__(self, key, scratch):
        self.key, self.scratch = key, scratch

    def __eq__(self, other):
        return isinstance(other, EqByKey) and other.key == self.key

    __hash__ = None


class EqByKeyReset:
    """Own __eq__, and a reducer that changes the field __eq__ looks at."""
    def __init__(self, key):
        self.key = key

    def __eq__(self, other):
        return isinstance(other, EqByKeyReset) and other.key == self.key

    __hash__ = None

    def __reduce__(self):
        return (EqByKeyReset, ("reset",))


class EqRaises:
    def __init__(self):
        self.v = 1

    def __eq__(self, other):
        raise RuntimeError("no comparisons here")

    __hash__ = None


# -- the default is an abstention ---------------------------------------------

def test_default_for_an_unruled_picklable_object_is_unknown():
    v, reason = verdict(Plain(1, 2), validate=False)
    assert v == UNKNOWN, reason


def test_undecidable_object_stays_unknown_even_with_the_validator():
    it = iter([1, 2, 3]); next(it)
    v, reason = verdict(it)
    assert v == UNKNOWN
    assert "C type" in reason


def test_manifest_properties_distinguish_admissible_from_fully_validated():
    it = iter([1])
    m = build_manifest({"it": it, "n": 3}, DEST)
    assert [n.name for n in m.unknown] == ["it"]
    assert m.admissible and not m.fully_validated
    assert m.counts[UNKNOWN] == 1 and m.counts[PRESERVED] == 1
    assert m.to_json()["unknown"] == ["it"]


# -- the critique's counterexample and its relatives -----------------------------

def test_custom_reducer_to_another_type_is_semantics_change():
    v, reason = verdict(ReducesToDict(100.0))
    assert v == SEMANTICS_CHANGE
    assert "restores as builtins.dict" in reason


def test_reducer_to_same_type_with_another_value_is_semantics_change():
    v, reason = verdict(ReducesToZero(5))
    assert v == SEMANTICS_CHANGE
    assert "x.n" in reason


def test_setstate_that_resets_a_field_is_semantics_change():
    v, reason = verdict(DropsField())
    assert v == SEMANTICS_CHANGE
    assert "hits" in reason


# -- the value rule ----------------------------------------------------------------

def test_builtin_values_are_preserved_by_the_value_rule():
    cyc = {"name": "root"}
    cyc["self"] = cyc
    shared = [1, 2]
    for obj in (None, True, 7, 2**91, 0.1 + 0.2, float("nan"), 1j, "s", b"b", bytearray(b"ba"),
                [1, (2, 3), {"k": {4, 5}}], frozenset({1, "a"}), cyc, {"a": shared, "b": shared}):
        v, reason = verdict(obj)
        assert (v, reason) == (PRESERVED, "builtin value"), (obj, v, reason)


def test_value_rule_is_exact_types_only():
    assert _is_builtin_value([1, {"a": (2, None)}])
    assert not _is_builtin_value([1, Plain(1, 2)])                 # a non-value member
    assert not _is_builtin_value(collections.OrderedDict(a=1))     # a subclass can carry its own reducer
    # Too deep for the rule: falls through to the validator, never to "preserved" by default.
    deep = cur = []
    for _ in range(200):
        nxt = []
        cur.append(nxt)
        cur = nxt
    assert not _is_builtin_value(deep)


# -- the validator's equality paths -------------------------------------------------

def test_plain_instance_that_round_trips_is_structurally_equal():
    assert verdict(Plain(1, [2, 3])) == (PRESERVED, "round-trip structurally equal")


def test_own_eq_that_holds_after_the_round_trip_validates():
    assert verdict(EqByKey("k", object())) == (PRESERVED, "validated by the type's own __eq__")


def test_own_eq_that_fails_after_the_round_trip_is_semantics_change():
    v, reason = verdict(EqByKeyReset("original"))
    assert v == SEMANTICS_CHANGE
    assert "own __eq__" in reason


def test_own_eq_that_raises_is_unknown():
    v, reason = verdict(EqRaises())
    assert v == UNKNOWN
    assert "__eq__ raised" in reason


def test_nested_container_of_user_objects_goes_through_the_validator():
    v, reason = verdict([Plain(1, 2), ReducesToZero(3)])
    assert v == SEMANTICS_CHANGE
    assert "x[1].n" in reason


def test_functions_are_compared_by_code_defaults_and_closure():
    def make(k):
        def inner(x, y=2):
            return x * k + y
        return inner
    assert verdict(make(7)) == (PRESERVED, "round-trip structurally equal")


def test_known_rules_keep_precedence_over_the_validator():
    open_file = open(__file__)
    try:
        v, _ = verdict(open_file)
        assert v == REJECTED
    finally:
        open_file.close()


# -- the package section: imported distributions only ---------------------------

@pytest.fixture
def dummy_distribution(tmp_path):
    """A distribution only this process has: a package plus a dist-info."""
    mod, dist = "clusy_test_probe_dist", "clusy-test-probe-dist"
    (tmp_path / mod).mkdir()
    (tmp_path / mod / "__init__.py").write_text("class Marker:\n    def __init__(self, n):\n        self.n = n\n")
    info = tmp_path / f"{mod}-0.2.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist}\nVersion: 0.2.0\n")
    (info / "top_level.txt").write_text(mod + "\n")
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        yield mod, dist, importlib.import_module(mod)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(mod, None)
        importlib.invalidate_caches()


def test_imported_distributions_maps_imported_modules_to_distributions(dummy_distribution):
    mod, dist, _ = dummy_distribution
    got = imported_distributions([mod, "json", "numpy.linalg"])
    assert got[dist] == {"version": "0.2.0", "modules": [mod]}
    assert "numpy" in got                       # a submodule counts through its top-level name
    assert all(d in (dist, "numpy") for d in got)   # stdlib maps to nothing


def test_missing_imported_distribution_blocks_only_when_unrepairable(dummy_distribution):
    mod, dist, pkg = dummy_distribution
    ns = {"marker": pkg.Marker(3)}
    imported = imported_distributions([mod, "numpy"])
    import numpy
    have = {"numpy": numpy.__version__}          # the destination has numpy, lacks the dummy
    # A default destination may repair the dummy (not on its forbidden list).
    m = build_manifest(ns, Destination(False, DEST.python_minor, DEST.dill_version, packages=have),
                       source_imported=imported)
    assert [p["distribution"] for p in m.packages_missing] == [dist]
    assert m.package_basis == "imported"
    assert m.admissible and m.blocking == []
    # A destination with NO repair path: every missing distribution blocks.
    m = build_manifest(ns, Destination(False, DEST.python_minor, DEST.dill_version, packages=have,
                                       unrepairable=("",)), source_imported=imported)
    assert not m.admissible
    assert m.blocking == [f"missing_distribution:{dist}"]
    # The per-name verdict cannot see it: in this process the module imports.
    assert m.names[0].verdict == PRESERVED


def test_distributions_installed_but_never_imported_are_not_compared(dummy_distribution):
    mod, dist, _ = dummy_distribution
    # The destination has NOTHING; only what was imported may be reported.
    m = build_manifest({}, Destination(False, DEST.python_minor, DEST.dill_version, packages={},
                                       unrepairable=("",)),
                       source_imported=imported_distributions([mod]))
    assert [p["distribution"] for p in m.packages_missing] == [dist]


def test_distribution_names_are_normalised_and_version_skew_is_reported(dummy_distribution):
    mod, dist, _ = dummy_distribution
    m = build_manifest({}, Destination(False, DEST.python_minor, DEST.dill_version,
                                       packages={"Clusy_Test.Probe_Dist": "0.1.0"}),
                       source_imported=imported_distributions([mod]))
    assert m.packages_missing == []
    assert m.packages_version_skew == [{"distribution": dist, "source_version": "0.2.0",
                                        "destination_version": "0.1.0"}]


def test_manifest_without_the_rejected_names_is_admissible():
    fh = open(__file__)
    try:
        m = build_manifest({"fh": fh, "n": 1}, DEST)
        assert not m.admissible
        assert m.without([n.name for n in m.rejected]).admissible
    finally:
        fh.close()


# -- the validator never does in the source what a load would do -----------------
#
# The validator loads pickles in the SOURCE process. dill rebuilds a file
# handle by re-opening its path with its original mode, so the first version
# truncated a user's open 'w+' log (and any file behind a CLOSED 'w' handle)
# while "judging" it. Each test checks the bytes on disk, not only the verdict.

class RunLog:
    """A user object that keeps its log file open, as training loops do."""
    def __init__(self, path):
        self.path = path
        self.fh = open(path, "w+")
        self.fh.write("epoch 1 loss 0.9\nepoch 2 loss 0.7\n")
        self.fh.flush()


class ReopensForWriting:
    """A reducer whose load re-opens a path for writing."""
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (open, (self.path, "w"))


FINALIZED: list[int] = []


class Finalized:
    def __init__(self):
        self.v = 1

    def __del__(self):
        FINALIZED.append(self.v)


class RefusesToLoad:
    def __init__(self):
        self.v = 1

    def __setstate__(self, s):
        raise RuntimeError("registry already holds this id")


def _bytes(p):
    return p.read_bytes()


@pytest.mark.parametrize("validate", [True, False])
def test_nested_open_handle_is_rejected_and_the_file_keeps_its_bytes(tmp_path, validate):
    p = tmp_path / "train.log"
    log = RunLog(str(p))
    try:
        before, pos = _bytes(p), log.fh.tell()
        v, reason = verdict(log, validate=validate)
        assert v == REJECTED and "open file handle inside the object" in reason
        assert _bytes(p) == before and len(before) == 34        # not truncated
        assert not log.fh.closed and log.fh.tell() == pos        # the handle itself untouched
    finally:
        log.fh.close()


def test_closed_write_handle_is_rejected_and_the_file_keeps_its_bytes(tmp_path):
    # `with open(path, "w") as f:` at the top of a cell leaves this `f` bound.
    p = tmp_path / "out.csv"
    with open(p, "w") as f:
        f.write("a,b\n1,2\n")
    v, reason = verdict(f)
    assert v == REJECTED and "closed file handle opened for writing" in reason
    assert _bytes(p) == b"a,b\n1,2\n"


def test_closed_read_handle_is_unknown_and_not_loaded(tmp_path):
    p = tmp_path / "in.csv"
    p.write_text("x\n")
    with open(p) as f:
        f.read()
    v, reason = verdict(f)
    assert v == UNKNOWN and "closed file handle" in reason


def test_a_reducer_that_calls_open_is_never_loaded(tmp_path):
    p = tmp_path / "keep.txt"
    p.write_text("precious")
    v, reason = verdict(ReopensForWriting(str(p)))
    assert v == UNKNOWN and "loading calls" in reason and "open" in reason
    assert p.read_text() == "precious"


def test_a_reference_into_a_resource_module_keeps_the_pickle_out_of_the_source():
    import shutil
    v, reason = verdict(Plain(shutil.rmtree, 1))
    assert v == UNKNOWN and "shutil.rmtree" in reason


def test_a_finalizer_on_the_copy_never_runs_in_the_source():
    import gc
    obj = Finalized()
    FINALIZED.clear()
    v, reason = verdict(obj)
    gc.collect()
    assert v == UNKNOWN and "__del__" in reason
    assert FINALIZED == []                                   # no copy was made to be finalized


def test_a_load_that_raises_in_the_source_is_unknown_not_rejected():
    v, reason = verdict(RefusesToLoad())
    assert v == UNKNOWN and "raised on load" in reason


def test_an_unscannable_pickle_is_never_loaded():
    from capsule.manifest import _round_trip_verdict
    v, reason = _round_trip_verdict(Plain(1, 2), b"\x80\x04\x95garbage", DEST)
    assert v == UNKNOWN and "not scannable" in reason


def test_the_scan_follows_ordinary_pickles_without_error():
    from capsule.manifest import _load_hazard, _scan_pickle
    import dill
    import numpy as np
    for obj in (Plain([1, 2], {"k": (3, 4)}), np.arange(5), [Plain(1, 2)] * 3, collections.OrderedDict(a=1),
                make_closure(), DropsField()):
        sc = _scan_pickle(dill.dumps(obj, recurse=True))
        assert sc.error is None and _load_hazard(sc) is None, obj


def make_closure():
    k = [1]

    def f(x):
        return x + k[0]
    return f


# -- sharing is part of the state ----------------------------------------------------

class SharedFields:
    def __init__(self):
        self.a = [1, 2]
        self.b = self.a


class CopiesOneAlias(SharedFields):
    def __getstate__(self):
        return {"a": self.a, "b": list(self.b)}


class MergesTwo:
    def __init__(self):
        self.a, self.b = [1], [1]

    def __getstate__(self):
        return {"a": self.a, "b": self.a}


class HoldsNumpyView:
    def __init__(self):
        import numpy as np
        self.w = np.arange(6.0)
        self.v = self.w[:3]


class HoldsInterleaved:
    def __init__(self):
        import numpy as np
        base = np.arange(8.0)
        self.even, self.odd = base[::2], base[1::2]      # overlapping extents, no shared element


def test_sharing_the_pickle_memo_keeps_is_preserved():
    assert verdict(SharedFields()) == (PRESERVED, "round-trip structurally equal")


def test_a_getstate_that_splits_an_alias_is_semantics_change():
    v, reason = verdict(CopiesOneAlias())
    assert v == SEMANTICS_CHANGE and "aliasing differs" in reason and "restores as two" in reason


def test_a_getstate_that_merges_two_objects_is_semantics_change():
    v, reason = verdict(MergesTwo())
    assert v == SEMANTICS_CHANGE and "restore as one" in reason


def test_a_numpy_view_of_a_sibling_field_is_semantics_change():
    v, reason = verdict(HoldsNumpyView())
    assert v == SEMANTICS_CHANGE and "memory shared with x.w" in reason


def test_interleaved_views_that_share_no_element_are_not_called_shared():
    assert verdict(HoldsInterleaved()) == (PRESERVED, "round-trip structurally equal")


def test_a_torch_view_of_a_sibling_field_is_semantics_change():
    torch = pytest.importorskip("torch")
    h = Plain(torch.zeros(4), None)
    h.b = h.a[:2]
    v, reason = verdict(h)
    assert v == SEMANTICS_CHANGE and "aliasing differs" in reason


def test_temporaries_read_during_the_walk_are_not_mistaken_for_aliases():
    # A Generator's `.state` is a fresh dict on every access; were it freed,
    # its id could be reused by the next one and look like an alias.
    import numpy as np
    obj = Plain([np.random.default_rng(i) for i in range(20)], None)
    assert verdict(obj) == (PRESERVED, "round-trip structurally equal")


# -- a distribution with no Name --------------------------------------------------

def test_a_distribution_without_a_name_is_skipped_not_a_crash(tmp_path):
    mod = "clusy_test_nameless_dist"
    (tmp_path / mod).mkdir()
    (tmp_path / mod / "__init__.py").write_text("X = 1\n")
    info = tmp_path / f"{mod}-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\n")
    (info / "top_level.txt").write_text(mod + "\n")
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        importlib.import_module(mod)
        got = imported_distributions([mod, "numpy"])
        assert None not in got and all(isinstance(k, str) and k for k in got)
        # And a caller that hands build_manifest a nameless entry anyway.
        m = build_manifest({}, Destination(False, DEST.python_minor, DEST.dill_version, packages={},
                                           unrepairable=("",)),
                           source_imported={None: {"version": None, "modules": [mod]}})
        assert m.packages_unattributed == [mod] and m.packages_missing == []
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(mod, None)
        importlib.invalidate_caches()
