"""Aliasing structure of a namespace, captured as a name partition.

Two variables alias when their reachable object graphs share storage. That
relation is what makes a set of variables an atomic unit of transfer: writing
through one name must still be visible through the other after the boundary,
and a serializer that walks each name independently silently breaks it.

The partition of top-level names induced by that relation is the thing worth
comparing across a boundary. Raw addresses are meaningless after a move (every
object lives somewhere new), so this module records addresses only to derive
the partition and then compares partitions.

Sharing is detected through three kinds of identity token:

- object identity, ``id(obj)``, for anything reachable;
- the numpy data pointer, which makes ``a`` and ``a[2:5]`` share a token even
  though the array objects differ;
- the torch untyped-storage pointer, which does the same for tensor views and,
  importantly, for parameters versus the optimizer slots that were created by
  ``torch.zeros_like`` on a view of them.

Traversal is bounded. A namespace holding a large graph is common, and an
unbounded walk would dominate the cost of the boundary being measured. When a
bound is hit the object is recorded as truncated, which is reported rather
than hidden, so a partition derived from an incomplete walk is never presented
as if it were complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# Types walked structurally rather than treated as leaves.
_CONTAINER_TYPES = (list, tuple, set, frozenset, dict)

# Walking these yields nothing useful and can be very expensive or can trigger
# imports/IO through descriptors.
_ATOMIC_TYPES = (
    str, bytes, bytearray, int, float, complex, bool, type(None),
    type, type(len),
)


@dataclass
class TraversalBudget:
    max_objects: int = 20_000
    max_depth: int = 12

    def fresh(self) -> "_BudgetState":
        return _BudgetState(self)


@dataclass
class _BudgetState:
    budget: TraversalBudget
    seen_objects: int = 0
    truncated: bool = False

    def spend(self) -> bool:
        self.seen_objects += 1
        if self.seen_objects > self.budget.max_objects:
            self.truncated = True
            return False
        return True


@dataclass
class AliasSnapshot:
    """The aliasing structure of a namespace at one instant."""

    #: Partition of top-level names. Each frozenset is one aliasing component.
    partition: frozenset[frozenset[str]]
    #: Names whose traversal hit the budget, so their component may be incomplete.
    truncated_names: frozenset[str] = field(default_factory=frozenset)
    #: Per-name count of distinct storage tokens, useful for triage only.
    token_counts: dict[str, int] = field(default_factory=dict)

    @property
    def components(self) -> list[list[str]]:
        return sorted([sorted(c) for c in self.partition])

    def component_of(self, name: str) -> frozenset[str] | None:
        for comp in self.partition:
            if name in comp:
                return comp
        return None


def _storage_tokens(obj: Any) -> Iterable[tuple[str, int]]:
    """Identity tokens for the memory an object owns or views into."""
    # numpy: a view's data pointer is the base pointer plus its offset, so two
    # slices of one buffer taken at different offsets do NOT share a pointer
    # value. Walking to the root of the base chain and using that object's
    # pointer is what makes them compare equal, which is the whole point.
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            root = obj
            while getattr(root, "base", None) is not None:
                root = root.base
                # The base chain can end at a non-array buffer (a memoryview or
                # a bytes object backing the array); its identity still works as
                # a shared token.
                if not isinstance(root, np.ndarray):
                    yield ("id", id(root))
                    root = None
                    break
            if root is not None:
                iface = getattr(root, "__array_interface__", None)
                if iface:
                    ptr = iface.get("data")
                    if isinstance(ptr, tuple) and ptr[0]:
                        yield ("npbuf", int(ptr[0]))
                # Unified with the generic object token so that the base array,
                # when it is itself a top-level name, joins the same component.
                yield ("id", id(root))
            return
    except Exception:
        pass

    # torch: untyped_storage().data_ptr() is stable across views and is what
    # makes a parameter and an optimizer slot built from it detectably related.
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            try:
                storage = obj.untyped_storage()
                ptr = storage.data_ptr()
                if ptr:
                    yield ("torch", int(ptr))
            except Exception:
                # Meta/fake/sparse tensors have no accessible storage.
                pass
            return
    except Exception:
        pass


def _walk(root: Any, budget: _BudgetState) -> set[tuple[str, int]]:
    """Collect identity tokens reachable from ``root``."""
    tokens: set[tuple[str, int]] = set()
    visited: set[int] = set()
    stack: list[tuple[Any, int]] = [(root, 0)]

    while stack:
        obj, depth = stack.pop()
        if depth > budget.budget.max_depth:
            budget.truncated = True
            continue
        if isinstance(obj, _ATOMIC_TYPES):
            continue
        oid = id(obj)
        if oid in visited:
            continue
        visited.add(oid)
        if not budget.spend():
            continue

        tokens.add(("id", oid))
        for tok in _storage_tokens(obj):
            tokens.add(tok)

        # Descend.
        try:
            if isinstance(obj, dict):
                for k, v in list(obj.items())[:4096]:
                    stack.append((k, depth + 1))
                    stack.append((v, depth + 1))
                continue
            if isinstance(obj, _CONTAINER_TYPES):
                for v in list(obj)[:4096]:
                    stack.append((v, depth + 1))
                continue
        except Exception:
            continue

        # torch modules and optimizers expose their tensors through ordinary
        # attributes, so the generic __dict__ walk below reaches them.
        try:
            slots = getattr(obj, "__dict__", None)
            if isinstance(slots, dict):
                for v in list(slots.values())[:4096]:
                    stack.append((v, depth + 1))
        except Exception:
            pass

    return tokens


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self._parent = {i: i for i in items}

    def find(self, x: str) -> str:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def groups(self) -> list[set[str]]:
        out: dict[str, set[str]] = {}
        for item in self._parent:
            out.setdefault(self.find(item), set()).add(item)
        return list(out.values())


def capture_alias_snapshot(
    namespace: dict[str, Any],
    *,
    budget: TraversalBudget | None = None,
    skip: frozenset[str] | None = None,
) -> AliasSnapshot:
    """Compute the aliasing partition of ``namespace``'s top-level names."""
    budget = budget or TraversalBudget()
    skip = skip or frozenset()

    names = [n for n in namespace if not n.startswith("_") and n not in skip]
    tokens_by_name: dict[str, set[tuple[str, int]]] = {}
    truncated: set[str] = set()

    for name in names:
        state = budget.fresh()
        try:
            tokens_by_name[name] = _walk(namespace[name], state)
        except Exception:
            tokens_by_name[name] = set()
            truncated.add(name)
        if state.truncated:
            truncated.add(name)

    # Invert to token -> names, then union every name that shares a token.
    owners: dict[tuple[str, int], list[str]] = {}
    for name, toks in tokens_by_name.items():
        for tok in toks:
            owners.setdefault(tok, []).append(name)

    uf = _UnionFind(names)
    for holders in owners.values():
        if len(holders) < 2:
            continue
        first = holders[0]
        for other in holders[1:]:
            uf.union(first, other)

    partition = frozenset(frozenset(g) for g in uf.groups())
    return AliasSnapshot(
        partition=partition,
        truncated_names=frozenset(truncated),
        token_counts={n: len(t) for n, t in tokens_by_name.items()},
    )


def diff_partitions(
    before: AliasSnapshot, after: AliasSnapshot
) -> tuple[list[str], list[str]]:
    """Describe how two partitions differ.

    Returns (split_descriptions, merge_descriptions). A split means names that
    aliased before the boundary no longer do, which is the failure that breaks
    write-visibility. A merge means names that were independent now share
    storage, which is rarer but equally wrong: it means the restore aliased two
    things that should have stayed distinct.
    """
    splits: list[str] = []
    merges: list[str] = []

    def pair_set(snap: AliasSnapshot) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for comp in snap.partition:
            members = sorted(comp)
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    pairs.add((a, b))
        return pairs

    before_pairs = pair_set(before)
    after_pairs = pair_set(after)

    for a, b in sorted(before_pairs - after_pairs):
        splits.append(f"{a} ~ {b} aliased before, independent after")
    for a, b in sorted(after_pairs - before_pairs):
        merges.append(f"{a} ~ {b} independent before, aliased after")

    return splits, merges
