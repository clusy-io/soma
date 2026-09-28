"""Optimizer class-identity adaptation: re-attach hidden submodules for the
duration of a dump.

PyTorch's `torch.optim.__init__` does `from .adamw import AdamW` and then
`del adamw`, removing the submodule attribute from the parent package. `dill`
resolves a dotted module name by attribute lookup on the parent package rather
than through `sys.modules`, so the optimizer class is recorded BY VALUE and the
restored object fails `isinstance(opt, torch.optim.Optimizer)`.

`reattach()` temporarily binds every loaded `pkg.leaf` submodule that is
missing as an attribute; `detach()` removes exactly those bindings. Wrap the
dump in try/finally. Re-attaches hidden submodule attributes for the duration
of a dump. In the ablation, re-attachment alone (not `byref`) restored identity
and shrank the capsule 24-54x because the class body is no longer copied.
"""

from __future__ import annotations

import sys
import types


def reattach() -> list[tuple[types.ModuleType, str]]:
    pairs: list[tuple[types.ModuleType, str]] = []
    for full in list(sys.modules):
        if "." not in full:
            continue
        sub = sys.modules.get(full)
        if not isinstance(sub, types.ModuleType):
            continue
        pkg_name, _, leaf = full.rpartition(".")
        pkg = sys.modules.get(pkg_name)
        if not isinstance(pkg, types.ModuleType):
            continue
        try:
            if getattr(pkg, leaf, None) is sub or leaf in vars(pkg):
                continue
        except BaseException:
            continue
        try:
            setattr(pkg, leaf, sub)
            pairs.append((pkg, leaf))
        except BaseException:
            pass
    return pairs


def detach(pairs: list[tuple[types.ModuleType, str]]) -> None:
    for pkg, leaf in pairs:
        try:
            delattr(pkg, leaf)
        except BaseException:
            pass
