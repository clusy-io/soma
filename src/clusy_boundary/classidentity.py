"""Class identity: is a restored object still an instance of the live class?

Serializers may write a class *by reference*, storing the import path and
looking it up again on load, or *by value*, embedding the class definition in
the payload and reconstructing a fresh class object on load. By-value
reconstruction produces an object that has the right attributes, the right
values, and the right methods, and that is nonetheless not an instance of the
class the rest of the process is using.

The failure this catches is real and quiet. ``dill.dump_session`` cannot
resolve ``torch.optim.adamw.AdamW`` by reference and falls back to by-value, so
after a restore ``isinstance(optimizer, torch.optim.Optimizer)`` is False while
``optimizer.step()`` continues to work perfectly. Nothing raises. Every library
that branches on ``isinstance`` then takes the wrong branch, and any registry
keyed on the class object misses.

No value comparison can see this, which is why it needs its own row: the bytes
are all correct.
"""

from __future__ import annotations

import importlib
from typing import Any


def type_path(obj: Any) -> str:
    cls = type(obj)
    module = getattr(cls, "__module__", "?")
    qualname = getattr(cls, "__qualname__", getattr(cls, "__name__", "?"))
    return f"{module}.{qualname}"


def resolves_by_reference(obj: Any) -> bool:
    """True when the object's class is reachable at its own import path.

    This is the exact predicate that distinguishes a by-reference restore from
    a by-value one: import the recorded module, walk the qualname, and check
    that the result *is* the same class object, not merely an equal-looking one.
    """
    cls = type(obj)
    module_name = getattr(cls, "__module__", None)
    qualname = getattr(cls, "__qualname__", None)
    if not module_name or not qualname or "<locals>" in qualname:
        return False
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return False
    target: Any = module
    for part in qualname.split("."):
        target = getattr(target, part, None)
        if target is None:
            return False
    return target is cls


def capture_class_identity(namespace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Record, per top-level name, its class path and whether it resolves."""
    out: dict[str, dict[str, Any]] = {}
    for name, value in namespace.items():
        if name.startswith("_"):
            continue
        try:
            entry: dict[str, Any] = {
                "type_path": type_path(value),
                "by_reference": resolves_by_reference(value),
            }
            # Record membership in the abstract base classes that framework code
            # actually branches on, since that is where the damage shows up.
            entry["isa"] = sorted(_framework_bases(value))
            out[name] = entry
        except Exception as exc:
            out[name] = {"type_path": "?", "by_reference": False, "error": repr(exc)}
    return out


def _framework_bases(obj: Any) -> set[str]:
    """Which well-known base classes this object is currently an instance of."""
    found: set[str] = set()
    try:
        import torch

        checks = {
            "torch.nn.Module": torch.nn.Module,
            "torch.optim.Optimizer": torch.optim.Optimizer,
            "torch.Tensor": torch.Tensor,
            "torch.utils.data.Dataset": torch.utils.data.Dataset,
            "torch.utils.data.DataLoader": torch.utils.data.DataLoader,
        }
        try:
            from torch.optim.lr_scheduler import LRScheduler

            checks["torch.optim.lr_scheduler.LRScheduler"] = LRScheduler
        except Exception:
            pass
        for label, base in checks.items():
            try:
                if isinstance(obj, base):
                    found.add(label)
            except Exception:
                continue
    except Exception:
        pass
    return found
