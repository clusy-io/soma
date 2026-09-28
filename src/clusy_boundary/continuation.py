"""Continuation: does training proceed the same way after the boundary?

Every other check inspects state at rest. This one runs the system forward a
fixed number of steps from the boundary and records the loss sequence, which is
the only way to catch a transition that restored every value correctly and
still changed what happens next.

Two design choices make the result mean something specific.

The probe reseeds every random stream from a fixed value before stepping. That
is deliberate: randomness has its own rows above, and leaving it free here would
mean a continuation mismatch could not be attributed. With randomness pinned,
this row isolates the question "given identical draws, does the restored state
compute the same updates".

The probe is undone afterwards. Training steps mutate parameters, optimizer
slots and scheduler counters, so a probe that left them changed would corrupt
the very state being certified. Everything it touches is saved before and
restored after, and if the restore cannot be completed the record says so
rather than reporting a loss sequence from a namespace it has damaged.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Callable

StepFn = Callable[[dict[str, Any], int], float]

DEFAULT_STEPS = 5
DEFAULT_SEED = 20260824


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _save_training_state(namespace: dict[str, Any]) -> dict[str, Any] | None:
    try:
        import torch
    except ImportError:
        return None

    saved: dict[str, Any] = {"modules": {}, "optimizers": {}, "others": {}}
    for name, value in namespace.items():
        if name.startswith("_"):
            continue
        try:
            if isinstance(value, torch.nn.Module):
                saved["modules"][name] = (
                    copy.deepcopy({k: v.detach().cpu() for k, v in value.state_dict().items()}),
                    value.training,
                )
            elif isinstance(value, torch.optim.Optimizer):
                saved["optimizers"][name] = copy.deepcopy(value.state_dict())
            elif hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                saved["others"][name] = copy.deepcopy(value.state_dict())
        except Exception:
            # An object that cannot be saved cannot be restored either, so the
            # probe must not run at all rather than run and leave it altered.
            return None

    saved["rng"] = _save_rng()
    return saved


def _save_rng() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    try:
        import torch

        state["torch_cpu"] = torch.get_rng_state().clone()
        if torch.cuda.is_available():
            state["torch_cuda"] = [s.clone() for s in torch.cuda.get_rng_state_all()]
    except Exception:
        pass
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        import torch

        torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])


def _restore_training_state(namespace: dict[str, Any], saved: dict[str, Any]) -> None:
    for name, (sd, was_training) in saved["modules"].items():
        module = namespace.get(name)
        if module is None:
            continue
        device = next((p.device for p in module.parameters()), None)
        target = {k: (v.to(device) if device is not None else v) for k, v in sd.items()}
        module.load_state_dict(target)
        module.train(was_training)
    for name, sd in saved["optimizers"].items():
        opt = namespace.get(name)
        if opt is not None:
            opt.load_state_dict(sd)
    for name, sd in saved["others"].items():
        obj = namespace.get(name)
        if obj is not None:
            obj.load_state_dict(sd)
    _restore_rng(saved["rng"])


def run_continuation(
    namespace: dict[str, Any],
    step_fn: StepFn,
    *,
    steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
    restore: bool = True,
) -> dict[str, Any]:
    """Step ``step_fn`` forward ``steps`` times and record the loss sequence."""
    result: dict[str, Any] = {"steps": steps, "seed": seed, "losses": [], "restored": False}

    saved = _save_training_state(namespace) if restore else None
    if restore and saved is None:
        result["error"] = (
            "training state could not be saved, so the probe was skipped rather "
            "than run destructively"
        )
        return result

    try:
        _seed_everything(seed)
        losses: list[float] = []
        for i in range(steps):
            losses.append(float(step_fn(namespace, i)))
        result["losses"] = losses
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if saved is not None:
            try:
                _restore_training_state(namespace, saved)
                result["restored"] = True
            except Exception as exc:
                result["restored"] = False
                result["error"] = (
                    f"{result.get('error', '')} probe could not be undone: "
                    f"{type(exc).__name__}: {exc}"
                ).strip()

    return result


def load_step_fn(spec: str) -> StepFn:
    """Load ``path/to/file.py:function`` or ``package.module:function``."""
    if ":" not in spec:
        raise ValueError(
            f"continuation spec {spec!r} must be 'file.py:function' or 'module:function'"
        )
    target, func_name = spec.rsplit(":", 1)

    if target.endswith(".py"):
        import importlib.util
        import os

        path = os.path.abspath(target)
        module_spec = importlib.util.spec_from_file_location("_boundary_continuation", path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        import importlib

        module = importlib.import_module(target)

    fn = getattr(module, func_name, None)
    if fn is None:
        raise AttributeError(f"{target} has no attribute {func_name!r}")
    return fn
