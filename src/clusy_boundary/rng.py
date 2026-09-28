"""Random-number state across the four streams a training loop actually uses.

Comparing the raw state buffers is necessary but weak: two encodings of the
same generator position can differ byte-wise, and, worse, a buffer can survive
a boundary intact while nothing ever installed it into the live generator. So
each stream is probed twice. The state digest answers "was the value carried",
and a forward draw answers "was it installed" by pulling a short sequence from
the live generator and putting the state back exactly as it was found.

The probe is non-destructive by construction: every stream is read, advanced by
the probe, and reset from the saved value. A boundary check must not perturb
the run it is measuring.

One asymmetry is unavoidable and is reported rather than smoothed over. A
namespace moved from a GPU host to a CPU-only host cannot install CUDA
generator state, because there is no device to install it into. That is not the
same as losing it. The stream is therefore judged on whether the bytes were
carried across intact, and the row says so, so that a later move back to a GPU
can be held to the stronger standard.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

PROBE_DRAWS = 8


def _digest(payload: Any) -> str:
    return hashlib.sha256(repr(payload).encode("utf-8", "replace")).hexdigest()[:32]


@dataclass
class StreamCapture:
    available: bool
    state_digest: str | None = None
    forward_draws: list[float] | list[int] | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "state_digest": self.state_digest,
            "forward_draws": self.forward_draws,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StreamCapture":
        return cls(
            available=d["available"],
            state_digest=d.get("state_digest"),
            forward_draws=d.get("forward_draws"),
            note=d.get("note", ""),
        )


@dataclass
class RngCapture:
    python: StreamCapture
    numpy_legacy: StreamCapture
    torch_cpu: StreamCapture
    torch_cuda: StreamCapture
    #: Named numpy Generator objects found in the namespace.
    generators: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "python": self.python.to_dict(),
            "numpy_legacy": self.numpy_legacy.to_dict(),
            "torch_cpu": self.torch_cpu.to_dict(),
            "torch_cuda": self.torch_cuda.to_dict(),
            "generators": self.generators,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RngCapture":
        return cls(
            python=StreamCapture.from_dict(d["python"]),
            numpy_legacy=StreamCapture.from_dict(d["numpy_legacy"]),
            torch_cpu=StreamCapture.from_dict(d["torch_cpu"]),
            torch_cuda=StreamCapture.from_dict(d["torch_cuda"]),
            generators=d.get("generators", {}),
        )


def _capture_python() -> StreamCapture:
    state = random.getstate()
    draws = [random.random() for _ in range(PROBE_DRAWS)]
    random.setstate(state)
    return StreamCapture(True, _digest(state), draws)


def _capture_numpy_legacy() -> StreamCapture:
    try:
        import numpy as np
    except ImportError:
        return StreamCapture(False, note="numpy not importable")
    state = np.random.get_state()
    draws = [float(x) for x in np.random.random(PROBE_DRAWS)]
    np.random.set_state(state)
    return StreamCapture(True, _digest(state), draws)


def _capture_torch_cpu() -> StreamCapture:
    try:
        import torch
    except ImportError:
        return StreamCapture(False, note="torch not importable")
    state = torch.get_rng_state()
    draws = [float(x) for x in torch.rand(PROBE_DRAWS)]
    torch.set_rng_state(state)
    return StreamCapture(True, _digest(state.tolist()), draws)


def _capture_torch_cuda() -> StreamCapture:
    try:
        import torch
    except ImportError:
        return StreamCapture(False, note="torch not importable")
    if not torch.cuda.is_available():
        return StreamCapture(False, note="no CUDA device on this host")
    states = torch.cuda.get_rng_state_all()
    device = torch.device("cuda", torch.cuda.current_device())
    draws = [float(x) for x in torch.rand(PROBE_DRAWS, device=device).cpu()]
    torch.cuda.set_rng_state_all(states)
    return StreamCapture(True, _digest([s.tolist() for s in states]), draws)


def _capture_named_generators(namespace: dict[str, Any]) -> dict[str, str]:
    """Digest every numpy Generator bound to a top-level name.

    A modern training loop usually threads an explicit ``np.random.Generator``
    rather than touching the legacy global, so a harness that only checked the
    global would report PASS while the generator that actually drives sampling
    was reset.
    """
    found: dict[str, str] = {}
    try:
        import numpy as np
    except ImportError:
        return found
    for name, value in namespace.items():
        if name.startswith("_"):
            continue
        if isinstance(value, np.random.Generator):
            try:
                found[name] = _digest(value.bit_generator.state)
            except Exception:
                found[name] = "unreadable"
    return found


def capture_rng(namespace: dict[str, Any] | None = None) -> RngCapture:
    return RngCapture(
        python=_capture_python(),
        numpy_legacy=_capture_numpy_legacy(),
        torch_cpu=_capture_torch_cpu(),
        torch_cuda=_capture_torch_cuda(),
        generators=_capture_named_generators(namespace or {}),
    )
