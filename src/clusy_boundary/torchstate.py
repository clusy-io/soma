"""Training-object state: parameters, buffers, optimizer, scheduler, scaler, loader.

Everything here reduces live objects to comparable, device-independent records.
Tensor values are hashed from their exact bytes after a move to host memory,
because moving a tensor between devices copies bits without changing them: a
parameter that differs after a GPU-to-CPU transfer differs because something
was lost, not because it changed device.

The subtle row is the optimizer's parameter references. An optimizer does not
merely hold values, it holds the identity of the tensors it will write into.
A restore that rebuilds the model and the optimizer separately can produce a
namespace where every value compares equal and training still appears to run,
while the optimizer updates tensors the model no longer uses. The loss curve
looks plausible and the model never improves. Recording which model parameter
each optimizer slot points to, by identity rather than by value, is what turns
that silent failure into a FAIL row.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


def _tensor_digest(t: Any) -> str:
    """Byte-exact digest of a tensor's contents, independent of its device."""
    import torch

    if not isinstance(t, torch.Tensor):
        return "not-a-tensor"
    try:
        host = t.detach().to("cpu", copy=True).contiguous()
        buf = host.view(torch.uint8) if host.dtype != torch.uint8 else host
        raw = buf.numpy().tobytes()
    except Exception:
        # Views of exotic dtypes (bfloat16 on old builds, complex, sparse)
        # cannot always be reinterpreted as bytes; fall back to a numeric
        # serialization that is still exact for the values it covers.
        try:
            raw = repr(t.detach().to("cpu").tolist()).encode()
        except Exception:
            return "undigestible"
    h = hashlib.sha256(raw)
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    return h.hexdigest()[:32]


@dataclass
class TensorRecord:
    digest: str
    shape: list[int]
    dtype: str
    device_type: str
    requires_grad: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "shape": self.shape,
            "dtype": self.dtype,
            "device_type": self.device_type,
            "requires_grad": self.requires_grad,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TensorRecord":
        return cls(
            digest=d["digest"],
            shape=d["shape"],
            dtype=d["dtype"],
            device_type=d["device_type"],
            requires_grad=d.get("requires_grad", False),
        )

    def value_equal(self, other: "TensorRecord") -> bool:
        return (
            self.digest == other.digest
            and self.shape == other.shape
            and self.dtype == other.dtype
        )


def _record(t: Any) -> TensorRecord:
    return TensorRecord(
        digest=_tensor_digest(t),
        shape=list(t.shape),
        dtype=str(t.dtype),
        device_type=t.device.type,
        requires_grad=bool(getattr(t, "requires_grad", False)),
    )


@dataclass
class ModuleCapture:
    parameters: dict[str, TensorRecord] = field(default_factory=dict)
    buffers: dict[str, TensorRecord] = field(default_factory=dict)
    training: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": {k: v.to_dict() for k, v in self.parameters.items()},
            "buffers": {k: v.to_dict() for k, v in self.buffers.items()},
            "training": self.training,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModuleCapture":
        return cls(
            parameters={k: TensorRecord.from_dict(v) for k, v in d["parameters"].items()},
            buffers={k: TensorRecord.from_dict(v) for k, v in d["buffers"].items()},
            training=d.get("training", True),
        )


@dataclass
class OptimizerCapture:
    kind: str
    #: Hyperparameters per group, minus the parameter list itself.
    groups: list[dict[str, Any]] = field(default_factory=list)
    #: (group index, slot index) -> the model-parameter name that slot points at.
    param_refs: dict[str, str] = field(default_factory=dict)
    #: "<param name>/<slot key>" -> record, for tensor-valued optimizer state.
    slots: dict[str, TensorRecord] = field(default_factory=dict)
    #: "<param name>/<slot key>" -> value, for scalar optimizer state (step counts).
    scalar_slots: dict[str, Any] = field(default_factory=dict)
    #: Parameters the optimizer references that belong to no known module.
    unbound: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "groups": self.groups,
            "param_refs": self.param_refs,
            "slots": {k: v.to_dict() for k, v in self.slots.items()},
            "scalar_slots": self.scalar_slots,
            "unbound": self.unbound,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OptimizerCapture":
        return cls(
            kind=d["kind"],
            groups=d.get("groups", []),
            param_refs=d.get("param_refs", {}),
            slots={k: TensorRecord.from_dict(v) for k, v in d.get("slots", {}).items()},
            scalar_slots=d.get("scalar_slots", {}),
            unbound=d.get("unbound", []),
        )


def build_param_identity_map(namespace: dict[str, Any]) -> dict[int, str]:
    """Map ``id(parameter tensor)`` to a stable ``module.param`` name.

    Identity is what the optimizer-reference check compares, but ``id()`` values
    are meaningless across a boundary. Resolving each id to a name on both sides
    gives a comparable representation of the same relation.
    """
    import torch

    identity: dict[int, str] = {}
    for var_name, value in namespace.items():
        if var_name.startswith("_") or not isinstance(value, torch.nn.Module):
            continue
        for pname, p in value.named_parameters():
            identity.setdefault(id(p), f"{var_name}.{pname}")
        for bname, b in value.named_buffers():
            identity.setdefault(id(b), f"{var_name}.{bname}")
    return identity


def capture_module(module: Any) -> ModuleCapture:
    return ModuleCapture(
        parameters={n: _record(p) for n, p in module.named_parameters()},
        buffers={n: _record(b) for n, b in module.named_buffers()},
        training=bool(module.training),
    )


def capture_optimizer(opt: Any, identity: dict[int, str]) -> OptimizerCapture:
    import torch

    cap = OptimizerCapture(kind=type(opt).__name__)

    for gi, group in enumerate(opt.param_groups):
        hyper = {k: v for k, v in group.items() if k != "params"}
        # Hyperparameters are occasionally tensors (a per-group LR schedule);
        # reduce those to values so the record stays JSON-comparable.
        for k, v in list(hyper.items()):
            if isinstance(v, torch.Tensor):
                hyper[k] = v.detach().to("cpu").tolist()
        cap.groups.append(hyper)

        for si, p in enumerate(group["params"]):
            name = identity.get(id(p))
            key = f"g{gi}.s{si}"
            if name is None:
                # The slot points at a tensor no live module owns. That is the
                # detached-copy failure, and it must not be silently normalized.
                shape = tuple(getattr(p, "shape", ()))
                name = f"<unbound {shape}>"
                cap.unbound.append(key)
            cap.param_refs[key] = name

    # Slots are keyed by the optimizer's own slot position, not by the model
    # parameter name. Position is what stays comparable when references break:
    # keying by name would make every slot key change the moment the optimizer
    # detaches from the model, so the value row would fail for the same reason
    # the reference row does and neither would localize the fault. With
    # positional keys the two rows stay independent, which is the point of
    # having both.
    position: dict[int, str] = {}
    for gi, group in enumerate(opt.param_groups):
        for si, p in enumerate(group["params"]):
            position[id(p)] = f"g{gi}.s{si}"

    for p, state in opt.state.items():
        slot = position.get(id(p), "g?.s?")
        if not isinstance(state, dict):
            continue
        for skey, sval in state.items():
            full = f"{slot}/{skey}"
            if isinstance(sval, torch.Tensor):
                cap.slots[full] = _record(sval)
            elif isinstance(sval, (int, float, bool, str)) or sval is None:
                cap.scalar_slots[full] = sval
            else:
                cap.scalar_slots[full] = repr(sval)[:128]

    return cap


@dataclass
class StateDictCapture:
    """A generic ``state_dict()``-bearing object: LR scheduler, AMP scaler."""

    kind: str
    entries: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "entries": self.entries}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StateDictCapture":
        return cls(kind=d["kind"], entries=d.get("entries", {}))


def capture_state_dict(obj: Any) -> StateDictCapture:
    import torch

    entries: dict[str, Any] = {}
    try:
        raw = obj.state_dict()
    except Exception as exc:
        return StateDictCapture(kind=type(obj).__name__, entries={"<error>": repr(exc)})

    for k, v in raw.items():
        if isinstance(v, torch.Tensor):
            entries[k] = _record(v).to_dict()
        elif isinstance(v, (int, float, bool, str)) or v is None:
            entries[k] = v
        elif isinstance(v, (list, tuple)):
            entries[k] = [x if isinstance(x, (int, float, bool, str)) else repr(x)[:64] for x in v]
        else:
            entries[k] = repr(v)[:128]
    return StateDictCapture(kind=type(obj).__name__, entries=entries)


@dataclass
class LoaderCapture:
    kind: str
    batch_size: int | None = None
    drop_last: bool | None = None
    dataset_len: int | None = None
    sampler_kind: str | None = None
    generator_digest: str | None = None
    #: The next indices the sampler would emit, read non-destructively.
    next_indices: list[int] | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "dataset_len": self.dataset_len,
            "sampler_kind": self.sampler_kind,
            "generator_digest": self.generator_digest,
            "next_indices": self.next_indices,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LoaderCapture":
        return cls(**d)


def capture_loader(loader: Any, probe: int = 16) -> LoaderCapture:
    """Record where a data loader is positioned and what it will emit next.

    Sampling order is state. A boundary that restores the model perfectly and
    resets the shuffle to epoch zero has changed the experiment, and does so
    invisibly, since the loss curve stays smooth. Reading the next indices
    directly is the only check that catches it, so the sampler's generator is
    saved, advanced, and restored around the probe.
    """
    import hashlib as _hashlib

    import torch

    cap = LoaderCapture(kind=type(loader).__name__)
    try:
        cap.batch_size = getattr(loader, "batch_size", None)
        cap.drop_last = getattr(loader, "drop_last", None)
        ds = getattr(loader, "dataset", None)
        cap.dataset_len = len(ds) if ds is not None and hasattr(ds, "__len__") else None
        sampler = getattr(loader, "sampler", None)
        cap.sampler_kind = type(sampler).__name__ if sampler is not None else None
    except Exception as exc:
        cap.note = f"metadata unreadable: {exc!r}"
        return cap

    gen = getattr(loader, "generator", None)
    if gen is None:
        gen = getattr(getattr(loader, "sampler", None), "generator", None)

    saved = None
    if isinstance(gen, torch.Generator):
        try:
            saved = gen.get_state().clone()
            cap.generator_digest = _hashlib.sha256(
                saved.numpy().tobytes()
            ).hexdigest()[:32]
        except Exception:
            saved = None

    sampler = getattr(loader, "sampler", None)
    if sampler is not None:
        try:
            indices: list[int] = []
            for i, idx in enumerate(iter(sampler)):
                if i >= probe:
                    break
                indices.append(int(idx))
            cap.next_indices = indices
        except Exception as exc:
            cap.note = (cap.note + f" sampler probe failed: {exc!r}").strip()
        finally:
            if saved is not None and isinstance(gen, torch.Generator):
                gen.set_state(saved)
    else:
        cap.note = (cap.note + " no sampler to probe").strip()

    if cap.generator_digest is None and cap.next_indices is not None:
        # A sampler with no generator is either sequential (position is not
        # random state) or seeded globally. Say which, rather than implying the
        # order was verified when only its determinism was.
        cap.note = (
            cap.note + " sampler has no dedicated generator; order derives from"
            " global RNG or is sequential"
        ).strip()

    return cap
