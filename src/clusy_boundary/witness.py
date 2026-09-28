"""A witness: everything about a namespace that a boundary must preserve.

A witness is taken twice, on either side of a transition, and the two are
compared. It is deliberately a plain, JSON-serializable record rather than a
handle to live objects, because the two sides of a boundary are usually two
processes on two machines, and often two different kinds of hardware.

Discovery is by inspection of the namespace, so the harness works on a kernel
it did not write. What it cannot find, it says it could not find.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import platform
import socket
import sys
from dataclasses import dataclass, field
from typing import Any

from .aliasgraph import AliasSnapshot, TraversalBudget, capture_alias_snapshot
from .classidentity import capture_class_identity
from .rng import RngCapture, capture_rng
from .torchstate import (
    LoaderCapture,
    ModuleCapture,
    OptimizerCapture,
    StateDictCapture,
    build_param_identity_map,
    capture_loader,
    capture_module,
    capture_optimizer,
    capture_state_dict,
)

WITNESS_FORMAT = 1

_WORKSPACE_IGNORE = (
    "*.pyc", "__pycache__/*", ".git/*", ".ipynb_checkpoints/*",
    "*.lock", ".venv/*", "venv/*", "*.log", ".DS_Store",
)


@dataclass
class HostInfo:
    hostname: str
    platform: str
    python: str
    torch: str | None
    cuda_available: bool
    cuda_devices: list[str] = field(default_factory=list)

    @property
    def device_class(self) -> str:
        return "cuda" if self.cuda_available else "cpu"

    def to_dict(self) -> dict[str, Any]:
        return {
            "hostname": self.hostname,
            "platform": self.platform,
            "python": self.python,
            "torch": self.torch,
            "cuda_available": self.cuda_available,
            "cuda_devices": self.cuda_devices,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HostInfo":
        return cls(
            hostname=d["hostname"],
            platform=d["platform"],
            python=d["python"],
            torch=d.get("torch"),
            cuda_available=d.get("cuda_available", False),
            cuda_devices=d.get("cuda_devices", []),
        )


def capture_host() -> HostInfo:
    torch_version: str | None = None
    cuda_available = False
    devices: list[str] = []
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            devices = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except Exception:
        pass
    return HostInfo(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python=sys.version.split()[0],
        torch=torch_version,
        cuda_available=cuda_available,
        cuda_devices=devices,
    )


def hash_workspace(root: str, ignore: tuple[str, ...] = _WORKSPACE_IGNORE) -> dict[str, str]:
    """Digest every file under ``root``, keyed by path relative to it."""
    out: dict[str, str] = {}
    if not root or not os.path.isdir(root):
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if not any(fnmatch.fnmatch(d + "/", pat) or fnmatch.fnmatch(d, pat.rstrip("/*"))
                       for pat in ignore)
        ]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(fn, pat) for pat in ignore):
                continue
            try:
                h = hashlib.sha256()
                with open(full, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                out[rel] = h.hexdigest()[:32]
            except OSError:
                out[rel] = "unreadable"
    return out


@dataclass
class Witness:
    format: int
    label: str
    host: HostInfo
    modules: dict[str, ModuleCapture] = field(default_factory=dict)
    optimizers: dict[str, OptimizerCapture] = field(default_factory=dict)
    schedulers: dict[str, StateDictCapture] = field(default_factory=dict)
    scalers: dict[str, StateDictCapture] = field(default_factory=dict)
    loaders: dict[str, LoaderCapture] = field(default_factory=dict)
    rng: RngCapture | None = None
    alias_components: list[list[str]] = field(default_factory=list)
    alias_truncated: list[str] = field(default_factory=list)
    workspace: dict[str, str] = field(default_factory=dict)
    workspace_root: str | None = None
    #: Loss values from a fixed continuation, filled by continuation.py.
    continuation: dict[str, Any] | None = None
    #: Per-name class path and whether that class resolves by reference.
    class_identity: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Top-level names present, so a dropped variable is visible even if it
    #: belonged to no recognized category.
    names: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "label": self.label,
            "host": self.host.to_dict(),
            "modules": {k: v.to_dict() for k, v in self.modules.items()},
            "optimizers": {k: v.to_dict() for k, v in self.optimizers.items()},
            "schedulers": {k: v.to_dict() for k, v in self.schedulers.items()},
            "scalers": {k: v.to_dict() for k, v in self.scalers.items()},
            "loaders": {k: v.to_dict() for k, v in self.loaders.items()},
            "rng": self.rng.to_dict() if self.rng else None,
            "alias_components": self.alias_components,
            "alias_truncated": self.alias_truncated,
            "workspace": self.workspace,
            "workspace_root": self.workspace_root,
            "continuation": self.continuation,
            "class_identity": self.class_identity,
            "names": self.names,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Witness":
        if d.get("format") != WITNESS_FORMAT:
            raise ValueError(
                f"witness format {d.get('format')} is not supported "
                f"(this build reads format {WITNESS_FORMAT})"
            )
        return cls(
            format=d["format"],
            label=d["label"],
            host=HostInfo.from_dict(d["host"]),
            modules={k: ModuleCapture.from_dict(v) for k, v in d.get("modules", {}).items()},
            optimizers={
                k: OptimizerCapture.from_dict(v) for k, v in d.get("optimizers", {}).items()
            },
            schedulers={
                k: StateDictCapture.from_dict(v) for k, v in d.get("schedulers", {}).items()
            },
            scalers={k: StateDictCapture.from_dict(v) for k, v in d.get("scalers", {}).items()},
            loaders={k: LoaderCapture.from_dict(v) for k, v in d.get("loaders", {}).items()},
            rng=RngCapture.from_dict(d["rng"]) if d.get("rng") else None,
            alias_components=d.get("alias_components", []),
            alias_truncated=d.get("alias_truncated", []),
            workspace=d.get("workspace", {}),
            workspace_root=d.get("workspace_root"),
            continuation=d.get("continuation"),
            class_identity=d.get("class_identity", {}),
            names=d.get("names", []),
            notes=d.get("notes", []),
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True, default=str)

    @classmethod
    def load(cls, path: str) -> "Witness":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


def _is_scheduler(obj: Any) -> bool:
    try:
        from torch.optim.lr_scheduler import LRScheduler  # torch >= 2.0

        if isinstance(obj, LRScheduler):
            return True
    except Exception:
        pass
    # ReduceLROnPlateau is not an LRScheduler subclass in several torch releases,
    # and it holds real state (best, num_bad_epochs) that a boundary can drop.
    return type(obj).__name__ in {"ReduceLROnPlateau", "LambdaLR", "_LRScheduler"}


def _is_scaler(obj: Any) -> bool:
    return type(obj).__name__ == "GradScaler"


def capture_witness(
    namespace: dict[str, Any],
    *,
    label: str,
    workspace_root: str | None = None,
    alias_budget: TraversalBudget | None = None,
) -> Witness:
    """Take a full witness of ``namespace``."""
    host = capture_host()
    w = Witness(format=WITNESS_FORMAT, label=label, host=host)
    w.names = sorted(n for n in namespace if not n.startswith("_"))

    try:
        import torch  # noqa: F401
    except ImportError:
        w.notes.append("torch not importable: module, optimizer and loader rows are n/a")
        w.rng = capture_rng(namespace)
        w.class_identity = capture_class_identity(namespace)
        alias = capture_alias_snapshot(namespace, budget=alias_budget)
        w.alias_components = alias.components
        w.alias_truncated = sorted(alias.truncated_names)
        if workspace_root:
            w.workspace_root = workspace_root
            w.workspace = hash_workspace(workspace_root)
        return w

    import torch

    identity = build_param_identity_map(namespace)

    for name, value in namespace.items():
        if name.startswith("_"):
            continue
        try:
            if isinstance(value, torch.nn.Module):
                w.modules[name] = capture_module(value)
            elif isinstance(value, torch.optim.Optimizer):
                w.optimizers[name] = capture_optimizer(value, identity)
            elif _is_scaler(value):
                w.scalers[name] = capture_state_dict(value)
            elif _is_scheduler(value):
                w.schedulers[name] = capture_state_dict(value)
            elif isinstance(value, torch.utils.data.DataLoader):
                w.loaders[name] = capture_loader(value)
        except Exception as exc:
            w.notes.append(f"capture of {name!r} failed: {exc!r}")

    w.rng = capture_rng(namespace)
    w.class_identity = capture_class_identity(namespace)

    alias = capture_alias_snapshot(namespace, budget=alias_budget)
    w.alias_components = alias.components
    w.alias_truncated = sorted(alias.truncated_names)

    if workspace_root:
        w.workspace_root = workspace_root
        w.workspace = hash_workspace(workspace_root)

    return w


def alias_snapshot_from_witness(w: Witness) -> AliasSnapshot:
    return AliasSnapshot(
        partition=frozenset(frozenset(c) for c in w.alias_components),
        truncated_names=frozenset(w.alias_truncated),
    )
