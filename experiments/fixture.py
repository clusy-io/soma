"""The state fixture, partitioned into what crosses and what is probed.

This is the single most important file for Table 1 succeeding, because the
natural way to build the fixture makes every transition fail.

§5.1 describes one fixture containing "both supported and deliberately
unsupported constructs", including open files. Built literally, that namespace
cannot cross *any* boundary: an open file handle pickles cleanly and then raises
during ``load_session``, and because the capsule is all-or-nothing it takes the
entire namespace with it. The exclusion probe does not save it — that probe runs
only after a dump *failure*, and this dump succeeds. Two such restores then
retire the checkpoint via the unrestorable fence, so retrying does not help
either. The result is 0/3 in every cell, for a reason that says nothing about
the system.

So the fixture is split in two, and the split is not a convenience — it *is* the
portability contract from §3.1 expressed as code:

``TRANSPORTED``
    Constructs measured as class A in E1: they cross a boundary and remain
    semantically correct. This namespace is what a transition actually moves,
    and every one of its members is expected to arrive intact. A failure here
    is a real finding.

``PROBED_IN_ISOLATION``
    Constructs measured as B, C or D. Each is exercised in its own capsule, the
    way ``survivability_probe.py`` already does, so that a construct which is
    *designed* to be unsupported cannot destroy the measurement of one that is
    supported. Their results belong in Figure 2, not Table 1.

Putting an unsupported construct into a transported namespace does not measure
"does the system handle unsupported state". It measures "does an all-or-nothing
capsule survive a poison object", and the answer is already known to be no.
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable

# Constructs measured as class A by an earlier survivability probe on
# torch 2.8.0 / numpy 2.4.6, through the platform's capsule dump path.
TRANSPORTED_CONSTRUCTS = (
    "scalars int/float/str/bool",
    "containers list/dict/set/tuple",
    "shared reference (two names, one object)",
    "custom class instance",
    "closure over a captured variable",
    "lambda with a default argument",
    "imported module object",
    "iterator over a container",
    "self-referential structure",
    "ndarray",
    "numpy Generator (PCG64)",
    "CPU tensor",
    "nn.Module with running stats",
    "optimizer state and a real training step",
    "LR scheduler",
    "AMP GradScaler",
    "DataLoader sampler position",
    "custom nn.Module subclass with a buffer",
)

# Measured B, C or D. Probed one capsule at a time; never transported together.
PROBED_IN_ISOLATION = (
    "mutated module global",          # B
    "sys.path modification",          # B
    "current working directory",      # B
    "environment variable",           # B
    "logging configuration",          # B
    "running thread",                 # B
    "subprocess handle",              # B
    "ndarray view sharing a buffer",  # B
    "tensor view sharing storage",    # B
    "legacy global numpy RNG",        # B
    "torch CPU RNG",                  # B
    "bound UDP socket",               # C
    "mmap region",                    # C
    "numpy memmap",                   # C
    "multiprocessing Process handle", # C
    "generator mid-suspension",       # C
    "open file handle and position",  # D — the poison object
    "tempfile handle",                # D
)


@dataclass
class FixtureSpec:
    """How to build one transported namespace."""

    seed: int = 20260825
    #: Target capsule size for the E4 sweep. None means "whatever it comes to".
    padding_bytes: int | None = None
    #: 'cpu' or 'cuda'. Tensors are created on this device.
    device: str = "cpu"
    #: Include workload state (model, optimizer, scheduler, loader).
    include_workload: bool = True
    #: Small enough to train on CPU in seconds; we are not benchmarking ML.
    workload_width: int = 32

    #: Which model the workload builds.
    #:
    #: ``tinyblock`` is the synthetic default: seconds to train on CPU, no
    #: download, and it exercises every construct the oracles check.
    #: ``resnet18`` is the paper's named workload, on real CIFAR-10, which
    #: additionally exercises the workspace layer at true scale.
    workload: str = "tinyblock"

    #: Where the raw dataset archive is cached. Deliberately OUTSIDE the project
    #: workspace, because the workspace is flushed to the vault on every
    #: transition and a 170 MB archive riding each of 48 rows would inflate
    #: latency and transfer cost for a reason unrelated to namespace
    #: materialization — and would quietly make the size sweep measure the
    #: dataset rather than the capsule.
    data_root: str = "/tmp/clusy-cifar10"

    #: Bytes of REAL FILES to materialise INSIDE the workspace.
    #:
    #: The workspace column of Table 1 checks that files reconstruct with their
    #: hashes, sizes and modes intact. That needs real files at a realistic
    #: count and size; it does not need the whole archive. So a subset is
    #: written into the workspace as actual files, and the archive stays
    #: outside. This is what makes the workspace layer genuinely exercised
    #: without paying 170 MB per transition.
    workspace_corpus_bytes: int = 8 * 1024 * 1024
    #: Spread across this many files, so directory structure, per-file modes and
    #: a realistic file count are all exercised rather than one large blob.
    workspace_corpus_files: int = 64
    #: Root of the corpus inside the workspace.
    workspace_root: str = "data"

    #: Use the real CIFAR-10 archive rather than a deterministic substitute of
    #: the same shape. Off by default because a cold sandbox pays the download
    #: on every row, and the workspace-layer claim is carried by the corpus
    #: above rather than by the archive. When it is on and the download fails,
    #: the fixture RAISES rather than substituting: a silent fallback would let
    #: the paper say CIFAR-10 about a run that measured something else.
    use_real_cifar10: bool = False
    #: How many CIFAR-10 examples to keep. The workload tests reconstruction,
    #: not accuracy, so a small subset trains in seconds and still gives a real
    #: dataloader with a real sampler position.
    workload_samples: int = 512


#: Where the platform scopes its workspace flush (the platform flushes only
#: files under /home/user). Files outside this tree are not transported, so
#: the corpus must be written inside it.
WORKSPACE_DIR = "/home/user"


def _write_workspace_corpus(spec: "FixtureSpec") -> dict[str, Any]:
    """Materialise real files inside the workspace, and describe them.

    Table 1's workspace column asserts that files reconstruct with their bytes,
    sizes and modes intact. That is a claim about the file layer, and it needs
    real files at a realistic count — it does not need the whole dataset
    archive, which is why the archive stays outside the workspace and only this
    corpus rides each transition.

    Deterministic from the seed, so the same fixture produces byte-identical
    files on both sides of a boundary and a hash difference means the transfer
    changed something rather than that the generator did.
    """
    import numpy as np

    # The flush is scoped to /home/user (the platform flushes only files under
    # /home/user). A relative
    # root resolved against the process CWD is a coin flip: if the kernel's cwd
    # is anywhere else, the corpus is written OUTSIDE the flush scope, never
    # rides the transition, and the workspace oracle reports a total loss that
    # is a harness artifact rather than a system finding. Anchor it explicitly,
    # and fall back to the cwd only when that directory does not exist, which
    # is the local-test case.
    root = spec.workspace_root
    if not os.path.isabs(root):
        anchor = WORKSPACE_DIR if os.path.isdir(WORKSPACE_DIR) else os.getcwd()
        root = os.path.join(anchor, root)
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(spec.seed)
    per_file = max(1, spec.workspace_corpus_bytes // max(1, spec.workspace_corpus_files))

    manifest: dict[str, dict[str, Any]] = {}
    for i in range(spec.workspace_corpus_files):
        # A nested layout so directory creation order and per-directory modes
        # are exercised, not just a flat listing.
        sub = os.path.join(root, f"shard{i % 8:02d}")
        os.makedirs(sub, exist_ok=True)
        path = os.path.join(sub, f"part{i:03d}.bin")
        payload = rng.integers(0, 256, size=per_file, dtype=np.uint8).tobytes()
        with open(path, "wb") as fh:
            fh.write(payload)
        # Vary modes deliberately: the restore is supposed to preserve them, and
        # a corpus that is uniformly 0644 cannot show that it did.
        mode = 0o640 if i % 3 == 0 else 0o644
        os.chmod(path, mode)
        rel = os.path.relpath(path, root)
        manifest[rel] = {
            "sha256": hashlib.sha256(payload).hexdigest()[:32],
            "size": len(payload),
            "mode": oct(mode),
        }

    return {
        "workspace_root": root,
        "workspace_manifest": manifest,
        "workspace_total_bytes": sum(v["size"] for v in manifest.values()),
    }


def _build_resnet18_workload(spec: "FixtureSpec", dev: Any) -> dict[str, Any]:
    """ResNet-18 mid-training, with a real file corpus in the workspace.

    Two things the paper's named workload is meant to supply, and they are
    separable: a real model with real optimizer and scheduler state and real
    training progress, and a workspace that actually holds files.

    The model half never needs a download. The data half does, and on a cold
    sandbox the CIFAR-10 archive is minutes of wall clock on every one of the
    48 rows. So the archive is used when it is already cached, and a
    deterministic tensor set of the same shape is used when it is not — and
    which one was used is RECORDED as ``data_source``, because a paper that
    says CIFAR-10 while the run measured something else would be false in a way
    no reader could detect.
    """
    import numpy as np  # noqa: F401
    import torch
    from torch import nn

    try:
        from torchvision.models import resnet18
    except ImportError as exc:  # pragma: no cover - depends on the image
        raise RuntimeError(
            "the resnet18 workload needs torchvision, which this image does not "
            "have. Use sandbox_type='ml' at both ends, or workload='tinyblock'. "
            "Falling back silently would measure a different model than the one "
            "the paper names."
        ) from exc

    data_source = "synthetic-cifar-shaped"
    train_x = train_y = None
    if spec.use_real_cifar10:
        try:
            from torchvision import transforms
            from torchvision.datasets import CIFAR10

            os.makedirs(spec.data_root, exist_ok=True)
            tfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ])
            full = CIFAR10(root=spec.data_root, train=True, download=True, transform=tfm)
            n = min(spec.workload_samples, len(full))
            xs = torch.stack([full[i][0] for i in range(n)])
            ys = torch.tensor([full[i][1] for i in range(n)])
            train_x, train_y = xs.to(dev), ys.to(dev)
            data_source = "cifar10"
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"real CIFAR-10 was requested and could not be obtained ({exc!r}). "
                f"Refusing to silently substitute synthetic data: set "
                f"use_real_cifar10=False to ask for the substitute explicitly."
            ) from exc

    if train_x is None:
        # CIFAR-10 shaped: 3x32x32, ten classes, normalised. Deterministic from
        # the seed so both sides of a boundary agree bit for bit.
        g = torch.Generator().manual_seed(spec.seed)
        n = spec.workload_samples
        train_x = torch.randn(n, 3, 32, 32, generator=g).to(dev)
        train_y = torch.randint(0, 10, (n,), generator=g).to(dev)

    model = resnet18(num_classes=10).to(dev)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss_fn = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(train_x.cpu(), train_y.cpu())
    gen = torch.Generator()
    gen.manual_seed(99)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, generator=gen, num_workers=0
    )

    # A fixed batch, so the forward oracle does not depend on loader position,
    # which has an oracle of its own.
    batch_x, batch_y = train_x[:32], train_y[:32]
    for _ in range(3):
        optimizer.zero_grad()
        loss_fn(model(batch_x), batch_y).backward()
        optimizer.step()
        scheduler.step()

    out: dict[str, Any] = {
        "model": model, "optimizer": optimizer, "scheduler": scheduler,
        "scaler": scaler, "loader": loader, "loss_fn": loss_fn,
        "train_x": batch_x, "train_y": batch_y, "step_count": 3,
        "t_cpu": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "workload_kind": "resnet18",
        "data_source": data_source,
    }
    out.update(_write_workspace_corpus(spec))
    return out


def build_transported_namespace(spec: FixtureSpec) -> dict[str, Any]:
    """Build a namespace containing only constructs measured as portable.

    Everything here is expected to arrive intact. If something in this namespace
    fails to reconstruct, that is a result worth reporting, not a fixture bug.
    """
    import numpy as np
    import torch
    from torch import nn

    random.seed(spec.seed)
    np.random.seed(spec.seed % (2**32 - 1))
    torch.manual_seed(spec.seed)

    dev = torch.device(spec.device)
    ns: dict[str, Any] = {}

    # -- plain Python ----------------------------------------------------
    ns["sc_int"] = 2**91 + 7
    ns["sc_float"] = 0.1 + 0.2
    ns["sc_str"] = "héllo wörld"
    ns["sc_bool"] = True
    ns["co_list"] = [1, "a", (2, 3), None]
    ns["co_dict"] = {"b": 2, "a": 1, 3: "three"}
    ns["co_set"] = {"x", "y", "z"}
    ns["co_tuple"] = (1, [2], {"k": "v"})

    shared = {"n": 1}
    ns["shared_a"] = shared
    ns["shared_b"] = shared            # same object; the memo preserves this

    recur: dict[str, Any] = {"name": "root"}
    recur["self"] = recur
    ns["recur"] = recur

    ns["it"] = iter([10, 20, 30, 40])
    next(ns["it"])

    def _make(factor: int) -> Callable[[int], int]:
        def inner(x: int) -> int:
            return x * factor
        return inner

    ns["closure_fn"] = _make(7)
    ns["lam"] = lambda x, y=3: x**y

    import json as _json

    ns["mod_ref"] = _json

    # -- numpy -----------------------------------------------------------
    ns["arr"] = np.arange(24, dtype=np.float64).reshape(4, 6)
    ns["np_gen"] = np.random.default_rng(spec.seed)
    ns["np_gen"].random(5)

    # -- torch workload --------------------------------------------------
    if spec.include_workload and spec.workload == "resnet18":
        ns.update(_build_resnet18_workload(spec, dev))
    elif spec.include_workload:
        # Stated for the same reason the resnet branch states it: a row whose
        # provenance keys are absent is indistinguishable from a row emitted by
        # a build that predates them, and the two mean different things.
        ns["workload_kind"] = "tinyblock"
        ns["data_source"] = "synthetic"
        # The corpus is not part of the MODEL, it is part of the transition:
        # namespace and workspace travel by different transports and a size
        # sweep that carries only the namespace would leave the workspace path
        # untested at every rung. It is a fixed 8 MiB, which is why the size
        # ladder compensates for the fixture floor rather than ignoring it.
        ns.update(_write_workspace_corpus(spec))

        w = spec.workload_width

        class TinyBlock(nn.Module):
            """A custom subclass with a registered buffer, as E1 measured."""

            def __init__(self, width: int) -> None:
                super().__init__()
                self.fc1 = nn.Linear(16, width)
                self.bn = nn.BatchNorm1d(width)
                self.fc2 = nn.Linear(width, 4)
                self.register_buffer("forward_calls", torch.zeros(1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.forward_calls += 1
                return self.fc2(torch.relu(self.bn(self.fc1(x))))

        model = TinyBlock(w).to(dev)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        # GradScaler is constructed for the DESTINATION-independent case: it is
        # disabled, so it carries state without requiring CUDA to be present.
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        xb = torch.randn(32, 16, device=dev)
        yb = torch.randn(32, 4, device=dev)
        loss_fn = nn.MSELoss()
        for _ in range(3):
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()
            scheduler.step()

        class IndexDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return 128

            def __getitem__(self, i: int) -> int:
                return i

        gen = torch.Generator()
        gen.manual_seed(99)
        loader = torch.utils.data.DataLoader(
            IndexDataset(), batch_size=8, shuffle=True, generator=gen
        )

        ns.update(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            loader=loader, loss_fn=loss_fn, train_x=xb, train_y=yb,
            t_cpu=torch.arange(12, dtype=torch.float32).reshape(3, 4),
        )
        ns["step_count"] = 3

    # -- optional padding, for the size sweep ----------------------------
    if spec.padding_bytes:
        from .sizes import validate_size

        validate_size(spec.padding_bytes, label="fixture padding")
        # float64 so the payload is incompressible-ish and the size is honest.
        n = max(1, spec.padding_bytes // 8)
        ns["payload"] = np.random.default_rng(spec.seed).random(n)

    return ns


@dataclass
class Expectation:
    """What the destination must reproduce, computed on the source."""

    values: dict[str, str] = field(default_factory=dict)
    step_count: int | None = None
    next_loader_indices: list[int] | None = None
    scheduler_lr: float | None = None
    scheduler_epoch: int | None = None
    forward_output: float | None = None
    param_digest: str | None = None
    device_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": self.values,
            "step_count": self.step_count,
            "next_loader_indices": self.next_loader_indices,
            "scheduler_lr": self.scheduler_lr,
            "scheduler_epoch": self.scheduler_epoch,
            "forward_output": self.forward_output,
            "param_digest": self.param_digest,
            "device_type": self.device_type,
        }


def _digest_params(model: Any) -> str:
    import torch

    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().to("cpu", copy=True).contiguous().numpy().tobytes())
    for name, b in sorted(model.named_buffers()):
        h.update(name.encode())
        t = b.detach().to("cpu", copy=True).contiguous()
        h.update(t.numpy().tobytes() if t.numel() else b"empty")
    return h.hexdigest()[:32]


def compute_expectation(ns: dict[str, Any]) -> Expectation:
    """Record, on the source, what the destination will be checked against."""
    import numpy as np
    import torch

    exp = Expectation()
    exp.values = {
        "sc_int": repr(ns["sc_int"]),
        "sc_float": repr(ns["sc_float"]),
        "sc_str": ns["sc_str"],
        "co_dict": repr(sorted(ns["co_dict"].items(), key=repr)),
        "arr_sum": repr(float(ns["arr"].sum())),
    }

    model = ns.get("model")
    if model is not None:
        exp.param_digest = _digest_params(model)
        exp.device_type = next(model.parameters()).device.type
        exp.step_count = ns.get("step_count")
        with torch.no_grad():
            was = model.training
            model.eval()          # eval() so BatchNorm does not update on probe
            exp.forward_output = float(model(ns["train_x"]).sum())
            model.train(was)

    sched = ns.get("scheduler")
    opt = ns.get("optimizer")
    if sched is not None and opt is not None:
        exp.scheduler_lr = float(opt.param_groups[0]["lr"])
        exp.scheduler_epoch = int(sched.last_epoch)

    loader = ns.get("loader")
    if loader is not None:
        gen = loader.generator
        saved = gen.get_state().clone() if gen is not None else None
        try:
            exp.next_loader_indices = [int(i) for i in list(iter(loader.sampler))[:8]]
        finally:
            if saved is not None:
                gen.set_state(saved)

    return exp
