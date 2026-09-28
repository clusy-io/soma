"""A small but realistic training namespace, and restores of varying fidelity.

The point of this module is to give the checks something to be right and wrong
about. ``build_namespace`` produces a namespace with every feature the report
has a row for, and the ``restore_*`` functions each damage exactly one of them,
so a test can assert that the corresponding row, and only that row, fails.
"""

from __future__ import annotations

import contextlib
import copy
import random
from typing import Any

import numpy as np
import torch
from torch import nn


@contextlib.contextmanager
def _rng_neutral():
    """Run a restore without perturbing the process's random state.

    These restores execute in the same interpreter as the source, which real
    ones do not: a real destination is a fresh process whose generators are
    installed from the checkpoint envelope. Constructing an ``nn.Module`` inside
    a restore therefore draws from the very generator the RNG rows are about,
    and the row would report the restore implementation's own consumption
    instead of the transition's fidelity.

    Almost every realistic restore path constructs a module, so without this the
    RNG rows would fail on faults that have nothing to do with randomness, and
    the harness would be measuring itself.
    """
    py = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class TinyNet(nn.Module):
    """Small, but with BatchNorm so there are real non-parameter buffers."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, width)
        self.bn = nn.BatchNorm1d(width)
        self.fc2 = nn.Linear(width, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.bn(self.fc1(x))))


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 256) -> None:
        g = torch.Generator().manual_seed(7)
        self.x = torch.randn(n, 8, generator=g)
        self.y = (self.x.sum(dim=1) > 0).long()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]


def build_namespace(*, steps: int = 3, seed: int = 1234) -> dict[str, Any]:
    """A namespace mid-training: warm optimizer, advanced RNG, moved loader."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = TinyNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    dataset = ToyDataset()
    loader_generator = torch.Generator().manual_seed(99)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, generator=loader_generator
    )
    rng = np.random.default_rng(seed)

    # An explicit alias: a view sharing storage with a base array. A serializer
    # that writes each name independently silently breaks this.
    features = np.arange(64, dtype=np.float64).reshape(8, 8)
    feature_view = features[2:5]

    loss_fn = nn.CrossEntropyLoss()
    batches = iter(loader)
    for _ in range(steps):
        try:
            xb, yb = next(batches)
        except StopIteration:
            batches = iter(loader)
            xb, yb = next(batches)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Advance every stream so a reset is detectable rather than coincidental.
    random.random()
    np.random.random()
    torch.rand(1)
    rng.random()

    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "loader": loader,
        "dataset": dataset,
        "rng": rng,
        "features": features,
        "feature_view": feature_view,
        "loss_fn": loss_fn,
        "notes": "mid-training namespace",
    }


def step(namespace: dict[str, Any], index: int) -> float:
    """One training step, used as the continuation probe."""
    model = namespace["model"]
    optimizer = namespace["optimizer"]
    loss_fn = namespace["loss_fn"]
    dataset = namespace["dataset"]

    # A fixed slice keeps the probe independent of loader position, which has
    # its own row; this row is about whether the restored state computes the
    # same update given the same input.
    start = (index * 16) % (len(dataset) - 16)
    xb = dataset.x[start:start + 16]
    yb = dataset.y[start:start + 16]

    optimizer.zero_grad()
    loss = loss_fn(model(xb), yb)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


# --------------------------------------------------------------------------
# Restores. Each one is what some plausible implementation would actually do.
# --------------------------------------------------------------------------


def restore_faithful(source: dict[str, Any]) -> dict[str, Any]:
    """A correct transfer: the namespace moves as one unit, aliasing included.

    ``deepcopy`` gets most of the way there, and it does preserve torch view
    relationships, but it does not preserve numpy ones: ``ndarray.__deepcopy__``
    returns an independent copy, so a base array and a slice of it arrive as two
    unrelated arrays. A restore that is genuinely co-variable-aware has to
    re-derive the view from the copied base, which is what this does.

    That extra step is the whole point of the co-variable concept: an aliasing
    component has to be rebuilt as a component, not reassembled from members
    that were each transferred correctly on their own.
    """
    with _rng_neutral():
        ns = copy.deepcopy(
            {k: v for k, v in source.items() if k not in ("loader", "dataset")}
        )
    # Re-derive the view against the copied base so the component survives.
    ns["feature_view"] = ns["features"][2:5]

    # DataLoader holds a worker context that does not deepcopy cleanly, so it is
    # rebuilt around the copied dataset with the generator state carried over.
    dataset = copy.deepcopy(source["dataset"])
    gen = torch.Generator()
    gen.set_state(source["loader"].generator.get_state().clone())
    ns["dataset"] = dataset
    ns["loader"] = torch.utils.data.DataLoader(
        dataset, batch_size=source["loader"].batch_size, shuffle=True, generator=gen
    )
    return ns


def restore_dill_session(source: dict[str, Any]) -> dict[str, Any]:
    """What Clusy's checkpoint actually does: plain-pickle the whole namespace.

    ``dill.dump_session`` serializes each object through the ordinary pickle
    protocol. The pickle memo preserves object *identity*, so two names bound to
    the same tensor still arrive as one object. It does not preserve *storage
    sharing* between distinct objects, so a base array and a view of it arrive
    as two independent buffers, and a write through one is no longer visible
    through the other.

    This restore is included so the alias row has a realistic failure to detect
    and not only a synthetic one.
    """
    import dill

    payload = {
        k: v for k, v in source.items()
        if k not in ("loader",)  # DataLoader worker state is not picklable here
    }
    ns = dill.loads(dill.dumps(payload))
    gen = torch.Generator()
    gen.set_state(source["loader"].generator.get_state().clone())
    ns["loader"] = torch.utils.data.DataLoader(
        ns["dataset"], batch_size=source["loader"].batch_size, shuffle=True, generator=gen
    )
    return ns


def restore_detached_optimizer(source: dict[str, Any]) -> dict[str, Any]:
    """Rebuild model and optimizer separately: values equal, references broken.

    This is the failure the ``optimizer refs`` row exists for. Every parameter
    compares equal, training runs without error, and the optimizer updates
    tensors the model does not use, so the model never improves.
    """
    ns = restore_faithful(source)
    with _rng_neutral():
        fresh = TinyNet()
        fresh.load_state_dict(ns["model"].state_dict())
        opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3, weight_decay=0.01)
        opt.load_state_dict(ns["optimizer"].state_dict())
    # The namespace keeps the ORIGINAL model, while the optimizer now points at
    # the clone's parameters.
    ns["optimizer"] = opt
    return ns


def restore_reset_rng(source: dict[str, Any]) -> dict[str, Any]:
    """Everything transfers, but the destination reseeds on start."""
    ns = restore_faithful(source)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    return ns


def restore_broken_aliasing(source: dict[str, Any]) -> dict[str, Any]:
    """Each name serialized independently, so the view stops sharing storage."""
    ns = restore_faithful(source)
    ns["features"] = np.array(source["features"], copy=True)
    ns["feature_view"] = np.array(source["feature_view"], copy=True)
    return ns


def restore_dropped_optimizer_state(source: dict[str, Any]) -> dict[str, Any]:
    """Parameters transfer; Adam's moment estimates silently restart."""
    ns = restore_faithful(source)
    with _rng_neutral():
        fresh = torch.optim.AdamW(ns["model"].parameters(), lr=1e-3, weight_decay=0.01)
    ns["optimizer"] = fresh
    return ns


def restore_reset_loader(source: dict[str, Any]) -> dict[str, Any]:
    """Model state is perfect; the shuffle order restarts from epoch zero."""
    ns = restore_faithful(source)
    gen = torch.Generator().manual_seed(99)
    ns["loader"] = torch.utils.data.DataLoader(
        ns["dataset"], batch_size=16, shuffle=True, generator=gen
    )
    return ns


def restore_dropped_buffers(source: dict[str, Any]) -> dict[str, Any]:
    """BatchNorm running statistics reset while weights transfer correctly."""
    ns = restore_faithful(source)
    with _rng_neutral():
        ns["model"].bn.reset_running_stats()
    return ns
