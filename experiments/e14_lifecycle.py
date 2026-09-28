"""E14: a resource lifecycle (CPU prep, GPU train, CPU analysis) under executed arms.

The job has three phases. PREP builds ResNet-18 and a SYNTHETIC CIFAR-shaped
dataset (the fixture's `use_real_cifar10` is off, so the data are Gaussian
inputs with random labels, recorded as `data_source`), iterates the shuffling
loader once, and derives preparation artefacts: per-channel statistics, a
horizontally flipped cache with aligned labels, and a feature bank from
`--prep-passes` passes of the untrained backbone. TRAIN runs `--steps`
optimizer steps on the accelerator with the LR scheduler stepped once per
completed epoch. ANALYSE computes plain accuracy and mean loss, then metrics
that genuinely CONSUME the prep artefacts (test-time-augmentation accuracy from
the cache, representation shift against the feature bank, and an input
standardisation check against the statistics).

Arms, all executed end to end:

  uninterrupted       the whole job on cpu
  always              the whole job on the GPU profile (default gpu_t4)
  switch              one project, PATCH cpu -> GPU -> cpu; the namespace,
                      files and training state follow the project
  restart             a fresh project per phase, nothing carried; each phase
                      re-runs what it depends on
  appckpt_complete    the best-effort application checkpoint: model weights
                      and per-module training flags, optimizer, scheduler,
                      step count, the global RNG streams (CUDA when saved on a
                      GPU runtime), the loader generator and every prep
                      artefact, moved between runtimes through the user's own
                      object storage
  appckpt_incomplete  the earlier "competent" checkpoint (weights, optimizer,
                      scheduler, step count, three RNG streams, map_location)
                      with no prep artefacts, no loader generator and no model
                      mode: the demonstration of omitted dependencies
  appckpt_naive       the incomplete checkpoint without map_location; expected
                      to fail loudly when a GPU-saved file is opened on CPU

What each arm records:

  * TRAIN-input and TRAIN-output fingerprints (params and buffers, optimizer
    state, scheduler, loader generator, global RNG streams, per-module
    training flags, step counters), so the analysis can say which arms trained
    from identical inputs and which produced identical outputs, instead of
    attributing an accuracy difference to a guess.
  * The same fingerprint at the END of PREP, digests of every prep artefact
    there and again at the top of ANALYSE, and the host PREP ran on. Carry
    fidelity is judged WITHIN an arm (TRAIN input against the arm's own PREP
    output; ANALYSE's artefacts against PREP's), because across arms the
    PREPs ran in different sandboxes and the fixture's CPU arithmetic depends
    on the host's thread count and ISA.
  * The lr at every step, so a reader can see the schedule is per epoch.
  * The application code each checkpoint arm needs, COUNTED from the marked
    blocks in this file (non-blank, non-comment lines), never hard-coded.
  * Estimated runtime cost, never "billing": each runtime is charged by one
    stated rule (CLOCK_RULE) at list rates whose source is recorded with them.

Only switching carries state without application code; the uninterrupted arms
also recompute nothing and need no checkpoint code, because they never move.
The comparison is therefore about what a MOVE costs, not about the arms that
avoid moving.

Dry run: `--local` drives the same programs through `experiments/localapi.py`
(persistent local kernels, simulated switch) and writes to a scratch
directory, never to `results/`. `--local-break rng_restore|rng_perturb` breaks
the simulated switch on purpose so the fingerprint checks can be seen to fail.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
import tempfile
import textwrap
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from handoff.controller import Api, MARK, FIXTURE_SRC  # noqa: E402
import runmeta  # noqa: E402

SCHEMA = "e14/2"
SEED = 20260925
STEPS = 120
PREP_PASSES = 3
CKPT_PATH = "data/app_checkpoint.pt"
ARMS = ("uninterrupted", "always", "switch", "restart",
        "appckpt_complete", "appckpt_incomplete", "appckpt_naive")
APP_ARMS = {"appckpt_complete": "complete", "appckpt_incomplete": "incomplete", "appckpt_naive": "naive"}
PREP_ARTEFACTS = ("prep_mean", "prep_std", "aug_cache", "prep_labels", "prep_features", "prep_seconds")

# The job's own construction parameters. PREP builds the namespace from them,
# and the application checkpoint's load must rebuild the same objects before
# it can load state into them, so both receive this one dict.
SPEC_KW = {"seed": SEED, "workload": "resnet18", "device": "cpu",
           "workspace_corpus_files": 4, "workspace_corpus_bytes": 16 * 1024}

# ---- estimated runtime cost -----------------------------------------------
# Two rate tables, both recorded in every record so a reader can recompute.
# The per-SKU SECONDS are the measurement; a rate table only prices them.
#
# `server_list_price` prices the exact sandboxes these projects run on at
# provider list prices: the E2B template used is 8 vCPU / 8 GiB, and a Modal
# GPU sandbox pays GPU + memory + CPU cores additively at the profile's
# default RAM. `campaign1_ledger` is the rough table the earlier E14 run used
# (a 2 vCPU E2B sandbox and the GPU-only Modal price); it is kept so the old
# numbers can be reproduced, and it understates both runtimes, the CPU one by
# about 5x.
_MODAL_MEM_USD_PER_GIB_HR = 0.0079992
_MODAL_CPU_USD_PER_CORE_HR = 0.04716


def _modal_usd_per_s(gpu_usd_per_hr: float, mem_mib: int) -> float:
    gib = mem_mib / 1024
    cores = min(16.0, max(0.5, round((gib / 8) * 2) / 2))   # CPU cores scale with the profile's memory
    return (gpu_usd_per_hr + gib * _MODAL_MEM_USD_PER_GIB_HR + cores * _MODAL_CPU_USD_PER_CORE_HR) / 3600


RATE_TABLES = {
    "server_list_price": {
        "usd_per_s": {
            "cpu": 0.5328 / 3600,
            "gpu_t4": _modal_usd_per_s(0.59, 16_384),
            "gpu_a100_40": _modal_usd_per_s(2.1, 32_768),
        },
        "source": ("provider list prices as of 2026-09 (E2B 8 vCPU / 8 GiB = $0.5328/hr; Modal: GPU list price + "
                   "$0.0079992/GiB-hr memory + $0.04716/core-hr CPU at the profile's default memory)"),
        "note": "list prices, not an invoice; the runtime seconds are the measurement",
    },
    "campaign1_ledger": {
        "usd_per_s": {"cpu": 0.000028, "gpu_t4": 0.000164, "gpu_a100_40": 0.000583},
        "source": ("an earlier campaign's RATE_USD_PER_SEC: rough per-second estimates kept for the "
                   "budget ledger (a 2 vCPU E2B sandbox; Modal GPU-only price without CPU or memory)"),
        "note": "kept only to reproduce the earlier E14 numbers; understates both runtimes",
    },
}
PRIMARY_RATE_TABLE = "server_list_price"

CLOCK_RULE = (
    "estimated runtime cost: a runtime is charged from immediately BEFORE the create_project request that "
    "creates it until immediately AFTER its delete_project returns. For a PATCH switch the source SKU is "
    "charged until the instant its destroy_returned step completes (PATCH start + the cumulative switch timeline "
    "ms up to and including that step; PATCH return when the step is absent) and the destination SKU from that "
    "same instant. Two runtimes alive at once (application-checkpoint moves) are both charged.")


# ============================================================================
# In-kernel code.
#
# Everything below that starts with `k_` runs INSIDE a sandbox kernel. It is
# shipped by source (inspect.getsource) inside ONE wrapper function that is
# called and then deleted, so a program leaves nothing of the harness in the
# kernel's `__main__`: the next switch dumps `__main__`, and a harness helper
# left there would be captured as if it were user state (see
# experiments/e13_chain.py verify_program). Helpers are nested siblings in the
# wrapper, so they call each other as closures. Each function imports what it
# uses, because nothing at module level here exists in the kernel. Names such
# as `FixtureSpec` resolve in the kernel's `__main__`, where the fixture is
# exec'd so dill records its classes and closures by value.
# ============================================================================

def k_sha(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()[:32]


def k_tensor_bytes(t):
    x = t.detach().to("cpu", copy=True).contiguous()
    return x.numpy().tobytes() if x.numel() else b"empty"


def k_params_digest(model):
    import hashlib
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        h.update(name.encode()); h.update(k_tensor_bytes(p))
    for name, b in sorted(model.named_buffers(), key=lambda kv: kv[0]):
        h.update(name.encode()); h.update(k_tensor_bytes(b))
    return h.hexdigest()[:32]


def k_optimizer_digest(opt):
    # Over the state_dict, which indexes state by parameter position, so the
    # digest does not depend on object identity and is comparable across
    # processes: every state tensor's bytes and every scalar (including the
    # per-parameter step), then every param group's hyperparameters.
    import hashlib
    import torch
    sd = opt.state_dict()
    h = hashlib.sha256()
    for idx in sorted(sd["state"]):
        for key in sorted(sd["state"][idx]):
            v = sd["state"][idx][key]
            h.update(f"{idx}:{key}:".encode())
            h.update(k_tensor_bytes(v) if torch.is_tensor(v) else repr(v).encode())
    for group in sd["param_groups"]:
        h.update(repr(sorted(group.items(), key=lambda kv: kv[0])).encode())
    return h.hexdigest()[:32]


def k_optimizer_step(opt):
    st = opt.state_dict()["state"]
    if not st:
        return None
    first = st[min(st)]
    v = first.get("step")
    return float(v) if v is not None else None


def k_scheduler_fp(sched):
    sd = sched.state_dict()
    return {"digest": k_sha(repr(sorted(sd.items(), key=lambda kv: kv[0])).encode()),
            "last_epoch": int(sched.last_epoch), "lr": [float(x) for x in sched.get_last_lr()]}


def k_generator_digest(gen):
    return "none" if gen is None else k_sha(k_tensor_bytes(gen.get_state()))


def k_rng_digests():
    # Full state of every process-global stream, read without drawing from
    # any of them. numpy's legacy state includes the Gaussian cache.
    import hashlib
    import random
    import numpy as np
    import torch
    out = {"python": k_sha(repr(random.getstate()).encode())}
    st = np.random.get_state()
    h = hashlib.sha256()
    h.update(st[1].tobytes()); h.update(repr((st[0], int(st[2]), int(st[3]), float(st[4]).hex())).encode())
    out["numpy"] = h.hexdigest()[:32]
    out["torch_cpu"] = k_sha(k_tensor_bytes(torch.get_rng_state()))
    if torch.cuda.is_available():
        was = torch.cuda.is_initialized()
        n = torch.cuda.device_count()
        out["cuda"] = {"available": True, "initialized_before": was, "device_count": n,
                       "states": [k_sha(k_tensor_bytes(torch.cuda.get_rng_state(i))) for i in range(n)]}
    else:
        out["cuda"] = {"available": False}
    return out


def k_module_training(model):
    flags = [(name, bool(m.training)) for name, m in model.named_modules()]
    return {"digest": k_sha(repr(flags).encode()),
            "train": sum(1 for _, f in flags if f), "eval": sum(1 for _, f in flags if not f)}


def k_fingerprint(g):
    loader = g["loader"]
    return {
        "params": k_params_digest(g["model"]),
        "optimizer": k_optimizer_digest(g["optimizer"]),
        "optimizer_step": k_optimizer_step(g["optimizer"]),
        "scheduler": k_scheduler_fp(g["scheduler"]),
        "loader_generator": k_generator_digest(getattr(loader, "generator", None)),
        "rng": k_rng_digests(),
        "module_training": k_module_training(g["model"]),
        "step_count": g.get("step_count"),
    }


def k_artefact_digests(g, names):
    # One digest per prep artefact (dtype, shape and bytes for a tensor, repr
    # otherwise), or "missing". Read at the end of PREP and again at the top
    # of ANALYSE, so the analysis can show that the artefacts it consumed are
    # the ones PREP made, not merely objects of the right name and shape.
    import torch
    out = {}
    for n in names:
        if n not in g:
            out[n] = "missing"
        elif torch.is_tensor(g[n]):
            t = g[n]
            out[n] = k_sha(f"{t.dtype}:{tuple(t.shape)}:".encode() + k_tensor_bytes(t))
        else:
            out[n] = k_sha(repr(g[n]).encode())
    return out


def k_host():
    # Where this program's CPU arithmetic ran. The fixture's three AdamW steps
    # and the feature passes run on the CPU, and their bits depend on the
    # intra-op thread count and on the ISA the kernels dispatch to (the same
    # seed gives different weights at 1, 2 and 8 threads), so two runtimes can
    # build different weights from identical code. Recorded beside every PREP
    # fingerprint so a cross-arm difference can be attributed to the host.
    import os
    import platform
    import torch
    model = platform.processor() or ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        cap = torch.backends.cpu.get_cpu_capability()
    except Exception:  # noqa: BLE001
        cap = None
    return {"cpu_model": model, "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(), "cpu_capability": cap, "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def k_prep_view(p, xs, aug):
    # Pass 0 is the original inputs (ANALYSE compares against these rows),
    # pass 1 the flipped cache, pass 2 the centre crop. Passes beyond three
    # are further 28x28 crops at every other offset of the original and of the
    # flipped cache (49 distinct views, then they repeat): a standard
    # multi-crop embedding, and a CPU-time knob that stays linear in N.
    if p == 0:
        return xs, "original"
    if p == 1:
        return aug, "hflip cache"
    if p == 2:
        return xs[:, :, 2:30, 2:30], "original crop 28 at (2,2)"
    offs = [(f, dy, dx) for f in (0, 1) for dy in range(5) for dx in range(5) if (f, dy, dx) != (0, 2, 2)]
    f, dy, dx = offs[(p - 3) % len(offs)]
    base, name = (xs, "original") if f == 0 else (aug, "hflip cache")
    return base[:, :, dy:dy + 28, dx:dx + 28], f"{name} crop 28 at ({dy},{dx})"


def k_prep(mark, spec_kw, passes, artefacts):
    import json
    import sys
    import time
    import torch
    g = sys.modules["__main__"].__dict__
    t0 = time.perf_counter()
    # Builds ResNet-18, its optimizer and scheduler, and a synthetic
    # CIFAR-shaped dataset (use_real_cifar10 is off in the fixture spec).
    g.update(build_transported_namespace(FixtureSpec(**spec_kw)))  # noqa: F821
    t_build = time.perf_counter()
    # One epoch of the SHUFFLING loader, as a user would iterate their data.
    # This advances loader.generator by one epoch, which is state TRAIN
    # depends on: the next epoch's batch order follows from it.
    xs_parts, ys_parts = [], []
    for xb, yb in g["loader"]:
        xs_parts.append(xb); ys_parts.append(yb)
    xs, ys = torch.cat(xs_parts), torch.cat(ys_parts)
    t_epoch = time.perf_counter()
    g["prep_mean"] = xs.mean(dim=(0, 2, 3))
    g["prep_std"] = xs.std(dim=(0, 2, 3))
    g["aug_cache"] = torch.stack([torch.flip(x, dims=[2]) for x in xs])
    g["prep_labels"] = ys.clone()                  # row i labels aug_cache[i]
    t_stats = time.perf_counter()
    # Sequential over the model's own children: .eval() switches those shared
    # submodules to eval mode and leaves the top-level model in train mode.
    # That mixed per-module mode is real namespace state (TRAIN resets it with
    # model.train()); an application checkpoint that ignores module flags
    # restores a different model mode, which the TRAIN-input fingerprint shows.
    backbone = torch.nn.Sequential(*list(g["model"].children())[:-1]).eval()
    # The bank holds one block per view kind: the original inputs (block 0),
    # the flipped cache (block 1) and the MEAN embedding over every crop view
    # (block 2). At the default three passes this is exactly one block per
    # pass. Averaging the extra crops instead of appending them keeps the
    # stored state the same size whatever --prep-passes is, so the crossover
    # sweep lengthens the CPU phase without also growing what a switch or a
    # checkpoint has to move.
    blocks, crop_sum, n_crop, pass_s, views = [], None, 0, [], []
    with torch.no_grad():
        for p in range(passes):
            tp = time.perf_counter()
            src, view = k_prep_view(p, xs, g["aug_cache"])
            f = torch.cat([backbone(src[i:i + 32]).flatten(1) for i in range(0, src.shape[0], 32)])
            if p < 2:
                blocks.append(f)
            else:
                crop_sum = f if crop_sum is None else crop_sum + f
                n_crop += 1
            pass_s.append(time.perf_counter() - tp)
            views.append(view)
    if crop_sum is not None:
        blocks.append(crop_sum / n_crop)
    g["prep_features"] = torch.cat(blocks)
    g["prep_seconds"] = time.perf_counter() - t0
    # The last thing PREP does (outside prep_seconds): fingerprint the state
    # it hands on, with the same function TRAIN uses on its inputs, and digest
    # every artefact. Each arm's TRAIN input is then compared with its OWN
    # PREP output, which isolates what carrying the state did from the host
    # arithmetic that built it (see k_host).
    fp_out = k_fingerprint(g)
    arts = k_artefact_digests(g, artefacts)
    print(mark + json.dumps({
        "prep_seconds": g["prep_seconds"], "build_s": t_build - t0, "loader_epoch_s": t_epoch - t_build,
        "stats_s": t_stats - t_epoch, "pass_s": pass_s, "passes": passes, "views": views,
        "samples": int(xs.shape[0]), "features": list(g["prep_features"].shape), "crop_views_averaged": n_crop,
        "data_source": g.get("data_source"), "torch_threads": torch.get_num_threads(),
        "prep_output": fp_out, "artefacts": arts, "host": k_host(),
    }))


def k_train(mark, steps):
    import json
    import sys
    import time
    import torch
    g = sys.modules["__main__"].__dict__
    # The first thing TRAIN does: fingerprint everything it is about to
    # depend on, before it moves or touches anything.
    fp_in = k_fingerprint(g)
    # Process state that no capsule carries, so it is set in the program.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    t0 = time.perf_counter()
    model, opt, sched, loss_fn, loader = g["model"], g["optimizer"], g["scheduler"], g["loss_fn"], g["loader"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    for st in opt.state.values():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                st[k] = v.to(dev)
    model.train()
    n_batches = len(loader)
    it, in_epoch = None, 0
    lrs, losses, boundaries = [], [], []
    for i in range(steps):
        if it is None:
            # A new epoch's iterator is made at its first batch, as
            # `for xb, yb in loader` does. Making a DataLoader iterator draws
            # its base seed from loader.generator, so one made eagerly after
            # the last completed epoch would advance the generator for an
            # epoch that never runs.
            it, in_epoch = iter(loader), 0
        xb, yb = next(it)
        lrs.append(float(opt.param_groups[0]["lr"]))
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())   # float(loss) on a grad tensor warns, leaving __warningregistry__ in __main__
        g["step_count"] = g.get("step_count", 0) + 1
        in_epoch += 1
        if in_epoch == n_batches:
            # The epoch is complete. End it the way `for xb, yb in loader`
            # does: that loop's final next() raises StopIteration, and on the
            # way out torch's RandomSampler draws one more randperm from
            # loader.generator even though nothing is left to yield. An
            # iterator abandoned after its last batch skips that draw and puts
            # every later epoch on a different order from the canonical loop.
            if next(it, None) is not None:
                raise RuntimeError("the loader yielded more batches than len(loader)")
            # StepLR is a per-epoch schedule: step it once per COMPLETED
            # epoch, right after the epoch (the canonical place, after the
            # inner loop). Stepping only when the next epoch starts would skip
            # the last epoch whenever --steps is a multiple of the epoch
            # length. The earlier harness stepped it per batch, which drove lr
            # to ~1e-18 within 120 steps.
            sched.step()
            boundaries.append(i + 1)      # the first step of the next epoch
            it = None
    g["train_seconds"] = time.perf_counter() - t0
    fp_out = k_fingerprint(g)
    print(mark + json.dumps({
        "train_seconds": g["train_seconds"], "device": str(dev), "step_count": g["step_count"],
        "steps": steps, "last_loss": losses[-1] if losses else None,
        "batches_per_epoch": n_batches, "epochs_completed": len(boundaries), "epoch_boundaries": boundaries,
        "host": k_host(),
        "lr_trajectory": lrs, "losses": losses,
        "determinism": {"cudnn_deterministic": torch.backends.cudnn.deterministic,
                        "cudnn_benchmark": torch.backends.cudnn.benchmark, "set_at": "top of TRAIN"},
        "train_input": fp_in, "train_output": fp_out, "data_source": g.get("data_source"),
    }))


def k_analyse(mark, artefacts):
    import json
    import sys
    import time
    import torch
    g = sys.modules["__main__"].__dict__
    # The first thing ANALYSE does: digest the prep artefacts it is about to
    # consume, to compare with the digests PREP recorded when it made them.
    artefacts_at_entry = k_artefact_digests(g, artefacts)
    t0 = time.perf_counter()
    model, loss_fn = g["model"], g["loss_fn"]
    model.to("cpu").eval()
    # The dataset in index order, not the shuffling loader: evaluation then
    # draws nothing from the loader generator, so the same weights give the
    # same numbers in every arm wherever that stream happens to stand.
    xs_all, ys_all = g["loader"].dataset.tensors
    res = {}
    with torch.no_grad():
        correct, total, loss_sum = 0, 0, 0.0
        for i in range(0, xs_all.shape[0], 64):
            out = model(xs_all[i:i + 64]); yb = ys_all[i:i + 64]
            loss_sum += float(loss_fn(out, yb)) * int(yb.numel())
            correct += int((out.argmax(1) == yb).sum()); total += int(yb.numel())
    res["accuracy"] = correct / total
    res["mean_loss"] = loss_sum / total

    # Metrics that CONSUME the prep artefacts. Each runs on its own and
    # records the exact exception, so a missing artefact is a named failure
    # (the analysis would have stopped there), not a silently different number.
    def tta():
        aug, labels = g["aug_cache"], g["prep_labels"]
        right = 0
        with torch.no_grad():
            for i in range(0, aug.shape[0], 64):
                a = aug[i:i + 64]
                logits = (model(a) + model(torch.flip(a, dims=[3]))) / 2   # flip of the cache is the original
                right += int((logits.argmax(1) == labels[i:i + 64]).sum())
        return {"accuracy": right / int(aug.shape[0]), "samples": int(aug.shape[0])}

    def representation_shift():
        feats = g["prep_features"]
        aug = g["aug_cache"]
        n = int(aug.shape[0])
        ref = feats[:n]                                  # pass 0: the original inputs, in PREP's order
        x0 = torch.flip(aug, dims=[3])
        backbone = torch.nn.Sequential(*list(model.children())[:-1])
        with torch.no_grad():
            cur = torch.cat([backbone(x0[i:i + 64]).flatten(1) for i in range(0, n, 64)])
        cos = torch.nn.functional.cosine_similarity(ref, cur, dim=1)
        return {"mean_cosine": float(cos.mean()), "min_cosine": float(cos.min()), "rows": n}

    def standardisation():
        mean, std = g["prep_mean"], g["prep_std"]
        z = (xs_all - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
        m, s = z.mean(dim=(0, 2, 3)), z.std(dim=(0, 2, 3))
        a, b = float(m.abs().max()), float((s - 1).abs().max())
        return {"max_abs_channel_mean": a, "max_abs_channel_std_minus_1": b, "consistent": a < 1e-3 and b < 1e-3}

    missing, errors = {}, {}
    for name, fn in (("tta_accuracy", tta), ("representation_shift", representation_shift),
                     ("input_standardisation", standardisation)):
        try:
            res[name] = fn()
        except KeyError as e:
            missing[name] = f"KeyError: {e}"
        except Exception as e:  # noqa: BLE001
            errors[name] = f"{type(e).__name__}: {e}"
    res["missing_dependencies"] = missing
    res["metric_errors"] = errors
    res["artefacts_at_entry"] = artefacts_at_entry
    res["final_params_digest"] = k_params_digest(model)
    res["step_count"] = g.get("step_count")
    res["data_source"] = g.get("data_source")
    g["analysis"] = res
    g["analysis_seconds"] = time.perf_counter() - t0
    print(mark + json.dumps({"analysis_seconds": g["analysis_seconds"], **res}))


# ---- the application checkpoint: what the user writes -------------------
# Only lines between the markers are application code; `app_code_loc` counts
# them (non-blank, non-comment) for each arm. The `return` after each block
# hands the checkpoint to the harness wrapper for reporting and is not
# counted. `ns` is the notebook namespace (the kernel's `__main__`).

def k_app_save_complete(ns, path):
    # --- application code begin ---
    import random, numpy as np, torch
    ckpt = {
        "model": ns["model"].state_dict(),
        "module_training": {name: m.training for name, m in ns["model"].named_modules()},
        "optimizer": ns["optimizer"].state_dict(),
        "scheduler": ns["scheduler"].state_dict(),
        "step_count": ns["step_count"],
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()},
        "loader_generator": ns["loader"].generator.get_state(),
        "prep": {k: ns[k] for k in ("prep_mean", "prep_std", "aug_cache", "prep_labels", "prep_features", "prep_seconds")},
    }
    if torch.cuda.is_available():
        ckpt["rng"]["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(ckpt, path)
    # --- application code end ---
    return ckpt


def k_app_save_incomplete(ns, path):
    # --- application code begin ---
    import random, numpy as np, torch
    ckpt = {
        "model": ns["model"].state_dict(),
        "optimizer": ns["optimizer"].state_dict(),
        "scheduler": ns["scheduler"].state_dict(),
        "step_count": ns["step_count"],
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()},
    }
    torch.save(ckpt, path)
    # --- application code end ---
    return ckpt


def k_app_rebuild(ns, spec_kw):
    # The objects the state_dicts load INTO must exist first: re-run the
    # job's construction code on the new runtime.
    # --- application code begin ---
    ns.update(build_transported_namespace(FixtureSpec(**spec_kw)))
    # --- application code end ---


def k_app_load_complete(ns, path):
    # --- application code begin ---
    import random, numpy as np, torch
    ck = torch.load(path, weights_only=False, map_location="cpu")
    ns["model"].load_state_dict(ck["model"])
    for name, m in ns["model"].named_modules():
        m.training = ck["module_training"][name]
    ns["optimizer"].load_state_dict(ck["optimizer"])
    ns["scheduler"].load_state_dict(ck["scheduler"])
    ns["step_count"] = ck["step_count"]
    random.setstate(ck["rng"]["python"])
    np.random.set_state(ck["rng"]["numpy"])
    torch.set_rng_state(ck["rng"]["torch"])
    if "cuda" in ck["rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
    ns["loader"].generator.set_state(ck["loader_generator"])
    ns.update(ck["prep"])
    # --- application code end ---
    return ck


def k_app_load_incomplete(ns, path):
    # --- application code begin ---
    import random, numpy as np, torch
    ck = torch.load(path, weights_only=False, map_location="cpu")
    ns["model"].load_state_dict(ck["model"])
    ns["optimizer"].load_state_dict(ck["optimizer"])
    ns["scheduler"].load_state_dict(ck["scheduler"])
    ns["step_count"] = ck["step_count"]
    random.setstate(ck["rng"]["python"])
    np.random.set_state(ck["rng"]["numpy"])
    torch.set_rng_state(ck["rng"]["torch"])
    # --- application code end ---
    return ck


def k_app_load_naive(ns, path):
    # Without map_location a checkpoint saved on the GPU raises on a CPU
    # destination: "Attempting to deserialize object on a CUDA device but
    # torch.cuda.is_available() is False".
    # --- application code begin ---
    import random, numpy as np, torch
    ck = torch.load(path, weights_only=False)
    ns["model"].load_state_dict(ck["model"])
    ns["optimizer"].load_state_dict(ck["optimizer"])
    ns["scheduler"].load_state_dict(ck["scheduler"])
    ns["step_count"] = ck["step_count"]
    random.setstate(ck["rng"]["python"])
    np.random.set_state(ck["rng"]["numpy"])
    torch.set_rng_state(ck["rng"]["torch"])
    # --- application code end ---
    return ck


def k_ckpt_contents(ck):
    keys = []
    for k in sorted(ck):
        if isinstance(ck[k], dict) and k in ("rng", "prep"):
            keys.extend(f"{k}.{j}" for j in sorted(ck[k]))
        else:
            keys.append(k)
    return keys


def k_app_save_run(mark, path, variant):
    import json
    import os
    import sys
    import time
    import torch
    ns = sys.modules["__main__"].__dict__
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    t0 = time.perf_counter()
    save = k_app_save_complete if variant == "complete" else k_app_save_incomplete
    ck = save(ns, path)
    print(mark + json.dumps({"variant": variant, "save_s": time.perf_counter() - t0,
                             "bytes": os.path.getsize(path), "contents": k_ckpt_contents(ck),
                             "saved_on_cuda_runtime": torch.cuda.is_available()}))


def k_app_load_run(mark, path, variant, spec_kw):
    import json
    import sys
    import time
    import torch
    ns = sys.modules["__main__"].__dict__
    t0 = time.perf_counter()
    k_app_rebuild(ns, spec_kw)
    t1 = time.perf_counter()
    load = {"complete": k_app_load_complete, "incomplete": k_app_load_incomplete, "naive": k_app_load_naive}[variant]
    ck = load(ns, path)
    t2 = time.perf_counter()
    if "cuda" not in ck.get("rng", {}):
        cuda = "not in checkpoint (saved on a runtime without CUDA)" if variant == "complete" else "not saved by this checkpoint"
    elif torch.cuda.is_available():
        cuda = "restored"
    else:
        cuda = "declared drop: destination has no CUDA"
    print(mark + json.dumps({"variant": variant, "rebuild_s": t1 - t0, "load_s": t2 - t1,
                             "loaded_step": ns.get("step_count"), "contents": k_ckpt_contents(ck),
                             "cuda_rng": cuda, "destination_cuda": torch.cuda.is_available()}))


# ---- program assembly -------------------------------------------------------

_HELPERS = [k_sha, k_tensor_bytes, k_params_digest, k_optimizer_digest, k_optimizer_step, k_scheduler_fp,
            k_generator_digest, k_rng_digests, k_module_training, k_fingerprint]
_APP_FUNCS = [k_app_save_complete, k_app_save_incomplete, k_app_rebuild,
              k_app_load_complete, k_app_load_incomplete, k_app_load_naive, k_ckpt_contents]


def _indent(block: str, pad: str = "    ") -> str:
    return "\n".join(pad + line if line.strip() else line for line in block.splitlines())


def _program(entry, helpers, args, *, fixture: bool = False) -> str:
    """One kernel program: the fixture (when the program builds the job) at
    top level, then a single wrapper holding `entry` and its helpers, called
    once and deleted in a `finally`, so nothing of it outlives the call."""
    body = "\n\n".join(_indent(textwrap.dedent(inspect.getsource(f))) for f in [*helpers, entry])
    call = f"    {entry.__name__}({', '.join(repr(a) for a in args)})\n"
    pre = f'exec(compile({FIXTURE_SRC!r}, "clusy_fixture.py", "exec"), globals())\n' if fixture else ""
    return (pre + "def __clusy_e14_program():\n" + body + "\n" + call
            + "try:\n    __clusy_e14_program()\nfinally:\n    del __clusy_e14_program\n")


def prep_program(passes: int) -> str:
    return _program(k_prep, [k_prep_view, *_HELPERS, k_artefact_digests, k_host],
                    [MARK, SPEC_KW, passes, list(PREP_ARTEFACTS)], fixture=True)


def train_program(steps: int) -> str:
    return _program(k_train, [*_HELPERS, k_host], [MARK, steps])


def analyse_program() -> str:
    return _program(k_analyse, [k_sha, k_tensor_bytes, k_params_digest, k_artefact_digests],
                    [MARK, list(PREP_ARTEFACTS)])


def app_save_program(variant: str) -> str:
    return _program(k_app_save_run, _APP_FUNCS, [MARK, CKPT_PATH, "complete" if variant == "complete" else "incomplete"])


def app_load_program(variant: str) -> str:
    return _program(k_app_load_run, _APP_FUNCS, [MARK, CKPT_PATH, variant, SPEC_KW], fixture=True)


APP_BEGIN, APP_END = "# --- application code begin ---", "# --- application code end ---"


def app_code_loc(funcs) -> tuple[int, list[str]]:
    """Count application code: non-blank, non-comment lines between the
    markers in each function's source. Returns the count and the lines."""
    lines: list[str] = []
    for f in funcs:
        inside = False
        for raw in inspect.getsource(f).splitlines():
            s = raw.strip()
            if s == APP_BEGIN:
                inside = True; continue
            if s == APP_END:
                inside = False; continue
            if inside and s and not s.startswith("#"):
                lines.append(s)
        if inside:
            raise ValueError(f"{f.__name__}: application code block is not closed")
    return len(lines), lines


def app_code_for(variant: str) -> dict:
    save = k_app_save_complete if variant == "complete" else k_app_save_incomplete
    load = {"complete": k_app_load_complete, "incomplete": k_app_load_incomplete, "naive": k_app_load_naive}[variant]
    n_save, l_save = app_code_loc([save])
    n_rebuild, l_rebuild = app_code_loc([k_app_rebuild])
    n_load, l_load = app_code_loc([load])
    return {"save_loc": n_save, "rebuild_loc": n_rebuild, "load_loc": n_load,
            "total_loc": n_save + n_rebuild + n_load,
            "counted_from": "non-blank, non-comment lines between the application code markers",
            "functions": [save.__name__, k_app_rebuild.__name__, load.__name__],
            "lines": l_save + l_rebuild + l_load}


# ============================================================================
# Harness side
# ============================================================================

class Runtimes:
    """The estimated-runtime-cost clock for one arm (see CLOCK_RULE).

    Every interval is (SKU, start, end) relative to the arm's start, with the
    event that opened and closed it, so the per-SKU seconds are auditable."""

    def __init__(self, api, arm: str, *, list_retries: int = 4, list_wait_s: float = 2.0):
        self.api, self.arm = api, arm
        self.t0 = time.perf_counter()
        self.live: dict[str, dict] = {}
        self.closed: list[dict] = []
        self.pids: list[str] = []
        self.create_failures: list[dict] = []
        self.list_retries, self.list_wait_s = list_retries, list_wait_s

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def create(self, profile: str) -> str:
        # The name is fixed BEFORE the request, so a project the server made
        # for a request that then failed on the client side can still be found.
        name = f"clusy-exp-e14-{self.arm}-{uuid.uuid4().hex[:12]}"
        t = self.now()
        try:
            pid = self.api.create_project(name, profile)
        except Exception as e:  # noqa: BLE001
            self._after_failed_create(name, profile, t, e)
            raise
        self.live[pid] = {"pid": pid, "name": name, "sku": profile, "start_s": t,
                          "start_event": "before create_project", "create_returned_s": self.now()}
        self.pids.append(pid)
        return pid

    def _after_failed_create(self, name: str, profile: str, t: float, err: Exception) -> None:
        """A create request can fail AFTER the server has made the project: a
        client-side timeout (urllib's default 120 s in Api._req), or a non-201
        answer to a create that went through. Such a project is found by its
        name, charged from before the request like any other runtime, and
        deleted by the arm's teardown. When no listing succeeds, the gap is
        recorded (`resolved: False`), never hidden; the clock check fails on it."""
        entry = {"name": name, "sku": profile, "request_started_s": t,
                 "error": f"{type(err).__name__}: {err}"[:300], "found": [], "listing_errors": []}
        items = None
        for attempt in range(self.list_retries):
            try:
                items = self.api.list_projects()
            except Exception as le:  # noqa: BLE001
                entry["listing_errors"].append(f"{type(le).__name__}: {le}"[:200])
                items = None
            # A listing can lag a create, as the handoff reclaim also allows for.
            if (items is not None and any(str(it.get("name", "")) == name for it in items)) \
                    or attempt == self.list_retries - 1:
                break
            time.sleep(self.list_wait_s)
        entry["resolved"] = items is not None
        for it in items or []:
            if str(it.get("name", "")) == name and it.get("id") and it["id"] not in self.live:
                pid = it["id"]
                entry["found"].append(pid)
                self.live[pid] = {"pid": pid, "name": name, "sku": profile, "start_s": t,
                                  "start_event": "before create_project (the request raised; project found by name)",
                                  "create_returned_s": self.now()}
                self.pids.append(pid)
        self.create_failures.append(entry)

    def delete(self, pid: str, note: str = "") -> int:
        st = self.api.delete_project(pid)
        t = self.now()
        iv = self.live.pop(pid)
        iv.update(end_s=t, end_event="after delete_project returned" + (f" ({note})" if note else ""),
                  delete_status=st)
        self._close(iv)
        return st

    def switch(self, pid: str, profile: str) -> tuple[int, dict, dict]:
        t_start = self.now()
        st, payload = self.api.patch_profile(pid, profile)
        t_end = self.now()
        info = {"to": profile, "status": st, "patch_start_s": t_start, "patch_return_s": t_end}
        if st != 200:
            return st, payload, info   # the source is still the runtime; nothing to split
        timeline = payload.get("switchTimeline") or {}
        cum, found = 0.0, False
        for step in timeline.get("steps") or []:
            cum += float(step.get("ms") or 0.0)
            if step.get("step") == "destroy_returned":
                found = True
                break
        if found:
            split = t_start + cum / 1000.0
            info.update(split_rule="destroy_returned: PATCH start + cumulative timeline ms",
                        destroy_cumulative_ms=cum, split_capped_at_patch_return=split > t_end)
            split = min(split, t_end)
        else:
            split = t_end
            info.update(split_rule="PATCH returned (no destroy_returned step in the timeline)")
        info["split_s"] = split
        steps = timeline.get("steps") or []
        info["timeline_sum_ms"] = sum(float(s.get("ms") or 0.0) for s in steps)
        info["timeline_total_ms"] = timeline.get("total_ms")
        iv = self.live.pop(pid)
        iv.update(end_s=split, end_event="source destroy_returned" if found else "PATCH returned")
        self._close(iv)
        self.live[pid] = {"pid": pid, "sku": profile, "start_s": split,
                          "start_event": "destination from source destroy_returned" if found else "PATCH returned"}
        return st, payload, info

    def _close(self, iv: dict) -> None:
        iv["seconds"] = iv["end_s"] - iv["start_s"]
        self.closed.append(iv)

    def teardown(self, note: str) -> None:
        for pid in list(self.live):
            try:
                self.delete(pid, note)
            except Exception as e:  # noqa: BLE001
                iv = self.live.pop(pid)
                iv.update(end_s=self.now(), end_event=f"delete failed: {type(e).__name__}: {e}"[:200])
                self._close(iv)

    def summary(self) -> dict:
        per_sku: dict[str, float] = {}
        for iv in self.closed:
            per_sku[iv["sku"]] = per_sku.get(iv["sku"], 0.0) + iv["seconds"]
        cost = {}
        for name, table in RATE_TABLES.items():
            rates = table["usd_per_s"]
            unknown = [s for s in per_sku if s not in rates]
            cost[name] = None if unknown else sum(per_sku[s] * rates[s] for s in per_sku)
        return {"runtime_s": per_sku, "runtimes": self.closed, "est_runtime_cost_usd": cost,
                "create_failures": self.create_failures}


def move_file_s3(api, src_pid: str, dst_pid: str, path: str = CKPT_PATH) -> dict:
    """What a user does between runtimes with their own cloud storage: upload
    the checkpoint from the old runtime, download it on the new one. The
    harness only mints the presigned URLs; the bytes move sandbox-to-bucket-
    to-sandbox, which is the real cost of this approach."""
    import boto3
    from botocore.config import Config
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        raise RuntimeError("the app-checkpoint arms need S3_BUCKET, S3_REGION and AWS credentials in the environment")
    key = f"e14/{uuid.uuid4().hex}.pt"
    # A plain PUT sends no checksum headers; newer boto3 signs them into a
    # presigned PUT unless told not to, and the upload then fails with 403.
    s3 = boto3.client("s3", region_name=os.environ.get("S3_REGION", "us-east-1"),
                      config=Config(signature_version="s3v4", request_checksum_calculation="when_required",
                                    response_checksum_validation="when_required"))
    put = s3.generate_presigned_url("put_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=3600)
    get = s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=3600)
    t0 = time.perf_counter()
    up = api.witness(src_pid, f"""
def __clusy_e14_up():
    import httpx, json, os
    with open({path!r}, "rb") as f:
        r = httpx.put({put!r}, content=f, headers={{"Content-Length": str(os.path.getsize({path!r}))}}, timeout=1800.0)
    r.raise_for_status()
    print({MARK!r} + json.dumps({{"uploaded": os.path.getsize({path!r})}}))
try:
    __clusy_e14_up()
finally:
    del __clusy_e14_up
""", timeout_ms=1_800_000)
    down = api.witness(dst_pid, f"""
def __clusy_e14_down():
    import httpx, json, os
    os.makedirs(os.path.dirname({path!r}) or ".", exist_ok=True)
    with httpx.stream("GET", {get!r}, timeout=1800.0) as r, open({path!r}, "wb") as f:
        r.raise_for_status()
        for chunk in r.iter_bytes(1 << 20):
            f.write(chunk)
    print({MARK!r} + json.dumps({{"downloaded": os.path.getsize({path!r})}}))
try:
    __clusy_e14_down()
finally:
    del __clusy_e14_down
""", timeout_ms=1_800_000)
    return {"transport": "s3 presigned PUT/GET", "seconds": time.perf_counter() - t0,
            "bytes": down.get("downloaded"), "uploaded": up.get("uploaded")}


def move_file_local(api, src_pid: str, dst_pid: str, path: str = CKPT_PATH) -> dict:
    """The dry-run stand-in for `move_file_s3`: copy between the two local
    kernels' working directories."""
    t0 = time.perf_counter()
    src = Path(api.projects[src_pid]["cwd"]) / path
    dst = Path(api.projects[dst_pid]["cwd"]) / path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return {"transport": "local file copy (dry run)", "seconds": time.perf_counter() - t0,
            "bytes": dst.stat().st_size}


_NAMES_PROGRAM = f"""
def __clusy_e14_names():
    import sys, json
    print({MARK!r} + json.dumps(sorted(sys.modules["__main__"].__dict__)))
try:
    __clusy_e14_names()
finally:
    del __clusy_e14_names
"""


def local_namespace_probe(api) -> dict:
    """Rule for every shipped program: nothing of the harness may remain in
    `__main__`, because the next switch dumps it. The local simulated switch
    filters underscore names, so a leftover wrapper would pass a local run
    and be captured on the platform; this probe therefore reads `__main__`
    directly after each program and checks that each one added exactly the
    job state it is meant to add."""
    def names(pid):
        return set(api.witness(pid, _NAMES_PROGRAM)) - {"__clusy_e14_names"}

    def leftovers(ns):
        return sorted(n for n in ns if n.startswith("k_") or "clusy_e14" in n or n in ("ck", "ckpt"))

    steps = []
    p1 = api.create_project("e14-hygiene-src", "cpu")
    p2 = api.create_project("e14-hygiene-dst", "cpu")
    try:
        prev = names(p1)
        plan = [("prep", p1, prep_program(1), None), ("train", p1, train_program(2), {"train_seconds"}),
                ("analyse", p1, analyse_program(), {"analysis", "analysis_seconds"}),
                ("app_save", p1, app_save_program("complete"), set())]
        for label, pid, code, expected in plan:
            api.witness(pid, code)
            now = names(pid)
            added, removed, left = sorted(now - prev), sorted(prev - now), leftovers(now)
            if expected is None:     # PREP adds the whole job; check it holds the job and the artefacts
                ok = not left and not removed and set(PREP_ARTEFACTS) <= now and "model" in now
                added = f"{len(added)} names (fixture, job namespace, prep artefacts)"
            else:
                ok = not left and not removed and set(added) == expected
            steps.append({"program": label, "added": added, "removed": removed, "harness_names_left": left, "ok": ok})
            prev = now
        move_file_local(api, p1, p2)
        before = names(p2)
        api.witness(p2, app_load_program("complete"))
        now = names(p2)
        left = leftovers(now)
        steps.append({"program": "app_load", "added": f"{len(now - before)} names", "harness_names_left": left,
                      "ok": not left and set(PREP_ARTEFACTS) <= now})
    finally:
        api.delete_project(p1)
        api.delete_project(p2)
    return {"name": "namespace_hygiene", "ok": all(st["ok"] for st in steps), "gate": True,
            "detail": {"steps": steps}}


def phase(api, pid: str, code: str, label: str, log, timeout_ms: int = 1_800_000) -> tuple[float, dict]:
    t0 = time.perf_counter()
    w = api.witness(pid, code, timeout_ms=timeout_ms)
    s = time.perf_counter() - t0
    brief = {k: v for k, v in w.items() if not isinstance(v, (list, dict))}
    log(f"    {label:<14} {s:7.1f}s  {json.dumps(brief, default=str)[:120]}")
    return s, w


def run_arm(api, arm: str, *, steps: int, passes: int, gpu: str, mover, log) -> dict:
    rt = Runtimes(api, arm)
    # `prep_feeding` names the PREP witness whose state each later phase
    # consumed: the arm's own reference for the TRAIN-input and artefact
    # comparisons. Only restart has more than one PREP.
    rec: dict = {"arm": arm, "profiles": {"cpu": "cpu", "gpu": gpu}, "phases": {}, "outcome": "completed",
                 "work_recomputed_s": 0.0, "work_recomputed_steps": 0, "work_recomputed": [],
                 "app_loc": 0, "projects": rt.pids, "prep_feeding": {"train": "prep", "analysis": "prep"}}
    current = {"phase": None}

    def start(profile: str) -> str:
        current["phase"] = f"create_project {profile}"
        return rt.create(profile)

    def run(pid: str, code: str, label: str) -> tuple[float, dict]:
        current["phase"] = label
        return phase(api, pid, code, label, log)

    def recomputed(label: str, seconds: float, basis: str, steps_redone: int = 0) -> None:
        rec["work_recomputed_s"] += seconds
        rec["work_recomputed_steps"] += steps_redone
        rec["work_recomputed"].append({"what": label, "seconds": seconds, "basis": basis, "steps": steps_redone})

    PREP, TRAIN, ANALYSE = prep_program(passes), train_program(steps), analyse_program()
    try:
        if arm in ("uninterrupted", "always"):
            pid = start("cpu" if arm == "uninterrupted" else gpu)
            rec["phases"]["prep"], rec["prep"] = run(pid, PREP, "prep")
            rec["phases"]["train"], rec["train"] = run(pid, TRAIN, "train")
            rec["phases"]["analyse"], rec["analysis"] = run(pid, ANALYSE, "analyse")
            rt.delete(pid)

        elif arm == "switch":
            rec["switches"] = []
            pid = start("cpu")
            rec["phases"]["prep"], rec["prep"] = run(pid, PREP, "prep")
            for label, to, nxt in (("switch_to_gpu", gpu, "train"), ("switch_to_cpu", "cpu", "analyse")):
                current["phase"] = label
                t0 = time.perf_counter()
                st, payload, info = rt.switch(pid, to)
                rec["phases"][label] = time.perf_counter() - t0
                info.update(phases=payload.get("switchPhases"), timeline=payload.get("switchTimeline"))
                rec["switches"].append(info)
                log(f"    {label:<14} {rec['phases'][label]:7.1f}s  PATCH {st}")
                if st != 200:
                    raise RuntimeError(f"PATCH {to} returned {st}: {json.dumps(payload)[:300]}")
                if nxt == "train":
                    rec["phases"]["train"], rec["train"] = run(pid, TRAIN, "train")
                else:
                    rec["phases"]["analyse"], rec["analysis"] = run(pid, ANALYSE, "analyse")
            rt.delete(pid)

        elif arm == "restart":
            # Nothing carried: each phase is a fresh runtime that redoes what it depends on.
            p1 = start("cpu")
            rec["phases"]["prep"], rec["prep"] = run(p1, PREP, "prep")
            rt.delete(p1)
            rec["prep_again"] = {}
            rec["prep_feeding"] = {"train": "prep_again_gpu", "analysis": "prep_again_cpu"}
            p2 = start(gpu)
            s, rec["prep_again"]["prep_again_gpu"] = run(p2, PREP, "prep_again_gpu")
            recomputed("prep on the GPU runtime", s, "phase wall time")
            rec["phases"]["train"], rec["train"] = run(p2, TRAIN, "train")
            rt.delete(p2)
            p3 = start("cpu")
            s, rec["prep_again"]["prep_again_cpu"] = run(p3, PREP, "prep_again_cpu")
            recomputed("prep on the analysis runtime", s, "phase wall time")
            s, w = run(p3, TRAIN, "train_again_cpu")
            recomputed("train on the analysis runtime (cpu)", s, "phase wall time", steps)
            rec["train_again"] = w
            rec["phases"]["analyse"], rec["analysis"] = run(p3, ANALYSE, "analyse")
            rt.delete(p3)

        elif arm in APP_ARMS:
            variant = APP_ARMS[arm]
            code = app_code_for(variant)
            rec["app_code"] = code
            rec["app_loc"] = code["total_loc"]
            rec["app"] = {"variant": variant, "saves": [], "loads": [], "moves": []}
            SAVE, LOAD = app_save_program(variant), app_load_program(variant)
            p1 = start("cpu")
            rec["phases"]["prep"], rec["prep"] = run(p1, PREP, "prep")
            rec["phases"]["app_save_1"], w = run(p1, SAVE, "app_save_1")
            rec["app"]["saves"].append(w)
            rec["checkpoint_contents"] = w["contents"]
            p2 = start(gpu)
            current["phase"] = "move_1"
            m = mover(api, p1, p2); rec["phases"]["move_1"] = m["seconds"]; rec["app"]["moves"].append(m)
            rt.delete(p1)
            rec["phases"]["app_load_1"], w = run(p2, LOAD, "app_load_1")
            rec["app"]["loads"].append(w)
            recomputed("rebuild the job's objects before loading (fixture build)", w["rebuild_s"], "in-kernel seconds")
            rec["phases"]["train"], rec["train"] = run(p2, TRAIN, "train")
            rec["phases"]["app_save_2"], w = run(p2, SAVE, "app_save_2")
            rec["app"]["saves"].append(w)
            p3 = start("cpu")
            current["phase"] = "move_2"
            m = mover(api, p2, p3); rec["phases"]["move_2"] = m["seconds"]; rec["app"]["moves"].append(m)
            rt.delete(p2)
            rec["phases"]["app_load_2"], w = run(p3, LOAD, "app_load_2")
            rec["app"]["loads"].append(w)
            recomputed("rebuild the job's objects before loading (fixture build)", w["rebuild_s"], "in-kernel seconds")
            rec["phases"]["analyse"], rec["analysis"] = run(p3, ANALYSE, "analyse")
            rt.delete(p3)
        else:
            raise ValueError(f"unknown arm {arm!r}")
    except Exception as e:  # noqa: BLE001
        rec["outcome"] = "failed"
        rec["failed_phase"] = current["phase"]
        rec["error"] = f"{type(e).__name__}: {e}"[:2000]
        if arm == "appckpt_naive":
            rec["expected_failure"] = "torch.load without map_location on a CPU runtime, for a checkpoint saved on the GPU"
    finally:
        rt.teardown("cleanup after error" if rec["outcome"] == "failed" else "cleanup")
    rec["wall_s"] = rt.now()
    rec.update(rt.summary())
    rec["cost_label"] = "estimated runtime cost"
    rec["clock_rule"] = CLOCK_RULE
    rec["rates"] = RATE_TABLES
    rec["primary_rate_table"] = PRIMARY_RATE_TABLE
    rec["data_source"] = (rec.get("analysis") or rec.get("train") or rec.get("prep") or {}).get("data_source")
    return rec


def _attempt(path: Path, cohort: str, arm: str, passes: int) -> int:
    n = 0
    if path.exists():
        for line in path.open():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("schema") == SCHEMA and r.get("cohort") == cohort and r.get("arm") == arm \
                    and r.get("prep_passes") == passes:
                n += 1
    return n + 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--api-url", default=os.environ.get("CLUSY_API_URL", "http://localhost:8010"))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--cohort", default=None, help="cohort label written into every record (default: the run_id)")
    ap.add_argument("--prep-passes", type=int, default=PREP_PASSES,
                    help="feature-extraction passes in PREP; the crossover sweep raises it to lengthen the CPU phase")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--gpu-profile", default="gpu_t4")
    ap.add_argument("--out", default=None, help="output directory (default results/e14; a scratch dir with --local)")
    ap.add_argument("--local", action="store_true", help="dry run against experiments/localapi.py LocalApi")
    ap.add_argument("--local-break", choices=("none", "rng_restore", "rng_perturb"), default="none",
                    help="break the simulated switch on purpose (LocalApi rng_restore=False / rng_perturb=True)")
    ap.add_argument("--keep-workdir", action="store_true", help="keep the local kernels' working directories")
    args = ap.parse_args()
    if args.prep_passes < 1:
        ap.error("--prep-passes must be >= 1 (ANALYSE compares against pass 0)")
    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        ap.error(f"unknown arms {bad}; choose from {ARMS}")

    if args.local:
        from localapi import LocalApi
        out = Path(args.out) if args.out else Path(tempfile.gettempdir()) / "clusy-e14-local"
        if (ROOT / "results") in out.resolve().parents or out.resolve() == (ROOT / "results").resolve():
            print("--local must not write under results/; pass a scratch --out", file=sys.stderr)
            return 2
        out.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix="e14-kernels-", dir=str(out)))
        api = LocalApi(rng_restore=args.local_break != "rng_restore", rng_perturb=args.local_break == "rng_perturb",
                       workdir=workdir)
        mover, api_url = move_file_local, None
    else:
        key = os.environ.get("CLUSY_HARNESS_API_KEY")
        if not key:
            print("export CLUSY_HARNESS_API_KEY", file=sys.stderr)
            return 1
        api, mover, api_url = Api(args.api_url, key), move_file_s3, args.api_url
        out = Path(args.out) if args.out else ROOT / "results" / "e14"
        why = runmeta.shipped_record_conflict(out / "e14_arms.jsonl", args.cohort)
        if why:
            print(why, file=sys.stderr)
            return 2
        out.mkdir(parents=True, exist_ok=True)

    meta = runmeta.start_run("e14", api_url=api_url, args=vars(args), cohort=args.cohort,
                             extra_files=["experiments/e14_analyse.py"] + (["experiments/localapi.py"] if args.local else []))
    cohort = args.cohort or meta["run_id"]
    meta["cohort"] = cohort
    meta["local"] = args.local
    if args.local:
        meta["local_break"] = args.local_break
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    path = out / "e14_arms.jsonl"
    log(f"E14 run {meta['run_id']} cohort {cohort} -> {path}" + (f"  [LOCAL dry run, break={args.local_break}]" if args.local else ""))
    rows = []
    try:
        for arm in arms:
            log(f"[{arm}]")
            attempt = _attempt(path, cohort, arm, args.prep_passes)
            r = run_arm(api, arm, steps=args.steps, passes=args.prep_passes, gpu=args.gpu_profile, mover=mover, log=log)
            r.update(schema=SCHEMA, experiment="e14", cohort=cohort, attempt=attempt, prep_passes=args.prep_passes,
                     steps=args.steps, seed=SEED, local_simulation=args.local)
            if args.local:
                r["local_break"] = args.local_break
                r["executed_on"] = "local cpu (every profile)"
            runmeta.stamp(r, meta)
            rows.append(r)
            with path.open("a") as f:
                f.write(json.dumps(r, default=str) + "\n")
            cost = r["est_runtime_cost_usd"].get(PRIMARY_RATE_TABLE)
            if r["outcome"] == "failed":
                log(f"  failed at {r.get('failed_phase')}: {(r.get('error') or '')[:160]}")
            else:
                log("  completed")
            log(f"  wall {r['wall_s']:.1f}s  runtime {json.dumps({k: round(v, 1) for k, v in r['runtime_s'].items()})}  "
                f"est. cost ${cost if cost is None else round(cost, 5)}  recomputed {r['work_recomputed_s']:.1f}s/"
                f"{r['work_recomputed_steps']} steps  app LOC {r['app_loc']}")
    finally:
        runmeta.finish_run(meta, api_url=api_url)
        runmeta.write_meta(meta, out / "runs_meta.jsonl")
        if args.local:
            api.close()
            if not args.keep_workdir:
                shutil.rmtree(workdir, ignore_errors=True)

    print(f"\n{'arm':<20} {'wall s':>7} {'cpu s':>7} {'gpu s':>7} {'est $':>9} {'recomp s':>8} {'LOC':>4} "
          f"{'acc':>7} {'tta acc':>8}  missing dependencies")
    for r in rows:
        a = r.get("analysis") or {}
        if r["outcome"] != "completed":
            print(f"{r['arm']:<20} FAILED at {r.get('failed_phase')}: {r.get('error', '')[:70]}")
            continue
        rs = r["runtime_s"]
        cost = r["est_runtime_cost_usd"].get(PRIMARY_RATE_TABLE) or 0.0
        tta = (a.get("tta_accuracy") or {}).get("accuracy")
        print(f"{r['arm']:<20} {r['wall_s']:>7.1f} {rs.get('cpu', 0.0):>7.1f} {rs.get(args.gpu_profile, 0.0):>7.1f} "
              f"{cost:>9.5f} {r['work_recomputed_s']:>8.1f} {r['app_loc']:>4} {a.get('accuracy', float('nan')):>7.4f} "
              f"{'-' if tta is None else f'{tta:.4f}':>8}  {', '.join(a.get('missing_dependencies', {}).values()) or 'none'}")

    if args.local:
        import e14_analyse
        recs = [r for r in e14_analyse.load(path)
                if r.get("cohort") == cohort and r.get("prep_passes") == args.prep_passes]
        # The arms THIS invocation asked for: one that failed, or left no
        # record, is a failed gate, never "not applicable".
        results = e14_analyse.checks(recs, requested=arms)
        hygiene_api = LocalApi(workdir=Path(tempfile.mkdtemp(prefix="e14-hygiene-", dir=str(out))))
        try:
            results.append(local_namespace_probe(hygiene_api))
        finally:
            hygiene_api.close()
            shutil.rmtree(hygiene_api.root, ignore_errors=True)
        print("\nlocal verification checks (cohort %s)" % cohort)
        e14_analyse.print_checks(results)
        return 0 if all(c["ok"] is not False for c in results if c.get("gate")) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
