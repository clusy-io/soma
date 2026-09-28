"""In-sandbox workload source for the resource-escalation experiments.

These strings are executed inside the sandbox kernel. They are kept here as one
source of truth so that calibration and the measured runs exercise **identical**
code — a hand-written calibration variant would be a different experiment
wearing the same label, and any drift would be invisible in the results.

Two workloads, two purposes:

  gpt2  GPT-2 small on WikiText-2. One escalation, with the OOM point swept
        across training progress. Answers "when does switching pay off?"
  vit   ViT-B/16 on CIFAR-10 with progressive resolution. Two escalations
        across three tiers. Answers "can the runtime be replaced repeatedly?"

Every cell prints a single `__ESC__:{...}` line that the driver parses. Nothing
is inferred from free-form output.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Shared prelude — determinism, memory accounting, and the OOM contract.
# --------------------------------------------------------------------------

PRELUDE = r'''
import json, os, time, gc, math
import torch

SEED = 20260825

def _emit(**kw):
    print("__ESC__:" + json.dumps(kw, default=str))

def _seed():
    import random, numpy as np
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"

def _gpu():
    if not torch.cuda.is_available():
        return {"name": "cpu", "vram_gb": None}
    p = torch.cuda.get_device_properties(0)
    return {"name": p.name, "vram_gb": round(p.total_memory / 2**30, 2)}

def _peak_gb():
    return round(torch.cuda.max_memory_allocated() / 2**30, 3) if torch.cuda.is_available() else None

def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def _free_after_oom():
    """Release the failed attempt.

    A CUDA OOM is a Python exception, not a process death, so the namespace
    survives it. The partially-allocated blocks are freed when the failed
    graph is collected; empty_cache then returns them to the driver so the
    subsequent capture has headroom to move tensors to host. Without this the
    capture can itself OOM and the experiment would measure the harness rather
    than the system.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
'''

# --------------------------------------------------------------------------
# GPT-2 small / WikiText-2
# --------------------------------------------------------------------------

GPT2_BUILD = PRELUDE + r'''
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

_seed()
_t0 = time.perf_counter()

tok = GPT2TokenizerFast.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(_dev())
model.gradient_checkpointing_disable()
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

# datasets 5.x requires a namespaced repo id; the bare "wikitext" alias that
# older tutorials use now raises HfUriError.
_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
_text = "\n\n".join(t for t in _ds["text"] if t.strip())
_ids = tok(_text, return_tensors="pt").input_ids[0]

# Drop the dataset and the tokenizer before any capture can see them.
#
# This is not tidiness, it is the contract. A `datasets.Dataset` is backed by a
# MEMORY-MAPPED Arrow table and the fast tokenizer is a Rust object; both pickle
# cleanly and then fail during load, which — because the capsule is
# all-or-nothing — destroys the entire namespace rather than just themselves.
# Measured: leaving them resident made every restore fail with
# `checkpoint_restore_failed`, the kernel rotate-and-retry fire twice, and the
# checkpoint retire. This is exactly the class-D construct E1 identified ("a
# successful dump does not imply a loadable capsule"), showing up unprompted in
# an ordinary HuggingFace training script.
#
# Neither object is needed after tokenisation: what the training loop consumes is
# the `_ids` tensor.
del _ds, _text, tok
gc.collect()

WORKLOAD = "gpt2-small/wikitext-2"
TOTAL_STEPS = TOTAL_STEPS_PLACEHOLDER
step_count = 0
loss_log = []

def make_batch(seq_len, batch, offset):
    need = seq_len * batch
    start = (offset * need) % max(1, (_ids.numel() - need - 1))
    chunk = _ids[start:start + need].view(batch, seq_len)
    return chunk.to(_dev())

def train_steps(n, seq_len, batch):
    """Run n optimizer steps. Raises torch.cuda.OutOfMemoryError on OOM."""
    global step_count
    model.train()
    t0 = time.perf_counter()
    for _ in range(n):
        x = make_batch(seq_len, batch, step_count)
        out = model(x, labels=x)
        out.loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        step_count += 1
        loss_log.append(float(out.loss.detach()))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0

_emit(event="built", workload=WORKLOAD, gpu=_gpu(),
      params=sum(p.numel() for p in model.parameters()),
      tokens=int(_ids.numel()), total_steps=TOTAL_STEPS,
      build_s=round(time.perf_counter() - _t0, 2))
'''

# --------------------------------------------------------------------------
# ViT-B/16 / CIFAR-10, progressive resolution
# --------------------------------------------------------------------------

VIT_BUILD = PRELUDE + r'''
import torchvision
from torchvision.models import vit_b_16
from torchvision.models.vision_transformer import interpolate_embeddings

_seed()
_t0 = time.perf_counter()

WORKLOAD = "vit-b-16/cifar-10"
TOTAL_STEPS = TOTAL_STEPS_PLACEHOLDER
cur_res = INITIAL_RES_PLACEHOLDER
step_count = 0
loss_log = []

# Full fine-tune, not head-only: the point is to carry meaningful model AND
# optimizer state across the boundary, and a frozen backbone would carry almost
# none.
model = vit_b_16(weights=None, image_size=cur_res, num_classes=10).to(_dev())
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# Real CIFAR-10, pulled from the HF hub rather than torchvision's mirror: that
# mirror throttles to ~80 kB/s from inside the sandbox, which is 35 minutes for
# 170 MB. Substituting synthetic data would have been faster and would have made
# the sentence "CIFAR-10" false in a way no reader could check, so the source
# changed instead of the dataset.
#
# The cache lives in /tmp, deliberately outside the persisted workspace, so the
# dataset is not counted as workspace payload in any switch measurement.
os.environ.setdefault("HF_HOME", "/tmp/clusy-hf")
from datasets import load_dataset as _load_ds
_ds = _load_ds("uoft-cs/cifar10", split="train[:512]")
import numpy as _np
_imgs = torch.from_numpy(
    _np.stack([_np.asarray(im, dtype=_np.float32) / 255.0 for im in _ds["img"]])
).permute(0, 3, 1, 2).contiguous()
_labs = torch.tensor(_ds["label"], dtype=torch.long)
# Same reason as the GPT-2 path: the Dataset wraps a memory-mapped Arrow table,
# which survives capture and then fails the load, taking the whole namespace with
# it. The tensors above are the only thing the training loop needs.
del _ds
gc.collect()

def make_batch(res, batch, offset):
    idx = torch.arange(offset * batch, offset * batch + batch) % _imgs.shape[0]
    x = _imgs[idx]
    x = torch.nn.functional.interpolate(x, size=(res, res), mode="bilinear",
                                        align_corners=False)
    return x.to(_dev()), _labs[idx].to(_dev())

def escalate_resolution(new_res):
    """Reshape the model for a higher input resolution.

    This is a WORKLOAD decision, not a system operation, and it is the standard
    progressive-resizing recipe: every transformer block carries over unchanged,
    and only the positional embedding is interpolated to the new patch count.
    The optimizer is rebuilt and its moments are restored for every parameter
    whose shape is unchanged — which is all of them except `pos_embedding` — so
    what the escalation costs is one tensor's optimizer moments, not the run.
    """
    global model, opt, cur_res
    old_sd = model.state_dict()
    new_sd = interpolate_embeddings(new_res, 16, old_sd)
    new_model = vit_b_16(weights=None, image_size=new_res, num_classes=10)
    new_model.load_state_dict(new_sd)
    new_model = new_model.to(_dev())

    old_state = opt.state_dict()
    old_params = list(model.parameters())
    new_params = list(new_model.parameters())
    carried, reset = 0, 0
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-4)
    new_opt_state = {"state": {}, "param_groups": new_opt.state_dict()["param_groups"]}
    for i, (op, np_) in enumerate(zip(old_params, new_params)):
        st = old_state["state"].get(i)
        if st is not None and op.shape == np_.shape:
            new_opt_state["state"][i] = st; carried += 1
        elif st is not None:
            reset += 1
    new_opt.load_state_dict(new_opt_state)

    model, opt, cur_res = new_model, new_opt, new_res
    return {"carried": carried, "reset": reset, "res": new_res}

def train_steps(n, res, batch):
    global step_count
    model.train()
    t0 = time.perf_counter()
    for _ in range(n):
        x, y = make_batch(res, batch, step_count)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        step_count += 1
        loss_log.append(float(loss.detach()))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0

_emit(event="built", workload=WORKLOAD, gpu=_gpu(),
      params=sum(p.numel() for p in model.parameters()),
      total_steps=TOTAL_STEPS, res=cur_res,
      build_s=round(time.perf_counter() - _t0, 2))
'''

# --------------------------------------------------------------------------
# Cells reused by both workloads
# --------------------------------------------------------------------------

#: Run a phase and report throughput. Does NOT catch OOM — the caller decides
#: whether an OOM here is expected.
RUN_PHASE = r'''
_reset_peak()
_dur = train_steps({n}, {dim}, {batch})
_emit(event="phase_ok", steps_done=step_count, phase_steps={n},
      seconds=round(_dur, 3), per_step_s=round(_dur / max(1, {n}), 4),
      peak_gb=_peak_gb(), loss=round(loss_log[-1], 4) if loss_log else None)
'''

#: Attempt the escalated phase and let it genuinely fail. The system must not
#: anticipate the OOM: the workload really attempts the higher-demand phase and
#: really receives CUDA OOM, and only then is the failed attempt discarded.
ATTEMPT_OOM = r'''
_reset_peak()
_oom, _err, _dur = False, None, None
_t = time.perf_counter()
try:
    _dur = train_steps({n}, {dim}, {batch})
except torch.cuda.OutOfMemoryError as e:
    _oom = True; _err = str(e)[:200]
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        _oom = True; _err = str(e)[:200]
    else:
        raise
_elapsed = time.perf_counter() - _t
if _oom:
    _free_after_oom()
_emit(event="oom_attempt", oom=_oom, error=_err,
      steps_done=step_count, attempted_steps={n},
      elapsed_s=round(_elapsed, 3), peak_gb=_peak_gb(),
      freed_gb=round(torch.cuda.memory_allocated()/2**30, 3) if torch.cuda.is_available() else None)
'''

#: Confirm the namespace survived the switch and the workload can continue.
VERIFY_AFTER_SWITCH = r'''
_emit(event="post_switch", gpu=_gpu(), steps_done=step_count,
      device=str(next(model.parameters()).device),
      opt_is_optimizer=isinstance(opt, torch.optim.Optimizer),
      opt_step=float(list(opt.state.values())[0]["step"]) if opt.state else None,
      loss_len=len(loss_log), res=(cur_res if "cur_res" in dir() else None))
'''
