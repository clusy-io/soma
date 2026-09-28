"""E13 analysis: read one cohort of chain and control records and say what they show.

This module is the single place where E13's comparisons are defined. The
harness (`e13_chain.py`) imports `rng_verdicts` to stamp each hop record and
`summarise` to run its local self-checks, so the logic that is dry-run against
`LocalApi` is exactly the logic that later reads the live records.

What a cohort is expected to contain (default `--chains` of the harness):

  control_t4_a, control_t4_b   gpu_t4, one uninterrupted process each, same
                               seed, identical per-block operations to the
                               `same` chain. Two of them measure the
                               run-to-run noise floor on one SKU.
  same                         gpu_t4 with a real switch (RAM tier change)
                               between blocks. Compared BITWISE with both T4
                               controls (every loss, the per-block parameter,
                               optimizer, loader and scheduler state, and the
                               full content of every RNG stream at the end of
                               each train block and the start of each verify
                               block): if control_a equals control_b, the
                               platform is deterministic on this workload and
                               any difference between `same` and a control is
                               attributable to the switch; if the controls
                               already differ, inequality proves nothing.
  control_cpu                  cpu, nine blocks. An observational reference
                               for the hetero chain, which spans devices, so no
                               single-device control can match it bitwise.
  hetero                       T4 / CPU / A100 route. Reported per hop: RNG
                               verdicts, and the drift of two recomputed
                               quantities (fixed-batch loss, forward-output sum)
                               under the process's default flags and with TF32
                               disabled on both ends of the hop.

Usage:
  python experiments/e13_analyse.py                       # latest cohort in results/e13
  python experiments/e13_analyse.py --in <jsonl> --cohort <id> --out <json>

When the input is the canonical results file and no --out is given, the
summary is written to results/e13/e13_summary.json (regenerated from the
records, so overwriting it is correct). For any other input nothing is written
unless --out is given.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_IN = ROOT / "results" / "e13" / "e13_chains.jsonl"
CANONICAL_OUT = ROOT / "results" / "e13" / "e13_summary.json"

#: The three process-global streams a capsule's RNG envelope carries.
GLOBAL_STREAMS = ("python", "numpy", "torch_cpu")

#: Comparisons the summary makes, with the role each one plays. Only the first
#: two can support a claim about the switch, and only when the noise floor
#: (third) shows the controls agree with each other.
COMPARISONS = (
    ("same", "control_t4_a", "exact test: same-device switch vs uninterrupted control"),
    ("same", "control_t4_b", "exact test: same-device switch vs uninterrupted control"),
    ("control_t4_a", "control_t4_b", "noise floor: two uninterrupted runs on one SKU"),
    ("hetero", "control_cpu", "observational: the hetero chain spans devices, so no bitwise agreement is expected"),
)
STEP_PAIRS = (("same", "control_t4_a"), ("same", "control_t4_b"), ("hetero", "control_cpu"))


# ---------------------------------------------------------------------------
# RNG verdicts
# ---------------------------------------------------------------------------

def _stream_verdict(src: dict | None, dst: dict | None) -> dict[str, Any]:
    """`equal` needs BOTH the full-state hash and the clone draws to agree.

    The hash covers the whole state (for Python and NumPy including the stream
    position and the cached Gaussian), and the draws show that the state
    produces the same next values, which is what a user's program observes.
    """
    if not src or not dst or src.get("loaded") is False or dst.get("loaded") is False:
        side = "source" if (not src or src.get("loaded") is False) else "destination"
        return {"verdict": "not_applicable", "reason": f"stream not loaded on the {side}"}
    if "sha256" not in src or "sha256" not in dst:
        return {"verdict": "not_applicable", "reason": "no fingerprint recorded"}
    hash_eq = src["sha256"] == dst["sha256"]
    draws_eq = src.get("draws") == dst.get("draws")
    out = {"verdict": "equal" if hash_eq and draws_eq else "state_differs",
           "hash_equal": hash_eq, "draws_equal": draws_eq}
    if hash_eq and not draws_eq:
        out["note"] = "state bytes equal but the clone draws differ (device-dependent generation)"
    return out


def _cuda_verdict(src: dict | None, dst: dict | None) -> dict[str, Any]:
    src, dst = src or {}, dst or {}
    s_fp = bool(src.get("fingerprinted") and src.get("devices"))
    d_fp = bool(dst.get("fingerprinted") and dst.get("devices"))
    s_hashes = [d["sha256"] for d in src.get("devices") or []]
    d_hashes = [d["sha256"] for d in dst.get("devices") or []]
    if not src.get("available") and not dst.get("available"):
        return {"verdict": "not_applicable", "reason": "no CUDA on either side"}
    if s_fp and not dst.get("available"):
        # The capsule carries the CUDA generator states, and a destination
        # without CUDA cannot hold them: the restore drops them by design.
        return {"verdict": "not_applicable", "reason": "destination has no CUDA: the source CUDA stream is a declared drop",
                "declared_drop": True, "source_sha256": s_hashes, "source_device_count": src.get("device_count")}
    if not s_fp:
        reason = ("source had no CUDA" if not src.get("available")
                  else "source had no initialized CUDA state")
        out = {"verdict": "not_applicable", "reason": reason + ": the destination CUDA stream was not carried",
               "destination_stream_carried": False}
        if d_fp:
            out["destination_sha256"] = d_hashes
            out["destination_device_count"] = dst.get("device_count")
        return out
    if not d_fp:
        return {"verdict": "not_applicable", "reason": "destination CUDA not fingerprinted",
                "source_sha256": s_hashes}
    if len(s_hashes) != len(d_hashes):
        return {"verdict": "state_differs", "reason": f"device count {len(s_hashes)} -> {len(d_hashes)}",
                "source_sha256": s_hashes, "destination_sha256": d_hashes}
    per = [_stream_verdict(a, b) for a, b in zip(src["devices"], dst["devices"])]
    out = {"verdict": "equal" if all(p["verdict"] == "equal" for p in per) else "state_differs",
           "per_device": per, "source_devices": [d.get("name") for d in src["devices"]],
           "destination_devices": [d.get("name") for d in dst["devices"]]}
    return out


def rng_verdicts(src_fp: dict | None, dst_fp: dict | None) -> dict[str, Any]:
    """Per-stream verdicts between a source fingerprint (end of a training
    block) and a destination fingerprint (start of the next program)."""
    src_fp, dst_fp = src_fp or {}, dst_fp or {}
    out: dict[str, Any] = {s: _stream_verdict(src_fp.get(s), dst_fp.get(s)) for s in GLOBAL_STREAMS}
    out["cuda"] = _cuda_verdict(src_fp.get("cuda"), dst_fp.get("cuda"))
    named: dict[str, Any] = {}
    s_named, d_named = src_fp.get("named") or {}, dst_fp.get("named") or {}
    for name in sorted(set(s_named) | set(d_named)):
        a, b = s_named.get(name), d_named.get(name)
        if a is None:
            named[name] = {"verdict": "not_applicable", "reason": "absent at the source"}
        elif b is None:
            named[name] = {"verdict": "state_differs", "reason": "absent at the destination"}
        elif a.get("kind") != b.get("kind"):
            named[name] = {"verdict": "state_differs", "reason": f"type changed: {a.get('kind')} -> {b.get('kind')}"}
        elif "error" in a or "error" in b:
            named[name] = {"verdict": "not_applicable", "reason": f"not fingerprintable: {a.get('error') or b.get('error')}"}
        else:
            named[name] = _stream_verdict(a, b)
    out["named"] = named
    return out


def verdict_letter(v: dict | None) -> str:
    return (v or {}).get("verdict", "missing")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_cohort(path: Path, cohort: str | None = None) -> tuple[str | None, dict[str, dict], list[str]]:
    """Records of one cohort, keyed by chain name. Records without a cohort
    predate this schema and are ignored. The default cohort is the cohort of
    the last record in the file."""
    recs = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    recs = [r for r in recs if r.get("cohort")]
    if not recs:
        return None, {}, []
    if cohort is None:
        cohort = recs[-1]["cohort"]
    chains: dict[str, dict] = {}
    notes: list[str] = []
    for r in recs:
        if r["cohort"] != cohort:
            continue
        if r["chain"] in chains:
            notes.append(f"{r['chain']}: more than one record in cohort {cohort}; the last one is used")
        chains[r["chain"]] = r
    return cohort, chains, notes


def entries(rec: dict) -> list[dict]:
    """Per-block entries: chain hops or control blocks, in order."""
    return list(rec.get("hops") or rec.get("blocks") or [])


def entry_label(e: dict) -> str:
    if "hop" in e:
        return f"{e['hop']} {e.get('from', '?')}>{e.get('to', '?')}"
    return f"b{e.get('block')}"


# ---------------------------------------------------------------------------
# Bitwise trajectory comparison
# ---------------------------------------------------------------------------

def _hex(x: Any) -> str | None:
    return None if x is None else float(x).hex()


def rng_stream_view(fp: dict | None) -> dict[str, Any] | None:
    """The stream content of an RNG fingerprint (full-state hash and clone
    draws of every stream), without the fields that describe the process
    rather than a stream (`cuda.initialized_before`, reasons, seeds). Two runs
    whose views are equal hold the same state in every RNG stream."""
    if not fp:
        return None

    def s(x: dict | None) -> Any:
        x = x or {}
        if x.get("loaded") is False:
            return "not loaded"
        if "error" in x:
            return "error"
        return [x.get("sha256"), x.get("draws")]

    cu = fp.get("cuda") or {}
    if cu.get("fingerprinted"):
        cuda: Any = [s(d) for d in cu.get("devices") or []]
    else:
        cuda = "not fingerprinted" if cu.get("available") else "no CUDA"
    return {"python": s(fp.get("python")), "numpy": s(fp.get("numpy")), "torch_cpu": s(fp.get("torch_cpu")),
            "cuda": cuda, "named": {k: s(v) for k, v in (fp.get("named") or {}).items()}}


def _view_diff(a: dict | None, b: dict | None) -> list[str]:
    """Names of the streams whose content differs between two views."""
    if a is None or b is None:
        return [] if a is b else ["missing"]
    out = [k for k in (*GLOBAL_STREAMS, "cuda") if a.get(k) != b.get(k)]
    an, bn = a.get("named") or {}, b.get("named") or {}
    out += [f"named:{k}" for k in sorted(set(an) | set(bn)) if an.get(k) != bn.get(k)]
    return out


def block_trace(rec: dict) -> list[dict]:
    """The per-block fingerprints, in order: what the train block printed,
    plus the full RNG stream content at the end of the train block (the
    source fingerprint) and at the start of the verify block that follows it
    (the destination fingerprint, after the switch on a chain). The RNG views
    are part of the exact test because a switch that loses a global stream
    leaves the loss trajectory untouched whenever training draws only from
    the loader's own generator, as this workload does."""
    out = []
    for e in entries(rec):
        pre = e.get("pre") or {}
        blk = pre.get("block")
        if not blk:
            continue
        out.append({"losses": blk.get("losses") or [], "param_digest": blk.get("param_digest"),
                    "optimizer_digest": blk.get("optimizer_digest"),
                    "loader_generator_sha256": blk.get("loader_generator_sha256"),
                    "scheduler": blk.get("scheduler"), "step_count": blk.get("step_count"),
                    "fixed_batch_loss": pre.get("fixed_batch_loss"),
                    "rng_train_end": rng_stream_view(pre.get("rng")),
                    "rng_verify_start": rng_stream_view((e.get("post") or {}).get("rng"))})
    return out


_FIELDS = ("losses", "fixed_batch_loss", "param_digest", "optimizer_digest", "loader_generator_sha256",
           "scheduler", "step_count", "rng_train_end", "rng_verify_start")


def compare_runs(a_rec: dict, b_rec: dict) -> dict[str, Any]:
    """Bitwise comparison of two runs, block by block, then at the end.

    Losses are compared as `float.hex` strings, so equality is equality of
    the bits and a NaN compares equal to a NaN. The per-block parameter and
    optimizer digests make the comparison independent of loss floats alone:
    two runs whose losses agree but whose weights do not are reported unequal.
    The per-block RNG views do the same for process randomness: two runs whose
    weights agree but whose Python, NumPy, torch or CUDA stream does not are
    reported unequal, with the differing streams named.
    """
    A, B = block_trace(a_rec), block_trace(b_rec)
    steps = a_rec.get("steps_per_hop") or 0
    n = min(len(A), len(B))
    per_block, first, max_abs = [], None, 0.0
    for i in range(n):
        a, b = A[i], B[i]
        la, lb = [_hex(x) for x in a["losses"]], [_hex(x) for x in b["losses"]]
        row = {"block": i + 1,
               "losses_equal": la == lb,
               "fixed_batch_loss_equal": _hex(a["fixed_batch_loss"]) == _hex(b["fixed_batch_loss"]),
               "param_digest_equal": a["param_digest"] == b["param_digest"],
               "optimizer_digest_equal": a["optimizer_digest"] == b["optimizer_digest"],
               "loader_generator_sha256_equal": a["loader_generator_sha256"] == b["loader_generator_sha256"],
               "scheduler_equal": a["scheduler"] == b["scheduler"],
               "step_count_equal": a["step_count"] == b["step_count"],
               "rng_train_end_equal": a["rng_train_end"] == b["rng_train_end"],
               "rng_verify_start_equal": a["rng_verify_start"] == b["rng_verify_start"]}
        row["bitwise_equal"] = all(v for k, v in row.items() if k.endswith("_equal"))
        rng_streams = {f: _view_diff(a[f], b[f]) for f in ("rng_train_end", "rng_verify_start") if not row[f + "_equal"]}
        if rng_streams:
            row["rng_streams_differ"] = rng_streams
        for x, y in zip(a["losses"], b["losses"]):
            if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
                max_abs = max(max_abs, abs(x - y))
        if first is None and not row["bitwise_equal"]:
            first = {"block": i + 1, "fields": [k[:-6] for k, v in row.items() if k.endswith("_equal") and k != "bitwise_equal" and not v]}
            if rng_streams:
                first["rng_streams"] = rng_streams
            if not row["losses_equal"]:
                k = next((j for j, (x, y) in enumerate(zip(la, lb)) if x != y), min(len(la), len(lb)))
                first.update({"step_in_block": k + 1, "global_step": i * steps + k + 1,
                              "a": a["losses"][k] if k < len(la) else None,
                              "b": b["losses"][k] if k < len(lb) else None})
        per_block.append(row)
    fa, fb = a_rec.get("final") or {}, b_rec.get("final") or {}
    final = {"present": bool(fa) and bool(fb)}
    if final["present"]:
        final.update({
            "loss_trace_equal": [_hex(x) for x in fa.get("loss_trace") or []] == [_hex(x) for x in fb.get("loss_trace") or []],
            "param_digest_equal": fa.get("param_digest") == fb.get("param_digest"),
            "optimizer_digest_equal": fa.get("optimizer_digest") == fb.get("optimizer_digest"),
            "optimizer_step_equal": fa.get("optimizer_step") == fb.get("optimizer_step"),
            "optimizer_step": [fa.get("optimizer_step"), fb.get("optimizer_step")],
            "rng_equal": rng_stream_view(fa.get("rng")) == rng_stream_view(fb.get("rng")),
        })
        if not final["rng_equal"]:
            final["rng_streams_differ"] = _view_diff(rng_stream_view(fa.get("rng")), rng_stream_view(fb.get("rng")))
        final["bitwise_equal"] = all(v for k, v in final.items() if k.endswith("_equal"))
    same_len = len(A) == len(B) and len(A) > 0
    return {"a": a_rec.get("chain"), "b": b_rec.get("chain"), "blocks_a": len(A), "blocks_b": len(B),
            "blocks_compared": n, "same_length": same_len,
            "bitwise_equal": bool(same_len and all(r["bitwise_equal"] for r in per_block) and final.get("bitwise_equal", False)),
            "blocks_equal": sum(1 for r in per_block if r["bitwise_equal"]),
            "first_divergence": first, "max_abs_loss_diff": max_abs, "final": final, "per_block": per_block}


# ---------------------------------------------------------------------------
# Drift of recomputed quantities, default flags vs TF32 off
# ---------------------------------------------------------------------------

def _rel(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs(b - a) / max(abs(a), 1e-12)


def describe_divergence(fd: dict | None) -> str:
    """One line for a first divergence: where, which fields, and for an RNG
    field which streams."""
    if not fd:
        return ""
    txt = f"block {fd['block']} {fd['fields']}"
    if fd.get("global_step"):
        txt += f" step {fd['global_step']}"
    for field, streams in (fd.get("rng_streams") or {}).items():
        txt += f" {field}: {', '.join(streams)}"
    return txt


def _on_cuda(t: dict) -> bool:
    """Whether the model computed on a GPU at this end. Keyed on the device
    the parameters were on, not on whether the host has CUDA: after a CPU ->
    GPU switch the restored model is still on the CPU when verify runs."""
    if "compute_on_cuda" in t:
        return bool(t["compute_on_cuda"])
    return str(t.get("compute_device") or "").startswith("cuda")


def _tf32_sides(t: dict) -> tuple[dict, dict, bool]:
    """(default values, TF32-off values, whether TF32-off was measured). The
    TF32 flags govern CUDA matmul and cuDNN only, so where the model computed
    on the CPU the program records the values once and TF32-off IS the
    default. A TF32-off value recorded for a CPU-resident model (the program
    before this was corrected keyed on host CUDA) is ignored for the same
    reason: it cannot differ from the default by anything TF32 does."""
    d = t.get("default") or {}
    off = t.get("tf32_off")
    measured = off is not None and _on_cuda(t)
    return d, (off if measured else d), measured


def drift_rows(rec: dict) -> list[dict]:
    rows = []
    for e in entries(rec):
        pre, post = e.get("pre") or {}, e.get("post") or {}
        if not pre.get("tf32") or not post.get("tf32"):
            continue
        pt, qt = pre["tf32"], post["tf32"]
        pd, po, p_meas = _tf32_sides(pt)
        qd, qo, q_meas = _tf32_sides(qt)
        p_cuda, q_cuda = _on_cuda(pt), _on_cuda(qt)
        rows.append({
            "hop": e.get("hop", e.get("block")), "from": e.get("from"), "to": e.get("to"),
            "src_device": pre.get("device"), "dst_device": post.get("device"),
            # Where the two recomputed values were actually computed. On a
            # CPU -> GPU hop both are "cpu": the profile says GPU, the
            # arithmetic was a CPU host's.
            "src_compute": pt.get("compute_device"), "dst_compute": qt.get("compute_device"),
            "src_on_cuda": p_cuda, "dst_on_cuda": q_cuda,
            "src_gpu": pt.get("device_name") if p_cuda else None, "dst_gpu": qt.get("device_name") if q_cuda else None,
            "src_capability": pt.get("capability") if p_cuda else None,
            "dst_capability": qt.get("capability") if q_cuda else None,
            "src_default_flags": pt.get("default_flags"), "dst_default_flags": qt.get("default_flags"),
            "default_flags_equal": pt.get("default_flags") == qt.get("default_flags"),
            "tf32_off_measured": {"src": p_meas, "dst": q_meas},
            "loss_rel_default": _rel(pd.get("fixed_batch_loss"), qd.get("fixed_batch_loss")),
            "loss_rel_tf32_off": _rel(po.get("fixed_batch_loss"), qo.get("fixed_batch_loss")),
            "forward_rel_default": _rel(pd.get("forward_sum"), qd.get("forward_sum")),
            "forward_rel_tf32_off": _rel(po.get("forward_sum"), qo.get("forward_sum")),
            # Did disabling TF32 change the value computed on that end at all?
            "tf32_changed_src_value": p_meas and (_hex(pd.get("fixed_batch_loss")) != _hex(po.get("fixed_batch_loss"))
                                                  or _hex(pd.get("forward_sum")) != _hex(po.get("forward_sum"))),
            "tf32_changed_dst_value": q_meas and (_hex(qd.get("fixed_batch_loss")) != _hex(qo.get("fixed_batch_loss"))
                                                  or _hex(qd.get("forward_sum")) != _hex(qo.get("forward_sum"))),
        })
    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _rng_rows(rec: dict) -> list[dict]:
    rows = []
    for e in entries(rec):
        pre, post = e.get("pre") or {}, e.get("post") or {}
        if not pre.get("rng") or not post.get("rng"):
            continue
        v = rng_verdicts(pre["rng"], post["rng"])
        rows.append({"entry": entry_label(e), "hop": e.get("hop"), "block": e.get("block"),
                     "from": e.get("from"), "to": e.get("to"),
                     **{s: verdict_letter(v[s]) for s in GLOBAL_STREAMS},
                     "cuda": verdict_letter(v["cuda"]), "cuda_reason": v["cuda"].get("reason"),
                     "cuda_declared_drop": bool(v["cuda"].get("declared_drop")),
                     "named": {k: verdict_letter(x) for k, x in v["named"].items()},
                     "stored_verdicts_agree": (e.get("rng_verdicts") is None or e["rng_verdicts"] == v),
                     "fingerprint_idempotent": bool(pre.get("rng_fingerprint_idempotent")) and bool(post.get("rng_fingerprint_idempotent"))})
    return rows


def _timing_rows(rec: dict) -> list[dict]:
    """Per-hop switch and first-execute times, with and without the
    destination RNG fingerprint. On a CPU -> GPU hop that fingerprint is the
    first thing to initialize CUDA, so only the `excl_fingerprint` column is
    comparable with runs (or hop types) where it did not."""
    rows = []
    for e in rec.get("hops") or []:
        post = e.get("post") or {}
        if not post:
            continue
        rows.append({"hop": e.get("hop"), "from": e.get("from"), "to": e.get("to"),
                     "switch_s": e.get("switch_s"), "first_execute_s": e.get("first_execute_s"),
                     "first_execute_excl_fingerprint_s": e.get("first_execute_excl_fingerprint_s"),
                     "rng_fingerprint_s": post.get("rng_fingerprint_s"),
                     "cuda_initialized_by_fingerprint": post.get("cuda_initialized_by_fingerprint")})
    return rows


def _chain_overview(rec: dict) -> dict[str, Any]:
    es = entries(rec)
    posts = [e.get("post") for e in es if e.get("post")]
    failed = {entry_label(e): e["post"]["oracles_failed"] for e in es if e.get("post") and e["post"].get("oracles_failed")}
    fin = rec.get("final") or {}
    return {
        "kind": rec.get("kind"), "route": rec.get("route"), "profile": rec.get("profile"),
        "entries": len(es), "verified": len(posts), "error": rec.get("error"),
        "data_source": rec.get("data_source"),
        "local_simulation": rec.get("local_simulation", False),
        "oracles_all_pass": bool(posts) and not failed, "oracles_failed": failed,
        "tied_identity_all": all(p.get("tied_identity") for p in posts) if posts else None,
        "view_shares_storage": [p.get("view_shares_storage") for p in posts],
        "step_count_matches_all": all(p.get("step_count_matches") for p in posts) if posts else None,
        "loss_continuous_all": all(p.get("loss_continuous") for p in posts) if posts else None,
        "final_optimizer_step": fin.get("optimizer_step"), "final_step_count": fin.get("step_count"),
        "run_id": (rec.get("run") or {}).get("run_id"),
    }


def summarise(chains: dict[str, dict], cohort: str | None = None, notes: list[str] | None = None) -> dict[str, Any]:
    s: dict[str, Any] = {"cohort": cohort, "chains_present": sorted(chains), "notes": list(notes or [])}
    s["chains"] = {name: _chain_overview(rec) for name, rec in chains.items()}
    s["rng"] = {name: _rng_rows(rec) for name, rec in chains.items()}
    s["fingerprint_idempotent_all"] = all(r["fingerprint_idempotent"] for rows in s["rng"].values() for r in rows)
    comps = {}
    for a, b, role in COMPARISONS:
        if a in chains and b in chains:
            comps[f"{a}_vs_{b}"] = {"role": role, **compare_runs(chains[a], chains[b])}
    s["comparisons"] = comps
    steps = {}
    for a, b in STEP_PAIRS:
        if a in chains and b in chains:
            sa = (chains[a].get("final") or {}).get("optimizer_step")
            sb = (chains[b].get("final") or {}).get("optimizer_step")
            steps[f"{a}_vs_{b}"] = {a: sa, b: sb, "equal": sa is not None and sa == sb}
    s["step_counts"] = steps
    s["drift"] = {name: drift_rows(chains[name]) for name in ("hetero", "same") if name in chains}
    s["timing"] = {name: _timing_rows(rec) for name, rec in chains.items() if rec.get("kind") == "chain"}
    s["interpretation"] = _interpret(s)
    return s


def _interpret(s: dict) -> dict[str, str]:
    """Plain statements that follow mechanically from the comparisons, so the
    reading of the exact test is fixed before anyone looks at the numbers.
    The exact test covers the loss trajectory, the per-block parameter,
    optimizer, loader and scheduler state, and the full content of every RNG
    stream at the end of each training block and the start of each verify."""
    out = {}
    c = s["comparisons"]
    floor = c.get("control_t4_a_vs_control_t4_b")
    for key in ("same_vs_control_t4_a", "same_vs_control_t4_b"):
        if key not in c:
            continue
        eq = c[key]["bitwise_equal"]
        fd = c[key]["first_divergence"] or {}
        rng_only = bool(fd) and set(fd.get("fields") or []) <= {"rng_train_end", "rng_verify_start"}
        if floor is None:
            out[key] = ("bitwise equal" if eq else "differs") + "; no noise-floor pair in this cohort, so a difference is not attributable"
        elif floor["bitwise_equal"]:
            out[key] = ("bitwise equal, and the two controls agree bitwise: the switch left the trajectory, the training state "
                        "and every RNG stream unchanged" if eq else
                        "differs while the two controls agree bitwise: the difference is attributable to the switch"
                        + (" (first seen in RNG stream state only, before any loss or weight differs)" if rng_only else ""))
        else:
            out[key] = ("bitwise equal, but the two controls themselves differ (run-to-run nondeterminism)" if eq else
                        "differs, and so do the two controls: run-to-run nondeterminism on this SKU; the difference is not attributable to the switch")
    if "hetero_vs_control_cpu" in c:
        out["hetero_vs_control_cpu"] = ("observational only (the chain spans devices): "
                                        + ("bitwise equal" if c["hetero_vs_control_cpu"]["bitwise_equal"] else
                                           f"differs, max |loss difference| {c['hetero_vs_control_cpu']['max_abs_loss_diff']:.3g}"))
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_SHORT = {"equal": "equal", "state_differs": "DIFFERS", "not_applicable": "n/a", "missing": "-"}


def render(s: dict) -> str:
    L = [f"E13 summary, cohort {s['cohort']}", "=" * 60]
    for n in s.get("notes") or []:
        L.append(f"note: {n}")
    for name, ov in s["chains"].items():
        L.append(f"{name:<13} {ov['kind'] or '?':<8} entries={ov['entries']} verified={ov['verified']} "
                 f"oracles_all_pass={ov['oracles_all_pass']} final_opt_step={ov['final_optimizer_step']} "
                 f"data={ov['data_source']}" + (f" ERROR={str(ov['error'])[:80]}" if ov["error"] else ""))
    L.append("")
    L.append("RNG verdicts (source = end of the training block, destination = start of the next program)")
    for name, rows in s["rng"].items():
        if not rows:
            continue
        L.append(f"  {name}")
        L.append(f"    {'entry':<22}{'python':<9}{'numpy':<9}{'torch':<9}{'cuda':<9}named")
        for r in rows:
            named = " ".join(f"{k}={_SHORT.get(v, v)}" for k, v in r["named"].items())
            L.append(f"    {r['entry']:<22}{_SHORT.get(r['python']):<9}{_SHORT.get(r['numpy']):<9}"
                     f"{_SHORT.get(r['torch_cpu']):<9}{_SHORT.get(r['cuda']):<9}{named}"
                     + ("" if r["fingerprint_idempotent"] else "  [fingerprint NOT idempotent]"))
    L.append(f"  fingerprint idempotent everywhere: {s['fingerprint_idempotent_all']}")
    L.append("")
    L.append("Bitwise trajectory comparisons")
    for key, c in s["comparisons"].items():
        fd = c["first_divergence"]
        L.append(f"  {key:<32} equal={c['bitwise_equal']} blocks {c['blocks_equal']}/{c['blocks_compared']} "
                 f"(lengths {c['blocks_a']}/{c['blocks_b']}) max|dloss|={c['max_abs_loss_diff']:.3g}"
                 + (f" first divergence: {describe_divergence(fd)}" if fd else ""))
        L.append(f"  {'':<32} role: {c['role']}")
    for key, txt in s["interpretation"].items():
        L.append(f"  => {key}: {txt}")
    L.append("")
    L.append("Final optimizer step, chain vs control")
    for key, v in s["step_counts"].items():
        L.append(f"  {key:<28} {v}")
    for name, rows in s["drift"].items():
        if not rows:
            continue
        L.append("")
        L.append(f"{name}: drift of recomputed values across each hop (relative); computed on = where the model's "
                 "parameters were at each end")
        L.append(f"    {'hop':<6}{'from':<13}{'to':<13}{'computed on':<14}{'loss dflt':<12}{'loss tf32off':<14}"
                 f"{'fwd dflt':<12}{'fwd tf32off':<12}")
        for r in rows:
            f = lambda x: "-" if x is None else f"{x:.2e}"  # noqa: E731
            on = f"{'gpu' if r['src_on_cuda'] else 'cpu'}>{'gpu' if r['dst_on_cuda'] else 'cpu'}"
            L.append(f"    {str(r['hop']):<6}{str(r['from']):<13}{str(r['to']):<13}{on:<14}{f(r['loss_rel_default']):<12}"
                     f"{f(r['loss_rel_tf32_off']):<14}{f(r['forward_rel_default']):<12}{f(r['forward_rel_tf32_off']):<12}")
    for name, rows in (s.get("timing") or {}).items():
        if not rows:
            continue
        L.append("")
        L.append(f"{name}: first execute after each switch, with and without the destination RNG fingerprint")
        L.append(f"    {'hop':<6}{'from':<13}{'to':<13}{'switch s':<10}{'first-exec s':<14}{'excl. fp s':<12}{'fp s':<9}CUDA init by fp")
        for r in rows:
            fp_s = "-" if r["rng_fingerprint_s"] is None else f"{r['rng_fingerprint_s']:.2f}"
            L.append(f"    {str(r['hop']):<6}{str(r['from']):<13}{str(r['to']):<13}{str(r['switch_s']):<10}"
                     f"{str(r['first_execute_s']):<14}{str(r['first_execute_excl_fingerprint_s']):<12}{fp_s:<9}"
                     f"{r['cuda_initialized_by_fingerprint']}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="inp", default=str(CANONICAL_IN))
    ap.add_argument("--cohort", default=None, help="default: the cohort of the last record in the file")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", action="store_true", help="print the JSON summary instead of the table")
    args = ap.parse_args()
    inp = Path(args.inp)
    cohort, chains, notes = load_cohort(inp, args.cohort)
    if not chains:
        print(f"no cohort records in {inp}", file=sys.stderr)
        return 1
    s = summarise(chains, cohort, notes)
    s["source_file"] = str(inp)
    print(json.dumps(s, indent=1, default=str) if args.json else render(s))
    out = Path(args.out) if args.out else (CANONICAL_OUT if inp.resolve() == CANONICAL_IN.resolve() else None)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(s, indent=1, default=str) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
