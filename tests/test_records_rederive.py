"""The paper's numbers, re-derived from the shipped records.

Each test reads one record under results/ (read-only) and runs it through the
analyser that `docs/EXPERIMENTS.md` names for it (`rederive.py`,
`e13_analyse.py`, `e15_analyse.py`, `figures/share_model.py`), or, where a
harness computes its summary inline (XSUB-ISA), recomputes that summary from
the raw fields. The expected values are the ones the paper states. A change to
an analyser that moves a published number, or a record edited so that it no
longer supports one, fails here.

Nothing is written: not to results/, and not to figures/ (share_model is
loaded without writing bytecode beside it).
"""

from __future__ import annotations

import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import e12_crash  # noqa: E402
import e13_analyse  # noqa: E402
import e15_analyse  # noqa: E402
import rederive  # noqa: E402
from handoff.controller import DEST_CREATED, admission_for  # noqa: E402


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def shipped_rederive(monkeypatch):
    """rederive with its records directory pinned to the shipped results/."""
    monkeypatch.setattr(rederive, "RES", RESULTS)
    return rederive


def test_e12_every_crash_point_recovers_as_the_paper_reports(shipped_rederive, capsys):
    # Paper: 17 of 17 crash points recovered, 8 writes refused while down,
    # 1 orphan reclaimed, median recovery 40 s (0 to 65 s), a duplicate
    # request yields one destination.
    shipped_rederive.e12()
    out = capsys.readouterr().out
    head = re.search(r"^E12 e12-r7: (\d+)/(\d+) crash points as expected; refused while down (\d+); "
                     r"orphans reclaimed (\d+); fully cleaned (\d+)/(\d+)$", out, re.M)
    assert head, out
    ok, n, refused, reclaimed, cleaned, n2 = map(int, head.groups())
    assert (ok, n, refused, reclaimed, cleaned, n2) == (17, 17, 8, 1, 17, 17)
    rec = re.search(r"recovery median ([\d.]+) s \(min ([\d.]+), max ([\d.]+)\)", out)
    assert rec, out
    # Printed to 0.1 s; the paper rounds half up to whole seconds.
    assert tuple(map(float, rec.groups())) == (40.1, 0.0, 64.5)
    assert "destinations_alive=1 one_joined=True" in out

    trials = [r for r in _jsonl(RESULTS / "e12" / "e12_runs.jsonl")
              if r.get("cohort") == "e12-r7" and "probe" not in r]
    # The 17 points are exactly the harness's default trial list: every
    # journaled phase hook before and after its write, plus the create window
    # (which exists only before a journal write).
    points = [(r["crash_phase"], r["after_journal"]) for r in trials]
    expected = {(p, a) for p in e12_crash.CRASH_PHASES for a in (True, False)} | {(DEST_CREATED, False)}
    assert len(points) == len(set(points)) == 17
    assert set(points) == expected
    # A write routed while the controller is down is refused exactly when the
    # journal phase at the crash implies a closed gate, per the controller.
    for r in trials:
        closed = admission_for(r["journal_phase_at_crash"]) == "closed"
        assert r["write_while_down"]["refused"] is closed, (r["crash_phase"], r["after_journal"])
        assert r["exactly_one_runtime_accepts"] and r["committed_state_recoverable"]
    # The single reclaimed orphan is the destination created but never journaled.
    assert [r["crash_phase"] for r in trials if r["reclaimed"]] == [DEST_CREATED]


def test_e13_same_chain_is_bitwise_equal_to_both_t4_controls_over_150_losses():
    # Paper, Table 2 T4 to T4: 5 moves, 150 losses, parameter, optimizer,
    # scheduler and loader digests and every generator state (including CUDA)
    # bitwise equal to two uninterrupted controls.
    cohort, chains, notes = e13_analyse.load_cohort(RESULTS / "e13" / "e13_chains.jsonl", "e13-v2")
    assert cohort == "e13-v2" and notes == []
    assert {"same", "control_t4_a", "control_t4_b"} <= set(chains)
    same = chains["same"]
    hops = same["hops"]
    assert len(hops) == 5 and all(h["from"] == h["to"] == "gpu_t4" for h in hops)

    losses = [x for b in e13_analyse.block_trace(same) for x in b["losses"]]
    assert len(losses) == 150 == len(same["final"]["loss_trace"])

    s = e13_analyse.summarise(chains, cohort, notes)
    assert s["chains"]["same"]["verified"] == 5 and s["chains"]["same"]["oracles_all_pass"]
    # The noise floor first: the two controls agree bitwise, so equality below
    # is attributable to the switch.
    assert s["comparisons"]["control_t4_a_vs_control_t4_b"]["bitwise_equal"]
    for ctl in ("control_t4_a", "control_t4_b"):
        c = s["comparisons"][f"same_vs_{ctl}"]
        assert c["bitwise_equal"], c["first_divergence"]
        assert (c["blocks_equal"], c["blocks_compared"]) == (5, 5)
        assert c["first_divergence"] is None and c["max_abs_loss_diff"] == 0.0
        assert all(c["final"][k] for k in ("loss_trace_equal", "param_digest_equal", "optimizer_digest_equal",
                                           "optimizer_step_equal", "rng_equal"))
        # The same claim straight from the record, bit for bit.
        ctl_losses = [x for b in e13_analyse.block_trace(chains[ctl]) for x in b["losses"]]
        assert [float(x).hex() for x in losses] == [float(x).hex() for x in ctl_losses]
        assert s["step_counts"][f"same_vs_{ctl}"]["equal"]
    # Every generator, CUDA included, equal across each of the five moves.
    rng = s["rng"]["same"]
    assert len(rng) == 5
    for row in rng:
        assert (row["python"], row["numpy"], row["torch_cpu"], row["cuda"]) == ("equal",) * 4, row
        assert row["named"] and set(row["named"].values()) == {"equal"}, row
        assert row["stored_verdicts_agree"], row


def test_e15_host_container_record_judges_101_pass_0_fail_6_skip():
    # Paper, Table 2 macOS process and Linux container: 101 checks pass, 63
    # acknowledged writes present once, 73 refused writes absent (3 while the
    # controller was down), abort reopened the source, crashed controller
    # resumed. The docs: 101 PASS, 0 FAIL, 6 SKIP (CUDA checks skipped).
    path = RESULTS / "xsub" / "host_container" / "e15_records.jsonl"
    records = e15_analyse.load(path, cohort="xsub-host-container")
    assert records
    res = e15_analyse.judge(records)
    assert (res["passed"], res["failed"], res["skipped"]) == (101, 0, 6)
    assert res["ok"]
    # Every skip is a CUDA check the local substrates cannot exercise.
    skips = [r for r in res["rows"] if r["verdict"] == "SKIP"]
    assert all(r["detail"].startswith(e15_analyse.NO_CUDA) and "cuda" in r["check"].lower()
               for r in skips), skips

    w = res["writes"]
    assert (w["acknowledged"], w["refused"], w["errors"]) == (63, 73, 0)
    by = {r["name"]: r for r in records}
    fin = by["final"]["writes"]
    assert not set(fin["acked_so_far"]) & set(fin["refused_so_far"])
    down = by["hop2b"]["down_writes"]
    assert len(down) == 3 and all(d["outcome"] == "refused" for d in down)
    assert {d["seq"] for d in down} <= set(fin["refused_so_far"])

    verdict = {(r["section"], r["check"]): r["verdict"] for r in res["rows"]}
    hop2a, hop2b = e15_analyse.SECTIONS["hop2a"], e15_analyse.SECTIONS["hop2b"]
    for key in [(hop2a, "authority stays on the GPU runtime"),
                (hop2a, "the gate closed for the capture and reopened on the GPU runtime"),
                (hop2b, "writes routed while it was down were refused"),
                (hop2b, "a fresh controller resumed with takeover"),
                (hop2b, "every acknowledged write present exactly once"),
                (hop2b, "every refused write absent"),
                ("trajectory", "every acknowledged write of the lifecycle is on the final runtime")]:
        assert verdict.get(key) == "PASS", key
    assert by["hop2a"]["result"]["phase"] == "ABORTED"
    assert by["hop2b"]["result"]["resumed"] is True and by["hop2b"]["result"]["phase"] == "DONE"


def test_xsub_isa_record_both_moves_done_with_state_equal():
    # Paper, Table 2 arm64 and x86-64: two moves DONE, state equal at both
    # boundaries, all relationships kept, losses bitwise equal to the arm64
    # control until the first x86-64 step; Figure fidelity: 1.5e-14 across ISAs.
    rec = json.loads((RESULTS / "xsub" / "isa" / "xsub_isa.json").read_text())
    assert rec["schema"] == "xsub-isa/1"
    arm, x86 = rec["substrates"]["arm64"], rec["substrates"]["x86_64"]
    assert (arm["machine"], x86["machine"]) == ("aarch64", "x86_64")
    assert (arm["python"], arm["dill"]) == (x86["python"], x86["dill"])

    hops = rec["hops"]
    assert [(h["from"], h["to"], h["phase"]) for h in hops] == [("arm64", "x86_64", "DONE"),
                                                                 ("x86_64", "arm64", "DONE")]
    assert [h["after"]["machine"] for h in hops] == ["x86_64", "aarch64"]
    # State equality re-derived on the fields the harness compares, not read
    # from its flag, and the flag and the summary agree with it.
    for h in hops:
        equal = all(h["before"][k] == h["after"][k] for k in ("step", "losses", "W", "log_lines"))
        assert equal and h["state_equal"] is True, h["hop"]
    s = rec["summary"]
    assert (s["hops_done"], s["hops"], s["state_equal_at_every_boundary"]) == (2, 2, True)

    fin = rec["final"]
    rel = ("view_shares_W", "opt_params_is_W", "alias_is_W", "cycle", "log_matches_losses")
    assert all(fin[k] is True for k in rel) and s["relationships_kept"] == {k: True for k in rel}
    steps = rec["steps_per_block"]
    assert fin["step"] == s["steps"] == 3 * steps == len(fin["losses"])

    # The first x86-64 step is the one after hop 1's cut; every loss before it
    # equals the arm64 control bit for bit, and it is the first that differs.
    ctl = rec["controls"]["arm64"]["losses"]
    first = next(i for i, (a, b) in enumerate(zip(fin["losses"], ctl)) if a != b)
    assert first == hops[0]["before"]["step"] == steps
    assert s["first_differing_step_vs_control"]["arm64"] == first
    rel_diff = max(abs(float.fromhex(a) - float.fromhex(b)) / abs(float.fromhex(b))
                   for a, b in zip(fin["losses"], ctl))
    assert f"{rel_diff:.1e}" == "1.5e-14"


def test_e16_micro_record_has_four_state_equal_moves_and_never_two_gpus(monkeypatch):
    # Paper, Figure share and "Two sessions, one GPU": 4 of 4 moves equal,
    # never two GPUs alive, swap 104 s, GPU busy 90% of the time held, 31%
    # cheaper over the two completed rounds (figures/share_model.py).
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spec = importlib.util.spec_from_file_location("share_model_rederive", ROOT / "figures" / "share_model.py")
    share_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(share_model)

    rec = json.loads((RESULTS / "e16" / "e16-micro-2.json").read_text())
    assert rec["cohort"] == "e16-micro-2" and rec["gpu"] == "gpu_a100_40"
    moves = rec["moves"]
    assert len(moves) == 4 and all(m["status"] == 200 for m in moves)
    assert len(rec["checks"]) == 4
    for c in rec["checks"]:
        assert c["ok"] is True and c["before"] == c["after"], c

    # Never two GPU runtimes alive, by a sweep over both sessions' intervals
    # (a release at the same instant as an acquire counts first).
    events = sorted((iv[t], d) for w in "AB" for iv in rec["runtimes"][w]["runtimes"] if iv["sku"] != "cpu"
                    for t, d in (("start_s", 1), ("end_s", -1)))
    alive, peak = 0, 0
    for _, d in events:
        alive += d
        peak = max(peak, alive)
    assert peak == 1

    win = share_model.measured_window(rec)
    assert (win["rounds"], win["moves"], win["moves_state_equal"]) == (2, 4, 4)
    assert win["gpu_runtimes_ever_overlap"] is False
    assert round(statistics.median(share_model.swap_overheads(rec))) == 104
    assert round(win["gpu_busy_pct_shared"]) == 90
    assert round(win["saving_pct"]) == 31
    # The record's own end-to-end summary includes the failed third round,
    # which is why the paper does not use it (docs: a saving of -29.9%).
    assert rec["error"] and round(rec["summary"]["saving_pct"], 1) == -29.9


def test_escalation_tally_is_7_verified_1_inconclusive_2_refused(shipped_rederive, capsys):
    # Paper, "Outgrowing a GPU": GPT-2 needs 16.2 GiB at sequence length 1024;
    # T4 to A100 after 20, 120 and 1080 steps; ViT T4 to L4 to A100; 7 of 10
    # moves verified, 1 inconclusive, 2 refused.
    shipped_rederive.escalation()
    out = capsys.readouterr().out
    assert re.search(r"^escalation moves: 7 verified, 1 inconclusive, 2 refused$", out, re.M), out
    assert "absent" not in out

    moves = re.findall(r"^\s+(\S+\.jsonl)\s+(gpt2|vit) f=(\S+) rep=(\d+) -> (\S+)\s+"
                       r"(verified|inconclusive|refused)\s+(.*)$", out, re.M)
    assert len(moves) == 10
    gpt2_steps = {int(m[6].split()[1]) for m in moves if m[1] == "gpt2" and m[5] == "verified"}
    assert gpt2_steps == {20, 120, 1080}
    assert all(m[4] == "gpu_a100_40" for m in moves if m[1] == "gpt2")
    vit_rep1 = [(m[4], m[5]) for m in moves if m[1] == "vit" and m[3] == "1"]
    assert vit_rep1 == [("gpu_l4", "verified"), ("gpu_a100_40", "verified")]
    assert [m[6] for m in moves if m[5] == "refused"] == ["RUNTIME_STATE_CAPTURE_FAILED"] * 2

    # The refusals, counted straight from the records.
    files = ["escalation_runs.jsonl", "escalation_runs_n40_verify.jsonl",
             "escalation_runs_timeout_fail.jsonl", "escalation_runs_failed_sweep1.jsonl"]
    switches = [s for f in files for r in _jsonl(RESULTS / "escalation" / f) for s in r.get("switches") or []]
    assert len(switches) == 10 and sum(1 for s in switches if s.get("status") != 200) == 2

    cal = {(p, d, b): v for p, d, b, v in re.findall(
        r"^\s+gpt2\s+(\S+)\s+dim=(\d+)\s+batch=(\d+)\s+(\S+)$", out, re.M)}
    assert cal[("gpu_t4", "1024", "8")] == "cuda_oom"
    assert round(float(cal[("gpu_a100_40", "1024", "8")]), 1) == 16.2
