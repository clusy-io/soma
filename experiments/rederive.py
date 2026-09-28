"""Re-derive paper numbers that have no dedicated analysis script.

E11 and E12 print their verdicts at the end of a live run, the escalation
harness has no analyser, and E10's harness is not part of this artifact. This
script recomputes those numbers from the shipped records, offline and with the
standard library only.

    python experiments/rederive.py              # everything
    python experiments/rederive.py e12 escalation
    python experiments/rederive.py --results /tmp/soma-rerun --cohort e11-rerun e11   # a rerun

`--results` names a directory laid out like results/ (e11/e11_runs.jsonl,
escalation/escalation_runs.jsonl, ...); `--cohort` replaces the paper's cohort
for the one E11, E12 or E13 step selected.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"          # replaced by --results
PAPER_COHORTS = {"e11": "e11-r4b", "e12": "e12-r7", "e13": "e13-v2"}


def _rows(path: Path) -> list[dict]:
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else [data]
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def e10() -> None:
    """Switch time of a 64-file workspace, before and after batched persistence."""
    for name in ("baseline", "opt"):
        rows = _rows(RES / "e10" / f"e10_reconciled.e10-v2-{name}.files.f1-f64-f256.json")
        by_label: dict[str, list[float]] = {}
        for r in rows:
            by_label.setdefault(r["label"], []).append(r["request_ms"] / 1000)
        cells = ", ".join(f"{k} median {st.median(v):.1f} s (n={len(v)})" for k, v in by_label.items())
        print(f"E10 {name:<8} {cells}")


def e11(cohort: str = PAPER_COHORTS["e11"]) -> None:
    """Fault campaign, cohort e11-r4b: the latest row per case decides."""
    rows = [r for r in _rows(RES / "e11" / "e11_runs.jsonl") if r.get("cohort") == cohort]
    latest: dict[str, dict] = {}
    for r in rows:
        if r.get("probe"):
            print(f"  probe {r['probe']}: {json.dumps(r.get('results'))} "
                  f"exactly_one_200={r.get('exactly_one_200')}")
            continue
        latest[r.get("fault") or "none"] = r
        print(f"  row  fault={str(r.get('fault')):<31} phase={str(r.get('phase')):<15} "
              f"reason={str(r.get('reason'))[:52]:<52} source_accepts_after={r.get('source_accepts_after')} "
              f"as_expected={r.get('as_expected')}")
    ok = sum(1 for r in latest.values() if r.get("as_expected"))
    print(f"E11 {cohort}: {ok}/{len(latest)} cases as designed (latest row per case; "
          f"earlier rows of a rerun case are kept in the record)")


def e12(cohort: str = PAPER_COHORTS["e12"]) -> None:
    """Crash campaign, cohort e12-r7."""
    rows = [r for r in _rows(RES / "e12" / "e12_runs.jsonl") if r.get("cohort") == cohort]
    trials = [r for r in rows if "probe" not in r]
    dup = [r for r in rows if r.get("probe") == "duplicate_request"]
    rec = [r["recovery_s"] for r in trials]
    print(f"E12 {cohort}: {sum(r['as_expected'] for r in trials)}/{len(trials)} crash points as expected; "
          f"refused while down {sum(1 for r in trials if r['write_while_down']['refused'])}; "
          f"orphans reclaimed {sum(len(r['reclaimed']) for r in trials)}; "
          f"fully cleaned {sum(r['fully_cleaned'] for r in trials)}/{len(trials)}")
    print(f"    recovery median {st.median(rec):.1f} s (min {min(rec):.1f}, max {max(rec):.1f})")
    for d in dup:
        print(f"    duplicate request: A={d.get('A')} B={d.get('B')} "
              f"destinations_alive={d.get('destinations_alive')} one_joined={d.get('one_joined')}")


def e13(cohort: str = PAPER_COHORTS["e13"]) -> None:
    """End-to-end move time on the heterogeneous chain (switch + first execute)."""
    rows = _rows(RES / "e13" / "e13_chains.jsonl")
    het = [r for r in rows if r.get("cohort") == cohort and r.get("chain") == "hetero"][-1]
    e2e = [h["switch_s"] + h["first_execute_s"] for h in het["hops"]]
    print(f"E13 {cohort} hetero: {len(e2e)} moves, per move "
          f"{[round(x, 1) for x in e2e]} s, median {st.median(e2e):.1f} s")


def escalation() -> None:
    """OOM-triggered moves: verified, inconclusive or refused, over all four run files."""
    files = ["escalation_runs.jsonl", "escalation_runs_n40_verify.jsonl",
             "escalation_runs_timeout_fail.jsonl", "escalation_runs_failed_sweep1.jsonl"]
    tally = {"verified": 0, "inconclusive": 0, "refused": 0}
    for f in files:
        if not (RES / "escalation" / f).exists():
            print(f"  {f:<38} absent")
            continue
        for r in _rows(RES / "escalation" / f):
            # verify_cell appends one restore row per attempt; attempt 1 opens a move.
            moves: list[list[dict]] = []
            for x in r.get("restores") or []:
                if x.get("attempt") in (1, None) or not moves:
                    moves.append([])
                moves[-1].append(x)
            k = 0
            for s in r.get("switches") or []:
                if s.get("status") != 200:
                    verdict, detail = "refused", s.get("code")
                else:
                    last = (moves[k] if k < len(moves) else [{}])[-1]
                    k += 1
                    good = (last.get("http_status") == 200 and last.get("restore_state") == "restored"
                            and last.get("opt_is_optimizer") is True
                            and last.get("steps_done") is not None
                            and last.get("opt_step") == last.get("steps_done"))
                    verdict = "verified" if good else "inconclusive"
                    detail = f"after {last.get('steps_done')} steps" if good else "no post-move state"
                tally[verdict] += 1
                print(f"  {f:<38} {r.get('experiment')} f={r.get('oom_fraction')} rep={r.get('rep')} "
                      f"-> {s.get('to'):<12} {verdict:<12} {detail}")
    print(f"escalation moves: {tally['verified']} verified, {tally['inconclusive']} inconclusive, "
          f"{tally['refused']} refused")
    print("  calibration (peak GiB, or OOM):")
    cal = RES / "escalation" / "escalation_calibration.jsonl"
    for r in _rows(cal) if cal.exists() else []:
        print(f"    {r['workload']:<5} {r['profile']:<12} dim={r['dim']:<4} batch={r['batch']:<3} "
              f"{r['peak_gb'] if r['fits'] else r['error']}")


STEPS = {"e10": e10, "e11": e11, "e12": e12, "e13": e13, "escalation": escalation}


def main() -> int:
    global RES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="*", help=f"any of {', '.join(STEPS)} (default: all)")
    ap.add_argument("--results", type=Path, default=RES, help="records directory (default: results/)")
    ap.add_argument("--cohort", default=None, help="cohort for the one e11, e12 or e13 step selected "
                                                   "(default: the paper's)")
    args = ap.parse_args()
    unknown = [w for w in args.which if w not in STEPS]
    if unknown:
        ap.error(f"unknown: {', '.join(unknown)}")
    which = args.which or list(STEPS)
    if args.cohort is not None and (len(which) != 1 or which[0] not in PAPER_COHORTS):
        ap.error("--cohort needs exactly one of " + ", ".join(PAPER_COHORTS))
    RES = args.results
    for name in which:
        STEPS[name](args.cohort) if args.cohort is not None else STEPS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
