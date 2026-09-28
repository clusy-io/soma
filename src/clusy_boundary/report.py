"""The record a transition produces, and how it is printed.

The machine-readable record is primary: one JSON object per transition, which
accumulates across runs into the evidence the paper reports. The table is a
rendering of that record, not a separate thing that could drift from it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

from .status import CheckResult, Mode, Status
from .witness import Witness

RECORD_FORMAT = 1
_ROW_WIDTH = 18


@dataclass
class BoundaryRecord:
    format: int
    transition: str
    source: dict[str, Any]
    destination: dict[str, Any]
    canonical: list[CheckResult]
    supplementary: list[CheckResult] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def all_rows(self) -> list[CheckResult]:
        return self.canonical + self.supplementary

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.all_rows if r.status is Status.FAIL]

    @property
    def errored(self) -> list[CheckResult]:
        return [r for r in self.all_rows if r.status is Status.ERROR]

    @property
    def not_applicable(self) -> list[CheckResult]:
        return [r for r in self.all_rows if r.status is Status.NA]

    @property
    def verified(self) -> list[CheckResult]:
        return [r for r in self.all_rows if r.status is Status.PASS]

    @property
    def ok(self) -> bool:
        """True only when nothing failed and nothing errored.

        NA rows do not make a transition fail. They do stop it from being
        described as fully verified, which the summary line states outright.
        """
        return not self.failed and not self.errored

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "transition": self.transition,
            "source": self.source,
            "destination": self.destination,
            "canonical": [r.to_dict() for r in self.canonical],
            "supplementary": [r.to_dict() for r in self.supplementary],
            "context": self.context,
            "summary": {
                "verified": len(self.verified),
                "failed": len(self.failed),
                "errored": len(self.errored),
                "not_applicable": len(self.not_applicable),
                "ok": self.ok,
            },
        }

    def append_to(self, path: str) -> None:
        """Append this record to a JSON Lines file.

        One line per transition, so records from many runs concatenate without
        a merge step and can be filtered with ordinary tools.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(self.to_dict(), sort_keys=True, default=str) + "\n")


def build_record(
    before: Witness,
    after: Witness,
    canonical: list[CheckResult],
    supplementary: list[CheckResult],
    *,
    transition: str,
    context: dict[str, Any] | None = None,
) -> BoundaryRecord:
    return BoundaryRecord(
        format=RECORD_FORMAT,
        transition=transition,
        source={"label": before.label, **before.host.to_dict()},
        destination={"label": after.label, **after.host.to_dict()},
        canonical=canonical,
        supplementary=supplementary,
        context=context or {},
    )


def _row_line(r: CheckResult, verbose: bool) -> str:
    line = f"{r.name.ljust(_ROW_WIDTH)}{r.status.value}"
    if r.status is Status.PASS and r.mode is Mode.TOLERANCE:
        # A tolerance pass is a weaker claim than an exact one and is never
        # printed as though it were the same result.
        line += "  (tolerance)"
    if verbose or r.status in (Status.FAIL, Status.NA, Status.ERROR):
        if r.detail:
            line += f"  {r.detail}"
    return line


def render(record: BoundaryRecord, *, verbose: bool = False, color: bool = False) -> str:
    def paint(text: str, status: Status) -> str:
        if not color:
            return text
        code = {
            Status.PASS: "32", Status.FAIL: "31",
            Status.NA: "33", Status.ERROR: "35",
        }[status]
        return f"\033[{code}m{text}\033[0m"

    out: list[str] = []
    out.append("Boundary correctness")
    out.append("-" * 20)
    for r in record.canonical:
        out.append(paint(_row_line(r, verbose), r.status))

    if record.supplementary:
        out.append("")
        for r in record.supplementary:
            out.append(paint(_row_line(r, verbose), r.status))

    out.append("")
    out.append(
        f"{record.transition}: "
        f"{record.source['label']} ({record.source['hostname']}, "
        f"{'cuda' if record.source['cuda_available'] else 'cpu'}) -> "
        f"{record.destination['label']} ({record.destination['hostname']}, "
        f"{'cuda' if record.destination['cuda_available'] else 'cpu'})"
    )

    n_ver, n_fail = len(record.verified), len(record.failed)
    n_na, n_err = len(record.not_applicable), len(record.errored)
    parts = [f"{n_ver} verified"]
    if n_fail:
        parts.append(f"{n_fail} failed")
    if n_err:
        parts.append(f"{n_err} errored")
    if n_na:
        parts.append(f"{n_na} not applicable")
    out.append(", ".join(parts))

    if n_na and not n_fail and not n_err:
        out.append(
            "Not fully verified: the rows above marked n/a were not checked, "
            "because the namespace had nothing to check them against."
        )

    for r in record.failed + record.errored:
        out.append("")
        out.append(f"{r.name}: {r.detail}")
        for off in r.offenders[:12]:
            out.append(f"  - {off}")
        if len(r.offenders) > 12:
            out.append(f"  ... and {len(r.offenders) - 12} more")

    return "\n".join(out)


def render_aggregate(records: Iterable[dict[str, Any]]) -> str:
    """Summarize many records: per-row pass rate across every transition."""
    rows: dict[str, dict[str, int]] = {}
    order: list[str] = []
    total = 0
    for rec in records:
        total += 1
        for r in rec.get("canonical", []) + rec.get("supplementary", []):
            name = r["name"]
            if name not in rows:
                rows[name] = {"PASS": 0, "FAIL": 0, "n/a": 0, "ERROR": 0}
                order.append(name)
            rows[name][r["status"]] = rows[name].get(r["status"], 0) + 1

    out = [f"Boundary correctness across {total} transition(s)", "-" * 46]
    header = f"{'row'.ljust(_ROW_WIDTH)}{'pass':>6}{'fail':>6}{'n/a':>6}{'err':>6}"
    out.append(header)
    for name in order:
        c = rows[name]
        out.append(
            f"{name.ljust(_ROW_WIDTH)}{c['PASS']:>6}{c['FAIL']:>6}"
            f"{c['n/a']:>6}{c['ERROR']:>6}"
        )
    failed_total = sum(c["FAIL"] for c in rows.values())
    out.append("")
    out.append(
        f"{total} transition(s), {failed_total} failing row(s) overall"
        if failed_total else f"{total} transition(s), no failing rows"
    )
    return "\n".join(out)
