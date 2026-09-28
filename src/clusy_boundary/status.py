"""Check outcomes.

The central discipline of this harness: a check that did not actually run
reports NA, never PASS. A report full of PASS rows is only evidence if every
PASS corresponds to a comparison that was genuinely performed. NA rows are
counted and displayed separately so a transition can never look verified
because a probe silently found nothing to look at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NA = "n/a"
    ERROR = "ERROR"

    @property
    def is_verdict(self) -> bool:
        """True when this status reflects a comparison that actually ran."""
        return self in (Status.PASS, Status.FAIL)


class Mode(str, Enum):
    """How equality was decided for a row.

    EXACT means byte-for-byte identity was required and obtained. TOLERANCE
    means the boundary crossed a hardware or library change where bitwise
    identity is not physically expected (a CUDA kernel and a CPU kernel do not
    produce the same bits for the same reduction), so the row is judged against
    a declared numeric threshold and the measured deviation is recorded. A
    TOLERANCE pass is a weaker claim than an EXACT pass and is rendered as such.
    """

    EXACT = "exact"
    TOLERANCE = "tolerance"
    STRUCTURAL = "structural"
    NONE = "none"


@dataclass
class CheckResult:
    name: str
    status: Status
    mode: Mode = Mode.NONE
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    #: Names of the specific items that failed, for triage.
    offenders: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "mode": self.mode.value,
            "detail": self.detail,
            "metrics": self.metrics,
            "offenders": self.offenders[:64],
        }
