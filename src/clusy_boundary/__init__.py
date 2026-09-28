"""Automated boundary-correctness verification for live kernel state.

A *boundary* is any point where a running kernel's state is moved: a GPU-to-CPU
sandbox switch, a branch fork, a checkpoint and restore, a pause and resume.
This package answers one question about such a move, with evidence:

    did the destination namespace preserve everything the source namespace had?

"Training still works" is not an answer to that question. This package checks
parameters, non-parameter buffers, optimizer slot tensors, the identity of the
tensors the optimizer will write into, LR-scheduler and AMP-scaler state, all
four random-number streams, the data loader's position and sampling order, the
aliasing structure of the namespace, workspace file contents, and whether
training continues along the same trajectory afterwards.

Usage in a kernel, on each side of the boundary::

    from clusy_boundary import capture_witness
    capture_witness(globals(), label="t4-source").save("before.json")

and then, anywhere both witnesses can be read::

    from clusy_boundary import Witness, build_record, render, run_all
    before, after = Witness.load("before.json"), Witness.load("after.json")
    canonical, supplementary = run_all(before, after)
    record = build_record(before, after, canonical, supplementary, transition="gpu-to-cpu")
    print(render(record))
"""

from .checks import run_all
from .continuation import run_continuation
from .report import BoundaryRecord, build_record, render, render_aggregate
from .status import CheckResult, Mode, Status
from .witness import Witness, capture_witness

__all__ = [
    "BoundaryRecord",
    "CheckResult",
    "Mode",
    "Status",
    "Witness",
    "build_record",
    "capture_witness",
    "render",
    "render_aggregate",
    "run_all",
    "run_continuation",
]

__version__ = "0.1.0"
