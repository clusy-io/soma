"""Every row must fail when, and only when, the property it names is broken.

A correctness harness that cannot fail is decoration. These tests damage one
property at a time and assert both directions: the row for that property goes
FAIL, and the rows for the other properties stay out of FAIL. The second half
is what stops the harness from degenerating into "something changed somewhere".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from clusy_boundary import capture_witness, run_all  # noqa: E402
from clusy_boundary.report import build_record, render  # noqa: E402
from clusy_boundary.status import Mode, Status  # noqa: E402
from clusy_boundary.continuation import run_continuation  # noqa: E402

import scenario  # noqa: E402


#: Rows about process-global state rather than about the namespace. They are
#: legitimately comparable even for an empty namespace, so the "nothing may
#: pass unexamined" assertion excludes them.
_PROCESS_ROWS = {"Python RNG", "NumPy RNG", "Torch CPU RNG", "Torch CUDA RNG"}


def _transition(build, restore, *, with_continuation=False, workspace=None):
    """Run a transition the way a real one runs, and collect the rows.

    Ordering matters and is the reason this helper exists. The random-number
    rows read process-global state, so the source witness has to be taken
    *before* the transition executes. Taking both witnesses afterwards would
    compare the destination's RNG with itself and report PASS for a stream that
    had in fact been reset.
    """
    source = build()
    before = capture_witness(source, label="source", workspace_root=workspace)
    if with_continuation:
        before.continuation = run_continuation(source, scenario.step, steps=4)

    dest = restore(source)

    after = capture_witness(dest, label="dest", workspace_root=workspace)
    if with_continuation:
        after.continuation = run_continuation(dest, scenario.step, steps=4)

    canonical, supplementary = run_all(before, after)
    return {r.name: r for r in canonical + supplementary}, before, after, source, dest


def _rows(source, dest, *, with_continuation=False, workspace=None):
    before = capture_witness(source, label="source", workspace_root=workspace)
    after = capture_witness(dest, label="dest", workspace_root=workspace)
    if with_continuation:
        before.continuation = run_continuation(source, scenario.step, steps=4)
        after.continuation = run_continuation(dest, scenario.step, steps=4)
    canonical, supplementary = run_all(before, after)
    return {r.name: r for r in canonical + supplementary}, before, after


def _failed(rows):
    return {name for name, r in rows.items() if r.status is Status.FAIL}


def test_faithful_restore_passes_everything():
    rows, *_ = _transition(
        scenario.build_namespace, scenario.restore_faithful, with_continuation=True
    )

    assert _failed(rows) == set(), {
        n: (r.detail, r.offenders[:3]) for n, r in rows.items() if r.status is Status.FAIL
    }

    # The properties this scenario actually exercises must be verified, not n/a.
    for name in (
        "parameters", "buffers", "optimizer slots", "optimizer refs",
        "scheduler", "Python RNG", "NumPy RNG", "Torch CPU RNG",
        "data cursor", "alias graph", "continuation", "variables",
    ):
        assert rows[name].status is Status.PASS, f"{name}: {rows[name].detail}"


def test_detached_optimizer_is_caught_by_refs_row():
    """Values all match; only the reference row may fail."""
    source = scenario.build_namespace()
    dest = scenario.restore_detached_optimizer(source)
    rows, _, _ = _rows(source, dest)

    assert rows["optimizer refs"].status is Status.FAIL
    assert rows["parameters"].status is Status.PASS, (
        "the point of this scenario is that parameter VALUES are identical"
    )
    assert rows["optimizer slots"].status is Status.PASS
    assert _failed(rows) <= {"optimizer refs", "alias graph"}


def test_reset_rng_is_caught_on_every_stream():
    rows, *_ = _transition(scenario.build_namespace, scenario.restore_reset_rng)

    for name in ("Python RNG", "NumPy RNG", "Torch CPU RNG"):
        assert rows[name].status is Status.FAIL, f"{name} did not notice the reseed"
    assert rows["parameters"].status is Status.PASS


def test_dill_session_restore_breaks_storage_aliasing():
    """The serializer Clusy actually uses does not preserve storage sharing.

    Pickle's memo preserves object identity, so two names bound to one object
    survive. It does not preserve sharing between distinct objects, so a base
    array and a view of it arrive independent. This is the realistic version of
    the aliasing fault, and the alias row has to catch it.
    """
    rows, *_ = _transition(scenario.build_namespace, scenario.restore_dill_session)

    alias = rows["alias graph"]
    assert alias.status is Status.FAIL
    assert any("feature" in o for o in alias.offenders), alias.offenders
    # Parameter values come back byte-identical, which is what makes this
    # failure mode dangerous in production and worth a dedicated row.
    assert rows["parameters"].status is Status.PASS
    assert rows["buffers"].status is Status.PASS


def test_dill_session_restore_breaks_optimizer_class_identity():
    """dill writes torch.optim classes by value, so isinstance stops holding.

    After the restore the optimizer still steps, still holds the right moment
    estimates, and is no longer an instance of ``torch.optim.Optimizer``. Every
    value in it is correct, so only a row that compares class identity can see
    the fault.
    """
    import torch

    source = scenario.build_namespace()
    dest = scenario.restore_dill_session(source)

    assert isinstance(source["optimizer"], torch.optim.Optimizer)
    assert not isinstance(dest["optimizer"], torch.optim.Optimizer), (
        "this test encodes an observed dill behaviour; if dill or torch fixed "
        "it, delete the test rather than weakening the check"
    )

    rows, *_ = _transition(scenario.build_namespace, scenario.restore_dill_session)
    identity = rows["class identity"]
    assert identity.status is Status.FAIL
    assert any("optimizer" in o for o in identity.offenders), identity.offenders


def test_class_identity_passes_for_a_faithful_restore():
    rows, *_ = _transition(scenario.build_namespace, scenario.restore_faithful)
    assert rows["class identity"].status is Status.PASS


def test_broken_aliasing_is_caught():
    source = scenario.build_namespace()
    dest = scenario.restore_broken_aliasing(source)
    rows, _, _ = _rows(source, dest)

    alias = rows["alias graph"]
    assert alias.status is Status.FAIL
    assert alias.metrics["splits"] >= 1
    assert any("feature" in o for o in alias.offenders)
    assert rows["parameters"].status is Status.PASS


def test_dropped_optimizer_state_is_caught():
    source = scenario.build_namespace()
    dest = scenario.restore_dropped_optimizer_state(source)
    rows, _, _ = _rows(source, dest)

    assert rows["optimizer slots"].status is Status.FAIL
    assert rows["parameters"].status is Status.PASS, (
        "weights are untouched; only the moment estimates were lost"
    )


def test_reset_loader_is_caught_by_data_cursor():
    source = scenario.build_namespace()
    dest = scenario.restore_reset_loader(source)
    rows, _, _ = _rows(source, dest)

    cursor = rows["data cursor"]
    assert cursor.status is Status.FAIL
    assert cursor.metrics["order_probed"] >= 1
    assert rows["parameters"].status is Status.PASS


def test_dropped_buffers_is_caught_without_touching_parameters():
    source = scenario.build_namespace()
    dest = scenario.restore_dropped_buffers(source)
    rows, _, _ = _rows(source, dest)

    assert rows["buffers"].status is Status.FAIL
    assert rows["parameters"].status is Status.PASS


def test_workspace_hashes(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "data.csv").write_text("a,b\n1,2\n")
    (work / "sub").mkdir()
    (work / "sub" / "model.txt").write_text("weights")

    source = scenario.build_namespace()
    dest = scenario.restore_faithful(source)
    rows, _, _ = _rows(source, dest, workspace=str(work))
    assert rows["workspace hashes"].status is Status.PASS
    assert rows["workspace hashes"].metrics["files_before"] == 2

    before = capture_witness(source, label="source", workspace_root=str(work))
    (work / "data.csv").write_text("a,b\n1,3\n")
    after = capture_witness(dest, label="dest", workspace_root=str(work))
    canonical, supp = run_all(before, after)
    row = {r.name: r for r in canonical + supp}["workspace hashes"]
    assert row.status is Status.FAIL
    assert row.metrics["changed"] == 1


def test_unexercised_properties_report_na_not_pass():
    """An empty namespace must produce no PASS rows at all.

    This is the discipline the whole report rests on: nothing may look verified
    because there was nothing to look at.
    """
    rows, _, _ = _rows({}, {})
    passed = {n for n, r in rows.items() if r.status is Status.PASS}
    # The RNG rows read process state, not namespace state, so comparing them
    # across an empty namespace is a real comparison and may legitimately pass.
    assert passed <= _PROCESS_ROWS, f"empty namespace reported PASS for {passed - _PROCESS_ROWS}"
    namespace_rows = {n: r for n, r in rows.items() if n not in _PROCESS_ROWS}
    assert all(r.status is Status.NA for r in namespace_rows.values()), {
        n: r.status for n, r in namespace_rows.items() if r.status is not Status.NA
    }


def test_cuda_rng_reports_na_on_a_cpu_only_host():
    source = scenario.build_namespace()
    dest = scenario.restore_faithful(source)
    rows, _, _ = _rows(source, dest)
    import torch

    if not torch.cuda.is_available():
        assert rows["Torch CUDA RNG"].status is Status.NA
        assert "no CUDA device" in rows["Torch CUDA RNG"].detail


def test_continuation_detects_a_silent_parameter_change():
    """A namespace whose weights were perturbed after capture must not pass."""
    import torch

    source = scenario.build_namespace()
    dest = scenario.restore_faithful(source)
    with torch.no_grad():
        dest["model"].fc1.weight.add_(1e-3)

    rows, _, _ = _rows(source, dest, with_continuation=True)
    assert rows["parameters"].status is Status.FAIL
    assert rows["continuation"].status is Status.FAIL
    assert rows["continuation"].mode is Mode.EXACT


def test_continuation_probe_leaves_the_namespace_unchanged():
    """The probe must not corrupt the state it is certifying."""
    source = scenario.build_namespace()
    before = capture_witness(source, label="pre-probe")
    result = run_continuation(source, scenario.step, steps=4)
    assert result["restored"] is True
    assert not result.get("error")
    after = capture_witness(source, label="post-probe")

    canonical, supp = run_all(before, after)
    failed = [r.name for r in canonical + supp if r.status is Status.FAIL]
    assert failed == [], f"the continuation probe perturbed: {failed}"


def test_record_and_render_round_trip():
    source = scenario.build_namespace()
    dest = scenario.restore_dropped_buffers(source)
    rows, before, after = _rows(source, dest)
    canonical, supplementary = run_all(before, after)
    record = build_record(
        before, after, canonical, supplementary, transition="unit-test"
    )
    assert record.ok is False
    payload = record.to_dict()
    assert payload["summary"]["failed"] >= 1
    text = render(record)
    assert "Boundary correctness" in text
    assert "buffers" in text
    # The canonical rows appear in the order the report commits to.
    names = [r["name"] for r in payload["canonical"]]
    assert names[:4] == ["parameters", "buffers", "optimizer slots", "optimizer refs"]
    assert names[-1] == "continuation"


@pytest.mark.parametrize(
    "restore_fn,expected_row",
    [
        (scenario.restore_dropped_buffers, "buffers"),
        (scenario.restore_dropped_optimizer_state, "optimizer slots"),
        (scenario.restore_detached_optimizer, "optimizer refs"),
        (scenario.restore_reset_loader, "data cursor"),
        (scenario.restore_broken_aliasing, "alias graph"),
    ],
)
def test_each_fault_is_attributed_to_its_own_row(restore_fn, expected_row):
    source = scenario.build_namespace()
    dest = restore_fn(source)
    rows, _, _ = _rows(source, dest)
    assert rows[expected_row].status is Status.FAIL
