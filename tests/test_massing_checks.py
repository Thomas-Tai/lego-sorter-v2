"""The four MVP static-geometry checks (runs in CI, no OCP)."""

from massing.model import AlignmentPair, Box, Machine, Station
from massing.checks import (
    check_alignment,
    check_clash,
    check_reach,
    check_stackup,
)


def test_clash_hard_overlap_fails() -> None:
    m = Machine(
        stations=[
            Station("a", Box(0, 0, 0, 10, 10, 10)),
            Station("b", Box(5, 5, 5, 10, 10, 10)),
        ]
    )
    result = check_clash(m)
    assert result.passed is False
    assert any("HARD CLASH" in d for d in result.details)


def test_clash_allowlisted_overlap_is_ignored() -> None:
    m = Machine(
        stations=[
            Station("a", Box(0, 0, 0, 10, 10, 10)),
            Station("b", Box(5, 5, 5, 10, 10, 10)),
        ],
        clash_allow={frozenset({"a", "b"})},
    )
    assert check_clash(m).passed is True


def test_clash_tight_gap_reports_but_passes() -> None:
    m = Machine(
        stations=[
            Station("a", Box(0, 0, 0, 10, 10, 10)),
            Station("b", Box(12, 0, 0, 10, 10, 10)),  # 2 mm gap
        ],
        min_clearance=3.0,
    )
    result = check_clash(m)
    assert result.passed is True  # tight gap warns, never fails
    assert any("TIGHT" in d for d in result.details)


def test_reach_fails_when_target_outside_travel() -> None:
    m = Machine(
        gantry_travel=Box(0, 0, 0, 355, 160, 1),
        reach_targets=[("bin_far", (400, 80))],  # x=400 > 355
    )
    assert check_reach(m).passed is False


def test_reach_passes_and_reports_backoff() -> None:
    m = Machine(
        gantry_travel=Box(0, 0, 0, 355, 160, 1),
        reach_targets=[("overflow", (280, 140))],
    )
    result = check_reach(m)
    assert result.passed is True
    assert any("back-off" in d for d in result.details)


def test_alignment_offset_within_tol_passes() -> None:
    m = Machine(
        alignment_pairs=[
            AlignmentPair("feeder_out", (-332.5, 0, 0), (-332.5, 0, 0), ("x",), 1.0)
        ]
    )
    assert check_alignment(m).passed is True


def test_alignment_offset_beyond_tol_fails() -> None:
    m = Machine(
        alignment_pairs=[
            AlignmentPair("feeder_out", (-332.5, 0, 0), (-330.0, 0, 0), ("x",), 1.0)
        ]
    )
    assert check_alignment(m).passed is False  # 2.5 mm > 1.0


def test_stackup_descending_passes() -> None:
    m = Machine(z_chain=[("belt", 75), ("funnel", 67), ("gate", 42), ("rim", 35)])
    assert check_stackup(m).passed is True


def test_stackup_non_descending_fails() -> None:
    m = Machine(z_chain=[("belt", 75), ("funnel", 80)])  # goes up
    result = check_stackup(m)
    assert result.passed is False
    assert any("NON-DESCENDING" in d for d in result.details)


def test_alignment_off_axis_diff_is_suppressed() -> None:
    # y differs by 50, but the pair only checks x -> the y diff is masked out.
    m = Machine(
        alignment_pairs=[AlignmentPair("masked", (0, 0, 0), (0, 50, 0), ("x",), 1.0)]
    )
    assert check_alignment(m).passed is True
