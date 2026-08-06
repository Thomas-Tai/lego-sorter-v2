"""Pure geometry helpers for the massing model (runs in CI, no OCP)."""

from massing.model import Box, aabb_overlap, aabb_gap, contains_point


def test_overlap_volume_of_intersecting_boxes() -> None:
    a = Box(0, 0, 0, 10, 10, 10)
    b = Box(5, 5, 5, 10, 10, 10)
    assert aabb_overlap(a, b) == 5 * 5 * 5


def test_overlap_zero_when_disjoint() -> None:
    a = Box(0, 0, 0, 10, 10, 10)
    b = Box(20, 0, 0, 10, 10, 10)
    assert aabb_overlap(a, b) == 0.0


def test_overlap_zero_when_touching_faces() -> None:
    a = Box(0, 0, 0, 10, 10, 10)
    b = Box(10, 0, 0, 10, 10, 10)  # shares the x=10 face, no volume
    assert aabb_overlap(a, b) == 0.0


def test_gap_along_single_axis() -> None:
    a = Box(0, 0, 0, 10, 10, 10)
    b = Box(13, 0, 0, 10, 10, 10)  # 3 mm apart in x
    assert aabb_gap(a, b) == 3.0


def test_gap_is_zero_when_overlapping() -> None:
    a = Box(0, 0, 0, 10, 10, 10)
    b = Box(5, 5, 5, 10, 10, 10)
    assert aabb_gap(a, b) == 0.0


def test_diagonal_gap_is_euclidean() -> None:
    a = Box(0, 0, 0, 10, 10, 10)
    b = Box(13, 14, 10, 10, 10, 10)  # 3 in x, 4 in y, touching in z
    assert aabb_gap(a, b) == 5.0  # hypot(3, 4, 0)


def test_contains_point_inside_and_outside() -> None:
    box = Box(0, 0, 0, 10, 10, 10)
    assert contains_point(box, (5, 5, 5)) is True
    assert contains_point(box, (11, 5, 5)) is False
