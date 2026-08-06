"""Pure geometry model for the massing pre-check (no geometry kernel).

All proxies are axis-aligned boxes in the machine frame (mm): origin =
gantry home, +X toward bins, +Y away from operator, +Z up, base top Z 0.
Because the proxies are axis-aligned, clash is exact closed-form AABB
math -- no OCP is needed to *check*. OCP (build123d) is used only to
*render* these same boxes to STEP/GLB in massing/geometry.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Point = tuple[float, float, float]


@dataclass(frozen=True)
class Box:
    """Axis-aligned bounding box: min corner (x, y, z) + positive sizes."""

    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float


@dataclass
class Station:
    name: str
    box: Box


@dataclass
class AlignmentPair:
    name: str
    a: Point
    b: Point
    axes: tuple[str, ...]  # any of "x", "y", "z"
    tol: float


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: list[str]


@dataclass
class Machine:
    stations: list[Station] = field(default_factory=list)
    clash_allow: set[frozenset[str]] = field(default_factory=set)
    min_clearance: float = 3.0
    gantry_travel: Box = field(default_factory=lambda: Box(0, 0, 0, 0, 0, 0))
    reach_targets: list[tuple[str, tuple[float, float]]] = field(default_factory=list)
    alignment_pairs: list[AlignmentPair] = field(default_factory=list)
    z_chain: list[tuple[str, float]] = field(default_factory=list)


def _overlap_1d(amin: float, alen: float, bmin: float, blen: float) -> float:
    lo = max(amin, bmin)
    hi = min(amin + alen, bmin + blen)
    return max(0.0, hi - lo)


def aabb_overlap(a: Box, b: Box) -> float:
    """Intersection volume in mm^3 (0.0 if the boxes do not overlap)."""
    return (
        _overlap_1d(a.x, a.dx, b.x, b.dx)
        * _overlap_1d(a.y, a.dy, b.y, b.dy)
        * _overlap_1d(a.z, a.dz, b.z, b.dz)
    )


def _sep_1d(amin: float, alen: float, bmin: float, blen: float) -> float:
    amax, bmax = amin + alen, bmin + blen
    if bmin > amax:
        return bmin - amax
    if amin > bmax:
        return amin - bmax
    return 0.0


def aabb_gap(a: Box, b: Box) -> float:
    """Euclidean separation in mm (0.0 if the boxes overlap or touch)."""
    return math.hypot(
        _sep_1d(a.x, a.dx, b.x, b.dx),
        _sep_1d(a.y, a.dy, b.y, b.dy),
        _sep_1d(a.z, a.dz, b.z, b.dz),
    )


def contains_point(box: Box, p: Point) -> bool:
    return (
        box.x <= p[0] <= box.x + box.dx
        and box.y <= p[1] <= box.y + box.dy
        and box.z <= p[2] <= box.z + box.dz
    )
