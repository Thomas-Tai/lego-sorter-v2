"""The four MVP static-geometry checks. Pure functions over a Machine.

Each returns a CheckResult with human-readable numeric detail lines.
Hard box overlap fails a check; a sub-min_clearance gap is reported but
does NOT fail (spec Draft-v3 Idea K -- proximity flag, not an access
check).
"""

from __future__ import annotations

from massing.model import CheckResult, Machine, aabb_gap, aabb_overlap

_AXIS = {"x": 0, "y": 1, "z": 2}


def check_clash(machine: Machine) -> CheckResult:
    details: list[str] = []
    passed = True
    stations = machine.stations
    for i in range(len(stations)):
        for j in range(i + 1, len(stations)):
            a, b = stations[i], stations[j]
            if frozenset({a.name, b.name}) in machine.clash_allow:
                continue
            vol = aabb_overlap(a.box, b.box)
            if vol > 0:
                passed = False
                details.append(f"HARD CLASH {a.name} & {b.name}: {vol:.0f} mm^3")
                continue
            gap = aabb_gap(a.box, b.box)
            if gap < machine.min_clearance:
                details.append(
                    f"TIGHT {a.name} ~ {b.name}: gap {gap:.2f} mm "
                    f"(< {machine.min_clearance} mm; proximity only)"
                )
    if not details:
        details.append("no hard clashes; no sub-clearance pairs")
    return CheckResult("clash", passed, details)


def check_reach(machine: Machine) -> CheckResult:
    tb = machine.gantry_travel
    xmin, xmax = tb.x, tb.x + tb.dx
    ymin, ymax = tb.y, tb.y + tb.dy
    details: list[str] = []
    passed = True
    backoffs: list[float] = []
    for name, (x, y) in machine.reach_targets:
        inside = xmin <= x <= xmax and ymin <= y <= ymax
        mx = min(x - xmin, xmax - x)
        my = min(y - ymin, ymax - y)
        if not inside:
            passed = False
        details.append(
            f"{name}: ({x:.1f}, {y:.1f}) inside={inside} "
            f"x-margin={mx:.1f} y-margin={my:.1f}"
        )
        backoffs.append(min(mx, my))
    if backoffs:
        details.append(f"worst back-off room to travel limit: {min(backoffs):.1f} mm")
    return CheckResult("reach", passed, details)


def check_alignment(machine: Machine) -> CheckResult:
    details: list[str] = []
    passed = True
    for pair in machine.alignment_pairs:
        offsets = []
        worst = 0.0
        for ax in pair.axes:
            off = abs(pair.a[_AXIS[ax]] - pair.b[_AXIS[ax]])
            offsets.append(f"{ax}={off:.2f}")
            worst = max(worst, off)
        ok = worst <= pair.tol
        if not ok:
            passed = False
        details.append(
            f"{pair.name}: {', '.join(offsets)} (tol {pair.tol}) "
            f"{'PASS' if ok else 'FAIL'}"
        )
    return CheckResult("alignment", passed, details)


def check_stackup(machine: Machine) -> CheckResult:
    details: list[str] = []
    passed = True
    chain = machine.z_chain
    for (la, za), (lb, zb) in zip(chain, chain[1:]):
        delta = za - zb
        ok = delta > 0
        if not ok:
            passed = False
        details.append(
            f"{la}({za:.1f}) -> {lb}({zb:.1f}): d={delta:+.1f} mm "
            f"{'ok' if ok else 'NON-DESCENDING'}"
        )
    return CheckResult("stackup", passed, details)
