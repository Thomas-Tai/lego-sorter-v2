"""C3 Locked-Numbers Wall guard for ai-sw-build specs.

The text2cad C3 wall says: a build123d *draft* informs topology and
proportion only -- never a measurement. Every SIZE dimension in the SW
spec must trace to a named local (an ``{"rhs": "\\"VAR\\""}`` expression
into a ``*_locals.txt``), so no raw number eyeballed off the throwaway
draft can reach SolidWorks.

This guard reads one or more ai-sw-build spec JSON files and fails if any
*size/length* field is a bare numeric literal instead of an ``rhs``
citation.

WHAT IT CHECKS (hard-fail): the rhs-capable length fields per
``docs/spec_reference.md`` -- width, height, depth, depth2, diameter,
radius, length, major_radius, minor_radius, distance, spacing, and each
``circles[].diameter``. A bare number in any of these is treated as a
leaked draft measurement.

WHAT IT DELIBERATELY DOES NOT CHECK: positional and angular fields --
``center`` (x/y/z/u/v), ``centerline``, ``start``/``end``/``points``/
``position``, pattern ``direction``/``axis``, fillet/chamfer ``edges``,
``circles[].u/v``, and ``angle``/``angle_deg``/``total_angle``. The
bridge schema forbids ``rhs`` on these (they are literal-only) or they are
placement, not proportion. A placement number copied from the draft can
therefore still slip through here -- those must be verified by hand
against the locals. This guard hardens the high-risk, checkable half of
the wall; it is not a substitute for the C4 leakage audit.

Exit codes: 0 = clean, 1 = leakage found, 2 = usage / bad spec file.

Usage:
    python -m tools.check_spec_leakage <spec.json> [<spec2.json> ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# rhs-capable "length" fields (docs/spec_reference.md "Length values" +
# per-feature tables). A bare number in any of these is a wall violation.
SIZE_FIELDS = frozenset(
    {
        "width",
        "height",
        "depth",
        "depth2",
        "diameter",
        "radius",
        "length",
        "major_radius",
        "minor_radius",
        "distance",
        "spacing",
    }
)


def _is_literal_number(value: object) -> bool:
    """True for a raw int/float dimension (bool is excluded on purpose)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass
class Leak:
    """One size field carrying a raw literal instead of an rhs citation."""

    location: str  # "FeatureName.field"
    value: float
    feature_type: str


@dataclass
class LeakReport:
    leaks: list[Leak] = field(default_factory=list)
    checked: int = 0  # number of size fields inspected
    features: int = 0

    @property
    def has_leaks(self) -> bool:
        return bool(self.leaks)


def find_leaks(spec: dict) -> LeakReport:
    """Scan a parsed spec dict for raw size literals. Pure; no I/O.

    Raises ValueError if the spec is not shaped like an ai-sw-build spec
    (no ``features`` list) -- the caller maps that to exit code 2.
    """
    features = spec.get("features")
    if not isinstance(features, list):
        raise ValueError("spec has no 'features' list")

    report = LeakReport(features=len(features))
    for feat in features:
        if not isinstance(feat, dict):
            continue
        fname = feat.get("name", "<unnamed>")
        ftype = feat.get("type", "<unknown>")

        for key, val in feat.items():
            if key == "circles" and isinstance(val, list):
                for idx, circ in enumerate(val):
                    if not isinstance(circ, dict) or "diameter" not in circ:
                        continue
                    report.checked += 1
                    if _is_literal_number(circ["diameter"]):
                        report.leaks.append(
                            Leak(
                                location=f"{fname}.circles[{idx}].diameter",
                                value=float(circ["diameter"]),
                                feature_type=ftype,
                            )
                        )
            elif key in SIZE_FIELDS:
                report.checked += 1
                if _is_literal_number(val):
                    report.leaks.append(
                        Leak(
                            location=f"{fname}.{key}",
                            value=float(val),
                            feature_type=ftype,
                        )
                    )
    return report


def format_report(path: Path, report: LeakReport) -> str:
    lines: list[str] = []
    if report.has_leaks:
        lines.append(
            f"LEAKAGE (C3 wall) in {path.name}: raw size literals found -- "
            "every size dimension must be an rhs into a *_locals.txt:"
        )
        for leak in report.leaks:
            lines.append(f"  {leak.location} = {leak.value:g}   [{leak.feature_type}]")
        lines.append(
            "  Fix: replace each literal with "
            '{"rhs": "\\"YOUR_LOCAL\\""} and add the local (cited) to the '
            "part's *_locals.txt."
        )
    else:
        lines.append(
            f"C3 wall clean: {path.name} -- {report.checked} size "
            f"dimension(s) across {report.features} feature(s), all cited "
            "to locals (rhs)."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "C3 Locked-Numbers Wall guard: fail if any size dimension in an "
            "ai-sw-build spec is a raw literal instead of an rhs citation."
        )
    )
    parser.add_argument(
        "specs",
        nargs="+",
        help="One or more ai-sw-build spec JSON files to check.",
    )
    args = parser.parse_args(argv)

    any_leak = False
    for spec_arg in args.specs:
        path = Path(spec_arg)
        if not path.is_file():
            print(f"ERROR: spec not found: {path}", file=sys.stderr)
            return 2
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"ERROR: could not parse {path}: {exc}", file=sys.stderr)
            return 2
        if not isinstance(spec, dict):
            print(f"ERROR: {path} is not a spec object.", file=sys.stderr)
            return 2
        try:
            report = find_leaks(spec)
        except ValueError as exc:
            print(f"ERROR: {path}: {exc}", file=sys.stderr)
            return 2

        print(format_report(path, report))
        any_leak = any_leak or report.has_leaks

    return 1 if any_leak else 0


if __name__ == "__main__":
    raise SystemExit(main())
