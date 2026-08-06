"""Tier-B pre-flight: reconcile the interface ledger against Hardware/.

Blocking gate for the SM-IMP-009 SETUP step (spec §5, F7). A non-zero
exit blocks the modeling session; the only way past a red is
--allow-drift WITH --reason, which is echoed so it can be pasted into
the session note. Never wire this to auto-skip.

Usage:
    python -m tools.reconcile_interfaces --hardware-root <path-to-Hardware>
    python -m tools.reconcile_interfaces --hardware-root <path> \\
        --allow-drift --reason "why I'm proceeding despite drift"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hardware.interfaces import LEDGER, STATIONS
from tools.ledger_checks import ReconcileReport, reconcile


def format_report(report: ReconcileReport) -> str:
    lines: list[str] = []
    if report.drifts:
        lines.append("DRIFT (blocking):")
        for d in report.drifts:
            shown = "<absent>" if d.found is None else d.found
            lines.append(f"  {d.name} @ {d.location}: ledger={d.expected} file={shown}")
    if report.strays:
        lines.append("STRAY interfaces (candidates to add to the ledger):")
        for loc, name in report.strays:
            lines.append(f"  {name} @ {loc}")
    if report.unconsumed:
        lines.append(
            "UNCONSUMED interfaces (locked, no consumer yet -- informational):"
        )
        for name in report.unconsumed:
            lines.append(f"  {name}")
    if not lines:
        lines.append("Ledger reconciled clean: no drift.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile interface ledger vs Hardware/."
    )
    parser.add_argument(
        "--hardware-root",
        required=True,
        help="Path to the Hardware/ directory (non-git, machine-specific).",
    )
    parser.add_argument(
        "--allow-drift",
        action="store_true",
        help="Proceed despite drift. Requires --reason.",
    )
    parser.add_argument(
        "--reason",
        default="",
        help="Justification recorded when --allow-drift is used.",
    )
    args = parser.parse_args(argv)

    root = Path(args.hardware_root)
    if not root.is_dir():
        print(f"ERROR: --hardware-root not found: {root}", file=sys.stderr)
        return 2
    if args.allow_drift and not args.reason.strip():
        print("ERROR: --allow-drift requires a non-empty --reason.", file=sys.stderr)
        return 2

    report = reconcile(LEDGER, STATIONS, root)
    print(format_report(report))

    if report.has_blocking:
        if args.allow_drift:
            print(
                f"OVERRIDE (--allow-drift): proceeding. Reason: {args.reason.strip()}"
            )
            print("Record this override in the SM-IMP-009 session note.")
            return 0
        print(
            "BLOCKED: interface drift detected. Fix the locals, or re-run with "
            '--allow-drift --reason "...".',
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
