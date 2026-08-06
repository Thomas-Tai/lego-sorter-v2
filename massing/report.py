"""Numeric pass/fail report with the mandatory scope line (spec §7)."""

from __future__ import annotations

from massing.model import CheckResult

SCOPE_LINE = (
    "Massing-level geometry only -- precise SW mates and the physics in "
    "spec §11 are NOT verified. A green run means the geometry is sane, "
    "never that the machine works."
)


def overall_passed(results: list[CheckResult]) -> bool:
    return all(r.passed for r in results)


def exit_code(results: list[CheckResult]) -> int:
    return 0 if overall_passed(results) else 1


def format_report(results: list[CheckResult]) -> str:
    lines = ["=== Full-System Massing Pre-Check ==="]
    for r in results:
        lines.append(f"[{'PASS' if r.passed else 'FAIL'}] {r.name}")
        for d in r.details:
            lines.append(f"    {d}")
    verdict = "ALL CHECKS PASS" if overall_passed(results) else "CHECKS FAILED"
    lines.append(f"--- {verdict} ---")
    lines.append(SCOPE_LINE)
    return "\n".join(lines)
