"""Run the four massing checks on the real ledger and print a report.

Pure (no OCP): this is the check/report half of the tool. The STEP/GLB
demo is massing/smoke_build.py (OCP, local-only). _load_ledger is kept
indirect so this module imports (and tests) without C1 present.

Usage:
    python -m massing.run_checks
"""

from __future__ import annotations

import argparse

from massing.checks import (
    check_alignment,
    check_clash,
    check_reach,
    check_stackup,
)
from massing.envelopes import build_machine
from massing.model import CheckResult
from massing.report import exit_code, format_report


def _load_ledger() -> dict:
    from hardware.interfaces import LEDGER  # lazy: needs C1 complete

    return LEDGER


def build_and_check(ledger: dict) -> list[CheckResult]:
    machine = build_machine(ledger)
    return [
        check_clash(machine),
        check_reach(machine),
        check_alignment(machine),
        check_stackup(machine),
    ]


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(
        description="Full-system massing static pre-check."
    ).parse_args(argv)
    results = build_and_check(_load_ledger())
    print(format_report(results))
    return exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())
