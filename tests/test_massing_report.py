"""Report formatting + pure check CLI (runs in CI, no OCP)."""

from massing.model import CheckResult
from massing.report import SCOPE_LINE, exit_code, format_report, overall_passed
from massing import run_checks
from tests.test_massing_envelopes import _fixture_ledger  # reuse the fixture


def test_exit_code_zero_when_all_pass() -> None:
    results = [CheckResult("clash", True, []), CheckResult("reach", True, [])]
    assert exit_code(results) == 0
    assert overall_passed(results) is True


def test_exit_code_one_when_any_fail() -> None:
    results = [CheckResult("clash", True, []), CheckResult("reach", False, [])]
    assert exit_code(results) == 1


def test_report_contains_scope_line_and_check_names() -> None:
    results = [CheckResult("clash", True, ["no hard clashes"])]
    text = format_report(results)
    assert SCOPE_LINE in text
    assert "clash" in text
    assert "PASS" in text


def test_build_and_check_runs_all_four(monkeypatch) -> None:
    results = run_checks.build_and_check(_fixture_ledger())
    assert [r.name for r in results] == ["clash", "reach", "alignment", "stackup"]


def test_main_returns_exit_code_using_injected_ledger(monkeypatch) -> None:
    monkeypatch.setattr(run_checks, "_load_ledger", lambda: _fixture_ledger())
    assert run_checks.main([]) == 0
