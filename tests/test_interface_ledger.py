"""Tier A -- the interface ledger's own integrity (runs in CI).

Checks hardware/interfaces.py against the record schema and validates
its AST purity (F4). Does NOT touch the non-git Hardware/ path -- that
is Tier B (test_reconcile_interfaces.py + the live SETUP pre-flight).
"""

from pathlib import Path

from hardware.interfaces import LEDGER, STATIONS
import hardware.interfaces as interfaces_mod
from tools.ledger_checks import validate_ledger_schema, assert_pure_data


def test_ledger_schema_is_valid() -> None:
    """The seeded ledger has no schema violations."""
    errors = validate_ledger_schema(LEDGER, STATIONS)
    assert errors == [], "ledger schema errors:\n" + "\n".join(errors)


def test_every_constant_is_a_d_name() -> None:
    """Ledger keys are program-wide interface dims (D_*, not _D_ or Sxx_)."""
    for name in LEDGER:
        assert name.startswith("D_"), f"{name} is not a D_* interface name"


def test_every_binding_token_is_known() -> None:
    """Every binding refers to a declared STATIONS token."""
    for name, rec in LEDGER.items():
        bindings = rec["bindings"]
        assert isinstance(bindings, list)  # also narrows `object` for mypy
        for token in bindings:
            assert token in STATIONS, f"{name} binds unknown station {token!r}"


def test_seed_covers_verified_restatements() -> None:
    """D_Z_BELT is restated in globals + S1a + S1b + S5 (verified in the files).

    S5 was added to the coverage by the M4 audit (Task 7): s5_gantry_locals
    restates D_Z_BELT = 75 as the belt-height reference for the gantry stack.
    """
    assert LEDGER["D_Z_BELT"]["bindings"] == ["globals", "S1a", "S1b", "S5"]


def test_real_ledger_is_pure_data() -> None:
    """hardware/interfaces.py contains only literal data (F4)."""
    errors = assert_pure_data(interfaces_mod.__file__)
    assert errors == [], "purity errors:\n" + "\n".join(errors)


def test_guard_rejects_import(tmp_path: Path) -> None:
    bad = tmp_path / "bad_import.py"
    bad.write_text("import os\nLEDGER = {}\n", encoding="utf-8")
    assert assert_pure_data(bad) != []


def test_guard_rejects_expression_rhs(tmp_path: Path) -> None:
    bad = tmp_path / "bad_expr.py"
    bad.write_text("D_X = 75 - 1\n", encoding="utf-8")
    assert assert_pure_data(bad) != []


def test_guard_rejects_call_rhs(tmp_path: Path) -> None:
    bad = tmp_path / "bad_call.py"
    bad.write_text("LEDGER = dict(a=1)\n", encoding="utf-8")
    assert assert_pure_data(bad) != []


def test_guard_allows_docstring_and_literal_dict(tmp_path: Path) -> None:
    good = tmp_path / "good.py"
    good.write_text('"""doc."""\nLEDGER = {"D_X": {"value": 1}}\n', encoding="utf-8")
    assert assert_pure_data(good) == []
