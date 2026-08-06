"""Tier B logic -- locals parser + reconcile core + CLI (runs in CI).

Uses tmp fixtures, never the real Hardware/ path. The live run against
Hardware/ is the SM-IMP-009 SETUP pre-flight, not a CI job.
"""

from decimal import Decimal
from pathlib import Path

from tools.ledger_checks import parse_interface_lines


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_parses_d_star_numeric_lines(tmp_path: Path) -> None:
    f = _write(
        tmp_path / "locals.txt",
        '"D_Z_BELT"              = 75.0\n'
        '"D_CONVEYOR_W"          = 80.0\n'
        '"D_HOPPER_INLET_DIA"    = 150\n',
    )
    parsed = parse_interface_lines(f)
    assert parsed == {
        "D_Z_BELT": Decimal("75.0"),
        "D_CONVEYOR_W": Decimal("80.0"),
        "D_HOPPER_INLET_DIA": Decimal("150"),
    }


def test_handles_globals_no_space_before_equals(tmp_path: Path) -> None:
    # lego_sorter_globals.txt uses `"NAME"= value` (no space before =).
    f = _write(tmp_path / "g.txt", '"D_BASE_W"= 1000\n')
    assert parse_interface_lines(f) == {"D_BASE_W": Decimal("1000")}


def test_skips_underscore_d_semi_private(tmp_path: Path) -> None:
    f = _write(tmp_path / "g.txt", '"_D_X_ENDSTOP_X"= 32.5\n')
    assert parse_interface_lines(f) == {}


def test_skips_part_internal_sxx_names(tmp_path: Path) -> None:
    f = _write(tmp_path / "l.txt", '"S1B_BELT_T"            = 1.5\n')
    assert parse_interface_lines(f) == {}


def test_skips_expression_rhs(tmp_path: Path) -> None:
    # A D_* name assigned an expression is NOT a restated literal (B2).
    f = _write(
        tmp_path / "l.txt",
        '"D_Z_BELT"    = 75.0\n' '"D_FOO"       = "D_Z_BELT"-1.5\n',
    )
    assert parse_interface_lines(f) == {"D_Z_BELT": Decimal("75.0")}


def test_skips_comment_as_variable_lines(tmp_path: Path) -> None:
    f = _write(
        tmp_path / "l.txt",
        '"// leg_H"= floor_Z - 5\n' '"D_VFEEDER_L"           = 200.0\n',
    )
    assert parse_interface_lines(f) == {"D_VFEEDER_L": Decimal("200.0")}


def test_tolerates_encoding_errors(tmp_path: Path) -> None:
    # A lone invalid byte in a comment must not crash the parser (B13).
    f = tmp_path / "moji.txt"
    f.write_bytes(b'"// bad \xff byte here"= 1\n"D_Z_BELT" = 75.0\n')
    assert parse_interface_lines(f) == {"D_Z_BELT": Decimal("75.0")}


def test_decimal_preserves_75_vs_75_0(tmp_path: Path) -> None:
    f = _write(tmp_path / "l.txt", '"D_A" = 75\n"D_B" = 75.0\n')
    parsed = parse_interface_lines(f)
    assert parsed["D_A"] == parsed["D_B"]  # Decimal("75") == Decimal("75.0")
