"""Tier B logic -- locals parser + reconcile core + CLI (runs in CI).

Uses tmp fixtures, never the real Hardware/ path. The live run against
Hardware/ is the SM-IMP-009 SETUP pre-flight, not a CI job.
"""

from decimal import Decimal
from pathlib import Path

from tools.ledger_checks import parse_interface_lines, reconcile, ReconcileReport


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


def test_tolerates_leading_utf8_bom(tmp_path: Path) -> None:
    # SolidWorks/Windows exports commonly prepend a UTF-8 BOM; it must not
    # break the anchor on the first D_* line (else that dim reads absent).
    f = tmp_path / "g.txt"
    f.write_bytes(b"\xef\xbb\xbf" + b'"D_BASE_W"= 1000\n"D_BASE_D"= 450.0\n')
    assert parse_interface_lines(f) == {
        "D_BASE_W": Decimal("1000"),
        "D_BASE_D": Decimal("450.0"),
    }


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


def _hardware(tmp_path: Path, files: dict[str, str]) -> Path:
    """Build a fake Hardware/ root: {relpath: file-text}."""
    root = tmp_path / "Hardware"
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


_STATIONS = {
    "globals": "00/glob.txt",
    "S1a": "S1a/loc.txt",
}


def test_clean_reconcile_has_no_drift(tmp_path: Path) -> None:
    ledger = {
        "D_Z_BELT": {
            "value": 75.0,
            "unit": "mm",
            "lock_id": "IF-24",
            "bindings": ["globals", "S1a"],
        },
    }
    root = _hardware(
        tmp_path,
        {
            "00/glob.txt": '"D_Z_BELT"= 75.0\n',
            "S1a/loc.txt": '"D_Z_BELT" = 75\n',  # 75 vs 75.0 must still match (F3)
        },
    )
    report = reconcile(ledger, _STATIONS, root)
    assert report.drifts == []
    assert report.has_blocking is False


def test_value_mismatch_is_drift(tmp_path: Path) -> None:
    ledger = {
        "D_BASE_W": {
            "value": 1000,
            "unit": "mm",
            "lock_id": "DL-02/D-10",
            "bindings": ["globals"],
        },
    }
    root = _hardware(tmp_path, {"00/glob.txt": '"D_BASE_W"= 650\n'})
    report = reconcile(ledger, _STATIONS, root)
    assert report.has_blocking is True
    assert report.drifts[0].name == "D_BASE_W"
    assert report.drifts[0].found == "650"
    assert report.drifts[0].expected == "1000"


def test_declared_but_missing_is_drift(tmp_path: Path) -> None:
    ledger = {
        "D_Z_BELT": {
            "value": 75.0,
            "unit": "mm",
            "lock_id": "IF-24",
            "bindings": ["globals", "S1a"],
        },
    }
    root = _hardware(
        tmp_path,
        {
            "00/glob.txt": '"D_Z_BELT"= 75.0\n',
            "S1a/loc.txt": '"S1A_V_ANGLE" = 60.0\n',  # D_Z_BELT absent here
        },
    )
    report = reconcile(ledger, _STATIONS, root)
    assert report.has_blocking is True
    assert report.drifts[0].location == "S1a"
    assert report.drifts[0].found is None


def test_unconsumed_interface_is_informational(tmp_path: Path) -> None:
    ledger = {
        "D_CAMERA_H": {
            "value": 150.0,
            "unit": "mm",
            "lock_id": "SM-DES-004 §2.3",
            "bindings": [],
        },
    }
    root = _hardware(tmp_path, {"00/glob.txt": "\n"})
    report = reconcile(ledger, _STATIONS, root)
    assert report.drifts == []
    assert report.unconsumed == ["D_CAMERA_H"]


def test_stray_d_name_is_warning(tmp_path: Path) -> None:
    ledger = {
        "D_Z_BELT": {
            "value": 75.0,
            "unit": "mm",
            "lock_id": "IF-24",
            "bindings": ["globals"],
        },
    }
    root = _hardware(
        tmp_path,
        {
            "00/glob.txt": '"D_Z_BELT"= 75.0\n"D_MYSTERY"= 42.0\n',
        },
    )
    report = reconcile(ledger, _STATIONS, root)
    assert report.drifts == []
    assert ("globals", "D_MYSTERY") in report.strays


from tools import reconcile_interfaces


def _mk_ledger_files(tmp_path: Path, glob_text: str) -> Path:
    root = tmp_path / "Hardware"
    (root / "00").mkdir(parents=True)
    (root / "00" / "glob.txt").write_text(glob_text, encoding="utf-8")
    return root


def test_cli_exit_0_when_clean(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        reconcile_interfaces,
        "LEDGER",
        {
            "D_BASE_W": {
                "value": 1000,
                "unit": "mm",
                "lock_id": "DL-02/D-10",
                "bindings": ["globals"],
            },
        },
    )
    monkeypatch.setattr(reconcile_interfaces, "STATIONS", {"globals": "00/glob.txt"})
    root = _mk_ledger_files(tmp_path, '"D_BASE_W"= 1000\n')
    assert reconcile_interfaces.main(["--hardware-root", str(root)]) == 0


def test_cli_exit_1_on_drift(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        reconcile_interfaces,
        "LEDGER",
        {
            "D_BASE_W": {
                "value": 1000,
                "unit": "mm",
                "lock_id": "DL-02/D-10",
                "bindings": ["globals"],
            },
        },
    )
    monkeypatch.setattr(reconcile_interfaces, "STATIONS", {"globals": "00/glob.txt"})
    root = _mk_ledger_files(tmp_path, '"D_BASE_W"= 650\n')
    assert reconcile_interfaces.main(["--hardware-root", str(root)]) == 1


def test_cli_allow_drift_needs_reason(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        reconcile_interfaces,
        "LEDGER",
        {
            "D_BASE_W": {
                "value": 1000,
                "unit": "mm",
                "lock_id": "DL-02/D-10",
                "bindings": ["globals"],
            },
        },
    )
    monkeypatch.setattr(reconcile_interfaces, "STATIONS", {"globals": "00/glob.txt"})
    root = _mk_ledger_files(tmp_path, '"D_BASE_W"= 650\n')
    assert (
        reconcile_interfaces.main(["--hardware-root", str(root), "--allow-drift"]) == 2
    )


def test_cli_allow_drift_with_reason_exits_0(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        reconcile_interfaces,
        "LEDGER",
        {
            "D_BASE_W": {
                "value": 1000,
                "unit": "mm",
                "lock_id": "DL-02/D-10",
                "bindings": ["globals"],
            },
        },
    )
    monkeypatch.setattr(reconcile_interfaces, "STATIONS", {"globals": "00/glob.txt"})
    root = _mk_ledger_files(tmp_path, '"D_BASE_W"= 650\n')
    code = reconcile_interfaces.main(
        ["--hardware-root", str(root), "--allow-drift", "--reason", "WIP rebuild"]
    )
    assert code == 0
    assert "WIP rebuild" in capsys.readouterr().out


def test_cli_missing_hardware_root_exits_2(tmp_path: Path) -> None:
    assert reconcile_interfaces.main(["--hardware-root", str(tmp_path / "nope")]) == 2
