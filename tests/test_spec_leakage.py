"""Tests for the C3 Locked-Numbers Wall guard (tools/check_spec_leakage.py)."""

from __future__ import annotations

import json

import pytest

from tools.check_spec_leakage import find_leaks, main


def _rhs(var: str) -> dict:
    return {"rhs": f'"{var}"'}


# A representative clean part, in the bake-off style: Front-plane bosses +
# +z-face holes, every SIZE field an rhs; positional fields literal.
CLEAN_SPEC = {
    "schema_version": 1,
    "name": "SM-HW-S5-014_CableChainMount_v01",
    "locals": "C:\\path\\s5_gantry_locals.txt",
    "features": [
        {
            "type": "sketch_rectangle_on_plane",
            "name": "SK_Clip",
            "plane": "Front",
            "width": _rhs("S5_CCM_D"),
            "height": _rhs("S5_CCM_W"),
            "center": {"x": 0.0, "y": 0.0},
        },
        {
            "type": "boss_extrude_blind",
            "name": "Extrude_Clip",
            "sketch": "SK_Clip",
            "depth": _rhs("S5_CCM_H"),
            "flip": False,
        },
        {
            "type": "sketch_circle_on_face",
            "name": "SK_Hole_A",
            "of_feature": "Extrude_Clip",
            "face": "+z",
            "diameter": _rhs("S5_M3_CLEAR"),
            "center": {"u": 0.0, "v": 6.0},
        },
    ],
}


def test_clean_spec_has_no_leaks():
    report = find_leaks(CLEAN_SPEC)
    assert not report.has_leaks
    assert report.features == 3
    # width, height, depth, diameter = 4 size fields inspected
    assert report.checked == 4


def test_literal_size_field_is_flagged():
    spec = json.loads(json.dumps(CLEAN_SPEC))  # deep copy
    spec["features"][1]["depth"] = 10.0  # raw literal leaked
    report = find_leaks(spec)
    assert report.has_leaks
    assert len(report.leaks) == 1
    leak = report.leaks[0]
    assert leak.location == "Extrude_Clip.depth"
    assert leak.value == 10.0
    assert leak.feature_type == "boss_extrude_blind"


def test_positional_literals_are_exempt():
    # center u/v, x/y are literal-only per the bridge schema -- never flagged.
    spec = {
        "features": [
            {
                "type": "sketch_circle_on_face",
                "name": "SK_Hole",
                "face": "+z",
                "diameter": _rhs("S5_M3_CLEAR"),
                "center": {"u": 13.5, "v": -7.5},  # raw literals, but exempt
            }
        ]
    }
    report = find_leaks(spec)
    assert not report.has_leaks


def test_angular_and_count_fields_are_exempt():
    spec = {
        "features": [
            {
                "type": "circular_pattern",
                "name": "CP",
                "seed": "SK_Hole",
                "axis": {"x": 0.0, "y": 0.0, "z": 5.0},
                "count": 6,  # integer count, exempt
                "total_angle": 360.0,  # angular, exempt
            },
            {
                "type": "sketch_polygon",
                "name": "Hex",
                "plane": "Front",
                "center": {"x": 0.0, "y": 0.0},
                "sides": 6,  # count, exempt
                "radius": _rhs("S5_HEX_R"),
                "angle_deg": 0.0,  # angular, exempt
            },
        ]
    }
    report = find_leaks(spec)
    assert not report.has_leaks


def test_circles_diameter_literal_flagged_but_uv_exempt():
    spec = {
        "features": [
            {
                "type": "sketch_circles_on_face",
                "name": "SK_Pattern",
                "face": "+z",
                "circles": [
                    {"u": 12.5, "v": 0.0, "diameter": _rhs("S5_M3_CLEAR")},
                    {"u": -12.5, "v": 0.0, "diameter": 3.2},  # literal leak
                ],
            }
        ]
    }
    report = find_leaks(spec)
    assert report.has_leaks
    assert len(report.leaks) == 1
    assert report.leaks[0].location == "SK_Pattern.circles[1].diameter"
    assert report.leaks[0].value == 3.2


def test_multiple_leaks_across_features():
    spec = {
        "features": [
            {"type": "boss_extrude_blind", "name": "A", "depth": 5.0},
            {"type": "cut_extrude_blind", "name": "B", "depth": 2.0},
        ]
    }
    report = find_leaks(spec)
    assert len(report.leaks) == 2


def test_bool_is_not_treated_as_number():
    # flip is a bool, and even a stray bool in a size field must not count
    # as a numeric literal (bool is an int subclass in Python).
    spec = {
        "features": [{"type": "boss_extrude_blind", "name": "A", "depth": _rhs("D")}]
    }
    assert not find_leaks(spec).has_leaks


def test_missing_features_raises():
    with pytest.raises(ValueError):
        find_leaks({"schema_version": 1, "name": "x"})


def test_expect_and_comment_blocks_are_ignored():
    spec = {
        "features": [
            {
                "type": "boss_extrude_blind",
                "name": "A",
                "_comment": "base per §13.4",
                "depth": _rhs("D"),
                "_expect": {"mass_delta_mm3": 5000.0, "tolerance_mm3": 50.0},
            }
        ]
    }
    assert not find_leaks(spec).has_leaks


# --- CLI ---


def _write(tmp_path, name, spec) -> str:
    p = tmp_path / name
    p.write_text(json.dumps(spec), encoding="utf-8")
    return str(p)


def test_main_clean_returns_zero(tmp_path, capsys):
    path = _write(tmp_path, "clean.json", CLEAN_SPEC)
    assert main([path]) == 0
    assert "C3 wall clean" in capsys.readouterr().out


def test_main_leaky_returns_one(tmp_path, capsys):
    leaky = json.loads(json.dumps(CLEAN_SPEC))
    leaky["features"][1]["depth"] = 10.0
    path = _write(tmp_path, "leaky.json", leaky)
    assert main([path]) == 1
    assert "LEAKAGE" in capsys.readouterr().out


def test_main_missing_file_returns_two(capsys):
    assert main(["C:/no/such/spec.json"]) == 2
    assert "not found" in capsys.readouterr().err


def test_main_bad_json_returns_two(tmp_path, capsys):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert main([str(p)]) == 2
    assert "could not parse" in capsys.readouterr().err


def test_main_multiple_specs_one_leaky(tmp_path):
    clean = _write(tmp_path, "clean.json", CLEAN_SPEC)
    leaky_spec = json.loads(json.dumps(CLEAN_SPEC))
    leaky_spec["features"][1]["depth"] = 10.0
    leaky = _write(tmp_path, "leaky.json", leaky_spec)
    assert main([clean, leaky]) == 1
