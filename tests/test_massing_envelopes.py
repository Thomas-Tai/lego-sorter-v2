"""Envelope + build_machine against a synthetic fixture ledger (CI, no OCP).

Uses KEYS to synthesize a ledger, so the test is independent of the exact
D_* names C1 finally chose -- it exercises whatever KEYS maps to.
"""

import pytest

from massing.envelopes import KEYS, MissingLedgerKeys, build_machine, required_keys

# Role -> value, matching current SW_Design_Guide / seeded ledger numbers.
_ROLE_VALUES = {
    "base_w": 1000.0,
    "base_d": 450.0,
    "gantry_x_max": 355.0,
    "gantry_y_max": 160.0,
    "z_belt": 75.0,
    "z_funnel_lip": 67.0,
    "z_gate": 42.0,
    "z_bin_rim": 35.0,
    "grid_origin_x": 55.0,
    "grid_origin_y": 30.0,
    "grid_pitch_x": 75.0,
    "grid_pitch_y": 55.0,
    "bin_l": 70.0,
    "bin_w": 50.0,
    "bin_h": 35.0,
    "overflow_x": 280.0,
    "overflow_y": 140.0,
    "x_hopper": -502.5,
    "x_vfeeder_out": -332.5,
    "x_conveyor_right": -32.5,
    "x_camera": -132.5,
    "z_hopper_outlet": 93.0,
    "hopper_inlet_dia": 150.0,
    "camera_h": 150.0,
    "z_camera": 225.0,
    "head_inlet": 40.0,
    "conveyor_l": 300.0,
    "conveyor_w": 80.0,
    "vfeeder_l": 200.0,
}


def _fixture_ledger() -> dict:
    return {
        KEYS[role]: {"value": v, "unit": "mm", "lock_id": "TEST-1", "bindings": []}
        for role, v in _ROLE_VALUES.items()
    }


def test_required_keys_reports_all_missing_on_empty_ledger() -> None:
    missing = required_keys({})
    assert set(missing) == set(KEYS.values())


def test_build_machine_raises_on_missing_key() -> None:
    ledger = _fixture_ledger()
    del ledger[KEYS["z_belt"]]
    with pytest.raises(MissingLedgerKeys):
        build_machine(ledger)


def test_build_machine_populates_all_four_check_inputs() -> None:
    m = build_machine(_fixture_ledger())
    assert len(m.stations) == 4  # hopper, vfeeder, conveyor, camera (fixed only)
    assert len(m.reach_targets) == 9  # 4x2 bins + 1 overflow
    assert [z for _, z in m.z_chain] == [75.0, 67.0, 42.0, 35.0]
    assert m.alignment_pairs  # at least one cross-key pair


def test_feeder_belt_infeed_alignment_is_zero_on_clean_ledger() -> None:
    # x_vfeeder_out (-332.5) must equal the belt infeed edge
    # (x_conveyor_right - conveyor_l = -32.5 - 300 = -332.5): offset 0.
    from massing.checks import check_alignment

    m = build_machine(_fixture_ledger())
    assert check_alignment(m).passed is True
