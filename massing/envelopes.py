"""Rough per-station envelopes + assembly of the Machine from the ledger.

LOW AUTHORITY. Exact positions/heights are read from the C1 ledger by the
names in KEYS. The rough box SIZES below are approximate, transcribed from
Hardware/00_Master_Assembly/SW_Design_Guide.md, and NEVER feed back into
the ledger. If the ledger and a size here disagree, the ledger wins.

Reconcile KEYS against the completed C1 ledger (see the plan, Task 3
Step 1). MVP static clash covers the FIXED stations only; the moving
sorting head/carriage is Phase-2 swept-volume, not a static clash box.
"""

from __future__ import annotations

from massing.model import AlignmentPair, Box, Machine, Station

# role -> ledger key name. THE reconciliation point.
KEYS: dict[str, str] = {
    "base_w": "D_BASE_W",
    "base_d": "D_BASE_D",
    "gantry_x_max": "D_GANTRY_X_MAX",
    "gantry_y_max": "D_GANTRY_Y_MAX",
    "z_belt": "D_Z_BELT",
    "z_funnel_lip": "D_Z_FUNNEL_LIP",
    "z_gate": "D_Z_GATE",
    "z_bin_rim": "D_Z_BIN_RIM",
    "grid_origin_x": "D_GRID_ORIGIN_X",
    "grid_origin_y": "D_GRID_ORIGIN_Y",
    "grid_pitch_x": "D_GRID_PITCH_X",
    "grid_pitch_y": "D_GRID_PITCH_Y",
    "bin_l": "D_BIN_L",
    "bin_w": "D_BIN_W",
    "bin_h": "D_BIN_H",
    "overflow_x": "D_OVERFLOW_X",
    "overflow_y": "D_OVERFLOW_Y",
    "x_hopper": "D_X_HOPPER",
    "x_vfeeder_out": "D_X_VFEEDER_OUT",
    "x_conveyor_right": "D_X_CONVEYOR_R",
    "x_camera": "D_X_CAMERA",
    "z_hopper_outlet": "D_Z_HOPPER_OUTLET",
    "hopper_inlet_dia": "D_HOPPER_INLET_DIA",
    "camera_h": "D_CAMERA_H",
    "z_camera": "D_Z_CAMERA",
    "head_inlet": "D_HEAD_INLET",
    "conveyor_l": "D_CONVEYOR_L",
    "conveyor_w": "D_CONVEYOR_W",
    "vfeeder_l": "D_VFEEDER_L",
}

# Layout topology (bin grid), not a locked interface length (SW_Design_Guide
# bin grid 4x2). Kept here as a massing layout constant.
_BIN_COLS = 4
_BIN_ROWS = 2

# Rough station heights not carried as interface dims (approximate, mm).
_HOPPER_BODY_H = 100.0  # inlet top ~Z193 - outlet ~Z93 (SW_Design_Guide)
_VFEEDER_H = 60.0  # rough channel body height
_CONVEYOR_FRAME_H = 30.0  # belt structure below the belt surface
_CAMERA_MAST_FOOT = 80.0  # rough mast/LED-ring footprint side


class MissingLedgerKeys(Exception):
    pass


def required_keys(ledger: dict) -> list[str]:
    """Ledger key names in KEYS that are absent from `ledger`, sorted."""
    return sorted(k for k in KEYS.values() if k not in ledger)


def _v(ledger: dict, role: str) -> float:
    return float(ledger[KEYS[role]]["value"])


def build_machine(
    ledger: dict, align_tol: float = 1.0, min_clearance: float = 3.0
) -> Machine:
    missing = required_keys(ledger)
    if missing:
        raise MissingLedgerKeys(
            "ledger is missing required interface keys: " + ", ".join(missing)
        )

    x_hopper = _v(ledger, "x_hopper")
    x_vf = _v(ledger, "x_vfeeder_out")
    x_conv_r = _v(ledger, "x_conveyor_right")
    x_cam = _v(ledger, "x_camera")
    conv_l = _v(ledger, "conveyor_l")
    conv_w = _v(ledger, "conveyor_w")
    vf_l = _v(ledger, "vfeeder_l")
    dia = _v(ledger, "hopper_inlet_dia")
    z_belt = _v(ledger, "z_belt")
    z_hop = _v(ledger, "z_hopper_outlet")
    z_cam = _v(ledger, "z_camera")

    # --- Fixed-station rough boxes (centered on the belt centerline Y=0) ---
    hopper = Station(
        "hopper",
        Box(x_hopper - dia / 2, -dia / 2, z_hop, dia, dia, _HOPPER_BODY_H),
    )
    vfeeder = Station(
        "vfeeder",
        Box(x_vf - vf_l, -conv_w / 2, z_belt, vf_l, conv_w, _VFEEDER_H),
    )
    conveyor = Station(
        "conveyor",
        Box(
            x_conv_r - conv_l,
            -conv_w / 2,
            z_belt - _CONVEYOR_FRAME_H,
            conv_l,
            conv_w,
            _CONVEYOR_FRAME_H,
        ),
    )
    camera = Station(
        "camera",
        Box(
            x_cam - _CAMERA_MAST_FOOT / 2,
            -_CAMERA_MAST_FOOT / 2,
            z_belt,
            _CAMERA_MAST_FOOT,
            _CAMERA_MAST_FOOT,
            z_cam - z_belt,
        ),
    )
    stations = [hopper, vfeeder, conveyor, camera]
    clash_allow = {
        frozenset({"hopper", "vfeeder"}),  # hopper drops into feeder
        frozenset({"vfeeder", "conveyor"}),  # feeder hands off to belt
    }

    # --- Reach: gantry travel box + bin/overflow centers ---
    gantry_travel = Box(
        0, 0, 0, _v(ledger, "gantry_x_max"), _v(ledger, "gantry_y_max"), 1
    )
    gx0, gy0 = _v(ledger, "grid_origin_x"), _v(ledger, "grid_origin_y")
    px, py = _v(ledger, "grid_pitch_x"), _v(ledger, "grid_pitch_y")
    reach_targets: list[tuple[str, tuple[float, float]]] = []
    for c in range(_BIN_COLS):
        for r in range(_BIN_ROWS):
            reach_targets.append((f"bin_{c}{r}", (gx0 + c * px, gy0 + r * py)))
    reach_targets.append(
        ("overflow", (_v(ledger, "overflow_x"), _v(ledger, "overflow_y")))
    )

    # --- Alignment: cross-key coincidences (offset 0 on a clean ledger) ---
    belt_infeed_x = x_conv_r - conv_l
    alignment_pairs = [
        AlignmentPair(
            "feeder_out<->belt_infeed",
            (x_vf, 0.0, 0.0),
            (belt_infeed_x, 0.0, 0.0),
            ("x",),
            align_tol,
        ),
        AlignmentPair(
            "feeder_out<->belt_centerline",
            (x_vf, 0.0, 0.0),
            (x_vf, 0.0, 0.0),
            ("y",),
            align_tol,
        ),
        AlignmentPair(
            "camera<->belt_centerline",
            (x_cam, 0.0, 0.0),
            (x_cam, 0.0, 0.0),
            ("y",),
            align_tol,
        ),
    ]

    # --- Stack-up: the handoff Z-chain (must be strictly descending) ---
    z_chain = [
        ("belt", z_belt),
        ("funnel_lip", _v(ledger, "z_funnel_lip")),
        ("gate", _v(ledger, "z_gate")),
        ("bin_rim", _v(ledger, "z_bin_rim")),
    ]

    return Machine(
        stations=stations,
        clash_allow=clash_allow,
        min_clearance=min_clearance,
        gantry_travel=gantry_travel,
        reach_targets=reach_targets,
        alignment_pairs=alignment_pairs,
        z_chain=z_chain,
    )
