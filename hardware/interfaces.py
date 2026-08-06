"""Published interface ledger (executable form of SM-DES-004 §2.2b).

AUTHORITY. This file is pure literal data: no imports, no logic, no
expressions. It is the git-tracked source of truth for cross-part
interface dimensions. lego_sorter_globals.txt and each part's
*_locals.txt are reconciliation TARGETS (see tools/reconcile_interfaces.py).

Record shape (validated at runtime by tools.ledger_checks, never by
static types -- imports are forbidden here so TypedDict cannot be used):
    "D_NAME": {
        "value":    <int|float>,      # nominal locked value
        "unit":     "<mm|deg|...>",
        "lock_id":  "<SM-DES-004 lock reference>",  # IF-xx / DL-xx / section
        "bindings": [<location tokens>],  # where this value is restated
    }
Binding tokens are keys of STATIONS below. Editing any value here is a
design-lock change (spec §5, B4): same approval as amending SM-DES-004.

COMPLETE (M4 audit, plan Task 7, 2026-08-06): the full line-by-line
transcription from locked SM-DES-004 (+ SM-DES-006 §4.3 for bin-grid
coordinates). Every value was cross-checked against SM-DES-004; none were
guessed. Bin-grid coordinates (offsets) are governed by SM-DES-006, the
authoritative doc per SM-DES-004 §8.

Three Hardware-source edits are still owed (gated SW-source changes,
tracked for the next Stream-2 CAD session) -- until they land, the live
reconcile pre-flight will (correctly) report them:
  - S0_Hopper local D_HOPPER_INLET_DIA 140 -> 150 (authority is 150;
    ledger records 150, so the S0 restatement DRIFTS until edited).
  - globals rename D_Z_HEAD_FUNNEL -> D_Z_FUNNEL_LIP and
    D_Z_HEAD_GATE -> D_Z_GATE_BOTTOM (canonical = S5's names; the old
    globals names remain as STRAY warnings until renamed).
"""

STATIONS = {
    "globals": "00_Master_Assembly/Model/lego_sorter_globals.txt",
    "S0": "S0_Hopper/Model/s0_hopper_locals.txt",
    "S1a": "S1a_VFeeder/Model/s1a_vfeeder_locals.txt",
    "S1b": "S1b_Conveyor/Model/s1b_conveyor_locals.txt",
    "S5": "S5_Gantry/Model/s5_gantry_locals.txt",
}

LEDGER = {
    # --- Base plate (SM-DES-004 §2.2b / §6.1, DL-02 / D-10) ---
    "D_BASE_W": {
        "value": 1000,
        "unit": "mm",
        "lock_id": "DL-02/D-10",
        "bindings": ["globals", "S5"],
    },
    "D_BASE_D": {
        "value": 450.0,
        "unit": "mm",
        "lock_id": "DL-02/D-10",
        "bindings": ["globals", "S5"],
    },
    "D_BASE_T": {
        "value": 8.0,
        "unit": "mm",
        "lock_id": "DL-02/D-10",
        "bindings": ["globals", "S5"],
    },
    "D_GANTRY_RIGHT_EDGE_X": {
        "value": 450.0,
        "unit": "mm",
        "lock_id": "DL-02/D-10",
        "bindings": ["S5"],
    },
    # --- Gantry travel + rails + drivetrain (SM-DES-004 §6) ---
    "D_GANTRY_X_MAX": {
        "value": 355.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §6.5",
        "bindings": ["globals"],
    },
    "D_GANTRY_Y_MAX": {
        "value": 160.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §6.5",
        "bindings": ["globals"],
    },
    "D_X_RAIL_L": {
        "value": 420.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §6.3",
        "bindings": ["globals"],
    },
    "D_Y_RAIL_L": {
        "value": 220.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §6.4",
        "bindings": ["globals"],
    },
    "D_RAIL_WIDTH": {
        "value": 12.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §6.3",
        "bindings": ["globals"],
    },
    "D_CARRIAGE_H": {
        "value": 13.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §6.2",
        "bindings": ["globals"],
    },
    "D_NEMA17_FACE": {
        "value": 42.3,
        "unit": "mm",
        "lock_id": "SM-DES-004 §6.3",
        "bindings": ["globals"],
    },
    "D_NEMA17_HOLE_SPACING": {
        "value": 31.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §6.3",
        "bindings": ["globals"],
    },
    "D_GT2_PULLEY_OD": {
        "value": 12.7,
        "unit": "mm",
        "lock_id": "SM-DES-004 §6.3",
        "bindings": ["globals"],
    },
    "D_GT2_PULLEY_BORE": {
        "value": 5.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §6.3",
        "bindings": ["globals"],
    },
    # --- X-carriage plate (SM-DES-004 §6.2) ---
    "D_X_PLATE_L": {
        "value": 250.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §6.2",
        "bindings": ["globals"],
    },
    "D_X_PLATE_W": {
        "value": 50.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §6.2",
        "bindings": ["globals"],
    },
    "D_X_PLATE_T": {
        "value": 5.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §6.2",
        "bindings": ["globals"],
    },
    # --- Master Z heights (SM-DES-004 §2.3 / §6.2 stack-up) ---
    "D_Z_BASE": {
        "value": 0.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §2.3",
        "bindings": ["globals"],
    },
    "D_Z_RAIL_MOUNT": {
        "value": 50.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §6.2",
        "bindings": ["globals"],
    },
    "D_Z_RAIL_TOP": {
        "value": 58.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §6.2",
        "bindings": ["globals"],
    },
    "D_Z_BELT": {
        "value": 75.0,
        "unit": "mm",
        "lock_id": "IF-24",
        "bindings": ["globals", "S1a", "S1b", "S5"],
    },
    "D_Z_CARRIAGE": {
        "value": 76.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §6.2",
        "bindings": ["globals"],
    },
    "D_Z_BIN_RIM": {
        "value": 35.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §2.3",
        "bindings": ["globals", "S5"],
    },
    # --- Sorting head + belt-end handoff (SM-DES-004 §7.2) ---
    # Canonical names = S5's (head-mount owner); globals still restates the
    # same two Z-heights as D_Z_HEAD_FUNNEL / D_Z_HEAD_GATE -> gated rename.
    "D_Z_FUNNEL_LIP": {
        "value": 67.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §7.2",
        "bindings": ["S5"],
    },
    "D_Z_GATE_BOTTOM": {
        "value": 42.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §7.2",
        "bindings": ["S5"],
    },
    "D_HEAD_INLET": {
        "value": 40.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §7.2",
        "bindings": ["globals"],
    },
    "D_HEAD_CHAMBER": {
        "value": 35.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §7.2",
        "bindings": ["globals"],
    },
    "D_HEAD_TOTAL_H": {
        "value": 25.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §7.2",
        "bindings": ["globals"],
    },
    # --- Conveyor + V-feeder (SM-DES-004 §2.2b / §4) ---
    "D_CONVEYOR_L": {
        "value": 300.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §2.2b",
        "bindings": ["globals", "S1b"],
    },
    "D_CONVEYOR_W": {
        "value": 80.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §2.2b",
        "bindings": ["globals", "S1b"],
    },
    "D_VFEEDER_L": {
        "value": 200.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §2.2b",
        "bindings": ["globals", "S1a"],
    },
    # --- Hopper (SM-DES-004 §3) ---
    "D_HOPPER_OUTLET_W": {
        "value": 28.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §3",
        "bindings": ["globals", "S1a", "S0"],
    },
    "D_HOPPER_AIR_GAP": {
        "value": 8.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §3.3",
        "bindings": ["globals", "S1a", "S0"],
    },
    "D_Z_HOPPER_OUTLET": {
        "value": 93.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §2.3",
        "bindings": ["globals", "S1a", "S0"],
    },
    "D_HOPPER_INLET_DIA": {
        # Authority (SM-DES-004 §3.2/§3.3) + globals = 150; S0 local still
        # restates 140 (stale) -> intentional visible DRIFT until S0 edited.
        "value": 150,
        "unit": "mm",
        "lock_id": "SM-DES-004 §3.2",
        "bindings": ["globals", "S0"],
    },
    # --- Camera station (SM-DES-004 §2.3 / §5.2) ---
    "D_CAMERA_H": {
        "value": 150.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §2.3",
        "bindings": ["globals"],
    },
    "D_Z_CAMERA": {
        "value": 225.0,
        "unit": "mm",
        "lock_id": "IF-19",
        "bindings": ["globals"],
    },
    "D_LED_RING_OD": {
        "value": 80.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §5.2",
        "bindings": ["globals"],
    },
    # --- Bin sizes (SM-DES-004 §8.1) ---
    "D_BIN_W": {
        "value": 70.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §8.1",
        "bindings": ["globals"],
    },
    "D_BIN_D": {
        "value": 50.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §8.1",
        "bindings": ["globals"],
    },
    "D_BIN_H": {
        "value": 35.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §8.1",
        "bindings": ["globals"],
    },
    "D_BIN_WALL": {
        "value": 2.5,
        "unit": "mm",
        "lock_id": "SM-DES-004 §8.1",
        "bindings": ["globals"],
    },
    "D_BIN_PITCH_X": {
        "value": 75.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §8.1",
        "bindings": ["globals"],
    },
    "D_BIN_PITCH_Y": {
        "value": 55.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §8.1",
        "bindings": ["globals"],
    },
    # --- Bin-grid coordinates (SM-DES-006 §4.3, authoritative per §8) ---
    "D_GRID_X_OFFSET": {
        "value": 55.0,
        "unit": "mm",
        "lock_id": "SM-DES-006 §4.3",
        "bindings": ["globals"],
    },
    "D_GRID_Y_OFFSET": {
        "value": 30.0,
        "unit": "mm",
        "lock_id": "SM-DES-006 §4.3",
        "bindings": ["globals"],
    },
    "D_OVERFLOW_X": {
        "value": 280.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §8.1",
        "bindings": ["globals"],
    },
    "D_OVERFLOW_Y": {
        "value": 140.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §8.1",
        "bindings": ["globals"],
    },
}
