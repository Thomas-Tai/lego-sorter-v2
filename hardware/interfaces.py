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

INCOMPLETE: this is the audited seed (~12 constants). The remaining D_*
constants are transcribed under the one-time line-by-line audit (M4) --
see the implementation plan, Task 7. Do NOT guess values or lock IDs.
"""

STATIONS = {
    "globals": "00_Master_Assembly/Model/lego_sorter_globals.txt",
    "S1a": "S1a_VFeeder/Model/s1a_vfeeder_locals.txt",
    "S1b": "S1b_Conveyor/Model/s1b_conveyor_locals.txt",
}

LEDGER = {
    "D_BASE_W": {
        "value": 1000,
        "unit": "mm",
        "lock_id": "DL-02/D-10",
        "bindings": ["globals"],
    },
    "D_BASE_D": {
        "value": 450.0,
        "unit": "mm",
        "lock_id": "DL-02/D-10",
        "bindings": ["globals"],
    },
    "D_Z_BELT": {
        "value": 75.0,
        "unit": "mm",
        "lock_id": "IF-24",
        "bindings": ["globals", "S1a", "S1b"],
    },
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
    "D_HOPPER_OUTLET_W": {
        "value": 28.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §3",
        "bindings": ["globals", "S1a"],
    },
    "D_HOPPER_AIR_GAP": {
        "value": 8.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §3.3",
        "bindings": ["globals", "S1a"],
    },
    "D_Z_HOPPER_OUTLET": {
        "value": 93.0,
        "unit": "mm",
        "lock_id": "SM-DES-004 §2.3",
        "bindings": ["globals", "S1a"],
    },
    "D_HOPPER_INLET_DIA": {
        "value": 150,
        "unit": "mm",
        "lock_id": "SM-DES-004 §3.2",
        "bindings": ["globals"],
    },
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
}
