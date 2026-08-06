"""Interface-ledger checks: schema, AST purity, locals parser, reconcile.

Pure library, no side effects. tools/reconcile_interfaces.py wraps
reconcile() with a CLI. tests/ cover every function.
"""

from __future__ import annotations

ALLOWED_UNITS = {"mm", "deg", "mm^2", "count", "ratio"}

_REQUIRED_KEYS = {"value", "unit", "lock_id", "bindings"}
_PLACEHOLDER_LOCKS = {"", "todo", "tbd", "fixme", "?", "xxx"}


def validate_ledger_schema(ledger: dict, stations: dict) -> list[str]:
    """Return a list of schema-violation messages (empty means valid).

    Each ledger record must have exactly the required keys; value must be
    numeric; unit must be an allowed unit; lock_id must be a non-empty,
    non-placeholder string that references a numbered lock (contains a
    digit); bindings must be a list of known station tokens. A record
    with an empty bindings list is allowed (an unconsumed interface, F2)
    -- it is validated here but reported informational by reconcile().
    """
    errors: list[str] = []
    for name, rec in ledger.items():
        if not isinstance(rec, dict):
            errors.append(f"{name}: record is not a dict")
            continue
        missing = _REQUIRED_KEYS - rec.keys()
        extra = rec.keys() - _REQUIRED_KEYS
        if missing:
            errors.append(f"{name}: missing keys {sorted(missing)}")
        if extra:
            errors.append(f"{name}: unexpected keys {sorted(extra)}")
        if missing:
            continue
        if not isinstance(rec["value"], (int, float)) or isinstance(rec["value"], bool):
            errors.append(f"{name}: value must be int or float")
        if rec["unit"] not in ALLOWED_UNITS:
            errors.append(
                f"{name}: unit {rec['unit']!r} not in {sorted(ALLOWED_UNITS)}"
            )
        lock = rec["lock_id"]
        if (
            not isinstance(lock, str)
            or lock.strip().lower() in _PLACEHOLDER_LOCKS
            or not any(ch.isdigit() for ch in lock)
        ):
            errors.append(f"{name}: lock_id {lock!r} is empty/placeholder/unnumbered")
        bindings = rec["bindings"]
        if not isinstance(bindings, list):
            errors.append(f"{name}: bindings must be a list")
            continue
        for token in bindings:
            if token not in stations:
                errors.append(f"{name}: binding {token!r} not a known station")
    return errors
