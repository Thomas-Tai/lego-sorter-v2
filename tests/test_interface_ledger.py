"""Tier A -- the interface ledger's own integrity (runs in CI).

Checks hardware/interfaces.py against the record schema and validates
its AST purity (F4). Does NOT touch the non-git Hardware/ path -- that
is Tier B (test_reconcile_interfaces.py + the live SETUP pre-flight).
"""

from hardware.interfaces import LEDGER, STATIONS
from tools.ledger_checks import validate_ledger_schema


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
    """D_Z_BELT is restated in globals + S1a + S1b (verified in the files)."""
    assert LEDGER["D_Z_BELT"]["bindings"] == ["globals", "S1a", "S1b"]
