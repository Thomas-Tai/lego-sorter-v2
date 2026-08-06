"""Interface-ledger checks: schema, AST purity, locals parser, reconcile.

Pure library, no side effects. tools/reconcile_interfaces.py wraps
reconcile() with a CLI. tests/ cover every function.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

ALLOWED_UNITS = {"mm", "deg", "mm^2", "count", "ratio"}

_REQUIRED_KEYS = {"value", "unit", "lock_id", "bindings"}
_PLACEHOLDER_LOCKS = {"", "todo", "tbd", "fixme", "?", "xxx"}

# Matches exactly `"D_NAME" = <number>` (optionally negative / decimal).
# The opening quote must be immediately followed by `D_`, so `_D_*` and
# `"// ..."` never match. An expression RHS (a quote, letter, or paren)
# fails the numeric group, so only restated literals are captured.
_INTERFACE_LINE = re.compile(r'^\s*"(D_[A-Za-z0-9_]+)"\s*=\s*(-?\d+(?:\.\d+)?)\s*$')


def parse_interface_lines(path: str | Path) -> dict[str, Decimal]:
    """Return {D_name: Decimal} for restated interface literals in a file.

    Scope (spec §5 / B2): only lines of the form `"D_NAME" = <number>`.
    Derived Sxx_ expressions, _D_ semi-privates, and "// ..." comment
    pseudo-entries are skipped. Values are Decimals for exact compare
    (F3). The file is read as utf-8-sig so a leading BOM (SolidWorks/
    Windows exports commonly emit one) does not break the anchor on the
    first line, with errors="replace" so encoding mojibake in comments
    cannot crash the parse (B13).
    """
    result: dict[str, Decimal] = {}
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    for line in text.splitlines():
        m = _INTERFACE_LINE.match(line)
        if m:
            result[m.group(1)] = Decimal(m.group(2))
    return result


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


def assert_pure_data(source_path: str | Path) -> list[str]:
    """Return purity violations for a data-only module (empty = pure).

    Allowed top-level statements: a string docstring (Expr wrapping a str
    constant) and assignments (Assign / AnnAssign) whose right-hand side
    is ast.literal_eval-able (numbers, strings, bools, None, and dict/
    list/tuple/set of those). Everything else -- imports, calls, BinOp,
    Name references, def/class -- is a violation. This is what keeps the
    authority trivially diffable and free of hidden derived values (F4).
    """
    errors: list[str] = []
    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue  # module docstring
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            rhs = node.value
            if rhs is None:
                errors.append(f"line {node.lineno}: annotation without a value")
                continue
            try:
                ast.literal_eval(rhs)
            except (ValueError, SyntaxError, TypeError):
                errors.append(f"line {node.lineno}: non-literal right-hand side")
            continue
        errors.append(f"line {node.lineno}: disallowed statement {type(node).__name__}")
    return errors


@dataclass
class Drift:
    name: str
    location: str
    expected: str
    found: str | None  # None == name absent at this location


@dataclass
class ReconcileReport:
    drifts: list[Drift] = field(default_factory=list)
    unconsumed: list[str] = field(default_factory=list)
    strays: list[tuple[str, str]] = field(default_factory=list)

    @property
    def has_blocking(self) -> bool:
        return bool(self.drifts)


def reconcile(
    ledger: dict, stations: dict, hardware_root: str | Path
) -> ReconcileReport:
    """Reconcile every restated D_* number against the ledger authority.

    hardware_root is the machine-specific Hardware/ path (config, not git).
    A station file that does not exist parses as empty, so declared
    bindings to it surface as declared-but-missing drifts.
    """
    root = Path(hardware_root)
    parsed: dict[str, dict[str, Decimal]] = {}
    for token, rel in stations.items():
        p = root / rel
        parsed[token] = parse_interface_lines(p) if p.exists() else {}

    report = ReconcileReport()
    for name, rec in ledger.items():
        bindings = rec["bindings"]
        if not bindings:
            report.unconsumed.append(name)
            continue
        expected = Decimal(str(rec["value"]))
        for token in bindings:
            found = parsed.get(token, {}).get(name)
            if found is None:
                report.drifts.append(
                    Drift(
                        name=name,
                        location=token,
                        expected=str(rec["value"]),
                        found=None,
                    )
                )
            elif found != expected:
                report.drifts.append(
                    Drift(
                        name=name,
                        location=token,
                        expected=str(rec["value"]),
                        found=format(found, "f"),
                    )
                )

    ledger_names = set(ledger)
    for token, values in parsed.items():
        for name in values:
            if name not in ledger_names:
                report.strays.append((token, name))
    return report
