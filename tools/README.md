# tools/ — interface ledger reconciliation

`hardware/interfaces.py` is the authoritative interface ledger (SM-DES-004
§2.2b in executable form). Two checks guard it:

- **Tier A (CI, automatic):** `tests/test_interface_ledger.py` validates the
  ledger's schema and AST purity on every push. Nothing to run by hand.
- **Tier B (local pre-flight, blocking):** reconciles the ledger against the
  non-git `Hardware/*_locals.txt` files. Run it at the start of every
  Stream-2 CAD session (SM-IMP-009 SETUP):

      python -m tools.reconcile_interfaces --hardware-root "<path-to>/Hardware"

  Exit 0 = clean, proceed. Exit 1 = drift found, session is BLOCKED: fix the
  drifted locals (edit the `*_locals.txt`, reload in SW) until it is green.
  To proceed anyway (rare), pass an audited override:

      python -m tools.reconcile_interfaces --hardware-root "<path>/Hardware" \
          --allow-drift --reason "why proceeding despite drift"

  The reason is echoed — paste it into the SM-IMP-009 session note.

Editing a value in `hardware/interfaces.py` is a design-lock change (spec §5):
it needs the same approval as amending SM-DES-004 §2.2b. A commit is not
authority to move a locked interface dim.

## `check_spec_leakage` — text2cad C3 Locked-Numbers Wall guard

When a part is modelled via the C2/C3 draft workflow (build123d draft →
ai-sw-bridge rebuild), the C3 wall says the draft informs topology and
proportion **only** — never a measurement. Every *size* dimension in the SW
spec must trace to a named local (`{"rhs": "\"VAR\""}` into a `*_locals.txt`).
This guard fails if any size field is a bare number instead:

    python -m tools.check_spec_leakage <spec.json> [<spec2.json> ...]

Exit 0 = clean (every size field cited to a local). Exit 1 = a raw literal
leaked into a size field (width/height/depth/depth2/diameter/radius/length/
major_radius/minor_radius/distance/spacing, or a `circles[].diameter`). Exit
2 = usage / unreadable spec.

**Coverage boundary (by design):** it does **not** check positional or angular
fields — `center` (x/y/z/u/v), `centerline`, `angle`/`angle_deg`, pattern
`direction`/`axis`. The bridge schema forbids `rhs` on those (literal-only),
so a placement number copied off the draft can still slip through and must be
checked by hand against the locals. This hardens the checkable half of the
wall; it is not a substitute for the C4 leakage audit. Runbook: step 5 of
`PM/4_Sorting Mechanism/03_Implementation/text2cad_Development_TODO.md`.
