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
