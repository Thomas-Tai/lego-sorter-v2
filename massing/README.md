# massing/ — full-system spatial pre-check (MVP)

Assembles the whole machine as rough axis-aligned box proxies from the C1
interface ledger and runs four static-geometry checks. Catches interference,
reach, alignment, and stack-up errors in seconds, before precise SolidWorks
modeling. Design spec: `PM/4_Sorting Mechanism/02_Design/2026-08-05-full-system-massing-model-design.md`.

## Two halves, by dependency weight

- **Checks (pure, CI-tested):** no geometry kernel. Run anytime:

      python -m massing.run_checks

  Prints a PASS/FAIL report with numbers and a scope line; exit 0 = all pass,
  1 = a check failed. Reads the real ledger (`hardware/interfaces.py`), so C1
  must be complete.

- **Demo (OCP, local-only, NOT in CI):** STEP + GLB + a self-contained HTML
  viewer, via `build123d`. Run in the build123d venv:

      python -m massing.smoke_build --out-dir build/massing

  Writes `massing.step`, `massing.glb`, `massing.html`, and runs the R7
  golden-dimension round-trip assertion (base-plate width must survive export).

## Scope — read this before trusting a green run

A green report means the *geometry is sane*, not that the machine works. It
does NOT verify: precise SW mates, structural deflection/creep, vibration,
motor torque/control, thermal/EMI, fastener/tool access. See spec §11.

## Boundaries

- Exact positions come from the ledger (authority). Rough box **sizes** live in
  `envelopes.py` (approximate, low-authority) and never feed back into the ledger.
- `build123d`/OCP is out of CI. No `tests/` file may import `massing.geometry`
  or `massing.smoke_build`.
- Ledger key names are reconciled in one place: `envelopes.KEYS`.
