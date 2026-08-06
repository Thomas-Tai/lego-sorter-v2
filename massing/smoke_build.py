"""Local end-to-end smoke: build -> STEP + GLB -> R7 assert -> HTML viewer.

NOT a pytest (imports build123d, which CI lacks). Run in the build123d
venv, with C1 complete:

    python -m massing.smoke_build --out-dir build/massing
"""

from __future__ import annotations

import argparse
from pathlib import Path

from massing.geometry import export_glb, export_step, golden_dimension_span
from massing.report import format_report
from massing.run_checks import build_and_check
from massing.viewer import write_viewer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local massing STEP/GLB smoke build.")
    parser.add_argument("--out-dir", default="build/massing")
    args = parser.parse_args(argv)

    from hardware.interfaces import LEDGER
    from massing.envelopes import build_machine

    machine = build_machine(LEDGER)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    step_path, glb_path = out / "massing.step", out / "massing.glb"
    export_step(machine, step_path)
    export_glb(machine, glb_path)

    # R7 golden-dimension round-trip: the model's own X-span must survive
    # export to STEP and re-import within STEP precision. There is no
    # base-plate solid in the MVP assembly, so compare the model-computed
    # span to the re-imported span (isolates export precision, not a ledger
    # dim). The exported solids are exactly the station boxes + gantry travel.
    boxes = [s.box for s in machine.stations] + [machine.gantry_travel]
    model_span = max(b.x + b.dx for b in boxes) - min(b.x for b in boxes)
    span = golden_dimension_span(step_path)
    ok = abs(span - model_span) < 1.0
    print(
        f"R7 span check: model X-span={model_span:.2f} vs "
        f"STEP re-import X-span={span:.2f} -> {'PASS' if ok else 'FAIL'}"
    )

    write_viewer(
        glb_path.read_bytes(), out / "massing.html", Path("massing/viewer/vendor")
    )
    print(format_report(build_and_check(LEDGER)))
    print(f"Viewer: {out / 'massing.html'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
