"""Local end-to-end smoke: build -> STEP + GLB -> R7 assert -> HTML viewer.

NOT a pytest (imports build123d, which CI lacks). Run in the build123d
venv, with C1 complete:

    python -m massing.smoke_build --hardware-root "<path-to>/Hardware" \\
        --out-dir build/massing
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

    # R7 golden-dimension round-trip: base plate width must survive export.
    expected = float(LEDGER["D_BASE_W"]["value"])  # type: ignore[arg-type]
    span = golden_dimension_span(step_path)
    ok = abs(span - expected) < 1.0
    print(
        f"R7 span check: STEP X-span={span:.2f} vs D_BASE_W={expected:.2f} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )

    write_viewer(
        glb_path.read_bytes(), out / "massing.html", Path("massing/viewer/vendor")
    )
    print(format_report(build_and_check(LEDGER)))
    print(f"Viewer: {out / 'massing.html'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
