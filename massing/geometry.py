"""OCP rendering of the massing proxies to STEP/GLB (local-only, NOT CI).

Imports build123d, which is absent from requirements*.txt and CI. No test
may import this module (it would break pytest collection in CI). Run it
via massing/smoke_build.py in the build123d venv.
"""

from __future__ import annotations

from pathlib import Path

import build123d as bd

from massing.model import Box, Machine


def _solid(box: Box) -> bd.Part:
    # build123d Box is centered; translate to our min-corner + size frame.
    part = bd.Box(box.dx, box.dy, box.dz)
    part = bd.Pos(box.x + box.dx / 2, box.y + box.dy / 2, box.z + box.dz / 2) * part
    return part


def build_assembly(machine: Machine) -> bd.Compound:
    solids = [_solid(s.box) for s in machine.stations]
    solids.append(_solid(machine.gantry_travel))
    return bd.Compound(children=solids)


def export_step(machine: Machine, path: str | Path) -> None:
    bd.export_step(build_assembly(machine), str(path))


def export_glb(machine: Machine, path: str | Path) -> None:
    # build123d exposes glTF export; confirm the exact symbol in the
    # installed version (export_gltf) and adjust if the API differs.
    bd.export_gltf(build_assembly(machine), str(path), binary=True)


def golden_dimension_span(step_path: str | Path) -> float:
    """Re-import the STEP and return the overall X-span (R7 round-trip)."""
    shape = bd.import_step(str(step_path))
    bb = shape.bounding_box()
    return bb.max.X - bb.min.X
