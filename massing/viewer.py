"""Self-contained HTML viewer (base64 GLB + vendored three.js).

Pure: no OCP. The GLB bytes are produced elsewhere (massing/geometry.py,
local-only); this module only packages them. The page declares its up
axis (data-up-axis) and draws a labelled origin triad so a mirror/axis
mismap (spec R7 / Idea H) is visible rather than silent.
"""

from __future__ import annotations

import base64
from pathlib import Path

_TEMPLATE = """<!doctype html>
<html data-up-axis="{up_axis}">
<head><meta charset="utf-8"><title>Massing Model</title>
<style>html,body{{margin:0;height:100%;background:#111}}#c{{width:100%;height:100%}}</style>
<script>{three_js}</script>
<script>{loader_js}</script>
</head>
<body>
<canvas id="c"></canvas>
<script>
// Machine frame is {up_axis}-up. If the model looks mirrored, the export
// axis map is wrong (spec R7) -- do not trust the render until fixed.
const GLB_B64 = "{glb_b64}";
// (viewer bootstrap: decode GLB_B64, add AxesHelper as the origin triad,
//  set camera up to {up_axis}, load with GLTFLoader) -- see README.
</script>
</body>
</html>
"""


def render_html(glb_b64: str, three_js: str, loader_js: str, up_axis: str = "Z") -> str:
    return _TEMPLATE.format(
        up_axis=up_axis, three_js=three_js, loader_js=loader_js, glb_b64=glb_b64
    )


def write_viewer(glb_bytes: bytes, out_html: Path, vendor_dir: Path) -> None:
    glb_b64 = base64.b64encode(glb_bytes).decode("ascii")
    three_js = (vendor_dir / "three.min.js").read_text(encoding="utf-8")
    loader_js = (vendor_dir / "GLTFLoader.js").read_text(encoding="utf-8")
    Path(out_html).write_text(
        render_html(glb_b64, three_js, loader_js), encoding="utf-8"
    )
