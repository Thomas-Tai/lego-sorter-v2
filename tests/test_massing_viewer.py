"""HTML viewer templating -- pure string assembly, no OCP (runs in CI)."""

import base64

from massing.viewer import render_html


def test_render_html_embeds_glb_and_up_axis() -> None:
    glb_b64 = base64.b64encode(b"dummy-glb-bytes").decode("ascii")
    html = render_html(
        glb_b64, three_js="/*three*/", loader_js="/*loader*/", up_axis="Z"
    )
    assert glb_b64 in html  # the model is embedded, not linked
    assert "/*three*/" in html  # three.js is inlined (self-contained)
    assert 'data-up-axis="Z"' in html  # the frame assertion is present
    assert "http" not in html.lower().split("data:")[0]  # no external CDN refs
