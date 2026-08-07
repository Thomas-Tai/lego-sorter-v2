"""Self-contained HTML viewer (base64 GLB + vendored three.js).

Pure: no OCP. The GLB bytes are produced elsewhere (massing/geometry.py,
local-only); this module only packages them. The page declares its up
axis (data-up-axis) and draws a labelled origin triad so a mirror/axis
mismap (spec R7 / Idea H) is visible rather than silent.

Templating uses plain token replacement (not str.format) so the inlined
three.js/GLTFLoader.js/bootstrap JS can contain literal `{`/`}` without
needing to be brace-escaped.
"""

from __future__ import annotations

import base64
from pathlib import Path

_BOOTSTRAP_JS = """
(function () {
  var axisMap = {
    X: new THREE.Vector3(1, 0, 0),
    Y: new THREE.Vector3(0, 1, 0),
    Z: new THREE.Vector3(0, 0, 1)
  };
  var upVec = axisMap[UP_AXIS] || axisMap.Z;

  var canvas = document.getElementById("c");
  var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);

  var scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111111);

  var camera = new THREE.PerspectiveCamera(
    50,
    (canvas.clientWidth || 1) / (canvas.clientHeight || 1),
    0.1,
    1e7
  );
  camera.up.copy(upVec);

  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  var dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(300, -300, 400);
  scene.add(dirLight);

  // Origin triad (R7 mirror-guard): red=+X (toward bins), green=+Y,
  // blue=+Z (up). If the model looks mirrored against these axes, the
  // export axis map is wrong -- do not trust the render until fixed.
  var axes = new THREE.AxesHelper(500);
  scene.add(axes);

  function resize() {
    var w = canvas.clientWidth || window.innerWidth;
    var h = canvas.clientHeight || window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / (h || 1);
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);
  resize();

  // Minimal drag-to-orbit / wheel-to-zoom control that works for any
  // up axis: rotate into a canonical Y-up frame, do standard spherical
  // orbit math there, then rotate the result back out. (Same trick
  // three.js's own OrbitControls uses for non-Y-up scenes -- vendored
  // here directly so we don't need to ship OrbitControls.js too.)
  var quat = new THREE.Quaternion().setFromUnitVectors(upVec, new THREE.Vector3(0, 1, 0));
  var quatInverse = quat.clone().invert();
  var target = new THREE.Vector3(0, 0, 0);
  var spherical = new THREE.Spherical(1000, Math.PI / 3, Math.PI / 4);

  function applyCamera() {
    var offset = new THREE.Vector3().setFromSpherical(spherical);
    offset.applyQuaternion(quatInverse);
    camera.position.copy(target).add(offset);
    camera.lookAt(target);
  }
  applyCamera();

  var dragging = false;
  var lastX = 0;
  var lastY = 0;
  canvas.addEventListener("pointerdown", function (e) {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
  });
  window.addEventListener("pointerup", function () {
    dragging = false;
  });
  window.addEventListener("pointermove", function (e) {
    if (!dragging) return;
    var dx = e.clientX - lastX;
    var dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    spherical.theta -= dx * 0.005;
    spherical.phi = Math.min(Math.max(spherical.phi - dy * 0.005, 0.001), Math.PI - 0.001);
    applyCamera();
  });
  canvas.addEventListener(
    "wheel",
    function (e) {
      e.preventDefault();
      spherical.radius = Math.max(10, spherical.radius * (1 + e.deltaY * 0.001));
      applyCamera();
    },
    { passive: false }
  );

  // Decode the embedded GLB (base64 -> ArrayBuffer) and load it via
  // GLTFLoader.parse -- no network fetch, fully self-contained.
  function base64ToArrayBuffer(b64) {
    var bin = atob(b64);
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes.buffer;
  }

  var loader = new THREE.GLTFLoader();
  loader.parse(
    base64ToArrayBuffer(GLB_B64),
    "",
    function (gltf) {
      scene.add(gltf.scene);

      var box = new THREE.Box3().setFromObject(gltf.scene);
      if (!box.isEmpty()) {
        var center = box.getCenter(new THREE.Vector3());
        var size = box.getSize(new THREE.Vector3()).length();
        target.copy(center);
        spherical.radius = Math.max(size * 1.5, 100);
        applyCamera();

        scene.remove(axes);
        axes = new THREE.AxesHelper(Math.max(size * 0.5, 50));
        scene.add(axes);
      }
    },
    function (err) {
      console.error("GLB load failed:", err);
    }
  );

  function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
  }
  animate();
})();
"""

_TEMPLATE = """<!doctype html>
<html data-up-axis="__UP_AXIS__">
<head><meta charset="utf-8"><title>Massing Model</title>
<style>html,body{margin:0;height:100%;background:#111}#c{width:100%;height:100%;display:block}</style>
<script>__THREE_JS__</script>
<script>__LOADER_JS__</script>
</head>
<body>
<canvas id="c"></canvas>
<script>
// Machine frame is __UP_AXIS__-up. If the model looks mirrored, the export
// axis map is wrong (spec R7) -- do not trust the render until fixed.
const GLB_B64 = "__GLB_B64__";
const UP_AXIS = "__UP_AXIS__";
__BOOTSTRAP_JS__
</script>
</body>
</html>
"""


def render_html(glb_b64: str, three_js: str, loader_js: str, up_axis: str = "Z") -> str:
    html = _TEMPLATE
    html = html.replace("__UP_AXIS__", up_axis)
    html = html.replace("__BOOTSTRAP_JS__", _BOOTSTRAP_JS)
    html = html.replace("__THREE_JS__", three_js)
    html = html.replace("__LOADER_JS__", loader_js)
    html = html.replace("__GLB_B64__", glb_b64)
    return html


def write_viewer(glb_bytes: bytes, out_html: Path, vendor_dir: Path) -> None:
    glb_b64 = base64.b64encode(glb_bytes).decode("ascii")
    three_js = (vendor_dir / "three.min.js").read_text(encoding="utf-8")
    loader_js = (vendor_dir / "GLTFLoader.js").read_text(encoding="utf-8")
    Path(out_html).write_text(
        render_html(glb_b64, three_js, loader_js), encoding="utf-8"
    )
