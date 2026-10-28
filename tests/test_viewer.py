"""Exercise viewer gestures with real Three.js math and a headless renderer."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from eyehand import browser_cursor


def test_viewer_rotation_zoom_and_release(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for viewer interaction check')
    viewer = Path(browser_cursor.__file__).parent / 'viewer'
    check = r'''
import * as Math3D from 'THREE_URL';
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const elements = {};
const el = id => elements[id] ||= {hidden: false, clientWidth: 800, clientHeight: 600,
  getBoundingClientRect() {return {left: 0, top: 0, width: 800, height: 600};},
  listeners: {}, addEventListener(type, fn) { this.listeners[type] = fn; }, focus() {}, setPointerCapture() {}};
let pivot, camera;
const THREE = {...Math3D,
  WebGLRenderer: class {setPixelRatio() {} setSize() {} render() {}},
  Group: class extends Math3D.Group {constructor() {super(); pivot = this;}},
  PerspectiveCamera: class extends Math3D.PerspectiveCamera {constructor(...args) {super(...args); camera = this;}}
};
const window = {addEventListener() {}};
const geometry = new Math3D.BoxGeometry(4, 2, 1);
geometry.translate(20, 30, 40);
const STLLoader = class {load(url, success) {success(geometry);}};
vm.runInNewContext(fs.readFileSync('VIEWER_PATH', 'utf8').replace(/^import .*;$/gm, ''), {
  THREE, STLLoader, window, document: {querySelector: el, hidden: false, hasFocus: () => true, addEventListener() {}},
  innerWidth: 800, innerHeight: 600, devicePixelRatio: 1, requestAnimationFrame() {}, console
});
const controls = window.propellerViewer;
assert.ok(controls);
geometry.computeBoundingBox();
assert.ok(geometry.boundingBox.getCenter(new Math3D.Vector3()).length() < 1e-6);
assert.equal(pivot.position.length(), 0);
const before = pivot.quaternion.clone();
assert.equal(controls.begin('R', {valid: true, x: .5, y: .5, hand_mm: [0, 0, -300]}), true);
controls.move('R', {valid: true, x: .6, y: .55, hand_mm: [10, 15, -280]});
assert.ok(before.angleTo(pivot.quaternion) > 0.01);
assert.ok(camera.position.z > 2.7); // Forward hand movement reverses the previous zoom direction.
const radius = 600 / (2 * Math.tan(Math3D.MathUtils.degToRad(camera.fov / 2)) * 2.7);
const expected = new Math3D.Vector3(80 / radius, -30 / radius, Math.sqrt(1 - (80 / radius) ** 2 - (30 / radius) ** 2));
const rotation = pivot.quaternion.clone().multiply(before.clone().invert());
assert.ok(new Math3D.Vector3(0, 0, 1).applyQuaternion(rotation).distanceTo(expected) < 1e-6);
assert.equal(el('#viewer-status').textContent, '');
controls.move('R', {valid: true, x: .6, y: .55, hand_mm: [10, 15, -300]});
assert.ok(Math.abs(camera.position.z - 2.7) < 1e-6); // Moving back restores the original zoom.
const stopped = pivot.quaternion.clone(), distance = camera.position.z;
controls.move('R', {valid: false});
assert.equal(camera.position.z, distance);
controls.end('R');
controls.move('R', {valid: true, x: .7, y: .6, hand_mm: [20, 30, -260]});
assert.ok(stopped.angleTo(pivot.quaternion) < 1e-7);
el('#reset-view').listeners.click();
assert.equal(camera.position.z, 2.7);
const canvas = el('#viewer-canvas');
canvas.listeners.pointerdown({button: 0, pointerId: 1, clientX: 400, clientY: 300, preventDefault() {}});
canvas.listeners.pointermove({pointerId: 1, clientX: 480, clientY: 330});
assert.ok(before.angleTo(pivot.quaternion) > 0.01);
const mouseRotation = pivot.quaternion.clone().multiply(before.clone().invert());
assert.ok(new Math3D.Vector3(0, 0, 1).applyQuaternion(mouseRotation).distanceTo(expected) < 1e-6);
canvas.listeners.pointerup();
const released = pivot.quaternion.clone();
canvas.listeners.pointermove({pointerId: 1, clientX: 180, clientY: 90});
assert.ok(released.angleTo(pivot.quaternion) < 1e-7);
el('#viewer-panel').hidden = true;
assert.equal(controls.begin('R', {valid: true, x: .5, y: .5, hand_mm: [0, 0, -300]}), false);
'''
    check = check.replace('THREE_URL', (viewer / 'three.module.js').as_uri()).replace('VIEWER_PATH', str(viewer / 'viewer.js'))
    script = tmp_path / 'check.mjs'
    script.write_text(check)
    result = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_supplied_mesh_is_complete_and_finite():
    mesh = Path(browser_cursor.__file__).parent / 'viewer' / 'propeller.stl'
    data = mesh.read_bytes()
    count = int.from_bytes(data[80:84], 'little')
    assert len(data) == 84 + count * 50
    triangles = np.frombuffer(data, offset=84, dtype=[('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    assert count > 0
    assert np.isfinite(triangles['vertices']).all()
