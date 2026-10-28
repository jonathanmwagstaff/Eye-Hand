import numpy as np
import pytest

from eyehand import browser_cursor, cursor
from eyehand.intrinsics import Intrinsics
from eyehand.landmarks import Landmarks, LEFT_IRIS, RIGHT_IRIS


K = np.array([[500.0, 0, 640], [0, 500.0, 480], [0, 0, 1]])
INTR_L = Intrinsics("L", (1280, 960), K, np.zeros(5))
INTR_F = Intrinsics("F", (1280, 960), K, np.zeros(5))
T_F_L = np.eye(4)
T_F_L[0, 3] = -100.0
T_S_L = np.diag([-1.0, 1.0, -1.0, 1.0])
SCREEN_MM = (344.0, 194.0)


def test_browser_page_is_utf8():
    assert "Eye-Hand browser cursor" in browser_cursor.PAGE.decode()
    assert b'id="entry-tab"' not in browser_cursor.PAGE
    assert b'id="entry-panel"' not in browser_cursor.PAGE
    assert b'/viewer/entry.js' not in browser_cursor.PAGE


def test_browser_holds_each_cursor_for_three_seconds_without_delaying_new_positions():
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to exercise the browser script")
    check = r"""
const assert = require('node:assert/strict'), vm = require('node:vm'), fs = require('node:fs');
const elements = {};
function element(id) {
  return elements[id] ||= {id: id.slice(1), style: {}, classes: new Set(), attrs: {}, offsetWidth: 144, offsetHeight: 144, clientLeft: 4, clientTop: 4,
    prepend() {},
    getBoundingClientRect() { return id === '#playground' ? {left: 20, top: 150, width: 900, height: 600} : {left: 100, top: 250, right: 244, bottom: 394, width: 144, height: 144}; },
    getTotalLength() { return 2 * Math.PI * 44; },
    getPointAtLength(distance) { return {x: 50 + 44 * Math.cos(distance / 44), y: 50 + 44 * Math.sin(distance / 44)}; },
    isPointInFill(point) { return Math.hypot(point.x - 50, point.y - 50) <= 44; },
    setPointerCapture() {},
    classList: {toggle(name, on) { on ? elements[id].classes.add(name) : elements[id].classes.delete(name); }},
    setAttribute(name, value) { this.attrs[name] = value; }, listeners: {},
    addEventListener(name, listener) { this.listeners[name] = listener; },
    querySelector() { return element(id + '-label'); }};
}
let now = 0, frame, stream;
const clicks = [], events = [], listeners = {}, windowListeners = {};
const pageTarget = {dispatchEvent(event) {
  events.push(event);
  if (event.type === 'click' || event.type === 'contextmenu') clicks.push(event);
}};
let hitShape = null;
const backendSettings = {eye: 'R', primary_hand: 'R', thumb_weight: .75}, savedWeights = [];
let focused = true;
vm.runInNewContext(fs.readFileSync(0, 'utf8'), {
  document: {querySelector: element, hidden: false, hasFocus: () => focused,
    createElement() {
      const el = element('#created-' + Object.keys(elements).length);
      Object.defineProperty(el, 'id', {set(value) { elements['#' + value] = el; }});
      return el;
    },
    addEventListener: (name, listener) => { listeners[name] = listener; },
    elementFromPoint: () => hitShape || pageTarget},
  window: {addEventListener: (name, listener) => { windowListeners[name] = listener; }}, MouseEvent: class { constructor(type, options) { this.type = type; Object.assign(this, options); } },
  innerWidth: 1000, innerHeight: 800,
  performance: {now: () => now}, requestAnimationFrame: callback => { frame = callback; },
  SharedWorker: class { constructor() { this.port = stream = {start() {}, postMessage() {}}; } },
  fetch: async (url, options = {}) => {
    if (options.method === 'POST') {
      Object.assign(backendSettings, JSON.parse(options.body));
      savedWeights.push(backendSettings.thumb_weight);
    }
    return {ok: true, json: async () => ({...backendSettings})};
  }
});
function send(pointers, primary_hand = 'R') { stream.onmessage({data: {primary_hand, pointers}}); frame(); }
frame();
assert.equal(element('#dot-R').style.display, 'none');
send({R: {valid: true, x: .2, y: .5}, L: {valid: true, x: .7, y: .5}});
assert.equal(element('#dot-R').style.transform, 'translate(200px, 400px)');
now = 100;
send({R: {valid: false}, L: {valid: true, x: .8, y: .5}});
assert.equal(element('#dot-R').style.display, 'block');
assert.equal(element('#dot-R').style.transform, 'translate(200px, 400px)');
assert.equal(element('#dot-L').style.transform, 'translate(800px, 400px)');
now = 2999;
send({R: {valid: false}, L: {valid: false}});
assert.equal(element('#dot-R').style.display, 'block');
now = 3000; frame();
assert.equal(element('#dot-R').style.display, 'none');
assert.equal(element('#dot-L').style.display, 'block');
now = 3100; frame();
assert.equal(element('#dot-L').style.display, 'none');
now = 3101;
send({R: {valid: true, x: .4, y: .6}});
assert.equal(element('#dot-R').style.display, 'block');
assert.equal(element('#dot-R').style.transform, 'translate(400px, 480px)');
now = 6101; frame();
assert.equal(element('#dot-R').style.display, 'none');
now = 6200;
const open = {valid: true, x: .3, y: .4, pinch_ratio: 1.0};
const closed = {...open, pinch_ratio: .28};
send({R: closed});
assert.equal(events.length, 0); // Starting already pinched must rearm first.
send({R: open, L: open});
send({R: closed, L: open});
assert.equal(events.at(-1).type, 'mousedown');
assert.equal(events.at(-1).buttons, 1);
assert.equal(clicks.length, 0); // Clicking completes on release, not on press.
assert.equal(element('#dot-R').classes.has('clicking'), true);
for (const ratio of [.35, .45, .55]) {
  send({R: {...closed, pinch_ratio: ratio}, L: open});
  assert.equal(element('#dot-R').classes.has('clicking'), true);
}
send({R: {valid: false}, L: open});
assert.equal(events.length, 1); // Missing tracking cannot generate mouseup.
send({R: closed, L: open});
assert.equal(events.length, 1);
send({R: open, L: open});
assert.equal(events.at(-2).type, 'mouseup');
assert.equal(clicks.length, 1);
assert.equal(clicks[0].type, 'click');
assert.equal(element('#dot-R').classes.has('clicking'), false);
send({R: open, L: closed});
assert.equal(events.at(-1).type, 'mousedown');
assert.equal(events.at(-1).button, 2);
assert.equal(clicks.length, 1);
send({R: open, L: open});
assert.equal(clicks.at(-1).type, 'contextmenu');
send({L: open}, 'L');
send({L: closed}, 'L');
send({L: open}, 'L');
assert.equal(clicks.at(-1).type, 'click');
assert.equal(clicks.length, 3);
send({R: open});
focused = false;
send({R: closed});
assert.equal(clicks.length, 3);
focused = true;
send({R: open});
now += 3000;
send({R: {valid: false, pinch_ratio: .2}});
assert.equal(clicks.length, 3);
function space(type, extra = {}) {
  const event = {code: 'Space', repeat: false, target: {closest: () => false},
    prevented: false, preventDefault() { this.prevented = true; }, ...extra};
  listeners[type](event); frame(); return event;
}
send({R: open, L: open});
assert.equal(space('keydown').prevented, true);
assert.equal(events.at(-1).type, 'mousedown');
assert.equal(clicks.length, 3);
assert.equal(element('#dot-R').classes.has('clicking'), true);
const heldEvents = events.length;
space('keydown', {repeat: true});
assert.equal(events.length, heldEvents);
send({L: open, R: {valid: false}});
space('keyup');
assert.equal(clicks.length, 4);
assert.equal(element('#dot-R').classes.has('clicking'), false);
space('keydown');
assert.equal(events.at(-1).button, 2);
assert.equal(element('#dot-L').classes.has('clicking'), true);
windowListeners.blur(); frame();
assert.equal(events.at(-1).type, 'mouseup');
assert.equal(clicks.length, 4); // Cancelling a held press must not activate a control.
assert.equal(element('#dot-L').classes.has('clicking'), false);
send({R: open, L: open}, 'L');
space('keydown'); space('keyup');
assert.equal(clicks.at(-1).type, 'click');
assert.equal(clicks.length, 5);
assert.equal(space('keydown', {target: {closest: () => true}}).prevented, false);
assert.equal(clicks.length, 5);
now += 301;
space('keydown'); space('keyup');
assert.equal(clicks.length, 5); // A frozen claw without current hand tracking cannot start a press.
focused = false;
send({L: open}, 'L');
space('keydown');
assert.equal(clicks.length, 5);
focused = true;

// Space and the physical pinch share a single held press.
send({R: open});
space('keydown');
const pressCount = events.length;
send({R: closed});
space('keyup');
assert.equal(events.length, pressCount); // Releasing Space cannot drop an active physical pinch.
send({R: open});
assert.equal(events.at(-2).type, 'mouseup');
assert.equal(events.at(-1).type, 'click');

// Mouse and claw share movement constraints and release behavior.
const circle = element('#shape-circle');
circle.listeners.pointerdown({button: 0, clientX: 120, clientY: 270, pointerId: 1, preventDefault() {}});
circle.listeners.pointermove({clientX: 2000, clientY: -100});
assert.equal(circle.style.left, '756px');
assert.equal(circle.style.top, '0px');
circle.listeners.pointerup();
assert.equal(circle.classes.has('grabbed'), false);
hitShape = {closest: () => circle, dispatchEvent() {}};
send({R: open});
send({R: closed});
assert.equal(circle.classes.has('grabbed'), true);
send({R: {...closed, x: .5, y: .6}});
assert.equal(circle.style.left, '280px');
assert.equal(circle.style.top, '260px');
send({R: open});
assert.equal(circle.classes.has('grabbed'), false);
send({R: closed});
now += 301; frame();
assert.equal(circle.classes.has('grabbed'), true); // Missing samples pause, rather than release, the drag.
const heldLeft = circle.style.left, heldTop = circle.style.top;
stream.onmessage({data: {valid: false, reason: 'Waiting for cameras'}}); frame();
assert.equal(circle.classes.has('grabbed'), true);
assert.equal(circle.style.left, heldLeft);
assert.equal(circle.style.top, heldTop);
send({R: {...closed, x: .55, y: .65, pinch_ratio: .5}});
assert.equal(circle.classes.has('grabbed'), true);
assert.notEqual(circle.style.left, heldLeft);
send({R: open});
assert.equal(circle.classes.has('grabbed'), false); // A clearly open hand releases it.

// Every movable object supports mouse dragging, including takeover from a held claw.
send({R: closed});
assert.equal(circle.classes.has('grabbed'), true);
for (const name of ['circle', 'triangle', 'square', 'card', 'folder']) {
  const object = element(`#shape-${name}`);
  object.listeners.pointerdown({button: 0, clientX: 120, clientY: 270, pointerId: 3, preventDefault() {}});
  assert.equal(object.classes.has('grabbed'), true);
  object.listeners.pointermove({clientX: 200, clientY: 350});
  assert.equal(object.style.left, '160px');
  assert.equal(object.style.top, '180px');
  object.listeners.pointerup();
  assert.equal(object.classes.has('grabbed'), false);
}
send({R: open});

const target = element('#target-circle');
target.getBoundingClientRect = () => ({left: 300, top: 300, right: 516, bottom: 516, width: 216, height: 216});
let bounds = {left: 336, top: 336, right: 480, bottom: 480, width: 144, height: 144};
circle.getBoundingClientRect = () => bounds;
circle.listeners.keydown({key: 'ArrowRight', preventDefault() {}});
assert.equal(target.classes.has('complete'), true);
bounds = {left: 303, top: 336, right: 447, bottom: 480, width: 144, height: 144};
circle.listeners.keydown({key: 'ArrowRight', preventDefault() {}});
assert.equal(target.classes.has('complete'), false); // Even partial contact with the outline fails.
bounds = {left: 304, top: 304, right: 448, bottom: 448, width: 144, height: 144};
circle.listeners.keydown({key: 'ArrowRight', preventDefault() {}});
assert.equal(target.classes.has('complete'), false); // Inside the box, outside the circular outline.
bounds = {left: 347, top: 347, right: 491, bottom: 491, width: 144, height: 144};
circle.listeners.keydown({key: 'ArrowRight', preventDefault() {}});
assert.equal(target.classes.has('complete'), true);
// Targets only change color: dragging through them never snaps, drops, or locks the shape.
bounds = {left: 336, top: 336, right: 480, bottom: 480, width: 144, height: 144};
circle.listeners.pointerdown({button: 0, clientX: 350, clientY: 350, pointerId: 2, preventDefault() {}});
circle.listeners.pointermove({clientX: 420, clientY: 420});
assert.equal(target.classes.has('complete'), true);
assert.equal(circle.classes.has('grabbed'), true);
assert.equal(circle.style.left, '386px');
bounds = {left: 600, top: 300, right: 744, bottom: 444, width: 144, height: 144};
circle.listeners.pointermove({clientX: 700, clientY: 500});
assert.equal(target.classes.has('complete'), false);
assert.equal(circle.classes.has('grabbed'), true);
assert.equal(circle.style.left, '666px');
circle.listeners.pointerup();
for (const [name, label] of [['square', 'Upload photos drop zone'], ['card', 'Ace of Spades target'], ['folder', 'Recycle Bin target']]) {
  const object = element(`#shape-${name}`), destination = element(`#target-${name}`);
  assert.equal(destination.attrs['aria-label'], label);
  destination.getBoundingClientRect = () => ({left: 300, top: 300, right: 516, bottom: 516, width: 216, height: 216});
  object.getBoundingClientRect = () => ({left: 336, top: 336, right: 480, bottom: 480, width: 144, height: 144});
  object.listeners.keydown({key: 'ArrowRight', preventDefault() {}});
  assert.equal(destination.classes.has('complete'), true);
  object.getBoundingClientRect = () => ({left: 290, top: 336, right: 434, bottom: 480, width: 144, height: 144});
  object.listeners.keydown({key: 'ArrowRight', preventDefault() {}});
  assert.equal(destination.classes.has('complete'), false);
}
element('#viewer-tab').listeners.click();
assert.equal(element('#drag-panel').hidden, true);
assert.equal(element('#viewer-tab').attrs['aria-selected'], 'true');
element('#drag-tab').listeners.click();
assert.equal(element('#drag-panel').hidden, false);
assert.equal(element('#viewer-panel').hidden, true);

setImmediate(async () => {
  for (const value of ['0', '40', '100']) {
    element('#thumb-weight').value = value;
    element('#thumb-weight').listeners.input();
  }
  await new Promise(setImmediate);
  assert.equal(backendSettings.thumb_weight, 1);
  assert.equal(savedWeights.at(-1), 1); // Dragging cannot let an older save win.
  assert.equal(element('#ratio-value').textContent, '100% thumb');
  assert.equal(element('#applied-ratio').textContent, 'Applied: 100% thumb');
  element('#thumb-weight').value = '0';
  element('#thumb-weight').listeners.input();
  await new Promise(setImmediate);
  assert.equal(backendSettings.thumb_weight, 0);
  assert.equal(element('#ratio-value').textContent, '0% thumb');
  hitShape = null;
  element('#release-sensitivity').value = '0';
  element('#release-sensitivity').listeners.input();
  await new Promise(setImmediate);
  assert.equal(backendSettings.release_sensitivity, 0);
  send({R: open}); send({R: closed});
  send({R: {...closed, pinch_ratio: .6}});
  assert.equal(element('#dot-R').classes.has('clicking'), true);
  element('#release-sensitivity').value = '100';
  element('#release-sensitivity').listeners.input();
  await new Promise(setImmediate);
  assert.equal(backendSettings.release_sensitivity, 1);
  assert.equal(element('#release-value').textContent, '100%');
  send({R: {...closed, pinch_ratio: .4}});
  assert.equal(element('#dot-R').classes.has('clicking'), false);
  assert.equal(events.at(-2).type, 'mouseup');
  const views = (L, F) => ({...open, pinch_views: {L, F}, pinch_ratio: Math.min(L, F)});
  send({R: views(1, 1)});
  send({R: views(.1, .1)});
  assert.equal(events.at(-1).type, 'mousedown');
  send({R: views(.1, 1)}); // F alone sees an open hand.
  assert.equal(events.at(-2).type, 'mouseup');
  assert.equal(element('#dot-R').classes.has('clicking'), false);
  const releasedEvents = events.length;
  for (let i = 0; i < 4; i++) send({R: views(.1, 1)});
  assert.equal(events.length, releasedEvents); // Disagreement must not repeatedly press/release.
  send({R: views(.1, .1)});
  assert.equal(events.at(-1).type, 'mousedown');
  send({R: views(1, .1)}); // L alone can release too.
  assert.equal(events.at(-2).type, 'mouseup');
  assert.equal(element('#dot-R').classes.has('clicking'), false);
});

"""
    script = browser_cursor.PAGE.decode().split("<script>")[1].split("</script>")[0]
    subprocess.run([node, "-e", check], input=script, text=True, check=True, timeout=10,
                   capture_output=True)


def test_pairing_prefers_fresh_aligned_frames():
    old_L, old_F = type("L", (), {"t_ms": 0.0})(), type("L", (), {"t_ms": 0.0})()
    new_L, new_F = type("L", (), {"t_ms": 100.0})(), type("L", (), {"t_ms": 115.0})()
    assert browser_cursor._best_pair([old_L, new_L], [old_F, new_F]) == (new_L, new_F)


def test_tracking_worker_shares_one_stream_across_tabs():
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to exercise the tracking worker")
    check = r"""
const assert = require('node:assert/strict'), vm = require('node:vm'), fs = require('node:fs');
const streams = [], scope = {EventSource: class {
  constructor(url) { assert.equal(url, '/stream'); streams.push(this); }
  close() { this.closed = true; }
}};
vm.runInNewContext(fs.readFileSync(0, 'utf8'), scope);
const ports = Array.from({length: 8}, () => ({messages: [], start() {}, close() {},
  postMessage(value) { this.messages.push(value); }}));
for (const port of ports) scope.onconnect({ports: [port]});
assert.equal(streams.length, 1);
streams[0].onmessage({data: JSON.stringify({seq: 1})});
for (const port of ports) assert.equal(port.messages[0].seq, 1);
for (const port of ports.slice(1)) port.onmessage({data: 'disconnect'});
assert.equal(streams[0].closed, undefined);
ports[0].onmessage({data: 'disconnect'});
assert.equal(streams[0].closed, true);
scope.onconnect({ports: [ports[0]]});
assert.equal(streams.length, 2);
"""
    subprocess.run([node, "-e", check], input=browser_cursor.STREAM_WORKER.decode(),
                   text=True, check=True, timeout=10, capture_output=True)


def project(points, transform=np.eye(4)):
    camera = (transform[:3, :3] @ points.T).T + transform[:3, 3]
    return camera[:, :2] / camera[:, 2:] * 500 + [640, 480]


def observations():
    # Eye and fingertip in screen coordinates. Their ray hits the screen at (175, 65) mm.
    points_S = np.array([[100.0, 50.0, -600.0], [150.0, 60.0, -200.0]])
    points_L = (T_S_L[:3, :3].T @ points_S.T).T
    return project(points_L), project(points_L, T_F_L)


def test_pointer_intersects_screen():
    px_L, px_F = observations()
    point = cursor.solve_pointer(px_L, px_F, INTR_L, INTR_F, T_F_L, T_S_L, SCREEN_MM)
    assert point.x_norm == pytest.approx(175 / 344)
    assert point.y_norm == pytest.approx(65 / 194)
    assert point.reprojection_px < 1e-6


@pytest.mark.parametrize("thumb_weight", [0, 0.25, 0.5, 0.75, 1])
def test_pointer_aims_through_weighted_3d_pinch_point(thumb_weight):
    points_S = np.array([[100., 50., -600.], [150., 60., -200.], [110., 80., -240.]])
    points_L = (T_S_L[:3, :3].T @ points_S.T).T
    point = cursor.solve_pointer(project(points_L), project(points_L, T_F_L),
                                 INTR_L, INTR_F, T_F_L, T_S_L, SCREEN_MM, thumb_weight=thumb_weight)
    pinch = (1 - thumb_weight) * points_S[1] + thumb_weight * points_S[2]
    hit = points_S[0] + (pinch - points_S[0]) * (-points_S[0, 2] / (pinch[2] - points_S[0, 2]))
    assert point.x_norm == pytest.approx(hit[0] / SCREEN_MM[0])
    assert point.y_norm == pytest.approx(hit[1] / SCREEN_MM[1])


def test_pointer_rejects_mismatched_landmarks():
    px_L, px_F = observations()
    px_F[1] += [80, 50]
    with pytest.raises(cursor.InvalidPointer, match="reprojection"):
        cursor.solve_pointer(px_L, px_F, INTR_L, INTR_F, T_F_L, T_S_L, SCREEN_MM)


def two_hand_observations():
    # Independent hands with different screen intersections, plus both iris positions.
    points_S = np.array([[100., 50., -600.], [110., 50., -600.],
                         [150., 60., -200.], [145., 60., -200.],
                         [120., 70., -200.], [115., 70., -200.]])
    points_L = (T_S_L[:3, :3].T @ points_S.T).T
    observations = []
    for transform in (np.eye(4), T_F_L):
        px = project(points_L, transform)
        face = np.zeros((478, 2))
        face[RIGHT_IRIS], face[LEFT_IRIS] = px[:2]
        hands = []
        for label, indices in (("Right", (2, 3)), ("Left", (4, 5))):
            hand = np.zeros((21, 2))
            hand[[8, 4]] = px[list(indices)]
            hands.append((label, 0.99, hand))
        observations.append(Landmarks(100, face, hands))
    return observations


def test_independent_hands_and_selected_eye():
    a, b = two_hand_observations()
    intr = {"L": INTR_L, "F": INTR_F}
    right_eye = browser_cursor._pointer_samples(a, b, "R", intr, T_F_L, T_S_L, SCREEN_MM)
    assert right_eye["L"]["valid"] and right_eye["R"]["valid"]
    assert right_eye["L"]["x"] != right_eye["R"]["x"]
    assert len(right_eye["R"]["hand_mm"]) == 3
    assert np.isfinite(right_eye["R"]["hand_mm"]).all()
    assert right_eye["R"]["hand_mm"][2] < 0
    left_eye = browser_cursor._pointer_samples(a, b, "L", intr, T_F_L, T_S_L, SCREEN_MM)
    assert left_eye["R"]["valid"]
    assert left_eye["R"]["x"] != right_eye["R"]["x"]
    b.hands = [h for h in b.hands if h[0] == "Right"]
    missing = browser_cursor._pointer_samples(a, b, "R", intr, T_F_L, T_S_L, SCREEN_MM)
    assert missing["R"]["valid"] and not missing["L"]["valid"]


def test_pinch_points_preserves_default_and_chooses_same_hand():
    a, _ = two_hand_observations()
    np.testing.assert_array_equal(a.pinch_points(), a.right_hand()[[8, 4]])
    np.testing.assert_array_equal(a.pinch_points("L"), a.hand("L")[[8, 4]])
    with pytest.raises(ValueError):
        a.pinch_points("invalid")


def test_settings_validate_sides_and_return_a_copy():
    samples = browser_cursor.Samples()
    assert samples.settings() == {"eye": "R", "primary_hand": "R", "thumb_weight": 0.75, "release_sensitivity": 0.5}
    samples.set_settings({"eye": "L", "primary_hand": "L"})
    copy = samples.settings()
    copy["eye"] = "R"
    assert samples.settings()["eye"] == "L"
    for invalid in ({"eye": "X", "primary_hand": "R"}, {"eye": "L"}, [],
                    {"eye": "L", "primary_hand": ["R"]}):
        with pytest.raises(ValueError):
            samples.set_settings(invalid)


def test_settings_endpoint_rejects_foreign_origins_and_malformed_payloads():
    import io
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock

    samples = browser_cursor.Samples()
    handler_type = browser_cursor._handler(samples)
    handler = handler_type.__new__(handler_type)
    handler.path = "/settings"
    handler.server = SimpleNamespace(server_port=8765)
    handler.send_error, handler.send_response, handler.end_headers = Mock(), Mock(), Mock()
    body = json.dumps({"eye": "L", "primary_hand": "L", "thumb_weight": 0.25}).encode()
    handler.headers = {"Origin": "http://127.0.0.1:8765", "Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler.do_POST()
    handler.send_response.assert_called_once_with(204)
    assert samples.settings() == {"eye": "L", "primary_hand": "L", "thumb_weight": 0.25, "release_sensitivity": 0.5}
    handler.headers["Origin"] = "https://example.com"
    handler.do_POST()
    handler.send_error.assert_called_with(403)
    handler.headers = {"Content-Length": "999999"}
    handler.do_POST()
    handler.send_error.assert_called_with(400)
    handler.headers = {"Content-Length": "1"}
    handler.rfile = io.BytesIO(b"{")
    handler.do_POST()
    handler.send_error.assert_called_with(400)


@pytest.mark.parametrize("field", ["thumb_weight", "release_sensitivity"])
@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5", None])
def test_invalid_cursor_setting_is_rejected(field, weight):
    samples = browser_cursor.Samples()
    with pytest.raises(ValueError):
        samples.set_settings({"eye": "R", "primary_hand": "R", field: weight})
    assert samples.settings()["thumb_weight"] == 0.75


def test_ratio_applies_to_both_independent_cursors():
    a, b = two_hand_observations()
    intr = {"L": INTR_L, "F": INTR_F}
    index = browser_cursor._pointer_samples(a, b, "R", intr, T_F_L, T_S_L, SCREEN_MM, thumb_weight=0)
    thumb = browser_cursor._pointer_samples(a, b, "R", intr, T_F_L, T_S_L, SCREEN_MM, thumb_weight=1)
    for hand in ("L", "R"):
        assert index[hand]["valid"] and thumb[hand]["valid"]
        assert index[hand]["x"] != thumb[hand]["x"]


def test_pinch_views_preserve_each_cameras_open_and_closed_readings():
    a, b = two_hand_observations()
    for lm, scale in ((a, 1), (b, 3)):
        hand = lm.hand("R")
        hand[5], hand[17] = [0, 0], [100 * scale, 0]
        hand[8], hand[4] = [40 * scale, 0], [50 * scale, 0]
    assert browser_cursor._pinch_views(a, b, "R") == pytest.approx({"L": .1, "F": .1})
    b.hand("R")[8] = [0, 0]
    assert browser_cursor._pinch_views(a, b, "R") == pytest.approx({"L": .1, "F": .5})
    b.hands = []
    assert browser_cursor._pinch_views(a, b, "R") == pytest.approx({"L": .1})
    a.hands = []
    assert browser_cursor._pinch_views(a, b, "R") == {}


def test_unsynchronized_views_can_still_report_a_pinch():
    a, b = two_hand_observations()
    a.hand("R")[5], a.hand("R")[17] = [0, 0], [100, 0]
    a.hand("R")[8], a.hand("R")[4] = [40, 0], [50, 0]
    b.hands = []
    b.t_ms += 30
    result = browser_cursor._pointer_samples(a, b, "R", {"L": INTR_L, "F": INTR_F},
                                             T_F_L, T_S_L, SCREEN_MM)
    assert not result["R"]["valid"]
    assert result["R"]["pinch_ratio"] == pytest.approx(.1)
