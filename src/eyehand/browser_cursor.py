"""Development cursor: live eye/hand geometry rendered only inside a localhost page."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from eyehand import cursor, intrinsics, mirror, status, stereo
from eyehand.capture import Camera
from eyehand.config import load_camera_configs
from eyehand.landmarks import Landmarks, Tracker
from eyehand.screen_calib import screen_pitch_mm
from eyehand.timebase import FrameClock

MAX_PAIR_GAP_MS = 20.0

STREAM_WORKER = """const ports = new Set();
let stream = null;
onconnect = event => {
  const port = event.ports[0];
  ports.add(port);
  port.onmessage = event => {
    if (event.data !== 'disconnect') return;
    ports.delete(port);
    port.close();
    if (!ports.size) { stream?.close(); stream = null; }
  };
  port.start();
  if (!stream) {
    stream = new EventSource('/stream');
    stream.onmessage = event => {
      const sample = JSON.parse(event.data);
      for (const client of ports) client.postMessage(sample);
    };
  }
};""".encode()

CLAW_SVG = """<svg viewBox="-32 -12 64 72" width="48" height="54" aria-hidden="true">
  <g fill="currentColor" stroke="#e4f5ff" stroke-width="1.5" stroke-linejoin="round">
    <path d="M-8 28 C-20 33 -14 51 1 53 C17 52 22 35 8 28 Z"/>
    <path d="M2 32 C20 32 27 18 20 6 C17 -2 9 -6 0 0 L9 9 L7 12 L12 14
             C12 23 8 25 2 25 Z"/>
    <g class="jaw" transform="rotate(-30 0 30)">
      <path d="M-4 32 C-17 26 -21 13 -15 3 C-11 -3 -6 -5 0 0 L-7 7 L-5 10 L-10 12
               C-12 21 -7 25 -3 25 Z"/>
    </g>
  </g>
  <path d="M-9 35 Q-10 44 0 47 M12 4 Q19 10 16 18" fill="none" stroke="#ffffff88" stroke-width="2"/>
</svg>"""

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Eye-Hand browser cursor</title>
<style>
  html, body { height: 100%; margin: 0; overflow: hidden; font: 16px system-ui; color: #dce7f3;
               background: #111923; }
  body { background-image: linear-gradient(#ffffff12 1px, transparent 1px),
                           linear-gradient(90deg, #ffffff12 1px, transparent 1px);
         background-size: 10vw 10vh; }
  #message { position: fixed; left: 16px; bottom: 8px; z-index: 3; padding: 8px 12px; border-radius: 8px;
             background: #07101bd9; padding-right: 52px; }
  #dismiss { position: absolute; right: 4px; top: 4px; width: 40px; height: 40px;
             border: 0; background: transparent; color: inherit; font-size: 24px; cursor: pointer; }
  #dismiss:focus-visible { outline: 2px solid white; }
  .dot { position: fixed; left: 0; top: 0; width: 48px; height: 54px;
         margin-left: -24px; margin-top: -9px; color: #ff7857;
         filter: drop-shadow(0 2px 3px #000a); pointer-events: none; display: none; z-index: 10; }
  .dot.secondary { color: #3bc9f1; }
  .dot.clicking { color: #42e58c; }
  #dot-L svg { transform: scaleX(-1); }
  #controls { position: fixed; right: 24px; top: 20px; z-index: 2; max-width: 300px;
              background: #07101bf2; border-radius: 8px; padding: 12px 16px; }
  summary { cursor: pointer; padding: 4px; }
  fieldset { border: 0; margin: 12px 0 0; padding: 0; }
  label { display: flex; justify-content: space-between; align-items: center; gap: 18px; margin: 10px 0; }
  select { font: inherit; min-width: 72px; padding: 6px; }
  #settings-error { color: #ffb4ab; }
  small { color: #9bb0c3; }
  #demo { position: fixed; inset: 12px 16px 84px; display: flex; flex-direction: column; }
  #tabs { display: flex; gap: 12px; margin-bottom: 12px; padding-right: 205px; }
  #tabs button { font: inherit; color: inherit; background: #233449; border: 2px solid #60758e;
                 border-radius: 12px; padding: 16px 28px; cursor: pointer; }
  #tabs button[aria-selected="true"] { background: #245b75; border-color: #77ddff; }
  button:focus-visible { outline: 3px solid #fff; outline-offset: 4px; }
  #drag-panel { flex: 1; min-height: 0; display: flex; flex-direction: column; }
  #drag-panel[hidden], #entry-panel[hidden] { display: none; }
  #playground { flex: 1; position: relative; overflow: hidden; border: 2px dashed #60758e;
                border-radius: 20px; background: #111923cc; min-height: 110px; }
  .shape { position: absolute; width: 144px; height: 144px; padding: 0; border: 0;
           background: none; touch-action: none; cursor: grab; user-select: none; }
  .shape svg { width: 100%; height: 100%; pointer-events: none; filter: drop-shadow(0 4px 3px #0006); }
  #shape-card { width: 182px; height: 182px; }
  .shape.grabbed { cursor: grabbing; filter: brightness(1.3); z-index: 1; }
  .shape-target { position: absolute; width: 216px; height: 216px; bottom: 24px;
                  color: #a5b6ca; pointer-events: none; }
  .shape-target.complete { color: #42e58c; }
  .shape-target svg { width: 100%; height: 100%; overflow: visible; }
  .shape-target:not(.special) svg > * { fill: transparent; stroke: currentColor; stroke-width: 1.8;
                          stroke-dasharray: .1 3; stroke-linecap: round; }
  .shape-target.special { box-sizing: border-box; border: 3px dotted currentColor; border-radius: 16px; padding: 14px; }
  .shape-target.special svg > * { stroke-dasharray: none; }
  .shape-target.special svg .card-face { fill: #fffdf5; stroke: #c6cad1; stroke-width: 1; }
  .shape-target.special svg text { fill: #17202b; }
  .shape-target.special.complete { background: #42e58c22; }
  #viewer-panel { flex: 1; min-height: 0; position: relative; background: #111923; border-radius: 20px; overflow: hidden; }
  #viewer-panel[hidden] { display: none; }
  #viewer-canvas { width: 100%; height: 100%; display: block; touch-action: none; cursor: grab; }
  #viewer-tools { position: absolute; top: 12px; left: 16px; right: 16px; pointer-events: none; }
  #viewer-tools button { pointer-events: auto; font: inherit; padding: 12px 20px; border-radius: 8px; cursor: pointer; }
  #viewer-status { margin: 8px 0; }
  #viewer-credit { position: absolute; bottom: 8px; right: 12px; font-size: 11px; color: #9bb0c3; pointer-events: none; }
  #entry-panel { flex: 1; padding: 40px; background: #172535; border-radius: 20px; }

  #name-form { max-width: 900px; margin: 8vh auto; }
  #name-form label { display: block; font-size: 28px; font-weight: 600; }
  #patient-name { box-sizing: border-box; width: 100%; font: 36px system-ui; padding: 24px;
                  border: 3px solid #60758e; border-radius: 14px; color: #fff; background: #101c2b; }
  #patient-name:focus { outline: 4px solid #77ddff; outline-offset: 4px; }
  #patient-name[data-mode] { border-color: #42e58c; background: #123b32; }
  #voice-status { font-size: 21px; min-height: 30px; color: #77ddff; }
  #name-form button { font: inherit; padding: 14px 22px; border-radius: 10px; cursor: pointer; margin-right: 12px; }
  #voice-preview { min-height: 24px; color: #9bb0c3; }
</style>
<div id="message"><span id="tracking-status">Waiting for both cameras…</span><br><small>Page clicks · C6/C8 not calibrated</small>
<button id="dismiss" type="button" aria-label="Close status message" onclick="document.querySelector('#message').hidden = true">×</button></div>
<details id="controls">
<summary>Cursor controls</summary>
<fieldset id="settings" disabled>
<label>Aiming eye <select id="eye"><option value="L">L</option><option value="R" selected>R</option></select></label>
<label>Dominant clicker hand <select id="primary-hand"><option value="L">L</option><option value="R" selected>R</option></select></label>
<label for="thumb-weight">Thumb / index ratio <output id="ratio-value" for="thumb-weight">75% thumb</output></label>
<input id="thumb-weight" type="range" min="0" max="100" step="1" value="75" style="width:100%">
<small>0%: index tip · 50%: midpoint · 100%: thumb tip</small>
<small id="applied-ratio">Checking applied ratio…</small>
<label for="release-sensitivity">Release sensitivity <output id="release-value" for="release-sensitivity">50%</output></label>
<input id="release-sensitivity" type="range" min="0" max="100" step="1" value="50" style="width:100%">
<small>Higher sensitivity releases with a smaller opening.</small>
</fieldset>
<p id="settings-error" role="status"></p>
</details>
<main id="demo">
<nav id="tabs" role="tablist" aria-label="Demos">
<button id="drag-tab" role="tab" aria-selected="true" aria-controls="drag-panel">Move shapes</button>
<button id="entry-tab" role="tab" aria-selected="false" aria-controls="entry-panel">Data entry</button>
<button id="viewer-tab" role="tab" aria-selected="false" aria-controls="viewer-panel">3D Viewer</button>
</nav>
<section id="drag-panel" role="tabpanel" aria-labelledby="drag-tab">
<div id="playground" aria-label="Shape play area">
<button class="shape" id="shape-circle" aria-label="Circle"><svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="44" fill="#62d8ed"/></svg></button>
<button class="shape" id="shape-triangle" aria-label="Triangle"><svg viewBox="0 0 100 100"><path d="M50 5 L96 92 H4 Z" fill="#ffbd69"/></svg></button>
<button class="shape" id="shape-square" aria-label="Photo"><svg viewBox="0 0 100 100"><rect x="6" y="6" width="88" height="88" rx="8" fill="#f1f5ff"/><rect x="12" y="12" width="76" height="76" rx="4" fill="#9ccfff"/><circle cx="68" cy="32" r="10" fill="#ffe28a"/><path d="M12 77 L36 43 L57 67 L69 52 L88 77 V88 H12 Z" fill="#4e997c"/></svg></button>
<button class="shape" id="shape-card" aria-label="Two of Spades"><svg viewBox="0 0 100 100"><rect class="card-face" x="19" y="2" width="62" height="96" rx="5" fill="#fffdf5" stroke="#c6cad1"/><g fill="#17202b" font-family="Georgia, serif"><text x="24" y="21" font-size="19">2</text><text x="24" y="36" font-size="18">♠</text><text x="50" y="43" text-anchor="middle" font-size="29">♠</text><text x="50" y="78" text-anchor="middle" font-size="29" transform="rotate(180 50 66)">♠</text></g></svg></button>
<button class="shape" id="shape-folder" aria-label="junk files folder"><svg viewBox="0 0 100 100"><path d="M7 22 H40 L49 31 H93 V83 H7 Z" fill="#d89a31"/><path d="M7 37 Q7 33 12 33 H88 Q94 33 93 39 L88 83 H12 Z" fill="#ffd16c"/><text x="50" y="97" text-anchor="middle" fill="#e4edf7" font-family="system-ui" font-size="13">junk files</text></svg></button>
</div></section>
<section id="entry-panel" role="tabpanel" aria-labelledby="entry-tab" hidden>
<div id="name-form">
<label for="patient-name">Patient Name</label>
<input id="patient-name" type="text" autocomplete="off" placeholder="Click here with the mouse or claw" aria-describedby="voice-instructions voice-status">
<p id="voice-instructions">Say <strong>“Write this”</strong> or <strong>“Spell this”</strong>, then say the name.</p>
<p id="voice-status" role="status">Select Patient Name to begin.</p>
<p id="voice-preview" aria-live="polite"></p>
<button id="enable-microphone" type="button">Enable microphone</button>
<button id="clear-name" type="button">Clear name</button>
<p><small>Spelling: letter names, space, hyphen, apostrophe, period, and digits. Three seconds of silence ends entry. Use fictional patient names.</small></p>
</div></section>
<section id="viewer-panel" role="tabpanel" aria-labelledby="viewer-tab" hidden>
<canvas id="viewer-canvas" aria-label="Toroidal propeller. Drag to rotate; scroll to zoom. Arrow keys rotate; plus and minus zoom." tabindex="0"></canvas>
<div id="viewer-tools"><button id="reset-view" type="button">Reset view</button>
<p id="viewer-status" role="status">Loading propeller…</p>
</div>
<small id="viewer-credit">Propeller model by Akshay Venugopal</small>
</section>
</main>
<div id="dot-L" class="dot" aria-hidden="true"><!--claw--></div>
<div id="dot-R" class="dot" aria-hidden="true"><!--claw--></div>
<script>
const playground = document.querySelector('#playground');
let drag = null;
function endDrag() {
  if (drag) drag.shape.classList.toggle('grabbed', false);
  drag = null;
}
function moveDrag(x, y) {
  if (!drag) return;
  const area = playground.getBoundingClientRect();
  drag.shape.style.left = `${Math.max(0, Math.min(area.width - drag.shape.offsetWidth, x - area.left - drag.dx))}px`;
  drag.shape.style.top = `${Math.max(0, Math.min(area.height - drag.shape.offsetHeight, y - area.top - drag.dy))}px`;
  updateTargets();
}
function startDrag(shape, x, y, owner) {
  if (drag) return;
  const rect = shape.getBoundingClientRect();
  drag = {shape, dx: x - rect.left, dy: y - rect.top, owner};
  shape.classList.toggle('grabbed', true);
}
const shapeNames = ['circle', 'triangle', 'square', 'card', 'folder'];
const shapeBoundaries = new Map();
function shapeInsideTarget(shape, target) {
  const rect = shape.getBoundingClientRect(), bounds = target.getBoundingClientRect();
  if (!rect.width || !bounds.width) return false;
  if (['shape-square', 'shape-card', 'shape-folder'].includes(shape.id)) {
    return rect.left > bounds.left && rect.top > bounds.top &&
           rect.right < bounds.right && rect.bottom < bounds.bottom;
  }
  const outline = target.querySelector('svg > *');
  return shapeBoundaries.get(shape).every(point => outline.isPointInFill({
    x: 100 * (rect.left + point.x * rect.width / 100 - bounds.left) / bounds.width,
    y: 100 * (rect.top + point.y * rect.height / 100 - bounds.top) / bounds.height
  }));
}
function updateTargets() {
  for (const name of shapeNames) {
    const shape = document.querySelector(`#shape-${name}`), target = document.querySelector(`#target-${name}`);
    target.classList.toggle('complete', shapeInsideTarget(shape, target));
  }
}
for (const [index, name] of shapeNames.entries()) {
  const shape = document.querySelector(`#shape-${name}`);
  const geometry = shape.querySelector('svg > *'), length = geometry.getTotalLength();
  // shortcut: sample contours at one SVG unit; use exact intersections for subpixel scoring.
  shapeBoundaries.set(shape, Array.from({length: Math.ceil(length)}, (_, i) => geometry.getPointAtLength(i)));
  const position = 5 + 22.5 * index;
  shape.style.left = `calc(${position}% - ${shape.offsetWidth * position / 100}px)`;
  shape.style.top = '24px';
  const target = document.createElement('div');
  target.id = `target-${name}`;
  target.className = 'shape-target';
  target.setAttribute('role', 'img');
  target.setAttribute('aria-label', `${name} target`);
  target.style.left = `calc(${position}% - ${2.16 * position}px)`;
  target.innerHTML = shape.innerHTML;
  if (name === 'square') {
    target.className += ' special';
    target.setAttribute('aria-label', 'Upload photos drop zone');
    target.innerHTML = '<svg viewBox="0 0 100 100"><path d="M50 59 V22 M36 36 L50 22 L64 36 M23 57 V72 H77 V57" fill="none" stroke="currentColor" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/><text x="50" y="93" text-anchor="middle" font-family="system-ui" font-size="12" style="fill:currentColor">Upload photos</text></svg>';
  } else if (name === 'card') {
    target.className += ' special';
    target.setAttribute('aria-label', 'Ace of Spades target');
    target.innerHTML = '<svg viewBox="0 0 100 100"><rect class="card-face" x="19" y="2" width="62" height="96" rx="5"/><text x="24" y="22" font-family="Georgia, serif" font-size="20">A</text><text x="24" y="38" font-size="18">♠</text><text x="50" y="68" text-anchor="middle" font-size="48">♠</text></svg>';
  } else if (name === 'folder') {
    target.className += ' special';
    target.setAttribute('aria-label', 'Recycle Bin target');
    target.innerHTML = '<svg viewBox="0 0 100 100"><path d="M24 22 L29 82 Q50 91 71 82 L76 22 Z" fill="#d7e6ec" stroke="#91a6b5"/><ellipse cx="50" cy="22" rx="26" ry="8" fill="#f1f8fb" stroke="#91a6b5"/><ellipse cx="50" cy="22" rx="19" ry="4" fill="#8fa6b5"/><text x="50" y="65" text-anchor="middle" fill="#238e53" font-size="35">♻</text><text x="50" y="98" text-anchor="middle" fill="#e4edf7" font-family="system-ui" font-size="12">Recycle Bin</text></svg>';
  }
  playground.prepend(target);
  shape.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    event.preventDefault();
    endDrag();
    startDrag(shape, event.clientX, event.clientY, 'mouse');
    shape.setPointerCapture(event.pointerId);
  });
  shape.addEventListener('pointermove', event => {
    if (drag?.owner === 'mouse' && drag.shape === shape) moveDrag(event.clientX, event.clientY);
  });
  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    shape.addEventListener(type, () => { if (drag?.owner === 'mouse') endDrag(); });
  }
  shape.addEventListener('keydown', event => {
    const delta = {ArrowLeft: [-20, 0], ArrowRight: [20, 0], ArrowUp: [0, -20], ArrowDown: [0, 20]}[event.key];
    if (!delta || drag) return;
    event.preventDefault();
    const rect = shape.getBoundingClientRect();
    startDrag(shape, rect.left, rect.top, 'keyboard');
    moveDrag(rect.left + delta[0], rect.top + delta[1]);
    endDrag();
  });
}
updateTargets();
for (const name of ['drag', 'entry', 'viewer']) {
  document.querySelector(`#${name}-tab`).addEventListener('click', () => {
    endDrag();
    window.propellerViewer?.end();
    if (name !== 'entry') window.nameEntry?.stop();
    for (const other of ['drag', 'entry', 'viewer']) {
      document.querySelector(`#${other}-tab`).setAttribute('aria-selected', String(other === name));
      document.querySelector(`#${other}-panel`).hidden = other !== name;
    }
    updateTargets();
  });
}
window.addEventListener('resize', () => {
  endDrag();
  for (const name of ['circle', 'triangle', 'square', 'card', 'folder']) {
    const shape = document.querySelector(`#shape-${name}`), rect = shape.getBoundingClientRect();
    startDrag(shape, rect.left, rect.top, 'resize');
    moveDrag(rect.left, rect.top);
    endDrag();
  }
});
const message = document.querySelector('#tracking-status');
const eye = document.querySelector('#eye'), primaryHand = document.querySelector('#primary-hand');
const settings = document.querySelector('#settings'), settingsError = document.querySelector('#settings-error');
const thumbWeight = document.querySelector('#thumb-weight'), ratioValue = document.querySelector('#ratio-value');
const appliedRatio = document.querySelector('#applied-ratio');
const releaseSensitivity = document.querySelector('#release-sensitivity'), releaseValue = document.querySelector('#release-value');
let pinchOpen = 0.60;
releaseSensitivity.addEventListener('input', () => saveSettings());
thumbWeight.addEventListener('input', () => {
  ratioValue.textContent = `${thumbWeight.value}% thumb`;
  saveSettings();
});
function applySettings(value) {
  eye.value = value.eye;
  primaryHand.value = value.primary_hand;
  thumbWeight.value = Math.round(value.thumb_weight * 100);
  ratioValue.textContent = `${thumbWeight.value}% thumb`;
  releaseSensitivity.value = Math.round((value.release_sensitivity ?? 0.5) * 100);
  releaseValue.textContent = `${releaseSensitivity.value}%`;
  pinchOpen = 0.85 - 0.5 * (value.release_sensitivity ?? 0.5);
}
async function loadSettings() {
  const response = await fetch('/settings', {cache: 'no-store'});
  if (!response.ok) throw new Error('Could not load cursor settings. Reload to retry.');
  applySettings(await response.json());
  appliedRatio.textContent = `Applied: ${thumbWeight.value}% thumb`;
}
loadSettings().then(() => { settings.disabled = false; }).catch(error => { settingsError.textContent = error.message; });
let savingSettings = false, pendingSettings = null;
async function saveSettings() {
  pendingSettings = {eye: eye.value, primary_hand: primaryHand.value,
                     thumb_weight: Number(thumbWeight.value) / 100,
                     release_sensitivity: Number(releaseSensitivity.value) / 100};
  applySettings(pendingSettings);
  if (savingSettings) return;
  savingSettings = true;
  settingsError.textContent = '';
  try {
    while (pendingSettings) {
      const value = pendingSettings;
      pendingSettings = null;
      const response = await fetch('/settings', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(value)});
      if (!response.ok) throw new Error('Could not save cursor settings.');
      appliedRatio.textContent = `Applied: ${Math.round(value.thumb_weight * 100)}% thumb`;
    }
  } catch (error) {
    pendingSettings = null;
    settingsError.textContent = `${error.message} Adjust a control to retry.`;
  } finally {
    savingSettings = false;
  }
}
settings.addEventListener('change', event => {
  if (event.target !== thumbWeight && event.target !== releaseSensitivity) saveSettings();
});
let sample, received = 0;
const lastValid = {L: null, R: null};
const pinch = {L: {down: false, armed: {}}, R: {down: false, armed: {}}};
const pressed = {L: null, R: null};
const pinchClose = 0.30;
let spaceDown = false, spaceHand = null;
function pressHand(hand) {
  if (pressed[hand]) return;
  const last = lastValid[hand];
  if (!last || performance.now() - last.time >= 3000 || document.hidden || !document.hasFocus()) return;
  const x = last.pointer.x * innerWidth, y = last.pointer.y * innerHeight;
  const target = document.elementFromPoint(x, y), primary = hand === sample.primary_hand;
  const shape = target?.closest?.('.shape');
  if (primary && target?.closest?.('#viewer-panel') && !target.closest('button')) {
    if (window.propellerViewer?.begin(hand, last.pointer)) {
      pressed[hand] = {target, primary, dragging: true};
      return;
    }
  }
  if (!target || (primary && shape && drag)) return;
  pressed[hand] = {target, primary, dragging: primary && !!shape};
  if (primary && shape) startDrag(shape, x, y, hand);
  target.dispatchEvent(new MouseEvent('mousedown', {
    bubbles: true, cancelable: true, view: window, button: primary ? 0 : 2,
    buttons: primary ? 1 : 2,
    clientX: x, clientY: y
  }));
}
function releaseHand(hand, cancel = false) {
  window.propellerViewer?.end(hand);
  const press = pressed[hand];
  pressed[hand] = null;
  if (drag?.owner === hand) endDrag();
  if (!press) return;
  const last = lastValid[hand].pointer, x = last.x * innerWidth, y = last.y * innerHeight;
  const options = {bubbles: true, cancelable: true, view: window,
                   button: press.primary ? 0 : 2, buttons: 0, clientX: x, clientY: y};
  press.target.dispatchEvent(new MouseEvent('mouseup', options));
  if (!cancel && !press.dragging && document.elementFromPoint(x, y) === press.target) {
    press.target.dispatchEvent(new MouseEvent(press.primary ? 'click' : 'contextmenu', options));
  }
}
function handTracked(hand) {
  const pointer = sample?.pointers?.[hand];
  return performance.now() - received < 300 && (pointer?.valid || Number.isFinite(pointer?.pinch_ratio));
}
function releaseSpace(cancel = false) {
  if (spaceHand && (cancel || !pinch[spaceHand].down)) releaseHand(spaceHand, cancel);
  spaceDown = false; spaceHand = null;
}
document.addEventListener('keydown', event => {
  if (event.code !== 'Space' || event.target.closest('input, select, textarea, button, summary, [contenteditable]') ||
      document.hidden || !document.hasFocus()) return;
  event.preventDefault();
  if (event.repeat || spaceDown) return;
  spaceDown = true;
  const primary = sample?.primary_hand, secondary = primary === 'R' ? 'L' : 'R';
  spaceHand = handTracked(primary) ? primary : handTracked(secondary) ? secondary : null;
  if (spaceHand && !pinch[spaceHand].down) pressHand(spaceHand);
});
document.addEventListener('keyup', event => { if (event.code === 'Space') releaseSpace(); });
function cancelPresses() {
  releaseSpace(true);
  for (const hand of ['L', 'R']) {
    releaseHand(hand, true);
    pinch[hand].down = false;
    pinch[hand].armed = {};
  }
  endDrag();
}
window.addEventListener('blur', cancelPresses);
document.addEventListener('visibilitychange', () => { if (document.hidden) cancelPresses(); });
function receiveSample(value) {
  sample = {...value, primary_hand: value.primary_hand ?? sample?.primary_hand ?? primaryHand.value};
  if (Number.isFinite(sample.release_sensitivity)) pinchOpen = 0.85 - 0.5 * sample.release_sensitivity;
  if (Number.isFinite(sample.thumb_weight)) {
    appliedRatio.textContent = `Applied: ${Math.round(sample.thumb_weight * 100)}% thumb`;
  }
  received = performance.now();
  for (const hand of ['L', 'R']) {
    const pointer = sample?.pointers?.[hand];
    if (pointer?.valid) lastValid[hand] = {pointer, time: received};
    const state = pinch[hand];
    const views = pointer?.pinch_views ?? {legacy: pointer?.pinch_ratio};
    if (document.hidden || !document.hasFocus()) {
      releaseHand(hand, true);
      state.down = false;
      state.armed = {};
      continue;
    }
    let press = false, release = false, observed = false;
    for (const [camera, ratio] of Object.entries(views)) {
      if (!Number.isFinite(ratio)) continue;
      observed = true;
      if (ratio >= pinchOpen) {
        state.armed[camera] = true;
        release = true;
      } else if (ratio <= pinchClose && state.armed[camera]) {
        state.armed[camera] = false;
        press = true;
      }
    }
    if (!observed && !state.down) state.armed = {};
    if (state.down && release) {
      if (spaceHand !== hand) releaseHand(hand);
      state.down = false;
    } else if (!state.down && press) {
      state.down = true;
      if (spaceHand !== hand) pressHand(hand);
    }
  }
}
let tracking;
function connectTracking() {
  tracking = new SharedWorker('/cursor-stream.js');
  tracking.onerror = () => { settingsError.textContent = 'Tracking connection failed. Reload the page.'; };
  tracking.port.onmessage = event => receiveSample(event.data);
  tracking.port.start();
}
connectTracking();
window.addEventListener('pagehide', () => tracking.port.postMessage('disconnect'));
window.addEventListener('pageshow', event => { if (event.persisted) connectTracking(); });
function draw() {
  const fresh = performance.now() - received < 300;
  if (drag && ['L', 'R'].includes(drag.owner)) {
    const hand = drag.owner, pointer = sample?.pointers?.[hand];
    if (hand !== sample?.primary_hand ||
        (!pinch[hand].down && spaceHand !== hand) || document.hidden || !document.hasFocus()) endDrag();
    else if (fresh && pointer?.valid) moveDrag(pointer.x * innerWidth, pointer.y * innerHeight);
  }
  if (fresh && sample?.primary_hand) {
    const hand = sample.primary_hand;
    window.propellerViewer?.move(hand, sample.pointers?.[hand]);
  }
  const statuses = [];
  for (const hand of ['L', 'R']) {
    const dot = document.querySelector(`#dot-${hand}`), pointer = sample?.pointers?.[hand];
    const last = lastValid[hand];
    const primary = hand === sample?.primary_hand;
    dot.classList.toggle('secondary', !primary);
    const clicking = (pinch[hand].down || (spaceHand === hand && handTracked(hand))) &&
                     !document.hidden && document.hasFocus();
    dot.classList.toggle('clicking', clicking);
    dot.querySelector('.jaw').setAttribute('transform', clicking ? 'rotate(0 0 30)' : 'rotate(-30 0 30)');
    if (last && performance.now() - last.time < 3000) {
      dot.style.display = 'block';
      dot.style.transform = `translate(${last.pointer.x * innerWidth}px, ${last.pointer.y * innerHeight}px)`;
      statuses.push(`${hand}: ${fresh && pointer?.valid ? 'tracking' : 'holding last position'}`);
    } else {
      dot.style.display = 'none';
      statuses.push(`${hand}: ${fresh ? (pointer?.reason || sample?.reason || 'waiting') : 'waiting for cameras'}`);
    }
  }
  message.textContent = statuses.join(' · ');
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);
</script>
<script type="importmap">{"imports":{"three":"/viewer/three.module.js"}}</script>
<script type="module" src="/viewer/viewer.js"></script>
<script type="module" src="/viewer/entry.js"></script>
""".replace("<!--claw-->", CLAW_SVG).encode()


class Samples:
    def __init__(self):
        self._condition = threading.Condition()
        self._seq = 0
        self._sample = {"valid": False, "reason": "Starting cameras…"}
        self._settings = {"eye": "R", "primary_hand": "R", "thumb_weight": 0.75, "release_sensitivity": 0.5}

    def settings(self) -> dict:
        with self._condition:
            return self._settings.copy()

    def set_settings(self, settings: dict) -> None:
        if (not isinstance(settings, dict) or not {"eye", "primary_hand"} <= set(settings) or
                set(settings) - {"eye", "primary_hand", "thumb_weight", "release_sensitivity"} or any(
                settings[key] not in ("L", "R") for key in ("eye", "primary_hand"))):
            raise ValueError("eye and primary_hand must each be L or R")
        weight = settings.get("thumb_weight", 0.75)
        if type(weight) not in (int, float) or not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("thumb_weight must be between 0 and 1")
        sensitivity = settings.get("release_sensitivity", 0.5)
        if type(sensitivity) not in (int, float) or not np.isfinite(sensitivity) or not 0 <= sensitivity <= 1:
            raise ValueError("release_sensitivity must be between 0 and 1")
        with self._condition:
            self._settings = {**settings, "thumb_weight": weight, "release_sensitivity": sensitivity}

    def publish(self, sample: dict) -> None:
        with self._condition:
            self._seq += 1
            self._sample = {"seq": self._seq, **sample}
            self._condition.notify_all()

    def after(self, seq: int, timeout: float = 15.0) -> tuple[int, dict] | None:
        with self._condition:
            self._condition.wait_for(lambda: self._seq > seq, timeout)
            return (self._seq, self._sample) if self._seq > seq else None


def _handler(samples: Samples):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/settings":
                self.send_error(404)
                return
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{self.server.server_port}",
                                         f"http://localhost:{self.server.server_port}"):
                self.send_error(403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 256:
                    raise ValueError("invalid settings length")
                samples.set_settings(json.loads(self.rfile.read(length)))
            except (ValueError, UnicodeError):
                self.send_error(400)
                return
            self.send_response(204)
            self.end_headers()

        def do_GET(self):
            if self.path == "/cursor-stream.js":
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
                self.send_header("Content-Length", str(len(STREAM_WORKER)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(STREAM_WORKER)
                return
            viewer_files = {"/viewer/" + name: name for name in
                            ("viewer.js", "entry.js", "three.module.js", "STLLoader.js", "propeller.stl")}
            if self.path in viewer_files:
                body = (Path(__file__).parent / "viewer" / viewer_files[self.path]).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript" if self.path.endswith(".js") else "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/events":
                # Retire old per-tab streams so stale tabs cannot exhaust browser connections.
                self.send_response(204)
                self.end_headers()
                return
            if self.path == "/settings":
                body = json.dumps(samples.settings()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
                return
            if self.path != "/stream":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = -1
            try:
                while True:
                    item = samples.after(seq)
                    if item is None:
                        self.wfile.write(b": keepalive\n\n")
                    else:
                        seq, sample = item
                        self.wfile.write(b"data: " + json.dumps(sample).encode() + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    return Handler


class CameraTracker(threading.Thread):
    def __init__(self, config):
        super().__init__(name=f"cursor-{config.name}", daemon=True)
        self.config = config
        self.history: deque[Landmarks] = deque(maxlen=4)
        self.lock = threading.Lock()
        self.stop_requested = threading.Event()
        self.error: Exception | None = None

    def run(self) -> None:
        tracker = None
        try:
            clock = FrameClock(self.config.frame_period_ms)
            tracker = Tracker()
            with Camera(self.config) as camera:
                while not self.stop_requested.is_set():
                    frame = camera.read()
                    t_ms = clock.add(frame.t_soe_ms).t_fit_ms
                    landmarks = tracker.process(frame.image, t_ms)
                    with self.lock:
                        self.history.append(landmarks)
        except Exception as exc:
            self.error = exc
        finally:
            if tracker is not None:
                tracker.close()

    def snapshot(self) -> list[Landmarks]:
        with self.lock:
            return list(self.history)


def _landmark_pixels(a: Landmarks, b: Landmarks, eye: str = "R",
                     hand: str = "R") -> tuple[np.ndarray, np.ndarray]:
    if eye not in ("L", "R"):
        raise ValueError("eye must be L or R")
    eye_a, eye_b = (a.left_eye(), b.left_eye()) if eye == "L" else (a.right_eye(), b.right_eye())
    tip_a, tip_b = a.pinch_points(hand), b.pinch_points(hand)
    if any(value is None for value in (eye_a, tip_a, eye_b, tip_b)):
        raise cursor.InvalidPointer(f"Show {eye} eye and {hand} hand's thumb and index tips to both cameras")
    return np.vstack((eye_a, tip_a)), np.vstack((eye_b, tip_b))


def _pinch_views(a: Landmarks, b: Landmarks, side: str) -> dict[str, float]:
    ratios = {}
    for camera, lm in (("L", a), ("F", b)):
        hand = lm.hand(side)
        if hand is None:
            continue
        width = float(np.linalg.norm(hand[5] - hand[17]))
        gap = float(np.linalg.norm(hand[8] - hand[4]))
        if not np.isfinite([width, gap]).all() or width <= 0:
            continue
        ratios[camera] = gap / width
    return ratios


def _pointer_samples(a, b, eye, intr, T_F_L, T_S_L, screen_mm, *, thumb_weight=0.75) -> dict:
    pointers = {}
    for hand in ("L", "R"):
        try:
            if abs(a.t_ms - b.t_ms) > MAX_PAIR_GAP_MS:
                raise cursor.InvalidPointer("Waiting for aligned frames")
            px_L, px_F = _landmark_pixels(a, b, eye, hand)
            point = cursor.solve_pointer(px_L, px_F, intr["L"], intr["F"], T_F_L, T_S_L, screen_mm,
                                         thumb_weight=thumb_weight)
            pointers[hand] = {"valid": True, "x": point.x_norm, "y": point.y_norm,
                              "reprojection_px": point.reprojection_px,
                              "ray_angle_deg": point.ray_angle_deg, "hand_mm": point.hand_mm}
        except cursor.InvalidPointer as exc:
            pointers[hand] = {"valid": False, "reason": str(exc)}
        views = _pinch_views(a, b, hand)
        pointers[hand]["pinch_views"] = views
        pointers[hand]["pinch_ratio"] = min(views.values()) if views else None
    return pointers


def _best_pair(history_L: list[Landmarks], history_F: list[Landmarks]):
    pairs = [(a, b) for a in history_L for b in history_F]
    aligned = [pair for pair in pairs if abs(pair[0].t_ms - pair[1].t_ms) <= MAX_PAIR_GAP_MS]
    return max(aligned, key=lambda pair: pair[0].t_ms + pair[1].t_ms) if aligned else min(
        pairs, key=lambda pair: abs(pair[0].t_ms - pair[1].t_ms))


def load_geometry(configs):
    statuses = status.calibration_status(configs)
    missing = [s.id for s in statuses if s.id in ("C1", "C2", "C4", "C5") and not s.usable]
    if missing:
        raise RuntimeError(f"calibration not usable: {', '.join(missing)}; run `eyehand status`")
    intr = {name: intrinsics.load_intrinsics(configs[name]) for name in ("L", "F")}
    T_F_L = np.linalg.inv(stereo.load_stereo(intr["L"], intr["F"]))
    T_S_L = mirror.load_T_S_L(intr["L"])
    pitch_x, pitch_y, _ = screen_pitch_mm()
    screen_mm = (1920 * pitch_x, 1080 * pitch_y)
    return intr, T_F_L, T_S_L, screen_mm


def run(port: int = 8765, *, stop_event: threading.Event | None = None,
        open_browser: bool = False, samples: Samples | None = None) -> int:
    stop_event = stop_event or threading.Event()
    configs = load_camera_configs()
    intr, T_F_L, T_S_L, screen_mm = load_geometry(configs)

    samples = samples if samples is not None else Samples()
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler(samples))
    server_thread = threading.Thread(target=server.serve_forever, name="cursor-http", daemon=True)
    workers = {name: CameraTracker(configs[name]) for name in ("L", "F")}
    print(f"Open http://127.0.0.1:{port} in your browser. Ctrl-C stops the cameras.")
    print("Development mode: movement only; C6 timing and C8 aiming correction are not complete.")
    server_thread.start()
    last_pair = None
    started = []
    try:
        for worker in workers.values():
            worker.start()
            started.append(worker)
        if open_browser:
            webbrowser.open(f"http://127.0.0.1:{port}")
        while not stop_event.is_set():
            for name, worker in workers.items():
                if worker.error is not None:
                    raise RuntimeError(f"camera {name}: {worker.error}") from worker.error
            histories = {name: worker.snapshot() for name, worker in workers.items()}
            if not all(histories.values()):
                time.sleep(0.01)
                continue
            a, b = _best_pair(histories["L"], histories["F"])
            settings = samples.settings()
            pair = (a.t_ms, b.t_ms, settings["eye"], settings["primary_hand"], settings["thumb_weight"])
            if pair == last_pair:
                time.sleep(0.005)
                continue
            last_pair = pair
            pointers = _pointer_samples(a, b, settings["eye"], intr, T_F_L, T_S_L, screen_mm,
                                        thumb_weight=settings["thumb_weight"])
            samples.publish({"pointers": pointers, **settings})
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    finally:
        for worker in started:
            worker.stop_requested.set()
        for worker in started:
            worker.join()
        server.shutdown()
        server.server_close()
    return 0
