import * as THREE from 'three';
import {STLLoader} from './STLLoader.js';

const panel = document.querySelector('#viewer-panel');
const canvas = document.querySelector('#viewer-canvas');
const status = document.querySelector('#viewer-status');
try {
  const renderer = new THREE.WebGLRenderer({canvas, antialias: true});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#111923');
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  const pivot = new THREE.Group();
  scene.add(pivot);
  scene.add(new THREE.HemisphereLight(0xe6f4ff, 0x384354, 2.5));
  const light = new THREE.DirectionalLight(0xffffff, 3);
  light.position.set(3, 4, 5);
  scene.add(light);
  let grab = null, ready = false;
  function reset() {
    grab = null;
    pivot.rotation.set(-0.55, 0.3, 0);
    camera.position.set(0, 0, 2.7);
    camera.lookAt(0, 0, 0);
  }
  function rotate(dx, dy) {
    const rotation = new THREE.Quaternion().setFromEuler(new THREE.Euler(dy, dx, 0, 'YXZ'));
    pivot.quaternion.premultiply(rotation).normalize();
  }
  function cursorVector(x, y) {
    const rect = canvas.getBoundingClientRect();
    const radius = rect.height / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * camera.position.z);
    const point = new THREE.Vector3((x - rect.left - rect.width / 2) / radius,
                                   (rect.top + rect.height / 2 - y) / radius, 0);
    point.z = Math.sqrt(Math.max(0, 1 - point.lengthSq()));
    return point.normalize();
  }
  function followCursor(x, y) {
    const from = cursorVector(grab.x, grab.y), to = cursorVector(x, y);
    pivot.quaternion.premultiply(new THREE.Quaternion().setFromUnitVectors(from, to)).normalize();
    grab.x = x; grab.y = y;
  }
  function zoom(delta) { camera.position.z = THREE.MathUtils.clamp(camera.position.z * Math.exp(delta), 1.8, 9); }
  reset();
  document.querySelector('#reset-view').addEventListener('click', reset);
  const coordinates = pointer => pointer?.valid && Number.isFinite(pointer.x) && Number.isFinite(pointer.y) && pointer.hand_mm?.length === 3 && pointer.hand_mm.every(Number.isFinite)
    ? pointer.hand_mm : null;
  window.propellerViewer = {
    begin(hand, pointer) {
      const position = coordinates(pointer);
      if (!ready || panel.hidden || !position) return false;
      grab = {owner: hand, position: [...position], x: pointer.x * innerWidth, y: pointer.y * innerHeight};
      return true;
    },
    move(hand, pointer) {
      if (!grab || grab.owner !== hand) return;
      if (panel.hidden || document.hidden || !document.hasFocus()) { grab = null; return; }
      const position = coordinates(pointer);
      if (!position) return; // Missing tracking holds the view until a valid sample returns.
      const delta = position.map((v, i) => v - grab.position[i]);
      grab.position = [...position];
      // shortcut: reject >100 mm frame jumps; revisit after real-hardware motion tuning.
      if (Math.hypot(...delta) > 100) return;
      followCursor(pointer.x * innerWidth, pointer.y * innerHeight);
      zoom(delta[2] * 0.003);
    },
    end(owner) { if (!owner || grab?.owner === owner) grab = null; }
  };
  canvas.addEventListener('pointerdown', event => {
    if (event.button !== 0 || !ready) return;
    event.preventDefault();
    canvas.focus();
    grab = {owner: 'mouse', x: event.clientX, y: event.clientY, pointerId: event.pointerId};
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', event => {
    if (grab?.owner !== 'mouse' || grab.pointerId !== event.pointerId) return;
    followCursor(event.clientX, event.clientY);
  });
  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    canvas.addEventListener(type, () => { if (grab?.owner === 'mouse') grab = null; });
  }
  window.addEventListener('blur', () => { grab = null; });
  document.addEventListener('visibilitychange', () => { if (document.hidden) grab = null; });
  canvas.addEventListener('wheel', event => { event.preventDefault(); zoom(event.deltaY * 0.001); }, {passive: false});
  canvas.addEventListener('keydown', event => {
    const direction = {ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, -0.1], ArrowDown: [0, 0.1]}[event.key];
    if (direction) { event.preventDefault(); rotate(...direction); }
    if (event.key === '+' || event.key === '=') zoom(-0.1);
    if (event.key === '-') zoom(0.1);
  });
  canvas.addEventListener('webglcontextlost', event => {
    event.preventDefault(); ready = false; grab = null;
    status.textContent = '3D graphics interrupted. Reload the page to retry.';
  });
  new STLLoader().load('/viewer/propeller.stl', geometry => {
    geometry.computeBoundingBox();
    const size = geometry.boundingBox.getSize(new THREE.Vector3());
    const diameter = Math.max(size.x, size.y, size.z);
    if (!Number.isFinite(diameter) || diameter <= 0) {
      status.textContent = 'The propeller mesh is empty or invalid.';
      return;
    }
    geometry.center();
    geometry.rotateX(Math.PI / 2);
    geometry.scale(2 / diameter, 2 / diameter, 2 / diameter);
    geometry.computeVertexNormals();
    pivot.add(new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({color: 0x9bc4db, metalness: 0.55, roughness: 0.38, side: THREE.DoubleSide})));
    ready = true;
    status.textContent = '';
  }, undefined, () => { status.textContent = 'Could not load the propeller. Reload to retry.'; });
  function draw() {
    requestAnimationFrame(draw);
    if (panel.hidden) return;
    const width = panel.clientWidth, height = panel.clientHeight;
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.render(scene, camera);
  }
  requestAnimationFrame(draw);
} catch (error) {
  status.textContent = `3D viewer unavailable: ${error.message}`;
}
