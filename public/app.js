// Tessa3D web: a game-like front end for Tessa Ripple. All quantum work runs in the
// Vercel Python function (api/tessa.py → tessa_ripple/bridge.py → Tessa); this file
// draws the mesh as a pulsing point cloud, runs the arc menus from the Figma design
// ("Tessa3D"), and blends, interpolates and colours what the function returns.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const app = $('app'), home = $('home'), menuEl = $('menu'), panelEl = $('panel');
const W_STAGE = 1440, H_STAGE = 1024, CX = 720, CY = 512;
const HOVER = matchMedia('(hover: hover)').matches;
const OBJ_R = 270; // stage px: a click inside this radius of the object opens the menu

// ---------- settings: everything a run or the view reads
const P = {
  shape: 'blob', resolution: 1, payload: 'radial', mode: 'theta', layout: 'mesh', // resolution: 0 low (decimated), 1 medium (as is), 2 high (subdivided)
  clamp: 0, clamp_b: 0, compression: 0, quant_bits: 0, headroom: 0.1,
  spectral_k: 0, // spectral load: modes to encode, 0 = auto (1 % truncation); past one run's cap = the cap
  backend: 'noiseless', placement: 'quality', shots: 12, noise_strength: 1, seed: 0, drr: 'none',
  readout: 'z', // how θ is read: z, bloch (Z, X and Y: noise's shrink cancels) or snap (64 Bloch-sphere levels)
  frames: 12, morph_t: 0.5, shapeB: null,
  hold_t: false, interference: false, shared_topology: false, a_dirs: false, // morph options
  morph_front: false, front_width: 0.15, front_source: null, // the morph front (null source: auto)
  view: 'fidelity', colour: 'glow', gain: 3, between: 8, fps: 24, loop: true, orig: true, spin: true,
  gain_by: 'shots', // Material: the gain made by fewer shots (the readout's own noise), or by scaling (×)
};
// S: what the page has loaded and is showing
const S = { mesh: null, meshKey: null, data: null, run: null, orig: null, frames: [], refs: [], ts: null,
  seq: [], step: 0, playing: false, acc: 0, edges: null, focus: null, busy: false,
  keys: [], keyNames: [], seed0: 0, // keys: the positions S.seq glides through (gain applied), named for export
  objText: null, objName: null, objTextB: null, objNameB: null,
  front: null, picking: false }; // front: per-step weights of the last run's morph front

// ---------- stage scaling: the Figma frame (1440 × 1024) fitted to the window
// The design's content sits in the middle ~1100 px of the frame, so narrow windows crop
// the decorative sides rather than shrinking everything. Portrait windows use a narrower
// view (760 px on phones, 900 on tablets) and a taller one: the extra height above and
// below the frame (E) moves the logo and titles up and the run buttons down, and the
// morph drop box goes under the object.
let scale = 1, portrait = false, compact = false;
const layout = { E: 0, T: 0, B: 0, vl: 0 };
function fitStage() {
  const w = innerWidth, h = innerHeight;
  portrait = w / h < 0.9;
  compact = portrait || w < 900;
  const minW = portrait ? (w < 600 ? 760 : 900) : 1100;
  // portrait keeps ~40 px under the frame for the status line
  scale = Math.min(w / minW, (h - (portrait ? 40 : 0)) / H_STAGE, Math.max(w / W_STAGE, h / H_STAGE));
  layout.E = Math.max(0, (h / scale - H_STAGE) / 2);
  layout.T = portrait ? Math.min(layout.E, 140) : 0;
  // run buttons: down into spare height, but always clear of the status line (may be < 0)
  layout.B = portrait ? Math.min(layout.E - 44 / scale, 220) : 0;
  layout.vl = CX - w / scale / 2; // stage x of the window's left edge
  app.classList.toggle('portrait', portrait);
  app.classList.toggle('compact', compact);
  app.style.setProperty('--s', scale);
  // wide windows: spread the landing shapes out with the width (up to 1.35× the frame's spacing)
  app.style.setProperty('--spread', portrait ? 0 : Math.min(Math.max(w / scale / W_STAGE, 1), 1.35) - 1);
  app.style.setProperty('--T', `${layout.T}px`);
  app.style.setProperty('--B', `${layout.B}px`);
  app.style.setProperty('--vl', `${Math.max(0, layout.vl)}px`);
  app.style.setProperty('--vlx', `${layout.vl}px`); // may be < 0: the window is wider than the frame
  app.style.setProperty('--E', `${layout.E}px`);
  resize();
  syncShift();
  placeDrawer();
  placeMorphBox();
  relayoutMenus();
}
const toStage = (x, y) => [(x - innerWidth / 2) / scale + CX, (y - innerHeight / 2) / scale + CY];
// [min, max] of a mesh-sized array: Math.min(...a) passes every item as an argument and
// overflows the call stack past ~10^5 of them (a big uploaded OBJ)
function extent(a) {
  let lo = Infinity, hi = -Infinity;
  for (const x of a) { if (x < lo) lo = x; if (x > hi) hi = x; }
  return [lo, hi];
}

// ---------- three.js scene
const viewer = $('viewer');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
viewer.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, 1, 0.01, 100);
camera.up.set(0, 0, 1);
const VIEW_DIR = new THREE.Vector3(1.6, -5.4, 1.5).normalize();
const controls = new OrbitControls(camera, renderer.domElement);
Object.assign(controls, { enableDamping: true, dampingFactor: 0.06, enableZoom: false, enablePan: false, autoRotateSpeed: 0.55 });
let objRadius = 1;

// Positions and colours live in float textures (one texel per vertex), so the point
// cloud's samples (barycentric points on every face) follow the decoded vertices on
// the GPU with no per-sample work on the CPU.
const TEX_W = 256;
const CREASE = Math.cos(THREE.MathUtils.degToRad(38)); // sharper bends than this stay hard edges
let posTex = null, colTex = null, nrmTex = null;
const cloudGeom = new THREE.BufferGeometry();
// The surface both the dots and the thin edges sit on: positions from the vertex
// texture, curved between vertices, coloured by height (or by the data colour), and
// dimmer on the far side. At rest, size and brightness are steady (the only motion is
// the data's); while a run is encoding or reading, the dots shimmer and pulse (uBusy).
// Light mode inverts the whole page (CSS on <html>). Data colours must survive that with
// their hues, so in light mode they are first turned through CSS's hue-rotate(180deg)
// matrix (invert then hue-rotate keeps hues). The glow is the Moth palette, given per mode.
const LIGHT_GLSL = /* glsl */`
    uniform float uLight;
    const mat3 HUE180 = mat3(-0.574, 0.426, 0.426, 1.43, 0.43, 1.43, 0.144, 0.144, -0.856); // columns
    vec3 forTheme(vec3 c) { return uLight > 0.5 ? clamp(HUE180 * c, 0.0, 1.0) : c; }`;
const SURFACE_GLSL = /* glsl */`${LIGHT_GLSL}
    uniform sampler2D uPos; uniform sampler2D uCol; uniform sampler2D uNrm; uniform int uW;
    uniform float uMix, uZ0, uZ1, uCurve, uR;
    uniform vec3 uGlow0, uGlow1, uGlow2;
    attribute vec3 aTri; attribute vec3 aBary; attribute float aFace;
    varying vec3 vCol; varying float vA;
    vec3 at(sampler2D t, float i) { int k = int(i + 0.5); return texelFetch(t, ivec2(k % uW, k / uW), 0).xyz; }
    // returns the view-space position; sets vCol and vA (the far side dimmer)
    vec4 surface() {
      vec3 p1 = at(uPos, aTri.x), p2 = at(uPos, aTri.y), p3 = at(uPos, aTri.z);
      vec3 p = p1 * aBary.x + p2 * aBary.y + p3 * aBary.z;
      // Phong tessellation: bend the flat face towards each corner's tangent plane, so a
      // low-poly mesh reads as a smooth surface (the corners themselves stay exact). Corner
      // normals respect hard edges, so a cube stays a cube.
      vec3 n1 = at(uNrm, 3.0 * aFace), n2 = at(uNrm, 3.0 * aFace + 1.0), n3 = at(uNrm, 3.0 * aFace + 2.0);
      vec3 q = (p - dot(p - p1, n1) * n1) * aBary.x + (p - dot(p - p2, n2) * n2) * aBary.y + (p - dot(p - p3, n3) * n3) * aBary.z;
      p = mix(p, q, uCurve);
      vec3 c = at(uCol, aTri.x) * aBary.x + at(uCol, aTri.y) * aBary.y + at(uCol, aTri.z) * aBary.z;
      float h = clamp((p.z - uZ0) / max(uZ1 - uZ0, 1e-5), 0.0, 1.0);
      // Moth palette by height (see GLOW): base, middle, crown
      vec3 glow = h < 0.45 ? mix(uGlow0, uGlow1, h / 0.45) : mix(uGlow1, uGlow2, (h - 0.45) / 0.55);
      vCol = mix(glow, forTheme(c), uMix);
      vec4 mv = modelViewMatrix * vec4(p, 1.0);
      float depth = (mv.z - (modelViewMatrix * vec4(0.0, 0.0, 0.0, 1.0)).z) / uR;
      vA = mix(0.22, 1.0, smoothstep(-0.7, 0.45, depth));
      return mv;
    }`;
// The default glow, base to crown, in sRGB. Dark ground: electric blue, information blue,
// and the highlighter-yellow signal at the crown. Light ground (the page is inverted, so
// these are stored inverted): information blue, the electric-blue signal, navy.
const GLOW = { dark: ['#2306FF', '#00BBFF', '#DCF900'], light: ['#00BBFF', '#2306FF', '#19238E'] };
const glowVec = (hex, invert) => new THREE.Vector3(...[1, 3, 5].map((k) => {
  const v = parseInt(hex.slice(k, k + 2), 16) / 255;
  return invert ? 1 - v : v;
}));
const lightUniform = { value: 0 }; // shared by every material that draws data colours
const surfaceUniforms = {
  uLight: lightUniform,
  uGlow0: { value: glowVec(GLOW.dark[0]) }, uGlow1: { value: glowVec(GLOW.dark[1]) }, uGlow2: { value: glowVec(GLOW.dark[2]) },
  uPos: { value: null }, uCol: { value: null }, uNrm: { value: null }, uCurve: { value: 0.75 }, uR: { value: 1 },
  uW: { value: TEX_W }, uMix: { value: 0 }, uZ0: { value: -1 }, uZ1: { value: 1 },
  uBusy: { value: 0 }, uTime: { value: 0 },
};
const cloudMat = new THREE.ShaderMaterial({
  uniforms: { ...surfaceUniforms, uSize: { value: 0.02 }, uScale: { value: 500 }, uPx: { value: 1 } },
  vertexShader: /* glsl */`${SURFACE_GLSL}
    uniform float uSize, uScale, uPx, uBusy, uTime;
    attribute float aVert;
    void main() {
      vec4 mv = surface();
      vA *= aVert > 0.5 ? 1.0 : 0.85;
      // busy: each dot pulses on its own phase and a scan line sweeps up the object
      float seed = fract(sin(float(gl_VertexID) * 12.9898) * 43758.5453);
      float pulse = 0.5 + 0.5 * sin(uTime * 5.5 + seed * 6.2832);
      float h = clamp((at(uPos, aTri.x).z - uZ0) / max(uZ1 - uZ0, 1e-5), 0.0, 1.0);
      float band = exp(-pow((fract(h * 0.9 - uTime * 0.3) - 0.5) * 12.0, 2.0));
      vCol *= 1.0 + uBusy * (0.35 * pulse - 0.15 + 0.75 * band);
      vA *= 1.0 - uBusy * 0.4 * (1.0 - pulse);
      gl_Position = projectionMatrix * mv;
      float size = uSize * (aVert > 0.5 ? 1.35 : 1.0) * (1.0 + uBusy * (0.55 * pulse - 0.25));
      gl_PointSize = max(size * uScale / -mv.z, 1.6 * uPx);
    }`,
  fragmentShader: /* glsl */`
    varying vec3 vCol; varying float vA;
    void main() {
      float d = length(gl_PointCoord - 0.5) * 2.0;
      if (d > 1.0) discard;
      float a = (smoothstep(1.0, 0.55, d) * 0.85 + pow(1.0 - d, 3.0) * 0.6) * vA;
      gl_FragColor = vec4(vCol * a, a);
    }`,
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
});
// Thin edges: every mesh edge, split into short segments along the same curved
// surface, faint beside the dots (WebGL lines are one pixel wide)
const WIRE_SEGMENTS = 6;
const wireGeom = new THREE.BufferGeometry();
const wireMat = new THREE.ShaderMaterial({
  uniforms: cloudMat.uniforms, // shared, so the dots and edges always agree
  vertexShader: /* glsl */`${SURFACE_GLSL}
    void main() { gl_Position = projectionMatrix * surface(); }`,
  fragmentShader: /* glsl */`
    varying vec3 vCol; varying float vA;
    void main() { float a = 0.32 * vA; gl_FragColor = vec4(vCol * a, a); }`,
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
});
const wire = new THREE.LineSegments(wireGeom, wireMat);
wire.frustumCulled = false;
const cloud = new THREE.Points(cloudGeom, cloudMat);
cloud.frustumCulled = false;

const edgeGeom = new THREE.BufferGeometry();
/**
 * Make a built-in material's output follow light mode (see LIGHT_GLSL).
 *
 * Args:
 *   mat: a three.js material; its final sRGB colour is turned for the theme.
 *
 * Returns:
 *   The same material.
 */
function themed(mat) {
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uLight = lightUniform;
    sh.fragmentShader = `${LIGHT_GLSL}\n${sh.fragmentShader}`.replace(/}\s*$/, '  gl_FragColor.rgb = forTheme(gl_FragColor.rgb);\n}');
  };
  return mat;
}
const edgeLines = new THREE.LineSegments(edgeGeom, themed(new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false })));
// Qubit map: every vertex on the hovered data qubit gets an orange dot.
const discTexture = (fill, ring, width) => {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const x = c.getContext('2d');
  x.beginPath(); x.arc(32, 32, 30 - width / 2, 0, 2 * Math.PI);
  x.fillStyle = fill; x.fill();
  x.lineWidth = width; x.strokeStyle = ring; x.stroke();
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace; // so the orange matches the qubit map's exactly
  return t;
};
const mateGeom = new THREE.BufferGeometry();
const mates = new THREE.Points(mateGeom, themed(new THREE.PointsMaterial({ map: discTexture('#FFA158', '#100F0F', 12), alphaTest: 0.5, size: 0.08 })));
// the morph front's source vertex
const srcGeom = new THREE.BufferGeometry();
srcGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3));
const srcMark = new THREE.Points(srcGeom, themed(new THREE.PointsMaterial({ map: discTexture('#FDE725', '#100F0F', 12), alphaTest: 0.5, size: 0.12 })));
edgeLines.visible = mates.visible = cloud.visible = wire.visible = srcMark.visible = false;
scene.add(wire, cloud, edgeLines, mates, srcMark);

function resize() {
  const w = viewer.clientWidth || innerWidth, h = viewer.clientHeight || innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  cloudMat.uniforms.uScale.value = (h * renderer.getPixelRatio()) / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
  cloudMat.uniforms.uPx.value = renderer.getPixelRatio();
  frameCamera();
}
// Put the camera so the object's radius spans ~275 stage px, like the design's 582 px object.
function frameCamera() {
  const h = viewer.clientHeight || innerHeight;
  const d = (objRadius * h) / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 275 * scale);
  const dir = camera.position.lengthSq() > 0 ? camera.position.clone().sub(controls.target).normalize() : VIEW_DIR;
  camera.position.copy(dir.multiplyScalar(d));
  controls.target.set(0, 0, 0);
  camera.near = d / 50; camera.far = d * 10;
  camera.updateProjectionMatrix();
}
camera.position.copy(VIEW_DIR);
addEventListener('resize', fitStage);

// ---------- colour definitions, mirrored in tessa_ripple/app.py so both apps match
const stops = (hexes) => hexes.map((h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16) / 255)); // sRGB
const ERROR_GLOW_HEX = ['#5ef2ff', '#fcfdbf', '#fd9668', '#cf4070', '#b02cff']; // on black: cyan = no error
const EDGE_HEX = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']; // viridis
const ERROR = stops(ERROR_GLOW_HEX), EDGE = stops(EDGE_HEX);
const GOLDEN = 0.618034;
// HSL is given in sRGB (what the eye sees), not three.js's linear working space.
const hsl = (h, s, l) => new THREE.Color().setHSL(((h % 1) + 1) % 1, s, l, THREE.SRGBColorSpace);
// blends in sRGB, like CSS gradients and matplotlib, so mesh and legend agree; `out` lets
// per-vertex loops reuse one Color instead of allocating one per vertex every frame
function ramp(cs, t, out = new THREE.Color()) {
  t = Math.min(1, Math.max(0, t));
  const x = t * (cs.length - 1), i = Math.min(Math.floor(x), cs.length - 2), f = x - i, a = cs[i], b = cs[i + 1];
  return out.setRGB(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, THREE.SRGBColorSpace);
}

// ---------- API
const MORPH_OPTS = ['hold_t', 'interference', 'shared_topology', 'a_dirs'];
// Settings the server reads; an uploaded mesh travels as OBJ text.
function params() {
  const p = {
    resolution: P.resolution, payload: P.payload, layout: P.layout, mode: P.mode,
    clamp: P.clamp * Math.PI || null, clamp_b: P.clamp_b * Math.PI || null,
    compression: P.compression, quant_bits: P.quant_bits || null, headroom: P.headroom,
    backend: P.backend, placement: P.placement, shots: 2 ** P.shots, noise_strength: P.noise_strength,
    seed: P.seed, drr: P.drr, morph_t: P.morph_t, readout: P.readout,
  };
  if (P.payload === 'spectral') p.spectral_k = P.spectral_k || null; // null: auto
  if (S.objText) p.obj = S.objText; else p.shape = P.shape;
  if (P.mode === 'morph') {
    if (S.objTextB) p.obj_b = S.objTextB; else if (P.shapeB) p.shape_b = P.shapeB;
    for (const k of MORPH_OPTS) p[k] = P[k];
    if (P.morph_front) { p.morph_front = true; p.front_width = P.front_width; p.front_source = P.front_source; }
    p.clamp_b = p.clamp; // morph uses one clamp for both fields (bridge.morph_clamp)
  }
  return p;
}
// Material's gain by shots: the run reads at shots / gain² (bridge.gain_shots), so the
// motion is the readout's own; null when the gain is a classical scale (or Fidelity)
const gainShots = () => (P.view === 'material' && P.gain_by === 'shots' ? P.gain : null);
// The same settings with uploads named instead of inlined: what "settings changed" compares.
const runKey = () => JSON.stringify({ ...params(), obj: S.objName, obj_b: S.objNameB, gain_shots: gainShots() });
const meshKeyNow = () => `${S.objName ? `upload:${S.objName}` : P.shape}/${P.resolution}`;
const objLabel = () => S.objName || P.shape;
const bLabel = () => S.objNameB || P.shapeB;

async function call(action, extra = {}) {
  const res = await fetch('/api/tessa', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, ...params(), ...extra }),
  });
  let body;
  try { body = await res.json(); } catch { throw new Error(`server error ${res.status} (the function may have timed out)`); }
  if (!res.ok) throw new Error(body.error || `server error ${res.status}`);
  return body;
}
// the log's lines each start with a capital (the first letter, after any symbol like ⚠)
const capLines = (text) => text.split('\n').map((l) => l.replace(/\p{L}/u, (c) => c.toUpperCase())).join('\n');
function status(msg, err = false) {
  $('status').textContent = capLines(msg);
  $('status').classList.toggle('err', err);
  if (err) setLog(true); // a problem opens the log so it is seen
}
// the log: open by default; its arrow collapses it
function setLog(open) {
  $('info').classList.toggle('open', open);
  $('logToggle').setAttribute('aria-expanded', open);
  $('logToggle').setAttribute('aria-label', open ? 'Hide the log' : 'Show the log');
}
$('logToggle').addEventListener('click', () => setLog(!$('info').classList.contains('open')));
setLog(true);
// light mode (the default): inverts the whole page (CSS on <html>); a choice is remembered in this browser
const tipEl = $('tip'); // the shared hover label (see showTip)
let tipFor = null;
const THEME = 'tessa3d.light';
function setLight(on) {
  document.documentElement.classList.toggle('light', on);
  lightUniform.value = on ? 1 : 0;
  GLOW[on ? 'light' : 'dark'].forEach((hex, i) => { cloudMat.uniforms[`uGlow${i}`].value = glowVec(hex, on); });
  $('themeToggle').setAttribute('aria-pressed', on);
  $('themeToggle').setAttribute('aria-label', on ? 'Switch to dark mode' : 'Switch to light mode');
  $('themeToggle').dataset.tip = on ? 'dark mode' : 'light mode'; // its hover label: what a click switches to
  if (tipFor === $('themeToggle')) showTip(tipFor);
}
$('themeToggle').addEventListener('click', () => {
  const on = !document.documentElement.classList.contains('light');
  setLight(on); drawMap();
  try { localStorage.setItem(THEME, on ? '1' : '0'); } catch { /* storage blocked: this visit only */ }
});
let storedTheme = null;
try { storedTheme = localStorage.getItem(THEME); } catch { /* storage blocked */ }
setLight(storedTheme !== '0'); // light unless dark was chosen
async function busy(label, fn) {
  S.busy = true;
  app.classList.add('busy');
  syncPlay();
  syncHud(); // every tool greys, its hover label saying a run is going
  status(label);
  try { await fn(); } catch (err) { status(`⚠ ${err.message}`, true); } finally {
    S.busy = false;
    app.classList.remove('busy');
    syncPlay();
    syncHud();
  }
}

// ---------- the object: load, sample, draw
async function loadMesh() {
  const key = meshKeyNow();
  if (key === S.meshKey && S.mesh) return;
  const d = await call('mesh');
  setMesh(d, key);
}

function setMesh(d, key) {
  if (key !== S.meshKey) P.front_source = null; // a picked source belongs to its mesh
  S.mesh = d; S.meshKey = key; S.data = null; S.run = null; S.edges = null; S.focus = null;
  setEdges(null);
  S.orig = new Float32Array(d.verts.flat());
  S.frames = []; S.refs = []; S.ts = null;
  buildCloud(d);
  rebuild();
  S.step = 0; S.playing = false;
  $('stats').textContent = capLines(`${objLabel()} · ${d.verts.length} vertices · ${d.faces.length} faces`);
  drawMap();
  syncHud(); syncStale();
}

// Sample the surface: points spread over every face by area, plus every vertex itself
// (drawn larger), each tied to its triangle's corners by barycentric weights.
function buildCloud(d) {
  const n = d.verts.length, V = d.verts;
  const tris = [];
  for (const f of d.faces) for (let k = 1; k < f.length - 1; k++) tris.push([f[0], f[k], f[k + 1]]);
  const area = tris.map(([a, b, c]) => {
    const u = [0, 1, 2].map((i) => V[b][i] - V[a][i]), v = [0, 1, 2].map((i) => V[c][i] - V[a][i]);
    return 0.5 * Math.hypot(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]);
  });
  const total = area.reduce((s, a) => s + a, 0) || 1;
  // bounding radius about the centroid, for the camera and the sample density
  const c = [0, 1, 2].map((i) => V.reduce((s, p) => s + p[i], 0) / n);
  objRadius = extent(V.map((p) => Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2])))[1] || 1;
  // dot pitch: ~70 dots across the object, coarser if that would pass ~60k dots
  let spacing = objRadius / 27;
  spacing = Math.max(spacing, Math.sqrt(total / 60000));
  const T = [], B = [], verts = [];
  const WT = [], WB = [], WF = []; // the thin edges: segment end points, as (triangle, weights)
  const len = (a, b) => Math.hypot(V[a][0] - V[b][0], V[a][1] - V[b][1], V[a][2] - V[b][2]);
  const F = [];
  const put = (t, b, face, isVert = 0) => { T.push(...t); B.push(...b); F.push(face); verts.push(isVert); };
  // a regular lattice on each face: interior points per face, edge points once per edge,
  // and every vertex (drawn larger), so the cloud reads as even rows of dots
  const edgeSeen = new Set();
  tris.forEach(([a, b, c], t) => {
    const g = Math.max(1, Math.round(Math.max(len(a, b), len(b, c), len(c, a)) / spacing));
    for (let i = 1; i < g; i++) for (let j = 1; i + j < g; j++) put([a, b, c], [1 - (i + j) / g, i / g, j / g], t);
    [[0, 1], [1, 2], [2, 0]].forEach(([p, q]) => {
      const x = [a, b, c][p], y = [a, b, c][q], key = Math.min(x, y) * 1e6 + Math.max(x, y);
      if (edgeSeen.has(key)) return;
      edgeSeen.add(key);
      const k = Math.max(1, Math.round(len(x, y) / spacing));
      const along = (u) => { const w = [0, 0, 0]; w[p] = 1 - u; w[q] = u; return w; };
      for (let i = 1; i < k; i++) put([a, b, c], along(i / k), t);
      for (let i = 0; i < WIRE_SEGMENTS; i++) { // the edge itself, curved like its dots
        for (const u of [i / WIRE_SEGMENTS, (i + 1) / WIRE_SEGMENTS]) { WT.push(a, b, c); WB.push(...along(u)); WF.push(t); }
      }
    });
  });
  for (let i = 0; i < n; i++) put([i, i, i], [1, 0, 0], 0, 1);
  const count = verts.length;
  const tri = Float32Array.from(T), bary = Float32Array.from(B), vert = Float32Array.from(verts);
  // faces round each vertex, for the corner normals
  S.vertFaces = Array.from({ length: n }, () => []);
  tris.forEach((t, k) => t.forEach((i) => S.vertFaces[i].push(k)));
  cloudGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3)); // unused; three needs it
  cloudGeom.setAttribute('aTri', new THREE.BufferAttribute(tri, 3));
  cloudGeom.setAttribute('aBary', new THREE.BufferAttribute(bary, 3));
  cloudGeom.setAttribute('aVert', new THREE.BufferAttribute(vert, 1));
  cloudGeom.setAttribute('aFace', new THREE.BufferAttribute(Float32Array.from(F), 1));
  wireGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(WF.length * 3), 3)); // unused
  wireGeom.setAttribute('aTri', new THREE.BufferAttribute(Float32Array.from(WT), 3));
  wireGeom.setAttribute('aBary', new THREE.BufferAttribute(Float32Array.from(WB), 3));
  wireGeom.setAttribute('aFace', new THREE.BufferAttribute(Float32Array.from(WF), 1));
  const rows = Math.ceil(n / TEX_W);
  const tex = () => new THREE.DataTexture(new Float32Array(TEX_W * rows * 4), TEX_W, rows, THREE.RGBAFormat, THREE.FloatType);
  posTex?.dispose(); colTex?.dispose(); nrmTex?.dispose();
  posTex = tex(); colTex = tex();
  const nrows = Math.max(1, Math.ceil((3 * tris.length) / TEX_W));
  nrmTex = new THREE.DataTexture(new Float32Array(TEX_W * nrows * 4), TEX_W, nrows, THREE.RGBAFormat, THREE.FloatType);
  cloudMat.uniforms.uPos.value = posTex;
  cloudMat.uniforms.uCol.value = colTex;
  cloudMat.uniforms.uNrm.value = nrmTex;
  cloudMat.uniforms.uSize.value = spacing * 0.7;
  cloudMat.uniforms.uR.value = objRadius;
  S.tris = tris;
  [cloudMat.uniforms.uZ0.value, cloudMat.uniforms.uZ1.value] = extent(V.map((p) => p[2]));
  mates.material.size = spacing * 2.5;
  srcMark.material.size = spacing * 4;
  cloud.visible = wire.visible = true;
  frameCamera();
}

function timeline(keys, steps, loop) {
  const k = loop && keys.length > 1 ? [...keys, keys[0]] : keys;
  const out = [];
  for (let i = 0; i < k.length - 1; i++) {
    for (let s = 0; s <= steps; s++) {
      const t = s / (steps + 1), e = t * t * (3 - 2 * t), a = k[i], b = k[i + 1];
      const o = new Float32Array(a.length);
      for (let j = 0; j < a.length; j++) o[j] = a[j] + (b[j] - a[j]) * e;
      out.push(o);
    }
  }
  if (!loop || k.length === 1) out.push(k[k.length - 1]);
  return out;
}

// Classical layer: gain, interpolation and colour, applied to cached frames. Every
// frame has its own reference: the original mesh, or in a morph the noiseless morph
// at that frame's t. Gain and error act on the deviation from it.
function rebuild() {
  if (!S.orig) return;
  const n = S.orig.length / 3, refs = S.refs.length ? S.refs : S.frames.map(() => S.orig);
  let keys = S.frames;
  if (P.view === 'material' && P.gain_by === '×') { // by shots, the frames already carry the gain
    keys = S.frames.map((f, k) => {
      const o = new Float32Array(f.length), r = refs[k];
      for (let j = 0; j < f.length; j++) o[j] = r[j] + P.gain * (f[j] - r[j]);
      return o;
    });
  }
  const lead = P.orig || !keys.length ? [S.orig] : [];
  const glide = (k) => (k.length === 1 ? k : timeline(k, P.between, P.loop));
  S.keys = lead.concat(keys);
  S.keyNames = (lead.length ? ['original'] : []).concat(keys.map((_, k) =>
    `${S.ts ? `t ${S.ts[k].toFixed(3)}` : `readout ${k + 1}`} · seed ${S.seed0 + k}`));
  S.seq = glide(S.keys);
  S.step = Math.min(S.step, S.seq.length - 1);
  // Error colour follows the playback: each step's true decoded error (the same timeline
  // without gain), on one scale fixed to the largest error anywhere in the run.
  const real = glide(lead.concat(S.frames)), ref = glide(lead.concat(refs));
  S.err = real.map((v, s) => {
    const e = new Float32Array(n), r = ref[s];
    for (let i = 0; i < n; i++) e[i] = Math.hypot(v[3 * i] - r[3 * i], v[3 * i + 1] - r[3 * i + 1], v[3 * i + 2] - r[3 * i + 2]);
    return e;
  });
  S.errTop = S.err.reduce((top, e) => Math.max(top, extent(e)[1]), 0) || 1;
  // front weights follow the same timeline (the original leads in at 0: all A)
  const fr = S.data?.front;
  if (fr && fr.weights.length === S.frames.length) {
    const z = new Float32Array(n), w = fr.weights.map((x) => Float32Array.from(x)), sm = fr.smooth.map((x) => Float32Array.from(x));
    const lw = lead.length ? [z] : [];
    S.front = { w: glide(lw.concat(w)), smooth: glide(lw.concat(sm)) };
  } else S.front = null;
  colourise();
  draw();
  syncHud(); // download .glb needs two keys
}

function colourise(stepOnly = false) {
  if (!colTex) return;
  const n = S.orig.length / 3, col = colTex.image.data;
  const set = (i, c) => { col[4 * i] = c.r; col[4 * i + 1] = c.g; col[4 * i + 2] = c.b; };
  const c = new THREE.Color(); // reused by every ramp below
  let mix = 0.85;
  if (P.colour === 'error' && S.frames.length) {
    const err = S.err[S.step];
    for (let i = 0; i < n; i++) set(i, ramp(ERROR, err[i] / S.errTop, c));
    if (!stepOnly) legend('faceLegend', ERROR_GLOW_HEX, '0', S.errTop.toFixed(4), `vertex error at this step (distance${S.ts ? ' from the noiseless morph' : ''}, mesh units)`);
  } else if (P.colour === 'front' && S.front) {
    const w = S.front.w[S.step], sm = S.front.smooth[S.step];
    let gap = 0;
    for (let i = 0; i < n; i++) { set(i, ramp(EDGE, w[i], c)); gap += Math.abs(w[i] - sm[i]); }
    if (!stepOnly) legend('faceLegend', EDGE_HEX, '0 = A', '1 = B', `front: share of the way to B · source vertex ${S.data.front.source}`);
    $('faceLegend').querySelector('.gap')?.remove();
    $('faceLegend').insertAdjacentHTML('beforeend', `<div class="gap">Patch steps: ${(gap / n).toFixed(3)} from a smooth front</div>`);
  } else if (P.colour === 'qubit' && S.data?.spectral) { // rows are modes: no qubit of a vertex's own
    mix = 0;
    $('faceLegend').hidden = false;
    $('faceLegend').innerHTML = 'Qubit: the spectral load spreads every vertex over all modes (see the qubit map)';
  } else if (P.colour === 'qubit' && S.data) {
    const d = S.data, used = [...new Set(d.qubits)].sort((a, b) => a - b), rank = new Map(used.map((q, r) => [q, r]));
    for (let i = 0; i < n; i++) set(i, hsl(rank.get(d.qubits[i]) * GOLDEN, 0.8, 0.6));
    $('faceLegend').hidden = false;
    $('faceLegend').innerHTML = `<div class="swatches">${used.map((q, r) => `<i title="qubit ${q}" style="background:hsl(${((r * GOLDEN) % 1 * 360).toFixed(0)} 80% 60%)"></i>`).join('')}</div>One colour per data qubit (${used.length})`;
  } else {
    mix = 0;
    $('faceLegend').hidden = true;
  }
  cloudMat.uniforms.uMix.value = mix;
  colTex.needsUpdate = true;
}

function legend(id, colours, lo, hi, caption) {
  const el = $(id);
  el.hidden = false;
  el.innerHTML = `<div class="bar" style="background:linear-gradient(to right, ${colours.join(', ')})"></div>`
    + `<div class="ends"><span>${lo}</span><span>${hi}</span></div>${capLines(caption)}`;
}

function copyEnds(target, pairs, v) {
  pairs.forEach((vi, j) => { target[3 * j] = v[3 * vi]; target[3 * j + 1] = v[3 * vi + 1]; target[3 * j + 2] = v[3 * vi + 2]; });
}

const faceNormals = { unit: new Float32Array(0), area: new Float32Array(0) }; // draw's scratch buffers
function draw() {
  if (!S.seq.length || !posTex) return;
  if ((P.colour === 'error' && S.frames.length) || (P.colour === 'front' && S.front)) colourise(true);
  const v = S.seq[S.step], pos = posTex.image.data, nrm = nrmTex.image.data;
  for (let i = 0; i < v.length / 3; i++) {
    pos[4 * i] = v[3 * i]; pos[4 * i + 1] = v[3 * i + 1]; pos[4 * i + 2] = v[3 * i + 2];
  }
  // Corner normals of the current shape, for the curved faces: each corner averages
  // (by area) the faces round its vertex that bend less than CREASE from its own face,
  // so smooth surfaces round off and hard edges stay sharp.
  const T = S.tris;
  if (faceNormals.unit.length !== T.length * 3) { // sized once per mesh, reused every step
    faceNormals.unit = new Float32Array(T.length * 3);
    faceNormals.area = new Float32Array(T.length * 3);
  }
  const fn = faceNormals.unit, fu = faceNormals.area;
  T.forEach(([a, b, c], t) => {
    const ux = v[3 * b] - v[3 * a], uy = v[3 * b + 1] - v[3 * a + 1], uz = v[3 * b + 2] - v[3 * a + 2];
    const wx = v[3 * c] - v[3 * a], wy = v[3 * c + 1] - v[3 * a + 1], wz = v[3 * c + 2] - v[3 * a + 2];
    const nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx, l = Math.hypot(nx, ny, nz) || 1;
    fu[3 * t] = nx; fu[3 * t + 1] = ny; fu[3 * t + 2] = nz;
    fn[3 * t] = nx / l; fn[3 * t + 1] = ny / l; fn[3 * t + 2] = nz / l;
  });
  T.forEach((corners, t) => corners.forEach((i, k) => {
    let x = 0, y = 0, z = 0;
    for (const u of S.vertFaces[i]) {
      const d = fn[3 * u] * fn[3 * t] + fn[3 * u + 1] * fn[3 * t + 1] + fn[3 * u + 2] * fn[3 * t + 2];
      if (Math.abs(d) < CREASE) continue; // abs: faces may be wound either way
      const sgn = Math.sign(d);
      x += sgn * fu[3 * u]; y += sgn * fu[3 * u + 1]; z += sgn * fu[3 * u + 2];
    }
    const l = Math.hypot(x, y, z) || 1, o = 4 * (3 * t + k);
    nrm[o] = x / l; nrm[o + 1] = y / l; nrm[o + 2] = z / l;
  }));
  posTex.needsUpdate = true;
  nrmTex.needsUpdate = true;
  if (S.edges) {
    copyEnds(edgeGeom.attributes.position.array, S.edges.ends, v);
    edgeGeom.attributes.position.needsUpdate = true;
  }
  edgeLines.visible = !!S.edges;
  wire.visible = cloud.visible && !S.edges; // the read-edges overlay replaces the thin edges
  placeMates();
  placeSource();
}

// the front's source: the picked vertex, else the one the last run resolved (auto)
const frontSource = () => (P.front_source ?? (S.data?.front ? S.data.front.source : null));
function placeSource() {
  const i = frontSource();
  srcMark.visible = P.mode === 'morph' && P.morph_front && i != null && cloud.visible && !!(S.seq.length || S.orig);
  if (!srcMark.visible) return;
  const v = S.seq[S.step] || S.orig;
  srcGeom.attributes.position.array.set(v.subarray(3 * i, 3 * i + 3));
  srcGeom.attributes.position.needsUpdate = true;
  srcGeom.computeBoundingSphere();
}

// pick the front's source: the vertex nearest the click on screen, front-most if several
function pickVertex(cx, cy) {
  const v = S.seq[S.step] || S.orig;
  if (!v) return null;
  const r = renderer.domElement.getBoundingClientRect(), p = new THREE.Vector3();
  let best = null, bestZ = Infinity;
  for (let i = 0; i < v.length / 3; i++) {
    p.set(v[3 * i], v[3 * i + 1], v[3 * i + 2]).project(camera);
    const x = r.left + (p.x + 1) / 2 * r.width, y = r.top + (1 - p.y) / 2 * r.height;
    const d = Math.hypot(x - cx, y - cy);
    if (d < 22 && p.z < bestZ - 0.002) { best = i; bestZ = p.z; } else if (d < 22 && Math.abs(p.z - bestZ) <= 0.002 && best != null) {
      // same depth: keep the closer one on screen
      p.set(v[3 * best], v[3 * best + 1], v[3 * best + 2]).project(camera);
      if (Math.hypot(r.left + (p.x + 1) / 2 * r.width - cx, r.top + (1 - p.y) / 2 * r.height - cy) > d) best = i;
    }
  }
  return best;
}
function setPicking(on) {
  S.picking = on;
  app.classList.toggle('picking', on);
  syncMorphBox();
}
let pickDown = null;
addEventListener('pointerdown', (e) => { pickDown = S.picking && !insideUi(e.target) ? [e.clientX, e.clientY] : null; }, true);
addEventListener('pointerup', (e) => {
  if (!pickDown || !S.picking) return;
  const moved = Math.hypot(e.clientX - pickDown[0], e.clientY - pickDown[1]);
  pickDown = null;
  if (moved > 6) return; // a drag rotates the view
  const i = pickVertex(e.clientX, e.clientY);
  if (i == null) return status('Front: no vertex there. Click on a vertex of the object (or press pick source again to cancel).', true);
  P.front_source = i;
  setPicking(false);
  placeSource();
  syncStale();
  status(`Front source: vertex ${i}. Press Run & play to morph from there.`);
}, true);

const READOUT_NAMES = { z: 'in Z', bloch: 'as a Bloch direction (Z, X, Y)', snap: 'by snapping to 64 Bloch-sphere levels' };
function showInfo(d, p) {
  const s = d.stats;
  $('stats').textContent = capLines([
    `${objLabel()}${d.ts ? ` → ${bLabel()} (morph, ${d.match.method}, ${(100 * d.match.coverage).toFixed(0)}% of B covered)` : ''} · ${d.verts.length} verts · ${p.payload}${d.spectral ? ', modes low first' : `, ${p.layout} layout`}`,
    ...(d.ts ? [`${d.options.interference ? `interference readout · selector 0 = A, 1 = B${d.kept ? ` · decoded from ${(100 * d.kept.reduce((a, b) => a + b, 0) / d.kept.length).toFixed(0)}% of shots` : ''}` : 'basis sweep · θ = A, φ = B'} · clamp ${d.clamp.toFixed(2)}π (θ and φ)`
      + `${d.options.hold_t ? ` · held at t = ${d.ts[0]}` : ''}${d.options.a_dirs ? " · A's directions" : ''}`
      + `${d.front ? ` · front from vertex ${d.front.source}, width ${d.front.width.toFixed(2)}` : ''}`] : []),
    ...(d.spectral ? [`spectral: ${d.spectral.k} modes${d.spectral.rows > d.spectral.k ? ' + constant' : ''} × xyz${P.spectral_k ? '' : ' (auto)'}`
      + ` · truncation A ${(100 * d.spectral.error_a).toFixed(2)}%${d.spectral.error_b != null ? `, B ${(100 * d.spectral.error_b).toFixed(2)}%` : ''} of diag`] : []),
    s.graph + (s.quality != null ? ` · data-qubit quality ${s.quality.toFixed(2)}` : ''),
    `${s.values} values on ${s.qubits} qubits${s.qubits < s.graph_qubits ? ` (of ${s.graph_qubits})` : ''} = ${s.ratio.toFixed(2)} / qubit`
      + (s.runs > 1 ? ` · ${s.runs} runs of ${s.values_per_run.join(' / ')} values` : ''),
    `depth ${s.depth} · 2q gates ${s.twoq}` + (s.twoq_single !== s.twoq ? ` (θ alone ${s.twoq_single})` : ''),
    `${d.shots.toLocaleString()} shots · read ${READOUT_NAMES[d.readout]}${P.drr === 'pilots' ? ' · pilot DRR' : ''}`,
    `mean error ${(d.error).toFixed(4)} of bbox${P.view !== 'material' ? ' · true positions'
      : P.gain_by === 'shots' ? ` · material gain ${P.gain}× by shots (true positions)` : ` · material gain ${P.gain}× scaled`}`,
  ].join('\n'));
}

// ---------- actions
async function run(frames) {
  if (P.mode === 'morph' && !bLabel()) {
    openPanel('encode', 'mode');
    return status('⚠ Morph needs a second mesh: drop one in the box, or pick a shape', true);
  }
  app.classList.add('running'); // marks a run in progress until the readouts are back (or fail)
  await busy(frames > 1 ? `${P.mode === 'morph' ? 'Morphing' : 'Running'} ${frames} readouts on ${P.backend}…` : `Encoding and reading once on ${P.backend}…`, async () => {
    // a mesh too big for one lattice is read in several runs (stats.runs), never decimated.
    // The settings are read before the request: ones changed while it runs belong to the
    // next run, so the run button turns stale rather than claiming them.
    const p = params(), key = runKey(), meshKey = meshKeyNow();
    const d = await call('run', { frames, gain_shots: gainShots() });
    if (meshKey !== S.meshKey) setMesh(d, meshKey);
    Object.assign(S, { data: d, run: key, frames: d.frames.map((f) => new Float32Array(f.flat())),
      refs: d.refs ? d.refs.map((r) => new Float32Array(r.flat())) : [], ts: d.ts || null, step: 0, seed0: p.seed });
    showInfo(d, p);
    S.focus = null;
    rebuild();
    drawMap();
    S.playing = d.frames.length > 1;
    if (!S.playing) { S.step = S.seq.length - 1; draw(); }
    const cut = d.frames.length < d.requested ? ` (fitted to the time limit; asked for ${d.requested})` : '';
    const what = d.ts ? (d.ts.length > 1 ? `morph ${objLabel()} → ${bLabel()}: ${d.frames.length} readouts, t 0 → 1` : `morph at t = ${d.ts[0]}`) : `${d.frames.length} readout(s)`;
    const runs = d.stats.runs > 1 ? ` · too many vertices for one ${d.stats.graph}, so each readout ran in ${d.stats.runs} runs` : '';
    status(`${what} in ${d.seconds.toFixed(1)} s on ${p.backend}${cut}${runs}`);
  });
  app.classList.remove('running');
  syncPlay(); syncStale();
}

function setEdges(r) {
  S.edges = r;
  $('edges').textContent = r ? 'Hide edges' : 'Read edges';
  $('edgeLegend').hidden = !r;
  if (!r) { edgeLines.visible = false; return; }
  const [lo, hi] = extent(r.exact), span = hi - lo || 1; // true edge range
  legend('edgeLegend', EDGE_HEX, lo.toFixed(3), hi.toFixed(3), 'edge length read from coherences');
  r.ends = r.edges.flat();
  edgeGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(r.ends.length * 3), 3));
  const col = new Float32Array(r.ends.length * 3);
  r.measured.forEach((m, k) => {
    const c = ramp(EDGE, (m - lo) / span); // off-range lengths clip to the ends
    for (const j of [2 * k, 2 * k + 1]) { col[3 * j] = c.r; col[3 * j + 1] = c.g; col[3 * j + 2] = c.b; }
  });
  edgeGeom.setAttribute('color', new THREE.BufferAttribute(col, 3));
}

async function readEdges() {
  if (S.edges) { setEdges(null); status('edge overlay hidden'); return draw(); }
  S.playing = false; syncPlay();
  await busy('Reading edge lengths from coherences…', async () => {
    const r = await call('edges');
    setEdges(r);
    const err = r.measured.reduce((s, m, k) => s + Math.abs(m - r.exact[k]), 0) / r.exact.length;
    const mean = r.exact.reduce((s, x) => s + x, 0) / r.exact.length;
    status(`${r.edges.length} edge lengths read from the state (${3 * r.edges.length} pair circuits, ${r.seconds.toFixed(1)} s): mean error ${(100 * err / mean).toFixed(1)}% of the mean edge`);
    draw();
  });
}

function openDialog(html) {
  $('dialogBody').innerHTML = html;
  $('dialog').showModal();
}

// ---------- Why Tessa: the conclusion first, then the table that backs it
const PHOENIX_QUBITS = 120;
const num = (v) => v.toLocaleString();
// Shots each value gets when every shot is split over a cell's outcomes
const perValue = (shots, configs) => (shots >= configs ? num(Math.floor(shots / configs)) : 'under 1');
// The same with its noun: "1 shot", "256 shots", "under 1 shot"
const shotsPer = (shots, configs) => `${perValue(shots, configs)} ${Math.floor(shots / configs) > 1 ? 'shots' : 'shot'}`;
// "A, B and C"
const listOf = (xs) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs.at(-1)}`);
// "QCrank (parallel QPIXL)" → "QCrank"
const shortName = (name) => name.split(' (')[0];

/**
 * Why Tessa's opening: a one-line answer and the sentence that backs it, both built from
 * the table, so each claim is only made when this mesh's numbers support it.
 *
 * Args:
 *   rows: bridge.resources rows, Tessa first.
 *   values: number of values encoded.
 *   shots: shots per readout.
 *
 * Returns:
 *   {answer, detail}: two HTML strings.
 */
function whyAnswer(rows, values, shots) {
  const [t, ...base] = rows;
  const angle = base.find((b) => b.configs === 1); // one qubit per value, read directly
  const packed = base.filter((b) => b !== angle);  // the encodings that pack values into fewer qubits
  const minDepth = Math.min(...packed.map((b) => b.depth));
  const minSwaps = Math.min(...packed.map((b) => b.swaps));
  const minCfg = Math.min(...packed.map((b) => b.configs)); // their best shots per value
  const traits = [t.depth < minDepth && 'shallow', t.swaps === 0 && 'SWAP-free'].filter(Boolean);
  const answer = traits.length
    ? `Tessa fits these ${num(values)} values onto a real chip with a ${traits.join(', ')} circuit.`
    : `Tessa fits these ${num(values)} values onto a real chip, here with depth ${num(t.depth)} and ${num(t.swaps)} SWAPs.`;
  const costs = [
    t.depth < minDepth && (minDepth >= 2 * t.depth ? `at least ${Math.floor(minDepth / t.depth)}× the depth` : `more depth (${num(minDepth)} or more)`),
    minSwaps > t.swaps && `${num(minSwaps)} or more SWAPs`,
    t.configs < minCfg && `give each value at most ${shotsPer(shots, minCfg)} to Tessa's ${perValue(shots, t.configs)}`,
  ].filter(Boolean);
  const fewer = packed.every((b) => b.qubits < t.qubits);
  const names = listOf(packed.map((b) => (b.name === 'Amplitude' ? 'amplitude' : shortName(b.name))));
  const parts = [];
  if (costs.length) {
    parts.push(`${names} ${fewer ? 'use fewer qubits, but ' : ''}need ${listOf(costs.slice(0, 2))}${costs[2] ? `, and ${costs[2]}` : ''}.`);
  }
  if (angle) {
    parts.push(angle.fits
      ? `Angle encoding is shallow but spends one qubit per value (${num(angle.qubits)} here).`
      : `Angle encoding needs one qubit per value: ${num(angle.qubits)}, more than the chip's ${PHOENIX_QUBITS}.`);
  }
  return { answer, detail: parts.join(' ') };
}

/**
 * The scaled-up line: what each encoding would need for a mesh a few times larger.
 *
 * Args:
 *   s: bridge.scaled_resources, or null (spectral: no line).
 *   shots: shots per readout.
 *
 * Returns:
 *   An HTML string ('' for null).
 */
function whyScaled(s, shots) {
  if (!s) return '';
  const items = s.rows.map((x) => {
    const where = x.graph ? ` (${x.graph}${x.runs > 1 ? `, ${x.runs} runs` : ''})` : '';
    const read = x.fits ? `${shotsPer(shots, x.configs)} per value` : `more than the chip's ${s.chip_qubits}`;
    return `<span class="${x.name === 'Tessa' ? 'tessa' : ''}">${shortName(x.name)}: ${num(x.qubits)} qubits${where}, ${read}</span>`;
  });
  return `<p class="scaled"><b>Scaled up ${s.factor}× (${num(s.values)} values)</b> ${items.join(' <i>·</i> ')}</p>`;
}

async function why() {
  await busy('Transpiling Tessa and four baselines onto ibm_phoenix…', async () => {
    const r = await call('resources');
    const shots = 2 ** P.shots;
    const { answer, detail } = whyAnswer(r.rows, r.stats.values, shots);
    // the four columns that make the argument first; the rest (class "extra") behind the toggle
    const rows = r.rows.map((x, k) => `<tr${k ? '' : ' class="tessa"'}><td>${x.name}</td>`
      + `<td>${num(x.qubits)}${x.fits ? '' : ' <span class="flag">won\'t fit</span>'}</td>`
      + `<td>${num(x.swaps)}</td><td>${num(x.depth)}</td><td>${perValue(shots, x.configs)}</td>`
      + `<td class="extra">${x.fits ? 'yes' : 'no'}</td><td class="extra">${num(x.twoq)}</td>`
      + `<td class="extra">${x.settings}</td><td class="extra">${num(x.configs)}</td></tr>`).join('');
    openDialog(`<div class="why"><h2>Why Tessa</h2>
      <p class="answer">${answer}</p>
      <p class="lede">${detail}</p>
      <div class="table-scroll"><table>
        <tr><th>Encoding</th><th>Qubits</th><th>SWAPs</th><th>Depth</th><th>Shots / value</th>
          <th class="extra">Fits on chip</th><th class="extra">2q gates</th><th class="extra">Settings</th><th class="extra">Outcomes / cell</th></tr>
        ${rows}</table></div>
      ${whyScaled(r.scaled, shots)}
      <dl class="col-notes">
        <dt>Qubits</dt><dd>How many the circuit uses. The ibm_phoenix chip has ${PHOENIX_QUBITS}.</dd>
        <dt>SWAPs</dt><dd>Extra moves the chip makes when two qubits that must interact aren't neighbours. Each one adds noise.</dd>
        <dt>Depth</dt><dd>Steps from start to readout once fitted to the chip. Deeper circuits pick up more error.</dd>
        <dt>Shots / value</dt><dd>Of the ${num(shots)} shots, about how many land on each value. More shots give a sharper reading.</dd>
        <dt class="extra">Fits on chip</dt><dd class="extra">Whether it needs no more than the chip's ${PHOENIX_QUBITS} qubits.</dd>
        <dt class="extra">2q gates</dt><dd class="extra">Two-qubit gates after fitting to the chip, the noisiest step.</dd>
        <dt class="extra">Settings</dt><dd class="extra">Measurement settings one readout needs.</dd>
        <dt class="extra">Outcomes / cell</dt><dd class="extra">How many outcomes each shot is split over.</dd>
      </dl>
      <button type="button" class="more" id="whyAll" aria-expanded="false">Show all columns</button>
      <p class="note">${num(r.stats.values)} values (${r.payload}, ${r.mode}), each encoding transpiled onto the ibm_phoenix snapshot (120 qubits, square lattice). Generic random data in every row, no compression. SWAPs ≈ extra CZs from routing ÷ 3. Shots per value = shots ÷ outcomes per cell. Amplitude: Z readout gives magnitudes only. With a chip backend, Tessa's row is on that chip; the baselines stay on Phoenix.${r.scaled ? ' Scaled up counts qubits and outcomes from the sizes alone (no circuits built), so it has no depth or SWAPs.' : ''}</p></div>`);
    $('whyAll').addEventListener('click', (ev) => {
      const all = $('dialogBody').querySelector('.why').classList.toggle('all');
      ev.currentTarget.textContent = all ? 'Show fewer columns' : 'Show all columns';
      ev.currentTarget.setAttribute('aria-expanded', all);
    });
    status('Why Tessa: resource table opened');
  });
}

// ---------- charts (the Curves window)
// A tick label: thousands with separators (2,000 not 2e+3), small values to 2 significant figures
const fmtTick = (v) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : String(+v.toPrecision(Math.abs(v) >= 1 ? 3 : 2)));
// "Nice" ticks for a linear axis (steps of 1, 2 or 5 × a power of ten, about five of them)
function niceStep(range, integer) {
  const raw = range / 5, p = 10 ** Math.floor(Math.log10(raw || 1));
  const step = [1, 2, 5, 10].map((k) => k * p).find((s) => s >= raw) || raw;
  return integer ? Math.max(1, Math.round(step)) : step;
}
// Ticks for a log axis: 1, 2 and 5 × powers of ten inside the range (just 1 and 5, or just
// powers, when that would be too many)
function logTicks(lo, hi) {
  for (const ms of [[1, 2, 5], [1, 5], [1]]) {
    const t = [];
    for (let k = Math.floor(Math.log10(lo)); k <= Math.ceil(Math.log10(hi)); k++) for (const m of ms) { const v = m * 10 ** k; if (v >= lo && v <= hi) t.push(v); }
    if (t.length <= 5) return t.length >= 2 ? t : [lo, hi];
  }
  return [lo, hi];
}
/**
 * One chart as an SVG string.
 *
 * Args:
 *   title, xlabel, ylabel: its heading and axis names.
 *   series: [{label, color, points: [[x, y]], dash?, mark?: 'ring' | 'square' | 'none'}].
 *   logx, logy: log scales.
 *
 * A linear y axis whose values all sit close together (spread under half the largest) starts
 * at zero, so a tiny difference is not stretched to fill the chart.
 */
function chart({ title, xlabel, ylabel, series, logx = false, logy = false }) {
  const W = 420, H = 270, m = { l: 58, r: 14, t: 44, b: 42 };
  const pts = series.flatMap((s) => s.points);
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  let [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  let yt;
  if (logy) yt = logTicks(y0, y1);
  else {
    if (y0 > 0 && (y1 - y0) / y1 < 0.5) y0 = 0;
    const integer = ys.every(Number.isInteger), step = niceStep(y1 - y0 || Math.abs(y1) || 1, integer);
    y0 = Math.floor(y0 / step) * step;
    y1 = Math.ceil(y1 / step) * step;
    if (Math.max(...ys) > y1 - step / 4) y1 += step; // room above the highest point
    yt = []; for (let v = y0; v <= y1 + step / 2; v += step) yt.push(+v.toPrecision(10));
  }
  const xt = logx ? logTicks(x0, x1) : (() => { const st = niceStep(x1 - x0 || 1, xs.every(Number.isInteger)), t = []; for (let v = Math.ceil(x0 / st) * st; v <= x1 + 1e-9; v += st) t.push(+v.toPrecision(10)); return t; })();
  const fx = logx ? Math.log10 : (x) => x, fy = logy ? Math.log10 : (y) => y;
  const X = (x) => m.l + ((fx(x) - fx(x0)) / (fx(x1) - fx(x0) || 1)) * (W - m.l - m.r);
  const Y = (y) => H - m.b - ((fy(y) - fy(y0)) / (fy(y1) - fy(y0) || 1)) * (H - m.t - m.b);
  // mark: 'ring' (hollow circle), 'square' or 'none'; dashed series draw on top, so
  // two series that coincide both stay visible
  const markAt = (s, x, y) => (s.mark === 'none' ? '' : s.mark === 'square'
    ? `<rect x="${x - 3}" y="${y - 3}" width="6" height="6" style="fill: ${s.color}"/>`
    : `<circle cx="${x}" cy="${y}" r="4.5" fill="var(--black)" style="stroke: ${s.color}" stroke-width="1.5"/>`);
  const lines = series.map((s) => `<polyline fill="none" style="stroke: ${s.color}" stroke-width="1.5" ${s.dash ? 'stroke-dasharray="5 4"' : ''} points="${s.points.map((p) => `${X(p[0])},${Y(p[1])}`).join(' ')}"/>`
    + s.points.map((p) => markAt(s, X(p[0]), Y(p[1]))).join('')).join('');
  // legend in a row under the title, each entry as wide as its words
  let lx = m.l;
  const legendRow = series.map((s) => {
    const x = lx;
    lx += 28 + s.label.length * 7 + 22; // about 7 units a letter at --fs-small
    return `<line x1="${x}" y1="31" x2="${x + 22}" y2="31" style="stroke: ${s.color}" stroke-width="1.5" ${s.dash ? 'stroke-dasharray="5 4"' : ''}/>`
      + markAt(s, x + 11, 31) + `<text class="key" x="${x + 28}" y="35">${s.label}</text>`;
  }).join('');
  const xT = xt.map((t) => `<text x="${X(t)}" y="${H - m.b + 16}" text-anchor="middle">${fmtTick(t)}</text>`).join('');
  const yT = yt.map((t) => `<text x="${m.l - 6}" y="${Y(t) + 4}" text-anchor="end">${fmtTick(t)}</text>`).join('');
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}">
    <text class="title" x="${m.l}" y="16">${title}</text>
    <line x1="${m.l}" y1="${H - m.b}" x2="${W - m.r}" y2="${H - m.b}" stroke="currentColor" opacity=".35"/>
    <line x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${H - m.b}" stroke="currentColor" opacity=".35"/>
    ${xT}${yT}${lines}${legendRow}
    <text x="${(W + m.l) / 2}" y="${H - 6}" text-anchor="middle">${xlabel}</text>
    <text transform="translate(14 ${(H - m.b + m.t) / 2}) rotate(-90)" text-anchor="middle">${ylabel}</text></svg>`;
}

// plain-language readings of the curves: what each chart says about this mesh
const pct = (a, b) => (100 * (b - a)) / a;
const sig = (v) => String(+v.toPrecision(2));
function shotsTakeaway(shots, guide) {
  const [s0, e0] = shots[0], [s1, e1] = shots[shots.length - 1], g1 = guide[guide.length - 1][1];
  const noise = e1 / g1 > 0.5 && e1 / g1 < 2;
  return `More shots help: going from ${s0.toLocaleString()} to ${s1.toLocaleString()} shots cuts the error about ${Math.round(e0 / e1)}×.`
    + (noise ? ' It falls in step with the dashed shot-noise line, so the error left is just shot noise: choose shots by how smooth you need the result against how long you can wait.'
      : ' It does not follow the dashed shot-noise line, so something besides shot noise (the backend\'s noise or the encoding) limits the error.');
}
function compressionTakeaway(r) {
  const mesh = r.mesh.map(([c, , e]) => [c, e]), naive = r.naive.map(([c, , e]) => [c, e]);
  const [c1, e1] = mesh[mesh.length - 1], d = pct(mesh[0][1], e1);
  const cost = Math.abs(d) < 0.5 ? `Compression is free here: even at ${c1}% the error doesn't change.`
    : Math.abs(d) < 5 ? `Compression is nearly free here: even at ${c1}% the error changes only ${d >= 0 ? '+' : ''}${d.toFixed(0)}%.`
    : `Compression costs accuracy: at ${c1}% the error rises from ${sig(mesh[0][1])} to ${sig(e1)} (${d >= 0 ? '+' : ''}${d.toFixed(0)}%).`;
  const lay = pct(naive[0][1], mesh[0][1]);
  const layouts = r.same ? ' Both layouts place every value the same way for this mesh, so they give the same result.'
    : Math.abs(lay) < 0.5 ? ' The two layouts give the same error.'
    : Math.abs(lay) < 5 ? ` The two layouts are about equal (the mesh layout's error is ${Math.abs(lay).toFixed(0)}% ${lay < 0 ? 'lower' : 'higher'}).`
      : ` The mesh layout's error is ${Math.abs(lay).toFixed(0)}% ${lay < 0 ? 'lower' : 'higher'} than the naive layout's.`;
  return cost + layouts;
}
function depthTakeaway(r) {
  const d = r.mesh.map(([, dep]) => dep), [a, b] = [d[0], d[d.length - 1]];
  return a === b ? `The circuit stays ${a} step${a === 1 ? '' : 's'} deep at every compression, so compressing won't shorten it for this mesh. (Shorter circuits pick up less noise on real chips.)`
    : `Compressing to ${r.mesh[r.mesh.length - 1][0]}% takes the circuit from ${a} to ${b} steps deep${b < a ? ': shorter circuits pick up less noise on real chips' : ''}.`;
}

async function curves() {
  await busy('Curves: 15 readouts on the current backend…', async () => {
    const r = await call('curves');
    // the app's colours: the measured / mesh line in white, the others in greys, dashed
    const A = 'var(--ink)', B = 'rgb(var(--ink-rgb) / 0.6)', G = 'rgb(var(--ink-rgb) / 0.45)';
    const shots = r.shots.map(([s, e]) => [s, e]);
    const guide = shots.map(([s]) => [s, shots[0][1] * Math.sqrt(shots[0][0] / s)]);
    const err = (k) => r[k].map(([c, , e]) => [c, e]);
    const depth = (k) => r[k].map(([c, d]) => [c, d]);
    // fixed per layout, as in the desktop app: mesh solid with rings, naive dashed with squares
    const layouts = (pick) => [{ label: 'Mesh layout', color: A, points: pick('mesh') },
      { label: 'Naive layout', color: B, dash: true, mark: 'square', points: pick('naive') }];
    const card = (svg, text) => `<figure class="chart">${svg}<figcaption>${text}</figcaption></figure>`;
    const cards = [
      card(chart({ title: 'Error vs Shots', xlabel: 'Shots', ylabel: 'Error (share of size)', logx: true, logy: true,
        series: [{ label: 'Measured', color: A, points: shots }, { label: 'Shot-noise limit', color: G, dash: true, mark: 'none', points: guide }] }), shotsTakeaway(shots, guide)),
      card(chart({ title: 'Error vs Compression', xlabel: 'Compression %', ylabel: 'Error (share of size)', series: layouts(err) }), compressionTakeaway(r)),
      card(chart({ title: 'Depth vs Compression', xlabel: 'Compression %', ylabel: 'Circuit depth (steps)', series: layouts(depth) }), depthTakeaway(r)),
    ];
    // one chart to a page; round arrow buttons (and the arrow keys) slide between pages
    const pages = cards.map((c) => `<div class="chart-page">${c}</div>`);
    const arrow = (dir, d) => `<button class="page-btn" data-dir="${dir}" aria-label="${dir < 0 ? 'Previous' : 'Next'} charts"><svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path d="${d}" fill="none" stroke="currentColor" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round"/></svg></button>`;
    openDialog(`<div class="curves"><h2>Fidelity Curves</h2><p class="lede">How the result's accuracy changes with the settings, for this mesh.</p>
      <div class="carousel">${arrow(-1, 'M10 3.5 5.5 8l4.5 4.5')}<div class="chart-pages"><div class="chart-track">${pages.join('')}</div></div>${arrow(1, 'M6 3.5 10.5 8 6 12.5')}</div>
      <p class="page-count"></p>
      <p class="note">Error is how far the read-back vertices sit from the true ones, on average, as a share of the object's size (0.01 = 1%). Compression throws away the smallest parts of each Tessa cell's data to save gates.</p></div>`);
    let page = 0;
    const body = $('dialogBody'), go = (to) => {
      page = Math.max(0, Math.min(pages.length - 1, to));
      body.querySelector('.chart-track').style.transform = `translateX(${-100 * page}%)`;
      body.querySelector('.page-count').textContent = `${page + 1} / ${pages.length}`;
      body.querySelectorAll('.page-btn').forEach((b) => { b.disabled = (+b.dataset.dir < 0 ? page === 0 : page === pages.length - 1); });
      body.querySelectorAll('.chart-page').forEach((pg, i) => { pg.inert = i !== page; });
    };
    body.querySelectorAll('.page-btn').forEach((b) => b.addEventListener('click', () => go(page + +b.dataset.dir)));
    body.querySelector('.chart-pages').addEventListener('keydown', (e) => { if (e.key === 'ArrowRight') go(page + 1); if (e.key === 'ArrowLeft') go(page - 1); });
    go(0);
    status('Fidelity curves opened');
  });
}

function downloadObj() {
  if (!S.seq.length) return;
  const v = S.seq[S.step], lines = [];
  for (let i = 0; i < v.length; i += 3) lines.push(`v ${v[i].toFixed(6)} ${v[i + 2].toFixed(6)} ${(-v[i + 1]).toFixed(6)}`); // Y-up
  for (const f of S.mesh.faces) lines.push(`f ${f.map((i) => i + 1).join(' ')}`);
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lines.join('\n') + '\n'], { type: 'text/plain' }));
  a.download = `tessa3d-${objLabel().replace(/\.obj$/i, '')}-step-${S.step + 1}.obj`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/**
 * Per-step morph weights that rebuild `timeline(keys, steps, loop)` from keys[0] and
 * the offsets of keys[1..]: in-between s of keys i → i+1 is (1 − e)·key i + e·key i+1,
 * e the smoothstep of s / (steps + 1), and keys[0] (Basis) carries no weight of its own.
 *
 * Args:
 *   n: number of keys (at least 2).
 *   steps: in-betweens after each key (P.between).
 *   loop: also glide from the last key back to the first (P.loop).
 *
 * Returns:
 *   Float32Array of (playback steps) × (n − 1) weights, row-major by step.
 */
function glbWeights(n, steps, loop) {
  const m = n - 1, rows = [], pairs = loop ? n : n - 1;
  const row = (i, e) => {
    const w = new Float32Array(m);
    if (i >= 1) w[i - 1] += 1 - e;
    if (i + 1 >= 1 && i + 1 < n) w[i] += e; // key i+1 is target i; key n (loop) is Basis
    rows.push(w);
  };
  for (let i = 0; i < pairs; i++) {
    for (let s = 0; s <= steps; s++) { const t = s / (steps + 1); row(i, t * t * (3 - 2 * t)); }
  }
  if (!loop) row(n - 1, 0);
  const out = new Float32Array(rows.length * m);
  rows.forEach((w, k) => out.set(w, k * m));
  return out;
}

/**
 * A binary glTF (.glb) of the playback: keys[0] is the mesh, every later key a morph
 * target (a Blender shape key, named in mesh.extras.targetNames), and one animation
 * drives the target weights through the same eased timeline the page plays.
 *
 * Args:
 *   keys: Float32Arrays of N·3 positions in the app's Z-up frame (at least 2).
 *   faces: polygons as vertex-index arrays (fan-triangulated here).
 *   names: one name per key; names[0] is the Basis (the node's name).
 *   steps: in-betweens after each key.
 *   loop: glide from the last key back to the first.
 *   fps: playback steps per second.
 *
 * Returns:
 *   ArrayBuffer holding the .glb file.
 */
function buildGlb(keys, faces, names, steps, loop, fps) {
  const n = keys[0].length / 3, m = keys.length - 1;
  // glTF is Y-up: (x, y, z) -> (x, z, -y), as downloadObj and mesh.write_obj do
  const yUp = (v) => { const o = new Float32Array(v.length); for (let i = 0; i < v.length; i += 3) { o[i] = v[i]; o[i + 1] = v[i + 2]; o[i + 2] = -v[i + 1]; } return o; };
  const base = yUp(keys[0]);
  const deltas = keys.slice(1).map((k) => { const d = yUp(k); for (let j = 0; j < d.length; j++) d[j] -= base[j]; return d; });
  const tris = [];
  for (const f of faces) for (let j = 1; j + 1 < f.length; j++) tris.push(f[0], f[j], f[j + 1]);
  const weights = glbWeights(keys.length, steps, loop), count = weights.length / m;
  const times = Float32Array.from({ length: count }, (_, k) => k / fps);
  const bounds = (v) => {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < v.length; i++) { const c = i % 3; lo[c] = Math.min(lo[c], v[i]); hi[c] = Math.max(hi[c], v[i]); }
    return { min: lo, max: hi };
  };
  // one buffer, one view per array; every array is 4-byte, so views stay aligned
  const parts = [], views = [], accessors = [];
  let offset = 0;
  const add = (arr, accessor, target) => {
    views.push({ buffer: 0, byteOffset: offset, byteLength: arr.byteLength, ...(target ? { target } : {}) });
    accessors.push({ bufferView: views.length - 1, ...accessor });
    parts.push(arr); offset += arr.byteLength;
    return accessors.length - 1;
  };
  const FLOAT = 5126, UINT = 5125, ARRAY = 34962, INDEX = 34963;
  const position = add(base, { componentType: FLOAT, count: n, type: 'VEC3', ...bounds(base) }, ARRAY);
  const indices = add(Uint32Array.from(tris), { componentType: UINT, count: tris.length, type: 'SCALAR' }, INDEX);
  const targets = deltas.map((d) => ({ POSITION: add(d, { componentType: FLOAT, count: n, type: 'VEC3', ...bounds(d) }, ARRAY) }));
  const input = add(times, { componentType: FLOAT, count, type: 'SCALAR', min: [0], max: [times[count - 1]] });
  const output = add(weights, { componentType: FLOAT, count: weights.length, type: 'SCALAR' });
  const gltf = {
    asset: { version: '2.0', generator: 'Tessa3D' },
    scene: 0, scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, name: names[0] }],
    meshes: [{ name: names[0], primitives: [{ attributes: { POSITION: position }, indices, targets, mode: 4 }],
      weights: Array(m).fill(0), extras: { targetNames: names.slice(1) } }],
    animations: [{ name: 'readouts', channels: [{ sampler: 0, target: { node: 0, path: 'weights' } }],
      samplers: [{ input, output, interpolation: 'LINEAR' }] }],
    accessors, bufferViews: views, buffers: [{ byteLength: offset }],
  };
  // GLB: 12-byte header, then a JSON chunk (space-padded) and a BIN chunk (zero-padded)
  const pad4 = (x) => (x + 3) & ~3;
  const json = new TextEncoder().encode(JSON.stringify(gltf));
  const jsonLen = pad4(json.length), binLen = pad4(offset), total = 12 + 8 + jsonLen + 8 + binLen;
  const out = new ArrayBuffer(total), dv = new DataView(out), bytes = new Uint8Array(out);
  dv.setUint32(0, 0x46546c67, true); dv.setUint32(4, 2, true); dv.setUint32(8, total, true); // "glTF", v2
  dv.setUint32(12, jsonLen, true); dv.setUint32(16, 0x4e4f534a, true); // "JSON"
  bytes.fill(0x20, 20, 20 + jsonLen); bytes.set(json, 20);
  let at = 20 + jsonLen;
  dv.setUint32(at, binLen, true); dv.setUint32(at + 4, 0x004e4942, true); at += 8; // "BIN\0"
  for (const p of parts) { bytes.set(new Uint8Array(p.buffer, p.byteOffset, p.byteLength), at); at += p.byteLength; }
  return out;
}

// download .glb: the playback as Blender shape keys (see buildGlb)
function downloadGlb() {
  if (S.keys.length < 2) return;
  const glb = buildGlb(S.keys, S.mesh.faces, S.keyNames, P.between, P.loop, P.fps);
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([glb], { type: 'model/gltf-binary' }));
  const name = (s) => String(s).replace(/\.obj$/i, '');
  a.download = `tessa3d-${name(objLabel())}${S.ts ? `-to-${name(bLabel())}` : ''}.glb`;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---------- qubit map: which data and address qubits hold each vertex
// map.vertices[i][c] = [group, address]: value c of vertex i sits in data qubit
// groups[group].data, in the slot where address qubit groups[group].addresses[j]
// reads bit j of address (Tessa's decoder bit order).
// colours come from the stylesheet's tokens, so the map follows light mode
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
let qmapOn = false;
const mapOn = () => qmapOn && !!S.data;
const comps = () => (S.data.map.vertices[0].length === 1 ? ['r'] : P.payload === 'spherical' ? ['polar', 'azimuth', 'r'] : ['x', 'y', 'z']);
// spectral: map rows are modes, low frequency first (a morph's row 0 is the constant)
const modeName = (r) => { const sp = S.data.spectral, lead = sp.rows - sp.k; return r < lead ? 'constant' : String(r - lead + 1); };

function mapGeometry() {
  const m = S.data.map, xs = m.coords.map((c) => c[0]), ys = m.coords.map((c) => c[1]);
  const nx = Math.max(...xs) + 1, ny = Math.max(...ys) + 1, step = Math.min(250 / nx, 220 / ny);
  return { m, step, w: nx * step, h: ny * step, pos: (q) => [(m.coords[q][0] + 0.5) * step, (m.coords[q][1] + 0.5) * step] };
}

function drawMap() {
  $('qmap').hidden = !mapOn();
  if (!mapOn()) return;
  const { m, step, w, h, pos } = mapGeometry(), c = $('qcanvas'), dpr = devicePixelRatio || 1;
  c.width = w * dpr; c.height = h * dpr; c.style.width = `${w}px`; c.style.height = `${h}px`;
  const ctx = c.getContext('2d');
  const ink = cssVar('--ink-rgb').split(' ').join(', ');
  const [muted, accent, line, panel, HOT] = [`rgba(${ink}, 0.45)`, cssVar('--accent'), `rgba(${ink}, 0.28)`, cssVar('--surface'), cssVar('--orange')];
  ctx.scale(dpr, dpr);
  const f = S.focus, hotGroups = new Set(f ? f.groups : []);
  const hotAddr = new Map(); // address qubit -> bit, when one value is hovered
  for (const g of hotGroups) for (const q of m.groups[g].addresses) hotAddr.set(q, null);
  const hotData = new Set([...hotGroups].map((g) => m.groups[g].data));
  for (const [a, b] of m.edges) {
    const hot = (hotData.has(a) && hotAddr.has(b)) || (hotData.has(b) && hotAddr.has(a));
    ctx.strokeStyle = hot ? HOT : line; ctx.lineWidth = hot ? 2.5 : 1.5;
    ctx.beginPath(); ctx.moveTo(...pos(a)); ctx.lineTo(...pos(b)); ctx.stroke();
  }
  const used = new Set(S.data.map.vertices.flat().map((v) => v[0]));
  const data = new Set(m.groups.map((g) => g.data));
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.font = `400 ${Math.max(8, step * 0.32)}px "RiformaLLWeb-Medium", sans-serif`;
  m.coords.forEach((_, q) => {
    if (data.has(q)) return;
    const [x, y] = pos(q), half = step * (hotAddr.has(q) ? 0.24 : 0.17);
    ctx.fillStyle = hotAddr.has(q) ? HOT : muted;
    ctx.fillRect(x - half, y - half, 2 * half, 2 * half);
    if (hotAddr.get(q) != null) { ctx.fillStyle = panel; ctx.fillText(hotAddr.get(q), x, y + 0.5); }
  });
  m.groups.forEach((g, k) => {
    const [x, y] = pos(g.data);
    ctx.beginPath(); ctx.arc(x, y, step * 0.3, 0, 2 * Math.PI);
    ctx.fillStyle = hotGroups.has(k) ? HOT : used.has(k) ? accent : line;
    ctx.fill();
  });
  $('qtext').textContent = focusText() + '\n● data qubit (2^k slots, k = its address qubits) ■ address qubit\nfilled ● = holds part of this mesh';
}

function focusText() {
  const f = S.focus, m = S.data.map;
  if (!f) return S.data.spectral ? 'Hover a data qubit to see its modes: spectral qubits hold modes of the whole surface, not vertices.'
    : 'Hover a data qubit to light its vertices on the object.';
  const qs = (g) => m.groups[g].addresses.map((q) => `q${q}`).join(' ');
  const g = f.groups[0], held = m.vertices.flat().filter((v) => v[0] === g).length;
  if (S.data.spectral) {
    return `data q${m.groups[g].data} · address ${qs(g)}\n${2 ** m.groups[g].addresses.length} slots, ${held} values (x, y, z of whole-surface modes)\n`
      + (f.modes.length ? `modes ${f.modes.map(modeName).join(', ')}` : 'no modes on it (padding)');
  }
  return `data q${m.groups[g].data} · address ${qs(g)}\n${2 ** m.groups[g].addresses.length} slots, ${held} values of this mesh (${comps().join(', ')})\n${f.mates.length} vertices on it (orange on the object)`;
}

function setFocus(group = null) {
  if (!mapOn() || group === null) S.focus = null;
  else {
    if (S.focus && S.focus.groups[0] === group) return;
    const rows = [];
    S.data.map.vertices.forEach((vs, i) => { if (vs.some((v) => v[0] === group)) rows.push(i); });
    // spectral rows are modes, not vertices: nothing to light on the object
    S.focus = S.data.spectral ? { groups: [group], mates: [], modes: rows } : { groups: [group], mates: rows };
  }
  placeMates();
  drawMap();
}

function placeMates() {
  mates.visible = !!(S.focus && S.seq.length);
  if (!mates.visible) return;
  const v = S.seq[S.step], pts = new Float32Array(S.focus.mates.length * 3);
  S.focus.mates.forEach((i, k) => pts.set(v.subarray(3 * i, 3 * i + 3), 3 * k));
  mateGeom.setAttribute('position', new THREE.BufferAttribute(pts, 3));
  mateGeom.computeBoundingSphere();
}

$('qcanvas').addEventListener('pointermove', (e) => {
  if (!mapOn()) return;
  const { m, step, pos } = mapGeometry(), r = e.currentTarget.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  const k = m.groups.findIndex((g) => Math.hypot(pos(g.data)[0] - x, pos(g.data)[1] - y) < step * 0.45);
  if (k >= 0) setFocus(k);
});
$('qcanvas').addEventListener('pointerleave', () => setFocus(null));

// ====================================================================================
// Game UI: screens, the click-to-open menu, and the settings panels
//   layer 0  home: the object and the run buttons
//   layer 1  after a click on the object: Encode and Decode tabs; clicking one opens its web
//   layer 2  a web of words on the left, every line from one hub beside the object
//   layer 3  one setting's own web: outlined squares, bare words, or sliders
// ====================================================================================
let landedAt = -Infinity; // when the landing last showed: the object fades out over 600 ms, then stops drawing
function setScreen(name) {
  app.dataset.screen = name;
  if (name === 'landing') { closeAll(); S.playing = false; landedAt = performance.now(); }
}

// object offset in stage px (left of, or in portrait above, the morph drop box)
function setShift(x, y) { app.style.setProperty('--ox', `${x}px`); app.style.setProperty('--oy', `${y}px`); }
let shift = [0, 0];
// the morph box sits right of the object (design) or, in portrait, below it
const MORPH_SHIFT = () => (portrait ? [0, -100] : [-240, 10]);
const morphBoxOpen = () => !panelEl.hidden && panelEl.dataset.key === 'mode' && P.mode === 'morph';
// On a phone the menus' webs need the left of the screen, so while one is open (layers 2
// and 3) the object slides right, partly off the edge.
const PORTRAIT_MENU_SHIFT = 230;
/**
 * Move the object for the morph box and, on portrait screens, for an open menu web.
 *
 * Args:
 *   layer: the menu layer to lay out for (defaults to the current one; openArc passes
 *     the layer it is about to show, so the web is laid out round the object's new place).
 */
function syncShift(layer = +home.dataset.layer) {
  shift = morphBoxOpen() ? MORPH_SHIFT() : [0, 0];
  if (portrait && layer >= 2) shift = [shift[0] + PORTRAIT_MENU_SHIFT, shift[1]];
  setShift(...shift);
}

// ---------- geometry
const rad = THREE.MathUtils.degToRad;
const svgNS = 'http://www.w3.org/2000/svg';
// Layer 1: two pill tabs, Encode and Decode, laid out like the first and last of layer 2's
// five words (Compression's and Load's places): on lines from the same hub, WEB.vstep2 × 2
// above and below it. layoutTabs fills in each pill's box (`poly`, for hit tests).
const TABS = { encode: { poly: [] }, decode: { poly: [] } };
// Layers 2 and 3 are webs: every option sits at the end of a thin straight line from one
// hub just left of the object, level with the centre of the right-hand tools (both sit
// at the window's mid-height, stage y = CY). Angles are in degrees: 180 points straight
// left, less points up, more points down. Encode's fan leans up and decode's down; each
// is spaced evenly and symmetrically about its centre line.
const WEB = {
  dx: 330, // the hub: this far left of the object's centre (clear of the object)
  vstep2: 66, r2: 270, // layer 2: pills, vstep2 apart top to bottom, the middle one level with the hub
  fan3: { encode: 180, decode: 180 }, step3: 46, span3: 124, // layer 3 with sliders: centred on the hub, so a pair sits symmetrically above and below
  rText: 230, rSquare: 190, rElbow: 150, track: 230, gap: 12,
};
const dirOf = (a) => [Math.cos(rad(a)), -Math.sin(rad(a))];
function fanAngles(n, centre, step, span = Infinity) {
  const s = n > 1 ? Math.min(step, span / (n - 1)) : 0;
  return [...Array(n).keys()].map((i) => centre + (i - (n - 1) / 2) * s);
}
/**
 * Lay out a fan of entries round the hub.
 *
 * Args:
 *   entries: per option, either a box {w, h, r, gap} (its line ends at radius r and the
 *     box sits gap beyond that, to its left or, for a steep line, above or below) or a slider {elbow: true, left} (its line
 *     bends at WEB.rElbow and runs `left` stage px straight left from there).
 *   angles: one direction per entry, in degrees (180 = straight left).
 *
 * Returns:
 *   {hub, at}: the hub point and, per entry, {end, c}: where its line ends and (boxes
 *   only) the box's centre. The hub moves right if the fan would cross the window's left edge.
 */
function fanLayout(entries, angles) {
  const [hx0, hy] = [CX + shift[0] - WEB.dx, CY];
  const rel = entries.map((e, i) => {
    if (e.elbow) return null;
    const [dx, dy] = dirOf(angles[i]), end = [dx * e.r, dy * e.r];
    // the line meets the middle of the box's right edge, or of its top or bottom edge when
    // the line is steep (within ~30° of vertical), so neighbours never overlap
    const c = Math.abs(dy) < 0.86 ? [end[0] - e.gap - e.w / 2, end[1]] : [end[0], end[1] + Math.sign(dy) * (e.gap + e.h / 2)];
    return { end, c, minX: c[0] - e.w / 2 };
  });
  // A slider's track runs straight left from its elbow, so the elbow sits deeper (further
  // above or below the hub) than every option on its side: the track never crosses a line.
  entries.forEach((e, i) => {
    if (!e.elbow) return;
    const [dx, dy] = dirOf(angles[i]);
    const depth = Math.max(0, ...rel.filter((r) => r?.c && Math.sign(r.c[1]) === Math.sign(dy)).map((r) => Math.abs(r.c[1]) + 24));
    const rE = Math.max(WEB.rElbow, Math.abs(dy) > 0.05 ? (depth + 40) / Math.abs(dy) : 0);
    const end = [dx * rE, dy * rE];
    rel[i] = { end, minX: end[0] - e.left };
  });
  const vl = Number.isFinite(layout.vl) ? layout.vl : -Infinity; // NaN while the window has no size
  // keep the fan clear of the window's left edge and of the back button there (its left
  // edge at --edge, 30px wide: the CSS clamp(14px, 5vw, 72px), in window px), plus a gap
  const edge = Math.min(72, Math.max(14, innerWidth * 0.05));
  const clear = Number.isFinite(scale) && scale > 0 ? (edge + 30 + 24) / scale : 28;
  const hx = Math.max(hx0, vl + clear - Math.min(...rel.map((r) => r.minX)));
  return { hub: [hx, hy], at: rel.map((r) => ({ end: [hx + r.end[0], hy + r.end[1]], c: r.c && [hx + r.c[0], hy + r.c[1]] })) };
}
/**
 * Draw the web's lines (hub to each end, drawn outward one after another) and its hub dot.
 *
 * Args:
 *   group: the SVG <g> to fill (emptied first).
 *   hub: the hub point, stage px.
 *   ends: each line's far end, stage px.
 */
function webLines(group, hub, ends) {
  group.replaceChildren();
  ends.forEach((end, i) => {
    const l = document.createElementNS(svgNS, 'line');
    l.setAttribute('class', 'web-line');
    l.setAttribute('pathLength', 1);
    for (const [k, v] of [['x1', hub[0]], ['y1', hub[1]], ['x2', end[0]], ['y2', end[1]]]) l.setAttribute(k, v.toFixed(1));
    l.style.setProperty('--d', `${55 * i}ms`);
    group.appendChild(l);
  });
  const dot = document.createElementNS(svgNS, 'circle');
  dot.setAttribute('class', 'web-hub');
  dot.setAttribute('cx', hub[0].toFixed(1)); dot.setAttribute('cy', hub[1].toFixed(1)); dot.setAttribute('r', 2.5);
  group.appendChild(dot);
}

/**
 * Lay out one side's layer 2 web: its words fanned round the hub on the left.
 *
 * Args:
 *   side: 'encode' or 'decode'.
 */
function layoutArc(side) {
  const labels = [...$(`${side}Arc`).querySelectorAll('.arc-label')];
  if (side === 'encode') labels.reverse(); // mirrored: each list starts nearest the horizontal
  // evenly spaced top to bottom (not by angle), centred on the hub: the middle pill sits
  // straight left of it. Each line still has length r2, so the pills follow a gentle arc.
  const angles = labels.map((_, i) => 180 + THREE.MathUtils.radToDeg(Math.asin(((i - (labels.length - 1) / 2) * WEB.vstep2) / WEB.r2)));
  const { hub, at } = fanLayout(labels.map((b) => ({ w: b.offsetWidth, h: b.offsetHeight, r: WEB.r2, gap: WEB.gap })), angles);
  labels.forEach((b, i) => {
    Object.assign(b.style, { left: `${at[i].c[0]}px`, top: `${at[i].c[1]}px` });
    b.style.setProperty('--d', `${120 + 80 * i}ms`);
  });
  webLines($(`${side}Web`), hub, at.map((p) => p.end));
}

/** Lay out layer 1's tabs where layer 2's top and bottom words go, with lines from the hub. */
function layoutTabs() {
  const h = 18 * 1.2 + 16; // padded like "Click for settings" (8px × 20px, 1.2-line text)
  const sides = ['encode', 'decode'];
  // each tab sits where its web's outer word goes: (words - 1) / 2 steps out (encode up, decode down)
  const dy = sides.map((side) => (($(`${side}Arc`).querySelectorAll('.arc-label').length - 1) / 2) * WEB.vstep2);
  const boxes = sides.map((side) => ({ w: $(`${side}Pill`).offsetWidth + 40, h, r: WEB.r2, gap: WEB.gap }));
  const angles = [-dy[0], dy[1]].map((d) => 180 + THREE.MathUtils.radToDeg(Math.asin(d / WEB.r2)));
  const { hub, at } = fanLayout(boxes, angles);
  sides.forEach((side, i) => {
    const [x, y] = at[i].c, { w } = boxes[i], r = h / 2;
    const [l, t, rt, b] = [x - w / 2, y - h / 2, x + w / 2, y + h / 2];
    TABS[side].poly = [[l, t], [rt, t], [rt, b], [l, b]];
    $(`${side}PillBand`).setAttribute('d', `M ${l + r} ${t} H ${rt - r} A ${r} ${r} 0 0 1 ${rt - r} ${b} H ${l + r} A ${r} ${r} 0 0 1 ${l + r} ${t} Z`);
    $(`${side}Lead`).setAttribute('d', `M ${hub.map((v) => v.toFixed(1)).join(' ')} L ${at[i].end.map((v) => v.toFixed(1)).join(' ')}`);
    Object.assign($(`${side}Pill`).style, { left: `${x}px`, top: `${y}px` });
  });
  $('tabHub').setAttribute('cx', hub[0].toFixed(1)); $('tabHub').setAttribute('cy', hub[1].toFixed(1));
}
/** Lay out layer 1's tabs and both layer 2 webs. */
function drawMenuShapes() {
  layoutTabs();
  for (const side of ['encode', 'decode']) layoutArc(side);
}
drawMenuShapes();
document.fonts?.ready.then(drawMenuShapes); // re-measure the words once Riforma has loaded

/** Re-lay the open web after the window (and so the visible stage) changes size. */
function relayoutMenus() {
  const st = menuEl.dataset.state;
  layoutTabs();
  if (st === 'encode' || st === 'decode') layoutArc(st);
  layoutPanel();
}


// menu state: closed | open (layer 1) | encode / decode (layer 2) | panel (layer 3)
function setMenu(state) {
  menuEl.dataset.state = state;
  home.dataset.layer = app.dataset.layer = { closed: 0, open: 1, encode: 2, decode: 2, panel: 3 }[state]; // the page frame reads it too
  if (state !== 'open') hotPill(null);
  syncShift();
}
let menuGen = 0; // bumped by every transition, so a superseded animation stops quietly

/**
 * Layer 1 → 2: the chosen side's web draws out from its hub.
 *
 * Args:
 *   side: 'encode' or 'decode'.
 *   fromKeyboard: focus the first word (opened with Enter or Space).
 */
function openArc(side, fromKeyboard = false) {
  ++menuGen;
  syncShift(2);
  layoutArc(side);
  setMenu(side);
  if (fromKeyboard) $(`${side}Arc`).querySelector('.arc-label')?.focus({ preventScroll: true });
}
/** Layer 2 → home: the web folds back into its hub. */
/**
 * Layer 2 → 3: the words give way to one setting's own web.
 *
 * Args:
 *   side: 'encode' or 'decode'.
 *   key: the setting, a key of PANELS.
 */
function openPanel(side, key) {
  ++menuGen;
  panelEl.dataset.side = side;
  panelEl.dataset.key = key;
  panelEl.hidden = false;
  setMenu('panel');
  buildPanel(side, key);
  requestAnimationFrame(() => { panelEl.classList.add('shown'); syncMorphBox(); });
}
/** Layer 3 → 2: the setting's web folds and the words come back. */
function backToArc() {
  const side = panelEl.dataset.side;
  ++menuGen;
  closePanel();
  layoutArc(side);
  setMenu(side);
}
function goBack() {
  const st = menuEl.dataset.state;
  if (st === 'panel') backToArc();
  else if (st === 'encode' || st === 'decode') { ++menuGen; setMenu('open'); } // back to the Encode / Decode tabs
  else if (st === 'open') setMenu('closed');
}
$('back').addEventListener('click', goBack);
addEventListener('keydown', (e) => {
  if (e.key !== 'Escape' || app.dataset.screen !== 'home' || $('dialog').open) return;
  if (mapOn()) setMapOpen(false); else goBack(); // Escape closes the qubit map first
});

function closePanel(now = false) {
  if (panelEl.hidden) return;
  panelEl.classList.remove('shown');
  $('dropB').classList.remove('shown');
  $('morphOpts').classList.remove('shown');
  const done = () => { if (!panelEl.classList.contains('shown')) { panelEl.hidden = true; $('dropB').hidden = true; $('morphOpts').hidden = true; syncShift(); } };
  if (now) done(); else setTimeout(done, 380);
  panelEl.dataset.key = '';
  syncShift();
}
function closeAll() {
  ++menuGen;
  closePanel(true);
  setMenu('closed');
}

// ---------- layer 1: click (or tap) the object to open the Encode and Decode tabs; click
// a tab to open its arc, anywhere else to close. A drag rotates the object instead.
const CLICK_SLOP = 8; // px a press may move and still count as a click
let press = null; // the pointerdown a click may complete: {id, x, y}
const objDist = (sx, sy) => Math.hypot(sx - (CX + shift[0]), sy - (CY + shift[1]));
const insideUi = (el) => !!el.closest?.('.pill-label, .arc-label, #back, .orb, .knob, .line-slider .hit, #dropB, #morphOpts, .run-buttons, #hint, #help, #hud, #drawer, #qmap, #info, #resRow, #side, dialog, .logo');
// which pill (if any) is at stage point (sx, sy)
function pillAt(sx, sy) {
  const PAD = 16; // a forgiving margin round each tab
  for (const side of ['encode', 'decode']) {
    const xs = TABS[side].poly.map((q) => q[0]), ys = TABS[side].poly.map((q) => q[1]);
    if (sx >= Math.min(...xs) - PAD && sx <= Math.max(...xs) + PAD && sy >= Math.min(...ys) - PAD && sy <= Math.max(...ys) + PAD) return side;
  }
  return null;
}
function hotPill(side) { for (const s of ['encode', 'decode']) { $(`${s}PillBand`).classList.toggle('hot', s === side); $(`${s}Pill`).classList.toggle('hot', s === side); } }

addEventListener('pointerdown', (e) => {
  press = app.dataset.screen === 'home' && e.button === 0 && !insideUi(e.target) ? { id: e.pointerId, x: e.clientX, y: e.clientY } : null;
});
addEventListener('pointerup', (e) => {
  const p = press;
  press = null;
  if (!p || e.pointerId !== p.id || Math.hypot(e.clientX - p.x, e.clientY - p.y) > CLICK_SLOP) return; // a drag: it rotated
  const [sx, sy] = toStage(e.clientX, e.clientY), st = menuEl.dataset.state;
  if (st === 'closed') {
    if (objDist(sx, sy) < OBJ_R) setMenu('open');
  } else if (st === 'open') {
    const side = pillAt(sx, sy); // a tab opens its arc
    if (side) openArc(side);
  }
  // an open menu (any layer) stays open on a click elsewhere: only the back button (or Escape) steps back
});
addEventListener('pointercancel', () => { press = null; });
// with the tabs open, light the one under the pointer
addEventListener('pointermove', (e) => {
  if (menuEl.dataset.state === 'open' && e.pointerType === 'mouse') hotPill(pillAt(...toStage(e.clientX, e.clientY)));
});

// keyboard (and a plain click on the hint): open layer 1, then pick a pill
$('hint').addEventListener('click', () => { if (menuEl.dataset.state === 'closed') { setMenu('open'); $('encodePill').focus(); } });
for (const pill of document.querySelectorAll('.pill-label')) pill.addEventListener('click', (e) => { if (menuEl.dataset.state === 'open') openArc(pill.dataset.side, e.detail === 0); }); // detail 0: Enter / Space
for (const label of document.querySelectorAll('.arc-label')) {
  label.addEventListener('click', () => openPanel(label.closest('.arc').classList.contains('encode') ? 'encode' : 'decode', label.dataset.panel));
}

// ---------- panel contents
// Line icons for the Load options, drawn like the round buttons' symbols: white strokes
// (currentColor, so a hover turns them dark), each a qubit drawn as a sphere outline
const qicon = (inner) => `<svg class="qicon" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor"`
  + ` stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/>${inner}</svg>`;
const iconDot = (x, y) => `<circle cx="${x}" cy="${y}" r="1.3" fill="currentColor" stroke="none"/>`;
const ICONS = {
  // one value per vertex: its distance from the centre, a single radius
  radial: qicon(`<path d="M12 12 18.2 5.8"/>${iconDot(12, 12)}${iconDot(18.2, 5.8)}`),
  // a direction in θ and φ: the equator and a vector tipped out from the pole
  spherical: qicon(`<ellipse cx="12" cy="12" rx="10" ry="3.2"/><path d="M12 12V3.5M12 12l5.2 4.6"/>${iconDot(17.2, 16.6)}`),
  // the mesh's smooth surface modes: a wave across the sphere
  spectral: qicon('<path d="M3.2 12c1.5-3.6 3-3.6 4.4 0s2.9 3.6 4.4 0 2.9-3.6 4.4 0 2.9 3.6 4.4 0"/>'),
};
const clampFmt = (v) => (v ? `${v.toFixed(2)}π` : P.mode === 'morph' ? '0.20π' : 'off');
const PANELS = {
  load: { title: 'load', key: 'payload', squares: true, items: [
    { v: 'xyz', html: '(x,y,z)', cls: 'label', name: 'xyz', tip: 'three values per vertex' },
    { v: 'radial', html: ICONS.radial, name: 'Radial', tip: 'one value per vertex (distance from the centre)' },
    { v: 'spherical', html: ICONS.spherical, name: 'Spherical', tip: 'direction in θ and φ, radius in θ' },
    { v: 'spectral', html: ICONS.spectral, name: 'Spectral', tip: 'the mesh\'s first k surface modes in x, y and z (smooth noise, one run)' }],
  sliders: [{ key: 'spectral_k', label: 'modes k', min: 0, max: 300, step: 1, fmt: (v) => (v ? `${v}` : 'auto'),
    show: () => P.payload === 'spectral' }] },
  mode: { title: 'mode', key: 'mode', squares: true, items: [
    { v: 'theta', html: 'θ', cls: 'greek', tip: 'shape in θ, read in Z' },
    { v: 'split', html: 'θ+φ', cls: 'greek', tip: 'shape over θ and φ: two values per slot' },
    { v: 'morph', html: 'Morph', cls: 'label', tip: 'A in θ, B in φ, read along a basis sweep' }] },
  layout: { title: 'vert layout', key: 'layout', items: [
    { v: 'mesh', html: 'mesh', tip: 'neighbouring vertices share data qubits' },
    { v: 'naive', html: 'naive', tip: 'values in file order' }] },
  // Morph uses one clamp for both fields (bridge.morph_clamp: φ follows θ, so equal
  // geometry gets equal angles), so there the two knobs move as one; 0 is Tessa's
  // pair default, 0.2π.
  clamps: { title: 'clamps', sliders: [
    { key: 'clamp', label: () => (P.mode === 'morph' ? 'clamp θ (= φ)' : 'clamp θ'), min: 0, max: 0.45, step: 0.01, fmt: clampFmt, link: 'clamp_b' },
    { key: 'clamp_b', label: () => (P.mode === 'morph' ? 'clamp φ (= θ)' : 'clamp φ'), min: 0, max: 0.45, step: 0.01, fmt: clampFmt, link: 'clamp',
      off: () => P.mode === 'theta' && 'φ clamp: θ+φ and morph modes only' }] },
  compression: { title: 'compression', sliders: [
    { key: 'compression', label: 'compression', min: 0, max: 100, step: 1, fmt: (v) => `${v}%` },
    { key: 'quant_bits', label: 'quant bits', min: 0, max: 8, step: 1, fmt: (v) => (v ? `${v} bit${v > 1 ? 's' : ''}` : 'off') }] },
  drr: { title: 'DRR', key: 'drr', items: ['none', 'min_max', 'mean_std', 'percentile', 'abs_max', 'min_avg_max'].map((v) => (
    { v, html: v.replace(/_/g, ' '), tip: `Tessa's ${v} dynamic-range restoration (from the input's statistics)` })).concat(
    { v: 'pilots', html: 'pilots', tip: 'known values in every spare slot, read with the mesh: the correction is measured on the chip (mode θ)' }) },
  // hints: each option's tip also shows in the bottom-left corner on hover, as for load and mode
  readout: { title: 'readout', key: 'readout', hints: true, items: [
    { v: 'z', html: 'Z', tip: 'reads θ in Z: one measurement setting' },
    { v: 'bloch', html: 'Bloch', tip: 'reads θ in Z, X and Y as a direction: noise shrinks it, not its angle (θ mode only)' },
    { v: 'snap', html: 'snap', tip: '64 points on the Bloch sphere, read back to the nearest: exact while noise allows (θ mode only)' }] },
  frames: { title: 'frames', sliders: [
    { key: 'frames', label: 'frames', min: 2, max: 30, step: 1, fmt: (v) => `${v}` },
    // playback only (no new run needed): steps per second, as in view → speed
    { key: 'fps', label: 'speed (fps)', min: 1, max: 60, step: 1, fmt: (v) => `${v} fps` }] },
  shots: { title: 'shots', sliders: [
    { key: 'shots', label: 'shots', min: 8, max: 16, step: 1, fmt: (v) => (2 ** v).toLocaleString() }] },
  placement: { title: 'chip placement', key: 'placement', items: [
    { v: 'quality', html: 'best qubits first', tip: 'the chip\'s best qubits hold the data' },
    { v: 'graph', html: 'Tessa\'s order', tip: 'Tessa\'s own qubit order' }] },
  backend: { title: 'backend', key: 'backend', items: [
    { v: 'noiseless', html: 'noiseless', tip: 'ideal simulator: shot noise only' },
    { v: 'flat noise', html: 'flat noise', tip: 'IBM-like flat error rates' },
    { v: 'phoenix', html: 'phoenix', tip: 'ibm_phoenix calibration snapshot' },
    { v: 'fez', html: 'fez', tip: 'ibm_fez calibration snapshot' }],
  sliders: [{ key: 'noise_strength', label: 'noise ×', min: 0, max: 3, step: 0.1, fmt: (v) => `${v.toFixed(1)}×`, show: () => P.backend === 'flat noise' }] },
};

/**
 * Build layer 3 for one setting: its options (outlined squares for shapes, bare words
 * for text) and sliders, each at the end of a line from the hub.
 *
 * Args:
 *   side: 'encode' or 'decode' (which way the fan leans).
 *   key: the setting, a key of PANELS.
 */
function buildPanel(side, key) {
  const spec = PANELS[key], body = $('panelBody');
  body.innerHTML = '';
  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('class', 'web-svg'); svg.setAttribute('viewBox', '0 0 1440 1024');
  svg.innerHTML = '<g class="web"></g>';
  body.appendChild(svg);
  for (const it of spec.items || []) {
    const b = document.createElement('button');
    b.className = `orb ${spec.squares ? 'square' : 'word'} ${it.cls || ''}`;
    // a word option starts with a capital (sentence case); graphics and symbols are left alone
    b.innerHTML = !spec.squares && !it.html.includes('<') ? it.html[0].toUpperCase() + it.html.slice(1) : it.html;
    b.setAttribute('aria-label', it.name ? `${it.name}: ${it.tip}` : it.tip); // pictures have no words: name them for screen readers
    if (spec.squares || spec.hints) b.dataset.tip = it.tip; // a graphic (or a hinted word): its tip shows on hover
    b.dataset.v = it.v;
    b.addEventListener('click', () => choose(spec.key, it.v));
    body.appendChild(b);
  }
  for (const sl of spec.sliders || []) body.appendChild(lineSlider(sl));
  syncPanel();
}

/** Lay out the open setting's web (again after a resize, a morph shift or a slider showing). */
function layoutPanel() {
  if (panelEl.hidden || !PANELS[panelEl.dataset.key]) return;
  const spec = PANELS[panelEl.dataset.key], body = $('panelBody');
  const orbs = [...body.querySelectorAll('.orb')];
  const sliders = [...body.querySelectorAll('.slider-wrap')].filter((w) => w.style.display !== 'none');
  const entries = [
    ...orbs.map((el) => ({ el, w: el.offsetWidth, h: el.offsetHeight, r: spec.squares ? WEB.rSquare : WEB.rText, gap: spec.squares ? 0 : WEB.gap })),
    ...sliders.map((el) => ({ el, elbow: true, left: WEB.track + 80 })),
  ];
  // options alone sit like layer 2's words: evenly spaced top to bottom, centred on the hub
  // (the middle one straight left of it), at least WEB.vstep2 apart; with a slider among
  // them the fan is spread by angle instead, so each track clears the lines
  let angles;
  if (!sliders.length) {
    const step = Math.max(WEB.vstep2, ...entries.map((e) => e.h + 24));
    angles = entries.map((e, i) => 180 + THREE.MathUtils.radToDeg(Math.asin(Math.max(-0.95, Math.min(0.95, ((i - (entries.length - 1) / 2) * step) / e.r)))));
  } else angles = fanAngles(entries.length, WEB.fan3[panelEl.dataset.side], WEB.step3, WEB.span3);
  const { hub, at } = fanLayout(entries, angles);
  entries.forEach((e, i) => {
    const d = `${120 + 80 * i}ms`;
    if (e.elbow) e.el.place(at[i].end, d);
    else { Object.assign(e.el.style, { left: `${at[i].c[0]}px`, top: `${at[i].c[1]}px` }); e.el.style.setProperty('--d', d); }
  });
  webLines(body.querySelector('.web'), hub, at.map((p) => p.end));
}

function syncPanel() {
  const key = panelEl.dataset.key, spec = PANELS[key];
  if (!spec) return;
  for (const b of $('panelBody').querySelectorAll('.orb')) {
    const on = String(P[spec.key]) === b.dataset.v;
    b.classList.toggle('on', on);
    b.setAttribute('aria-pressed', on);
  }
  for (const el of $('panelBody').querySelectorAll('.slider-wrap')) el.sync();
  layoutPanel(); // a slider may have shown or hidden (backend: the noise strength)
}

function choose(key, v) {
  P[key] = v;
  // the spherical payload fills θ and φ itself, so it only pairs with mode θ
  if (key === 'payload' && v === 'spherical' && P.mode !== 'theta') { P.mode = 'theta'; status('spherical fills θ and φ itself: mode set to θ'); }
  if (key === 'mode' && v !== 'theta' && P.payload === 'spherical') { P.payload = 'radial'; status('θ+φ modes need a radial or xyz load: load set to radial'); }
  // the spectral load's qubits hold modes of the whole surface, so there is no patch for a front to turn
  if (key === 'payload' && v === 'spectral' && P.morph_front) { P.morph_front = false; status('the morph front needs one row per vertex: front turned off for the spectral load'); }
  if (key === 'payload') syncMorphBox();
  if (key === 'mode' && v === 'morph') P.clamp_b = P.clamp; // one clamp for both fields
  if (key === 'mode') { syncMorphBox(); syncHud(); }
  if (key === 'backend') {
    if (v !== 'noiseless' && P.interference) { P.interference = false; status('interference readout needs the noiseless backend on the web: turned it off'); }
    syncMorphBox(); syncHud();
  }
  syncPanel(); syncStale();
}

// ---------- sliders for continuous settings: the web's line bends at an elbow and runs
// straight left as the track (minimum at the far end, maximum at the elbow)
/**
 * Make one slider.
 *
 * Args:
 *   sl: its spec from PANELS (key, label, min, max, step, fmt, and optional link, off, show).
 *
 * Returns:
 *   The slider's wrapper element, with sync() (redraw from P) and place(elbow, delay)
 *   (move it to the end of its web line).
 */
function lineSlider(sl) {
  const wrap = document.createElement('div');
  wrap.className = 'slider-wrap';
  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('class', 'line-slider'); svg.setAttribute('viewBox', '0 0 1440 1024');
  const mk = (tag, cls) => { const el = document.createElementNS(svgNS, tag); el.setAttribute('class', cls); svg.appendChild(el); return el; };
  const track = mk('path', 'track'), fill = mk('path', 'fill'), hit = mk('path', 'hit');
  const label = mk('text', 'slider-label'), value = mk('text', 'slider-value');
  value.setAttribute('text-anchor', 'end');
  const knob = document.createElement('button');
  knob.className = 'knob';
  knob.setAttribute('role', 'slider');
  knob.setAttribute('aria-valuemin', sl.min); knob.setAttribute('aria-valuemax', sl.max);
  wrap.append(svg, knob);
  const labelText = () => { const t = typeof sl.label === 'function' ? sl.label() : sl.label; return t[0].toUpperCase() + t.slice(1); }; // sentence case
  let T = [0, 0]; // the track's left end; it runs WEB.track px right to the elbow
  wrap.place = (elbow, delay) => {
    T = [elbow[0] - WEB.track, elbow[1]];
    const [x, y] = T;
    // the track, with a short upright tick at each end: the minimum (left) and the maximum,
    // where the track meets the line in from the hub
    track.setAttribute('d', `M ${x} ${y} H ${elbow[0]} M ${x} ${y - 6} V ${y + 6} M ${elbow[0]} ${y - 6} V ${y + 6}`);
    hit.setAttribute('d', `M ${x} ${y} H ${elbow[0]}`);
    // left edge in line with the knob's (14px square centred on the track's end); on the
    // outside of the fan: above the knob for a slider above the hub, below it for one below,
    // with the same 13px gap either way (18px type: capitals 12.6px tall)
    label.setAttribute('x', x - 7); label.setAttribute('y', y > CY ? y + 7 + 13 + 12.6 : y - 7 - 13);
    value.setAttribute('x', x - 14); value.setAttribute('y', y + 5);
    wrap.style.setProperty('--d', delay);
    wrap.sync();
  };
  const set = (v) => {
    v = Math.min(sl.max, Math.max(sl.min, Math.round(v / sl.step) * sl.step));
    v = +v.toFixed(4);
    if (v !== P[sl.key]) {
      P[sl.key] = v;
      if (sl.link && P.mode === 'morph') { P[sl.link] = v; syncPanel(); } // morph: θ and φ clamps move as one
      syncStale(); syncDrawer();
    }
    wrap.sync();
  };
  wrap.sync = () => {
    wrap.style.display = !sl.show || sl.show() ? '' : 'none';
    const off = sl.off && sl.off();
    wrap.style.opacity = off ? 0.35 : '';
    knob.title = off || '';
    label.textContent = labelText();
    knob.setAttribute('aria-label', labelText());
    const x = T[0] + (WEB.track * (P[sl.key] - sl.min)) / (sl.max - sl.min);
    knob.style.left = `${x}px`; knob.style.top = `${T[1]}px`;
    fill.setAttribute('d', `M ${T[0]} ${T[1]} H ${x}`);
    value.textContent = sl.fmt(P[sl.key]);
    knob.setAttribute('aria-valuenow', P[sl.key]);
    knob.setAttribute('aria-valuetext', sl.fmt(P[sl.key]));
  };
  const fromPointer = (e) => set(sl.min + ((toStage(e.clientX, e.clientY)[0] - T[0]) / WEB.track) * (sl.max - sl.min));
  const drag = (e) => {
    e.preventDefault();
    knob.classList.add('dragging');
    fromPointer(e);
    const move = (ev) => fromPointer(ev);
    const up = () => { knob.classList.remove('dragging'); removeEventListener('pointermove', move); removeEventListener('pointerup', up); };
    addEventListener('pointermove', move);
    addEventListener('pointerup', up);
  };
  knob.addEventListener('pointerdown', drag);
  hit.addEventListener('pointerdown', drag);
  knob.addEventListener('keydown', (e) => {
    const d = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[e.key];
    if (d) { e.preventDefault(); set(P[sl.key] + d * sl.step); }
  });
  wrap.sync();
  return wrap;
}

// ---------- morph: the drop box for mesh B
/**
 * Place the morph block (the drop box and the options under it) as one unit: centred on the
 * window's middle height, and halfway across the gap between the object's right edge and the
 * tools on the right (their open position). Stage px. Phones use the stylesheet's layout.
 */
const MORPH_W = 484, MORPH_BOX_H = 300, MORPH_GAP = 24, OBJ_FIT_R = 275;
function placeMorphBox() {
  const box = $('dropB'), opts = $('morphOpts');
  // the upper chips sit as far apart as the spread-out lower ones
  const low = [...$('chipsBottom').children], used = low.reduce((w, c) => w + c.offsetWidth, 0);
  if (low.length > 1 && $('chipsBottom').offsetWidth) opts.style.setProperty('--chip-gap', `${($('chipsBottom').offsetWidth - used) / (low.length - 1)}px`);
  if (portrait) { box.style.left = box.style.top = opts.style.left = opts.style.top = ''; return; }
  const objRight = CX + shift[0] + OBJ_FIT_R;
  const hud = $('hud'), toolsLeft = toStage(innerWidth - parseFloat(getComputedStyle(hud).marginRight) - hud.offsetWidth, 0)[0];
  const left = Math.max(objRight + MORPH_GAP, (objRight + toolsLeft) / 2 - MORPH_W / 2);
  const top = CY - (MORPH_BOX_H + MORPH_GAP + opts.offsetHeight) / 2;
  Object.assign(box.style, { left: `${left}px`, top: `${top}px` });
  Object.assign(opts.style, { left: `${left}px`, top: `${top + MORPH_BOX_H + MORPH_GAP}px` });
}
function syncMorphBox() {
  const on = !panelEl.hidden && panelEl.dataset.key === 'mode' && P.mode === 'morph';
  const box = $('dropB');
  if (on) { box.hidden = false; requestAnimationFrame(() => box.classList.add('shown')); } else box.classList.remove('shown');
  box.classList.toggle('ready', !!bLabel());
  const opts = $('morphOpts');
  if (on) { opts.hidden = false; requestAnimationFrame(() => opts.classList.add('shown')); } else opts.classList.remove('shown');
  for (const c of opts.querySelectorAll('.chip')) c.setAttribute('aria-pressed', P[c.dataset.opt]);
  // a chip that can't be used now stays greyed and says why on hover (bottom right, like the
  // tools); one that can keeps its description
  const setChip = (c, reason, about) => {
    c.setAttribute('aria-disabled', !!reason);
    if (reason) { c.dataset.tip = reason; c.removeAttribute('title'); } else { delete c.dataset.tip; c.title = about; }
  };
  setChip(opts.querySelector('[data-opt="interference"]'),
    noisy() ? `Interference readout with ${P.backend} noise takes minutes per frame on the web: noiseless backend only (or the desktop app)`
      : P.morph_front ? 'Interference readout: turn the front off first (the front turns the basis sweep)' : '',
    'Read by turning selector qubits from A (0) to B (1), instead of the basis sweep');
  setChip(opts.querySelector('[data-opt="morph_front"]'),
    P.interference ? 'Front: basis sweep only; turn interference readout off first'
      : P.payload === 'spectral' ? 'Front: not with the spectral load (its qubits hold modes of the whole surface, not patches)' : '',
    'Each data qubit turns at its own t, by its distance over the surface from a source: B spreads over A patch by patch');
  $('frontRow').hidden = !P.morph_front;
  $('front_width').value = P.front_width;
  $('front_width').nextElementSibling.textContent = P.front_width.toFixed(2);
  $('srcAuto').setAttribute('aria-pressed', P.front_source == null && !S.picking);
  $('srcPick').setAttribute('aria-pressed', S.picking || P.front_source != null);
  $('srcPick').textContent = S.picking ? 'Click a vertex…' : P.front_source != null ? `Source: vertex ${P.front_source}` : 'Pick source';
  if (!P.morph_front && S.picking) { S.picking = false; app.classList.remove('picking'); }
  placeSource();
  setChip(opts.querySelector('[data-opt="a_dirs"]'), P.payload === 'radial' ? '' : "A's directions only: radial load only",
    "Radial load: keep A's directions; only the radii morph");
  $('morph_t').value = P.morph_t;
  $('morph_t').nextElementSibling.textContent = P.morph_t.toFixed(2);
  $('morphNote').textContent = (P.interference ? 'Interference: t turns the selector qubits from A (0) to B (1).'
    : P.morph_front ? 'Front: each qubit turns from A to B at its own t, by its distance from the source.'
    : 'Basis sweep: t turns the readout from Z (A) to X (B).')
    + (P.hold_t ? ` Every frame reads at t = ${P.morph_t.toFixed(2)}.` : ` Frames run t 0 → 1; Run once reads at t = ${P.morph_t.toFixed(2)}.`);
  $('dropBText').textContent = bLabel() ? `B: ${bLabel()} ✓ — press Run & play to morph (drop to replace)` : 'Drop second mesh (.obj)';
  for (const b of box.querySelectorAll('[data-shape-b]')) b.classList.toggle('on', !S.objTextB && b.dataset.shapeB === P.shapeB);
  syncShift();
  placeMorphBox();
  layoutPanel();
}
for (const c of document.querySelectorAll('#morphOpts .chip')) {
  c.addEventListener('click', () => { if (c.getAttribute('aria-disabled') === 'true') return; P[c.dataset.opt] = !P[c.dataset.opt]; syncMorphBox(); syncStale(); });
}
$('morph_t').addEventListener('input', () => { P.morph_t = +$('morph_t').value; syncMorphBox(); syncStale(); });
$('front_width').addEventListener('input', () => { P.front_width = +$('front_width').value; syncMorphBox(); syncStale(); });
$('srcAuto').addEventListener('click', () => { P.front_source = null; S.picking = false; app.classList.remove('picking'); syncMorphBox(); syncStale(); });
$('srcPick').addEventListener('click', () => {
  if (!S.orig) return status('Load a mesh first, then pick the front\'s source on it.', true);
  setPicking(!S.picking);
  if (S.picking) status('Click a vertex on the object to start the front there (drag still rotates).');
});
$('dropB').addEventListener('click', (e) => {
  const b = e.target.closest('[data-shape-b]');
  if (b) { e.stopPropagation(); P.shapeB = b.dataset.shapeB; S.objTextB = S.objNameB = null; syncMorphBox(); syncStale(); return; }
  $('fileB').click();
});
$('fileB').addEventListener('change', async (e) => { if (e.target.files[0]) await loadFileB(e.target.files[0]); e.target.value = ''; });
async function loadFileB(file) {
  S.objTextB = await file.text(); S.objNameB = file.name; P.shapeB = null;
  status(`B: ${file.name} loaded. Press Run & play to morph ${objLabel()} → ${file.name}`);
  syncMorphBox(); syncStale();
}

// ---------- loading objects: presets, file picker, drag and drop
async function chooseShape(name) {
  P.shape = name; S.objText = S.objName = null; P.resolution = 1;
  syncDrawer();
  await enterHome();
}
const MAX_OBJ = 3_000_000; // characters; the server refuses more (Vercel caps requests at 4.5 MB)
// A dropped or picked file goes straight to the home screen, which shows the loading
// pulse while the file is read and sent; problems are reported there.
async function loadFileA(file) {
  setScreen('home');
  S.loading = true;
  status(`Loading ${file.name}…`);
  try {
    const text = await file.text();
    if (text.length > MAX_OBJ) throw new Error(`${file.name} is too large (${(text.length / 1e6).toFixed(1)} MB; the limit is 3 MB). Try a decimated export.`);
    if (!/^\s*v\s/m.test(text)) throw new Error(`${file.name} has no vertices ('v' lines): is it an OBJ?`);
    S.objText = text; S.objName = file.name; P.resolution = 1;
    syncDrawer();
    await enterHome();
  } catch (err) { status(`⚠ ${err.message}`, true); } finally { S.loading = false; }
}
async function enterHome() {
  setScreen('home');
  syncRes();
  S.loading = true;
  try {
    status(`Loading ${objLabel()}…`);
    await loadMesh();
    status(`${objLabel()} ready. Press Run once to encode and read it, or ${HOVER ? 'click' : 'tap'} it for settings.`);
  } catch (err) { status(`⚠ ${err.message}`, true); } finally { S.loading = false; }
}
for (const b of document.querySelectorAll('[data-shape]')) b.addEventListener('click', () => chooseShape(b.dataset.shape));
$('dropObj').addEventListener('click', () => $('fileA').click());
$('fileA').addEventListener('change', async (e) => { if (e.target.files[0]) await loadFileA(e.target.files[0]); e.target.value = ''; });
$('logo').addEventListener('click', () => setScreen('landing'));
// Hover labels for buttons that are only a graphic ([data-tip]): one shared label, just
// above the button (or below it when there is no room above), at half opacity. It is a
// popover so it sits in the top layer, over an open dialog too.
function showTip(el) {
  tipFor = el;
  const tip = el.dataset.tip;
  const greyed = el.getAttribute('aria-disabled') === 'true'; // its label is the reason it can't be used
  const corner = !greyed && el.matches('.orb'); // a setting's option circle: the label goes in the bottom-left corner
  // sentence case; in the corner, a label breaks after a colon and before a bracketed aside
  const text = tip[0].toUpperCase() + tip.slice(1);
  tipEl.textContent = corner ? text.replace(': ', ':\n').replace(' (', '\n(') : text;
  const c = (el.getBoundingClientRect().left + el.getBoundingClientRect().right) / 2;
  tipEl.style.textAlign = corner ? 'left' : greyed ? 'right' : c < innerWidth / 3 ? 'right' : c > (2 * innerWidth) / 3 ? 'left' : 'center'; // before measuring
  if (!tipEl.matches(':popover-open')) tipEl.showPopover();
  const t = tipEl.getBoundingClientRect(), gap = 6;
  const r = el.getBoundingClientRect();
  const mid = (r.left + r.right) / 2, third = innerWidth / 3;
  let x0, y;
  if (greyed) { // why a control is greyed: always in the bottom-right corner (above any colour key there)
    const cs = getComputedStyle(app), edge = parseFloat(cs.getPropertyValue('--edge')) || 24, bottom = parseFloat(cs.getPropertyValue('--bottom')) || 24;
    const keys = [...document.querySelectorAll('.legend:not([hidden])')].map((l) => l.getBoundingClientRect().top);
    x0 = innerWidth - edge - t.width;
    y = (keys.length ? Math.min(...keys) - 2 * gap : innerHeight - bottom) - t.height;
  } else if (corner) { // bottom-left, just above the Log button and in line with its left edge (and the name's)
    const log = $('logToggle').getBoundingClientRect();
    x0 = log.left;
    y = (log.height ? log.top : innerHeight - (parseFloat(getComputedStyle(app).getPropertyValue('--bottom')) || 24)) - 2 * gap - t.height;
  } else {
    y = r.top - gap - t.height >= 4 ? r.top - gap - t.height : r.bottom + gap; // above, or below if no room
    // grow away from the middle of the window, where the object sits: a button on the
    // left lines the label up with its right edge, one on the right with its left edge
    x0 = mid < third ? r.right - t.width : mid > 2 * third ? r.left : mid - t.width / 2;
  }
  const x = Math.min(Math.max(x0, 4), innerWidth - t.width - 4);
  Object.assign(tipEl.style, { left: `${x}px`, top: `${y}px` });
}
function hideTip() { tipFor = null; if (tipEl.matches(':popover-open')) tipEl.hidePopover(); }
document.addEventListener('pointerover', (e) => {
  if (e.pointerType !== 'mouse') return;
  const el = app.dataset.screen === 'landing' ? null : e.target.closest?.('[data-tip]'); // hints start on home
  if (el && el !== tipFor) showTip(el); else if (!el && tipFor) hideTip();
});
document.addEventListener('pointerdown', hideTip);
document.documentElement.addEventListener('pointerleave', hideTip);
addEventListener('blur', hideTip);
$('help').addEventListener('click', () => $('helpDialog').showModal());

let dragDepth = 0;
addEventListener('dragenter', (e) => { if (!Array.from(e.dataTransfer?.types || []).includes('Files')) return; e.preventDefault(); dragDepth++; app.classList.add('dragging'); $('dropOverlay').hidden = false; });
addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; app.classList.remove('dragging'); $('dropOverlay').hidden = true; $('dropB').classList.remove('over'); } });
addEventListener('dragover', (e) => {
  e.preventDefault();
  $('dropB').classList.toggle('over', !!e.target.closest?.('#dropB'));
  $('dropOverlay').querySelector('p').textContent = e.target.closest?.('#dropB') ? 'Drop mesh B' : 'Drop to load';
});
addEventListener('drop', async (e) => {
  e.preventDefault();
  dragDepth = 0; app.classList.remove('dragging'); $('dropOverlay').hidden = true; $('dropB').classList.remove('over');
  const file = [...(e.dataTransfer?.files || [])].find((f) => /\.obj$/i.test(f.name)) || e.dataTransfer?.files[0];
  if (!file) return;
  if (!/\.obj$/i.test(file.name)) {
    const msg = `${file.name} is not an .obj file`;
    if (app.dataset.screen === 'landing') { $('landingMsg').textContent = msg; $('landingMsg').hidden = false; return; }
    return status(`⚠ ${msg}`, true);
  }
  $('landingMsg').hidden = true;
  if (e.target.closest?.('#dropB') && $('dropB').classList.contains('shown')) await loadFileB(file);
  else await loadFileA(file);
});

// ---------- resolution row: low (decimated), medium (as is), high (subdivided)
const RES_NAMES = ['low', 'medium', 'high'];
function syncRes() {
  for (const b of document.querySelectorAll('#resRow [data-res]')) b.setAttribute('aria-checked', +b.dataset.res === P.resolution);
}
for (const b of document.querySelectorAll('#resRow [data-res]')) {
  b.addEventListener('click', async () => {
    if (+b.dataset.res === P.resolution && S.mesh) return;
    P.resolution = +b.dataset.res;
    syncRes();
    try {
      await loadMesh();
      status(`${objLabel()} at ${RES_NAMES[P.resolution]} res: ${S.mesh.verts.length} vertices`);
    } catch (err) { status(`⚠ ${err.message}`, true); }
    syncStale();
  });
}

// ---------- run buttons
$('run').addEventListener('click', () => { useErrorColour(); run(1); });
// a readout (one, or frames) shows how far each vertex moved, so both run buttons
// switch the colour to error
function useErrorColour() {
  const want = P.mode === 'morph' && P.morph_front ? 'front' : 'error'; // a front run shows the front
  if (P.colour === want) return;
  P.colour = want;
  syncDrawer();
  rebuild();
}
$('play').addEventListener('click', () => {
  const fresh = S.run === runKey() && S.frames.length > 1;
  useErrorColour();
  if (!fresh) return run(P.frames);
  S.playing = !S.playing;
  if (S.playing && S.step + 1 >= S.seq.length) S.step = 0;
  syncPlay();
});
// the second run button says what a click will do: run the frames (none yet, or the
// settings have changed since), or pause / play the frames already run
function syncPlay() {
  const fresh = S.run === runKey() && S.frames.length > 1;
  $('play').classList.toggle('playing', S.playing);
  $('play').textContent = !fresh ? 'Run & play' : S.playing ? 'Pause' : 'Play';
  $('play').setAttribute('aria-label', !fresh ? 'Run and play: run the frames, then play them' : S.playing ? 'Pause' : 'Play');
}
// the run button glows while the settings differ from the last run
function syncStale() { $('run').classList.toggle('stale', !!S.mesh && S.run !== runKey()); syncPlay(); }

// ---------- HUD and drawer
// Noise simulation runs ~9× slower on the web function than on a laptop, so some work
// cannot finish inside its 300 s limit (the server refuses it too: api/tessa.py too_slow).
const CHIPS = ['phoenix', 'fez'];
const noisy = () => P.backend !== 'noiseless';
// A tool that can't run right now stays greyed but still shows, on hover, why not (its
// reason as the hover label); a click on it does nothing.
function setTool(id, reason) {
  const b = $(id);
  b.setAttribute('aria-disabled', !!reason);
  if (reason) b.dataset.tip = reason; else delete b.dataset.tip;
}
function syncHud() {
  const morph = P.mode === 'morph', chip = CHIPS.includes(P.backend);
  const base = S.busy ? 'Busy: wait for the current run to finish' : !S.mesh ? 'Load an object first' : '';
  setTool('edges', base || (morph ? 'Edge readouts compare against one fixed mesh: not in morph mode'
    : chip ? `Edge readouts on the ${P.backend} chip take over a minute on the web: use noiseless or flat noise, or the desktop app` : ''));
  setTool('curves', base || (morph ? 'Curves compare against one fixed mesh: not in morph mode'
    : noisy() ? `Curves with ${P.backend} noise take minutes on the web: switch backend to noiseless, or use the desktop app` : ''));
  setTool('why', base || (morph && !bLabel() ? 'Morph: add a second mesh first' : ''));
  setTool('obj', S.seq.length ? '' : 'Run the object first, then download the result');
  setTool('glb', S.keys.length >= 2 ? '' : 'Run the object first, then download its readouts as shape keys');
  setTool('qmapOn', S.data ? '' : 'Run the object first to see its qubit map');
  setTool('viewToggle', '');
}
// clicks on a tool that can't run are stopped before they reach it
$('hud').addEventListener('click', (e) => {
  if (e.target.closest('button[aria-disabled="true"]')) { e.stopImmediatePropagation(); e.preventDefault(); }
}, true);
$('why').onclick = why;
$('curves').onclick = curves;
$('edges').onclick = readEdges;
$('obj').onclick = downloadObj;
$('glb').onclick = downloadGlb;
function setMapOpen(on) { qmapOn = on; $('qmapOn').setAttribute('aria-pressed', on); setFocus(null); }
$('qmapOn').onclick = () => setMapOpen(!qmapOn);
$('qmapClose').addEventListener('click', () => setMapOpen(false));
// The tools menu: closed behind its handle by default; the handle slides it in from the right.
function setSide(open) {
  $('side').classList.toggle('open', open);
  $('hud').inert = !open; // closed: its buttons are out of reach (and out of the tab order)
  $('sideHandle').setAttribute('aria-expanded', open);
  $('sideHandle').setAttribute('aria-label', open ? 'Hide the tools' : 'Show the tools');
  if (!open && !$('drawer').hidden) { $('drawer').hidden = true; $('viewToggle').setAttribute('aria-expanded', false); }
}
$('sideHandle').addEventListener('click', () => setSide(!$('side').classList.contains('open')));

$('viewToggle').onclick = () => {
  const open = $('drawer').hidden;
  $('drawer').hidden = !open;
  $('viewToggle').setAttribute('aria-expanded', open);
  placeDrawer();
};
// the View drawer sits just left of the tool pills (a 24px gap), clear of the object; on
// small screens the stylesheet places it instead
function placeDrawer() {
  const d = $('drawer');
  if (d.hidden) return;
  if (compact) { d.style.right = ''; return; }
  d.style.right = `${innerWidth - $('hud').getBoundingClientRect().left + 24}px`;
}

const DRAWER_RANGES = ['gain', 'between', 'fps', 'headroom'];
function syncDrawer() {
  for (const id of DRAWER_RANGES) { $(id).value = P[id]; $(id).nextElementSibling.textContent = P[id]; }
  $('seed').value = P.seed;
  for (const id of ['loop', 'orig', 'spin']) $(id).checked = P[id];
  for (const b of document.querySelectorAll('#drawer [data-set]')) b.classList.toggle('on', String(P[b.dataset.set]) === b.dataset.value);
}
for (const id of DRAWER_RANGES) {
  $(id).addEventListener('input', () => {
    P[id] = +$(id).value; $(id).nextElementSibling.textContent = P[id];
    if (['gain', 'between'].includes(id)) rebuild();
    syncStale();
  });
}
$('seed').addEventListener('input', () => { P.seed = +$('seed').value || 0; syncStale(); });
for (const id of ['loop', 'orig', 'spin']) $(id).addEventListener('input', () => { P[id] = $(id).checked; if (id !== 'spin') rebuild(); });
for (const b of document.querySelectorAll('#drawer [data-set]')) {
  b.addEventListener('click', async () => {
    const k = b.dataset.set;
    P[k] = b.dataset.value;
    syncDrawer();
    rebuild();
    if (S.data) showInfo(S.data, params());
    syncStale(); // view and gain by shots change what a run reads
  });
}

// ---------- animation loop
let last = performance.now(), busyLevel = 0;
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  controls.autoRotate = P.spin && app.dataset.screen === 'home';
  controls.update();
  // the shimmer eases in while a run encodes / reads, and back out when it lands
  busyLevel += ((S.busy || S.loading ? 1 : 0) - busyLevel) * Math.min(1, dt * 3);
  cloudMat.uniforms.uBusy.value = busyLevel;
  cloudMat.uniforms.uTime.value = now / 1000;
  if (S.playing && S.seq.length > 1) {
    S.acc = Math.min(S.acc + dt, 1);
    const step = 1 / P.fps;
    let moved = false;
    while (S.acc >= step) {
      S.acc -= step;
      if (S.step + 1 < S.seq.length) S.step++;
      else if (P.loop) S.step = 0;
      else { S.playing = false; syncPlay(); break; }
      moved = true;
    }
    if (moved) draw();
  }
  // the landing hides the viewer (opacity 0): once it has faded out, skip the GPU work
  if (app.dataset.screen !== 'landing' || now - landedAt < 700) renderer.render(scene, camera);
}

// touch screens have no hover: say "tap" instead
if (!HOVER) $('hint').textContent = 'Tap for settings';

window.tessa3d = { state: S, settings: P, buildGlb }; // read-only handle for debugging in the console
fitStage();
syncDrawer(); syncHud();
requestAnimationFrame(frame);
status('Drop an .obj file or pick a shape.');
// warm the function up while the landing page shows (a cold start can take ~10 s)
fetch('/api/tessa').catch(() => {});
