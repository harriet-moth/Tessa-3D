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
const OBJ_R = 270; // stage px: holding inside this radius of the object opens the menu

// ---------- settings: everything a run or the view reads
const P = {
  shape: 'blob', resolution: 0, payload: 'radial', mode: 'theta', layout: 'mesh',
  clamp: 0, clamp_b: 0, compression: 0, quant_bits: 0, headroom: 0.1,
  backend: 'noiseless', placement: 'quality', shots: 12, noise_strength: 1, seed: 0, drr: 'none',
  frames: 12, morph_t: 0.5, shapeB: null,
  hold_t: false, interference: false, shared_topology: false, a_dirs: false, // morph options
  view: 'fidelity', colour: 'glow', gain: 3, between: 8, fps: 24, loop: true, orig: true, spin: true,
};
// S: what the page has loaded and is showing
const S = { mesh: null, meshKey: null, data: null, run: null, orig: null, frames: [], refs: [], ts: null,
  seq: [], step: 0, playing: false, acc: 0, edges: null, focus: null, busy: false,
  objText: null, objName: null, objTextB: null, objNameB: null };

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
  // ribbons: as big as the frame would be if it covered the window (portrait: the frame's own size)
  app.style.setProperty('--c', portrait ? Math.max(w / W_STAGE, scale) : Math.max(w / W_STAGE, h / H_STAGE));
  // wide windows: spread the landing shapes out with the width (up to 1.35× the frame's spacing)
  app.style.setProperty('--spread', portrait ? 0 : Math.min(Math.max(w / scale / W_STAGE, 1), 1.35) - 1);
  app.style.setProperty('--T', `${layout.T}px`);
  app.style.setProperty('--B', `${layout.B}px`);
  app.style.setProperty('--vl', `${Math.max(0, layout.vl)}px`);
  resize();
  if (!panelEl.hidden) placeOrbs();
  syncShift();
}
const toStage = (x, y) => [(x - innerWidth / 2) / scale + CX, (y - innerHeight / 2) / scale + CY];

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
const SURFACE_GLSL = /* glsl */`
    uniform sampler2D uPos; uniform sampler2D uCol; uniform sampler2D uNrm; uniform int uW;
    uniform float uMix, uZ0, uZ1, uCurve, uR;
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
      // sci-fi palette of the design's point clouds: violet base, electric blue, cyan crown
      vec3 glow = h < 0.45 ? mix(vec3(0.56, 0.28, 1.0), vec3(0.16, 0.45, 1.0), h / 0.45)
                           : mix(vec3(0.16, 0.45, 1.0), vec3(0.36, 0.96, 1.0), (h - 0.45) / 0.55);
      vCol = mix(glow, c, uMix);
      vec4 mv = modelViewMatrix * vec4(p, 1.0);
      float depth = (mv.z - (modelViewMatrix * vec4(0.0, 0.0, 0.0, 1.0)).z) / uR;
      vA = mix(0.22, 1.0, smoothstep(-0.7, 0.45, depth));
      return mv;
    }`;
const surfaceUniforms = {
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
const edgeLines = new THREE.LineSegments(edgeGeom, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false }));
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
const mates = new THREE.Points(mateGeom, new THREE.PointsMaterial({ map: discTexture('rgb(230, 150, 0)', '#14161e', 12), alphaTest: 0.5, size: 0.08 }));
edgeLines.visible = mates.visible = cloud.visible = wire.visible = false;
scene.add(wire, cloud, edgeLines, mates);

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
function ramp(cs, t) { // blends in sRGB, like CSS gradients and matplotlib, so mesh and legend agree
  t = Math.min(1, Math.max(0, t));
  const x = t * (cs.length - 1), i = Math.min(Math.floor(x), cs.length - 2), f = x - i;
  const [r, g, b] = cs[i].map((c, k) => c + (cs[i + 1][k] - c) * f);
  return new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);
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
    seed: P.seed, drr: P.drr, morph_t: P.morph_t,
  };
  if (S.objText) p.obj = S.objText; else p.shape = P.shape;
  if (P.mode === 'morph') {
    if (S.objTextB) p.obj_b = S.objTextB; else if (P.shapeB) p.shape_b = P.shapeB;
    for (const k of MORPH_OPTS) p[k] = P[k];
    p.clamp_b = p.clamp; // morph uses one clamp for both fields (bridge.morph_clamp)
  }
  return p;
}
// The same settings with uploads named instead of inlined: what "settings changed" compares.
const runKey = () => JSON.stringify({ ...params(), obj: S.objName, obj_b: S.objNameB });
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
function status(msg, err = false) { $('status').textContent = msg; $('status').classList.toggle('err', err); }
async function busy(label, fn) {
  S.busy = true;
  app.classList.add('busy');
  syncPlay();
  document.querySelectorAll('#hud button').forEach((b) => { b.disabled = true; });
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
  S.mesh = d; S.meshKey = key; S.data = null; S.run = null; S.edges = null; S.focus = null;
  setEdges(null);
  S.orig = new Float32Array(d.verts.flat());
  S.frames = []; S.refs = []; S.ts = null;
  buildCloud(d);
  rebuild();
  S.step = 0; S.playing = false;
  $('stats').textContent = `${objLabel()} · ${d.verts.length} vertices · ${d.faces.length} faces`;
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
  objRadius = Math.max(...V.map((p) => Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]))) || 1;
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
  const zs = V.map((p) => p[2]);
  cloudMat.uniforms.uZ0.value = Math.min(...zs); cloudMat.uniforms.uZ1.value = Math.max(...zs);
  mates.material.size = spacing * 2.5;
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
  if (P.view === 'material') {
    keys = S.frames.map((f, k) => {
      const o = new Float32Array(f.length), r = refs[k];
      for (let j = 0; j < f.length; j++) o[j] = r[j] + P.gain * (f[j] - r[j]);
      return o;
    });
  }
  const lead = P.orig || !keys.length ? [S.orig] : [];
  const glide = (k) => (k.length === 1 ? k : timeline(k, P.between, P.loop));
  S.seq = glide(lead.concat(keys));
  S.step = Math.min(S.step, S.seq.length - 1);
  // Error colour follows the playback: each step's true decoded error (the same timeline
  // without gain), on one scale fixed to the largest error anywhere in the run.
  const real = glide(lead.concat(S.frames)), ref = glide(lead.concat(refs));
  S.err = real.map((v, s) => {
    const e = new Float32Array(n), r = ref[s];
    for (let i = 0; i < n; i++) e[i] = Math.hypot(v[3 * i] - r[3 * i], v[3 * i + 1] - r[3 * i + 1], v[3 * i + 2] - r[3 * i + 2]);
    return e;
  });
  S.errTop = Math.max(...S.err.map((e) => Math.max(...e))) || 1;
  colourise();
  draw();
}

function colourise(stepOnly = false) {
  if (!colTex) return;
  const n = S.orig.length / 3, col = colTex.image.data;
  const set = (i, c) => { col[4 * i] = c.r; col[4 * i + 1] = c.g; col[4 * i + 2] = c.b; };
  let mix = 0.85;
  if (P.colour === 'error' && S.frames.length) {
    const err = S.err[S.step];
    for (let i = 0; i < n; i++) set(i, ramp(ERROR, err[i] / S.errTop));
    if (!stepOnly) legend('faceLegend', ERROR_GLOW_HEX, '0', S.errTop.toFixed(4), `vertex error at this step (distance${S.ts ? ' from the noiseless morph' : ''}, mesh units)`);
  } else if (P.colour === 'qubit' && S.data) {
    const d = S.data, used = [...new Set(d.qubits)].sort((a, b) => a - b), rank = new Map(used.map((q, r) => [q, r]));
    for (let i = 0; i < n; i++) set(i, hsl(rank.get(d.qubits[i]) * GOLDEN, 0.8, 0.6));
    $('faceLegend').hidden = false;
    $('faceLegend').innerHTML = `<div class="swatches">${used.map((q, r) => `<i title="qubit ${q}" style="background:hsl(${((r * GOLDEN) % 1 * 360).toFixed(0)} 80% 60%)"></i>`).join('')}</div>one colour per data qubit (${used.length})`;
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
    + `<div class="ends"><span>${lo}</span><span>${hi}</span></div>${caption}`;
}

function copyEnds(target, pairs, v) {
  pairs.forEach((vi, j) => { target[3 * j] = v[3 * vi]; target[3 * j + 1] = v[3 * vi + 1]; target[3 * j + 2] = v[3 * vi + 2]; });
}

function draw() {
  if (!S.seq.length || !posTex) return;
  if (P.colour === 'error' && S.frames.length) colourise(true);
  const v = S.seq[S.step], pos = posTex.image.data, nrm = nrmTex.image.data;
  for (let i = 0; i < v.length / 3; i++) {
    pos[4 * i] = v[3 * i]; pos[4 * i + 1] = v[3 * i + 1]; pos[4 * i + 2] = v[3 * i + 2];
  }
  // Corner normals of the current shape, for the curved faces: each corner averages
  // (by area) the faces round its vertex that bend less than CREASE from its own face,
  // so smooth surfaces round off and hard edges stay sharp.
  const T = S.tris, fn = new Float32Array(T.length * 3), fu = new Float32Array(T.length * 3);
  T.forEach(([a, b, c], t) => {
    const ux = v[3 * b] - v[3 * a], uy = v[3 * b + 1] - v[3 * a + 1], uz = v[3 * b + 2] - v[3 * a + 2];
    const wx = v[3 * c] - v[3 * a], wy = v[3 * c + 1] - v[3 * a + 1], wz = v[3 * c + 2] - v[3 * a + 2];
    const nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx, l = Math.hypot(nx, ny, nz) || 1;
    fu.set([nx, ny, nz], 3 * t);
    fn.set([nx / l, ny / l, nz / l], 3 * t);
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
}

function showInfo(d, p) {
  const s = d.stats;
  $('stats').textContent = [
    `${objLabel()}${d.ts ? ` → ${bLabel()} (morph, ${d.match.method}, ${(100 * d.match.coverage).toFixed(0)}% of B covered)` : ''} · ${d.verts.length} verts · ${p.payload}, ${p.layout} layout`,
    ...(d.ts ? [`${d.options.interference ? `interference readout · selector 0 = A, 1 = B${d.kept ? ` · decoded from ${(100 * d.kept.reduce((a, b) => a + b, 0) / d.kept.length).toFixed(0)}% of shots` : ''}` : 'basis sweep · θ = A, φ = B'} · clamp ${d.clamp.toFixed(2)}π (θ and φ)`
      + `${d.options.hold_t ? ` · held at t = ${d.ts[0]}` : ''}${d.options.a_dirs ? " · A's directions" : ''}`] : []),
    s.graph + (s.quality != null ? ` · data-qubit quality ${s.quality.toFixed(2)}` : ''),
    `${s.values} values on ${s.qubits} qubits${s.qubits < s.graph_qubits ? ` (of ${s.graph_qubits})` : ''} = ${s.ratio.toFixed(2)} / qubit`,
    `depth ${s.depth} · 2q gates ${s.twoq}` + (s.twoq_single !== s.twoq ? ` (θ alone ${s.twoq_single})` : ''),
    `mean error ${(d.error).toFixed(4)} of bbox${P.view === 'material' ? ` · material gain ${P.gain}×` : ' · true positions'}`,
  ].join('\n');
}

// ---------- actions
const TOO_BIG = /do not fit|exceed|capacity/i;
async function run(frames) {
  if (P.mode === 'morph' && !bLabel()) {
    openPanel('encode', 'mode');
    return status('⚠ Morph needs a second mesh: drop one in the box, or pick a shape', true);
  }
  await busy(frames > 1 ? `${P.mode === 'morph' ? 'Morphing' : 'Running'} ${frames} readouts on ${P.backend}…` : `Encoding and reading once on ${P.backend}…`, async () => {
    let d;
    for (;;) {
      try { d = await call('run', { frames }); break; } catch (err) {
        // an uploaded mesh too big for the lattice: decimate (each step halves it) and retry
        if (!TOO_BIG.test(err.message) || P.resolution === 0 || !S.objText) throw err;
        P.resolution -= 1;
        syncDrawer();
        status(`Too many values for the lattice: decimating ${objLabel()} to ${['low', 'mid'][P.resolution]} detail and retrying…`);
      }
    }
    const p = params();
    if (meshKeyNow() !== S.meshKey) setMesh(d, meshKeyNow());
    Object.assign(S, { data: d, run: runKey(), frames: d.frames.map((f) => new Float32Array(f.flat())),
      refs: d.refs ? d.refs.map((r) => new Float32Array(r.flat())) : [], ts: d.ts || null, step: 0 });
    showInfo(d, p);
    S.focus = null;
    rebuild();
    drawMap();
    S.playing = d.frames.length > 1;
    if (!S.playing) { S.step = S.seq.length - 1; draw(); }
    const cut = d.frames.length < d.requested ? ` (fitted to the time limit; asked for ${d.requested})` : '';
    const what = d.ts ? (d.ts.length > 1 ? `morph ${objLabel()} → ${bLabel()}: ${d.frames.length} readouts, t 0 → 1` : `morph at t = ${d.ts[0]}`) : `${d.frames.length} readout(s)`;
    status(`${what} in ${d.seconds.toFixed(1)} s on ${p.backend}${cut}`);
  });
  syncPlay(); syncStale();
}

function setEdges(r) {
  S.edges = r;
  $('edges').textContent = r ? 'hide edges' : 'read edges';
  $('edgeLegend').hidden = !r;
  if (!r) { edgeLines.visible = false; return; }
  const lo = Math.min(...r.exact), hi = Math.max(...r.exact), span = hi - lo || 1; // true edge range
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

async function why() {
  await busy('Transpiling Tessa and four baselines onto ibm_phoenix…', async () => {
    const r = await call('resources');
    const shots = 2 ** P.shots;
    const rows = r.rows.map((x) => `<tr><td>${x.name}</td><td>${x.qubits}</td><td>${x.fits ? 'yes' : 'no'}</td><td>${x.twoq}</td><td>${x.depth}</td><td>${x.swaps}</td><td>${x.settings}</td><td>${x.configs}</td><td>${Math.floor(shots / x.configs)}</td></tr>`).join('');
    openDialog(`<h2>why tessa</h2>
      <p>${r.stats.values} values (${r.payload}, ${r.mode}) transpiled onto the ibm_phoenix snapshot (120 qubits, square lattice).</p>
      <table><tr><th>encoding</th><th>qubits</th><th>fits on chip</th><th>2q gates</th><th>depth</th><th>SWAPs</th><th>settings</th><th>outcomes / cell</th><th>shots / value (${shots})</th></tr>${rows}</table>
      <p class="note">Generic random data in every row, no compression. SWAPs ≈ extra CZs from routing ÷ 3. Each shot is split over a cell's outcomes, so each value gets about shots ÷ outcomes. Amplitude: Z readout gives magnitudes only. With a chip backend, Tessa's row is on that chip; the baselines stay on Phoenix.</p>`);
    status('Why Tessa: resource table opened');
  });
}

function chart({ title, xlabel, ylabel, series, logx = false, logy = false }) {
  const W = 420, H = 270, m = { l: 58, r: 12, t: 42, b: 42 };
  const fx = logx ? Math.log10 : (x) => x, fy = logy ? Math.log10 : (y) => y;
  const pts = series.flatMap((s) => s.points);
  const [x0, x1] = [Math.min(...pts.map((p) => fx(p[0]))), Math.max(...pts.map((p) => fx(p[0])))];
  let [y0, y1] = [Math.min(...pts.map((p) => fy(p[1]))), Math.max(...pts.map((p) => fy(p[1])))];
  if (y1 === y0) { y0 -= 1; y1 += 1; }
  const X = (x) => m.l + ((fx(x) - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const Y = (y) => H - m.b - ((fy(y) - y0) / (y1 - y0)) * (H - m.t - m.b);
  const fmt = (v) => (Math.abs(v) >= 1000 ? v.toExponential(0) : +v.toPrecision(2)).toString();
  const ticks = (a, b, log) => [0, 1, 2, 3].map((k) => { const t = a + (k / 3) * (b - a); return log ? 10 ** t : t; });
  // mark: 'ring' (hollow circle), 'square' or 'none'; dashed series draw on top, so
  // two series that coincide both stay visible
  const markAt = (s, x, y) => (s.mark === 'none' ? '' : s.mark === 'square'
    ? `<rect x="${x - 3}" y="${y - 3}" width="6" height="6" fill="${s.color}"/>`
    : `<circle cx="${x}" cy="${y}" r="5" fill="none" stroke="${s.color}" stroke-width="1.8"/>`);
  const mark = (s, p) => markAt(s, X(p[0]), Y(p[1]));
  const lines = series.map((s) => `<polyline fill="none" stroke="${s.color}" stroke-width="2" ${s.dash ? 'stroke-dasharray="5 4"' : ''} points="${s.points.map((p) => `${X(p[0])},${Y(p[1])}`).join(' ')}"/>`
    + s.points.map((p) => mark(s, p)).join('')).join('');
  // legend in a row under the title, clear of the data
  const legendRow = series.map((s, i) => {
    const x = m.l + 150 * i;
    return `<line x1="${x}" y1="29" x2="${x + 22}" y2="29" stroke="${s.color}" stroke-width="2" ${s.dash ? 'stroke-dasharray="5 4"' : ''}/>`
      + markAt(s, x + 11, 29)
      + `<text x="${x + 28}" y="33">${s.label}</text>`;
  }).join('');
  const xt = ticks(x0, x1, logx).map((t) => `<text x="${X(t)}" y="${H - m.b + 16}" text-anchor="middle">${fmt(t)}</text>`).join('');
  const yt = ticks(y0, y1, logy).map((t) => `<text x="${m.l - 6}" y="${Y(t) + 4}" text-anchor="end">${fmt(t)}</text>`).join('');
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}">
    <text x="${W / 2}" y="16" text-anchor="middle" style="font-weight:600">${title}</text>
    <line x1="${m.l}" y1="${H - m.b}" x2="${W - m.r}" y2="${H - m.b}" stroke="currentColor" opacity=".3"/>
    <line x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${H - m.b}" stroke="currentColor" opacity=".3"/>
    ${xt}${yt}${lines}${legendRow}
    <text x="${(W + m.l) / 2}" y="${H - 6}" text-anchor="middle">${xlabel}</text>
    <text transform="translate(14 ${(H - m.b + m.t) / 2}) rotate(-90)" text-anchor="middle">${ylabel}</text></svg>`;
}

async function curves() {
  await busy('Curves: 15 readouts on the current backend…', async () => {
    const r = await call('curves');
    const A = '#5ef2ff', B = '#ffa90a', G = '#999999';
    const shots = r.shots.map(([s, e]) => [s, e]);
    const guide = shots.map(([s]) => [s, shots[0][1] * Math.sqrt(shots[0][0] / s)]);
    const err = (k) => r[k].map(([c, , e]) => [c, e]);
    const depth = (k) => r[k].map(([c, d]) => [c, d]);
    // fixed per layout, as in the desktop app: mesh solid with rings, naive dashed with squares
    const layouts = (pick) => [{ label: 'mesh layout', color: A, points: pick('mesh') },
      { label: 'naive layout', color: B, dash: true, mark: 'square', points: pick('naive') }];
    openDialog(`<h2>fidelity curves</h2><div class="charts">
      ${chart({ title: 'error vs shots', xlabel: 'shots', ylabel: 'mean error (of bbox)', logx: true, logy: true,
        series: [{ label: 'measured', color: A, points: shots }, { label: '1/√shots', color: G, dash: true, mark: 'none', points: guide }] })}
      ${chart({ title: 'error vs compression', xlabel: 'compression %', ylabel: 'mean error (of bbox)', series: layouts(err) })}
      ${chart({ title: 'depth vs compression', xlabel: 'compression %', ylabel: 'circuit depth', series: layouts(depth) })}
      </div>${r.same ? '<p class="note"><b>Mesh and naive layouts place every value identically for this mesh</b>, so their lines coincide.</p>' : ''}<p class="note">Error is the mean coordinate error as a fraction of the bounding box. Compression drops the smallest Walsh–Hadamard coefficients of each Tessa cell.</p>`);
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

// ---------- qubit map: which data and address qubits hold each vertex
// map.vertices[i][c] = [group, address]: value c of vertex i sits in data qubit
// groups[group].data, in the slot where address qubit groups[group].addresses[j]
// reads bit j of address (Tessa's decoder bit order).
const HOT = 'rgb(230, 150, 0)';
let qmapOn = false;
const mapOn = () => qmapOn && !!S.data;
const comps = () => (S.data.map.vertices[0].length === 1 ? ['r'] : P.payload === 'spherical' ? ['polar', 'azimuth', 'r'] : ['x', 'y', 'z']);

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
  const [muted, accent, line, panel] = ['rgba(255,255,255,0.45)', '#5ef2ff', 'rgba(199,199,199,0.3)', '#1b1b1b'];
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
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.font = `600 ${Math.max(8, step * 0.32)}px ui-monospace, Menlo, monospace`;
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
  if (!f) return 'Hover a data qubit to light its vertices on the object.';
  const qs = (g) => m.groups[g].addresses.map((q) => `q${q}`).join(' ');
  const g = f.groups[0], held = m.vertices.flat().filter((v) => v[0] === g).length;
  return `data q${m.groups[g].data} · address ${qs(g)}\n${2 ** m.groups[g].addresses.length} slots, ${held} values of this mesh (${comps().join(', ')})\n${f.mates.length} vertices on it (orange on the object)`;
}

function setFocus(group = null) {
  if (!mapOn() || group === null) S.focus = null;
  else {
    if (S.focus && S.focus.groups[0] === group) return;
    const mates = [];
    S.data.map.vertices.forEach((vs, i) => { if (vs.some((v) => v[0] === group)) mates.push(i); });
    S.focus = { groups: [group], mates };
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
// Game UI: screens, the click-and-hold arc menu, and the settings panels
//   layer 0  home: the object and the run buttons
//   layer 1  while held: Encode and Decode pills; releasing on one opens its arc
//   layer 2  an arc of options, unsweeping from its back button
//   layer 3  one setting's toggles; the arc has swept back into the back button
// ====================================================================================
function setScreen(name) {
  app.dataset.screen = name;
  if (name === 'landing') { closeAll(); S.playing = false; }
}

// object offset in stage px (left of, or in portrait above, the morph drop box)
function setShift(x, y) { app.style.setProperty('--ox', `${x}px`); app.style.setProperty('--oy', `${y}px`); }
let shift = [0, 0];
// the morph box sits right of the object (design) or, in portrait, below it
const MORPH_SHIFT = () => (portrait ? [0, -100] : [-240, 10]);
const morphBoxOpen = () => !panelEl.hidden && panelEl.dataset.key === 'mode' && P.mode === 'morph';
function syncShift() {
  shift = morphBoxOpen() ? MORPH_SHIFT() : [0, 0];
  setShift(...shift);
}

// ---------- geometry: every menu shape is a band of constant width on a circle round
// the object, so the arcs are exactly even (the design's are hand-drawn prototypes)
const rad = THREE.MathUtils.degToRad;
const ARC = { R: 295, h: 32, a0: -60, a1: 60 }; // layer 2: the back button sits on the a0 cap
const PILL = { R: 300, h: 28, span: 23 }; // layer 1
// a point at angle a (degrees from straight up) on a circle of radius r round the
// object: the encode side above it, the decode side mirrored below
const onRing = (side, r, a) => [CX + r * Math.sin(rad(a)), side === 'decode' ? CY + r * Math.cos(rad(a)) : CY - r * Math.cos(rad(a))];
// The outline of a band from angle a0 to a1: outer arc, round cap, inner arc, round cap.
// With a0 = a1 it is a circle of diameter 2h: the arc collapsed into an ellipse.
function bandPath(side, R, h, a0, a1) {
  const p = (r, a) => onRing(side, r, a).map((v) => v.toFixed(2)).join(' ');
  const big = a1 - a0 > 180 ? 1 : 0, cw = side === 'decode' ? 0 : 1, ccw = 1 - cw;
  return `M ${p(R + h, a0)} A ${R + h} ${R + h} 0 ${big} ${cw} ${p(R + h, a1)} A ${h} ${h} 0 0 ${cw} ${p(R - h, a1)}`
    + ` A ${R - h} ${R - h} 0 ${big} ${ccw} ${p(R - h, a0)} A ${h} ${h} 0 0 ${cw} ${p(R + h, a0)} Z`;
}
const capAt = (side) => onRing(side, ARC.R, ARC.a0); // the arc's start cap: home of the back button

function drawMenuShapes() {
  for (const side of ['encode', 'decode']) {
    $(`${side}PillBand`).setAttribute('d', bandPath(side, PILL.R, PILL.h, -PILL.span, PILL.span));
    const [x, y] = onRing(side, PILL.R, 0);
    Object.assign($(`${side}Pill`).style, { left: `${x}px`, top: `${y}px` });
    $(`${side}ArcBand`).setAttribute('d', bandPath(side, ARC.R, ARC.h, ARC.a0, ARC.a0));
    // labels: measured, then spaced evenly along the arc between the back button and the
    // far cap, so long words never collide
    const labels = [...$(`${side}Arc`).querySelectorAll('.arc-label')];
    const widths = labels.map((b) => b.offsetWidth || 8 * b.textContent.length);
    const deg = (px) => THREE.MathUtils.radToDeg(px / ARC.R);
    const start = ARC.a0 + deg(34 + 10), end = ARC.a1 - deg(ARC.h * 0.6);
    const gap = (end - start - widths.reduce((t, w) => t + deg(w), 0)) / (labels.length - 1);
    let at = start;
    labels.forEach((b, i) => {
      const a = at + deg(widths[i]) / 2, [lx, ly] = onRing(side, ARC.R, a);
      at += deg(widths[i]) + gap;
      Object.assign(b.style, { left: `${lx}px`, top: `${ly}px` });
      b.style.setProperty('--r', `${side === 'decode' ? -a : a}deg`);
      b.dataset.a = a;
    });
  }
  // the thin line from each pill towards the object
  const inner = PILL.R - PILL.h;
  document.querySelector('#menuSvg .menu-line').setAttribute('d', `M ${CX} ${CY - inner} V ${CY - inner + 52} M ${CX} ${CY + inner} V ${CY + inner - 52}`);
}
drawMenuShapes();
document.fonts?.ready.then(drawMenuShapes); // re-space the labels once Funnel Sans has loaded

// ---------- the sweep: an arc grows from (or shrinks into) its start cap
const sweep = { encode: { a: ARC.a0, raf: 0 }, decode: { a: ARC.a0, raf: 0 } };
function setSweep(side, a) {
  sweep[side].a = a;
  $(`${side}ArcBand`).setAttribute('d', bandPath(side, ARC.R, ARC.h, ARC.a0, a));
  // a label shows once the sweep has passed it, and hides as it sweeps back
  for (const b of $(`${side}Arc`).querySelectorAll('.arc-label')) b.classList.toggle('lit', a >= +b.dataset.a + 8);
}
function sweepTo(side, to, ms) {
  cancelAnimationFrame(sweep[side].raf);
  const from = sweep[side].a, t0 = performance.now();
  const ease = to > from ? (t) => 1 - (1 - t) ** 3 : (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
  return new Promise((done) => {
    const step = (now) => {
      const t = Math.min(1, (now - t0) / ms);
      setSweep(side, from + (to - from) * ease(t));
      if (t < 1) sweep[side].raf = requestAnimationFrame(step); else done();
    };
    sweep[side].raf = requestAnimationFrame(step);
  });
}

// ---------- the back button: on the arc's start cap (shifted with the morph layout)
function placeBack(side) {
  const [x, y] = capAt(side), [dx, dy] = panelEl.hidden ? [0, 0] : morphBoxOpen() ? MORPH_SHIFT() : [0, 0];
  Object.assign($('back').style, { left: `${x + dx - 34}px`, top: `${y + dy - 34}px` });
}
const showBack = (on) => $('back').classList.toggle('shown', on);

// menu state: closed | open (layer 1) | encode / decode (layer 2) | panel (layer 3)
function setMenu(state) {
  menuEl.dataset.state = state;
  home.dataset.layer = { closed: 0, open: 1, encode: 2, decode: 2, panel: 3 }[state];
  syncShift();
}
let menuGen = 0; // bumped by every transition, so a superseded animation stops quietly

// layer 1 → 2: the chosen arc unsweeps out of its start cap
async function openArc(side) {
  const gen = ++menuGen;
  setSweep(side, ARC.a0);
  placeBack(side);
  setMenu(side);
  showBack(true);
  await sweepTo(side, ARC.a1, 560);
  if (gen === menuGen) $(`${side}Arc`).querySelector('.arc-label')?.focus({ preventScroll: true });
}
// layer 2 → home: the arc sweeps back into its cap, which then fades
async function closeArc() {
  const side = menuEl.dataset.state, gen = ++menuGen;
  if (side !== 'encode' && side !== 'decode') return setMenu('closed');
  await sweepTo(side, ARC.a0, 420);
  if (gen !== menuGen) return;
  showBack(false);
  setMenu('closed');
}
// layer 2 → 3: the arc sweeps into the back button and the toggles fade in
async function openPanel(side, key) {
  const gen = ++menuGen;
  if (menuEl.dataset.state === side) await sweepTo(side, ARC.a0, 420);
  if (gen !== menuGen) return;
  panelEl.dataset.side = side;
  panelEl.dataset.key = key;
  buildPanel(side, key);
  panelEl.hidden = false;
  setMenu('panel');
  placeBack(side);
  showBack(true);
  requestAnimationFrame(() => { panelEl.classList.add('shown'); syncMorphBox(); });
}
// layer 3 → 2: the toggles fade and the back button unsweeps into the arc again
async function backToArc() {
  const side = panelEl.dataset.side, gen = ++menuGen, moved = morphBoxOpen();
  closePanel();
  placeBack(side);
  setSweep(side, ARC.a0);
  setMenu(side);
  if (moved) await new Promise((r) => setTimeout(r, 450)); // let the back button slide home first
  if (gen === menuGen) await sweepTo(side, ARC.a1, 560);
}
function goBack() {
  const st = menuEl.dataset.state;
  if (st === 'panel') backToArc();
  else if (st === 'encode' || st === 'decode') closeArc();
  else if (st === 'open') setMenu('closed');
}
$('back').addEventListener('click', goBack);
addEventListener('keydown', (e) => { if (e.key === 'Escape' && app.dataset.screen === 'home' && !$('dialog').open) goBack(); });

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
  for (const side of ['encode', 'decode']) { cancelAnimationFrame(sweep[side].raf); setSweep(side, ARC.a0); }
  showBack(false);
  setMenu('closed');
}

// ---------- layer 1: click (or right-click, or touch) and hold the object; release on
// a pill to open its arc, anywhere else to cancel. A drag before the hold rotates the object.
const HOLD_MS = 220;
let hold = null;
const objDist = (sx, sy) => Math.hypot(sx - (CX + shift[0]), sy - (CY + shift[1]));
const insideUi = (el) => !!el.closest?.('.pill-label, .arc-label, #back, .orb, .knob, .arc-slider .hit, #dropB, #morphOpts, .run-buttons, #hint, #hud, #drawer, #qmap, dialog, .logo');
// which pill (if any) is at stage point (sx, sy)
function pillAt(sx, sy) {
  for (const side of ['encode', 'decode']) {
    const dx = sx - CX, dy = side === 'encode' ? CY - sy : sy - CY;
    const r = Math.hypot(dx, dy), a = THREE.MathUtils.radToDeg(Math.atan2(dx, dy));
    if (Math.abs(r - PILL.R) <= PILL.h + 16 && Math.abs(a) <= PILL.span + 8) return side;
  }
  return null;
}
function hotPill(side) { for (const s of ['encode', 'decode']) { $(`${s}PillBand`).classList.toggle('hot', s === side); $(`${s}Pill`).classList.toggle('hot', s === side); } }

addEventListener('pointerdown', (e) => {
  if (app.dataset.screen !== 'home' || insideUi(e.target)) return;
  const [sx, sy] = toStage(e.clientX, e.clientY), d = objDist(sx, sy), st = menuEl.dataset.state;
  if (st === 'closed' && d < OBJ_R && (e.button === 0 || e.button === 2)) {
    const h = { id: e.pointerId, x: e.clientX, y: e.clientY, open: false };
    h.timer = setTimeout(() => {
      h.open = true;
      controls.enabled = false; // the hold is for the menu now, not for rotating
      setMenu('open');
      hotPill(pillAt(...toStage(h.x, h.y)));
    }, HOLD_MS);
    hold = h;
  } else if ((st === 'encode' || st === 'decode' || st === 'panel') && d > OBJ_R) {
    // a click well away from the object and the menu goes home
    if (st === 'panel') { closePanel(); showBack(false); ++menuGen; setMenu('closed'); } else closeArc();
  }
});
addEventListener('pointermove', (e) => {
  if (!hold || e.pointerId !== hold.id) return;
  if (!hold.open) {
    if (Math.hypot(e.clientX - hold.x, e.clientY - hold.y) > 8) { clearTimeout(hold.timer); hold = null; } // a drag: rotate instead
    return;
  }
  hotPill(pillAt(...toStage(e.clientX, e.clientY)));
});
function endHold(e, cancelled) {
  if (!hold || e.pointerId !== hold.id) return;
  clearTimeout(hold.timer);
  const opened = hold.open;
  hold = null;
  controls.enabled = true;
  hotPill(null);
  if (!opened) return;
  const side = cancelled ? null : pillAt(...toStage(e.clientX, e.clientY));
  if (side) openArc(side); else setMenu('closed');
}
addEventListener('pointerup', (e) => endHold(e, false));
addEventListener('pointercancel', (e) => endHold(e, true));
// the browser's own menu would interrupt a right-click hold
addEventListener('contextmenu', (e) => { if (app.dataset.screen === 'home' && !e.target.closest('#hud, #drawer, dialog, #qmap, input')) e.preventDefault(); });

// keyboard (and a plain click on the hint): open layer 1, then pick a pill
$('hint').addEventListener('click', () => { if (menuEl.dataset.state === 'closed') { setMenu('open'); $('encodePill').focus(); } });
for (const pill of document.querySelectorAll('.pill-label')) pill.addEventListener('click', () => { if (menuEl.dataset.state === 'open') openArc(pill.dataset.side); });
for (const label of document.querySelectorAll('.arc-label')) {
  label.addEventListener('click', () => openPanel(label.closest('.arc').classList.contains('encode') ? 'encode' : 'decode', label.dataset.panel));
}

// ---------- panel contents
const icon = (src, cls = '') => `<img class="icon ${cls}" src="assets/${src}" alt="">`;
const MORPH_ORB = `<div class="morph"><img src="assets/morph-vector.svg" alt="" style="left:22.2px;top:16.74px;width:18px;height:21.522px">`
  + `<img src="assets/morph-ellipse-a.svg" alt="" style="left:0;top:0;width:32.4px;height:32.283px">`
  + `<img src="assets/morph-ellipse-b.svg" alt="" style="left:31.2px;top:21.52px;width:34.8px;height:33.478px"><span>θ+φ</span></div>`;
const clampFmt = (v) => (v ? `${v.toFixed(2)}π` : P.mode === 'morph' ? '0.20π' : 'off');
const PANELS = {
  load: { title: 'load', key: 'payload', items: [
    { v: 'xyz', html: '(x,y,z)', tip: 'xyz: three values per vertex' },
    { v: 'radial', html: icon('radial-icon.svg', 'r55'), tip: 'radial: one value per vertex (distance from the centre)' },
    { v: 'spherical', html: icon('spherical-icon.svg'), tip: 'spherical: direction in θ and φ, radius in θ' }] },
  mode: { title: 'mode', key: 'mode', items: [
    { v: 'theta', html: 'θ', cls: 'greek', tip: 'shape in θ, read in Z' },
    { v: 'split', html: 'θ+φ', cls: 'greek', tip: 'shape over θ and φ: two values per slot' },
    { v: 'morph', html: MORPH_ORB, cls: 'flat', tip: 'morph: A in θ, B in φ, read along a basis sweep' }] },
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
    { v, html: v.replace(/_/g, '‑<br>'), cls: 'tiny', tip: `Tessa's ${v} dynamic-range restoration` })) },
  frames: { title: 'frames', sliders: [
    { key: 'frames', label: 'frames', min: 2, max: 30, step: 1, fmt: (v) => `${v}` }] },
  shots: { title: 'shots', sliders: [
    { key: 'shots', label: 'shots', min: 8, max: 16, step: 1, fmt: (v) => (2 ** v).toLocaleString() }] },
  placement: { title: 'chip placement', key: 'placement', items: [
    { v: 'quality', html: 'best<br>qubits<br>first', cls: 'tiny', tip: 'the chip\'s best qubits hold the data' },
    { v: 'graph', html: 'Tessa\'s<br>order', cls: 'small', tip: 'Tessa\'s own qubit order' }] },
  backend: { title: 'backend', key: 'backend', items: [
    { v: 'noiseless', html: 'noise‑<br>less', cls: 'small', tip: 'ideal simulator: shot noise only' },
    { v: 'flat noise', html: 'flat<br>noise', cls: 'small', tip: 'IBM-like flat error rates' },
    { v: 'phoenix', html: 'phoe‑<br>nix', cls: 'small', tip: 'ibm_phoenix calibration snapshot' },
    { v: 'fez', html: 'fez', tip: 'ibm_fez calibration snapshot' }],
  sliders: [{ key: 'noise_strength', label: 'noise ×', min: 0, max: 3, step: 0.1, fmt: (v) => `${v.toFixed(1)}×`, inner: true, show: () => P.backend === 'flat noise' }] },
};

// The design's three toggles sit on a circle round the object (centre 720, 732.7;
// radius 490.7; 21.8° apart). More toggles squeeze closer; decode mirrors it below.
const RING = { r: 490.7, cyEnc: 732.7, cyDec: 1024 - 732.7 };
function ringPoint(side, deg, r = RING.r) {
  const a = THREE.MathUtils.degToRad(deg);
  return side === 'encode' ? [CX + r * Math.sin(a), RING.cyEnc - r * Math.cos(a)] : [CX + r * Math.sin(a), RING.cyDec + r * Math.cos(a)];
}
function orbAngles(n) {
  const gap = n <= 3 ? 21.8 : 52 / (n - 1);
  return [...Array(n).keys()].map((i) => (i - (n - 1) / 2) * gap);
}
const orbRadius = (n) => (n <= 3 ? RING.r : 505); // longer rows sit a little further out, clear of the back button

function buildPanel(side, key) {
  const spec = PANELS[key], body = $('panelBody');
  $('panelTitle').textContent = spec.title;
  body.innerHTML = '';
  if (spec.items) {
    const angles = orbAngles(spec.items.length);
    spec.items.forEach((it, i) => {
      const b = document.createElement('button');
      b.className = `orb ${it.cls || ''}`;
      b.innerHTML = it.html;
      b.title = it.tip; b.setAttribute('aria-label', it.tip);
      b.dataset.v = it.v;
      [b.dataset.x, b.dataset.y] = ringPoint(side, angles[i], orbRadius(spec.items.length));
      b.style.transitionDelay = `${120 + 60 * i}ms, ${120 + 60 * i}ms, 0s, 0s, 0s, 0s, 0s`;
      b.addEventListener('click', () => choose(spec.key, it.v));
      body.appendChild(b);
    });
  }
  for (const [k, sl] of (spec.sliders || []).entries()) body.appendChild(arcSlider(side, sl, sl.inner ? (side === 'encode' ? 580 : 375) : (spec.items ? 570 : [RING.r, 570][k])));
  placeOrbs();
  syncPanel();
}

// Toggles (and the back button) move with the object when the morph box needs the room.
function placeOrbs() {
  const [dx, dy] = morphBoxOpen() ? MORPH_SHIFT() : [0, 0];
  for (const b of $('panelBody').querySelectorAll('.orb')) {
    b.style.left = `${+b.dataset.x + dx}px`;
    b.style.top = `${+b.dataset.y + dy}px`;
  }
  if (!panelEl.hidden) placeBack(panelEl.dataset.side);
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
}

function choose(key, v) {
  P[key] = v;
  // the spherical payload fills θ and φ itself, so it only pairs with mode θ
  if (key === 'payload' && v === 'spherical' && P.mode !== 'theta') { P.mode = 'theta'; status('spherical fills θ and φ itself: mode set to θ'); }
  if (key === 'mode' && v !== 'theta' && P.payload === 'spherical') { P.payload = 'radial'; status('θ+φ modes need a radial or xyz load: load set to radial'); }
  if (key === 'mode' && v === 'morph') P.clamp_b = P.clamp; // one clamp for both fields
  if (key === 'mode') { placeOrbs(); syncMorphBox(); syncHud(); }
  syncPanel(); syncStale();
}

// ---------- arc sliders for continuous settings: drag the knob round the ring
function arcSlider(side, sl, r) {
  const A0 = -24, A1 = 30, svgNS = 'http://www.w3.org/2000/svg'; // starts clear of the back button
  const wrap = document.createElement('div');
  wrap.className = 'slider-wrap';
  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('class', 'arc-slider'); svg.setAttribute('viewBox', '0 0 1440 1024');
  const arcPath = (a0, a1) => {
    const [x0, y0] = ringPoint(side, a0, r), [x1, y1] = ringPoint(side, a1, r);
    return `M ${x0} ${y0} A ${r} ${r} 0 0 ${side === 'encode' ? 1 : 0} ${x1} ${y1}`;
  };
  const mk = (cls, d) => { const p = document.createElementNS(svgNS, 'path'); p.setAttribute('class', cls); p.setAttribute('d', d); svg.appendChild(p); return p; };
  mk('track', arcPath(A0, A1));
  const fill = mk('fill', arcPath(A0, A0));
  const hit = mk('hit', arcPath(A0, A1));
  const label = document.createElementNS(svgNS, 'text');
  const [lx, ly] = ringPoint(side, A0 + 6, r + 28); // just outside the track's start
  label.setAttribute('x', lx); label.setAttribute('y', ly + 5); label.setAttribute('text-anchor', 'middle');
  const labelText = () => (typeof sl.label === 'function' ? sl.label() : sl.label);
  label.textContent = labelText();
  svg.appendChild(label);
  const knob = document.createElement('button');
  knob.className = 'knob';
  knob.setAttribute('role', 'slider');
  knob.setAttribute('aria-label', labelText());
  knob.setAttribute('aria-valuemin', sl.min); knob.setAttribute('aria-valuemax', sl.max);
  wrap.append(svg, knob);
  const angleOf = (v) => A0 + ((A1 - A0) * (v - sl.min)) / (sl.max - sl.min);
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
    const shown = !sl.show || sl.show();
    wrap.style.display = shown ? '' : 'none';
    const off = sl.off && sl.off();
    wrap.style.opacity = off ? 0.35 : 1;
    knob.title = off || '';
    label.textContent = labelText();
    const a = angleOf(P[sl.key]), [x, y] = ringPoint(side, a, r);
    knob.style.left = `${x}px`; knob.style.top = `${y}px`;
    knob.textContent = sl.fmt(P[sl.key]);
    knob.setAttribute('aria-valuenow', P[sl.key]);
    fill.setAttribute('d', arcPath(A0, a));
  };
  const fromPointer = (e) => {
    const [sx, sy] = toStage(e.clientX, e.clientY), cy = side === 'encode' ? RING.cyEnc : RING.cyDec;
    let a = THREE.MathUtils.radToDeg(Math.atan2(sx - CX, side === 'encode' ? cy - sy : sy - cy));
    a = Math.min(A1, Math.max(A0, a));
    set(sl.min + ((a - A0) / (A1 - A0)) * (sl.max - sl.min));
  };
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
function syncMorphBox() {
  const on = !panelEl.hidden && panelEl.dataset.key === 'mode' && P.mode === 'morph';
  const box = $('dropB');
  if (on) { box.hidden = false; requestAnimationFrame(() => box.classList.add('shown')); } else box.classList.remove('shown');
  box.classList.toggle('ready', !!bLabel());
  const opts = $('morphOpts');
  if (on) { opts.hidden = false; requestAnimationFrame(() => opts.classList.add('shown')); } else opts.classList.remove('shown');
  for (const c of opts.querySelectorAll('.chip')) c.setAttribute('aria-pressed', P[c.dataset.opt]);
  const aDirs = opts.querySelector('[data-opt="a_dirs"]');
  aDirs.disabled = P.payload !== 'radial';
  aDirs.title = P.payload === 'radial' ? "Radial load: keep A's directions; only the radii morph" : "A's directions only: radial load only";
  $('morph_t').value = P.morph_t;
  $('morph_t').nextElementSibling.textContent = P.morph_t.toFixed(2);
  $('morphNote').textContent = (P.interference ? 'Interference: t turns the selector qubits from A (0) to B (1).' : 'Basis sweep: t turns the readout from Z (A) to X (B).')
    + (P.hold_t ? ` Every frame reads at t = ${P.morph_t.toFixed(2)}.` : ` Frames run t 0 → 1; one ▶ reads at t = ${P.morph_t.toFixed(2)}.`);
  $('dropBText').textContent = bLabel() ? `B: ${bLabel()} ✓ — press ▶ to morph (drop to replace)` : 'drop second mesh (.obj)';
  for (const b of box.querySelectorAll('[data-shape-b]')) b.classList.toggle('on', !S.objTextB && b.dataset.shapeB === P.shapeB);
  placeOrbs();
  syncShift();
}
for (const c of document.querySelectorAll('#morphOpts .chip')) {
  c.addEventListener('click', () => { P[c.dataset.opt] = !P[c.dataset.opt]; syncMorphBox(); syncStale(); });
}
$('morph_t').addEventListener('input', () => { P.morph_t = +$('morph_t').value; syncMorphBox(); syncStale(); });
$('dropB').addEventListener('click', (e) => {
  const b = e.target.closest('[data-shape-b]');
  if (b) { e.stopPropagation(); P.shapeB = b.dataset.shapeB; S.objTextB = S.objNameB = null; syncMorphBox(); syncStale(); return; }
  $('fileB').click();
});
$('fileB').addEventListener('change', async (e) => { if (e.target.files[0]) await loadFileB(e.target.files[0]); e.target.value = ''; });
async function loadFileB(file) {
  S.objTextB = await file.text(); S.objNameB = file.name; P.shapeB = null;
  status(`B: ${file.name} loaded. Press ▶ (frames) to morph ${objLabel()} → ${file.name}`);
  syncMorphBox(); syncStale();
}

// ---------- loading objects: presets, file picker, drag and drop
async function chooseShape(name) {
  P.shape = name; S.objText = S.objName = null; P.resolution = name === 'apple' || name === 'suzanne' ? 2 : 0;
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
    S.objText = text; S.objName = file.name; P.resolution = 2;
    syncDrawer();
    await enterHome();
  } catch (err) { status(`⚠ ${err.message}`, true); } finally { S.loading = false; }
}
async function enterHome() {
  setScreen('home');
  S.loading = true;
  try {
    status(`Loading ${objLabel()}…`);
    await loadMesh();
    status(`${objLabel()} ready. Press ▶ to encode and read it, or ${HOVER ? 'click' : 'press'} and hold it for settings.`);
  } catch (err) { status(`⚠ ${err.message}`, true); } finally { S.loading = false; }
}
for (const b of document.querySelectorAll('[data-shape]')) b.addEventListener('click', () => chooseShape(b.dataset.shape));
$('dropObj').addEventListener('click', () => $('fileA').click());
$('fileA').addEventListener('change', async (e) => { if (e.target.files[0]) await loadFileA(e.target.files[0]); e.target.value = ''; });
$('logo').addEventListener('click', () => setScreen('landing'));

let dragDepth = 0;
addEventListener('dragenter', (e) => { if (!Array.from(e.dataTransfer?.types || []).includes('Files')) return; e.preventDefault(); dragDepth++; app.classList.add('dragging'); $('dropOverlay').hidden = false; });
addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; app.classList.remove('dragging'); $('dropOverlay').hidden = true; $('dropB').classList.remove('over'); } });
addEventListener('dragover', (e) => {
  e.preventDefault();
  $('dropB').classList.toggle('over', !!e.target.closest?.('#dropB'));
  $('dropOverlay').querySelector('p').textContent = e.target.closest?.('#dropB') ? 'drop mesh B' : 'drop to load';
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

// ---------- run buttons
$('run').addEventListener('click', () => run(1));
// frames show how the readout moves the mesh, so they switch the colour to error
function useErrorColour() {
  if (P.colour === 'error') return;
  P.colour = 'error';
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
function syncPlay() {
  $('play').classList.toggle('playing', S.playing);
  home.classList.toggle('quiet', S.playing || S.busy); // no "click and hold" prompt while frames run or play
}
// the run button glows while the settings differ from the last run
function syncStale() { $('run').classList.toggle('stale', !!S.mesh && S.run !== runKey()); }

// ---------- HUD and drawer
function syncHud() {
  const morph = P.mode === 'morph';
  $('edges').disabled = S.busy || morph || !S.mesh;
  $('edges').title = morph ? 'Edge readouts compare against one fixed mesh: not in morph mode' : '';
  $('curves').disabled = S.busy || morph || !S.mesh;
  $('curves').title = morph ? 'Curves compare against one fixed mesh: not in morph mode' : '';
  $('why').disabled = S.busy || !S.mesh || (morph && !bLabel());
  $('obj').disabled = !S.seq.length;
  $('qmapOn').disabled = !S.data;
  $('viewToggle').disabled = false;
}
$('why').onclick = why;
$('curves').onclick = curves;
$('edges').onclick = readEdges;
$('obj').onclick = downloadObj;
$('qmapOn').onclick = () => { qmapOn = !qmapOn; $('qmapOn').setAttribute('aria-pressed', qmapOn); setFocus(null); };
$('viewToggle').onclick = () => {
  const open = $('drawer').hidden;
  $('drawer').hidden = !open;
  $('viewToggle').setAttribute('aria-expanded', open);
};

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
    P[k] = k === 'resolution' ? +b.dataset.value : b.dataset.value;
    syncDrawer();
    if (k === 'resolution') {
      try { await loadMesh(); status(`${objLabel()}: ${['low', 'mid', 'high'][P.resolution]} detail, ${S.mesh.verts.length} vertices`); } catch (err) { status(`⚠ ${err.message}`, true); }
      syncStale();
    } else rebuild();
    if (S.data) showInfo(S.data, params());
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
  renderer.render(scene, camera);
}

// touch screens have no hover: say "tap" instead
if (!HOVER) $('hint').textContent = 'press and hold';
// compact windows fold the tools into one button, and the stats behind a tap
$('hudMore').addEventListener('click', () => {
  const open = !$('hud').classList.contains('open');
  $('hud').classList.toggle('open', open);
  $('hudMore').setAttribute('aria-expanded', open);
});
$('hud').addEventListener('click', (e) => {
  if (compact && e.target.closest('button') && e.target.id !== 'hudMore') { $('hud').classList.remove('open'); $('hudMore').setAttribute('aria-expanded', false); }
});
$('info').addEventListener('click', () => { if (compact) $('info').classList.toggle('open'); });

window.tessa3d = { state: S, settings: P }; // read-only handle for debugging in the console
fitStage();
syncDrawer(); syncHud();
requestAnimationFrame(frame);
status('Drop an .obj file or pick a shape.');
// warm the function up while the landing page shows (a cold start can take ~10 s)
fetch('/api/tessa').catch(() => {});
