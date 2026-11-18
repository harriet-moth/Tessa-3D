// Tessa Ripple web: three.js viewer and controls. All quantum work runs in the
// Vercel Python function (api/tessa.py → tessa_ripple/bridge.py → Tessa); this file
// only draws, blends, interpolates and colours what it returns.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const val = (id) => $(id).value;
const num = (id) => +$(id).value;
const S = { data: null, run: null, meshKey: null, edges: null, seq: [], step: 0, playing: false, acc: 0 };

// ---------- scene
const viewer = $('viewer');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
viewer.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
camera.up.set(0, 0, 1);
camera.position.set(1.6, -5.4, 1.5);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x445066, 1.7));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(3, -3, 5);
scene.add(sun);

const geom = new THREE.BufferGeometry();
const material = new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, side: THREE.DoubleSide, roughness: 0.65, transparent: true });
const mesh = new THREE.Mesh(geom, material);
const wireGeom = new THREE.BufferGeometry();
const wire = new THREE.LineSegments(wireGeom, new THREE.LineBasicMaterial({ color: 0x1d2433, transparent: true, opacity: 0.22 }));
const edgeGeom = new THREE.BufferGeometry();
const edgeLines = new THREE.LineSegments(edgeGeom, new THREE.LineBasicMaterial({ vertexColors: true }));
const marker = new THREE.Mesh(new THREE.SphereGeometry(0.03, 12, 8), new THREE.MeshBasicMaterial({ color: 0xd0342c }));
edgeLines.visible = marker.visible = false;
scene.add(mesh, wire, edgeLines, marker);

function resize() {
  const w = viewer.clientWidth, h = viewer.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewer);

// ---------- colour ramps
const stops = (hexes) => hexes.map((c) => new THREE.Color(c));
const MAGMA = stops(['#fcfdbf', '#fe9f6d', '#de4968', '#8c2981', '#3b0f70']);
const VIRIDIS = stops(['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']);
function ramp(cs, t) {
  t = Math.min(1, Math.max(0, t));
  const x = t * (cs.length - 1), i = Math.min(Math.floor(x), cs.length - 2);
  return cs[i].clone().lerp(cs[i + 1], x - i);
}
const PLAIN = new THREE.Color('#8fb0d6');

// ---------- API
function params() {
  return {
    shape: val('shape'), resolution: num('resolution'), payload: val('payload'), layout: val('layout'),
    mode: val('mode'), clamp: num('clamp') * Math.PI || null, clamp_b: num('clamp_b') * Math.PI || null,
    compression: num('compression'), quant_bits: num('quant_bits') || null, headroom: num('headroom'),
    backend: val('backend'), shots: 2 ** num('shots'), noise_strength: num('noise_strength'),
    seed: num('seed'), drr: val('drr'),
  };
}
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
const status = (msg) => { $('status').textContent = msg; };
async function busy(label, fn) {
  const buttons = document.querySelectorAll('.buttons button');
  buttons.forEach((b) => { b.disabled = true; });
  status(label);
  try { await fn(); } catch (err) { status(`⚠ ${err.message}`); } finally { buttons.forEach((b) => { b.disabled = false; }); }
}

// ---------- mesh
function buildMesh(d) {
  const n = d.verts.length;
  geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  geom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  const idx = [];
  for (const f of d.faces) for (let k = 1; k < f.length - 1; k++) idx.push(f[0], f[k], f[k + 1]);
  geom.setIndex(idx);
  const seen = new Set();
  S.wire = [];
  for (const f of d.faces) {
    for (let k = 0; k < f.length; k++) {
      const a = f[k], b = f[(k + 1) % f.length], key = Math.min(a, b) * 1e6 + Math.max(a, b);
      if (!seen.has(key)) { seen.add(key); S.wire.push(a, b); }
    }
  }
  wireGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(S.wire.length * 3), 3));
}

function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randomMask(n, frac, seed) {
  const r = mulberry32(seed + 1);
  const order = [...Array(n).keys()].map((i) => [r(), i]).sort((a, b) => a[0] - b[0]);
  const m = new Float32Array(n);
  for (let j = 0; j < Math.round(frac * n); j++) m[order[j][1]] = 1;
  return m;
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

// Classical layer: gain, mask, interpolation and colour, applied to cached frames.
function rebuild() {
  if (!S.data) return;
  const n = S.data.verts.length, orig = S.orig;
  let keys = S.frames;
  if (val('view') === 'material') {
    const g = num('gain'), m = randomMask(n, num('mask') / 100, num('seed'));
    keys = S.frames.map((f) => {
      const o = new Float32Array(f.length);
      for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) {
        const j = 3 * i + c;
        o[j] = orig[j] + g * m[i] * (f[j] - orig[j]);
      }
      return o;
    });
  }
  keys = ($('orig').checked ? [orig] : []).concat(keys);
  S.seq = keys.length === 1 ? keys : timeline(keys, num('between'), $('loop').checked);
  S.step = Math.min(S.step, S.seq.length - 1);
  colourise();
  draw();
}

function colourise() {
  const d = S.data, n = d.verts.length, col = geom.attributes.color.array, mode = val('colour');
  const set = (i, c) => { col[3 * i] = c.r; col[3 * i + 1] = c.g; col[3 * i + 2] = c.b; };
  if (mode === 'error') {
    const err = new Float32Array(n);
    for (const f of S.frames) for (let i = 0; i < n; i++) {
      err[i] += Math.hypot(f[3 * i] - S.orig[3 * i], f[3 * i + 1] - S.orig[3 * i + 1], f[3 * i + 2] - S.orig[3 * i + 2]) / S.frames.length;
    }
    const top = Math.max(...err) || 1;
    for (let i = 0; i < n; i++) set(i, ramp(MAGMA, 0.1 + 0.8 * err[i] / top));
    S.colourLabel = `error per vertex (dark = ${top.toFixed(3)})`;
  } else if (mode === 'qubit') {
    for (let i = 0; i < n; i++) set(i, new THREE.Color().setHSL((d.qubits[i] * 0.618034) % 1, 0.55, 0.6));
    S.colourLabel = 'data qubit holding each vertex';
  } else if (mode === 'phi' && d.colour) {
    const hue = d.colours ? d.colour.map((_, i) => d.colours.reduce((s, c) => s + c[i], 0) / d.colours.length) : d.colour;
    for (let i = 0; i < n; i++) set(i, new THREE.Color().setHSL(hue[i], 0.75, 0.55));
    S.colourLabel = 'vertex colour decoded from φ';
  } else {
    for (let i = 0; i < n; i++) set(i, PLAIN);
    S.colourLabel = mode === 'phi' ? 'plain (φ colour needs pair mode)' : 'plain';
  }
  geom.attributes.color.needsUpdate = true;
}

function copyEnds(target, pairs, v) {
  pairs.forEach((vi, j) => { target[3 * j] = v[3 * vi]; target[3 * j + 1] = v[3 * vi + 1]; target[3 * j + 2] = v[3 * vi + 2]; });
}

function draw() {
  if (!S.seq.length) return;
  const v = S.seq[S.step];
  geom.attributes.position.array.set(v);
  geom.attributes.position.needsUpdate = true;
  geom.computeBoundingSphere();
  copyEnds(wireGeom.attributes.position.array, S.wire, v);
  wireGeom.attributes.position.needsUpdate = true;
  if (S.edges) {
    copyEnds(edgeGeom.attributes.position.array, S.edges.ends, v);
    edgeGeom.attributes.position.needsUpdate = true;
  }
  edgeLines.visible = !!S.edges;
  material.opacity = S.edges ? 0.35 : 1;
  marker.visible = false;
  $('tip').hidden = true;
  const head = val('view') === 'fidelity' ? 'Fidelity · gain 1 (true decoded positions)'
    : `Material · gain ${num('gain').toFixed(1)}× (noise exaggerated)`;
  const stale = S.run && JSON.stringify(params()) !== S.run ? ' · settings changed: press Run' : '';
  $('caption').textContent = `${head} · mean err ${S.data.error.toFixed(4)} of bbox · step ${S.step + 1}/${S.seq.length} · colour: ${S.colourLabel}`
    + (S.edges ? ' · edges: length read from coherences' : '') + stale;
}

function showInfo(d, p) {
  const s = d.stats;
  $('info').textContent = [
    `${p.shape} · ${d.verts.length} verts · ${p.payload}, ${p.layout} layout`,
    s.graph,
    `${s.values} values on ${s.qubits} qubits = ${s.ratio.toFixed(2)} / qubit`,
    `(angle encoding would need ${s.values} qubits)`,
    `capacity used ${s.values}/${s.capacity}`,
    `depth ${s.depth} · 2q gates ${s.twoq}` + (p.mode === 'pair' ? ` (θ alone ${s.twoq_single})` : ''),
  ].join('\n');
}

// ---------- actions
async function run(frames) {
  const p = params();
  await busy(frames > 1 ? `Running ${frames} readouts on ${p.backend}…` : `Running on ${p.backend}…`, async () => {
    const d = await call('run', { frames });
    const meshKey = `${p.shape}/${p.resolution}`;
    if (meshKey !== S.meshKey) { buildMesh(d); S.meshKey = meshKey; setEdges(null); }
    Object.assign(S, { data: d, run: JSON.stringify(p), orig: new Float32Array(d.verts.flat()),
      frames: d.frames.map((f) => new Float32Array(f.flat())), step: 0 });
    showInfo(d, p);
    rebuild();
    S.playing = d.frames.length > 1;
    if (!S.playing) { S.step = S.seq.length - 1; draw(); }
    const cut = d.frames.length < d.requested ? ` (stopped at the time limit; asked for ${d.requested})` : '';
    status(`${d.frames.length} readout(s) in ${d.seconds.toFixed(1)} s on ${p.backend}${cut}`);
  });
}

function setEdges(r) {
  S.edges = r;
  $('edges').textContent = r ? 'Hide edges' : 'Read edges';
  if (!r) return;
  const lo = Math.min(...r.exact), span = Math.max(...r.exact) - lo || 1;
  r.ends = r.edges.flat();
  edgeGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(r.ends.length * 3), 3));
  const col = new Float32Array(r.ends.length * 3);
  r.measured.forEach((m, k) => {
    const c = ramp(VIRIDIS, (m - lo) / span);
    for (const j of [2 * k, 2 * k + 1]) { col[3 * j] = c.r; col[3 * j + 1] = c.g; col[3 * j + 2] = c.b; }
  });
  edgeGeom.setAttribute('color', new THREE.BufferAttribute(col, 3));
}

async function readEdges() {
  if (S.edges) { setEdges(null); status('edge overlay hidden'); return draw(); }
  if (`${val('shape')}/${val('resolution')}` !== S.meshKey) return status('⚠ the shape changed: press Run first');
  S.playing = false;
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
    const shots = 2 ** num('shots');
    const rows = r.rows.map((x) => `<tr><td>${x.name}</td><td>${x.qubits}</td><td>${x.fits ? 'yes' : 'no'}</td><td>${x.twoq}</td><td>${x.depth}</td><td>${x.swaps}</td><td>${x.settings}</td><td>${x.configs}</td><td>${Math.floor(shots / x.configs)}</td></tr>`).join('');
    openDialog(`<h2>Why Tessa</h2>
      <p>${r.stats.values} values (${r.payload}, ${r.mode}) transpiled onto the ibm_phoenix snapshot (120 qubits, square lattice).</p>
      <table><tr><th>encoding</th><th>qubits</th><th>fits on chip</th><th>2q gates</th><th>depth</th><th>SWAPs</th><th>settings</th><th>outcomes / cell</th><th>shots / value (${shots})</th></tr>${rows}</table>
      <p class="note">Generic random data in every row, no compression. SWAPs ≈ extra CZs from routing ÷ 3. Each shot is split over a cell's outcomes, so each value gets about shots ÷ outcomes. Amplitude: Z readout gives magnitudes only. With a chip backend, Tessa's row is on that chip; the baselines stay on Phoenix.</p>`);
    status('Why Tessa: resource table opened');
  });
}

function chart({ title, xlabel, ylabel, series, logx = false, logy = false }) {
  const W = 420, H = 260, m = { l: 58, r: 12, t: 26, b: 42 };
  const fx = logx ? Math.log10 : (x) => x, fy = logy ? Math.log10 : (y) => y;
  const pts = series.flatMap((s) => s.points);
  const [x0, x1] = [Math.min(...pts.map((p) => fx(p[0]))), Math.max(...pts.map((p) => fx(p[0])))];
  let [y0, y1] = [Math.min(...pts.map((p) => fy(p[1]))), Math.max(...pts.map((p) => fy(p[1])))];
  if (y1 === y0) { y0 -= 1; y1 += 1; }
  const X = (x) => m.l + ((fx(x) - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const Y = (y) => H - m.b - ((fy(y) - y0) / (y1 - y0)) * (H - m.t - m.b);
  const fmt = (v) => (Math.abs(v) >= 1000 ? v.toExponential(0) : +v.toPrecision(2)).toString();
  const ticks = (a, b, log) => [0, 1, 2, 3].map((k) => { const t = a + (k / 3) * (b - a); return log ? 10 ** t : t; });
  const lines = series.map((s) => `<polyline fill="none" stroke="${s.color}" stroke-width="2" ${s.dash ? 'stroke-dasharray="4 3"' : ''} points="${s.points.map((p) => `${X(p[0])},${Y(p[1])}`).join(' ')}"/>`
    + s.points.map((p) => (s.dash ? '' : `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="3" fill="${s.color}"/>`)).join('')).join('');
  const legend = series.map((s, i) => `<text x="${W - m.r - 4}" y="${m.t + 12 + 14 * i}" text-anchor="end" style="fill:${s.color}">${s.label}</text>`).join('');
  const xt = ticks(x0, x1, logx).map((t) => `<text x="${X(t)}" y="${H - m.b + 16}" text-anchor="middle">${fmt(t)}</text>`).join('');
  const yt = ticks(y0, y1, logy).map((t) => `<text x="${m.l - 6}" y="${Y(t) + 4}" text-anchor="end">${fmt(t)}</text>`).join('');
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}">
    <text x="${W / 2}" y="16" text-anchor="middle" style="font-weight:600">${title}</text>
    <line x1="${m.l}" y1="${H - m.b}" x2="${W - m.r}" y2="${H - m.b}" stroke="currentColor" opacity=".3"/>
    <line x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${H - m.b}" stroke="currentColor" opacity=".3"/>
    ${xt}${yt}${lines}${legend}
    <text x="${(W + m.l) / 2}" y="${H - 6}" text-anchor="middle">${xlabel}</text>
    <text transform="translate(14 ${(H - m.b + m.t) / 2}) rotate(-90)" text-anchor="middle">${ylabel}</text></svg>`;
}

async function curves() {
  await busy('Curves: 15 readouts on the current backend…', async () => {
    const r = await call('curves');
    const A = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#3d5a98';
    const B = '#e07b39', G = '#999999';
    const shots = r.shots.map(([s, e]) => [s, e]);
    const guide = shots.map(([s]) => [s, shots[0][1] * Math.sqrt(shots[0][0] / s)]);
    const err = (k) => r[k].map(([c, , e]) => [c, e]);
    const depth = (k) => r[k].map(([c, d]) => [c, d]);
    openDialog(`<h2>Fidelity curves</h2><div class="charts">
      ${chart({ title: 'error vs shots', xlabel: 'shots', ylabel: 'mean error (of bbox)', logx: true, logy: true,
        series: [{ label: 'measured', color: A, points: shots }, { label: '1/√shots', color: G, dash: true, points: guide }] })}
      ${chart({ title: 'error vs compression', xlabel: 'compression %', ylabel: 'mean error',
        series: [{ label: 'mesh layout', color: A, points: err('mesh') }, { label: 'naive layout', color: B, points: err('naive') }] })}
      ${chart({ title: 'depth vs compression', xlabel: 'compression %', ylabel: 'circuit depth',
        series: [{ label: 'mesh layout', color: A, points: depth('mesh') }, { label: 'naive layout', color: B, points: depth('naive') }] })}
      </div><p class="note">Error is the mean coordinate error as a fraction of the bounding box. Compression drops the smallest Walsh–Hadamard coefficients of each Tessa cell.</p>`);
    status('Fidelity curves opened');
  });
}

function downloadObj() {
  if (!S.seq.length) return;
  const v = S.seq[S.step], lines = [];
  for (let i = 0; i < v.length; i += 3) lines.push(`v ${v[i].toFixed(6)} ${v[i + 2].toFixed(6)} ${(-v[i + 1]).toFixed(6)}`); // Y-up
  for (const f of S.data.faces) lines.push(`f ${f.map((i) => i + 1).join(' ')}`);
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lines.join('\n') + '\n'], { type: 'text/plain' }));
  a.download = `tessa-ripple-${val('shape')}-step-${S.step + 1}.obj`;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---------- hover (paused only)
const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
function hideTip() { marker.visible = false; $('tip').hidden = true; }
renderer.domElement.addEventListener('pointermove', (e) => {
  if (S.playing || !S.data) return hideTip();
  const r = renderer.domElement.getBoundingClientRect();
  ptr.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ptr, camera);
  const hit = ray.intersectObject(mesh)[0];
  if (!hit) return hideTip();
  const v = S.seq[S.step], o = S.orig;
  const at = (i, a) => new THREE.Vector3(a[3 * i], a[3 * i + 1], a[3 * i + 2]);
  const i = [hit.face.a, hit.face.b, hit.face.c].sort((p, q) => at(p, v).distanceTo(hit.point) - at(q, v).distanceTo(hit.point))[0];
  const cur = at(i, v), org = at(i, o), d = cur.clone().sub(org);
  const f = (p) => `(${[p.x, p.y, p.z].map((c) => (c >= 0 ? '+' : '') + c.toFixed(4)).join(', ')})`;
  const gain = val('view') === 'fidelity' ? 1 : num('gain');
  marker.position.copy(cur);
  marker.visible = true;
  const tip = $('tip'), stage = viewer.parentElement.getBoundingClientRect();
  tip.textContent = `vertex ${i} on data qubit ${S.data.qubits[i]}  (gain ${gain.toFixed(1)}×, step ${S.step + 1})\noriginal ${f(org)}\ncurrent  ${f(cur)}\ndelta    ${f(d)}\n|delta|  ${d.length().toFixed(4)}`;
  tip.style.left = `${Math.min(e.clientX - stage.left + 14, stage.width - 330)}px`;
  tip.style.top = `${e.clientY - stage.top + 10}px`;
  tip.hidden = false;
});
renderer.domElement.addEventListener('pointerleave', hideTip);

// ---------- wiring
document.querySelectorAll('input[type=range]').forEach((el) => {
  const out = el.nextElementSibling;
  const show = () => { if (out) out.textContent = el.id === 'shots' ? 2 ** +el.value : el.id === 'quant_bits' && +el.value === 0 ? 'off' : el.value; };
  el.addEventListener('input', show);
  show();
});
for (const id of ['shape', 'resolution', 'payload', 'mode', 'layout', 'clamp', 'clamp_b', 'compression', 'quant_bits', 'headroom', 'backend', 'shots', 'noise_strength', 'drr']) {
  $(id).addEventListener('input', draw);
}
for (const id of ['view', 'colour', 'gain', 'mask', 'between', 'loop', 'orig', 'seed']) $(id).addEventListener('input', rebuild);
$('run').onclick = () => run(1);
$('runFrames').onclick = () => run(num('frames'));
$('play').onclick = () => { S.playing = !S.playing; hideTip(); if (S.playing && S.step + 1 >= S.seq.length) S.step = 0; };
$('edges').onclick = readEdges;
$('why').onclick = why;
$('curves').onclick = curves;
$('obj').onclick = downloadObj;

let last = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  controls.update();
  if (S.playing && S.seq.length > 1) {
    S.acc = Math.min(S.acc + (now - last) / 1000, 1);
    const dt = 1 / num('fps');
    let moved = false;
    while (S.acc >= dt) {
      S.acc -= dt;
      if (S.step + 1 < S.seq.length) S.step++;
      else if ($('loop').checked) S.step = 0;
      else { S.playing = false; break; }
      moved = true;
    }
    if (moved) draw();
  }
  last = now;
  renderer.render(scene, camera);
}
resize();
requestAnimationFrame(frame);
status('Loading Tessa on the server (a cold start can take ~10 s)…');
run(1);
