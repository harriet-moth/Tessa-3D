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
// Unlit by default, so every face shows its mapped colour exactly; lighting would darken
// faces turned away from the light, which in error mode reads as "worse".
const flat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, transparent: true });
const lit = new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, side: THREE.DoubleSide, roughness: 0.65, transparent: true });
const mesh = new THREE.Mesh(geom, flat);
const wireGeom = new THREE.BufferGeometry();
const wire = new THREE.LineSegments(wireGeom, new THREE.LineBasicMaterial({ color: 0x1d2433, transparent: true, opacity: 0.22 }));
const edgeGeom = new THREE.BufferGeometry();
const edgeLines = new THREE.LineSegments(edgeGeom, new THREE.LineBasicMaterial({ vertexColors: true }));
const marker = new THREE.Mesh(new THREE.SphereGeometry(0.03, 12, 8), new THREE.MeshBasicMaterial({ color: 0xd0342c }));
// Faces get one flat colour each (the mean of their vertices, as in the desktop app);
// each vertex's exact colour is shown on its own dot, so no value is smeared across faces.
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
const dotTexture = discTexture('#fff', 'rgba(20, 22, 30, 0.75)', 7);
const dotGeom = new THREE.BufferGeometry();
const dots = new THREE.Points(dotGeom, new THREE.PointsMaterial({ vertexColors: true, map: dotTexture, alphaTest: 0.5, size: 0.06 }));
// Qubit map: every vertex on the hovered data qubit gets an orange dot.
const mateGeom = new THREE.BufferGeometry();
const mates = new THREE.Points(mateGeom, new THREE.PointsMaterial({ map: discTexture('rgb(230, 150, 0)', '#14161e', 12), alphaTest: 0.5, size: 0.08 }));
edgeLines.visible = marker.visible = mates.visible = false;
scene.add(mesh, wire, edgeLines, dots, marker, mates);

function resize() {
  const w = viewer.clientWidth, h = viewer.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewer);

// ---------- colour definitions, mirrored in tessa_ripple/app.py so both apps match
const stops = (hexes) => hexes.map((h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16) / 255)); // sRGB
const ERROR_HEX = ['#fcfdbf', '#fd9668', '#cf4070', '#752181', '#1a1042']; // magma_r on [0, 0.88]; light = no error
const EDGE_HEX = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']; // viridis
const ERROR = stops(ERROR_HEX), EDGE = stops(EDGE_HEX);
const GOLDEN = 0.618034;
// HSL is given in sRGB (what the eye sees), not three.js's linear working space.
const hsl = (h, s, l) => new THREE.Color().setHSL(((h % 1) + 1) % 1, s, l, THREE.SRGBColorSpace);
const hueCss = (h) => `hsl(${(h * 360).toFixed(0)} 75% 55%)`;
function ramp(cs, t) { // blends in sRGB, like CSS gradients and matplotlib, so mesh and legend agree
  t = Math.min(1, Math.max(0, t));
  const x = t * (cs.length - 1), i = Math.min(Math.floor(x), cs.length - 2), f = x - i;
  const [r, g, b] = cs[i].map((c, k) => c + (cs[i + 1][k] - c) * f);
  return new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);
}
const PLAIN = new THREE.Color('#8fb0d6');

// ---------- API
function params() {
  return {
    shape: val('shape'), resolution: num('resolution'), payload: val('payload'), layout: val('layout'),
    mode: val('mode'), clamp: num('clamp') * Math.PI || null, clamp_b: num('clamp_b') * Math.PI || null,
    compression: num('compression'), quant_bits: num('quant_bits') || null, headroom: num('headroom'),
    backend: val('backend'), placement: val('placement'), shots: 2 ** num('shots'), noise_strength: num('noise_strength'),
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
  const n = d.verts.length, corners = [];
  for (const f of d.faces) for (let k = 1; k < f.length - 1; k++) corners.push(f[0], f[k], f[k + 1]);
  S.corner = Int32Array.from(corners); // non-indexed: each triangle corner has its own (flat) colour
  geom.setIndex(null);
  geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(corners.length * 3), 3));
  geom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(corners.length * 3), 3));
  dotGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  dotGeom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  const seen = new Set();
  S.wire = [];
  for (const f of d.faces) {
    for (let k = 0; k < f.length; k++) {
      const a = f[k], b = f[(k + 1) % f.length], key = Math.min(a, b) * 1e6 + Math.max(a, b);
      if (!seen.has(key)) { seen.add(key); S.wire.push(a, b); }
    }
  }
  wireGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(S.wire.length * 3), 3));
  let len = 0; // dot size follows the mesh's mean edge length
  for (let j = 0; j < S.wire.length; j += 2) {
    const a = S.wire[j], b = S.wire[j + 1];
    len += Math.hypot(...[0, 1, 2].map((c) => d.verts[a][c] - d.verts[b][c]));
  }
  dots.material.size = 0.6 * len / (S.wire.length / 2);
  mates.material.size = 2 * dots.material.size;
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
  const d = S.data, n = d.verts.length, col = dotGeom.attributes.color.array, mode = val('colour');
  const set = (i, c) => { col[3 * i] = c.r; col[3 * i + 1] = c.g; col[3 * i + 2] = c.b; };
  if (mode === 'error') {
    const err = new Float32Array(n);
    for (const f of S.frames) for (let i = 0; i < n; i++) {
      err[i] += Math.hypot(f[3 * i] - S.orig[3 * i], f[3 * i + 1] - S.orig[3 * i + 1], f[3 * i + 2] - S.orig[3 * i + 2]) / S.frames.length;
    }
    const top = Math.max(...err) || 1;
    for (let i = 0; i < n; i++) set(i, ramp(ERROR, err[i] / top));
    S.colourLabel = `error per vertex (mean of ${S.frames.length} readouts)`;
    legend('faceLegend', ERROR_HEX, '0', top.toFixed(4), 'vertex error (distance, mesh units)');
  } else if (mode === 'qubit') {
    const used = [...new Set(d.qubits)].sort((a, b) => a - b), rank = new Map(used.map((q, r) => [q, r]));
    for (let i = 0; i < n; i++) set(i, hsl(rank.get(d.qubits[i]) * GOLDEN, 0.6, 0.55));
    S.colourLabel = `${d.split ? "data qubit holding each vertex's x (naive layout splits x, y, z)" : 'data qubit holding each vertex'}, ${used.length} qubits`;
    $('faceLegend').hidden = false;
    $('faceLegend').innerHTML = `<div class="swatches">${used.map((q, r) => `<i title="qubit ${q}" style="background:hsl(${((r * GOLDEN) % 1 * 360).toFixed(0)} 60% 55%)"></i>`).join('')}</div>one colour per data qubit (${used.length}); hover a vertex for its qubit`;
  } else if (mode === 'phi' && d.colour) {
    const hue = d.colours ? d.colour.map((_, i) => d.colours.reduce((s, c) => s + c[i], 0) / d.colours.length) : d.colour;
    for (let i = 0; i < n; i++) set(i, hsl(hue[i], 0.75, 0.55));
    S.colourLabel = d.colours ? `colour decoded from φ (mean of ${d.colours.length} readouts)` : 'original colour field';
    legend('faceLegend', [0, 0.2, 0.4, 0.6, 0.8, 1].map(hueCss), '0', '1', 'colour value stored in φ');
  } else {
    for (let i = 0; i < n; i++) set(i, PLAIN);
    S.colourLabel = mode === 'phi' ? 'plain (φ colour needs pair mode)' : 'plain';
    $('faceLegend').hidden = true;
  }
  // Faces: one flat colour each, the mean of their vertices' colours in sRGB (as in the desktop app).
  const fc = geom.attributes.color.array, tmp = new THREE.Color(), s = { r: 0, g: 0, b: 0 };
  let corner = 0;
  for (const f of d.faces) {
    let r = 0, g = 0, b = 0;
    for (const i of f) {
      tmp.setRGB(col[3 * i], col[3 * i + 1], col[3 * i + 2]).getRGB(s, THREE.SRGBColorSpace);
      r += s.r; g += s.g; b += s.b;
    }
    tmp.setRGB(r / f.length, g / f.length, b / f.length, THREE.SRGBColorSpace);
    for (let k = 0; k < 3 * (f.length - 2); k++, corner++) {
      fc[3 * corner] = tmp.r; fc[3 * corner + 1] = tmp.g; fc[3 * corner + 2] = tmp.b;
    }
  }
  geom.attributes.color.needsUpdate = true;
  dotGeom.attributes.color.needsUpdate = true;
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
  if (!S.seq.length) return;
  const v = S.seq[S.step];
  copyEnds(geom.attributes.position.array, S.corner, v);
  geom.attributes.position.needsUpdate = true;
  geom.computeBoundingSphere();
  dotGeom.attributes.position.array.set(v);
  dotGeom.attributes.position.needsUpdate = true;
  dotGeom.computeBoundingSphere();
  dots.visible = $('dots').checked;
  copyEnds(wireGeom.attributes.position.array, S.wire, v);
  wireGeom.attributes.position.needsUpdate = true;
  if (S.edges) {
    copyEnds(edgeGeom.attributes.position.array, S.edges.ends, v);
    edgeGeom.attributes.position.needsUpdate = true;
  }
  edgeLines.visible = !!S.edges;
  mesh.material = $('shade').checked ? lit : flat;
  mesh.material.opacity = S.edges ? 0.35 : 1;
  marker.visible = false;
  $('tip').hidden = true;
  placeMates();
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
    s.graph + (s.quality != null ? ` · data-qubit quality ${s.quality.toFixed(2)}` : ''),
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
    S.focus = null;
    rebuild();
    drawMap();
    S.playing = d.frames.length > 1;
    if (!S.playing) { S.step = S.seq.length - 1; draw(); }
    const cut = d.frames.length < d.requested ? ` (stopped at the time limit; asked for ${d.requested})` : '';
    status(`${d.frames.length} readout(s) in ${d.seconds.toFixed(1)} s on ${p.backend}${cut}`);
  });
}

function setEdges(r) {
  S.edges = r;
  $('edges').textContent = r ? 'Hide edges' : 'Read edges';
  $('edgeLegend').hidden = !r;
  if (!r) return;
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

// ---------- qubit map: which data and address qubits hold each vertex
// map.vertices[i][c] = [group, address]: value c of vertex i sits in data qubit
// groups[group].data, in the slot where address qubit groups[group].addresses[j]
// reads bit j of address (Tessa's decoder bit order).
const HOT = 'rgb(230, 150, 0)';
const mapOn = () => $('qmapOn').checked && !!S.data;
const comps = () => (S.data.map.vertices[0].length === 1 ? ['r'] : ['x', 'y', 'z']);

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
  const ctx = c.getContext('2d'), css = getComputedStyle(document.documentElement);
  const [muted, accent, line, panel] = ['--muted', '--accent', '--line', '--panel'].map((v) => css.getPropertyValue(v).trim());
  ctx.scale(dpr, dpr);
  const f = S.focus, hotGroups = new Set(f ? f.groups : []);
  const hotAddr = new Map(); // address qubit -> bit, when one value is hovered
  for (const g of hotGroups) for (const q of m.groups[g].addresses) hotAddr.set(q, null);
  if (f && f.vertex !== null && f.groups.length === 1 && S.data.map.vertices[f.vertex].length === 1) {
    const [g, a] = S.data.map.vertices[f.vertex][0];
    m.groups[g].addresses.forEach((q, j) => hotAddr.set(q, (a >> j) & 1));
  }
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
  const f = S.focus, m = S.data.map, pair = S.data.colour ? ' (θ: shape, φ: colour)' : '';
  if (!f) return 'Hover a vertex on the mesh, or a data qubit here.';
  const qs = (g) => m.groups[g].addresses.map((q) => `q${q}`).join(' ');
  if (f.vertex === null) {
    const g = f.groups[0], held = m.vertices.flat().filter((v) => v[0] === g).length;
    return `data q${m.groups[g].data} · address ${qs(g)}\n${2 ** m.groups[g].addresses.length} slots, ${held} hold values of this mesh${pair}\n${f.mates.length} vertices on it (orange on the mesh)`;
  }
  const lines = m.vertices[f.vertex].map(([g, a], c) => {
    const bits = m.groups[g].addresses.map((q, j) => `q${q}=${(a >> j) & 1}`).join(' ');
    return `${comps()[c]} → data q${m.groups[g].data}, slot ${a} of ${2 ** m.groups[g].addresses.length}\n    address ${bits}`;
  });
  return `vertex ${f.vertex}${pair}\n${lines.join('\n')}\n${f.mates.length} vertices share ${f.groups.length > 1 ? 'these data qubits' : 'this data qubit'} (orange)`;
}

function setFocus(vertex, group = null) {
  if (!mapOn() || (vertex === null && group === null)) S.focus = null;
  else {
    const groups = vertex !== null ? [...new Set(S.data.map.vertices[vertex].map((v) => v[0]))] : [group];
    if (S.focus && S.focus.vertex === vertex && S.focus.groups.join() === groups.join()) return;
    const mates = [];
    S.data.map.vertices.forEach((vs, i) => { if (vs.some((v) => groups.includes(v[0]))) mates.push(i); });
    S.focus = { vertex, groups, mates };
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
  if (k >= 0) setFocus(null, k);
});
$('qcanvas').addEventListener('pointerleave', () => setFocus(null));

// ---------- hover: coordinates while paused; the qubit map (if on) also while playing
const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
function hideTip() { marker.visible = false; $('tip').hidden = true; }
function clearHover() { hideTip(); if (S.focus) setFocus(null); }
renderer.domElement.addEventListener('pointermove', (e) => {
  if (!S.data || (S.playing && !mapOn())) return clearHover();
  const r = renderer.domElement.getBoundingClientRect();
  ptr.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ptr, camera);
  const hit = ray.intersectObject(mesh)[0];
  if (!hit) return clearHover();
  const v = S.seq[S.step], o = S.orig;
  const at = (i, a) => new THREE.Vector3(a[3 * i], a[3 * i + 1], a[3 * i + 2]);
  const i = [hit.face.a, hit.face.b, hit.face.c].map((c) => S.corner[c]).sort((p, q) => at(p, v).distanceTo(hit.point) - at(q, v).distanceTo(hit.point))[0];
  setFocus(i);
  if (S.playing) return hideTip();
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
renderer.domElement.addEventListener('pointerleave', clearHover);

// ---------- wiring
document.querySelectorAll('input[type=range]').forEach((el) => {
  const out = el.nextElementSibling;
  const show = () => { if (out) out.textContent = el.id === 'shots' ? 2 ** +el.value : el.id === 'quant_bits' && +el.value === 0 ? 'off' : el.value; };
  el.addEventListener('input', show);
  show();
});
for (const id of ['shape', 'resolution', 'payload', 'mode', 'layout', 'clamp', 'clamp_b', 'compression', 'quant_bits', 'headroom', 'backend', 'placement', 'shots', 'noise_strength', 'drr']) {
  $(id).addEventListener('input', draw);
}
for (const id of ['view', 'colour', 'gain', 'mask', 'between', 'loop', 'orig', 'seed']) $(id).addEventListener('input', rebuild);
$('shade').addEventListener('input', draw);
$('dots').addEventListener('input', draw);
$('qmapOn').addEventListener('input', () => setFocus(null));
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
window.tessaRipple = { state: S, geometry: geom }; // read-only handle for debugging in the console
resize();
requestAnimationFrame(frame);
status('Loading Tessa on the server (a cold start can take ~10 s)…');
run(1);
