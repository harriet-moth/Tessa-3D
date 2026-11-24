# Tessa Ripple — Direction morph and spectral payload: plan

Status: **Part B (spectral payload) done: gate passed, in both apps** (2026-11-23; results
below Part B's heading). **Part A not started.** Read `docs/morph-plan.md`,
`docs/interference-morph.md`, `docs/morph-front-plan.md` and the README's Morph section
first.

Two new payloads that keep every part of the shape in the qubits (no classical
directions, unlike radial) and aim to cut the noise that makes xyz morphs unreadable on
the chips (front prototype: xyz frame noise 0.25–0.36 against a mean displacement of
0.2–0.3, SNR < 1).

- **Part A, direction morph.** Each vertex's direction lives as a qubit's Bloch direction
  (θ, φ), A's and B's in separate slots under an interference selector, plus a radius slot.
  Decoded from ratios of ⟨X⟩, ⟨Y⟩, ⟨Z⟩, so the dominant noise (shrinking the Bloch vector)
  cancels. The morph is the interference readout, so the direction path from A to B is made
  by the quantum state, not by `morph_dirs`.
- **Part B, spectral payload.** xyz encoded as coefficients of A's mesh harmonics (Laplacian
  eigenvectors). Fewer values, a smaller circuit, and noise that reads as smooth swells
  instead of per-vertex grain. Works with the existing sweep and interference morphs.

Both are experiment-first: build headless, measure, and only add UI when the stop
conditions below pass.

---

## Verified facts this plan relies on (2026-11-23)

- **Tessa's `pair_full` decode is ratio-based** (`tessa/core/decode.py`):
  `theta = atan2(hypot(<X>, <Y>), <Z>)`, `phi = atan2(<Y>, <X>) mod 2π`. Single mode uses
  `arccos(<Z>)`; `pair` (half-circle) uses `arccos(<Z>)` and `arccos(<X>/sin θ)`.
- **Cell state** (`tessa/core/mode.py`): `cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩`. Pair modes need
  `clamps[0] > 0`: θ lives in `[c, π − c]`. On the full circle `clamps[1] = 0` identifies
  φ = 0 and 2π (right for azimuth, which is periodic).
- **Basis change:** `tessa.codec._measured(circuit, data_qubits, basis)` applies H (X) or
  S†H (Y) to data qubits only.
- **Noise model** (`tessa/noise.py`): depolarising + thermal relaxation (T1, T2) on `sx`/`x`
  and `cz`, plus **symmetric** bit-flip readout error per qubit.
- The current spherical payload is θ-mode only (`bridge.encode_mesh` raises for other
  modes). It already uses `pair_full` via `_spherical_circuit`, with scales (π, 2π,
  padded max radius).
- The interference morph (`interfere.py`) encodes in single mode and reads Z only;
  `pure()` shows that with real states the coherent frame follows the great circle between
  A's and B's Bloch vectors.

## Why ratio decoding should cut noise

What each noise channel does to a Bloch vector (X, Y, Z), and what survives a ratio decode:

| Channel | Effect | φ = atan2(Y, X) | θ = atan2(hypot(X, Y), Z) | θ = arccos(Z) (single mode) |
|---|---|---|---|---|
| Depolarising | all × (1 − p) | exact | exact | pulled to π/2 |
| Symmetric readout flip | all × (1 − 2r) | exact | exact | pulled to π/2 |
| Dephasing (T2) | X, Y shrink | exact | pulled to nearest pole | exact |
| Amplitude damping (T1) | X, Y × √(1−γ); Z → (1−γ)Z + γ | exact | pulled to 0 | pulled to 0 |

So under Tessa's noise model the **azimuth is protected from every channel**, and a
ratio-decoded θ is protected from depolarising and readout error (the "pull towards the
middle" the interference doc measured). Caveats to measure, not assume:
- Errors happen mid-circuit, before later multiplexed rotations, so they are transformed
  rather than only shrinking the final vector. Expect partial protection.
- Real chips have asymmetric readout error (not in the model), which offsets ⟨X⟩ and ⟨Y⟩
  and biases φ towards 45°.
- Why not store radius as the Bloch vector's length? Length is exactly what noise destroys.

---

## Part A — Direction morph

### Encoding

Builds on `interfere.py`. Half-graph slots per vertex: 2 (**dir**, **radius**). Each half slot
becomes an A slot (selector bit 0) and a B slot (bit 1) via `pair_slots`.

| Full-graph slot | θ field | φ field |
|---|---|---|
| A dir | A's polar / π | A's azimuth / 2π |
| A radius | A's radius / R | 0 |
| B dir | B's polar / π | B's azimuth / 2π |
| B radius | B's radius / R | 0 |

- Directions and radii from `radial(verts, centre)` and `radial(target, centre)`, centre =
  A's centroid (as every morph). Polar from +z, azimuth via `arctan2`, as `to_payload`.
- R = max radius of A and B × (1 + headroom), shared so equal radii get equal angles.
- One `tessa.encode(graph, a_theta, b_phi, mode="pair_full", clamp=morph_clamp(p),
  clamp_b=0.0, value_range=(0, 1), value_range_b=(0, 1))` per run, as `_spherical_circuit`.
  Padding as `encode_interfere` (empty pairs hold A's padding twice; data qubits outside
  every star take the mean).
- Capacity: 2 half slots per vertex, against 3 for xyz interference and 1 for radial.
  Estimate: Fez half capacity 224 → 112 vertices per run (apple, 247 vertices: 3 runs;
  xyz: 4).

### Readout

- Per star cluster: `_cluster_circuit` as now (selector RY(−β), β = tπ), then the basis
  change on the cluster's **data qubits only** (Z, X, Y: three circuits per cluster per
  frame, against one for the Z-only interference readout).
- Post-select selector = 0 in each basis's counts separately, then
  `decode_cells({"z", "x", "y"}, local, n, "pair_full", clamps)`.
- Frame: dir slots → (polar, azimuth), radius slots → r, then
  `centre + r·(sin p cos a, sin p sin a, cos p)` (generalise `from_payload`'s spherical
  branch).
- Dephasing control (`dephase=True`) works unchanged; run it in all three bases.

### References (all noiseless)

- `ideal_dir(ie, t)`: exact cluster statevectors, expectations in Z, X, Y, selector-0
  projection.
- `pure_dir(ie, t)`: each data qubit alone in its star:
  `cos(β/2)|ψ_A⟩ + sin(β/2)|ψ_B⟩` normalised, using Tessa's exact phase convention (check it
  against a one-slot `tessa.encode` statevector, including any e^{−iφ/2} global phase). With
  φ_A ≠ φ_B the states are complex, so the path is **not** guaranteed to be the great
  circle. This is the main open question (Q1).
- `slerp_ref(t)`: classical slerp of A's and B's directions (`morph.slerp`) and lerp of
  radii. The target look, and the comparison for Q1.
- `incoherent_dir(t)`: mixture, Bloch vector = cos²(β/2)·v_A + sin²(β/2)·v_B. Its direction is
  a weighted chord average (nlerp-like). Use with `coherence()` as now.

### Experiments (`dev_dirmorph.py` → `out/dirmorph/`)

**Phase 0 (no new encoding; gate for everything else).** Use the existing spherical payload
in θ mode against radial and xyz, single readout, no morph. Meshes: apple, Suzanne.
Backends: noiseless 10×12, flat noise, Phoenix and Fez noise models. Shots 4,096 and
65,536, 3 seeds.
- Per-vertex error split into radial and tangential parts (relative to the centroid
  direction). Compare spherical's tangential error with xyz's.
- Polar and azimuth error separately, with their mean bias. Prediction: azimuth unbiased,
  polar biased towards 0.
- Ratio vs arccos: decode the same counts with θ = arccos(⟨Z⟩) as well, to measure how much
  the ratio decode removes.
- Whether Tessa's DRR (`mean_std`) removes the polar bias.

**Phase 1 (direction morph, after Phase 0 passes).** Apple → bitten apple, Suzanne →
sphere (shared topology where it applies).
- Endpoints: t = 0 is A and t = 1 is B to ≤ 1e-12 noiseless.
- Q1: max and mean angular distance of `pure_dir` and `ideal_dir` from `slerp_ref` at
  t ∈ {0.25, 0.5, 0.75}. Also how much the radius path differs from a lerp.
- Chips: frame noise, SNR (mean displacement over RMS frame noise, as the front doc),
  coherent − control as in the interference doc, and the share of shots kept.
- Same metrics for xyz and radial interference at equal shots.
- Cost: runs, two-qubit gates, depth, circuits per frame, seconds per frame.

### Stop conditions

- **After Phase 0:** stop Part A if spherical's tangential error on Phoenix and Fez is not
  ≤ 0.7× xyz's at the same shots, or the azimuth bias is not clearly smaller than the polar
  bias. Write the numbers into this doc either way.
- **After Phase 1:** build UI only if the chip SNR is above 1 (xyz is below) and ideally at
  radial's level, and the Q1 path is smooth (no doubling back). If the path is smooth but far
  from slerp, report it and decide with the user: it is still a quantum interpolation, just
  not a slerp.

### Code changes

- `interfere.py`:
  - `_payload`: accept `"spherical"`; return dir and radius values for A and B, and scales.
  - `encode_interfere`: branch for spherical (2 half slots per vertex, `pair_full` encode,
    fields as the table above). Keep the radial and xyz paths unchanged.
  - `read`: bases from the encoding (`("z",)` or `("z", "x", "y")`); basis change on data
    qubits; post-select per basis; `_decode_star` gets the mode and the counts per basis.
  - `frame`: spherical branch (no `morph_dirs`).
  - New `ideal_dir`, `pure_dir`, `slerp_ref`, `incoherent_dir` (or extend the existing ones
    by payload); `cost` counts circuits per frame.
- `bridge.py`: error message for spherical + sweep morph names the interference readout as
  the way to morph with this payload. Factor the spherical position maths out of
  `from_payload` so both paths share it.
- `app.py` (only after the gate): allow Payload *spherical* in morph when Interference
  readout is on; disable the front controls; time estimate × 3 for the extra bases.
- README: Morph section, Payload notes, findings.

### Tests (`tests/test_interfere.py`, `tests/test_bridge.py`)

1. Spherical interference endpoints exact noiseless (apple radial-size mesh, small lattice).
2. `pure_dir` equals `ideal_dir` on a lattice where each star has one data qubit.
3. With φ_A = φ_B, `pure_dir` equals the great circle (reduces to today's `pure`).
4. Ratio decode removes a pure depolarising error: decoded directions equal noiseless ones
   (Aer depolarising-only model, many shots, tolerance).
5. Post-selection keeps the same shot share in all three bases (noiseless).
6. Dephasing control on the noiseless simulator gives `incoherent_dir`.
7. Capacity and run count: 2 half slots per vertex; the radial and xyz paths unchanged
   (existing tests pass).

---

## Part B — Spectral payload

### Results (2026-11-23, `dev_spectral.py` → `out/spectral/`: `truncation.csv`, `theta.csv`, `morph.csv`, `costs.csv`, `summary.json`, `strip.png`)

Built as planned (`tessa_ripple/spectral.py`, `Params(payload="spectral", spectral_k,
spectral_tol, spectral_band)`, sweep and interference morphs, tests in
`tests/test_spectral.py`). One change to the plan: **a morph keeps the constant mode.** The
centre is A's mass-weighted centroid, so A has no constant part, but B's matched points do:
B's centroid shift against A is 0.6 % of the bounding-box diagonal for the apple and **3.2 %
for Suzanne → sphere**, more than the whole 1 % budget. Dropping it would have counted that
shift as truncation, so a morph encodes the constant as row 0 (A's coefficient there is
exactly 0); θ and split readouts still drop it.

**Gate: passed** for both meshes on both chips (`summary.json` → `gate`), with or without
DRR. Numbers are means over 3 seeds at 65,536 shots; error is the RMS per-vertex distance
over the bounding-box diagonal, against the original mesh (truncation included).

| θ readout, Phoenix / Fez | apple error | apple roughness | Suzanne error | Suzanne roughness |
|---|---|---|---|---|
| xyz | 0.096 / 0.100 | 2.0 / 2.9 | 0.097 / 0.092 | 2.3 / 2.5 |
| radial | 0.010 / 0.014 | 1.6 / 1.6 | 0.020 / 0.021 | 1.2 / 2.2 |
| spectral, auto (k 56 / 119) | 0.046 / 0.068 | 0.26 / 0.21 | 0.047 / 0.053 | 0.32 / 0.39 |
| spectral, every mode one run holds | 0.061 / 0.071 | 0.30 / 0.26 | 0.053 / 0.053 | 0.39 / 0.40 |
| spectral auto, one shared range | 0.044 / 0.055 | 1.0 / 0.64 | 0.048 / 0.049 | 1.3 / 1.05 |

Roughness is |L_u E| / |E − mean| with L_u the uniform graph Laplacian: about the vertex
degree for independent per-vertex noise, near 0 for smooth swells.

1. **Error.** Spectral is about half of xyz's on both chips (and below it at every k
   tried). With Tessa's DRR (`mean_std`) it falls to 0.018–0.036 against xyz's 0.08–0.09.
   Radial stays the quietest payload, but its directions are classical. Noiseless (shot
   noise only), spectral's noise is 0.0037 against xyz's 0.0056.
2. **Roughness.** 6–14× lower than xyz's on the chips (the gate asked for 2×). The noise
   reads as slow swells of the whole surface, not grain (`strip.png`). Under gate noise
   most of every payload's error is a smooth shrink towards the middle of the range; after
   DRR removes it, xyz's roughness rises to 3–4 while spectral's stays at 0.2–0.4.
3. **Band scaling is what makes it work.** With one shared range the error is similar but
   the roughness is 3–4× higher, and the high modes' noise shows (high-mode energy share
   0.47 against 0.0003 noiseless, apple).
4. **Truncation and the bite.** The 1 % tolerance needs k = 56 for the apple alone, 84 for
   apple → bitten apple, 119 for Suzanne, 197 for Suzanne → sphere. The bite is the hard
   part: at k = 56–84 it keeps only about 70 % of its depth (bite-region truncation error 28 %
   of its displacement, gate < 25 %). It passes at k ≥ about 100: every apple mode fits one
   Phoenix run (k = 246, bite error 0) and Fez's sweep holds k = 148 (bite error 11 %).
   **For the bite, raise k to the cap** (the apps' k slider past the cap encodes the cap).
   The interference morph holds half as many (Fez: k = 73, bite error 34 %), so its bite is
   soft on Fez. Suzanne → sphere passes from k = 64.
5. **Cost.** Always one run. Apple: 196 two-qubit gates at auto k against xyz's 792 on
   Phoenix (Fez: 228 against 448 a run, in 2 runs); Suzanne: one run against xyz's 2
   (Phoenix) or 4 (Fez). Seconds per θ readout on the chip models: 15–64 % of xyz's.
6. **Morph SNR on the chips (t = 0.5, mean displacement of moving vertices over RMS frame
   noise): better than xyz, still below 1 without DRR.**

   | SNR, Phoenix / Fez | apple sweep | apple interference | Suzanne sweep | Suzanne interference |
   |---|---|---|---|---|
   | xyz | 0.26 / 0.26 | 0.32 / 0.28 | 0.36 / 0.42 | 0.51 / 0.52 |
   | radial | 2.2 / 1.9 | 2.6 / 1.1 | 2.5 / 2.8 | 3.7 / 2.4 |
   | spectral auto | 0.38 / 0.41 | 0.65 / 0.59 | 0.52 / 0.61 | 1.06 / 1.21 |

   Noiseless: spectral 6.0 (apple) and 9.4 (Suzanne) against xyz 3.4 and 4.7. With DRR the
   apple's spectral sweep reaches 2.0 on Phoenix (xyz 0.34). Endpoints are exact
   (t = 0 and t = 1 within 2e-15 of the truncated A and B) for every payload and readout.
   So on the chips a spectral morph reads clearly better than xyz and beats it 1.5–2.4×,
   but the motion itself is still under the noise; the gate (θ error, roughness, bite)
   did not ask for morph SNR above 1, so the apps got the payload anyway.

UI built per the plan (both apps): Payload / load *spectral*, a k slider (0 = auto, a k past
one run's cap encodes the cap), error measured against the truncated mesh, the front
disabled for spectral, the qubit colour replaced by a note, and the qubit map showing each
data qubit's modes instead of vertices.

### Encoding

New module `tessa_ripple/spectral.py`.

- **Basis.** Cotangent Laplacian L and lumped (barycentric) mass M of A, triangulated with
  `morph.triangles` (quads in `uv_sphere`). Generalised eigenproblem L φ = λ M φ, smallest
  k + 1 via `scipy.sparse.linalg.eigsh(L, k + 1, M, sigma=-1e-8)`. Drop mode 0 (constant):
  the centroid stays classical, as in every payload. Φ is (N, k), Φᵀ M Φ = I. Cache by mesh.
- **Coefficients.** C = Φᵀ M (V − centre), shape (k, 3). B: the same with
  `correspond(...)["target"]` (B on A's vertices), so A and B share the basis.
- **Band scaling (required, not optional).** s_j = max over xyz of |C_A[j]| and |C_B[j]|,
  × (1 + headroom); encode C / s in [−1, 1]. s is classical metadata like the spherical scales
  or Tessa's value range. Reason: energy piles into the low modes, so a single shared range
  is set by the largest coefficient, about √N × the coordinate scale. With uniform
  fractional noise σ per value, per-vertex error would be about σR√k (√k times worse than
  xyz). With band scaling it is about σ × the RMS radius, roughly xyz's level, before
  counting the smaller circuit. Measure both to confirm.
- **Choosing k.** Smallest k with truncation error (RMS of V − (centre + Φ C) over the
  bounding-box diagonal) ≤ `spectral_tol` (default 1%) for **both** A and B, capped so 3k
  fits one run. Report k and both truncation errors. Endpoints are then the truncated A and
  B, not the originals.
- **Layout.** Values ordered by mode (low first), a mode's x, y, z adjacent, naive layout
  (BFS patches have no meaning here). With quality placement the lowest modes, which move
  the most, land on the best qubits.
- **Decode.** V = centre + Φ (s ⊙ C′).

### Morph

- **Sweep morph:** works as is (scalar values, A in θ, B in φ).
- **Interference morph:** works as is (A and B coefficients paired under selectors).
- **Front:** not supported. A qubit holds modes, not a patch of surface; raise a clear error.
- No batching: one run by construction.

### Guards

Several features assume one row per vertex (`slots`, `batch`, `vertex_qubits`,
`qubit_map`, hover inspect, per-qubit error colour, `front_distances`). For spectral, either
give them a mode-based meaning (qubit map: which modes each qubit holds) or disable them
with a status-line note. List each in the README.

### Experiments (`dev_spectral.py` → `out/spectral/`)

Meshes: apple → bitten apple, Suzanne → sphere. k ∈ {16, 32, 64, auto}. Band scaling on and
off. Backends and shots as Part A.
- Truncation error of A and B, and of B's distinctive region (the bite: vertices with
  |B − A| > 10% of the max).
- Per-vertex error (over the bbox diagonal) and **roughness**: share of the error energy
  above the k-th mode (project the error field on A's full basis, or use the uniform
  Laplacian's norm against the error's norm). Prediction: much lower than xyz.
- Morph SNR on chips for sweep and interference, against xyz and radial.
- Cost: values, runs, two-qubit gates, depth, seconds per frame.

### Stop conditions

Build UI only if, on Phoenix and Fez at some k: the bite stays visible (bite-region
truncation error < 25% of its displacement), per-vertex error ≤ xyz's, and roughness at
least 2× lower. If error is lower but the bite needs k past one run's capacity, report and
stop.

### Code changes

- `spectral.py`: `cotan_laplacian(verts, faces)`, `basis(verts, faces, k)`,
  `coefficients(V, centre, basis)`, `reconstruct(C, centre, basis)`,
  `choose_k(a, b, faces, tol, k_max)`, `band_scales(Ca, Cb, headroom)`.
- `bridge.py`: Payload `"spectral"`; `Params.spectral_k: int | None = None`,
  `Params.spectral_tol: float = 0.01` (in `encoder_key` only when payload is spectral);
  `Encoded` carries the basis and scales; spectral paths in `to_payload`, `from_payload`
  and `encode_mesh` (rows are modes, n = k). Front raises for spectral.
- `interfere.py`: `_payload` accepts spectral.
- `app.py` (only after the gate): Payload *spectral*, k / tolerance field, guards above.
- README.

### Tests (`tests/test_spectral.py`, plus bridge and interfere)

1. Basis is M-orthonormal; mode 0 is constant and dropped.
2. Full basis (k = N − 1) reconstructs a mesh exactly; truncation error falls as k rises.
3. `choose_k` meets the tolerance for A and B and respects the capacity cap.
4. Noiseless θ-mode readout returns the truncated mesh to 1e-9.
5. Sweep and interference morph endpoints equal the truncated A and B noiseless.
6. Band scales: every encoded value lies in [−1, 1].
7. Front with spectral raises; vertex-based features are guarded.

---

## Order and estimate

| Milestone | Work | Days |
|---|---|---|
| M0 | Part A Phase 0: spherical vs radial vs xyz noise, ratio vs arccos, bias | 0.5 |
| M1 | `spectral.py`, spectral payload in θ mode, tests 1–4, 6 | 1 |
| M2 | Spectral with sweep and interference, guards, `dev_spectral.py`, gate | 1 |
| M3 | Direction interference encode and `pair_full` readout, references, tests | 1.5–2 |
| M4 | `dev_dirmorph.py` Phase 1, Q1, chips, gate | 1 |
| M5 | UI for whichever passed, README, findings in this doc | 0.5–1 |

About 5.5–6.5 days. M0 and M1 are independent; do M0 first, since its result decides
whether M3–M4 happen.

## Open questions

1. **Q1, the coherent direction path.** With complex states, does the selector-0 projection
   move a direction smoothly from A to B, and how far from slerp? Can a per-slot phase in B's
   φ (added classically before encoding) align the branches so the path becomes the great
   circle? Work this out on one qubit before M3.
2. Is ratio θ decoding worth using for other payloads too (any slot with φ = 0 read in Z and
   X)? It costs extra bases; Phase 0 shows how much it buys.
3. Spectral basis from A only, or from a blend of A and B? A-only is simpler and keeps A
   exact up to truncation; B's distinctive detail may need more modes.
4. Asymmetric readout on real hardware: add a per-qubit readout correction before the
   ratios, if the work moves to real chips.
