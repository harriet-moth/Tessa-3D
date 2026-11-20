# Tessa Ripple — Morph front: prototype brief

Status: **prototype done, stopped before the UI** (both stop conditions below fired).
Desktop app only.

## Results (2026-11-20, `dev_front.py` → `out/front/front.csv`, `chips.csv`, `strip_apple_radial.png`)

1. **Is the patch-stepped front visible, and different on Phoenix and Fez?** Partly.
   The front itself is real: at t = 0.5 it is 9–31 % of the morph's displacement away
   from the uniform sweep (apple radial 0.09, xyz 0.31; Suzanne → sphere 0.09). The
   *patch steps* are small: the quantisation gap (mean |front − smooth per-vertex
   reference| / mean morph displacement) is 0.3–8 % (apple radial on Phoenix 0.3–1.5 %,
   Suzanne 1.6–8 %). Apple frames hold only 2–4 distinct α (23 cells on Phoenix, 36 on
   Fez; the bite sits on 1–3 of them), Suzanne 6–25 radial and up to 84 xyz on Fez.
   Phoenix and Fez differ: 2–10 % of apple vertices (radial; up to 24 % xyz at t = 0.75)
   and 7–39 % of Suzanne vertices get an α more than 0.1 rad apart.
   **Quantisation gap is small**, so the front looks like a classical per-vertex wave.
2. **Is the edge readable above noise?** Not on the chips. At 65,536 shots the edge
   visibility (mean displacement of vertices with 0 < α < π/2, over the frame's RMS noise)
   is at most 1.6 for apple radial, 1.5 for Suzanne radial and 0.25 for xyz, all below 3.
   With w ≤ 0.15 on Phoenix the apple has *no* vertex mid-ramp at t = 0.5: the front
   jumps a whole patch at a time. Tessa's DRR (`mean_std`) halves the apple's chip noise
   (0.036 → 0.020) and lifts apple radial to 2.97 (Phoenix, w = 0.3), but it hurts
   Suzanne → sphere, whose statistics are far from A's. The reached region (vertices the
   front has turned and B moves) reads at 3–5 radial on chips (6–9 with DRR) and up to 22
   on the noiseless lattice. Chip noise barely falls with shots (apple radial Phoenix:
   0.045 at 4,096, 0.036 at 65,536): calibrated gate and readout error dominates, not
   shot noise. The noiseless lattice reads the edge at 11 (apple radial, t = 0.25).
3. **Do the endpoints stay exact?** Yes. Noiseless, t = 0 is A and t = 1 is B to within
   7e-16 for every mesh, payload and graph, and the sampled endpoints pass the existing
   tolerances (`test_front_endpoints_noiseless`).

Cost is identical to the scalar sweep: the same two-qubit gates and depth on every
graph (one RY per data qubit either way), and the same seconds per frame.

Also found while auditing:
- **Unreached cells were noisier than A.** The transition decode subtracts two
  independent Z readouts and divides by sin θ where α = 0. Any value read at α = 0 (a front's
  unreached part, or a whole frame at t = 0) now decodes from its own Z readout alone
  (`transition_values`), so it is A at A's noise. (2026-11-20 audit: this first used the
  shared Z reference, which froze the unreached part across frames and left t = 0
  frames after the first ~11 % noisier than A; the reference also shared the frame's
  seed, which under-reported noise at small t. It now has its own, `reference_seed`.)
- **xyz morphs do not read on the chips at all**, front or not: frame noise 0.25–0.36
  against a mean displacement of about 0.2–0.3 (front vs uniform SNR < 1).
- The plan's edge metric is empty whenever no qubit is mid-ramp; `reached_visibility`
  is recorded beside it.

Per *Experiments*, the UI was not built. The bridge functions, `Params` fields and tests
are in place, so a UI can be added later if the decision changes.
 Builds on Morph (see `docs/morph-plan.md` and the
README's Morph section); read both first.

## Goal

Today every data qubit is turned by the same sweep angle α = t·π/2, so the whole mesh
morphs at once. Noiseless, each frame is w·cos θ + (1 − w)·cos φ: a cross-fade in cosine
space that a classical shape key could trace. The only quantum signature is the noise.

The **morph front** gives each data qubit its own α_q(t), driven by surface distance from
a source point. B spreads across A as a wave. Mesh-aware layout makes one data qubit one
surface patch, so the front advances in **patch-sized steps whose shape is Tessa's layout
on that chip**: something a shape key cannot produce. It costs no extra gates (still one
RY per data qubit) and is the groundwork for the collision morph (open decision 3 in
`morph-plan.md`, "Collision morph" in the README's *Not done yet*).

The prototype has to answer three questions:
1. Is the patch-stepped front clearly visible, and does it look different on Phoenix and Fez?
2. Is the front's edge readable above the noise at 4,096 and 65,536 shots?
3. Do the endpoints stay exact (A at t = 0, B at t = 1) when α varies per qubit?

## What already exists (do not rebuild)

- `bridge.sweep_circuit(e, alpha, part)`: accepts a scalar or one angle per data qubit
  (`graph.groups` order) and applies `RY(-a)` per data qubit. Address qubits are never turned.
- `bridge._per_value(e, alpha)`: maps per-qubit α to per-value α via `e.groups_of`.
- `bridge.transition_values`, `bridge.sweep_values`, `bridge.tilted_angles`: all take a
  scalar or per-qubit α. `_transition_plan` also accepts **per-value** α arrays, which
  `ideal_frames` handles: use this for the smooth reference below.
- `mesh.edges`, `mesh.edge_lengths`, `mesh.adjacency`: graph for geodesic distance.
- `bridge.vertex_qubits(e)`, `bridge.qubit_map(e)`, the **Qubit** colour mode: show which
  data qubit holds each vertex.
- `e.batch`: each vertex's batch when the mesh needs several runs.

### Known gap: batches

`tilted_angles` passes the same `alpha` to every part:
`[sweep_circuit(e, alpha, k) for k in range(e.runs)]`, and `_per_value` indexes
`alpha[e.groups_of]` with no batch. When a mesh is read in K batches, each batch puts
**different vertices** on the same data qubit, so one α per data qubit is wrong. Extend α
to shape `(K, G)` (batch, data qubit), as `morph-plan.md` intended ("addressed by batch
and qubit"):
- `sweep_circuit(e, alpha, part)` takes `alpha[part]` when alpha is 2-D.
- `_per_value` uses `alpha[e.batch[:, None], e.groups_of]` (check that `groups_of` is
  per batch; adjust if it is global).
- Scalars and 1-D arrays keep working exactly as now (K = 1).

## The front

### Distance
1. Per-vertex geodesic distance from the source vertex: Dijkstra over the mesh edges
   weighted by `edge_lengths` (`scipy.sparse.csgraph.dijkstra`). Normalise to [0, 1] by
   the largest finite distance. Unreachable vertices (disconnected parts) get 1.
2. Per-qubit distance d_q: the **median** of the distances of every vertex with a value in
   that (batch, data qubit). In xyz a vertex's three components may sit on different
   qubits; count the vertex once per qubit it touches. A data qubit with no values gets 1.

### Angle
With width w > 0 (the front's softness, as a share of the normalised distance):

    α_q(t) = (π/2) · clip((t·(1 + w) − d_q) / w, 0, 1)

This gives α = 0 everywhere at t = 0 and α = π/2 everywhere at t = 1, so the endpoints are
the existing exact A and B. Default w = 0.15.

### Source
- Default: **auto**, the A vertex with the largest |B − A| (the `morph Δ` colour's
  maximum). For apple → bitten apple this lands in the bite.
- Optional: a vertex index (slider or entry). Picking by clicking in matplotlib 3D is
  unreliable, so leave it out of the prototype.

### Smooth reference (for comparison only)
The same formula with per-**vertex** distance instead of per-qubit, passed as a
per-value α array to `ideal_frames` through `_transition_plan`. No circuit can produce
this (it would need a different angle per slot), so it shows what the chip's
quantisation adds. Draw it as the noiseless reference ghost when front mode is on.

## Code

New functions (in `morph.py` or `bridge.py`, wherever fits best; keep the existing
Google-style docstrings):
- `geodesic(verts, faces, source) -> (N,)` normalised distances.
- `qubit_distance(e, d_vertex) -> (K, G)` per-(batch, data qubit) medians.
- `front_alpha(d, t, width) -> same shape as d`.
- `front_source(e) -> int` (auto source).

`Params` gains `morph_front: bool = False`, `front_width: float = 0.15`,
`front_source: int | None = None` (None = auto). Include them in `readout_key`,
not `encoder_key`: the front changes only the readout.

`decode_raw`, `morph_expected` and `morph_frame` use `front_alpha` instead of
`t·π/2` when `morph_front` is on. Everything else (Z readout shared across Run frames,
DRR, radial directions with slerp or A's directions only) is unchanged.

**Do not change any default.** With `morph_front` off, every output must be bit-for-bit
what it is today.

## UI (app.py, Morph options box)

- **Front** checkbox (off by default).
- **Front width** slider, 0.05–0.5.
- **Source**: auto, or a vertex index.
- When Front is on, the status line gives the number of distinct steps the front makes
  (distinct d_q values) and the source vertex.
- Suggest, but don't force, **Colour** *Qubit* while the front runs, so the steps can be
  read against the layout.

## Prototype script first

Before the UI, add `dev_front.py` (or a function in `dev.py`, matching how `dev.py` is
used) that runs the experiments below headlessly and writes PNG frames plus a CSV of the
metrics into `out/front/`. The UI comes after the numbers look worth it.

## Experiments

Meshes: `apple_full.obj` → `apple_bitten.obj` (shared topology if counts match, else
closest point), and Suzanne → a UV sphere written with `mesh.write_obj`. Payloads: radial
(A's directions only) and xyz. Backends: noiseless, phoenix, fez. Shots 4,096 and
65,536. t ∈ {0, 0.25, 0.5, 0.75, 1}, w ∈ {0.1, 0.15, 0.3}.

Record per run:
- **Steps**: distinct α values in the frame, and the mean patch size (vertices per data qubit).
- **Quantisation gap**: mean |front − smooth reference| per vertex at each t, noiseless.
  This is the visible signature; if it is tiny, the front looks like a shape key.
- **Edge visibility**: at t = 0.5, the mean displacement of vertices inside the front's
  edge (0 < α < π/2) divided by the frame's noise (RMS of noisy − noiseless frame).
  Above about 3 counts as readable.
- **Chip difference**: for the same mesh and t, the share of vertices whose α differs by
  more than 0.1 rad between Phoenix and Fez.
- **Cost**: two-qubit gates and depth (should match the scalar sweep), seconds per frame.

Save image strips for apple radial on Phoenix, Fez and the smooth reference side by side
at t = 0.25, 0.5 and 0.75.

## Tests (tests/test_morph.py)

1. `front_alpha` is 0 everywhere at t = 0, π/2 everywhere at t = 1, and non-decreasing in t.
2. Noiseless with the front: t = 0 equals A and t = 1 equals B (same tolerance as the
   existing endpoint tests), radial and xyz.
3. With `morph_front` off, `decode_raw` / `morph_expected` are unchanged (compare against
   the scalar path for several t).
4. A constant d (every qubit equal) reproduces the scalar sweep at the matching α.
5. Vertices whose values all sit on one data qubit get the same α.
6. Batched mesh (high Suzanne, xyz, Fez, K = 4): `alpha` of shape (K, G) is accepted,
   each part's circuit uses its own row, and a wrong shape raises `ValueError`.
7. `geodesic` is 0 at the source, 1 at the farthest vertex, and finite on a connected mesh.
8. `readout_key` changes with `morph_front`, `front_width` and `front_source`;
   `encoder_key` does not.

## Done when

- The tests pass, and so do the existing ones.
- The experiment CSV and image strips are in `out/front/`, with a short summary of the
  three questions from *Goal* at the top of this file (answer them with the numbers).
- The README's Morph section gets a **Front** paragraph, and "Collision morph" in
  *Not done yet* says what is now in place.

Stop and report after the experiments if the quantisation gap is small or the edge does
not read above noise at 65,536 shots. Don't build the UI in that case.
