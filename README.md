# Tessa Ripple

Tessa Ripple loads the vertices of a 3D mesh into a quantum state with Moth's
[Tessa](https://github.com/moth-quantum/Tessa) encoder, reads them back, and moves each
vertex to wherever the readout puts it.

It tells two stories, one per **View**:

- **Fidelity** (default): the technical case. Vertices sit at their true decoded
  positions (gain 1) and are coloured by error. Beside it, the app shows what
  makes Tessa different: many values per qubit, no SWAPs on a real chip, few
  readout outcomes per value, a second field in the phase, and edge lengths read
  straight from the state's coherences.
- **Material**: the artistic mode. A gain slider exaggerates the readout noise so
  it reads as a ripple, clearly labelled as exaggeration.

Nothing is added to the circuit to make things move. All movement comes from
variation Tessa's readout already has: **shot noise** and **simulated hardware
noise**, including real calibration snapshots of IBM chips. The one addition is a
Hadamard on the address qubit for the edge readout, which is a change of
measurement basis, not movement.

## Install and run

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/). Tessa is not on PyPI, so it
is pinned to a git commit in `pyproject.toml`. No IBM account or `ibm` extra is
needed: the chip backends use calibration snapshots that ship with Tessa.

```bash
uv venv --python 3.13 && uv pip install -e ".[dev]"
```

```bash
.venv/bin/tessa-ripple
```

```bash
.venv/bin/python -m pytest
```

To add your own meshes, pass OBJ files on the command line; each becomes a **Shape**
option. OBJ files are read as Y-up (the OBJ and Blender convention) and exported
the same way. Blender's Suzanne is included:

```bash
.venv/bin/tessa-ripple tessa_ripple/suzanne.obj
```

Suzanne has 507 vertices, measured at 4,096 shots:
- **Radial on a 9×9 lattice:** 507 values on 81 qubits (6.3 per qubit), 0.5 % error.
- **Pair mode (shape + colour):** 1,014 values on the same 81 qubits, 0.8 % error.
- **Whole `ibm_phoenix` chip with calibrated noise:** 1.7 % error, or 2.3 % in pair mode.
- **Read edges:** all 1,005 edges in about 6 s, with 3.1 % error noiseless.

As xyz (1,521 values) she is too big for any lattice here.

**Quick start:**
1. Press **Run frames** on the default blob. It plays its true readout error,
   coloured by error.
2. Press **Why Tessa** for the resource comparison.
3. Press **Read edges** to read edge lengths from coherences.
4. Switch **Backend** to *phoenix* or *fez* for realistic chip noise.
5. Switch **View** to *material* for the exaggerated ripple.

## Web version

Live at **https://tessa-ripple.vercel.app**. It's a three.js viewer in the browser; all
quantum work runs in one Vercel Python function (`api/tessa.py`) that calls the same
`tessa_ripple/bridge.py` as the desktop app. Every push to `main` redeploys it.

Differences from the desktop app:
- Changing **DRR** needs a new **Run**.
- **Run frames** is capped at 30 frames and stops after about 45 s to stay under
  Vercel's 60 s limit. On the chip backends a frame takes about 7 s there, so
  expect about 6 frames.
- There's no GIF export; **Download OBJ** saves the current step.
- Faces are unlit by default so colours are exact; tick **Shading** for lighting.

To run the web version locally:

```bash
.venv/bin/python dev.py
```

Then open http://localhost:8000.

## How a mesh goes through Tessa

1. **Mesh.** A blob, sphere, cube or OBJ (such as Suzanne) gives `N` vertices and faces.
2. **Payload** (what is encoded):
   - *radial*: one radius per vertex, measured from the centroid.
   - *xyz*: three coordinates per vertex.
3. **Prepare.** `tessa.prepare_field` optionally quantises the values (Quant bits).
4. **Pick a graph.**
   - On a simulator backend, the smallest near-square `tessa.Lattice` that holds
     the mesh, up to 10×12. That is the whole `ibm_phoenix` chip, so every lattice
     is a piece of a real device.
   - On a chip backend, that chip's whole coupling graph from Tessa's calibration
     snapshot: `ibm_phoenix` (120 qubits, capacity 792) or `ibm_fez` (156 qubits,
     capacity 448).
   - A data qubit with `k` address neighbours holds `2^k` values, so capacity is
     the sum of `2^k` over data qubits. That is why there are more values than
     qubits.
5. **Place.** **Layout** decides which capacity slot each value goes in. *mesh*
   fills each cell from a breadth-first walk over the mesh and keeps each vertex's
   values in one cell; *naive* is axis-major (see Layout below). Unused slots are
   padded with their cell's mean so cells stay smooth.
6. **Encode.** `tessa.encode` builds a measurement-free circuit of uniformly
   controlled rotations, **once** per mesh and encoder setting. In pair mode the
   colour field goes into the phase φ of the same cells.
7. **Decode.** `tessa.decode` measures the circuit on the chosen sampler. Each
   frame is one decode with its own seed.
8. **Restore and show.** `tessa.restore_field` applies optional dynamic-range
   recovery (DRR), the values become vertex positions again, and the view blends
   and animates them.

All Tessa calls are in [`tessa_ripple/bridge.py`](tessa_ripple/bridge.py). `mesh.py`,
`bridge.py` and `animate.py` are numpy-in, numpy-out with no UI state, so a Blender
add-on can reuse them.

## What the app shows about Tessa

| Tessa property | Where to see it |
|---|---|
| Capacity is the sum of `2^degree`, more values than qubits | Info panel: "86 values on 20 qubits = 4.30 values / qubit (angle encoding: 86 qubits)" |
| Runs on the chip's own couplers: no SWAPs, shallow | **Why Tessa** table, transpiled onto the Phoenix snapshot |
| One measurement setting reads every value (two in pair mode), with at most 16 outcomes per cell | **Why Tessa**: settings, outcomes per cell, shots per value |
| A second field in the phase | **Mode** *pair*: vertex colour rides in φ (**Colour** *φ colour*) |
| Smooth data compresses | **Curves**: error and depth against compression for both layouts |
| Coherences carry differences | **Read edges**: edge lengths read from the state |
| Real devices have good and bad qubits | **Backend** *phoenix* / *fez*, **Colour** *error* or *qubit* |

Example: the default blob (86 values, radial), **Why Tessa**, 4,096 shots.

| encoding | qubits | 2q gates | depth | SWAPs | outcomes per cell | shots per value |
|---|---|---|---|---|---|---|
| Tessa (4×5 lattice) | 20 | 96 | 84 | 0 | 16 | 256 |
| Angle (1 value per qubit) | 86 | 0 | 1 | 0 | 1 | 4096 |
| QCrank (parallel QPIXL) | 8 | 141 | 493 | 15 | 32 | 128 |
| FRQI | 8 | 176 | 687 | 16 | 128 | 32 |
| Amplitude | 7 | 189 | 788 | 23 | 128 | 32 |

Tessa sits between the extremes. Angle encoding needs a qubit per value (86 here,
and more than the chip has for big meshes). The compact encodings (QCrank, FRQI,
amplitude) need deep routed circuits and split each shot budget over many outcomes.
Tessa uses a few values per qubit, no SWAPs, and modest depth.

## Findings worth knowing

Measured while building this; the README states them as measured, not as claims.

- **Pair mode doubles the two-qubit gates.** Tessa's docstring says the second
  field comes "at no extra two-qubit gate cost". Tessa's own circuit builder
  (`core/encode.py`) appends a second RZ multiplexer with its own CX ladder, so
  the count goes from 84 to 172 on the default blob. The info panel shows both
  numbers. What pair mode does give for free is qubits and cells: the colour
  needs no extra qubits.
- **Layout and compression trade off.** The mesh-aware layout helps only when the
  vertex order is genuinely scrambled. On a blob with randomly shuffled vertices,
  75 % compression gives 1.0 % error with the *mesh* layout against 1.2 % with
  *naive* (radial), and 9.6 % against 12.0 % (xyz).

  Most real meshes are already ordered well enough that it doesn't pay:
  - **Built-in UV shapes:** the two layouts are identical for radial. For xyz,
    *naive* compresses better: 2.1 % against 4.4 % at 50 %, because keeping x, y
    and z in one cell puts jumps between axes inside each cell.
  - **Suzanne** (Blender's export order, radial): *naive* is slightly better,
    1.0 % against 1.3 % at 75 %.

  **Curves** shows this for whatever mesh is loaded.
- **The radial payload is the better default for star-shaped meshes.** It needs
  one value per vertex instead of three, and noise moves vertices along their
  direction from the centre. A plain sphere is a poor payload: every radius is 1,
  so Tessa's transform zeroes every coefficient (depth 1, no two-qubit gates).
  That is why the default is a smooth blob.
- **Edge readout is noise-sensitive but usable.** Mean error is 1.3 % of the mean
  edge noiseless, 6 % with flat noise, and 8–9 % on the Phoenix and Fez snapshots
  (blob 12×8, 4,096 shots).

What the app does *not* show: a quantum advantage. Apart from the edge readout,
nothing computes on the encoded state; storing a mesh on a QPU and reading it
back is strictly worse than RAM. The decode also simulates one commuting cell at a
time, which shows that this loader is classically easy to simulate. Tessa is a
loading stage, and the edge readout is the first step past loading.

## The controls

Controls fall into three tiers by how much work a change costs:

| Tier | Controls | What a change does |
|---|---|---|
| Encoder | Shape, Resolution, Payload, Mode, Layout, Clamp θ, Clamp φ, Compression, Quant bits, Headroom (and Backend, when it switches to or from a chip graph) | Rebuilds the circuit (milliseconds). Old frames are cleared. |
| Readout | Backend, Shots, Noise ×, Seed | Needs a new decode: press **Run** or **Run frames**, or tick **Auto-run**. The status line says "frames stale" until you do. |
| Post / view / animation | View, Colour, DRR, Gain, Mask %, Frames, In-between, Speed, Loop, Start from original | Applied to cached frames instantly. No quantum work (except **Frames**, which takes effect on the next **Run frames**). |

The **info panel** shows:
- the mesh, payload and layout;
- the graph (lattice size or chip);
- values on qubits, as values per qubit and against angle encoding;
- capacity used;
- circuit depth and two-qubit gates (in pair mode, also the θ field alone).

The **title** shows the view, the gain, the **mean err** (average coordinate
error over all frames as a fraction of the bounding box), the playback step and
what the colours mean.

### Mesh

**Shape** (blob / sphere / cube, plus any OBJ given on the command line)
- *In practice:* chooses the object.
  - *Blob* (default): a sphere with smooth bumps, star-shaped with varied radii.
  - *Sphere*: shows the degenerate constant payload.
  - *Cube*: only a few distinct coordinates, so it encodes what are basically
    symbols.
- *In Tessa:* changes how many values are encoded, and so the graph and depth.

**Resolution** (low / mid / high; ignored for OBJ files)

| | low | mid | high |
|---|---|---|---|
| Blob (segments × rings) | 12×8 → 86 verts | 16×10 → 146 verts | 24×12 → 266 verts |
| Sphere | 8×6 → 42 verts | 12×8 → 86 verts | 16×8 → 114 verts |
| Cube (per edge) | 3 → 26 verts | 4 → 56 verts | 5 → 98 verts |

### Encoder

**Payload** (radial / xyz)
- *In practice:*
  - *Radial*: one radius per vertex from the centroid, so noise moves each
    vertex in and out along its direction from the centre. That looks like a
    ripple and fits three times the vertices.
  - *xyz*: noise moves vertices in any direction. It suits meshes that aren't
    star-shaped.
- *In Tessa:* *radial* encodes `N` values on `value_range=(0, r_max(1 + headroom))`;
  *xyz* encodes `3N` values on one symmetric range `(-m, m)` shared by all axes, so
  the mesh isn't stretched per axis.

**Mode** (single / pair)
- *In practice:* *pair* adds a vertex colour (a hue around the vertical axis) to
  the same circuit. Show it with **Colour** *φ colour*: before a run it shows the
  original colours, and after a run the colours decoded from the phase.
- *In Tessa:* each cell is a qubit state `cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩`.
  - Shape goes in θ (read with Z) and colour in φ (read with X), at the same slots.
    The same qubits and cells hold both.
  - The two-qubit gate count doubles, because Tessa adds an RZ multiplexer (see
    Findings).

**Layout** (mesh / naive)
- *In practice:*
  - *Mesh*: each Tessa cell holds a patch of neighbouring vertices, with each
    vertex's values together. Correlated chip errors then show up as spatial
    patches rather than scatter.
  - *Naive*: axis-major in vertex-index order, the original V1 layout.
  - Use **Curves** to see which compresses better for your mesh.
- *In Tessa:* chooses which capacity slot (data qubit and address configuration)
  each value occupies. The Walsh–Hadamard compression works on the values inside
  one cell, so this choice is what decides whether compression has smooth data to
  work on.

**Clamp θ** (0–0.45 π; 0 means Tessa's default)
- *In practice:* keeps values away from the extreme ends of the angle range. A
  larger clamp leaves less of the range for the data, so noise becomes a larger
  share of each value.
- *In Tessa:* `encode(clamp=...)`. Values map to θ in `[c, π − c]`. Tessa's default
  is 0 in single mode and 0.2 π in pair mode; pair mode needs θ > 0, since a cell at
  θ = 0 or π carries no phase.

**Clamp φ** (pair only; 0–0.45 π, 0 means Tessa's default of 0)
- *In practice:* the same idea for the colour field, held in the phase.
- *In Tessa:* `encode(clamp_b=...)`, φ in `[c, π − c]`.

**Compression %** (0–100)
- *In practice:* drops gates to make a shallower circuit (watch **depth** fall),
  at the cost of accuracy. The distortion is the same every run, so it reshapes
  the mesh but doesn't animate it. How much it costs depends on the payload and
  layout; see **Curves** and Findings.
- *In Tessa:* `encode(compression=...)`. Each multiplexer's angles are
  Walsh–Hadamard transformed, and the smallest `compression` % of the coefficients
  are zeroed, together with their rotation gates.

**Quant bits** (off, 2–8)
- *In practice:* snaps every value to `2^bits` evenly spaced levels before
  encoding: a fixed, terraced reshape, not motion.
- *In Tessa:* `prepare_field(quant_bits=...)`, over the data's own min–max range.
  The quantised field is also the reference DRR matches against.

**Headroom** (0–50 %)
- *In practice:* spare room at the ends of the encoding range, so extreme vertices
  can move outward as well as inward.
- *In Tessa:* pads the top of `value_range` (and the bottom, for xyz) by this
  fraction.

### Readout

**Backend** (noiseless / flat noise / phoenix / fez)
- *In practice:*
  - *Noiseless*: shot noise only, an even shimmer.
  - *Flat noise*: every qubit equally bad (uniform rates), which gives a
    featureless shrink.
  - *Phoenix* / *fez*: real per-qubit calibration, so specific vertices on bad
    qubits misbehave. Colour by *error* or *qubit* to see the chip in the mesh.
    About 1–1.7 s per frame.
- *In Tessa:*
  - *Noiseless*: `aer_sampler`.
  - *Flat noise*: `noisy_sampler(rates=…)` with Tessa's fallback IBM-like averages,
    scaled by **Noise ×**.
  - *Phoenix* / *fez*: the mesh is encoded on the chip's whole graph from
    `tessa.hardware.devices` and decoded on a sampler built like
    `tessa.calibrated_sampler`, with each qubit's and coupler's own calibration
    (`build_local_noise_model`). The noise model is built once per chip and reused
    across seeds.

**Shots** (2^8–2^16; default 4,096)
- *In practice:* the main jitter control. Error falls as `1/√shots` (see **Curves**).
- *In Tessa:* `decode(shots=...)`, measurements per cell circuit and basis.

**Noise × (flat)** (0–3; flat noise only)
- *In Tessa:* error rates are multiplied by this value and T1/T2 times are divided
  by it. The base rates are single-qubit gate 3e-4, `cz` 8e-3, readout 1e-2,
  T1 300 µs and T2 250 µs.

**Seed** (0–999)
- *In practice:* the same seed gives the same noise sample; frame *i* of
  **Run frames** uses `seed + i`. It also seeds **Mask %**.
- *In Tessa:* seeds the Aer sampler.

### View and colour (instant)

**View** (fidelity / material)
- *Fidelity*: gain fixed at 1 and no mask, so what you see is the true decoded
  mesh.
- *Material*: **Gain** and **Mask %** apply, and the title says
  "gain 3.0× (noise exaggerated)".

**Colour** (error / qubit / φ colour / plain)
- *Error*: each vertex's mean distance from its original position over the
  readouts, in mesh units. The scale runs from light (no error) to dark (the
  largest error in this run); the legend shows the values. It is always the
  true decoded error, even in the material view. On a chip backend it shows the
  bad qubits.
- *Qubit*: which data qubit holds each vertex, one distinct colour per qubit in
  use (hues spread by the golden ratio). With the mesh layout, cells show up as
  patches, so the mesh becomes a picture of the chip. With the naive layout on
  xyz, a vertex's x, y and z sit on different qubits, so the colour shows the
  qubit holding x (the title says so). Hover a vertex for its qubit number.
- *φ colour*: pair mode's colour field as a hue (0 to 1 around the colour wheel):
  the original field before a run, and the field decoded from φ, averaged over
  the readouts, after one.
- *Plain*: a single colour.

Both apps use the same colour scales and show a legend for every scale (error,
φ colour, and edge length when edges are shown). The edge scale spans the true
edge lengths, so a measured length outside that range shows as the end colour.
The web view is unlit by default so every face shows its exact mapped colour;
**Shading** turns lighting on to read the shape, at the cost of darker faces
on the far side.

### Post-processing and animation (instant, no quantum work)

**DRR** (none / min_max / mean_std / percentile / abs_max / min_avg_max)
- *In practice:* undoes the shrink that hardware noise causes.
- *In Tessa:* `restore_field(dynamic_range=...)` stretches each decoded frame so a
  few summary statistics match the original values. `mean_std` is the paper's
  choice for hardware.

**Gain (material)** (0–10×; default 3)
- *In practice:* how visible the noise is, in the material view only.
- *In Tessa:* nothing. It is applied as `out = orig + gain · mask · (decoded − orig)`.

**Mask %** (0–100; material view only)
- *In practice:* only this percentage of vertices, chosen at random, move.
- *In Tessa:* nothing. It is the `mask` term in the blend above.

**Frames** (2–60)
- *In practice:* how many readouts **Run frames** collects.
- *In Tessa:* the circuit is encoded once, then `decode` runs once per frame.

**In-between** (0–30)
- *In practice:* how many eased in-between positions are drawn between frames,
  so vertices glide instead of jumping.

**Speed (fps)** (1–60)
- *In practice:* playback speed, changeable live while it plays. It is also the
  GIF frame rate.

**Loop**, **Start from original**, **Auto-run**
- *Loop*: playback repeats from the first frame.
- *Start from original*: the animation begins at the undistorted mesh.
- *Auto-run*: every encoder or readout change decodes a single frame straight away.

### Buttons

- **Run**: one decode with the current seed, shown as a still result.
- **Run frames**: decodes **Frames** readouts and starts playback.
- **Play/Pause**: starts or stops playback.
- **Read edges** / **Hide edges**: reads every mesh edge's length from the encoded
  state. While the edges are shown, the button reads **Hide edges** and clicking it
  removes the overlay. Changing the mesh also removes it.
  - Each edge and axis is encoded as an isolated two-qubit Tessa cell holding the
    two endpoint values. A Hadamard on its address qubit gives
    `K = 2P(0) − 1 = cos(π·Δv/2)`, so the length comes from the coherence rather
    than from decoding each vertex.
  - Each pair's range is zoomed to the largest edge component, so K uses its
    whole range. Reference pairs with equal values correct for each coupler's
    contrast loss.
  - On a chip backend the pairs are spread over its lowest-error couplers.
  - The edges are drawn coloured by measured length (viridis), and the status
    line gives the error against the true lengths.
  - This follows the paper's coherence experiment B (isolated pairs plus a
    reference circuit), not the full-lattice version. Edges are read from xyz
    whatever the payload.
- **Curves**: opens a window with error against shots (with a `1/√shots` guide),
  and error and depth against compression for both layouts. It takes 15 decodes.
- **Why Tessa**: opens the resource table.
  - The current values are encoded with Tessa, angle encoding, QCrank (parallel
    QPIXL, Tessa's stated prior art), FRQI and amplitude encoding, and transpiled
    onto the `ibm_phoenix` snapshot.
  - Columns: qubits, whether it fits on the chip, two-qubit gates, depth, SWAPs
    (estimated as the extra CZs from routing divided by 3), measurement settings,
    outcomes per cell, and shots per value.
  - Every row uses generic random data with no compression, so no row gets credit
    for this mesh.
- **Export OBJ**: writes the playback sequence to `exports/obj/frame_000.obj`, ….
- **Export GIF**: writes the playback to `exports/ripple.gif`, the 3D view only.

### Inspecting a vertex (hover while paused)

Pause, or show a single **Run** result, and hover over the mesh. The nearest
vertex is circled. Its tooltip shows the data qubit that holds it, its original
and current coordinates, and the delta. In the fidelity view, the delta is
exactly Tessa's readout error for that vertex.

## Not done yet

- **Quality-aware placement.** Tessa's `slot_order(graph, quality)` could put
  important vertices (silhouette, high curvature) on a chip's best qubits.
- **Full-lattice coherences.** Hadamards on the address qubits of the whole
  encoded mesh, instead of isolated pairs.
- **Real hardware.** Tessa would run whole circuits there with
  `decode(..., per_group=False)`, and that needs its `ibm` extra.

## Code layout

```
tessa_ripple/
  mesh.py      blob, cube, uv_sphere, edges, bfs_order, load_obj, write_obj
  bridge.py    all Tessa calls: Params, payloads, placement, fit_lattice, encode_mesh,
               samplers (incl. chip snapshots), decode_raw, restore, fidelity_curves,
               resources (Why Tessa), read_edges (coherence readout)
  animate.py   run_frames, random_mask, blend, interpolate, timeline, export_obj_sequence
  app.py       matplotlib window: views, colours, controls, playback, export
tests/         meshes, placement, round trips, pair colour, chip backend, resources,
               edge readout, seeding, interpolation, export
```
