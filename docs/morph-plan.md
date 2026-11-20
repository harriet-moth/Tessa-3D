# Tessa Ripple — Morph mode: implementation plan

Status: implemented (see README, Morph). Desktop app only. Payload xyz (default) or radial.
The *blend* style below was later removed: morph is now the basis sweep only, with the
transition decode as the default (raw stays as an option).
Meshes too large for one run are read in several batches (see Batching).

## What Morph does

Each slot's qubit holds two meshes at once. Mesh A (the selected Shape) sets the polar
angle θ and is read in Z. Mesh B (an uploaded OBJ) sets the azimuth φ and is read in X.
This is Tessa's existing pair mode, the same machinery as Mode "θ + colour φ", with B's
geometry in φ instead of a hue. It costs no extra qubits or two-qubit gates, only the
second measurement setting that pair mode already runs.

The morph runs one of two ways, chosen in the Morph options box:

| Style | Each frame | Morph slider t | At t = 1 |
|---|---|---|---|
| **Blend** | Tessa's pair decode: A′ from Z, B′ from X (two settings) | classical lerp A′ → B′; moving t needs no new readout | B′, B with its own (larger) shot noise |
| **Basis sweep** | one setting: RY(−α) on every data qubit, then Z, with α = t·π/2 | picks the measurement axis; every t is a new readout | "B seen through A": B scaled by sin θ, contracted toward mid-range |

Predicted look (from the design discussion, checked by simulation at 4,096 shots and the
default pair clamp 0.2π): B's angle noise is 3.2× A's where both sit at the clamp and
1.0× at mid-range, so B shimmers most at A's extremes. With A = B the basis sweep still
breathes: values that span 0 → 1 come back at about 0.24 → 0.78 at t = 1, with the
outermost ~8% at each end folding back inward.

## UI (app.py)

- **Mode** gets a fifth option, `morph A→B` → `MODE_LABELS[...] = "morph"`.
- A **Morph options** box appears only while Mode is morph (its axes are created at
  start-up and toggled with `set_visible`, like the qubit map). Proposed position: an
  inset at the lower left of the 3D view, above the colour legends, so it never fights
  the qubit map. Confirm the exact placement visually in M4. Contents:
  - **Style** radio: `blend` | `basis sweep`
  - **Shared topology** checkbox: off = closest point, on = pair by index
  - **Upload B (.obj)** button, and a label with B's file name and vertex count
  - **Morph t** slider, 0–1 (blend weight, or α = t·π/2 for the sweep)
  - **Hold t** checkbox: Run frames keeps t fixed and varies only the seed (today's
    ripple). Off: Run frames sweeps t from 0 to 1 over the frames, one seed per frame,
    so the morph and the noise move together. Loop plays it back and forth.
- While Mode is morph:
  - **Payload** switches to xyz when morph is first selected; radial stays available.
    Changing it re-encodes as usual.
  - **Clamp φ** follows **Clamp θ**, so equal geometry gets equal angles in both fields.
    The φ slider is greyed out and shows the shared value.
  - With no B uploaded, the status line reads "Morph: upload a B mesh" and nothing
    encodes.
- **Upload B**: the default macOS matplotlib backend does not mix well with a Tk dialog,
  so on macOS use `osascript -e 'POSIX path of (choose file of type {"obj"})'` through
  `subprocess`, with `tkinter.filedialog` as the fallback on other platforms. A
  cancelled dialog changes nothing.
- **Colour** radio gains `morph Δ`: per-vertex |B − A| after alignment, so you can see
  where the two shapes disagree before running anything.
- **Info panel** in morph mode: B's name, vertex count, correspondence method, mean
  aligned distance |B − A|, "values: 2 × N·comps on Q qubits", and "2 runs" (with the
  values per run) whenever batching kicks in.
- **Batch placement** radio in the Morph options box, shown only on a chip backend when
  there is more than one run: `next-best qubits` (default) | `same qubits`.
- **Qubit-map hover** shows both of the slot's values (A in θ, B in φ).

## Pipeline (new module `tessa_ripple/morph.py`, plus small changes in bridge.py)

### 1. Load B
`load_obj(path)` as today (it already centres and normalises each mesh on its own, so
world placement is discarded either way).

### 2. Align B to A
`align(b_verts, a_verts)`: translate B's centroid onto A's and scale B so its RMS
distance from the centroid equals A's. Uniform scale only: no rotation fit (no ICP).
B's orientation stays as authored, which keeps a deliberately rotated B rotated.

### 3. Correspond: one B point per A vertex
- **Shared topology on**: pair by index. Vertex counts must match, else a clear error:
  "Shared topology: A has 507 vertices, B has 512. Untick Shared topology to sample by
  closest point." Resolution needs care here, because `decimate` picks collapses from A's
  edge lengths, so decimating B on its own would give a different topology. Plan:
  `decimate` also returns its collapse log (the ordered `(a, b)` merges), and a new
  `replay_decimation(verts, log)` applies the same merges to B (midpoint of the same
  pairs). B is then loaded at full size and replayed down to A's Resolution.
- **Shared topology off**: closest point on B's surface for each A vertex. B stays at
  full resolution for accuracy. Fan-triangulate B's faces, build a `scipy.spatial.cKDTree`
  on triangle centroids, take the 16 nearest triangles per A vertex, and keep the exact
  closest point among them (vectorised point-triangle projection, Ericson's method).
  scipy is already installed with qiskit; add it to `pyproject.toml` explicitly.

### 4. Payload
- **xyz (default)**: every coordinate of A and B comes from readout. Slot j holds one
  coordinate of A in θ and the same coordinate of the matched B point in φ, so B needs
  no extra slots. Both fields share one range: ±max(|A|, |B|)·(1 + headroom), with
  centre 0 at A's (and aligned B's) centroid. Expected look: noise moves vertices in
  every direction, so the surface roughens rather than breathes, and the sweep ghost is
  anisotropic (B's x is squeezed where A's |x| is large, and so on per axis).
- **radial**: one radius per vertex; the direction from the centre is classical.
  B's radius r_B,i = |b_i − c_A|. Directions: see open decision 1. Shared range
  (0, max(r_A, r_B)·(1 + headroom)). Three times cheaper, smoother motion.

### 5. Encode (bridge.encode_mesh, new `mode == "morph"` branch)
- `encode(graph, a, b, clamp=c, clamp_b=c, value_range=rng, value_range_b=rng)`, with
  `b = fill(graph, slots, B_payload)` (B's xyz coordinates or radii, same shape as A's). The slots are A's (mesh-aware BFS patches), so a smooth
  correspondence keeps B's cells smooth and compressible too.
- `Params` gains `target: str | None` (B's path), `shared_topology: bool`,
  `morph_style: str` and `morph_t: float`. `encoder_key` adds target and
  shared_topology. `readout_key` adds morph_t only for the sweep (blend's t is applied
  after decoding).
- `Encoded` gains `target` (aligned, corresponded B, shape (N, 3)), `target_r`,
  `dirs_b`, and `stats["values"] = 2N`.
- Add "morph" to `MODES`. `resources()` treats morph like colour (two random fields).

### 5b. Batching (new `tessa_ripple/batch.py`)
When N·comps values exceed the graph's capacity, the mesh is read in K =
ceil(N·comps / capacity) runs. Capacities: 792 slots on the 10×12 lattice and
ibm_phoenix, 448 on ibm_fez. For Suzanne in xyz morph: low 1 run everywhere, mid
1 run on Phoenix (nearly full) and 2 on Fez, high 2 runs on Phoenix and 4 on Fez.
- **Split by surface patch.** `grow_patches` cuts the mesh into K connected regions of
  whole vertices, each with about N/K vertices, so the runs are even (1,521 values →
  761 + 760, not 792 + 729) and a seam is not a noise-level step.
- **One encoding per batch** on the whole graph, each with its own slots and fill
  padding. The shared value range is computed once over the whole mesh, so every batch
  maps values to angles identically.
- **Decode one after another, then stitch** back into one (N, 3) array by vertex index.
  Batch k uses seed `seed + 1000·k` so batches never share noise.
- **Batch placement (chip backends)**:
  - `next-best qubits` (default): batch k's cell order is the quality ranking rotated
    by the number of data qubits the earlier batches took first, so the best qubits are
    spread across batches instead of always serving the first patch. The chip's noise
    fingerprint does not repeat as a copy.
  - `same qubits`: every batch uses the ranking from the top. Its noise pattern repeats
    once per batch, which may be wanted as a texture.
  - Note: when each batch nearly fills the chip (Phoenix, 761 of 792), the two settings
    land on almost the same qubits; the difference shows when batches are small relative
    to the chip (Fez, 4 runs of ~381 of 448).
  - On lattice backends (noiseless, flat noise) placement makes no difference and the
    setting is hidden.
- **Both morph styles work unchanged.** Every batch gets the same sweep angle α; the
  per-qubit α for the later collision morph is addressed as (batch, data qubit).
- **Cost**: frame time × K. On the simulated chips that is a few seconds per batch per
  frame, so 12 frames of high-resolution Suzanne on Phoenix take a few minutes. The
  status line shows a time estimate before Run frames starts.
- **Story**: the mesh becomes K separate quantum states rather than one. Decoding is
  unaffected (Tessa reads each data qubit independently), and the info panel says
  "2 runs" so the piece doesn't claim "one chip, one state".
- Built generic in bridge (any mode can batch), but only switched on for morph in this
  pass. Lifting the cap for the other modes is a later one-line change plus tests.

### 6. Read
- **Blend**: `decode_raw` returns (r_A′, r_B′) from Tessa's pair decode. `restore` builds
  A′ and B′, and the classical layer computes the frame as lerp(A′, B′, t).
- **Basis sweep**: `sweep_circuit(e, alpha)` copies the pair circuit and adds `ry(-alpha)`
  on the data qubits only (address qubits are never rotated, as in Tessa's own
  `_measured`). It is decoded as a single field with A's clamp and range, the same trick
  as the existing `phi` mode at α = π/2. `alpha` may be a scalar or one angle per data
  qubit. Today the UI only sends a scalar; the per-qubit form is there for the later
  collision morph, where only the qubits under the contact patch rotate. Circuits are
  cached per α.
- **Expected positions** for error colouring and the Fidelity view: blend →
  lerp(A, B, t); sweep → noiseless closed form, angle = arccos(cos α·cos θ + sin α·sin θ·cos φ),
  mapped back through A's angle → value map. `positions()` in app.py changes from one
  reference mesh (`self.verts`) to one reference per frame, and the Material view's gain
  scales the deviation from that per-frame reference.

### 7. Animate and export
`animate.run_frames` gains a `ts` argument (one t per frame). Blend: one pair decode per
frame. Sweep: one sweep readout per frame at that frame's α. Export OBJ and GIF work
unchanged, since frames are still position arrays on A's faces.

## Tests (tests/test_morph.py)

1. `align`: centroids coincide and RMS radii match.
2. Shared topology with mismatched vertex counts raises the clear error.
3. `replay_decimation` on an exact copy of A reproduces `decimate(A)` exactly, with the
   same faces.
4. Closest point: a vertex of B maps to itself, and points off a cube project onto the
   correct face.
5. Blend, noiseless, 4,096 shots: t = 0 within 2% of A; t = 1 within 3% of B.
6. Sweep at t = 0 matches theta mode within shot noise.
7. Sweep at t = 1, noiseless, 65,536 shots, matches the closed form. This pins the sign
   of the RY rotation.
8. A = B: blend endpoints agree; the sweep at t = 1 spans about 0.24–0.78 of the range,
   with the predicted fold at the ends.
9. Morph forces clamp_b = clamp; xyz and radial both round-trip (xyz within 3% of A
   at t = 0 noiseless).
10. `encoder_key` changes with target and shared_topology; `readout_key` changes with t
    for the sweep only.
11. `sweep_circuit` accepts a per-qubit α array of the right length and rejects others.
12. Batching: high-resolution Suzanne in xyz morph gives K = 2 on Phoenix (761 + 760
    values) and K = 4 on Fez; every vertex appears in exactly one batch; noiseless
    stitched result within 3% of A.
13. Batch seeds differ, so two batches never share a noise draw.
14. `next-best qubits` gives batch 2 a different first data qubit from batch 1 on Fez;
    `same qubits` gives the same one.
15. A mesh that fits in one run is unchanged by the batching code (same slots and
    circuit as before).

## Milestones

| | Work | Rough size |
|---|---|---|
| M1 | morph.py: align, closest point, decimation replay + tests 1–4 | 1 day |
| M2 | bridge: morph encode + blend decode + tests 5, 9, 10 | 0.5 day |
| M3 | sweep circuit, closed form, per-qubit α + tests 6–8, 11 | 1 day |
| M3b | batching: patch split, per-batch encode/decode/stitch, batch placement + tests 12–15 | 1 day |
| M4 | app: Mode option, Morph options box, upload dialog, forced payload/clamp, info | 1 day |
| M5 | per-frame references, `morph Δ` colour, t-sweep frames, hover | 0.5–1 day |
| M6 | README section, try Suzanne → sculpted Suzanne and Suzanne → sphere in xyz and radial, on noiseless and phoenix | 0.5 day |

Total about 5.5–6 days.

## Open decisions

1. **Radial directions (radial payload only; xyz has no such choice).** With radial
   payload only the radius is quantum; each vertex's
   direction is classical. Options:
   - (a) Keep A's directions throughout. B appears as its radius along A's directions:
     exact for similar meshes, a "B wrapped onto A" for different ones. Purely quantum
     motion.
   - (b) Rotate each direction from A's to B's (slerp) with t. Lands on B's true shape,
     but the turning of directions is classical and should be labelled as such.
   - Recommendation: (b) as the default, with an "A's directions only" checkbox for the
     pure version.
2. **Closest point on very different shapes** (Suzanne → torus, say) can pile many A
   vertices onto one region of B, leaving parts of B unrepresented. That's acceptable for
   art, but the info panel should report coverage (the share of B's area that received a
   point).
3. **Later, collision**: drive the per-qubit α from contact distance. Mesh-aware layout
   means one data qubit is one surface patch, so the morph spreads patch by patch from the
   contact point.
