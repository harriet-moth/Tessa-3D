# Tessa Ripple

Tessa Ripple encodes the vertex coordinates of a 3D mesh into a quantum state with
Moth's [Tessa](https://github.com/moth-quantum/Tessa) encoder, reads them back, and
moves each vertex to wherever the readout puts it.

Nothing is added to the circuit on purpose. Every bit of movement comes from
variation Tessa's readout already has: **shot noise** (finite measurements) and,
optionally, **simulated hardware noise**. Reading the same circuit out several times
with different seeds and gliding the mesh from one result to the next makes that
noise look like a ripple.

## Install and run

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/). Tessa is not on PyPI, so it
is pinned to a git commit in `pyproject.toml`.

```bash
uv venv --python 3.13 && uv pip install -e ".[dev]"
```

```bash
.venv/bin/tessa-ripple
```

```bash
.venv/bin/python -m pytest
```

Quick start: press **Run frames** on the default cube. Raise **Gain** or lower
**Shots** for a bigger ripple, or switch **Backend** to *noisy* to see hardware noise.

## How a mesh goes through Tessa

1. **Mesh.** A cube or UV sphere gives `N` vertices, i.e. `3N` numbers.
2. **Flatten.** Coordinates are laid out axis-major: all x, then all y, then all z.
3. **Prepare.** `tessa.prepare_field` optionally quantises them (Quant bits).
4. **Pick a lattice.** `tessa.Lattice(rows, cols)` is a checkerboard of *data* qubits
   (which hold values) and *address* qubits (shared controls). A data qubit with `k`
   address neighbours stores `2^k` values, so capacity is the sum of `2^k` over data
   qubits. The app picks the smallest near-square lattice (up to 8×8, 64 qubits)
   whose capacity holds the values.
5. **Encode.** `tessa.encode` maps each value linearly from one shared, symmetric
   range `(-m, m)` onto a rotation angle and builds a measurement-free circuit of
   uniformly controlled `RY` rotations (multiplexers). Using one range for all
   three axes keeps the mesh from being stretched per axis. This runs **once** per
   mesh and encoder setting.
6. **Decode.** `tessa.decode` adds the measurements, runs the circuit on a local Aer
   sampler, and turns the measured probabilities back into angles and then values.
   Tessa cuts the circuit into one small cell per data qubit and simulates each on
   its own, which is fast and exact because the cells commute. Each frame is one
   decode with its own seed.
7. **Restore.** `tessa.restore_field` applies optional dynamic-range recovery (DRR),
   and the values are reshaped back to `(N, 3)`.
8. **Blend and animate.** `out = orig + gain · mask · (decoded − orig)`, then
   smoothstep-eased in-between positions glide from frame to frame.

Steps 1–8 are plain numpy-in, numpy-out functions. All Tessa calls are in
[`tessa_ripple/bridge.py`](tessa_ripple/bridge.py).

## The controls

Controls fall into three tiers by how much work a change costs:

| Tier | Controls | What a change does |
|---|---|---|
| Encoder | Shape, Resolution, Mode, Clamp θ, Clamp φ, Compression, Quant bits, Headroom | Rebuilds the circuit (milliseconds). Old frames are cleared. |
| Readout | Backend, Shots, Noise ×, Seed | Needs a new decode: press **Run** or **Run frames**, or tick **Auto-run**. The status line says "frames stale" until you do. |
| Post / animation | DRR, Gain, Mask %, Frames, In-between, Speed, Loop, Start from original | Applied to cached frames instantly. No quantum work (except **Frames**, which only takes effect on the next **Run frames**). |

The panel on the right shows the current mesh, lattice size, qubit count, capacity
used and circuit depth. The title above the mesh shows the playback step and
**mean err**: the average distance between decoded and original coordinates
(before gain), as a fraction of the mesh's size. So 0.02 means 2 % of the bounding box.

### Mesh

**Shape** (cube / sphere)
- *In practice:* chooses the object.
- *In Tessa:* changes how many values are encoded, which can change the lattice
  and so the qubit count and circuit depth.

**Resolution** (low / mid / high)

| | low | mid | high |
|---|---|---|---|
| Cube (vertices per edge) | 3 → 26 verts | 4 → 56 verts | 5 → 98 verts |
| Sphere (segments × rings) | 8×6 → 42 verts | 12×8 → 86 verts | 16×8 → 114 verts |

- *In practice:* more vertices means a finer mesh and a slower decode.
- *In Tessa:* `3N` values must fit the lattice capacity, so a higher resolution
  picks a bigger lattice (more qubits, deeper circuit, more cells to simulate). If
  it does not fit in 8×8, the status line says so; switch to pair mode.

### Encoder

**Mode** (single / pair)
- *In practice:* pair mode fits about twice as many values on the same qubits, at
  roughly twice the error. Use it when a mesh is too big for single mode.
- *In Tessa:* each cell is a qubit state `cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩`.
  - `single` stores one value in θ and is read with a Z measurement.
  - `pair` stores a second value in the phase φ at no extra two-qubit gate cost.
    It is read with both Z and X measurements, so twice as many circuits run.
  - The app splits the flat coordinate list in half, the first half into θ and
    the second into φ.

**Clamp θ** (0–0.45 π; 0 means Tessa's default)
- *In practice:* keeps values away from the extreme ends of the angle range. A
  larger clamp leaves less of the range for the data, so shot noise becomes a
  larger share of each value and the mesh shimmers more.
- *In Tessa:* `encode(clamp=...)`. Values map to θ in `[c, π − c]` instead of `[0, π]`.
  - Tessa's default is 0 in single mode and 0.2 π in pair mode.
  - Pair mode needs θ > 0, because a cell at θ = 0 or π is `|0⟩` or `|1⟩` and
    carries no phase. That is why 0 on this slider means "use Tessa's default"
    rather than "no clamp".

**Clamp φ** (pair mode only; 0–0.45 π, 0 means Tessa's default of 0)
- *In practice:* the same idea for the second half of the coordinates, held in
  the phase.
- *In Tessa:* `encode(clamp_b=...)`. φ is held in `[c, π − c]`. It has no effect
  in single mode.

**Compression %** (0–100)
- *In practice:* drops gates to make a shallower circuit (watch **depth** fall),
  at the cost of accuracy. The distortion is the same on every run, so it
  **reshapes** the mesh but does not animate it. Mesh coordinates are smooth, so
  moderate compression often loses little.
- *In Tessa:* `encode(compression=...)`. Each multiplexer's angles are turned into
  coefficients with a Walsh–Hadamard transform, and the smallest `compression` %
  of those coefficients are set to zero. Their rotation gates are then left out.

**Quant bits** (off, 2–8)
- *In practice:* snaps every coordinate to `2^bits` evenly spaced levels before
  encoding. Low values give a blocky, terraced shape. Like compression, this is a
  fixed reshape, not motion.
- *In Tessa:* `prepare_field(quant_bits=...)` quantises uniformly over the data's
  own min–max range. The quantised field also becomes the reference that DRR
  matches against.

**Headroom** (0–50 %)
- *In practice:* leaves spare room at the edges of the encoding range. With no
  headroom, the outermost vertices sit at the very end of the range, where noise
  can only push them inward, so they move less than the rest. Some headroom lets
  edge vertices move freely. Too much wastes resolution and makes the noise
  relatively larger.
- *In Tessa:* sets `value_range=(-m, m)` with `m = max|coordinate| × (1 + headroom)`.
  The same range is used for all three axes and both pair fields.

### Readout

**Backend** (noiseless / noisy)
- *In practice:*
  - *Noiseless:* only shot noise, which gives an even, zero-mean shimmer.
  - *Noisy:* adds simulated hardware errors, which give larger movement that also
    drifts: vertices sag toward the centre of the range, and the shape shrinks.
    Noisy decodes are slower.
- *In Tessa:* `tessa.backends.aer_sampler` or `noisy_sampler(rates=...)`. The noisy
  sampler builds an Aer noise model with:
  - depolarising and thermal-relaxation errors on single-qubit gates and on
    every `cz` edge,
  - bit-flip readout error on every qubit.

  Tessa's decode also carries the noise of neighbouring cells' gates on shared
  address qubits into each cell (`halo="traced"`).

**Shots** (2^8–2^16, i.e. 256–65,536; default 4,096)
- *In practice:* **the main ripple control.** Fewer shots give bigger, noisier
  jumps, and more shots give a calmer, more accurate mesh. Error falls roughly as
  `1/√shots`, so 4× the shots halves the jitter. Decode time grows with shots.
- *In Tessa:* `decode(shots=...)`, the number of times each cell circuit is
  measured in each basis. Each value is estimated from the measured frequencies,
  so a finite number of shots means statistical error.

**Noise ×** (0–3; noisy backend only)
- *In practice:* how strong the simulated hardware is. 1 is typical of IBM
  hardware, 0 is effectively noiseless, and 3 is a bad day.
- *In Tessa:* scales the `rates` dict passed to `noisy_sampler`. Single-qubit,
  `cz` and readout error rates are multiplied by the value, and T1/T2 coherence
  times are divided by it. The base rates are single-qubit gate 3e-4, `cz` 8e-3,
  readout 1e-2, T1 300 µs and T2 250 µs, which are Tessa's own fallback averages.

**Seed** (0–999)
- *In practice:* picks which noise sample you get. The same seed always gives
  the same result, so runs are repeatable. A new seed gives a fresh sample. It
  also seeds the random vertex choice for **Mask %**.
- *In Tessa:* seeds the Aer sampler (`aer_sampler(seed=...)`). In **Run frames**,
  frame *i* uses `seed + i`, so the frames are independent samples of the same
  noise.

### Post-processing (instant, no quantum work)

**DRR** (dynamic range recovery: none / min_max / mean_std / percentile / abs_max / min_avg_max)
- *In practice:* undoes the shrinking that hardware noise causes. Use it with the
  noisy backend to pull a collapsed mesh back out to its true size while the
  ripple stays. It does little when noiseless, because there is little to recover.
- *In Tessa:* `restore_field(dynamic_range=...)` stretches each decoded frame so a
  few summary statistics match the original (prepared) coordinates. It uses all
  `3N` values together, not each axis separately.
  - `min_max`: match minimum and maximum.
  - `mean_std`: match mean and standard deviation. The paper's choice for hardware.
  - `percentile`: match the 2nd and 98th percentiles.
  - `abs_max`: match the largest absolute value.
  - `min_avg_max`: match minimum, mean and maximum.

**Gain** (0–10×; default 3)
- *In practice:* how visible the noise is. Raw movement at sensible shot counts is
  only a few percent of the mesh, so gain scales it up. Gain 0 shows the original
  mesh exactly, and gain 1 shows the true decoded positions.
- *In Tessa:* nothing. It is applied after decoding as
  `out = orig + gain · mask · (decoded − orig)`. It changes how big the noise
  looks, not where it comes from.

**Mask %** (0–100)
- *In practice:* only that percentage of vertices, chosen at random, move. The
  rest stay put. 100 moves everything.
- *In Tessa:* nothing. It is the `mask` term in the blend above, seeded by **Seed**.

### Animation (instant, no quantum work)

**Frames** (2–60)
- *In practice:* how many readouts **Run frames** collects. More frames give a
  longer, less repetitive animation. Run time is about frames × the time per
  frame; the status line shows progress and the seconds per frame from the last run.
- *In Tessa:* the circuit is encoded once, then `decode` is called this many
  times with seeds `seed, seed+1, …`.

**In-between** (0–30)
- *In practice:* smoothness. This many eased in-between positions are drawn
  between each pair of frames, so vertices glide instead of jumping. 0 gives a
  raw flicker.
- *In Tessa:* nothing. Linear interpolation with smoothstep easing.

**Speed (fps)** (1–60; default 24)
- *In practice:* playback speed in steps per second, changeable live while it
  plays. It is also the frame rate of the exported GIF. One readout-to-readout
  glide takes `(In-between + 1) / Speed` seconds, so lower Speed or raise
  In-between for a slower, smoother ripple. Very high speeds on large meshes are
  capped by how fast matplotlib can redraw.
- *In Tessa:* nothing.

**Loop** (on / off)
- *In practice:* when on, playback returns from the last frame to the first and
  repeats. When off, it stops at the end.
- *In Tessa:* nothing.

**Start from original** (on / off)
- *In practice:* when on, the animation begins at the undistorted mesh and glides
  into the first readout.
- *In Tessa:* nothing.

**Auto-run** (on / off)
- *In practice:* when on, every encoder or readout change decodes a single frame
  straight away instead of waiting for **Run**. Convenient on small meshes, but
  sluggish with noise on large ones.
- *In Tessa:* calls `decode` once per change.

### Inspecting a vertex (hover while paused)

Pause playback (**Play/Pause**), or show a single **Run** result, then move the
mouse over the mesh. The nearest vertex within a few pixels is circled in red,
and a tooltip shows:

- **original**: the pre-encoding coordinate from the mesh.
- **current**: where that vertex is drawn at this step, after decoding, DRR,
  gain and mask (the tooltip header shows the gain and step).
- **delta** and **|delta|**: the movement, per axis and as a distance.

The tooltip disappears when playback resumes or anything redraws the mesh.
Because it picks the closest vertex on screen, a vertex at the back can be picked
where it overlaps one at the front; rotate the view to reach it. At gain 1 the
delta is exactly Tessa's readout error for that vertex.

### Buttons

- **Run**: one decode with the current seed, shown as a still result.
- **Run frames**: decodes **Frames** readouts and starts playback.
- **Play/Pause**: starts or stops playback.
- **Export OBJ**: writes the current playback sequence, including in-betweens, to
  `exports/obj/frame_000.obj`, `frame_001.obj`, … with gain, mask and DRR applied.
- **Export GIF**: writes the playback to `exports/ripple.gif` with Pillow. It
  records the whole window, controls included.

## Where the movement comes from (and where it doesn't)

| Source | Changes each run? | Effect |
|---|---|---|
| Shot noise (Shots) | Yes | Even, zero-mean shimmer; this is the ripple |
| Hardware noise (Backend, Noise ×) | Yes | Larger jitter plus a drift toward the centre; DRR counters the drift |
| Compression, Quant bits, Clamp, Headroom | No | Fixed reshaping or accuracy changes; they alter the mesh but do not animate it |
| Gain, Mask | No | Make the existing noise more or less visible |

## Code layout

```
tessa_ripple/
  mesh.py      cube(n), uv_sphere(segments, rings), write_obj()
  bridge.py    Params, fit_lattice, encode_mesh, decode_raw, restore, read_mesh  (all Tessa calls)
  animate.py   run_frames, random_mask, blend, interpolate, timeline, export_obj_sequence
  app.py       matplotlib window: 3D view, controls, playback, export
tests/         mesh counts and validity, round-trip accuracy, seeding, interpolation, export
```

`mesh.py`, `bridge.py` and `animate.py` have no UI state, so a Blender add-on can
call them directly later. Real hardware is out of scope for V1. Tessa runs whole
circuits there with `decode(..., per_group=False)` and needs its `ibm` extra, and
nothing in this design blocks that.
