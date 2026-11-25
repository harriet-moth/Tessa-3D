# Tessa Ripple for Blender

A Blender 5.1+ extension (Apple Silicon Macs) that encodes the active mesh's vertices
with Moth's Tessa quantum encoder, reads them back, and stores each readout as a
shape key. Everything is bundled: qiskit, qiskit-aer and scipy as wheels, and Tessa
and the Tessa Ripple core as source.

## Use

View3D > N > **Tessa**. Select a mesh (A) in Object Mode, choose settings, then:

- **Encode & read**: one readout (Morph: at Morph t), stored as one shape key.
- **Make frames**: an animation, one shape key per frame, keyframed **Timeline spacing**
  frames apart from the scene's start frame and blended linearly. The scene's frame
  range is set to the animation, so Space plays and loops it.
  - Morph A→B with *Frames show: Morph A→B*: t runs 0 → 1 across the frames, so A
    turns into B over time.
  - Morph with *Noise at t*, or any other mode: every frame is a fresh readout
    (seed, seed + 1, …), so the mesh moves only by the shot-to-shot readout variance.

Esc (or Cancel) stops a run after the current readout. Make frames shows an estimate
of its time from the last multi-frame run, and the first run after Blender starts is as
fast as later ones: qiskit and the core load in the background a moment after start-up.

- **Readout** (Mode θ, non-spherical payloads): *Z*, *Bloch* or *Snap*, as the desktop app.
- **Output > Loop / Start from original / Ease in-betweens / Speed (fps)**: playback, as the
  web app's drawer. *Start from original* puts the unmodified mesh on the start frame;
  *Loop* adds a final glide back to the start and ends the timeline one frame before it, so
  playback wraps without holding a frame twice; *Ease* eases in and out of every readout
  instead of blending linearly; *Speed* is the scene's frame rate. All re-apply to the
  last result without a new readout.

- **Gain** 1 shows the true decoded positions; above 1 exaggerates the deviation.
  Gain, timeline spacing and colour re-apply to the last result without a new readout,
  and each only rewrites its own part (positions, keyframes or attributes).
- **Colour** sets `tessa_colour` (see it in Solid shading > Color > Attribute), as the
  desktop app's Colour modes: *Error* per vertex, *Qubit* (one colour per data qubit,
  which is where the mesh and naive **Layouts** differ; not for spectral, whose qubits
  hold modes), *Morph Δ* (|B − A|) or *None*. Every run also writes the data attributes
  `tessa_error` (float), `tessa_qubit` (int; not spectral) and `tessa_morph_delta`
  (float; morph), for shaders and Geometry Nodes.
- **Morph A→B**: pick B (any mesh object). B is centred and scaled onto A, so its
  rotation relative to A counts but its position and size do not.
  Encode & read uses Morph t; Make frames follows *Frames show* (above).
  With **Front**, the turn spreads from a source vertex: *Auto* (the vertex that moves
  most) or *Selected* (select a vertex of A in Edit Mode, then press it).
- The **spherical** payload fills θ and φ itself, so it runs with Mode θ (Z) only.
- **Clear** removes Tessa's keys, their animation and its attributes.
  **Export OBJs** writes each readout as an OBJ file (Y-up).

The encoding is cached: changing only readout settings (seed, shots, readouts, DRR,
morph t) skips the encode.

## Develop

```bash
python3 build_ext.py                      # copy ../tessa_ripple core + Tessa into core/, tessa/
blender --command extension validate
blender --command extension build --output-dir ../dist
```

Edit the core in `../tessa_ripple`, never in `core/` (rebuilt by `build_ext.py`).
For live editing, symlink this folder into
`~/Library/Application Support/Blender/5.3/extensions/user_default/tessa_ripple`,
then Preferences > Get Extensions > Refresh Local, and F3 > Reload Scripts after edits.

| File | Role |
| --- | --- |
| `blender_manifest.toml` | Extension metadata, wheel list, permissions |
| `__init__.py` | Registers the classes and `scene.tessa` |
| `props.py` | Settings (the desktop app's widgets) and `params()` for `bridge.Params` |
| `engine.py` | Quantum pipeline in a worker thread: numpy only, no bpy |
| `mesh_io.py` | Mesh ⇄ numpy, shape keys, keyframes, attributes and colours |
| `ops.py` | Run (threaded, modal), Cancel, Front source, Clear, Export OBJs, Check |
| `ui.py` | Sidebar panels |
| `build_ext.py` | Generates `core/` and `tessa/` (excluded from the zip) |

Not carried over from the desktop app, on purpose: Read edges, Curves and Why Tessa
(charts, not mesh output), Resolution (use a Decimate modifier and apply it) and Export
GIF (render the animation instead).

## Before sharing

- Add Tessa's LICENSE to `tessa/` (`build_ext.py` warns when it is missing), and
  confirm with Moth that redistributing Tessa's source is fine.
- Bump `version` in the manifest for every build you share.
