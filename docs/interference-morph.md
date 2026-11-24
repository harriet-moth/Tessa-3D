# Tessa Ripple — Interference morph: prototype

Status: **implemented.** In both apps: Morph options → **Interference readout** (the web
version refuses it with noise, which takes minutes per frame). The diagnostics (dephased
control `read(..., dephase=True)`, `coherence`, the `pure` and `incoherent` references)
have no UI; only `ideal` is used, for the error colour. Code: `tessa_ripple/interfere.py`, tests:
`tests/test_interfere.py`, experiments: `dev_interfere.py` → `out/interfere/`
(`noiseless.csv`, `chips.csv`, `costs.csv`, `summary.json`, `progress.png`, `strip.png`).

## Why

The sweep morph (mode "morph") reads each slot along cos α Z + sin α X. Noiseless it is a
cross-fade in cosine space (Tessa's `ideal_frames`), which a classical shape key can
trace. The only quantum part a viewer sees is the noise. The Tessa paper's Appendix D
reads coherences *between* cells by turning address qubits before measuring. This
prototype puts the morph there, so a frame depends on quantum coherence.

## How

- **Selectors.** An integer program (`choose_selectors`, scipy `milp`) picks address
  qubits so every data qubit has at most one selector neighbour. A selector's data
  neighbours form its *star*. 6x6, 10x12, Phoenix and Fez pack perfectly (other
  lattices, e.g. 3x3, 5x5, 5x7, leave a few data qubits uncovered, holding padding
  only): there every data qubit is covered and the morph keeps exactly half of Tessa's
  capacity (Phoenix 396, Fez 224, against 792 and 448 A/B pairs for the sweep morph).
- **Encoding.** Plain `tessa.encode` (single mode). Each slot pair (selector bit 0, bit 1)
  holds A's value and B's value. With the selector in |+⟩, a star is
  (|0⟩|A patch⟩ + |1⟩|B patch⟩)/√2: a superposition of the two meshes' patches.
- **Frame at t.** RY(−β), β = tπ, on every selector, then Z on everything. Tessa's own
  per-slot decode of a selector-0 slot reads the data qubit after projecting the star
  onto cos(β/2)|A⟩ + sin(β/2)|B⟩.
- **Simulation.** Turning a shared address qubit breaks Tessa's one-cell-at-a-time
  readout, so each star runs as one cluster: its data qubits and all their address
  qubits (≤ 13 qubits on Phoenix, ≤ 10 on Fez), with the neighbouring data qubits
  as Tessa's traced halo. Every step reuses Tessa: `cell_circuit` (cluster
  extraction), `codec._run` (transpile, trace halo, sample), `decode_cells`,
  `ideal_frames`, and `restore_field` via `bridge.restore`.
- **References.** `ideal` gives the exact noiseless frame (cluster statevectors,
  entanglement included). `pure` treats each data qubit as alone in its star: the
  half-angles add as vectors, so the qubit follows the great circle. `incoherent`
  dephases the selector: cos θ = cos²(β/2) cos θ_A + sin²(β/2) cos θ_B, which is Tessa's
  cross-fade. `coherence` fits λ in frame ≈ mixed + λ(coherent − mixed).
- **Control.** `read(..., dephase=True)` runs half the shots with Z on every selector and
  sums the counts. That removes the A/B coherence while keeping everything else the
  same, including the noise.

## Results (2026-11-20)

Displacements below are means over the *moving* values (|B − A| > 10 % of the largest).

1. **Endpoints are exact.** t = 0 is A and t = 1 is B to 1e-15, radial and xyz, on
   every graph (tests).
2. **Noiseless, the coherent frame is far from the mixture, but mostly in timing.**
   - At t = 0.2 and t = 0.8 the coherent frame is 14–15 % of the displacement away from
     the mixture (apple and Suzanne, radial and xyz).
   - The coherent morph progresses almost linearly in t. The mixture eases in and out,
     like sin²(πt/2). See `progress.png`.
   - Re-timing the classical cross-fade (choosing the best t′ for each t) leaves only
     0.8–1.6 %. So a classical shape key with a different easing curve gets within 2 %.
   - The entanglement term inside a star (exact against `pure`) is at most 0.6 %:
     these meshes' A and B patches overlap almost fully, so stars barely decohere one
     another.
3. **On the Phoenix and Fez noise models, the coherence survives and can be measured,
   but not frame by frame.**
   - Raw λ is biased by the noise. The control should read 0 but reads −0.24 to
     +1.38, depending on the mesh and t, because noise pulls every frame towards the
     middle.
   - The noise-free quantity is coherent − control (same noise, coherence removed):

     | | apple t = .25 | apple t = .75 | Suzanne t = .25 | Suzanne t = .75 |
     |---|---|---|---|---|
     | Phoenix | 0.86 | 0.69 | 0.60 | 0.89 |
     | Fez | 0.82 | 0.81 | 0.67 | 0.78 |

     The spread over 3 seeds is ≤ 0.04, so each value is more than 15σ from 0.
   - Per vertex, however, the chip error (11–17 % of the displacement at 4,096 shots)
     is as large as the whole coherent-vs-mixed signature (13–15 %). The difference
     shows in the aggregate, not in one frame by eye (`strip.png`).
4. **Cost.**
   - Per run, two-qubit gates and depth are about the same as the sweep morph's
     (Phoenix apple 504 vs 496; the RY on the selectors adds no two-qubit gates).
   - Half the capacity means more runs: apple on Fez 2 runs (sweep 1), Suzanne on Fez
     3 (sweep 2), Suzanne on Phoenix 2 (sweep 1).
   - A frame needs no separate Z reference, but only the selector-0 shots decode it:
     about 80 % of the shots are kept at t = 0.25 and t = 0.75, 50 % at the endpoints.
   - Simulation is slower: 150–300 s per frame on Phoenix (13-qubit noisy clusters),
     6–11 s on Fez.

## Verdict

This is the first morph here whose noiseless frames depend on quantum coherence. The
chips' noise models measurably keep that coherence (coherent − control ≈ 0.6–0.9).

What you can see is mostly a different easing curve, which a classical keyframe could
copy to within 2 %. The entanglement term between neighbouring vertices is under 1 % for
these meshes. Pairs whose patches differ more (larger |θ_A − θ_B|) would raise it.

The quantum effect is real and testable with the control, but it reads as timing, not
as a shape, and is below per-vertex noise on today's simulated chips. A UI would be
worth building only as a "coherence witness" (show λ and the control next to the
morph), not as a visibly different morph.
