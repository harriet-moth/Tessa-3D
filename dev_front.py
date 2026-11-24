"""Morph front prototype: run the experiments of docs/morph-front-plan.md headlessly.

    .venv/bin/python dev_front.py            # everything (several minutes)
    .venv/bin/python dev_front.py --quick    # noiseless backend and 4,096 shots only

Writes ``out/front/front.csv`` (one row per mesh, payload, backend, shots, width, t),
``out/front/chips.csv`` (Phoenix vs Fez angles), ``out/front/summary.json`` and image
strips ``out/front/strip_*.png``.
"""

import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out" / "front"
PKG = ROOT / "tessa_ripple"
MESHES = {"apple": (PKG / "apple_full.obj", PKG / "apple_bitten.obj"),
          "suzanne": (PKG / "suzanne.obj", OUT / "uv_sphere.obj")}
PAYLOADS = ("radial", "xyz")
BACKENDS = ("noiseless", "phoenix", "fez")
SHOTS = (4096, 65_536)
TS = (0.0, 0.25, 0.5, 0.75, 1.0)
WIDTHS = (0.1, 0.15, 0.3)
NOISY_TS = (0.25, 0.5, 0.75)  # frames read with noise (t = 0 and 1 are the existing A and B)
STRIP_TS = (0.25, 0.5, 0.75)
DRR = "mean_std"  # Tessa's dynamic range recovery, the best of DRR_METHODS on chip frames


def load(mesh: str):
    """Mesh A and the morph Params for one experiment mesh.

    Args:
        mesh: Key of MESHES.

    Returns:
        tuple: ``(verts, faces, b_path, shared)``: A's (N, 3) vertices and faces, B's
        OBJ path and whether B shares A's topology (:func:`shares_topology`).
    """
    from tessa_ripple.mesh import load_obj

    a, b = MESHES[mesh]
    v, f = load_obj(a)
    return v, f, str(b), shares_topology(v, f, b)


def shares_topology(verts, faces, b_path) -> bool:
    """Whether B can be paired with A by vertex index: the same vertex count and faces.

    ``morph.correspond`` refuses a shared topology whose faces differ (the apples have
    equal vertex counts but a bite's worth of different faces), so those match by
    closest point.

    Args:
        verts: (N, 3) A's vertices.
        faces: A's faces.
        b_path: B's OBJ path.

    Returns:
        bool: True if B has A's vertex count and faces.
    """
    from tessa_ripple.mesh import load_obj

    vb, fb = load_obj(b_path)
    return len(vb) == len(verts) and [list(x) for x in fb] == [list(x) for x in faces]


def cost(e, alpha) -> dict:
    """Two-qubit gates and depth of one sweep frame, summed over the runs.

    Args:
        e: A morph encoding.
        alpha: The frame's angle (scalar or (runs, groups)).

    Returns:
        dict: ``twoq`` and ``depth`` totals over ``e.runs`` circuits.
    """
    from tessa_ripple.bridge import sweep_circuit, two_qubit

    qcs = [sweep_circuit(e, alpha, k)[0] for k in range(e.runs)]
    return {"twoq": sum(two_qubit(q) for q in qcs), "depth": sum(q.depth() for q in qcs)}


def experiment(job):
    """Every measurement of one (mesh, payload, backend) encoding.

    Args:
        job: ``(mesh, payload, backend, shots_list)``.

    Returns:
        dict: ``rows`` (list of CSV dicts), ``alpha`` {(width, t): (N,) per-vertex front
        angle} for the chip comparison, and ``strips`` {t: (front positions noisy at the
        largest shots, per-vertex alpha)} for apple radial on a chip.
    """
    from tessa_ripple.bridge import (Params, decode_raw, encode_mesh, front_distances, front_reference,
                                     front_smooth_alpha, morph_alpha, morph_expected, morph_frame,
                                     per_vertex, tilted_angles, _per_value)

    mesh, payload, backend, shots_list = job
    v, f, b_path, shared = load(mesh)
    a_dirs = payload == "radial"  # radial: A's directions only, as the plan asks
    p = Params(mode="morph", payload=payload, target=b_path, shared_topology=shared, backend=backend)
    t0 = time.time()
    e = encode_mesh(v, f, p)
    enc_sec = time.time() - t0
    source, d_vertex, d_qubit = front_distances(e)
    # what the morph actually moves each vertex (radial with A's directions: |r_B - r_A|)
    move = np.linalg.norm(morph_expected(e, 1.0, a_dirs) - morph_expected(e, 0.0, a_dirs), axis=1)
    cells = np.unique(np.column_stack([e.batches.repeat(e.slots.shape[1]), e.groups_of.ravel()]), axis=0)
    patch = len(v) / len(cells)  # vertices per used (batch, data qubit)
    rows, alphas, strips = [], {}, {}
    base = dict(mesh=mesh, payload=payload, backend=backend, graph=e.name, runs=e.runs,
                verts=len(v), source=source, cells=len(cells), patch_size=round(patch, 2),
                mean_move=float(move.mean()), encode_sec=round(enc_sec, 2))
    scalar_cost = cost(e, 0.3)
    expected = {}  # noiseless positions per (width or None, t)
    for w in WIDTHS:
        fr = (w, None)
        for t in TS:
            alpha = morph_alpha(e, t, fr)
            per = _per_value(e, alpha)
            alphas[(w, t)] = per_vertex(e, per)
            front = morph_expected(e, t, a_dirs, fr)
            smooth = front_reference(e, t, fr, a_dirs)
            scalar = morph_expected(e, t, a_dirs)
            expected[(w, t)], expected[(None, t)] = front, scalar
            gap = np.linalg.norm(front - smooth, axis=1)
            contrast = np.linalg.norm(front - scalar, axis=1)
            smooth_per = front_smooth_alpha(e, t, fr)
            moving = move > 0.1 * move.max()  # vertices B actually moves
            rows.append(dict(base, shots="", width=w, t=t,
                             steps=len(np.unique(np.round(per, 9))),
                             steps_smooth=len(np.unique(np.round(smooth_per, 9))),
                             gap=float(gap.mean()), gap_moving=float(gap[moving].mean()),
                             gap_rel=float(gap.mean() / move.mean()),
                             front_vs_scalar=float(contrast.mean()),
                             end_error_A=float(np.abs(front - v).max()) if t == 0 else "",
                             # B's end as the uniform sweep reads it (radial keeps A's directions)
                             end_error_B=float(np.abs(front - scalar).max()) if t == 1 else "",
                             **{f"{k}_front": c for k, c in cost(e, alpha).items()},
                             **{f"{k}_scalar": c for k, c in scalar_cost.items()}))
    for shots in shots_list:
        q = replace(p, shots=shots)
        z = tilted_angles(e, q, q.seed, 0.0)  # one Z reference, shared as run_frames does
        for k, t in enumerate(NOISY_TS):
            seed = 1 + k
            ts = time.time()
            raw_s = decode_raw(e, replace(q, morph_t=t), seed, z)
            sec_scalar = time.time() - ts
            got_s = morph_frame(e, raw_s, t, a_dirs_only=a_dirs)
            noise_s = np.sqrt((np.linalg.norm(got_s - expected[(None, t)], axis=1) ** 2).mean())
            for w in WIDTHS if t == 0.5 else (0.15,):
                fr = (w, None)
                qf = replace(q, morph_t=t, morph_front=True, front_width=w)
                ts = time.time()
                raw = decode_raw(e, qf, seed, z)
                sec = time.time() - ts
                got = morph_frame(e, raw, t, a_dirs_only=a_dirs, front=fr)
                want = expected[(w, t)]
                err = np.linalg.norm(got - want, axis=1)
                noise = np.sqrt((err ** 2).mean())
                per = _per_value(e, morph_alpha(e, t, fr)).reshape(e.slots.shape)
                edge = ((per > 0) & (per < np.pi / 2)).any(1)
                disp = np.linalg.norm(want - v, axis=1)
                contrast = np.linalg.norm(want - expected[(None, t)], axis=1)
                reached = (per > 0).any(1) & (move > 0.1 * move.max())  # front passed, B moves it
                # the same readout through Tessa's DRR (restore_field, A's statistics)
                got_d = morph_frame(e, raw, t, DRR, a_dirs_only=a_dirs, front=fr)
                noise_d = np.sqrt((np.linalg.norm(got_d - want, axis=1) ** 2).mean())
                rows.append(dict(base, shots=shots, width=w, t=t, noise_rms=float(noise),
                                 noise_rms_scalar=float(noise_s),
                                 noise_reached=float(np.sqrt((err[(per > 0).any(1)] ** 2).mean()))
                                 if (per > 0).any() else "",
                                 noise_unreached=float(np.sqrt((err[(per == 0).all(1)] ** 2).mean()))
                                 if (per == 0).all(1).any() else "",
                                 edge_vertices=int(edge.sum()),
                                 edge_disp=float(disp[edge].mean()) if edge.any() else "",
                                 edge_visibility=float(disp[edge].mean() / noise) if edge.any() else "",
                                 reached_vertices=int(reached.sum()),
                                 reached_visibility=float(disp[reached].mean() / noise) if reached.any() else "",
                                 front_vs_scalar_snr=float(contrast.mean() / noise),
                                 noise_rms_drr=float(noise_d),
                                 edge_visibility_drr=float(disp[edge].mean() / noise_d) if edge.any() else "",
                                 reached_visibility_drr=float(disp[reached].mean() / noise_d)
                                 if reached.any() else "",
                                 sec_frame=round(sec, 2), sec_frame_scalar=round(sec_scalar, 2)))
                if mesh == "apple" and payload == "radial" and w == 0.15 and shots == max(shots_list):
                    strips[t] = (got, per_vertex(e, per.ravel()))
    return {"key": (mesh, payload, backend), "rows": rows, "alpha": alphas, "strips": strips,
            "faces": f, "verts": v, "source": source, "d_vertex": d_vertex, "d_qubit_used": len(np.unique(d_qubit))}


def write_uv_sphere():
    """B for Suzanne: a UV sphere, written with ``mesh.write_obj``.

    Returns:
        Path: The OBJ path.
    """
    from tessa_ripple.mesh import uv_sphere, write_obj

    return write_obj(MESHES["suzanne"][1], *uv_sphere(24, 16))  # makes its folder


def draw_strip(results, path):
    """Apple radial at t = 0.25, 0.5, 0.75: Phoenix and Fez (noisy, 65,536 shots) and
    the smooth reference, coloured by each vertex's sweep angle.

    Args:
        results: ``experiment`` outputs keyed by (mesh, payload, backend).
        path: PNG to write.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from tessa_ripple.bridge import Params, encode_mesh, front_reference, front_smooth_alpha

    rows = [("phoenix", "ibm_phoenix · per-qubit front (noisy, 65,536 shots)"),
            ("fez", "ibm_fez · per-qubit front (noisy, 65,536 shots)"),
            (None, "smooth reference · per-vertex front (no circuit can do this)")]
    v, f, b_path, shared = load("apple")
    e = encode_mesh(v, f, Params(mode="morph", payload="radial", target=b_path, shared_topology=shared))
    cmap = colormaps["viridis"]
    src = results[("apple", "radial", "phoenix")]["source"]
    look = v[src] - v.mean(0)
    azim = np.degrees(np.arctan2(look[1], look[0]))
    elev = np.degrees(np.arcsin(look[2] / np.linalg.norm(look)))
    fig = plt.figure(figsize=(12, 12))
    for r, (backend, title) in enumerate(rows):
        for c, t in enumerate(STRIP_TS):
            ax = fig.add_subplot(3, 3, r * 3 + c + 1, projection="3d")
            if backend is None:
                pos = front_reference(e, t, (0.15, None), True)
                a = per_vertex(e, front_smooth_alpha(e, t, (0.15, None)))
            else:
                pos, a = results[("apple", "radial", backend)]["strips"][t]
            face_a = np.array([a[q].mean() for q in f]) / (np.pi / 2)
            ax.add_collection3d(Poly3DCollection([pos[q] for q in f], facecolor=cmap(face_a),
                                                 edgecolor="#20304a", lw=0.2))
            c0, half = v[src], 0.3 * np.ptp(v, axis=0).max()  # zoomed on the bite
            ax.set(xlim=(c0[0] - half, c0[0] + half), ylim=(c0[1] - half, c0[1] + half),
                   zlim=(c0[2] - half, c0[2] + half), box_aspect=(1, 1, 1))
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            ax.set_title(f"t = {t}" + (f"\n{title}" if c == 1 else ""), fontsize=9)
    fig.suptitle("Morph front, apple → bitten apple (radial, A's directions, w = 0.15), zoomed on the source.\n"
                 "Colour: each vertex's sweep angle α (dark = A, yellow = B)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(quick: bool = False):
    """Run every experiment and write the CSVs, summary and strips.

    Args:
        quick: Only the noiseless backend at 4,096 shots (for a smoke test).
    """
    write_uv_sphere()
    OUT.mkdir(parents=True, exist_ok=True)
    backends = ("noiseless",) if quick else BACKENDS
    shots = SHOTS[:1] if quick else SHOTS
    jobs = [(m, pl, b, shots) for m in MESHES for pl in PAYLOADS for b in backends]
    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for res in pool.map(experiment, jobs):
            results[res["key"]] = res
            print(f"{res['key']} done at {time.time() - t0:.0f}s", flush=True)
    rows = [r for res in results.values() for r in res["rows"]]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / "front.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fields)
        wr.writeheader()
        wr.writerows(rows)
    chips = []
    if not quick:
        for m in MESHES:
            for pl in PAYLOADS:
                a, b = (results[(m, pl, d)]["alpha"] for d in ("phoenix", "fez"))
                for (w, t), x in a.items():
                    chips.append(dict(mesh=m, payload=pl, width=w, t=t,
                                      share_differs=float((np.abs(x - b[(w, t)]) > 0.1).mean())))
        with open(OUT / "chips.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, list(chips[0]))
            wr.writeheader()
            wr.writerows(chips)
        draw_strip(results, OUT / "strip_apple_radial.png")
    (OUT / "summary.json").write_text(json.dumps({"seconds": round(time.time() - t0), "rows": len(rows),
                                                  "chips": chips}, indent=1))
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    if "--quick" in sys.argv:  # a smoke test never overwrites the full run's results
        OUT = OUT / "quick"
    main("--quick" in sys.argv)
