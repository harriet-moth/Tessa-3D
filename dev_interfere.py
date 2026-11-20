"""Interference morph prototype (tessa_ripple/interfere.py): run the experiments headlessly.

    .venv/bin/python dev_interfere.py            # everything (about half an hour)
    .venv/bin/python dev_interfere.py --quick    # noiseless only

Questions:
1. Noiseless, how far is the coherent frame from the classical mixture (Tessa's
   cross-fade, which the sweep morph also draws), and is it more than a retiming,
   i.e. can some other t' of the cross-fade reproduce it?
2. On the chips' noise models, does the coherence survive? λ = 1 is the noiseless
   frame, 0 a dephased selector; the control dephases the selectors on purpose.
3. Cost against the sweep morph: two-qubit gates, runs, capacity, shots kept.

Writes ``out/interfere/noiseless.csv``, ``chips.csv``, ``summary.json``,
``progress.png`` (mean progress towards B against t) and ``strip.png``.
"""

import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out" / "interfere"
PKG = ROOT / "tessa_ripple"
MESHES = {"apple": (PKG / "apple_full.obj", PKG / "apple_bitten.obj"),
          "suzanne": (PKG / "suzanne.obj", ROOT / "out" / "front" / "uv_sphere.obj")}
TS = np.round(np.linspace(0, 1, 11), 2)
CHIP_TS = (0.25, 0.75)  # where coherent and mixed differ most
CHIPS = ("phoenix", "fez")
SEEDS = (0, 1, 2)
SHOTS = 4096


def setup(mesh: str, payload: str = "radial", backend: str = "noiseless"):
    """Encode one experiment.

    Args:
        mesh: Key of MESHES.
        payload: "radial" or "xyz".
        backend: Params.backend.

    Returns:
        tuple: ``(ie, p)``: the interference encoding and its Params.
    """
    from tessa_ripple import interfere as I
    from tessa_ripple.bridge import Params
    from tessa_ripple.mesh import load_obj

    a, b = MESHES[mesh]
    v, f = load_obj(a)
    shared = len(load_obj(b)[0]) == len(v)
    p = Params(mode="morph", payload=payload, target=str(b), shared_topology=shared,
               backend=backend, shots=SHOTS)
    return I.encode_interfere(v, f, p), p


def moving(ie) -> np.ndarray:
    """Values whose B differs from A by more than a tenth of the largest change.

    Args:
        ie: The encoding.

    Returns:
        np.ndarray: (N*comps,) bool mask.
    """
    d = np.abs(ie.b01 - ie.a01)
    return d > 0.1 * d.max()


def progress(ie, x, mask) -> float:
    """Mean share of the way from A to B over the masked values.

    Args:
        ie: The encoding.
        x: (N*comps,) frame values.
        mask: Values to average.

    Returns:
        float: 0 at A, 1 at B.
    """
    lo, hi = ie.e.enc.ranges[0]
    a, b = lo + (hi - lo) * ie.a01, lo + (hi - lo) * ie.b01
    return float(((x - a) / (b - a))[mask].mean())


def noiseless(job):
    """Signature metrics of one mesh and payload, noiseless and exact.

    Args:
        job: ``(mesh, payload)``.

    Returns:
        list[dict]: One row per t.
    """
    from tessa_ripple import interfere as I

    mesh, payload = job
    ie, _ = setup(mesh, payload)
    m = moving(ie)
    lo, hi = ie.e.enc.ranges[0]
    disp = float(((hi - lo) * np.abs(ie.b01 - ie.a01))[m].mean())
    grid = np.linspace(0, 1, 401)
    mixes = np.array([I.incoherent(ie, s)[m] for s in grid])
    rows = []
    for t in TS:
        x, pu, mix = I.ideal(ie, t), I.pure(ie, t), I.incoherent(ie, t)
        res = np.abs(mixes - x[m]).mean(1)  # best retiming of the cross-fade
        best = int(res.argmin())
        rows.append({"mesh": mesh, "payload": payload, "t": t, "graph": ie.e.name, "values": int(m.size),
                     "moving": int(m.sum()), "disp": disp,
                     "coherent_vs_mixed": float(np.abs(x - mix)[m].mean() / disp),
                     "star_term": float(np.abs(x - pu)[m].mean() / disp),
                     "retimed_t": float(grid[best]), "retimed_residual": float(res[best] / disp),
                     "progress_coherent": progress(ie, x, m), "progress_pure": progress(ie, pu, m),
                     "progress_mixed": progress(ie, mix, m)})
    return rows


def chip(job):
    """One noisy frame and its dephased control on a chip.

    Args:
        job: ``(mesh, backend, t, seed)``.

    Returns:
        dict: λ, the control's λ, error, shots kept, timing.
    """
    from tessa_ripple import interfere as I

    mesh, backend, t, seed = job
    ie, p = setup(mesh, "radial", backend)
    m = moving(ie)
    x, mix = I.ideal(ie, t), I.incoherent(ie, t)
    lo, hi = ie.e.enc.ranges[0]
    disp = float(((hi - lo) * np.abs(ie.b01 - ie.a01))[m].mean())
    t0 = time.time()
    raw, kept = I.read(ie, p, t, seed=seed)
    secs = time.time() - t0
    ctrl, _ = I.read(ie, p, t, seed=seed + 7, dephase=True)
    return {"mesh": mesh, "backend": backend, "t": t, "seed": seed, "shots": SHOTS,
            "lambda": I.coherence(raw[m], x[m], mix[m]), "lambda_control": I.coherence(ctrl[m], x[m], mix[m]),
            "error": float(np.abs(raw - x)[m].mean() / disp),
            "signature": float(np.abs(x - mix)[m].mean() / disp),
            "progress": progress(ie, raw, m), "progress_control": progress(ie, ctrl, m),
            "kept": kept, "seconds_per_frame": secs, "raw": raw.tolist()}


def cost_row(job):
    """Resources of the interference morph against the sweep morph on one graph.

    Args:
        job: ``(mesh, backend)``.

    Returns:
        dict: Two-qubit gates, depth, runs, capacity of both.
    """
    from tessa_ripple import interfere as I
    from tessa_ripple.bridge import encode_mesh, two_qubit

    mesh, backend = job
    ie, p = setup(mesh, "radial", backend)
    sweep = encode_mesh(ie.e.verts, ie.e.faces, p, target=ie.e.match)
    sq = [c for c, _ in (sweep.parts or ((sweep.circuit, sweep.enc),))]
    c = I.cost(ie)
    return {"mesh": mesh, "backend": backend, "graph": ie.e.name,
            "interfere_twoq": c["twoq"], "interfere_depth": c["depth"], "interfere_runs": c["runs"],
            "interfere_capacity": c["capacity"], "stars": c["stars"],
            "sweep_twoq": sum(two_qubit(q) for q in sq), "sweep_depth": sum(q.depth() for q in sq),
            "sweep_runs": sweep.runs, "sweep_capacity": sweep.graph.capacity,
            "sweep_frame_circuits": 2 * sweep.runs, "interfere_frame_circuits": c["runs"]}


def draw_progress(rows, chips, path):
    """Mean progress towards B against t: mixture, coherent, pure, and the chip reads.

    Args:
        rows: Noiseless rows.
        chips: Chip rows.
        path: PNG to write.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, mesh in zip(axes, MESHES):
        r = [x for x in rows if x["mesh"] == mesh and x["payload"] == "radial"]
        t = [x["t"] for x in r]
        ax.plot(t, [x["progress_mixed"] for x in r], color="#888", lw=2,
                label="mixture = Tessa cross-fade (classical)")
        ax.plot(t, [x["progress_pure"] for x in r], color="#7a5", ls="--", label="lone qubit superposition")
        ax.plot(t, [x["progress_coherent"] for x in r], color="#25c", lw=2, label="coherent star (exact)")
        for b, mk in zip(CHIPS, "os"):
            for ctrl, col in ((False, "#d40"), (True, "#999")):
                pts = [x for x in chips if x["mesh"] == mesh and x["backend"] == b]
                for tt in CHIP_TS:
                    v = [x["progress_control" if ctrl else "progress"] for x in pts if x["t"] == tt]
                    if v:
                        ax.errorbar(tt + (0.015 if b == "fez" else -0.015), np.mean(v), yerr=np.std(v), fmt=mk,
                                    color=col, ms=5, capsize=2,
                                    label=f"ibm_{b}" + (" control (dephased)" if ctrl else "")
                                    if tt == CHIP_TS[0] else None)
        ax.set(title=mesh, xlabel="t", ylabel="mean progress towards B (moving vertices)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Interference morph (radial): coherent frames lead the classical mixture", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_strip(chips, path, t=0.25):
    """Apple and Suzanne at t: mixture, coherent, and one Phoenix and Fez read, coloured by progress.

    Args:
        chips: Chip rows (with ``raw``).
        path: PNG to write.
        t: The frame.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from tessa_ripple import interfere as I

    cmap = colormaps["viridis"]
    fig = plt.figure(figsize=(14, 7.5))
    for r, mesh in enumerate(MESHES):
        ie, _ = setup(mesh)
        lo, hi = ie.e.enc.ranges[0]
        a, b = lo + (hi - lo) * ie.a01, lo + (hi - lo) * ie.b01
        reads = {c: next(np.array(x["raw"]) for x in chips if (x["mesh"], x["backend"], x["t"], x["seed"])
                         == (mesh, c, t, 0)) for c in CHIPS}
        cols = [("mixture (classical cross-fade)", I.incoherent(ie, t)),
                ("coherent (noiseless)", I.ideal(ie, t)),
                *((f"ibm_{c} (noisy, {SHOTS} shots)", reads[c]) for c in CHIPS)]
        v, f = ie.e.verts, ie.e.faces
        src = int(np.argmax(np.abs(b - a)))
        for c, (title, x) in enumerate(cols):
            ax = fig.add_subplot(2, 4, r * 4 + c + 1, projection="3d")
            pos = I.frame(ie, x, t)
            with np.errstate(divide="ignore", invalid="ignore"):
                prog = np.clip(np.where(np.abs(b - a) > 1e-9, (x - a) / (b - a), 0), 0, 0.6) / 0.6
            face = np.array([prog[q].mean() for q in f])
            ax.add_collection3d(Poly3DCollection([pos[q] for q in f], facecolor=cmap(face),
                                                 edgecolor="#20304a", lw=0.15))
            c0 = v[src] if mesh == "apple" else v.mean(0)
            half = (0.3 if mesh == "apple" else 0.6) * np.ptp(v, axis=0).max()
            ax.set(xlim=(c0[0] - half, c0[0] + half), ylim=(c0[1] - half, c0[1] + half),
                   zlim=(c0[2] - half, c0[2] + half), box_aspect=(1, 1, 1))
            look = v[src] - v.mean(0)
            ax.view_init(elev=np.degrees(np.arcsin(look[2] / np.linalg.norm(look))),
                         azim=np.degrees(np.arctan2(look[1], look[0])))
            ax.set_axis_off()
            ax.set_title(f"{mesh} · {title}", fontsize=8)
    fig.suptitle(f"Interference morph at t = {t}. Colour: progress towards B (dark = A, yellow ≥ 60 %)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(quick: bool = False):
    """Run the experiments and write the CSVs, summary and figures.

    Args:
        quick: Noiseless only.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    if not MESHES["suzanne"][1].exists():
        from dev_front import write_uv_sphere
        write_uv_sphere()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = [r for rs in pool.map(noiseless, [(m, pl) for m in MESHES for pl in ("radial", "xyz")])
                for r in rs]
        print(f"noiseless done at {time.time() - t0:.0f}s", flush=True)
        costs = list(pool.map(cost_row, [(m, b) for m in MESHES for b in CHIPS]))
        chips = []
        if not quick:
            jobs = [(m, b, t, s) for b in ("phoenix", "fez") for m in MESHES for t in CHIP_TS for s in SEEDS]
            for res in pool.map(chip, jobs):
                chips.append(res)
                print(res["mesh"], res["backend"], res["t"], res["seed"],
                      f"λ {res['lambda']:.2f} control {res['lambda_control']:.2f} at {time.time() - t0:.0f}s",
                      flush=True)
    for name, data in (("noiseless", rows), ("costs", costs),
                       ("chips", [{k: v for k, v in r.items() if k != "raw"} for r in chips])):
        if data:
            with open(OUT / f"{name}.csv", "w", newline="") as fh:
                wr = csv.DictWriter(fh, list(data[0]))
                wr.writeheader()
                wr.writerows(data)
    summary = {"noiseless": {}, "chips": {}, "costs": costs}
    for m in MESHES:
        for pl in ("radial", "xyz"):
            r = [x for x in rows if (x["mesh"], x["payload"]) == (m, pl) and 0 < x["t"] < 1]
            summary["noiseless"][f"{m}/{pl}"] = {
                "max_coherent_vs_mixed": max(x["coherent_vs_mixed"] for x in r),
                "max_retimed_residual": max(x["retimed_residual"] for x in r),
                "max_star_term": max(x["star_term"] for x in r)}
        for b in CHIPS:
            for t in CHIP_TS:
                r = [x for x in chips if (x["mesh"], x["backend"], x["t"]) == (m, b, t)]
                if r:
                    lam, ctl = [x["lambda"] for x in r], [x["lambda_control"] for x in r]
                    summary["chips"][f"{m}/{b}/t={t}"] = {
                        "lambda": [float(np.mean(lam)), float(np.std(lam))],
                        "lambda_control": [float(np.mean(ctl)), float(np.std(ctl))],
                        "error_over_disp": float(np.mean([x["error"] for x in r])),
                        "signature_over_disp": r[0]["signature"],
                        "kept": float(np.mean([x["kept"] for x in r])),
                        "seconds_per_frame": float(np.mean([x["seconds_per_frame"] for x in r]))}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    draw_progress(rows, chips, OUT / "progress.png")
    if chips:
        draw_strip(chips, OUT / "strip.png")
    print(json.dumps(summary, indent=2))
    print(f"all done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    if "--quick" in sys.argv:  # a smoke test never overwrites the full run's results
        OUT = OUT / "quick"
    main(quick="--quick" in sys.argv)
