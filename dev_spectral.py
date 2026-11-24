"""Spectral payload (tessa_ripple/spectral.py, Params(payload="spectral")): run the experiments.

    .venv/bin/python dev_spectral.py            # everything (a few minutes on 10 cores)
    .venv/bin/python dev_spectral.py --quick    # truncation + noiseless/flat only, one seed

Questions (docs/direction-spectral-plan.md, Part B):
1. Truncation: how many modes do A, B and B's distinctive region (the bite) need?
2. θ readout: per-vertex error (over the bounding-box diagonal) against xyz and radial,
   with and without band scaling, and how smooth the error is (roughness).
3. Morph: SNR (mean displacement of the moving vertices at t = 0.5 over the frame's
   RMS noise) for the sweep and the interference readout, against xyz and radial.
4. Cost: values, runs, two-qubit gates, depth, seconds per readout.

Writes ``out/spectral/truncation.csv``, ``theta.csv``, ``morph.csv``, ``costs.csv``,
``summary.json`` (with the plan's gate) and ``strip.png``.
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
OUT = ROOT / "out" / "spectral"
PKG = ROOT / "tessa_ripple"
MESHES = {"apple": (PKG / "apple_full.obj", PKG / "apple_bitten.obj"),
          "suzanne": (PKG / "suzanne.obj", ROOT / "out" / "front" / "uv_sphere.obj")}
BACKENDS = ("noiseless", "flat noise", "phoenix", "fez")
CHIPS = ("phoenix", "fez")
SHOTS = (4096, 65536)
SEEDS = (0, 1, 2)
DRR = "mean_std"
MORPH_T = 0.5
# (label, payload, spectral_k, spectral_tol, band): "auto" is the default 1 % tolerance,
# "max" every mode one run holds (tolerance 0)
CONFIGS = [("xyz", "xyz", None, 0.01, True), ("radial", "radial", None, 0.01, True),
           ("spectral k16", "spectral", 16, 0.01, True), ("spectral k32", "spectral", 32, 0.01, True),
           ("spectral k64", "spectral", 64, 0.01, True), ("spectral auto", "spectral", None, 0.01, True),
           ("spectral max", "spectral", None, 0.0, True),
           ("spectral auto, one range", "spectral", None, 0.01, False)]
MORPH_CONFIGS = [c for c in CONFIGS if c[0] in ("xyz", "radial", "spectral auto", "spectral max")]


def load(mesh: str):
    """A's mesh and B's match for one experiment mesh (closest point: B's faces differ).

    Args:
        mesh: Key of MESHES.

    Returns:
        tuple: ``(verts, faces, b_path, match)``: A's vertices and faces, B's OBJ path
        and the :func:`morph.correspond` dict.
    """
    from tessa_ripple.mesh import load_obj
    from tessa_ripple.morph import correspond

    a, b = MESHES[mesh]
    v, f = load_obj(a)
    return v, f, str(b), correspond(v, f, str(b))


def params(config, backend: str, **kw):
    """Params of one configuration.

    Args:
        config: An entry of CONFIGS.
        backend: Params.backend.
        **kw: Further Params fields.

    Returns:
        Params: The settings.
    """
    from tessa_ripple.bridge import Params

    _, payload, k, tol, band = config
    return Params(payload=payload, backend=backend, spectral_k=k, spectral_tol=tol, spectral_band=band, **kw)


def rms(d, diag: float) -> float:
    """RMS per-vertex distance over the bounding-box diagonal.

    Args:
        d: (N, 3) displacement field.
        diag: Bounding-box diagonal.

    Returns:
        float: sqrt(mean |d|²) / diag.
    """
    return float(np.sqrt((np.asarray(d) ** 2).sum(1).mean()) / diag)


def roughness(E, faces) -> float:
    """How grainy an error field is: |L_u E| / |E - mean|, L_u the uniform graph Laplacian.

    About the vertex degree (5-6) for independent per-vertex noise; far smaller for a
    field made of low surface modes.

    Args:
        E: (N, 3) error field.
        faces: Polygons (for the edges).

    Returns:
        float: The ratio (0 for a constant field).
    """
    from tessa_ripple.mesh import edges

    E = np.asarray(E, float)
    ed = edges(faces)
    LE = np.zeros_like(E)
    np.add.at(LE, ed[:, 0], E[ed[:, 0]] - E[ed[:, 1]])
    np.add.at(LE, ed[:, 1], E[ed[:, 1]] - E[ed[:, 0]])
    den = np.linalg.norm(E - E.mean(0))
    return float(np.linalg.norm(LE) / den) if den > 0 else 0.0


def truncation(mesh: str) -> tuple[list[dict], dict]:
    """Truncation error of A, B and the bite for every k up to N - 1.

    Args:
        mesh: Key of MESHES.

    Returns:
        tuple: ``(rows, info)``: one row per k (A alone, and A + B with the constant
        mode), and the bite's vertex count and mean displacement.
    """
    from tessa_ripple import spectral as S

    v, f, _, match = load(mesh)
    b = np.asarray(match["target"], float)
    n = len(v)
    phi, _, mass = S.basis(v, f, n - 1, constant=True)
    c = S.centroid(v, mass)
    diag = S.diagonal(v)
    move = np.linalg.norm(b - v, axis=1)
    bite = move > 0.1 * move.max()
    ea = S.truncation_errors(v, c, phi, mass)
    eb = S.truncation_errors(b, c, phi, mass)
    cb = S.coefficients(b, c, phi, mass)
    resid = b - c
    rows = []
    for j in range(n):
        resid = resid - np.outer(phi[:, j], cb[j])
        if j == 0:
            continue  # k counts non-constant modes; row j holds the constant plus j modes
        rows.append({"mesh": mesh, "k": j, "error_a": float(ea[j]), "error_b": float(eb[j]),
                     "bite_error_over_disp": float(np.linalg.norm(resid[bite], axis=1).mean() / move[bite].mean())})
    info = {"verts": n, "bite_verts": int(bite.sum()), "bite_disp_over_diag": float(move[bite].mean() / diag),
            "b_centroid_shift_over_diag": float(np.linalg.norm(S.centroid(b, mass) - c) / diag)}
    return rows, info


def theta(job) -> list[dict]:
    """θ readouts of one mesh on one backend, every configuration, shots and seed.

    Args:
        job: ``(mesh, backend, seeds)``.

    Returns:
        list[dict]: One row per (configuration, shots, seed), plus a ``positions`` key
        on the apple's largest-shots seed-0 rows for the strip.
    """
    from tessa_ripple import spectral as S
    from tessa_ripple.bridge import decode_raw, encode_mesh, encoded_shape, restore

    mesh, backend, seeds = job
    v, f, _, _ = load(mesh)
    diag = S.diagonal(v)
    phi, _, mass = S.basis(v, f, len(v) - 1)
    rows = []
    for config in CONFIGS:
        p = params(config, backend)
        t0 = time.time()
        e = encode_mesh(v, f, p)
        enc_sec = time.time() - t0
        ref = encoded_shape(e)
        base = dict(mesh=mesh, backend=backend, config=config[0], graph=e.name, runs=e.runs,
                    k=None if e.spectral is None else e.spectral["k"], values=e.stats["values"],
                    twoq=e.stats["twoq"], depth=e.stats["depth"], trunc_error=rms(ref - v, diag),
                    encode_sec=round(enc_sec, 2))
        for shots in SHOTS:
            q = replace(p, shots=shots)
            for seed in seeds:
                t0 = time.time()
                raw = decode_raw(e, q, seed)
                sec = time.time() - t0
                got, got_d = restore(e, raw), restore(e, raw, DRR)
                noise = got - ref
                row = dict(base, shots=shots, seed=seed, sec=round(sec, 2),
                           error=rms(got - v, diag), noise=rms(noise, diag),
                           error_drr=rms(got_d - v, diag), noise_drr=rms(got_d - ref, diag),
                           roughness=roughness(noise, f), roughness_drr=roughness(got_d - ref, f),
                           high_share_32=S.high_share(noise, phi, mass, 32),
                           high_share_32_drr=S.high_share(got_d - ref, phi, mass, 32))
                if mesh == "apple" and shots == SHOTS[-1] and seed == seeds[0]:
                    row["positions"] = got
                rows.append(row)
    return rows


def morph(job) -> list[dict]:
    """Morph frames at MORPH_T for one mesh, backend, readout and configuration.

    Args:
        job: ``(mesh, backend, readout, config, seeds)``; readout "sweep" or "interference".

    Returns:
        list[dict]: One row per (shots, seed), plus a noiseless row (shots "") with the
        endpoint errors and the frame's mean displacement.
    """
    from tessa_ripple import interfere as I
    from tessa_ripple import spectral as S
    from tessa_ripple.bridge import (decode_raw, encode_mesh, encoded_shape, morph_expected, morph_frame,
                                     reference_seed, tilted_angles, two_qubit)

    mesh, backend, readout, config, seeds = job
    v, f, b_path, match = load(mesh)
    diag = S.diagonal(v)
    p = params(config, backend, mode="morph", target=b_path)
    t0 = time.time()
    if readout == "sweep":
        e = encode_mesh(v, f, p, match)
        expect = lambda t: morph_expected(e, t)  # noqa: E731
    else:
        ie = I.encode_interfere(v, f, p, match)
        e = ie.e
        expect = lambda t: I.frame(ie, I.ideal(ie, t), t)  # noqa: E731
    enc_sec = time.time() - t0
    a_ref, b_ref = encoded_shape(e), encoded_shape(e, "b")
    x0, x1, xt = expect(0.0), expect(1.0), expect(MORPH_T)
    move = np.linalg.norm(b_ref - a_ref, axis=1)
    moving = move > 0.1 * move.max()
    disp = float(np.linalg.norm(xt - x0, axis=1)[moving].mean())
    if readout == "sweep":
        circuits = 2 * e.runs  # the tilted frame and its α = 0 reference
    else:
        circuits = sum(len(I._clusters(ie, k)) for k in range(e.runs))
    base = dict(mesh=mesh, backend=backend, readout=readout, config=config[0], graph=e.name, runs=e.runs,
                k=None if e.spectral is None else e.spectral["k"], values=e.stats["values"],
                twoq=sum(two_qubit(c) for c, _ in (e.parts or ((e.circuit, e.enc),))),
                circuits_per_frame=circuits, disp_over_diag=disp / diag,
                trunc_a=rms(a_ref - v, diag), trunc_b=rms(b_ref - np.asarray(match["target"]), diag),
                encode_sec=round(enc_sec, 2))
    rows = [dict(base, shots="", seed="", end_error_a=float(np.abs(x0 - a_ref).max()),
                 end_error_b=float(np.abs(x1 - b_ref).max()))]
    for shots in SHOTS:
        q = replace(p, shots=shots, morph_t=MORPH_T)
        for seed in seeds:
            t0 = time.time()
            kept = ""
            if readout == "sweep":
                ref = tilted_angles(e, q, reference_seed(seed), 0.0)
                raw = decode_raw(e, q, seed, ref)
                got, got_d = morph_frame(e, raw, MORPH_T), morph_frame(e, raw, MORPH_T, DRR)
            else:
                raw, kept = I.read(ie, q, MORPH_T, seed)
                got, got_d = I.frame(ie, raw, MORPH_T), I.frame(ie, raw, MORPH_T, DRR)
            sec = time.time() - t0
            noise = float(np.sqrt((np.linalg.norm(got - xt, axis=1) ** 2).mean()))
            noise_d = float(np.sqrt((np.linalg.norm(got_d - xt, axis=1) ** 2).mean()))
            rows.append(dict(base, shots=shots, seed=seed, sec=round(sec, 2), kept=kept,
                             noise_over_diag=noise / diag, snr=disp / noise,
                             noise_drr_over_diag=noise_d / diag, snr_drr=disp / noise_d,
                             roughness=roughness(got - xt, f), roughness_drr=roughness(got_d - xt, f)))
    return rows


def mean_of(rows, key, **match) -> float:
    """Mean of ``key`` over the rows matching every ``match`` field.

    Args:
        rows: Dict rows.
        key: Field to average.
        **match: Field values the rows must have.

    Returns:
        float: The mean (nan if none match).
    """
    vals = [r[key] for r in rows if all(r.get(k) == x for k, x in match.items()) and r.get(key) not in ("", None)]
    return float(np.mean(vals)) if vals else float("nan")


def gate(trunc, theta_rows, morph_rows) -> dict:
    """The plan's stop condition, per mesh, chip and spectral configuration.

    Build UI only if on Phoenix and Fez at some k: the bite stays visible (bite-region
    truncation error < 25 % of its displacement, at the morph's k), per-vertex error ≤
    xyz's, and roughness at least 2× lower than xyz's (θ readout, largest shots).

    Args:
        trunc: Truncation rows.
        theta_rows: θ rows.
        morph_rows: Morph rows.

    Returns:
        dict: ``{mesh: {chip: {config: checks}}, "pass": {mesh: {"raw": bool, "drr": bool}}}``.
    """
    out, passed = {}, {}
    shots = SHOTS[-1]
    for mesh in MESHES:
        bite = {r["k"]: r["bite_error_over_disp"] for r in trunc if r["mesh"] == mesh}
        out[mesh], ok_chips = {}, []
        for chip in CHIPS:
            xyz_err = mean_of(theta_rows, "error", mesh=mesh, backend=chip, config="xyz", shots=shots)
            xyz_rough = mean_of(theta_rows, "roughness", mesh=mesh, backend=chip, config="xyz", shots=shots)
            xyz_err_d = mean_of(theta_rows, "error_drr", mesh=mesh, backend=chip, config="xyz", shots=shots)
            xyz_rough_d = mean_of(theta_rows, "roughness_drr", mesh=mesh, backend=chip, config="xyz", shots=shots)
            out[mesh][chip] = {}
            for config in [c[0] for c in CONFIGS if c[1] == "spectral" and c[4]]:
                err = mean_of(theta_rows, "error", mesh=mesh, backend=chip, config=config, shots=shots)
                rough = mean_of(theta_rows, "roughness", mesh=mesh, backend=chip, config=config, shots=shots)
                ks = {r["readout"]: r["k"] for r in morph_rows
                      if (r["mesh"], r["backend"], r["config"], r["shots"]) == (mesh, chip, config, "")}
                kt = next((r["k"] for r in theta_rows if (r["mesh"], r["backend"], r["config"]) == (mesh, chip, config)), None)
                k_sweep = ks.get("sweep", kt)
                checks = {"k_theta": kt, "k_sweep": k_sweep, "k_interference": ks.get("interference"),
                          "bite_error_over_disp": bite.get(k_sweep), "error": err, "xyz_error": xyz_err,
                          "roughness": rough, "xyz_roughness": xyz_rough}
                checks["bite_ok"] = bool(checks["bite_error_over_disp"] is not None and checks["bite_error_over_disp"] < 0.25)
                checks["error_ok"] = bool(err <= xyz_err)
                checks["roughness_ok"] = bool(rough * 2 <= xyz_rough)
                checks["pass"] = checks["bite_ok"] and checks["error_ok"] and checks["roughness_ok"]
                # the same checks after Tessa's DRR (mean_std) on both payloads, which removes
                # the shared shrink towards the centre and leaves the noise's own shape
                err_d = mean_of(theta_rows, "error_drr", mesh=mesh, backend=chip, config=config, shots=shots)
                rough_d = mean_of(theta_rows, "roughness_drr", mesh=mesh, backend=chip, config=config, shots=shots)
                checks.update(error_drr=err_d, xyz_error_drr=xyz_err_d, roughness_drr=rough_d,
                              xyz_roughness_drr=xyz_rough_d)
                checks["pass_drr"] = bool(checks["bite_ok"] and err_d <= xyz_err_d and rough_d * 2 <= xyz_rough_d)
                out[mesh][chip][config] = checks
            ok_chips.append([any(c[key] for c in out[mesh][chip].values()) for key in ("pass", "pass_drr")])
        passed[mesh] = {"raw": all(o[0] for o in ok_chips), "drr": all(o[1] for o in ok_chips)}
    out["pass"] = passed
    return out


def draw_strip(theta_rows, faces, path):
    """Apple on each chip: truncated A, xyz readout, spectral readout (largest shots, seed 0).

    Args:
        theta_rows: θ rows (those with ``positions``).
        faces: The apple's polygons.
        path: PNG path.

    Returns:
        None. Writes the image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tessa_ripple.morph import triangles

    tri = triangles(faces)
    picks = [(chip, cfg) for chip in CHIPS for cfg in ("xyz", "spectral auto", "spectral max")]
    found = [(c, g, next((r for r in theta_rows if r.get("positions") is not None
                          and (r["backend"], r["config"]) == (c, g)), None)) for c, g in picks]
    found = [x for x in found if x[2] is not None]
    if not found:
        return
    fig = plt.figure(figsize=(3 * len(found), 3.2))
    for i, (chip, cfg, r) in enumerate(found):
        ax = fig.add_subplot(1, len(found), i + 1, projection="3d")
        P = r["positions"]
        ax.plot_trisurf(P[:, 0], P[:, 1], P[:, 2], triangles=tri, color="#d9d4cc", edgecolor="#555", linewidth=0.15)
        ax.set_title(f"{chip}: {cfg}\nerror {r['error']:.3f}, rough {r['roughness']:.1f}", fontsize=8)
        ax.set_axis_off()
        ax.view_init(15, -60)
    fig.tight_layout(rect=(0, 0, 1, 0.9))  # room for the two-line titles
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_csv(name, rows):
    """Write rows to ``OUT/name.csv`` (array-valued fields dropped).

    Args:
        name: File stem.
        rows: Dict rows.

    Returns:
        None.
    """
    rows = [{k: v for k, v in r.items() if k != "positions"} for r in rows]
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / f"{name}.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, keys)
        wr.writeheader()
        wr.writerows(rows)


def main(quick: bool = False):
    """Run every experiment and write the results.

    Args:
        quick: Truncation and the noiseless and flat-noise θ readouts only, one seed.

    Returns:
        None.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    trunc, info = [], {}
    for mesh in MESHES:
        rows, info[mesh] = truncation(mesh)
        trunc += rows
    write_csv("truncation", trunc)
    print(f"truncation done at {time.time() - t0:.0f}s", flush=True)
    seeds = SEEDS[:1] if quick else SEEDS
    backends = BACKENDS[:2] if quick else BACKENDS
    with ProcessPoolExecutor(max_workers=8) as pool:
        theta_rows = []
        for rows in pool.map(theta, [(m, b, seeds) for m in MESHES for b in backends]):
            theta_rows += rows
            print(rows[0]["mesh"], rows[0]["backend"], f"θ done at {time.time() - t0:.0f}s", flush=True)
        write_csv("theta", theta_rows)
        morph_rows = []
        if not quick:
            jobs = [(m, b, ro, c, seeds) for m in MESHES for b in ("noiseless", *CHIPS)
                    for ro in ("sweep", "interference") for c in MORPH_CONFIGS]
            for rows in pool.map(morph, jobs):
                morph_rows += rows
                r = rows[0]
                print(r["mesh"], r["backend"], r["readout"], r["config"],
                      f"snr {mean_of(rows, 'snr', shots=SHOTS[-1]):.2f} at {time.time() - t0:.0f}s", flush=True)
            write_csv("morph", morph_rows)
    costs = [{k: r[k] for k in ("mesh", "backend", "config", "graph", "k", "values", "runs", "twoq", "depth")}
             for r in theta_rows if r["shots"] == SHOTS[0] and r["seed"] == seeds[0]]
    write_csv("costs", costs)
    summary = {"truncation": info, "theta": {}, "morph": {}}
    for mesh in MESHES:
        for b in backends:
            for c in CONFIGS:
                for s in SHOTS:
                    m = dict(mesh=mesh, backend=b, config=c[0], shots=s)
                    summary["theta"][f"{mesh}/{b}/{c[0]}/{s}"] = {
                        k: mean_of(theta_rows, k, **m) for k in ("k", "error", "noise", "error_drr", "noise_drr", "roughness",
                                                                 "roughness_drr", "high_share_32", "high_share_32_drr",
                                                                 "sec", "twoq", "runs")}
        for b in ("noiseless", *CHIPS):
            for ro in ("sweep", "interference"):
                for c in MORPH_CONFIGS:
                    m = dict(mesh=mesh, backend=b, readout=ro, config=c[0])
                    end = [r for r in morph_rows if all(r[k] == x for k, x in m.items()) and r["shots"] == ""]
                    if end:
                        summary["morph"][f"{mesh}/{b}/{ro}/{c[0]}"] = {
                            "k": end[0]["k"], "end_error": max(end[0]["end_error_a"], end[0]["end_error_b"]),
                            "disp_over_diag": end[0]["disp_over_diag"], "circuits_per_frame": end[0]["circuits_per_frame"],
                            **{f"{k}_{s}": mean_of(morph_rows, k, shots=s, **m)
                               for k in ("snr", "snr_drr", "noise_over_diag", "noise_drr_over_diag", "roughness",
                                         "roughness_drr", "sec", "kept") for s in SHOTS}}
    if not quick:
        summary["gate"] = gate(trunc, theta_rows, morph_rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    if not quick:
        draw_strip(theta_rows, load("apple")[1], OUT / "strip.png")
    print(json.dumps(summary.get("gate", {}), indent=2, default=float))
    print(f"all done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    if "--quick" in sys.argv:  # a smoke test never overwrites the full run's results
        OUT = OUT / "quick"
    main(quick="--quick" in sys.argv)
