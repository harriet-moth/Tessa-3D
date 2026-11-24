"""Read a mesh on a real IBM chip with Tessa's whole-chip path, or plan it offline.

The simulated readouts (desktop app, web, Blender) cut the chip into cells
(``tessa.decode(per_group=True)``); real hardware runs the whole encoded circuit at
once, in each basis its mode reads (Tessa's ``measurement_circuits`` and
``decode_from_counts``: :func:`tessa_ripple.bridge.read_hardware`). Every payload,
mode θ / θ+φ, readout (Z, Bloch, snap) and DRR (pilots included) works the same way.

Command line (``python -m tessa_ripple.hardware <command> ...``)::

    plan   MESH [settings]               transpile onto the chip's stored calibration
                                         snapshot and report the job; contacts nobody
    run    MESH [settings] --out DIR     submit one job, wait, decode, save everything
    run    MESH [settings] --out DIR --simulate
                                         the same whole-chip circuits on Aer (lattice
                                         backends small enough to simulate whole)
    decode DIR [--drr METHOD]            decode DIR's saved counts again, offline

``run`` submits only when called. It needs an IBM Quantum account: the token comes
from ``$QISKIT_IBM_TOKEN`` or the account saved with
``QiskitRuntimeService.save_account``, the instance from ``$QISKIT_IBM_INSTANCE``
(Tessa's ``ibm_sampler``). No credential is ever written to DIR, which holds ``settings.json`` (mesh and Params), ``job.json`` (job id and
backend), ``counts.json``, ``report.json`` (transpile report, error) and
``decoded.obj``.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

from . import bridge
from .mesh import load_obj, write_obj

SETTINGS = ("payload", "mode", "layout", "readout", "placement", "compression", "headroom", "clamp", "clamp_b")


def params_of(args) -> bridge.Params:
    """The encoder settings a command line names.

    Args:
        args: Parsed arguments (``chip``/``backend``, ``shots``, ``drr`` and SETTINGS).

    Returns:
        bridge.Params: Pilots on when ``drr`` is "pilots".
    """
    kw = {k: getattr(args, k) for k in SETTINGS if getattr(args, k, None) is not None}
    backend = args.backend if getattr(args, "simulate", False) else args.chip
    return bridge.Params(**kw, backend=backend, shots=args.shots, pilots=args.drr == "pilots")


def encode(mesh: str, p: bridge.Params) -> bridge.Encoded:
    """Load an OBJ and encode it.

    Args:
        mesh: OBJ path.
        p: Settings.

    Returns:
        bridge.Encoded: The encoding.
    """
    v, f = load_obj(mesh)
    return bridge.encode_mesh(v, f, p)


def plan(e: bridge.Encoded, p: bridge.Params) -> dict:
    """What a hardware run of ``e`` would submit, transpiled onto the chip's snapshot.

    Args:
        e: A chip encoding.
        p: Its settings.

    Returns:
        dict: ``backend``, ``circuits``, ``shots``, ``qubits`` (used), ``runs`` and
        Tessa's transpile report per circuit.
    """
    snapshot = bridge.devices.snapshot_backend(p.backend)
    jobs = bridge.hardware_circuits(e)
    reports = [bridge.transpile_to_backend(qc, snapshot)[1] for _, _, qc in jobs]
    return {"backend": bridge.snapshot_json(p.backend)["name"], "circuits": len(jobs), "shots": p.shots,
            "qubits": len(e.used_qubits), "runs": e.runs, "values": e.stats["values"],
            "bases": sorted({b for _, b, _ in jobs}), "reports": reports}


def _write(path: Path, data) -> None:
    """Write JSON (numpy numbers as Python ones).

    Args:
        path: File.
        data: JSON-able data.
    """
    path.write_text(json.dumps(data, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))


def finish(out: Path, e: bridge.Encoded, raw, drr: str, extra: dict | None = None) -> dict:
    """Restore a readout, write ``decoded.obj`` and ``report.json``.

    Args:
        out: Output folder.
        e: The encoding.
        raw: (N*comps,) raw values.
        drr: DRR method (one of ``bridge.DRR_CHOICES``).
        extra: More report fields.

    Returns:
        dict: The report.
    """
    pos = bridge.restore(e, raw, drr)
    write_obj(out / "decoded.obj", pos, e.faces)
    report = {"error": bridge.mesh_error(bridge.encoded_shape(e), pos), "drr": drr, **(extra or {})}
    _write(out / "report.json", report)
    return report


def run(args) -> dict:
    """Encode, run once (IBM, or Aer with ``--simulate``), decode and save.

    Args:
        args: Parsed ``run`` arguments.

    Returns:
        dict: The report.

    Raises:
        SystemExit: Missing ``qiskit-ibm-runtime``, or a simulation too big to run whole.
    """
    p = params_of(args)
    e = encode(args.mesh, p)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "settings.json", {"mesh": str(Path(args.mesh).resolve()), "drr": args.drr,
                                   "params": dataclasses.asdict(p)})
    if args.simulate:
        if e.graph.n_qubits > 26:
            sys.exit(f"{e.graph.n_qubits} qubits is too many to simulate whole; plan it for a chip instead")
        sampler, device, name = bridge.sampler(e.graph, p, p.seed), None, f"aer ({p.backend})"
    else:
        from tessa.backends.ibm import ibm_sampler, make_service
        name = bridge.snapshot_json(p.backend)["name"]
        device = make_service().backend(name)  # the real backend: transpile for it
        sampler = ibm_sampler(backend=device)
    _write(out / "job.json", {"backend": name, "shots": p.shots})
    raw, counts, reports = bridge.read_hardware(e, sampler, p.shots, device=device)
    _write(out / "counts.json", counts)
    return finish(out, e, raw, args.drr, {"backend": name, "shots": p.shots, "reports": reports})


def decode(args) -> dict:
    """Decode a run folder's saved counts again (offline), optionally with another DRR.

    Args:
        args: Parsed ``decode`` arguments (``folder``, ``drr``).

    Returns:
        dict: The report.

    Raises:
        ValueError: ``--drr pilots`` on a run encoded without pilots, or the reverse.
    """
    out = Path(args.folder)
    saved = json.loads((out / "settings.json").read_text())
    p = bridge.Params(**saved["params"])
    drr = args.drr or saved["drr"]
    if (drr == "pilots") != p.pilots:
        raise ValueError("pilots are part of the encoding: decode with the DRR the run was encoded for, "
                         "or any classical DRR if it had none")
    e = encode(saved["mesh"], p)
    counts = json.loads((out / "counts.json").read_text())
    return finish(out, e, bridge.values_from_counts(e, counts), drr, {"decoded_again": True})


def main(argv=None):
    """``python -m tessa_ripple.hardware``: see the module docstring.

    Args:
        argv: Arguments (None: sys.argv).
    """
    ap = argparse.ArgumentParser(prog="python -m tessa_ripple.hardware", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        c = sub.add_parser(name)
        c.add_argument("mesh")
        c.add_argument("--chip", choices=bridge.DEVICES, default="fez")
        c.add_argument("--shots", type=int, default=4096)
        c.add_argument("--drr", choices=bridge.DRR_CHOICES, default="none")
        c.add_argument("--payload", choices=bridge.PAYLOADS)
        c.add_argument("--mode", choices=("theta", "split"))
        c.add_argument("--layout", choices=("mesh", "naive"))
        c.add_argument("--readout", choices=bridge.READOUTS)
        c.add_argument("--placement", choices=bridge.PLACEMENTS)
        c.add_argument("--compression", type=float)
        c.add_argument("--headroom", type=float)
    run_p = sub.choices["run"]
    run_p.add_argument("--out", required=True)
    run_p.add_argument("--simulate", action="store_true", help="run the whole-chip circuits on Aer instead")
    run_p.add_argument("--backend", choices=("noiseless", "flat noise"), default="noiseless",
                       help="with --simulate: the lattice backend")
    dec = sub.add_parser("decode")
    dec.add_argument("folder")
    dec.add_argument("--drr", choices=bridge.DRR_CHOICES)
    args = ap.parse_args(argv)
    if args.command == "plan":
        p = params_of(args)
        report = plan(encode(args.mesh, p), p)
        r = report.pop("reports")
        report.update(swaps=sum(x["swaps"] for x in r), two_qubit_gates=max(x["two_qubit_gates"] for x in r),
                      transpiled_depth=max(x["transpiled_depth"] for x in r))
        print(json.dumps(report, indent=1))
    else:
        report = run(args) if args.command == "run" else decode(args)
        print(json.dumps({k: v for k, v in report.items() if k != "reports"}, indent=1))


if __name__ == "__main__":
    main()
