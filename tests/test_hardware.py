"""Tessa's whole-chip (real hardware) path: bridge.read_hardware and the hardware CLI."""

import json

import numpy as np
import pytest
from tessa.backends import aer_sampler

from tessa_ripple import bridge, hardware
from tessa_ripple.mesh import blob, cube, write_obj

V, F = blob(6, 4)


@pytest.mark.parametrize("kw", [dict(), dict(readout="bloch"), dict(readout="snap"), dict(pilots=True),
                                dict(mode="split"), dict(payload="spherical")],
                         ids=["z", "bloch", "snap", "pilots", "split", "spherical"])
def test_whole_chip_readout_matches_the_per_cell_simulation(kw):
    """The circuits a chip would run, simulated whole, decode as well as the per-cell readout."""
    p = bridge.Params(shots=8192, **kw)
    e = bridge.encode_mesh(V, F, p)
    raw, counts, reports = bridge.read_hardware(e, aer_sampler(e.graph, seed=1), p.shots)
    want = bridge.encoded_shape(e)
    whole = bridge.mesh_error(want, bridge.restore(e, raw))
    cells = bridge.mesh_error(want, bridge.restore(e, bridge.decode_raw(e, p)))
    assert whole < max(2 * cells, 0.005)
    assert all(r["swaps"] == 0 for r in reports) and len(counts) == e.runs
    assert np.array_equal(bridge.values_from_counts(e, counts), raw)  # saved counts decode the same later


def test_a_chip_plan_needs_no_swaps():
    p = bridge.Params(backend="fez", readout="bloch")
    report = hardware.plan(bridge.encode_mesh(*cube(), p), p)
    assert report["backend"] == "ibm_fez" and report["bases"] == ["x", "y", "z"]
    assert all(r["swaps"] == 0 for r in report["reports"])


def test_morph_is_simulated_only(tmp_path):
    path = write_obj(tmp_path / "b.obj", V * 1.1, F)
    e = bridge.encode_mesh(V, F, bridge.Params(mode="morph", target=str(path), shared_topology=True))
    with pytest.raises(ValueError, match="simulated only"):
        bridge.hardware_circuits(e)


def test_cli_runs_simulated_and_decodes_again(tmp_path, capsys):
    mesh, out = write_obj(tmp_path / "cube.obj", *cube()), tmp_path / "run"
    hardware.main(["run", str(mesh), "--simulate", "--readout", "snap", "--out", str(out)])
    first = json.loads((out / "report.json").read_text())
    hardware.main(["decode", str(out)])
    again = json.loads((out / "report.json").read_text())
    assert again["error"] == first["error"] < 0.01 and (out / "decoded.obj").exists()
    assert "token" not in "".join(p.read_text() for p in out.glob("*.json")).lower()
    with pytest.raises(ValueError, match="pilots are part of the encoding"):
        hardware.main(["decode", str(out), "--drr", "pilots"])
