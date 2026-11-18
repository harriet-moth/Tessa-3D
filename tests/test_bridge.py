import numpy as np
import pytest

from tessa_ripple.animate import blend
from tessa_ripple.bridge import (Params, cells, encode_mesh, fit_lattice, mesh_error, qubit_map, read_edges,
                                 read_mesh, resources, decode_raw)
from tessa_ripple.mesh import blob, cube, uv_sphere

V, F = blob(12, 8)


def test_fit_lattice_holds_values_and_cap_is_whole_phoenix():
    for n, comps in ((86, 1), (86, 3), (266, 1)):
        lat = fit_lattice(n, comps)
        assert sum((hi - lo) // comps for lo, hi in cells(lat)) >= n
    assert (fit_lattice(700, 1).rows, fit_lattice(700, 1).cols) <= (10, 12)


def test_capacity_error_is_clear():
    with pytest.raises(ValueError, match="exceed"):
        fit_lattice(10_000)


@pytest.mark.parametrize("payload", ["radial", "xyz"])
@pytest.mark.parametrize("layout", ["mesh", "naive"])
def test_noiseless_round_trip_within_2_percent(payload, layout):
    p = Params(payload=payload, layout=layout, shots=20_000)
    out = read_mesh(encode_mesh(V, F, p), p)
    assert out.shape == V.shape
    assert mesh_error(V, out) < 0.02


def test_mesh_layout_keeps_vertex_components_in_one_cell():
    e = encode_mesh(V, F, Params(payload="xyz"))
    stops = cells(e.graph)[:, 1]
    cell = np.searchsorted(stops, e.slots, side="right")
    assert (cell == cell[:, :1]).all()
    assert len(np.unique(e.slots)) == e.slots.size


def test_pair_mode_carries_colour_in_phi():
    p = Params(mode="pair", shots=20_000)
    e = encode_mesh(V, F, p)
    assert e.enc.mode == "pair" and e.colour is not None
    shape, colour = decode_raw(e, p)
    assert np.abs(colour - e.colour).mean() < 0.05
    assert e.stats["values"] == 2 * len(V)


def test_gain_zero_returns_original_exactly():
    p = Params()
    assert np.array_equal(blend(V, read_mesh(encode_mesh(V, F, p), p), gain=0), V)


def test_chip_backend_uses_whole_chip():
    p = Params(backend="phoenix")
    e = encode_mesh(V, F, p)
    assert e.graph.n_qubits == 120 and e.graph.capacity == 792
    assert mesh_error(V, read_mesh(e, p)) < 0.1


def test_resources_tessa_needs_no_swaps_and_fewer_qubits_than_angle():
    rows = {r["name"]: r for r in resources(encode_mesh(V, F, Params(payload="xyz")))}
    assert rows["Tessa"]["swaps"] == 0
    assert rows["Tessa"]["qubits"] < rows["Angle (1 value/qubit)"]["qubits"]


def test_edge_lengths_read_from_coherences():
    v, f = uv_sphere(8, 6)
    E, measured, exact = read_edges(v, f, Params(shots=20_000))
    assert len(E) == len(measured) == len(exact)
    assert np.abs(measured - exact).mean() / exact.mean() < 0.05


def test_cube_xyz_still_works():
    v, f = cube(3)
    p = Params(payload="xyz")
    assert mesh_error(v, read_mesh(encode_mesh(v, f, p), p)) < 0.05


@pytest.mark.parametrize("backend", ["noiseless", "phoenix"])
def test_qubit_map_points_each_value_at_its_slot(backend):
    v, f = blob(12, 8)
    e = encode_mesh(v, f, Params(backend=backend, payload="xyz"))
    m, bounds = qubit_map(e), cells(e.graph)
    assert len(m["coords"]) == e.graph.n_qubits
    for (g, addr), slot in zip(np.reshape(m["vertices"], (-1, 2)), e.slots.ravel()):
        assert bounds[g, 0] + addr == slot and addr < 2 ** len(m["groups"][g]["addresses"])
    data = {grp["data"] for grp in m["groups"]}
    assert all(a not in data for grp in m["groups"] for a in grp["addresses"])
    assert all({a, b} & data for a, b in m["edges"])  # every coupler touches a data qubit


@pytest.mark.parametrize("layout", ["mesh", "naive"])
def test_quality_placement_uses_only_the_best_data_qubits(layout):
    from tessa_ripple.bridge import chip_quality, probe_chip, snapshot_json, QUALITY_FILE  # noqa: F401
    import json
    stored = json.loads(QUALITY_FILE.read_text())["phoenix"]
    assert stored["calibrated_at"] == snapshot_json("phoenix")["calibrated_at"]  # file is current
    q = chip_quality("phoenix")
    v, f = blob(12, 8)
    e = encode_mesh(v, f, Params(backend="phoenix", layout=layout, placement="quality"))
    used = np.unique(np.searchsorted(cells(e.graph)[:, 1], e.slots, side="right"))
    assert q[used].min() >= np.sort(q)[::-1][len(used) - 1]  # no better data qubit was skipped
    g = encode_mesh(v, f, Params(backend="phoenix", layout=layout, placement="graph"))
    assert e.stats["quality"] > g.stats["quality"]
    assert Params(placement="quality").encoder_key == Params(placement="graph").encoder_key  # lattice: no effect


def test_qubit_count_is_only_what_the_mesh_uses():
    v, f = blob(16, 10)
    e = encode_mesh(v, f, Params(backend="phoenix"))
    s = e.stats
    assert s["graph_qubits"] == 120 and s["qubits"] == len(e.used_qubits) < 120
    data = {e.graph.groups[i].data_qubit for i in e.used_groups}
    assert data == {e.graph.groups[g].data_qubit for g, _ in np.reshape(qubit_map(e)["vertices"], (-1, 2))}


@pytest.mark.parametrize("layout", ["mesh", "naive"])
def test_importance_placement_puts_curved_vertices_on_better_qubits(layout):
    from tessa_ripple.mesh import importance, load_obj
    v, f = load_obj("tessa_ripple/suzanne.obj")
    w = importance(v, f)
    top = w >= np.quantile(w, 0.75)
    got = {}
    for pl in ("quality", "importance"):
        e = encode_mesh(v, f, Params(backend="phoenix", layout=layout, placement=pl))
        cell = np.searchsorted(cells(e.graph)[:, 1], e.slots[:, 0], side="right")
        got[pl] = (e.quality[cell][top].mean(), sorted(set(cell)))
        assert sorted(e.slots.ravel()) == sorted(set(e.slots.ravel()))  # every value has its own slot
    assert got["importance"][0] > got["quality"][0]
    assert got["importance"][1] == got["quality"][1]  # same data qubits, only reassigned
