from pathlib import Path

import numpy as np
import pytest

from tessa_ripple import bridge
from tessa_ripple.animate import run_frames
from tessa_ripple.bridge import (Params, cells, decode_raw, encode_mesh, mesh_error, morph_clamp, morph_expected, morph_frame,
                                 read_mesh, restore, sweep_circuit, sweep_values)
from tessa_ripple.mesh import blob, cube, decimate, load_decimated, load_obj, replay_decimation, write_obj
from tessa_ripple.morph import align, closest_points, correspond

SUZANNE = "tessa_ripple/suzanne.obj"
A, AF = blob(12, 8)


def sculpted(v):
    """Same topology, different shape: squash z, bulge the sides."""
    out = v.copy()
    out[:, 0] *= 1 + 0.3 * np.abs(v[:, 2])
    out[:, 2] *= 0.7
    return out


@pytest.fixture
def b_path(tmp_path):
    """B: A sculpted, saved as OBJ (load_obj recentres and rescales it; align undoes that)."""
    return str(write_obj(tmp_path / "b.obj", sculpted(A), AF))


def morph(verts, faces, path, shared=True, **kw):
    p = Params(mode="morph", target=path, shared_topology=shared, **kw)
    return encode_mesh(verts, faces, p), p


# --- M1: geometry ------------------------------------------------------------------
def test_align_matches_centroid_and_rms_radius():
    rng = np.random.default_rng(0)
    b = rng.normal(size=(50, 3)) * 3 + 7
    out = align(b, A)
    assert np.allclose(out.mean(0), A.mean(0))
    rms = lambda v: np.sqrt(((v - v.mean(0)) ** 2).sum(1).mean())  # noqa: E731
    assert np.isclose(rms(out), rms(A))
    assert np.allclose(np.corrcoef((out - out.mean(0)).ravel(), (b - b.mean(0)).ravel())[0, 1], 1)  # no rotation


def test_shared_topology_with_mismatched_counts_is_a_clear_error(tmp_path):
    v, f = blob(16, 10)
    path = write_obj(tmp_path / "b.obj", v, f)
    with pytest.raises(ValueError, match=rf"A has {len(A)} vertices, B has {len(v)}. Untick Shared"):
        correspond(A, AF, path, shared_topology=True)


def test_replay_decimation_reproduces_decimate():
    v, f = load_obj(SUZANNE)
    dv, df, log = decimate(v, f, return_log=True)
    rv, rf = replay_decimation(v, f, log)
    assert np.array_equal(rv, dv) and rf == df
    v2, f2, logs = load_decimated(SUZANNE, 2)
    assert len(logs) == 2 and np.array_equal(v2, decimate(*decimate(v, f))[0])


def test_shared_topology_follows_a_decimated_a(tmp_path):
    v, f, logs = load_decimated(SUZANNE, 1)
    full, _ = load_obj(SUZANNE)
    m = correspond(v, f, SUZANNE, shared_topology=True, a_logs=logs, a_full_count=len(full))
    assert m["target"].shape == v.shape and m["distance"] < 1e-9  # B = A: same points


def test_closest_point_on_a_cube():
    v, f = cube(3)
    q, _ = closest_points(v, v, f)
    assert np.allclose(q, v)  # a vertex of B maps to itself
    off = np.array([[2, 0.3, 0.2], [0.2, -3, 0.1], [2, 2, 2], [0.1, 0.4, 1.5], [-1.5, 0.9, -0.2]])
    want = np.array([[1, 0.3, 0.2], [0.2, -1, 0.1], [1, 1, 1], [0.1, 0.4, 1], [-1, 0.9, -0.2]])
    q, _ = closest_points(off, v, f)
    assert np.allclose(q, want)


def test_closest_point_correspondence_reports_coverage(tmp_path):
    path = write_obj(tmp_path / "s.obj", *blob(24, 12))
    m = correspond(A, AF, path)
    assert m["method"] == "closest point" and 0.5 < m["coverage"] <= 1


# --- M2: encode + read the endpoints -----------------------------------------------
def test_sweep_endpoints_noiseless(b_path):
    e, p = morph(A, AF, b_path)  # transition decode by default
    assert p.sweep_decode == "transition"
    for t, want in ((0.0, A), (1.0, e.target)):
        raw, second = decode_raw(e, Params(**{**p.__dict__, "morph_t": t}))
        assert second is None and mesh_error(want, morph_frame(e, raw, t)) < 0.03
    assert not np.allclose(e.target, A, atol=0.05)  # B really differs


@pytest.mark.parametrize("payload", ["xyz", "radial"])
def test_morph_forces_clamp_b_and_round_trips(b_path, payload):
    e, p = morph(A, AF, b_path, payload=payload, clamp=0.15 * np.pi, clamp_b=0.05 * np.pi)
    assert e.enc.mode == "pair" and e.enc.clamps == (0.15 * np.pi, 0.15 * np.pi)
    assert morph_clamp(Params()) == pytest.approx(0.2 * np.pi)
    assert e.stats["values"] == 2 * e.slots.size
    assert mesh_error(A, read_mesh(e, p)) < 0.03
    assert mesh_error(e.target, read_mesh(e, Params(**{**p.__dict__, "morph_t": 1.0}))) < 0.04


def test_keys():
    base = Params(mode="morph", target="b.obj")
    assert base.encoder_key != Params(mode="morph", target="c.obj").encoder_key
    assert base.encoder_key != Params(mode="morph", target="b.obj", shared_topology=True).encoder_key
    sweep = Params(mode="morph")
    assert sweep.readout_key != Params(mode="morph", morph_t=0.5).readout_key
    assert Params().readout_key == Params(morph_t=0.5).readout_key  # t only matters in morph
    assert Params(target="b.obj").encoder_key == Params().encoder_key  # only morph cares


# --- M3: basis sweep ---------------------------------------------------------------
def test_sweep_at_zero_matches_theta_mode(b_path):
    e, p = morph(A, AF, b_path, sweep_decode="raw", shots=20_000)
    sweep = read_mesh(e, p)
    theta = Params(clamp=morph_clamp(p), shots=20_000)
    z = read_mesh(encode_mesh(A, AF, theta), theta)
    assert mesh_error(A, sweep) < 0.02 and mesh_error(z, sweep) < 0.02


def test_sweep_at_one_matches_closed_form(b_path):
    e, p = morph(A, AF, b_path, sweep_decode="raw", morph_t=1.0, shots=65_536)
    got = decode_raw(e, p)[0].reshape(e.slots.shape)
    want = sweep_values(e, np.pi / 2, "raw")
    lo, hi = e.enc.ranges[0]
    assert np.abs(got - want).mean() / (hi - lo) < 0.01
    # the opposite rotation sign would read cos(-φ)... sin θ: compare against a flipped φ
    from tessa.core.mode import get_mode
    m = get_mode("pair")
    x = lambda v: np.clip((v.ravel() - lo) / (hi - lo), 0, 1)  # noqa: E731
    th, ph = m.angles(x(e.prep.reference), e.enc.clamps), m.phases(x(e.prep_b.reference), e.enc.clamps)
    flipped = np.arccos(-np.sin(th) * np.cos(ph))
    wrong = lo + (hi - lo) * m.values_from_angles(flipped, e.enc.clamps)
    assert np.abs(got.ravel() - wrong).mean() / (hi - lo) > 0.1


def test_a_equals_b(tmp_path):
    v, f = cube(5)  # xyz spans exactly -1..1, so with no headroom the values fill the range
    path = write_obj(tmp_path / "a.obj", v, f)
    e, p = morph(v, f, str(path), payload="xyz", headroom=0.0, shots=65_536)
    ends = [read_mesh(e, Params(**{**p.__dict__, "morph_t": t})) for t in (0.0, 1.0)]
    assert mesh_error(*ends) < 0.02  # transition: A = B reads the same at both ends
    q = Params(**{**p.__dict__, "sweep_decode": "raw", "morph_t": 1.0})
    lo, hi = e.enc.ranges[0]
    x_in = (e.prep.reference - lo) / (hi - lo)
    x_out = (decode_raw(e, q)[0] - lo) / (hi - lo)
    assert x_in.min() < 0.01 and x_in.max() > 0.99  # the values span the whole range
    assert x_out.min() == pytest.approx(0.222, abs=0.02) and x_out.max() == pytest.approx(0.778, abs=0.02)
    # the fold: the outermost values come back inside the extremes of the readout
    assert x_out[x_in.argmin()] > x_out.min() + 0.005 and x_out[x_in.argmax()] < x_out.max() - 0.005


def test_sweep_circuit_takes_one_angle_per_data_qubit(b_path):
    e, _ = morph(A, AF, b_path)
    g = len(e.graph.groups)
    qc, enc = sweep_circuit(e, np.linspace(0, 1, g))
    assert enc.mode == "single" and qc.count_ops().get("ry", 0) >= g
    assert sweep_circuit(e, np.linspace(0, 1, g))[0] is qc  # cached
    with pytest.raises(ValueError, match="one angle per data qubit"):
        sweep_circuit(e, np.zeros(g + 1))
    # per-qubit alpha: the closed form follows each qubit's own angle
    alpha = np.where(np.arange(g) % 2, np.pi / 2, 0.0)
    vals = sweep_values(e, alpha, "raw")
    cell = np.searchsorted(cells(e.graph)[:, 1], e.slots, side="right")
    at0 = alpha[cell] == 0
    assert np.allclose(vals[at0], e.prep.reference.reshape(e.slots.shape)[at0])


# --- M3b: batching -----------------------------------------------------------------
@pytest.fixture(scope="module")
def suzanne_batched():
    v, f = load_obj(SUZANNE)
    p = Params(mode="morph", payload="xyz", target=SUZANNE, shared_topology=True)
    return v, f, p, encode_mesh(v, f, p)


def test_batching_splits_suzanne_into_even_runs(suzanne_batched):
    v, f, p, e = suzanne_batched
    assert e.runs == 2 and sorted(e.stats["values_per_run"]) == [2 * 759, 2 * 762]  # whole vertices
    assert e.graph.capacity == 792
    assert np.bincount(e.batches).tolist() == [254, 253]
    for k in range(2):  # every vertex in exactly one batch, its own slot within the batch
        s = e.slots[e.batches == k].ravel()
        assert len(set(s)) == len(s)
    assert mesh_error(v, read_mesh(e, p)) < 0.03


@pytest.mark.parametrize("backend, runs", [("phoenix", 2), ("fez", 4)])
def test_batch_counts_on_chips(backend, runs):
    v, f = load_obj(SUZANNE)
    e = encode_mesh(v, f, Params(mode="morph", payload="xyz", target=SUZANNE, shared_topology=True,
                                 backend=backend))
    assert e.runs == runs and len(e.parts) == runs
    assert sum(e.stats["values_per_run"]) == 2 * 3 * len(v)


def test_batches_use_different_seeds(suzanne_batched, monkeypatch):
    v, f, p, e = suzanne_batched
    seeds, real = [], bridge.sampler
    monkeypatch.setattr(bridge, "sampler", lambda g, q, s: seeds.append(s) or real(g, q, s))
    decode_raw(e, p, seed=3)
    assert seeds == [3, 3 + bridge.BATCH_SEED_STEP]


def test_batch_placement_on_fez():
    v, f = load_obj(SUZANNE)
    first = {}
    for placement in ("next-best", "same"):
        e = encode_mesh(v, f, Params(mode="morph", payload="xyz", target=SUZANNE, shared_topology=True,
                                     backend="fez", batch_placement=placement))
        # each batch's first patch (it holds the batch's lowest vertex) goes on the top of its ranking
        lead = [np.flatnonzero(e.batches == k)[0] for k in range(e.runs)]
        cell = np.searchsorted(cells(e.graph)[:, 1], e.slots[lead, 0], side="right")
        first[placement] = [e.graph.groups[c].data_qubit for c in cell]
    best = int(np.argmax(e.quality))
    assert first["same"] == [e.graph.groups[best].data_qubit] * 4
    assert first["next-best"][0] == e.graph.groups[best].data_qubit
    assert len(set(first["next-best"])) == 4  # every batch starts somewhere new


def test_a_mesh_that_fits_is_one_run_as_before(b_path):
    e, p = morph(A, AF, b_path)
    assert e.runs == 1 and e.parts == () and e.batch is None
    assert np.array_equal(e.slots, encode_mesh(A, AF, Params(mode="colour")).slots)


@pytest.mark.parametrize("decode", ["raw", "transition"])
def test_morph_frames_sweep_t(b_path, decode):
    e, p = morph(A, AF, b_path, sweep_decode=decode)
    vals, second = run_frames(e, p, 3, ts=[0.0, 0.5, 1.0])
    assert vals.shape == (3, e.slots.size) and second is None
    assert mesh_error(A, restore(e, vals[0])) < 0.02
    for t, raw in zip([0.0, 0.5, 1.0], vals):  # each frame read at its own angle
        want = morph_expected(e, t, sweep_decode=decode)
        assert mesh_error(want, morph_frame(e, raw, t)) < 0.03
    assert mesh_error(morph_expected(e, 0, sweep_decode=decode), morph_expected(e, 1, sweep_decode=decode)) > 0.03


def test_radial_sweep_keeps_the_mesh_size(b_path):
    e, _ = morph(A, AF, b_path, payload="radial")
    lo, hi = e.enc.ranges[0]
    r = e.prep.reference
    assert lo < r.min() and r.max() < hi and abs((lo + hi) / 2 - r.mean()) < 1e-9  # centred on A's mean radius
    rms = lambda x: np.sqrt(((x - e.centre) ** 2).sum(1).mean())  # noqa: E731
    size = [rms(morph_expected(e, t, True, "raw")) / rms(A) for t in np.linspace(0, 1, 9)]
    assert max(abs(np.array(size) - 1)) < 0.03


# --- transition decode (Tessa's transition / tilted_readout.py) ----------------------
def test_transition_closed_form_runs_exactly_from_a_to_b(b_path):
    e, _ = morph(A, AF, b_path)
    a, b = (r.reshape(e.slots.shape) for r in (e.prep.reference, e.prep_b.reference))
    assert np.allclose(sweep_values(e, 0.0, "transition"), a)
    assert np.allclose(sweep_values(e, np.pi / 2, "transition"), b)
    same, _ = morph(A, AF, str(write_obj(Path(b_path).with_name("a.obj"), A, AF)))
    for alpha in np.linspace(0, np.pi / 2, 5):  # A = B: every frame is A
        assert np.allclose(sweep_values(same, alpha, "transition"),
                           same.prep.reference.reshape(same.slots.shape), atol=1e-9)


@pytest.mark.parametrize("payload", ["radial", "xyz"])
def test_transition_decode_matches_its_closed_form(b_path, payload):
    e, p = morph(A, AF, b_path, payload=payload, shots=65_536)
    lo, hi = e.enc.ranges[0]
    for t in (0.5, 1.0):
        q = Params(**{**p.__dict__, "morph_t": t})
        got = decode_raw(e, q)[0]
        assert np.abs(got - sweep_values(e, t * np.pi / 2, "transition").ravel()).mean() / (hi - lo) < 0.02
    assert mesh_error(e.target, read_mesh(e, Params(**{**p.__dict__, "morph_t": 1.0}))) < 0.04


def test_transition_frames_share_one_z_reference(b_path, monkeypatch):
    e, p = morph(A, AF, b_path)
    seeds, real = [], bridge.sampler
    monkeypatch.setattr(bridge, "sampler", lambda g, q, s: seeds.append(s) or real(g, q, s))
    vals, _ = run_frames(e, p, 3, ts=[0.0, 0.5, 1.0])
    assert seeds == [0, 0, 1, 2]  # the reference, then one readout per frame
    lo, hi = e.enc.ranges[0]
    assert np.abs(vals[0] - e.prep.reference).mean() / (hi - lo) < 0.04  # t = 0: the reference, i.e. A
    seeds.clear()
    decode_raw(e, Params(**{**p.__dict__, "morph_t": 0.5}), seed=7)  # a lone readout reads its own
    assert seeds == [7, 7]


def test_transition_keeps_the_size_when_a_equals_b(tmp_path):
    path = write_obj(tmp_path / "a.obj", A, AF)
    e, _ = morph(A, AF, str(path), payload="xyz")
    rms = lambda x: np.sqrt(((x - e.centre) ** 2).sum(1).mean())  # noqa: E731
    for t in np.linspace(0, 1, 5):
        assert rms(morph_expected(e, t)) == pytest.approx(rms(A))
    assert Params(mode="morph").readout_key != Params(mode="morph", sweep_decode="raw").readout_key


def test_sweep_circuit_is_tessas_tilted_for_one_angle(b_path):
    from tessa.hardware.pipeline import _tilted
    e, _ = morph(A, AF, b_path)
    for alpha in (0.0, 0.3, np.pi / 2):
        assert sweep_circuit(e, alpha)[0] == _tilted(e.circuit, e.enc, alpha)
    g = len(e.graph.groups)
    assert sweep_circuit(e, np.full(g, 0.3))[0] == _tilted(e.circuit, e.enc, 0.3)  # equal per-qubit angles too
    mixed = np.linspace(0, 1, g)
    want = e.circuit.copy()
    for grp, a in zip(e.graph.groups, mixed):
        want.ry(-a, grp.data_qubit)
    assert sweep_circuit(e, mixed)[0] == want
