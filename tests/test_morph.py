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
    """Same topology, different shape: squash z, bulge the sides (strong enough to move
    even the flat, tilted blob by several percent of its box)."""
    out = v.copy()
    out[:, 0] *= 1 + 0.5 * np.abs(v[:, 2])
    out[:, 2] *= 0.6
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


def test_shared_topology_with_other_faces_is_a_clear_error():
    """apple_bitten has apple_full's vertex count but its own vertex order and faces."""
    v, f = load_obj("tessa_ripple/apple_full.obj")
    with pytest.raises(ValueError, match="different faces"):
        correspond(v, f, "tessa_ripple/apple_bitten.obj", shared_topology=True)


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


def test_closest_point_survives_a_zero_area_triangle():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    pts = np.array([[0.2, 0.2, 0.5], [0.5, -0.5, 0.1]])
    q, _ = closest_points(pts, v, [[0, 0, 1], [0, 1, 2]])  # face 0 repeats a corner
    assert np.allclose(q, [[0.2, 0.2, 0], [0.5, 0, 0]])
    q, _ = closest_points(pts, v, [[0, 0, 1]])  # only the degenerate face: its edge
    assert np.allclose(q, [[0.2, 0, 0], [0.5, 0, 0]])


def test_closest_point_correspondence_reports_coverage(tmp_path):
    path = write_obj(tmp_path / "s.obj", *blob(24, 12))
    m = correspond(A, AF, path)
    assert m["method"] == "closest point" and 0.5 < m["coverage"] <= 1


# --- M2: encode + read the endpoints -----------------------------------------------
def test_sweep_endpoints_noiseless(b_path):
    e, p = morph(A, AF, b_path)
    for t, want in ((0.0, A), (1.0, e.target)):
        raw = decode_raw(e, Params(**{**p.__dict__, "morph_t": t}))
        assert mesh_error(want, morph_frame(e, raw, t)) < 0.03
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
    e, p = morph(A, AF, b_path, shots=20_000)
    sweep = read_mesh(e, p)
    theta = Params(clamp=morph_clamp(p), shots=20_000)
    z = read_mesh(encode_mesh(A, AF, theta), theta)
    assert mesh_error(A, sweep) < 0.02 and mesh_error(z, sweep) < 0.02


def test_a_equals_b(tmp_path):
    v, f = cube(5)  # xyz spans exactly -1..1, so with no headroom the values fill the range
    path = write_obj(tmp_path / "a.obj", v, f)
    e, p = morph(v, f, str(path), payload="xyz", headroom=0.0, shots=65_536)
    ends = [read_mesh(e, Params(**{**p.__dict__, "morph_t": t})) for t in (0.0, 1.0)]
    assert mesh_error(*ends) < 0.02  # A = B reads the same at both ends


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
    vals = sweep_values(e, alpha)
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


def test_later_batches_start_on_the_next_best_qubits_on_fez():
    v, f = load_obj(SUZANNE)
    e = encode_mesh(v, f, Params(mode="morph", payload="xyz", target=SUZANNE, shared_topology=True,
                                 backend="fez"))
    # each batch's first patch (it holds the batch's lowest vertex) goes on the top of its ranking
    lead = [np.flatnonzero(e.batches == k)[0] for k in range(e.runs)]
    cell = np.searchsorted(cells(e.graph)[:, 1], e.slots[lead, 0], side="right")
    first = [e.graph.groups[c].data_qubit for c in cell]
    assert first[0] == e.graph.groups[int(np.argmax(e.quality))].data_qubit
    assert len(set(first)) == 4  # every batch starts somewhere new


def test_a_mesh_that_fits_is_one_run_as_before(b_path):
    e, p = morph(A, AF, b_path)
    assert e.runs == 1 and e.parts == () and e.batch is None
    assert np.array_equal(e.slots, encode_mesh(A, AF, Params()).slots)


def test_morph_frames_sweep_t(b_path):
    e, p = morph(A, AF, b_path)
    vals = run_frames(e, p, 3, ts=[0.0, 0.5, 1.0])
    assert vals.shape == (3, e.slots.size)
    assert mesh_error(A, restore(e, vals[0])) < 0.02
    for t, raw in zip([0.0, 0.5, 1.0], vals):  # each frame read at its own angle
        want = morph_expected(e, t)
        assert mesh_error(want, morph_frame(e, raw, t)) < 0.03
    assert mesh_error(morph_expected(e, 0), morph_expected(e, 1)) > 0.03


def test_radial_morph_range_is_centred_on_a_mean_radius(b_path):
    e, _ = morph(A, AF, b_path, payload="radial")
    lo, hi = e.enc.ranges[0]
    r = e.prep.reference
    assert lo < r.min() and r.max() < hi and abs((lo + hi) / 2 - r.mean()) < 1e-9


# --- transition decode (Tessa's transition / tilted_readout.py) ----------------------
def test_transition_closed_form_runs_exactly_from_a_to_b(b_path):
    e, _ = morph(A, AF, b_path)
    a, b = (r.reshape(e.slots.shape) for r in (e.prep.reference, e.prep_b.reference))
    assert np.allclose(sweep_values(e, 0.0), a)
    assert np.allclose(sweep_values(e, np.pi / 2), b)
    same, _ = morph(A, AF, str(write_obj(Path(b_path).with_name("a.obj"), A, AF)))
    for alpha in np.linspace(0, np.pi / 2, 5):  # A = B: every frame is A
        assert np.allclose(sweep_values(same, alpha),
                           same.prep.reference.reshape(same.slots.shape), atol=1e-9)


@pytest.mark.parametrize("payload", ["radial", "xyz"])
def test_transition_decode_matches_its_closed_form(b_path, payload):
    e, p = morph(A, AF, b_path, payload=payload, shots=65_536)
    lo, hi = e.enc.ranges[0]
    for t in (0.5, 1.0):
        q = Params(**{**p.__dict__, "morph_t": t})
        got = decode_raw(e, q)
        assert np.abs(got - sweep_values(e, t * np.pi / 2).ravel()).mean() / (hi - lo) < 0.02
    assert mesh_error(e.target, read_mesh(e, Params(**{**p.__dict__, "morph_t": 1.0}))) < 0.04


def test_transition_frames_share_one_z_reference(b_path, monkeypatch):
    e, p = morph(A, AF, b_path)
    seeds, real = [], bridge.sampler
    monkeypatch.setattr(bridge, "sampler", lambda g, q, s: seeds.append(s) or real(g, q, s))
    vals = run_frames(e, p, 3, ts=[0.0, 0.5, 1.0])
    ref = bridge.reference_seed(0)
    assert seeds == [ref, 0, 1, 2]  # the reference (its own run), then one readout per frame
    lo, hi = e.enc.ranges[0]
    assert np.abs(vals[0] - e.prep.reference).mean() / (hi - lo) < 0.04  # t = 0: A
    seeds.clear()
    decode_raw(e, Params(**{**p.__dict__, "morph_t": 0.5}), seed=7)  # a lone readout reads its own
    assert seeds == [7, bridge.reference_seed(7)]


def test_frames_at_t0_are_plain_z_readouts(b_path):
    """At alpha = 0 a frame decodes from its own Z readout, never against the shared one."""
    e, p = morph(A, AF, b_path, shots=256)
    vals = run_frames(e, p, 3)
    own = bridge.tilted_angles(e, p, p.seed, 0.0, frames=3)  # the frames share one job
    for i in range(3):
        assert np.array_equal(vals[i], bridge.transition_values(e, own[i], own[i], 0.0))


def test_transition_keeps_the_size_when_a_equals_b(tmp_path):
    path = write_obj(tmp_path / "a.obj", A, AF)
    e, _ = morph(A, AF, str(path), payload="xyz")
    rms = lambda x: np.sqrt(((x - e.centre) ** 2).sum(1).mean())  # noqa: E731
    for t in np.linspace(0, 1, 5):
        assert rms(morph_expected(e, t)) == pytest.approx(rms(A))


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


# --- morph front: one alpha per (batch, data qubit) ----------------------------------
from tessa_ripple.bridge import front_distances, front_reference, front_source, morph_alpha, qubit_distance  # noqa: E402
from tessa_ripple.morph import front_alpha, geodesic  # noqa: E402


def front(e, p, width=0.15, source=None):
    """``p`` with the front on, and its ``(width, source)``."""
    q = Params(**{**p.__dict__, "morph_front": True, "front_width": width, "front_source": source})
    return q, q.front


def test_front_alpha_ends_and_is_monotone_in_t():
    d = np.linspace(0, 1, 11)
    for w in (0.05, 0.15, 0.5):
        assert np.all(front_alpha(d, 0.0, w) == 0)
        assert np.all(front_alpha(d, 1.0, w) == np.pi / 2)
        a = np.array([front_alpha(d, t, w) for t in np.linspace(0, 1, 41)])
        assert np.all(np.diff(a, axis=0) >= 0)
    with pytest.raises(ValueError):
        front_alpha(d, 0.5, 0.0)


@pytest.mark.parametrize("payload", ["xyz", "radial"])
def test_front_endpoints_noiseless(b_path, payload):
    e, p = morph(A, AF, b_path, payload=payload, shots=65_536)
    q, fr = front(e, p)
    for t, want in ((0.0, A), (1.0, e.target)):
        qt = Params(**{**q.__dict__, "morph_t": t})
        assert mesh_error(want, morph_frame(e, decode_raw(e, qt), t, front=fr)) < 0.03
        assert mesh_error(want, morph_expected(e, t, front=fr)) < 1e-9


def test_front_off_is_the_scalar_sweep(b_path):
    e, p = morph(A, AF, b_path, shots=1024)
    z = bridge.tilted_angles(e, p, bridge.reference_seed(p.seed), 0.0)
    for t in (0.0, 0.3, 0.7, 1.0):
        q = Params(**{**p.__dict__, "morph_t": t})
        alpha = t * np.pi / 2
        assert q.front is None and morph_alpha(e, t) == alpha
        want = bridge.transition_values(e, bridge.tilted_angles(e, q, q.seed, alpha), z, alpha)
        assert np.array_equal(decode_raw(e, q), want)  # bit for bit the scalar path
        dirs = bridge.morph_dirs(e, t)
        assert np.array_equal(morph_expected(e, t), bridge.from_payload(e, sweep_values(e, alpha), dirs))


def test_constant_distance_is_the_scalar_sweep(b_path, monkeypatch):
    e, p = morph(A, AF, b_path)
    g = len(e.graph.groups)
    d = np.full((e.runs, g), 0.4)
    monkeypatch.setattr(bridge, "front_distances", lambda e, s: (0, np.full(len(e.verts), 0.4), d))
    for t in (0.3, 0.5, 0.8):
        alpha = morph_alpha(e, t, (0.15, None))
        a0 = float(alpha.flat[0])
        assert np.all(alpha == a0)
        assert sweep_circuit(e, alpha)[0] == sweep_circuit(e, a0)[0]
        assert np.allclose(sweep_values(e, alpha), sweep_values(e, a0))


def test_vertices_on_one_data_qubit_share_alpha(b_path):
    e, p = morph(A, AF, b_path, payload="radial")
    _, fr = front(e, p)
    per = bridge._per_value(e, morph_alpha(e, 0.5, fr)).reshape(e.slots.shape)[:, 0]
    for grp in np.unique(e.groups_of[:, 0]):
        assert len(np.unique(per[e.groups_of[:, 0] == grp])) == 1
    assert len(np.unique(per)) > 1  # the front is somewhere in the middle at t = 0.5


@pytest.fixture(scope="module")
def suzanne_fez():
    v, f = load_obj(SUZANNE)
    p = Params(mode="morph", payload="xyz", target=SUZANNE, shared_topology=True, backend="fez")
    return encode_mesh(v, f, p), p


def test_batched_alpha_is_per_batch_and_qubit(suzanne_fez):
    e, p = suzanne_fez
    g = len(e.graph.groups)
    assert e.runs == 4
    alpha = np.arange(4 * g, dtype=float).reshape(4, g) / (4 * g)
    for k in range(4):  # each part's circuit uses its own row
        assert sweep_circuit(e, alpha, k)[0] == sweep_circuit(e, alpha[k], k)[0]
    per = bridge._per_value(e, alpha).reshape(e.slots.shape)
    assert np.array_equal(per, alpha[e.batches[:, None], e.groups_of])
    with pytest.raises(ValueError, match="per \\(batch, data qubit\\)"):
        sweep_circuit(e, np.zeros((3, g)))
    with pytest.raises(ValueError):
        bridge._per_value(e, np.zeros((4, g + 1)))
    # the front gives (K, G); a qubit holds different vertices per batch, so rows differ
    d = front_distances(e)[2]
    assert d.shape == (4, g) and not np.allclose(d[0], d[1])


def test_geodesic_is_normalised():
    d = geodesic(A, AF, 5)
    assert d[5] == 0 and d.max() == 1 and np.isfinite(d).all()
    v2 = np.vstack([A, A + 10])  # a second, disconnected copy is unreachable: 1
    f2 = AF + [[i + len(A) for i in f] for f in AF]
    d2 = geodesic(v2, f2, 5)
    assert np.allclose(d2[:len(A)], d) and np.all(d2[len(A):] == 1)


def test_qubit_distance_is_the_median_and_auto_source(b_path):
    e, p = morph(A, AF, b_path)
    d = np.random.default_rng(0).random(len(A))
    dq = qubit_distance(e, d)
    for grp in e.used_groups:
        held = np.unique(np.nonzero(e.groups_of == grp)[0])
        assert dq[0, grp] == np.median(d[held])
    unused = np.setdiff1d(np.arange(len(e.graph.groups)), e.used_groups)
    assert np.all(dq[0, unused] == 1)
    assert front_source(e) == int(np.argmax(np.linalg.norm(e.target - A, axis=1)))


def test_front_smooth_reference_ends_exact(b_path):
    e, p = morph(A, AF, b_path, payload="radial")
    _, fr = front(e, p)
    assert mesh_error(A, front_reference(e, 0.0, fr)) < 1e-9
    assert mesh_error(e.target, front_reference(e, 1.0, fr)) < 1e-9


def test_front_keys():
    base = Params(mode="morph", target="b.obj")
    on = Params(mode="morph", target="b.obj", morph_front=True)
    assert base.readout_key != on.readout_key
    assert on.readout_key != Params(mode="morph", target="b.obj", morph_front=True, front_width=0.3).readout_key
    assert on.readout_key != Params(mode="morph", target="b.obj", morph_front=True, front_source=3).readout_key
    for q in (on, Params(mode="morph", target="b.obj", morph_front=True, front_width=0.3, front_source=3)):
        assert q.encoder_key == base.encoder_key
    assert base.front is None and on.front == (0.15, None)
    assert Params(morph_front=True).front is None  # only morph has a front


def test_front_values_not_yet_reached_are_their_own_z_readout(b_path):
    e, p = morph(A, AF, b_path, payload="radial")
    q, fr = front(e, p)
    q = Params(**{**q.__dict__, "morph_t": 0.3})
    z = bridge.tilted_angles(e, q, 0, 0.0)
    got = decode_raw(e, q, seed=5, reference=z)
    alpha = morph_alpha(e, 0.3, fr)
    per = bridge._per_value(e, alpha)
    assert (per == 0).any() and (per > 0).any()
    own = bridge.tilted_angles(e, q, 5, alpha)  # unreached qubits were read in Z
    a_read = bridge.transition_values(e, own, own, 0.0)  # that readout read as A
    assert np.array_equal(got[per == 0], a_read[per == 0])
    assert not np.allclose(got[per > 0], a_read[per > 0])


@pytest.mark.parametrize("payload", ["radial", "xyz"])
def test_drr_stretches_a_frame_towards_its_own_t_not_towards_a(b_path, payload):
    """At t = 1 the frame is B: DRR against A's range would pull it back towards A."""
    e, p = morph(A, AF, b_path, payload=payload, shots=65536)
    raw = decode_raw(e, Params(**{**p.__dict__, "morph_t": 1.0}))
    want = morph_expected(e, 1.0)
    plain = mesh_error(want, morph_frame(e, raw, 1.0))
    assert mesh_error(want, morph_frame(e, raw, 1.0, "mean_std")) < 1.5 * plain
    against_a = mesh_error(want, bridge.restore(e, raw, "min_max", bridge.morph_dirs(e, 1.0)))  # A's input
    assert mesh_error(want, morph_frame(e, raw, 1.0, "min_max")) < 0.5 * against_a
