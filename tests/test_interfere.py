import numpy as np
import pytest
from tessa import Lattice
from tessa.hardware import devices

from tessa_ripple import interfere as I
from tessa_ripple.bridge import Params, cells, encode_mesh, two_qubit
from tessa_ripple.mesh import blob, write_obj

A, AF = blob(12, 8)


def sculpted(v):
    """Same topology, different shape: squash z, bulge the sides."""
    out = v.copy()
    out[:, 0] *= 1 + 0.3 * np.abs(v[:, 2])
    out[:, 2] *= 0.7
    return out


@pytest.fixture(scope="module")
def b_path(tmp_path_factory):
    return str(write_obj(tmp_path_factory.mktemp("b") / "b.obj", sculpted(A), AF))


@pytest.fixture(scope="module", params=["radial", "xyz"])
def ie(request, b_path):
    p = Params(mode="morph", payload=request.param, target=b_path, shared_topology=True)
    return I.encode_interfere(A, AF, p)


def values(ie, which):
    lo, hi = ie.e.enc.ranges[0]
    return lo + (hi - lo) * (ie.a01 if which == "a" else ie.b01)


@pytest.mark.parametrize("graph", [Lattice(6, 6), Lattice(10, 12), devices.graph("phoenix"),
                                   devices.graph("fez")], ids=["6x6", "10x12", "phoenix", "fez"])
def test_selectors_pack_every_data_qubit_into_one_star(graph):
    st = I.choose_selectors(graph)
    for g, s in zip(graph.groups, st.star):
        assert sum(a in g.address_qubits for a in st.selectors) == (s >= 0)  # at most one, and it is s
    assert (st.star >= 0).all()  # perfect packing: nothing left out
    assert st.capacity == graph.capacity // 2


def test_pair_slots_differ_only_in_the_selector_bit():
    graph = Lattice(6, 6)
    st = I.choose_selectors(graph)
    a, b = I.pair_slots(st, np.arange(st.capacity))
    assert len(set(a) | set(b)) == 2 * st.capacity  # every full slot used once
    g = np.searchsorted(cells(graph)[:, 1], a, side="right")
    local_a, local_b = a - cells(graph)[g, 0], b - cells(graph)[g, 0]
    assert ((local_a >> st.bit[g]) & 1 == 0).all()
    assert (local_b - local_a == 1 << st.bit[g]).all()


def test_endpoints_are_exactly_a_and_b(ie):
    assert np.allclose(I.ideal(ie, 0.0), values(ie, "a"), atol=1e-9)
    assert np.allclose(I.ideal(ie, 1.0), values(ie, "b"), atol=1e-9)


def test_frames_lie_between_the_superposition_and_the_mixture(ie):
    for t in (0.25, 0.5, 0.75):
        x, pu, mix = I.ideal(ie, t), I.pure(ie, t), I.incoherent(ie, t)
        assert ((x >= np.minimum(pu, mix) - 1e-9) & (x <= np.maximum(pu, mix) + 1e-9)).all()
        assert np.abs(pu - mix).max() > 1e-3  # coherence is visible at all


def test_noiseless_read_matches_ideal_and_the_control_reads_the_mixture(ie):
    p = Params(mode="morph", shots=65_536)
    x, mix = I.ideal(ie, 0.25), I.incoherent(ie, 0.25)
    raw, kept = I.read(ie, p, 0.25)
    span = np.subtract(*ie.e.enc.ranges[0][::-1])
    assert np.abs(raw - x).mean() < 0.02 * span
    assert kept > 0.5
    assert I.coherence(raw, x, mix) == pytest.approx(1, abs=0.2)
    assert I.coherence(I.read(ie, p, 0.25, dephase=True)[0], x, mix) == pytest.approx(0, abs=0.2)


def test_coherence_is_undefined_at_the_endpoints(ie):
    """At t = 0 and 1 the coherent and mixed frames agree up to rounding: no λ to measure."""
    for t in (0.0, 1.0):
        x, mix = I.ideal(ie, t), I.incoherent(ie, t)
        assert np.isnan(I.coherence(x + 1e-3, x, mix))


def test_positions_round_trip_at_the_endpoints(ie):
    for t, which in ((0.0, "a"), (1.0, "b")):
        pos = I.frame(ie, I.ideal(ie, t), t)
        ref = A if which == "a" else ie.e.target
        assert np.abs(pos - ref).max() < 1e-6


def test_about_as_many_two_qubit_gates_as_the_sweep_morph_on_a_chip(b_path):
    """Summed over every run: the counts depend on the data (out/interfere/costs.csv has
    apple on Phoenix at 504 against 496), so only a rough match is promised."""
    kw = dict(mode="morph", target=b_path, shared_topology=True, backend="fez")
    total = lambda e: sum(two_qubit(c) for c, _ in (e.parts or ((e.circuit, e.enc),)))  # noqa: E731
    ie = I.encode_interfere(A, AF, Params(**kw))
    sweep = encode_mesh(A, AF, Params(**kw))
    assert total(ie.e) <= 1.1 * total(sweep)


def test_run_frames_reads_each_frame_at_its_own_t_and_seed(ie, monkeypatch):
    calls, real = [], I.read
    monkeypatch.setattr(I, "read", lambda ie_, p, t, seed=None: calls.append((t, seed)) or real(ie_, p, t, seed))
    p = Params(mode="morph", shots=2048, seed=4)
    raw, kept = I.run_frames(ie, p, [0.0, 0.5, 1.0])
    assert calls == [(0.0, 4), (0.5, 5), (1.0, 6)]
    assert raw.shape == (3, ie.e.slots.size) and kept.shape == (3,)
    assert np.all((0 < kept) & (kept <= 1))


def test_why_tessa_costs_a_and_b_slots_as_one_field(ie):
    from tessa_ripple.bridge import resources
    where = [np.concatenate([ie.e.slots.ravel(), ie.b_slots.ravel()])]
    tessa = resources(ie.e, where)[0]
    assert tessa["name"] == "Tessa" and tessa["settings"] == 1  # one Z setting, no φ field


def test_ideal_frames_are_computed_once_per_t(ie):
    first = I.ideal(ie, 0.25)
    assert I.ideal(ie, 0.25) is first and not first.flags.writeable
