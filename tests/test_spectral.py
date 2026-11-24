import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from tessa.codec import decode_from_counts

from tessa_ripple import interfere as I
from tessa_ripple import spectral as S
from tessa_ripple.bridge import (Params, encode_mesh, encoded_shape, mesh_error, morph_alpha, morph_expected,
                                 qubit_map, read_mesh, restore, run_positions, vertex_qubits)
from tessa_ripple.mesh import blob, write_obj

A, AF = blob(12, 8)


def sculpted(v):
    """Same topology, different shape and centroid: squash z, bulge the sides, shift."""
    out = v.copy()
    out[:, 0] *= 1 + 0.3 * np.abs(v[:, 2])
    out[:, 2] *= 0.7
    return out + [0.05, 0.0, 0.0]


@pytest.fixture(scope="module")
def b_path(tmp_path_factory):
    return str(write_obj(tmp_path_factory.mktemp("b") / "b.obj", sculpted(A), AF))


def test_basis_is_m_orthonormal_and_has_no_constant():
    phi, lam, mass = S.basis(A, AF, 30)
    assert phi.shape == (len(A), 30)
    assert np.allclose(phi.T @ (mass[:, None] * phi), np.eye(30), atol=1e-9)
    assert np.allclose(phi.T @ mass, 0, atol=1e-9)  # orthogonal to the constant: it was dropped
    assert (np.diff(lam) >= -1e-9).all() and lam[0] > 1e-6  # low frequency first, no zero mode
    with_const, lam_c, _ = S.basis(A, AF, 30, constant=True)
    assert np.allclose(with_const[:, 0], 1 / np.sqrt(mass.sum())) and lam_c[0] == 0
    assert np.allclose(with_const[:, 1:], phi)


def test_laplacian_is_symmetric_with_zero_row_sums():
    L, mass = S.cotan_laplacian(A, AF)
    assert abs(L - L.T).max() < 1e-12
    assert np.abs(L.sum(1)).max() < 1e-9
    assert (mass > 0).all()


def test_full_basis_reconstructs_exactly_and_error_falls_with_k():
    phi, _, mass = S.basis(A, AF, len(A) - 1)
    c = S.centroid(A, mass)
    assert np.abs(S.reconstruct(S.coefficients(A, c, phi, mass), c, phi) - A).max() < 1e-9
    err = S.truncation_errors(A, c, phi, mass)
    # the projection is M-orthogonal, so the plain RMS may tick up by a hair between
    # neighbouring k; across the spectrum it falls to zero
    assert (np.diff(err[[0, 4, 19, 39, 84]]) < 0).all() and err[-1] < 1e-12


def test_k_outside_the_mesh_raises():
    with pytest.raises(ValueError, match="between 1 and"):
        S.basis(A, AF, len(A))


def test_choose_k_meets_tolerance_for_a_and_b_and_respects_the_cap():
    B = sculpted(A)
    k = S.choose_k(A, B, AF, 0.01, 200)
    phi, _, mass = S.basis(A, AF, k, constant=True)
    c = S.centroid(A, mass)
    assert S.truncation_errors(A, c, phi, mass)[-1] <= 0.01
    assert S.truncation_errors(B, c, phi, mass)[-1] <= 0.01
    phi1, _, _ = S.basis(A, AF, k - 1, constant=True)
    worst = max(S.truncation_errors(V, c, phi1, mass)[-1] for V in (A, B))
    assert worst > 0.01  # the smallest such k
    assert S.choose_k(A, B, AF, 1e-9, 7) == 7  # capped


def test_band_scales_put_every_value_in_range():
    p = Params(payload="spectral")
    e = encode_mesh(A, AF, p)
    v = e.prep.reference
    assert np.abs(v).max() <= 1 / (1 + p.headroom) + 1e-12
    assert np.isclose(np.abs(v.reshape(-1, 3)).max(1), 1 / (1 + p.headroom)).all()  # each mode fills its band
    assert e.enc.ranges[0] == (-1.0, 1.0)


def test_exact_theta_readout_is_the_truncated_mesh():
    e = encode_mesh(A, AF, Params(payload="spectral", spectral_k=8))
    probs = Statevector(e.circuit.remove_final_measurements(inplace=False)).probabilities_dict()
    raw = decode_from_counts({"z": probs}, e.enc)[e.slots].ravel()
    trunc = encoded_shape(e)
    assert np.abs(restore(e, raw) - trunc).max() < 1e-9
    assert np.abs(restore(e, e.prep.reference) - trunc).max() < 1e-12
    assert e.spectral["error_a"] == pytest.approx(
        np.sqrt(((trunc - A) ** 2).sum(1).mean()) / np.linalg.norm(np.ptp(A, 0)))


def test_sampled_theta_readout_and_one_run():
    p = Params(payload="spectral", shots=20_000)
    e = encode_mesh(A, AF, p)
    assert e.runs == 1 and e.slots.shape == (e.spectral["k"], 3)
    assert e.spectral["error_a"] <= p.spectral_tol
    assert mesh_error(encoded_shape(e), read_mesh(e, p)) < 0.01


def test_k_too_big_for_one_run_raises():
    big, bf = blob(40, 30)
    cap = run_positions(Params()) // 3
    with pytest.raises(ValueError, match="does not fit"):
        encode_mesh(big, bf, Params(payload="spectral", spectral_k=cap + 1))


def test_encoder_key_has_spectral_settings_only_for_spectral():
    assert Params(spectral_k=5).encoder_key == Params(spectral_k=9).encoder_key
    assert Params(payload="spectral", spectral_k=5).encoder_key != Params(payload="spectral", spectral_k=9).encoder_key


def test_sweep_morph_endpoints_are_truncated_a_and_b(b_path):
    p = Params(mode="morph", payload="spectral", target=b_path, shared_topology=True)
    e = encode_mesh(A, AF, p)
    assert e.spectral["rows"] == e.spectral["k"] + 1  # the constant mode carries B's centroid shift
    assert np.allclose(e.prep.reference.reshape(-1, 3)[0], 0)  # A has none
    assert mesh_error(encoded_shape(e), morph_expected(e, 0.0)) < 1e-12
    assert mesh_error(encoded_shape(e, "b"), morph_expected(e, 1.0)) < 1e-12
    assert e.spectral["error_b"] <= p.spectral_tol
    assert mesh_error(e.target, encoded_shape(e, "b")) < 0.01  # B as matched (aligned to A)


def test_interference_morph_endpoints_are_truncated_a_and_b(b_path):
    p = Params(mode="morph", payload="spectral", target=b_path, shared_topology=True)
    ie = I.encode_interfere(A, AF, p)
    assert mesh_error(encoded_shape(ie.e), I.frame(ie, I.ideal(ie, 0.0), 0.0)) < 1e-12
    assert mesh_error(encoded_shape(ie.e, "b"), I.frame(ie, I.ideal(ie, 1.0), 1.0)) < 1e-12
    assert ie.e.runs == 1 and 2 * ie.e.slots.size <= ie.e.graph.capacity


def test_front_is_refused_and_row_features_are_per_mode(b_path):
    p = Params(mode="morph", payload="spectral", target=b_path, shared_topology=True)
    with pytest.raises(ValueError, match="front"):
        encode_mesh(A, AF, Params(mode="morph", payload="spectral", target=b_path, shared_topology=True,
                                  morph_front=True))
    e = encode_mesh(A, AF, p)
    with pytest.raises(ValueError, match="front"):
        morph_alpha(e, 0.5, (0.15, None))
    rows = e.spectral["rows"]
    assert len(vertex_qubits(e)) == rows and len(qubit_map(e)["vertices"]) == rows
