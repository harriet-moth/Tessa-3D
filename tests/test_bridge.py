import numpy as np
import pytest

from tessa_ripple.animate import blend
from tessa_ripple.bridge import Params, encode_mesh, fit_lattice, read_mesh
from tessa_ripple.mesh import cube, uv_sphere


def test_fit_lattice_holds_values():
    for n in (78, 168, 294, 342):
        assert fit_lattice(n).capacity >= n


def test_capacity_error_is_clear():
    with pytest.raises(ValueError, match="exceed"):
        fit_lattice(10_000)


def test_noiseless_round_trip_within_2_percent():
    v, _ = cube(3)
    p = Params(shots=20_000)
    out = read_mesh(encode_mesh(v, p), p)
    assert out.shape == v.shape
    assert np.abs(out - v).mean() / np.ptp(v) < 0.02


def test_pair_mode_round_trip():
    v, _ = uv_sphere(8, 6)
    p = Params(mode="pair", shots=20_000)
    e = encode_mesh(v, p)
    assert e.enc.mode == "pair"
    assert np.abs(read_mesh(e, p) - v).mean() / np.ptp(v) < 0.05


def test_gain_zero_returns_original_exactly():
    v, _ = cube(3)
    p = Params()
    assert np.array_equal(blend(v, read_mesh(encode_mesh(v, p), p), gain=0), v)
