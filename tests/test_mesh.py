import numpy as np
import pytest

from tessa_ripple.mesh import cube, uv_sphere, write_obj


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_cube(n):
    v, f = cube(n)
    assert len(v) == 6 * n * n - 12 * n + 8
    assert len(np.unique(v.round(9), axis=0)) == len(v)
    assert all(0 <= i < len(v) for face in f for i in face)


@pytest.mark.parametrize("s,r", [(8, 6), (12, 8), (16, 8)])
def test_sphere(s, r):
    v, f = uv_sphere(s, r)
    assert len(v) == s * (r - 1) + 2
    assert len(np.unique(v.round(9), axis=0)) == len(v)
    assert all(0 <= i < len(v) for face in f for i in face)
    assert np.allclose(np.linalg.norm(v, axis=1), 1)


def test_write_obj(tmp_path):
    v, f = cube(2)
    text = write_obj(tmp_path / "c.obj", v, f).read_text().splitlines()
    assert sum(line.startswith("v ") for line in text) == 8
    assert sum(line.startswith("f ") for line in text) == 6
