import numpy as np
import pytest

from tessa_ripple.mesh import subdivide, _normal, bfs_order, blob, cube, edges, load_obj, uv_sphere, write_obj


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


def test_edges_and_bfs_order():
    v, f = blob(12, 8)
    E = edges(f)
    assert len(E) == len({tuple(e) for e in E}) and (E[:, 0] < E[:, 1]).all()
    order = bfs_order(len(v), f)
    assert sorted(order) == list(range(len(v)))


def test_load_obj_round_trip(tmp_path):
    v, f = cube(3)
    lv, lf = load_obj(write_obj(tmp_path / "c.obj", v, f))
    assert np.allclose(lv, v) and lf == [list(x) for x in f]


def _edge_use(faces):
    from collections import Counter
    return Counter(tuple(sorted((f[i], f[i - 1]))) for f in faces for i in range(len(f)))


def test_decimate_halves_suzanne_without_tearing_or_losing_her_shape():
    from tessa_ripple.mesh import decimate
    v, f = load_obj("tessa_ripple/suzanne.obj")
    dv, df = decimate(v, f)
    assert len(dv) == round(len(v) / 2)
    assert set(np.concatenate(df)) == set(range(len(dv)))  # no orphan vertices
    use, use0 = _edge_use(df), _edge_use(f)
    assert max(use.values()) <= 2  # still manifold
    assert sum(c == 1 for c in use.values()) == sum(c == 1 for c in use0.values())  # holes untouched
    assert np.allclose(np.ptp(dv, 0), np.ptp(v, 0), atol=0.1)



def test_decimating_twice_quarters_suzanne():
    from tessa_ripple.mesh import decimate
    v, f = load_obj("tessa_ripple/suzanne.obj")
    lv, lf = decimate(*decimate(v, f))
    assert len(lv) == round(round(len(v) / 2) / 2) and max(_edge_use(lf).values()) <= 2


def test_grow_patches_are_compact_and_cover_the_mesh():
    from tessa_ripple.mesh import grow_patches
    v, f = load_obj("tessa_ripple/suzanne.obj")
    sizes = [16] * 30
    patches, rest = grow_patches(v, f, sizes)
    every = np.concatenate(patches + [rest])
    assert sorted(every) == list(range(len(v))) and [len(p) for p in patches] == sizes
    E = edges(f)
    edge = np.linalg.norm(v[E[:, 0]] - v[E[:, 1]], axis=1).mean()
    radius = np.mean([np.linalg.norm(v[p] - v[p].mean(0), axis=1).mean() for p in patches])
    assert radius < 2.5 * edge  # BFS arcs were about 4 edge lengths


@pytest.mark.parametrize("n", [2, 3, 4])
def test_cube_faces_wind_outward(n):
    v, f = cube(n)
    for q in f:
        p = v[q]
        assert _normal(p) @ p.mean(0) > 0


# --- custom OBJ files -----------------------------------------------------------------
def test_load_obj_reads_latin1_and_line_continuations(tmp_path):
    from tessa_ripple.mesh import load_obj

    body = "# caf\xe9\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 \\\n 3\nf 1/1/1 2/2/2 4/4/4\n"
    (tmp_path / "m.obj").write_bytes(body.encode("latin-1"))
    v, f = load_obj(tmp_path / "m.obj")
    assert v.shape == (4, 3) and f == [[0, 1, 2], [0, 1, 3]]


@pytest.mark.parametrize("body, msg", [("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n", "names vertex 9"),
                                       ("v 0 0 zero\n", "line 1"), ("# empty\n", "no vertices")])
def test_load_obj_reports_unusable_files_clearly(tmp_path, body, msg):
    from tessa_ripple.mesh import load_obj

    (tmp_path / "bad.obj").write_text(body)
    with pytest.raises(ValueError, match=msg):
        load_obj(tmp_path / "bad.obj")


def test_a_large_uploaded_obj_is_decimated_to_fit(tmp_path):
    from tessa_ripple.app import FIT_VERTICES, fit_passes, obj_presets
    from tessa_ripple.mesh import uv_sphere, write_obj

    big = write_obj(tmp_path / "big.obj", *uv_sphere(80, 60))  # 4,722 vertices
    assert fit_passes(str(big)) > 0
    assert len(obj_presets(str(big))["high"]()[0]) <= FIT_VERTICES
    assert fit_passes("tessa_ripple/suzanne.obj") == 0  # fits as it is: unchanged


def test_subdivide_splits_triangles_into_four_without_changing_the_shape():
    """Each triangle becomes four; the originals stay; midpoints are shared; area is kept."""
    v, f = cube(3)
    V, F = subdivide(v, f)
    tris = sum(len(x) - 2 for x in f)
    assert len(F) == 4 * tris and np.allclose(V[:len(v)], v)
    edges = {tuple(sorted((x[k], x[(k + 1) % len(x)]))) for x in [[t[0], t[i], t[i + 1]] for t in f for i in range(1, len(t) - 1)] for k in range(3)}
    assert len(V) == len(v) + len(edges)  # one shared midpoint per edge

    def area(V, F):
        return sum(0.5 * np.linalg.norm(np.cross(V[b] - V[a], V[c] - V[a])) for a, b, c in F)
    assert area(V, F) == pytest.approx(area(v, [[t[0], t[i], t[i + 1]] for t in f for i in range(1, len(t) - 1)]))
