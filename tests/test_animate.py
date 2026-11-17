import numpy as np

from tessa_ripple.animate import blend, export_obj_sequence, interpolate, random_mask, run_frames, timeline
from tessa_ripple.bridge import Params, encode_mesh, restore
from tessa_ripple.mesh import cube

V, F = cube(3)


def frames(n=3, **kw):
    p = Params(**kw)
    return run_frames(encode_mesh(V, p), p, n)


def test_same_seed_same_frames_different_seeds_differ():
    a, b = frames(seed=5), frames(seed=5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a[0], a[1])


def test_gain_zero_is_still():
    e = encode_mesh(V, Params())
    for raw in frames():
        assert np.array_equal(blend(V, restore(e, raw), gain=0), V)


def test_interpolation_endpoints_exact():
    a, b = np.zeros((4, 3)), np.ones((4, 3))
    assert np.array_equal(interpolate(a, b, 0), a)
    assert np.array_equal(interpolate(a, b, 1), b)
    seq = timeline([a, b], steps=5)
    assert len(seq) == 7 and np.array_equal(seq[0], a) and np.array_equal(seq[-1], b)
    assert len(timeline([a, b], steps=5, loop=True)) == 12


def test_spread_shrinks_with_shots():
    lo, hi = frames(6, shots=512), frames(6, shots=16384)
    assert hi.std(axis=0).mean() < lo.std(axis=0).mean()


def test_mask_and_export(tmp_path):
    m = random_mask(len(V), 0.5, seed=1)
    assert m.sum() == round(0.5 * len(V))
    out = blend(V, V + 1, gain=1, mask=m)
    assert np.array_equal(out[m == 0], V[m == 0])
    paths = export_obj_sequence(tmp_path, [V, V + 1], F)
    assert [p.name for p in paths] == ["frame_000.obj", "frame_001.obj"]
