"""Headless smoke tests of the desktop app's morph readouts (matplotlib's Agg backend)."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from tessa_ripple import app as appmod  # noqa: E402
from tessa_ripple.bridge import mesh_error  # noqa: E402
from tessa_ripple.mesh import PRESETS, write_obj  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


def pick(app, name, label):
    """Select ``label`` on the radio ``name``, firing its callbacks as a click would.

    Args:
        app: The ``App``.
        name: Key of a RadioButtons in ``app.w``.
        label: The option to select.
    """
    w = app.w[name]
    w.set_active([t.get_text() for t in w.labels].index(label))


@pytest.fixture
def app(tmp_path):
    """The app on the low-resolution blob, morphing to a stretched copy (radial, 1,024 shots).

    Args:
        tmp_path: pytest's temporary folder, for B's OBJ.

    Returns:
        App: Encoded with the basis sweep, ready to toggle the interference readout.
    """
    v, f = next(iter(PRESETS["blob"].values()))()  # the Shape "blob" at resolution low
    b = v * [1.2, 1.0, 0.8]  # B: the blob stretched, same topology
    a = appmod.App([])
    a.b_path = str(write_obj(tmp_path / "b.obj", b, f))
    pick(a, "Shape", "blob")
    pick(a, "Mode", "morph A→B")
    pick(a, "Payload", "radial")
    a.w["Morph"].set_active(0)  # Shared topology
    a.w["Shots (log2)"].set_val(10)
    return a


def test_interference_readout_toggles_on_and_off(app):
    assert app.morphing and not app.interfering
    app.w["Morph"].set_active(3)  # Interference readout
    assert app.interfering and app.error is None
    assert app.morph_note.get_text().startswith("Interference")
    assert "stars: selector 0 = A, 1 = B" in app.info.get_text()
    app.w["Frames"].set_val(2)
    app.run_frames()
    assert app.kept is not None and len(app.decoded) == 2
    assert "Morph interference" in app.ax.get_title()
    e = app.encoded
    assert mesh_error(e.verts, app.decoded[0]) < 0.03  # t = 0: A
    assert mesh_error(e.target, app.decoded[1]) < 0.03  # t = 1: B
    for ref, t in zip(app.refs, app.raw_ts):  # references are the exact interference frames
        assert np.allclose(ref, appmod.interf.frame(app.ie, appmod.interf.ideal(app.ie, t), t))
    app.w["Morph"].set_active(3)  # off again: the basis sweep
    assert not app.interfering and app.raw is None
    assert app.morph_note.get_text().startswith("Basis sweep")
