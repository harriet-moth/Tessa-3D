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


def test_front_runs_shows_its_colour_and_stays_off_with_interference(app):
    """Front on: params carry it, Run frames reads it and switches Colour to front."""
    app.w["Morph"].set_active(4)  # Front
    p = app.params()
    assert p.front == (0.15, None) and app.morph_note.get_text().startswith("Front")
    assert "front: vertex" in app.info.get_text() and "(auto)" in app.src_label.get_text()
    app.w["Frames"].set_val(3)
    app.run_frames()
    assert app.val("Colour") == "front" and "Morph front" in app.ax.get_title()
    assert len(app.front_seq) == len(app.seq) and np.allclose(app.front_seq[0], 0)  # leads in from A
    assert np.allclose(app.front_weights(1.0, app.raw_front), 1)  # t = 1: every vertex at B
    assert max(mesh_error(r, d) for r, d in zip(app.refs, app.decoded)) < 0.05
    app.w["Morph"].set_active(3)  # Interference readout: the front needs the basis sweep
    assert app.params().front is None and app.src_label.get_text() == "front: basis sweep only"
    assert not app.w["Front width"].get_active()


def test_front_source_is_picked_by_clicking_a_vertex(app):
    """Pick source, then a click (not a drag) on a vertex makes it the source; Auto source resets."""
    from types import SimpleNamespace
    from mpl_toolkits.mplot3d import proj3d
    app.w["Morph"].set_active(4)  # Front
    app.fig.canvas.draw()
    v = app.seq[app.idx]
    xs, ys, _ = proj3d.proj_transform(*v.T, app.ax.get_proj())
    pix = app.ax.transData.transform(np.column_stack([xs, ys]))
    i = int(np.argmax(pix[:, 0]))  # the right-most vertex: nothing in front of it on screen
    click = SimpleNamespace(x=pix[i, 0], y=pix[i, 1], inaxes=app.ax)
    app.toggle_picking()
    assert app.picking and app.src_label.get_text() == "click a vertex"
    app.on_press(SimpleNamespace(x=pix[i, 0] - 40, y=pix[i, 1], inaxes=app.ax))
    app.on_release(click)  # moved 40 px: a drag, so still picking
    assert app.picking and app.front_source is None
    app.on_press(click)
    app.on_release(click)
    assert not app.picking and app.front_source == i and app.params().front_source == i
    assert app.src_label.get_text() == f"vertex {i}"
    app.auto_front_source()
    assert app.front_source is None and app.params().front_source is None


def test_front_source_resets_on_another_mesh(app):
    """A picked source belongs to its mesh: changing Resolution drops it."""
    app.w["Morph"].set_active(4)
    app.front_source = 3
    pick(app, "Resolution", "mid")
    assert app.front_source is None


def test_spectral_payload_encodes_runs_and_guards_vertex_features(app):
    pick(app, "Payload", "spectral")
    app.w["Morph"].set_active(4)  # Front: refused for spectral, so it stays off
    assert app.error is None and app.encoded.payload == "spectral" and not app.params().morph_front
    assert "spectral:" in app.info.get_text() and "not with spectral" in app.src_label.get_text()
    app.w["Morph"].set_active(4)
    app.w["Frames"].set_val(2)
    app.run_frames()
    assert len(app.decoded) == 2 and app.err < 0.05
    app.w["Morph"].set_active(3)  # interference readout with spectral
    assert app.interfering and app.error is None and app.encoded.spectral is not None
    app.w["Morph"].set_active(3)
    pick(app, "Mode", "shape θ (Z)")
    app.w["Spectral k"].set_val(appmod.SPECTRAL_K_MAX)  # past the cap (N - 1 here): encodes the cap
    assert app.error is None and app.encoded.spectral["k"] <= len(app.verts) - 1
    app.w["Spectral k"].set_val(8)
    assert app.encoded.spectral["k"] == 8 and app.encoded.slots.shape == (8, 3)
    pick(app, "Colour", "qubit")
    assert "spectral" in app.colour_label
    app.w["Toggles"].set_active(3)  # qubit map: a vertex hover focuses nothing, a data qubit lists modes
    app.set_focus(vertex=0)
    assert app.focus is None
    g = int(app.qmap["vertices"][0][0][0])
    app.set_focus(group=g)
    assert app.focus["mates"] == [] and "modes 1" in app.focus_text()
    app.run()
    assert np.isfinite(app.err)
