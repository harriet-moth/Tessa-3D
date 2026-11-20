"""Tests of the web API (api/tessa.py): uploaded OBJ meshes, the mesh action and morph."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("tessa_api", ROOT / "api" / "tessa.py")
api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(api)

pytestmark = pytest.mark.filterwarnings("ignore")

# A regular octahedron: 6 vertices, 8 faces.
OCTA = "\n".join(["v 1 0 0", "v -1 0 0", "v 0 1 0", "v 0 -1 0", "v 0 0 1", "v 0 0 -1",
                  "f 1 3 5", "f 3 2 5", "f 2 4 5", "f 4 1 5", "f 3 1 6", "f 2 3 6", "f 4 2 6", "f 1 4 6"])


def test_mesh_action_returns_the_preset_unencoded():
    """The mesh action returns the named preset's vertices and faces."""
    d = api.mesh({"shape": "cube", "resolution": 0})
    v, f = api.PRESETS["cube"]["3/edge"]()
    assert np.allclose(d["verts"], v, atol=1e-5) and len(d["faces"]) == len(f)


def test_uploaded_obj_runs_and_reuses_its_file():
    """An uploaded OBJ runs like a preset; the same text maps to the same saved file."""
    r = api.run({"obj": OCTA, "resolution": 2, "frames": 2})
    assert len(r["verts"]) == 6 and len(r["frames"]) == 2
    assert api.mesh_path({"obj": OCTA}) == api.mesh_path({"obj": OCTA})


def test_uploaded_obj_over_the_limit_is_refused(monkeypatch):
    """Text over MAX_OBJ characters is a ValueError (a 400), not a crash."""
    monkeypatch.setattr(api, "MAX_OBJ", 10)
    with pytest.raises(ValueError, match="too large"):
        api.mesh({"obj": OCTA})


def test_morph_frames_span_t_and_track_the_noiseless_morph():
    """A morph run reads frames along t = 0 -> 1, each close to its noiseless reference."""
    r = api.run({"shape": "blob", "mode": "morph", "shape_b": "sphere", "payload": "radial",
                 "frames": 5, "shots": 65536})
    assert r["ts"] == pytest.approx([0, 0.25, 0.5, 0.75, 1])
    assert len(r["frames"]) == len(r["refs"]) == 5
    errs = [np.abs(np.array(f) - np.array(ref)).max() for f, ref in zip(r["frames"], r["refs"])]
    assert max(errs) < 0.05
    assert np.allclose(r["refs"][0], r["verts"], atol=1e-4)  # t = 0 is A


def test_morph_single_run_reads_at_morph_t_with_an_uploaded_b():
    """One morph readout sits at ``morph_t``; B may be an uploaded OBJ."""
    r = api.run({"shape": "blob", "mode": "morph", "obj_b": OCTA, "morph_t": 0.3})
    assert r["ts"] == [0.3] and r["match"]["method"] == "closest point"


def test_morph_without_b_and_fixed_mesh_actions_are_refused():
    """A morph needs B; edges and curves are not available in morph mode."""
    with pytest.raises(ValueError, match="second mesh"):
        api.run({"shape": "blob", "mode": "morph"})
    for action in (api.edges, api.curves):
        with pytest.raises(ValueError, match="morph mode"):
            action({"shape": "blob", "mode": "morph", "shape_b": "sphere"})
