"""Tests of the web API (api/tessa.py): uploaded OBJ meshes, the mesh action and morph."""

import importlib.util
import time
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
    """The mesh action returns the named preset's vertices and faces (medium: as is)."""
    d = api.mesh({"shape": "cube", "resolution": 1})
    v, f = api.PRESETS["cube"]["3/edge"]()
    assert np.allclose(d["verts"], v, atol=1e-5) and len(d["faces"]) == len(f)


def test_uploaded_obj_runs_and_reuses_its_file():
    """An uploaded OBJ runs like a preset; the same text maps to the same saved file."""
    r = api.run({"obj": OCTA, "resolution": 1, "frames": 2})
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


def test_morph_hold_t_reads_every_frame_at_morph_t():
    """``hold_t`` reads every frame at ``morph_t`` instead of running t 0 -> 1."""
    r = api.run({"shape": "blob", "mode": "morph", "shape_b": "sphere", "frames": 3, "hold_t": True, "morph_t": 0.4})
    assert r["ts"] == pytest.approx([0.4] * 3) and r["options"]["hold_t"]


def test_morph_uses_one_clamp_for_both_fields():
    """Morph reports the one clamp θ and φ share: θ's, or Tessa's pair default (0.2π)."""
    base = {"shape": "blob", "mode": "morph", "shape_b": "sphere"}
    assert api.run(base)["clamp"] == pytest.approx(0.2)
    assert api.run({**base, "clamp": 0.3 * np.pi, "clamp_b": 0.1 * np.pi})["clamp"] == pytest.approx(0.3)


def test_morph_interference_readout_reports_its_kept_shots():
    """The interference readout runs on the web too and says how many shots it kept."""
    r = api.run({"shape": "blob", "mode": "morph", "shape_b": "sphere", "frames": 2, "interference": True,
                 "shots": 65536})
    assert r["options"]["interference"] and len(r["kept"]) == len(r["frames"]) == 2
    assert all(0 < k <= 1 for k in r["kept"]) and r["error"] < 0.05


def test_shared_topology_pairs_by_index_and_refuses_other_meshes():
    """Shared topology pairs B with A by vertex index, and refuses a B of another size."""
    v = api.mesh({"shape": "blob"})
    b = "\n".join([f"v {x * 1.2} {z} {-y}" for x, y, z in v["verts"]]
                  + ["f " + " ".join(str(i + 1) for i in f) for f in v["faces"]])
    r = api.run({"shape": "blob", "mode": "morph", "obj_b": b, "shared_topology": True})
    assert r["match"]["method"] == "by index"
    with pytest.raises(ValueError, match="Shared topology"):
        api.run({"shape": "blob", "mode": "morph", "shape_b": "cube", "shared_topology": True})


def _slow_readouts(monkeypatch, seconds):
    """Make every readout take ``seconds`` longer, like Vercel's slower CPU (each frame
    of a ``decode_frames`` batch too)."""
    for name in ("decode_raw", "tilted_angles", "decode_frames"):
        real = getattr(api.bridge, name)

        def slow(*a, _real=real, _name=name, **k):
            time.sleep(seconds * (a[2] if _name == "decode_frames" else 1))
            return _real(*a, **k)
        monkeypatch.setattr(api.bridge, name, slow)


def test_run_frames_stop_before_the_deadline(monkeypatch):
    """Frames stop before one would overrun the request's deadline (no 504)."""
    _slow_readouts(monkeypatch, 0.5)
    start = time.time()
    r = api.run({"shape": "blob", "frames": 30, "_deadline": start + 5.0})
    assert 1 <= len(r["frames"]) < 30
    assert time.time() - start < 5.0 + 0.6  # slack for the reply and a busy test machine


def test_morph_frames_fit_the_deadline_and_still_span_t(monkeypatch):
    """A slow morph reads fewer frames, still from t = 0 to 1, and ends by the deadline."""
    _slow_readouts(monkeypatch, 0.3)
    start = time.time()
    r = api.run({"shape": "blob", "mode": "morph", "shape_b": "sphere", "frames": 30, "_deadline": start + 8.0})
    assert 2 <= len(r["frames"]) < 30 and r["ts"][0] == 0 and r["ts"][-1] == pytest.approx(1)
    assert time.time() - start < 8.0 + 1.0  # slack for the references, the reply and a busy test machine


@pytest.mark.parametrize("action, req, words", [
    ("curves", {"shape": "blob", "backend": "flat noise"}, "Curves with the flat noise"),
    ("curves", {"shape": "blob", "backend": "phoenix"}, "Curves with the phoenix"),
    ("edges", {"shape": "blob", "backend": "fez"}, "Edge readouts on the fez"),
    ("run", {"shape": "blob", "backend": "phoenix", "mode": "morph", "shape_b": "sphere", "interference": True},
     "interference readout with phoenix"),
])
def test_work_too_slow_for_the_web_is_refused_up_front(action, req, words):
    """Work that cannot fit in the function's time limit is refused at once, with advice."""
    with pytest.raises(ValueError, match=words):
        api.ACTIONS[action](req)


def test_fast_enough_work_is_not_refused():
    """Edges with flat noise and the interference readout without noise still run."""
    assert len(api.edges({"shape": "cube", "backend": "flat noise"})["edges"]) > 0
    assert api.run({"shape": "blob", "mode": "morph", "shape_b": "sphere", "interference": True})["kept"]


def test_resolution_decimates_keeps_or_subdivides():
    """Low decimates, medium is the mesh as is, high subdivides it (same shape, more vertices)."""
    low, med, high = (api.mesh({"shape": "sphere", "resolution": r}) for r in (0, 1, 2))
    assert len(low["verts"]) < len(med["verts"]) < len(high["verts"])
    v, f = api.PRESETS["sphere"]["8x6"]()
    assert np.allclose(med["verts"], v, atol=1e-5)
    assert np.allclose(high["verts"][:len(v)], v, atol=1e-5)  # the originals stay, midpoints are added
    assert api.mesh({"obj": OCTA})["verts"] == api.mesh({"obj": OCTA, "resolution": 1})["verts"]  # default: medium


FRONT = {"shape": "blob", "mode": "morph", "shape_b": "sphere", "morph_front": True}


def test_morph_front_reads_each_vertex_at_its_qubits_angle():
    """The front runs from A (every weight 0) to B (every weight 1), turning patch by patch."""
    r = api.run({**FRONT, "frames": 3, "shots": 65536})
    f = r["front"]
    assert r["ts"] == pytest.approx([0, 0.5, 1]) and f["width"] == pytest.approx(0.15)
    assert len(f["weights"]) == len(f["smooth"]) == 3 and len(f["weights"][0]) == len(r["verts"])
    assert np.allclose(f["weights"][0], 0) and np.allclose(f["weights"][2], 1)
    mid = np.array(f["weights"][1])
    assert mid.min() < 0.5 < mid.max()  # a front, not the uniform sweep's 0.5 everywhere
    assert f["weights"][1][f["source"]] == max(f["weights"][1])  # the source turns first
    assert np.allclose(r["refs"][0], r["verts"], atol=1e-4)  # t = 0 is A


def test_morph_front_takes_a_source_and_width():
    """A chosen source and width come back as given (width clamped to [0.02, 1])."""
    r = api.run({**FRONT, "front_source": 3, "front_width": 5, "morph_t": 0.3})
    assert r["front"]["source"] == 3 and r["front"]["width"] == 1.0


def test_morph_front_is_refused_with_interference_or_a_bad_source():
    """The front needs the basis sweep, and its source must be a vertex of A."""
    with pytest.raises(ValueError, match="interference"):
        api.run({**FRONT, "interference": True})
    with pytest.raises(ValueError, match="not a vertex"):
        api.run({**FRONT, "front_source": 10**6})


def test_front_flag_is_ignored_outside_morph():
    """``morph_front`` on a plain run changes nothing and reports no front."""
    assert "front" not in api.run({"shape": "blob", "morph_front": True})


def test_spectral_payload_runs_with_its_truncated_reference_and_caps_k():
    """Spectral: θ frames come with the truncated mesh as reference; k past one run is capped; 0 is auto."""
    r = api.run({"shape": "blob", "payload": "spectral", "frames": 2, "shots": 65536})
    sp = r["spectral"]
    assert sp["k"] >= 1 and len(r["map"]["vertices"]) == len(r["qubits"]) == sp["rows"]
    assert len(r["refs"]) == 2 and r["error"] < 0.02
    assert not np.allclose(r["refs"][0], r["verts"])  # truncated, not the mesh itself
    n = len(r["verts"])
    cap = api.bridge.spectral_cap(n, api.bridge.Params(payload="spectral"))  # one run: 264 modes here
    assert cap < n - 1
    assert api.run({"shape": "blob", "payload": "spectral", "spectral_k": 10_000})["spectral"]["k"] == cap
    assert api.run({"shape": "blob", "payload": "spectral", "spectral_k": 0})["spectral"]["k"] == sp["k"]
    assert api.run({"shape": "blob", "payload": "radial"})["spectral"] is None


def test_spectral_morph_runs_and_refuses_the_front():
    """Spectral morphs through the sweep and the interference readout; the front is refused."""
    base = {"shape": "blob", "mode": "morph", "shape_b": "sphere", "payload": "spectral", "shots": 65536}
    for extra in ({}, {"interference": True}):
        r = api.run({**base, **extra, "frames": 3})
        assert r["spectral"]["rows"] == r["spectral"]["k"] + 1 and r["error"] < 0.05
        assert len(r["refs"]) == len(r["frames"]) == 3
    with pytest.raises(ValueError, match="front"):
        api.run({**base, "morph_front": True})


def test_run_reports_readout_shots_and_pilots():
    r = api.run({"shape": "blob", "readout": "bloch", "drr": "pilots", "shots": 4096, "gain_shots": 2})
    assert r["readout"] == "bloch" and r["shots"] == 1024  # Material's gain made by shots / gain²
    assert r["stats"]["values"] == len(r["verts"])
    with pytest.raises(ValueError, match="unknown readout"):
        api.params_of({"readout": "y"})
