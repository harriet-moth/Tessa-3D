"""The quantum side of the add-on: numpy in, numpy out, no bpy.

``run`` is called from a worker thread (see ops.py), so it must never touch Blender
data. It follows the desktop app (``tessa_ripple.app``) step for step: normalise the
mesh as ``mesh.load_obj`` does, encode once (cached), read ``frames`` readouts, decode
each, and return positions in the object's own coordinates.

Blender is Z-up, like the app's internal frame, so vertices go in without an axis turn.
"""

import hashlib
import os
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

# Inside Blender, sys.executable is Blender itself: qiskit must never spawn workers.
os.environ.setdefault("QISKIT_PARALLEL", "FALSE")


class Cancelled(Exception):
    """Raised between readouts when the user cancels a run."""


@dataclass
class Job:
    """Everything one run needs, copied out of Blender on the main thread.

    Attributes:
        verts: (N, 3) mesh A in object coordinates.
        faces: A's polygons (lists of vertex indices).
        params: Keyword arguments for ``bridge.Params``.
        frames: Number of readouts (seeds seed, seed + 1, ...).
        drr: Dynamic range recovery method (``bridge.DRR_METHODS``).
        interference: Morph: interference readout instead of the basis sweep.
        hold_t: Morph: every frame at ``morph_t``; else t runs 0 -> 1 over the frames.
        a_dirs_only: Morph, radial payload: keep A's directions.
        b_verts: Morph: (M, 3) mesh B in A's object coordinates.
        b_faces: Morph: B's polygons.
    """
    verts: np.ndarray
    faces: list
    params: dict
    frames: int = 1
    drr: str = "none"
    interference: bool = False
    hold_t: bool = False
    a_dirs_only: bool = False
    b_verts: np.ndarray | None = None
    b_faces: list | None = None


_cache = {}  # the last encoding: "key", "e", "ie", "match"


def normalise(verts):
    """Centre on the bounding box and scale the largest coordinate to 1 (as load_obj).

    Args:
        verts: (N, 3) positions.

    Returns:
        (verts, centre, scale): normalised (N, 3) and what undoes it.
    """
    v = np.asarray(verts, float)
    centre = (v.min(0) + v.max(0)) / 2
    scale = float(np.abs(v - centre).max()) or 1.0
    return (v - centre) / scale, centre, scale


def prepare():
    """Import qiskit on the calling thread: call it on Blender's main thread before a run.

    qiskit 2.6's compiled core (``_accelerate``, with its own allocator) sets itself
    up on whichever thread imports it first. If that is a worker thread, the next
    worker segfaults once the first has exited (reproduced outside Blender too:
    any second short-lived thread that builds a ``GenericBackendV2``). Imported on
    the main thread, which never exits, every later worker thread is safe. The first
    call blocks for about a second; later calls are free.
    """
    import qiskit  # noqa: F401


def warm():
    """Load everything a first run needs, so that run starts as fast as later ones.

    Importing the core (which pulls in Tessa, scipy and qiskit-aer) and starting
    Aer's first simulation cost several seconds, which otherwise land in the first run.
    Call it from a worker thread *after* :func:`prepare` has run on the main thread.

    Raises:
        ImportError: A bundled wheel is missing (the caller ignores it: the first run
        then reports the same error to the user).
    """
    from .core import animate, bridge, interfere  # noqa: F401  (the imports are the point)
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()
    AerSimulator().run(qc, shots=1).result()


def _digest(verts, faces) -> str:
    """Hash of a mesh's positions and faces, for the encoding cache.

    Args:
        verts: (N, 3) positions.
        faces: Polygon vertex-index lists.

    Returns:
        str: Hex digest.
    """
    h = hashlib.sha1(np.ascontiguousarray(verts, float).tobytes())
    h.update(repr([tuple(f) for f in faces]).encode())
    return h.hexdigest()


def clear_cache():
    """Forget the cached encoding, so the next run encodes afresh."""
    _cache.clear()


def _encode(job: Job, v, p, interf_on: bool, progress):
    """Encode A (and match B), reusing the last encoding when nothing changed.

    Args:
        job: The inputs.
        v: (N, 3) A, normalised.
        p: ``bridge.Params`` (spectral k already capped).
        interf_on: Encode for the interference readout.
        progress: ``progress(text, fraction)`` status callback.

    Returns:
        tuple: ``(e, ie, match)``: ``bridge.Encoded``, ``interfere.Interfered`` (or
        None) and ``morph.correspond``'s match (or None).

    Raises:
        ValueError: Morph without B, or settings Tessa cannot encode.
    """
    from .core import bridge, interfere as interf
    from .core.mesh import load_decimated, write_obj
    from .core.morph import correspond

    morph = p.mode == "morph"
    b_key = _digest(job.b_verts, job.b_faces) if morph else None
    key = (_digest(job.verts, job.faces), b_key, p.encoder_key, interf_on)
    if _cache.get("key") == key:
        return _cache["e"], _cache["ie"], _cache["match"]

    target = match = None
    if morph:
        if job.b_verts is None or not len(job.b_verts):
            raise ValueError("Morph: pick a mesh object as B")
        progress("Matching B to A…", 0.0)
        with tempfile.TemporaryDirectory() as tmp:  # correspond reads B from an OBJ path
            path = write_obj(Path(tmp) / "b.obj", job.b_verts, job.b_faces)
            match = correspond(v, job.faces, str(path), p.shared_topology)
        load_decimated.cache_clear()  # it keys on the temporary path: free B's copy
        target = match
    progress("Encoding…", 0.0)
    if interf_on:  # A and B in the two slots of each selector, one field
        ie = interf.encode_interfere(v, job.faces, p, target)
        e = ie.e
    else:
        ie, e = None, bridge.encode_mesh(v, job.faces, p, target)
    _cache.clear()
    _cache.update(key=key, e=e, ie=ie, match=match)
    return e, ie, match


def run(job: Job, progress=lambda text, frac: None, cancelled=lambda: False) -> dict:
    """Encode (or reuse) and read ``job.frames`` readouts.

    Args:
        job: The inputs.
        progress: ``progress(text, fraction)`` status callback (called from this thread).
        cancelled: Returns True to stop before the next readout.

    Returns:
        dict: ``decoded`` and ``refs`` (lists of (N, 3) positions in object coordinates;
        a ref is the noiseless frame each readout is compared against), ``ts`` (morph t
        per frame or None), ``errors`` (mean error per frame, fraction of the mesh's
        size), ``vertex_error`` ((N,) mean deviation per vertex, same units), ``stats``
        (Encoded.stats), ``kept`` (interference: share of shots kept per frame),
        ``match`` (morph: method, distance, coverage), ``vertex_qubit`` ((N,) data
        qubit per vertex; None for spectral, whose qubits hold modes), ``delta``
        (morph: (N,) |B - A| per vertex, fraction of the mesh's size; else None),
        ``front_source`` (morph front: the source vertex used; else None),
        ``spectral`` (spectral payload: k, rows and truncation errors; else None)
        and ``seconds``.

    Raises:
        ValueError: Settings Tessa cannot encode (shown to the user as is).
        Cancelled: The user cancelled.
    """
    from .core import animate, bridge, interfere as interf

    t0 = time.time()
    v, centre, scale = normalise(job.verts)
    p = bridge.Params(**job.params)
    morph = p.mode == "morph"
    interf_on = morph and job.interference
    if morph and interf_on and p.morph_front:
        p = replace(p, morph_front=False)  # the front turns the basis sweep, not the selectors
    p = bridge.cap_spectral(p, len(v), interf_on)  # spectral: k at most what one run holds

    e, ie, match = _encode(job, v, p, interf_on, progress)

    n = max(1, int(job.frames))
    ts = None
    if morph:
        ts = [p.morph_t] * n if job.hold_t or n == 1 else list(np.linspace(0, 1, n))
    runs = f" ({e.runs} circuits each)" if e.runs > 1 else ""

    def step(i, total):
        if cancelled():
            raise Cancelled
        progress(f"Readout {i + 1}/{total}{runs}…", i / total)

    a = job.a_dirs_only
    kept = None
    if interf_on:
        raw, kept = interf.run_frames(ie, p, ts, progress=step)
        decoded = [interf.frame(ie, r, t, job.drr, a) for r, t in zip(raw, ts)]
        refs = [interf.frame(ie, interf.ideal(ie, t), t, "none", a) for t in ts]  # ideal caches per t
    elif morph:
        raw = animate.run_frames(e, p, n, progress=step, ts=ts)
        decoded = [bridge.morph_frame(e, r, t, job.drr, a, p.front) for r, t in zip(raw, ts)]
        refs = [bridge.morph_expected(e, t, a, p.front) for t in ts]
    else:
        raw = animate.run_frames(e, p, n, progress=step)
        decoded = [bridge.restore(e, r, job.drr) for r in raw]
        refs = [bridge.encoded_shape(e)] * n
    progress("Decoding…", 1.0)

    size = np.ptp(v) or 1.0
    errors = [bridge.mesh_error(r, d) for r, d in zip(refs, decoded)]
    vertex_error = np.mean([np.linalg.norm(d - r, axis=1) for r, d in zip(refs, decoded)], axis=0) / size
    back = lambda x: np.asarray(x) * scale + centre  # noqa: E731
    return {
        "decoded": [back(d) for d in decoded],
        "refs": [back(r) for r in refs],
        "ts": ts,
        "errors": errors,
        "vertex_error": vertex_error,
        "stats": e.stats,
        "kept": None if kept is None else [float(k) for k in kept],
        "match": None if match is None else {k: match[k] for k in ("method", "distance", "coverage")},
        "vertex_qubit": None if e.payload == "spectral" else bridge.vertex_qubits(e),
        "delta": None if match is None else np.linalg.norm(match["target"] - v, axis=1) / size,
        "front_source": None if p.front is None else int(bridge.front_distances(e, p.front_source)[0]),
        "spectral": e.spectral,
        "seconds": time.time() - t0,
    }
