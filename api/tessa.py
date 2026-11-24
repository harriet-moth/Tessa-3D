"""Vercel Python function: every Tessa computation the web app needs, at POST /api/tessa.

The request body is JSON with an ``action`` (mesh / run / edges / resources / curves) plus
the mesh and Params settings. The mesh is a named shape or an uploaded OBJ (``obj``);
a morph names its B mesh the same way (``shape_b`` or ``obj_b``). All quantum work
goes through tessa_ripple.bridge, the same code the desktop app uses.
"""

import hashlib
import json
import sys
import tempfile
import time

_IMPORT_START = time.time()  # the first request after a cold start also pays for the imports below

from dataclasses import replace  # noqa: E402
from http.server import BaseHTTPRequestHandler  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tessa_ripple import bridge  # noqa: E402
from tessa_ripple import interfere as interf  # noqa: E402
from tessa_ripple.mesh import PRESETS, decimate, load_decimated, subdivide, write_obj  # noqa: E402
from tessa_ripple.morph import correspond  # noqa: E402

_IMPORT_SECONDS = time.time() - _IMPORT_START
_cold = True  # set until the first request has been served

# OBJ shapes that ship with the app, by name.
OBJ_FILES = {"suzanne": ROOT / "tessa_ripple" / "suzanne.obj",
             "apple": ROOT / "tessa_ripple" / "apple_full.obj"}
# The Params fields the web app sets, with their allowed values (None: any number).
# Morph's ``target`` (a file path) is never taken from a request: B comes from
# ``shape_b`` / ``obj_b`` (:func:`mesh_path`).
CHOICES = {"payload": bridge.PAYLOADS, "layout": ("mesh", "naive"),
           "mode": ("theta", "split", "morph"), "backend": bridge.BACKENDS, "placement": bridge.PLACEMENTS,
           "clamp": None, "clamp_b": None, "compression": None, "quant_bits": None,
           "headroom": None, "shots": None, "noise_strength": None, "seed": None, "spectral_k": None}
MAX_FRAMES = 30
# Vercel stops the function at maxDuration (vercel.json) and answers 504. Each request
# gets a deadline this many seconds after it arrives, leaving room to send the reply;
# the first request after a cold start also loses the import time.
FUNCTION_LIMIT = 300.0  # batched runs of a big mesh can take ~3 minutes
REPLY_MARGIN = 8.0
FRAME_MARGIN = 1.25  # a frame only starts if the slowest frame so far, ×1.25, still fits


def deadline_of(req) -> float:
    """When this request's work must be finished (time.time() seconds).

    Args:
        req: request dict; reads ``_deadline`` (set by the handler; tests may set it).

    Returns:
        float: The deadline.
    """
    return float(req.get("_deadline") or time.time() + FUNCTION_LIMIT - REPLY_MARGIN)


def fits(deadline: float, per: float) -> bool:
    """Whether another readout of about ``per`` seconds finishes before the deadline.

    Args:
        deadline: time.time() deadline.
        per: Slowest readout so far, in seconds.

    Returns:
        bool: True if there is time.
    """
    return time.time() + per * FRAME_MARGIN < deadline


def readouts(read, want: int, deadline: float) -> list:
    """Up to ``want`` frames at one setting, as many as finish before ``deadline``.

    The first call reads one frame, which times a readout; each later call reads as
    many frames as the time left allows at the last call's cost per frame. Frames read
    together share each circuit's simulator job (``bridge.decode_frames``), so after
    the first a frame costs less, and the estimate tightens as it goes. At least one
    frame always runs.

    Args:
        read: ``read(n, k)`` -> n frames, the first being frame k (it seeds by k).
        want: Frames asked for.
        deadline: time.time() by which the readouts must be done.

    Returns:
        list: The frames read, in order.
    """
    out, per = [], None
    while len(out) < want:
        left = want - len(out)
        n = 1 if per is None else int(np.clip((deadline - time.time()) // (per * FRAME_MARGIN), 0, left))
        if n == 0:
            break
        began = time.time()
        out.extend(read(n, len(out)))
        per = (time.time() - began) / n
    return out


def too_slow(req, what: str):
    """Refuse up front work that cannot finish inside the function's time limit.

    Noise simulation dominates: on Vercel's CPU it runs about 9× slower than on a
    laptop. Measured locally: curves take 15–18 s with any noisy backend (about 2–3
    minutes on Vercel), edge readouts 8–10 s on a chip (70–90 s), and one
    interference-readout frame with noise 30–40 s (several minutes).

    Args:
        req: request dict.
        what: "curves", "edges" or "interference".

    Raises:
        ValueError: the request's backend makes it too slow for the web function.
    """
    backend = req.get("backend", "noiseless")
    noisy, chip = backend != "noiseless", backend in bridge.DEVICES
    advice = "switch Backend to noiseless (or flat noise for edges), or use the desktop app."
    if what == "curves" and noisy:
        raise ValueError(f"Curves with the {backend} backend take minutes on the web (15 noisy readouts); {advice}")
    if what == "edges" and chip:
        raise ValueError(f"Edge readouts on the {backend} chip take over a minute on the web; {advice}")
    if what == "interference" and noisy:
        raise ValueError(f"The interference readout with {backend} noise takes minutes per frame on the web; "
                         "switch Backend to noiseless, turn off interference readout, or use the desktop app.")


MAX_OBJ = 3_000_000  # characters of uploaded OBJ text (Vercel caps request bodies at 4.5 MB)
UPLOADS = Path(tempfile.gettempdir()) / "tessa_ripple_obj"  # Vercel's writable /tmp


LOW, MEDIUM, HIGH = 0, 1, 2  # resolutions: decimated once, as is, subdivided once


def resolution_of(req) -> int:
    """The request's ``resolution`` (0 low, 1 medium, 2 high; default medium), clamped.

    Args:
        req: request dict.

    Returns:
        int in [0, 2].
    """
    return int(np.clip(int(req.get("resolution", MEDIUM)), LOW, HIGH))


def mesh_path(req, suffix: str = "") -> Path:
    """An OBJ file for the mesh a request names, so OBJ readers can take a path.

    ``obj<suffix>`` (uploaded OBJ text) wins over ``shape<suffix>``. Uploads are
    saved under their content hash, so the same text reuses one file and
    load_decimated's cache; generated presets are written once per resolution.

    Args:
        req: request dict.
        suffix: "" for mesh A, "_b" for a morph's B.

    Returns:
        Path of an OBJ file.

    Raises:
        KeyError: unknown shape.
        ValueError: uploaded text over MAX_OBJ characters.
    """
    text = req.get("obj" + suffix)
    if text:
        if len(text) > MAX_OBJ:
            raise ValueError(f"OBJ too large ({len(text):,} characters; the limit is {MAX_OBJ:,})")
        path = UPLOADS / f"{hashlib.sha1(text.encode()).hexdigest()}.obj"
        if not path.exists():
            UPLOADS.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return path
    name = req.get("shape" + suffix, "blob")
    if name in OBJ_FILES:
        return OBJ_FILES[name]
    presets = list(PRESETS[name].values())
    k = 0 if not suffix else len(presets) - 1  # A: the base preset; B: the finest, to match against
    path = UPLOADS / f"{name}-{k}.obj"
    if not path.exists():
        write_obj(path, *presets[k]())
    return path


def mesh_of(req):
    """The mesh a request names, at its resolution, with its decimation logs.

    The base mesh is the shape's first preset, or the OBJ (shipped or uploaded
    ``obj``) as loaded. Resolution medium is the base as is; low decimates it once
    (about half the vertices, :func:`mesh.decimate`); high subdivides it once (each
    triangle into four at its edge midpoints, :func:`mesh.subdivide`: the same shape
    described by about four times the vertices).

    Args:
        req: request dict; reads ``obj`` or ``shape`` (default "blob") and
            ``resolution`` (0 low, 1 medium, 2 high; default medium).

    Returns:
        (verts (N, 3), faces, logs, full_count): logs are the decimation passes and
        full_count the base's vertex count, for :func:`morph.correspond` (a subdivided
        mesh has no logs: shared topology then needs a B with its own vertex count).

    Raises:
        KeyError: unknown shape.
        ValueError: an unreadable OBJ.
    """
    name, res = req.get("shape", "blob"), resolution_of(req)
    if not req.get("obj") and name in PRESETS:
        v, f = next(iter(PRESETS[name].values()))()
        if res == LOW:
            low, lf, log = decimate(v, f, return_log=True)
            return low, lf, (log,), len(v)
    else:
        path = mesh_path(req)
        v, f = load_decimated(path, 0)[:2]
        if res == LOW:
            low, lf, logs = load_decimated(path, 1)  # cached per file
            return np.array(low), lf, logs, len(v)
    if res == HIGH:
        hv, hf = subdivide(v, f)
        return hv, hf, (), len(hv)
    return np.array(v), f, (), len(v)


def morph_options(req) -> dict:
    """The desktop app's morph check boxes, from a request (all default off).

    Args:
        req: request dict; reads ``shared_topology`` (pair B with A by vertex index),
            ``hold_t`` (every frame at ``morph_t``), ``a_dirs`` (radial: keep A's
            directions) and ``interference`` (read by turning selector qubits,
            :mod:`tessa_ripple.interfere`, instead of the basis sweep).

    Returns:
        dict of those four keys -> bool.
    """
    return {k: bool(req.get(k)) for k in ("shared_topology", "hold_t", "a_dirs", "interference")}


def spectral_capped(req, p, n_verts: int):
    """``p`` with ``spectral_k`` capped to what one run holds (the slider may ask for more).

    Args:
        req: request dict (reads ``interference``: its run holds half as many values).
        p: bridge.Params.
        n_verts: Vertices of A.

    Returns:
        bridge.Params: ``p``, with ``spectral_k`` at most ``bridge.spectral_cap``
        (:func:`bridge.cap_spectral`).
    """
    return bridge.cap_spectral(p, n_verts, p.mode == "morph" and morph_options(req)["interference"])


def encode_of(req, verts, faces, logs, full, p):
    """Encode the request's mesh. A morph first matches B to it (closest point, or by
    vertex index with ``shared_topology``), then encodes for the basis sweep or, with
    ``interference``, for the selector readout.

    Args:
        req: request dict (reads ``shape_b`` / ``obj_b`` and :func:`morph_options`).
        verts, faces, logs, full: from :func:`mesh_of`.
        p: bridge.Params.

    Returns:
        (Encoded, match, ie): match is the :func:`morph.correspond` dict and ie the
        ``interfere.Interfered`` (interference only), else None. A spectral ``k`` past
        what one run holds encodes the cap (``bridge.spectral_cap``).

    Raises:
        ValueError: a morph without B, a shared topology that does not match, a mesh
            that does not fit, a front with the interference readout or the spectral
            payload, or a front source that is not a vertex.
    """
    p = spectral_capped(req, p, len(verts))
    if p.mode != "morph":
        return bridge.encode_mesh(verts, faces, p), None, None
    if not (req.get("obj_b") or req.get("shape_b")):
        raise ValueError("Morph: drop a second mesh (B) first")
    opts = morph_options(req)
    if p.morph_front and opts["interference"]:
        raise ValueError("The morph front turns the basis sweep's qubits one patch at a time; "
                         "turn off interference readout to use it")
    if p.morph_front and p.payload == "spectral":
        raise ValueError(bridge.SPECTRAL_FRONT)
    if p.front_source is not None and not 0 <= p.front_source < len(verts):
        raise ValueError(f"Front source {p.front_source} is not a vertex of this mesh (0..{len(verts) - 1})")
    match = correspond(verts, faces, mesh_path(req, "_b"), opts["shared_topology"], logs, full)
    if opts["interference"]:
        ie = interf.encode_interfere(verts, faces, p, match)
        return ie.e, match, ie
    return bridge.encode_mesh(verts, faces, p, target=match), match, None


def params_of(req) -> bridge.Params:
    """The bridge.Params a request sets.

    Only the non-null CHOICES keys are read (never morph's file path); named
    choices are checked, and numbers clamped: shots to [256, 65536], compression to
    [0, 100], quant_bits to [1, 16], headroom and noise strength to [0, 10]. With
    ``morph_front`` true it also reads ``front_width`` (clamped to [0.02, 1], default
    0.15) and ``front_source`` (a vertex index; null = auto, the vertex that moves most).
    ``spectral_k`` (spectral payload: modes to encode) of 0 or less is auto (None); a
    k past one run's cap is capped later, in :func:`encode_of`.

    Args:
        req: request dict.

    Returns:
        bridge.Params.

    Raises:
        ValueError: a value outside its choices, or a number that does not parse.
        TypeError: a key of the wrong type for Params.
    """
    kw = {k: req[k] for k in CHOICES if req.get(k) is not None}
    for k, allowed in CHOICES.items():
        if allowed is None and k in kw:
            kw[k] = float(kw[k])
        elif allowed is not None and kw.get(k, allowed[0]) not in allowed:
            raise ValueError(f"unknown {k} {kw[k]!r}; use one of {list(allowed)}")
    for k, lo, hi in (("shots", 256, 65536), ("compression", 0, 100), ("quant_bits", 1, 16),
                      ("headroom", 0, 10), ("noise_strength", 0, 10)):
        if k in kw or k == "shots":
            kw[k] = float(np.clip(kw.get(k, 4096), lo, hi))
    for k in ("shots", "quant_bits", "seed", "spectral_k"):
        if k in kw:
            kw[k] = int(kw[k])
    if kw.get("spectral_k") is not None and kw["spectral_k"] <= 0:  # 0: auto (the 1 % tolerance)
        del kw["spectral_k"]
    if req.get("morph_front"):  # the morph front: width in [0.02, 1], source a vertex index or auto
        kw["morph_front"] = True
        kw["front_width"] = float(np.clip(float(req.get("front_width") or 0.15), 0.02, 1.0))
        if req.get("front_source") is not None:
            kw["front_source"] = int(req["front_source"])
    return bridge.Params(**kw)


def rounded(a, digits=5):
    """An array as nested lists of floats rounded for JSON.

    Args:
        a: array-like.
        digits: decimal places.

    Returns:
        Nested list of floats.
    """
    return np.round(np.asarray(a, dtype=float), digits).tolist()


def mesh(req):
    """The request's mesh as loaded (no encoding), so the page can show it before a run.

    Args:
        req: request dict; reads the mesh keys (:func:`mesh_of`), or with ``b`` true
            the morph B keys (``shape_b`` / ``obj_b``, B as its file loads).

    Returns:
        Dict: ``verts`` (N, 3) and ``faces``.
    """
    if req.get("b"):
        v, f = load_decimated(mesh_path(req, "_b"), 0)[:2]
    else:
        v, f = mesh_of(req)[:2]
    return {"verts": rounded(v), "faces": [list(map(int, x)) for x in f]}


def morph_frames(e, ie, p, want: int, drr: str, t_one: float, opts: dict, deadline: float):
    """Read a morph A -> B.

    Frames run along t = 0 -> 1, or all at ``t_one`` with ``hold_t`` (one readout
    reads at ``t_one`` either way). With ``p.front`` each data qubit reads at its own
    angle (bridge.morph_alpha). The basis sweep shares one alpha = 0 readout over
    its frames, as animate.run_frames does; frames at one t share each circuit's
    simulator job too (:func:`readouts`). The interference readout reads each frame
    on its own (interfere.read). The first frame times the readout; the frame count is
    then cut so the rest (and the noiseless references) finish before ``deadline``.

    Args:
        e: A morph encoding.
        ie: Its ``interfere.Interfered`` for the interference readout, else None.
        p: Readout settings.
        want: Frames asked for.
        drr: DRR method.
        t_one: The t of a single readout, and of every frame with ``hold_t``.
        opts: :func:`morph_options`.
        deadline: time.time() by which the readouts must be done (:func:`deadline_of`).

    Returns:
        (frames, refs, ts, kept): decoded (N, 3) positions; the noiseless morph at each
        t (bridge.morph_expected, or the interference ideal); each frame's t; and, for
        interference, the share of shots each frame was decoded from (else None).
    """
    hold, a_dirs = opts["hold_t"], opts["a_dirs"]
    start = time.time()
    reference = None if ie else bridge.tilted_angles(e, p, bridge.reference_seed(p.seed), 0.0)
    # a frame at a new t pays for a new circuit, as the reference did; frame 0 (t = 0)
    # reuses the reference's circuit, so on its own it would promise too many frames
    fresh = time.time() - start
    ts, frames, kept = ([t_one] if want == 1 or hold else [0.0]), [], []
    if not ie and (want == 1 or hold):  # one t: the frames share each circuit's simulator job
        q = replace(p, morph_t=t_one)
        raws = readouts(lambda n, k: bridge.decode_frames(e, q, n, p.seed + k, reference), want, deadline)
        frames = [bridge.morph_frame(e, raw, t_one, drr, a_dirs, p.front) for raw in raws]
        ts = ts * len(frames)
    slowest = fresh
    while len(frames) < len(ts):
        if frames and not fits(deadline, slowest):  # slower than the first frame promised: stop here
            ts = ts[:len(frames)]
            break
        k, t, began = len(frames), float(ts[len(frames)]), time.time()
        if ie:
            raw, share = interf.read(ie, p, t, p.seed + k)
            frames.append(interf.frame(ie, raw, t, drr, a_dirs))
            kept.append(float(share))
        else:
            raw = bridge.decode_raw(e, replace(p, morph_t=t), p.seed + k, reference)
            frames.append(bridge.morph_frame(e, raw, t, drr, a_dirs, p.front))
        slowest = max(slowest, time.time() - began)
        if want > 1 and k == 0:
            per = max(slowest, 1e-3)
            n = int(np.clip(1 + (deadline - time.time()) // (per * FRAME_MARGIN), 1, want))
            ts = [t_one] * n if hold else list(np.linspace(0.0, 1.0, n))
    refs = {}
    for t in ts:
        if t not in refs:
            refs[t] = (interf.frame(ie, interf.ideal(ie, t), t, "none", a_dirs) if ie
                       else bridge.morph_expected(e, t, a_dirs, p.front))
    return frames, [refs[t] for t in ts], [float(t) for t in ts], (kept if ie else None)


def front_info(e, p, ts) -> dict:
    """What the page shows of a morph front: its source and how far each vertex has turned.

    Args:
        e: A morph encoding.
        p: Params with the front on.
        ts: Each frame's t.

    Returns:
        Dict: ``source`` (vertex index, the automatic one resolved), ``width``,
        ``weights`` (per frame, (N,) share of the way to B each vertex was read at:
        its data qubits' alpha / (pi/2), averaged over its components) and ``smooth``
        (the same at each frame's t with every vertex's own distance, which no circuit
        can make: the gap to ``weights`` is the chip's patch quantisation).
    """
    width, source = p.front
    front = (width, bridge.front_distances(e, source)[0])  # the automatic source resolved
    weights = [bridge.front_weights(e, t, front) for t in ts]
    smooth = [bridge.front_weights(e, t, front, smooth=True) for t in ts]
    source = front[1]
    return {"source": int(source), "width": width,
            "weights": [rounded(w, 3) for w in weights], "smooth": [rounded(w, 3) for w in smooth]}


def run(req):
    """Encode once, decode up to ``frames`` readouts.

    The readouts share each circuit's simulator job (:func:`readouts`), as many as
    finish by the request's deadline (:func:`deadline_of`; at least one always runs).
    A morph (``mode`` "morph") reads its frames along t = 0 -> 1 (or at a held t)
    instead (:func:`morph_frames`), each compared with the noiseless morph at its t.

    Args:
        req: request dict; reads the mesh keys (:func:`mesh_of`), Params keys
            (:func:`params_of`), ``frames`` (1..MAX_FRAMES, default 1), ``drr``
            (a DRR method, default "none") and, for a morph, ``shape_b`` / ``obj_b``
            ``morph_t`` (the t of a single readout or of held frames, default 0.5)
            and the :func:`morph_options` flags.

    Returns:
        Dict: ``verts``, ``faces`` (the original mesh); ``frames`` (decoded
        positions per frame); ``requested`` (frames asked for); ``qubits`` (data
        qubit per vertex); ``map``
        (bridge.qubit_map); ``stats`` (Encoded.stats); ``split`` (some vertex spans
        several data qubits); ``error`` (mean bridge.mesh_error over frames, against
        each frame's reference); ``seconds``; ``spectral`` (``Encoded.spectral``: k,
        rows, truncation errors; None for other payloads). For the spectral payload
        ``qubits`` and ``map`` rows are modes, not vertices, and a θ or split readout
        adds ``refs`` (the truncated mesh, what a noiseless readout returns). A morph adds ``refs`` (noiseless
        positions per frame), ``ts`` (t per frame), ``match`` (B's matching:
        method, coverage, mean distance), ``options`` (:func:`morph_options`),
        ``clamp`` (the one clamp both fields use, in units of π) and, for the
        interference readout, ``kept`` (share of shots each frame was decoded from).
        A front adds ``front`` (:func:`front_info`).
    """
    t0, deadline = time.time(), deadline_of(req)
    if req.get("mode") == "morph" and req.get("interference"):
        too_slow(req, "interference")
    verts, faces, logs, full = mesh_of(req)
    p = params_of(req)
    e, match, ie = encode_of(req, verts, faces, logs, full, p)
    want, frames, drr = int(np.clip(req.get("frames", 1), 1, MAX_FRAMES)), [], req.get("drr", "none")
    extra = {}
    if match is not None:
        t_one = float(np.clip(float(req.get("morph_t", 0.5)), 0.0, 1.0))
        opts = morph_options(req)
        frames, refs, ts, kept = morph_frames(e, ie, p, want, drr, t_one, opts, deadline)
        if p.front is not None:
            extra["front"] = front_info(e, p, ts)
        extra = {**extra, "refs": [rounded(r) for r in refs], "ts": ts, "options": opts,
                 "match": {k: match[k] for k in ("method", "coverage", "distance")},
                 "clamp": bridge.morph_clamp(p) / np.pi, **({"kept": kept} if ie else {})}
    else:
        raws = readouts(lambda n, k: bridge.decode_frames(e, p, n, p.seed + k), want, deadline)
        frames = [bridge.restore(e, raw, drr) for raw in raws]
        refs = [bridge.encoded_shape(e)] * len(frames)  # the mesh, or spectral's truncated mesh
        if e.spectral is not None:
            extra["refs"] = [rounded(r) for r in refs]
    return {"verts": rounded(verts), "faces": [list(map(int, f)) for f in faces],
            "frames": [rounded(f) for f in frames], "requested": want,
            "qubits": bridge.vertex_qubits(e).tolist(), "map": bridge.qubit_map(e), "stats": e.stats,
            "split": e.straddles, "spectral": e.spectral,
            "error": float(np.mean([bridge.mesh_error(r, f) for r, f in zip(refs, frames)])),
            "seconds": time.time() - t0, **extra}


def not_morph(req, what: str):
    """Refuse an action that compares against one fixed mesh in morph mode.

    Args:
        req: request dict.
        what: The action's name, for the message.

    Raises:
        ValueError: ``mode`` is "morph".
    """
    if req.get("mode") == "morph":
        raise ValueError(f"{what} compare against one fixed mesh; not available in morph mode")


def edges(req):
    """Edge lengths read from coherences (bridge.read_edges).

    Args:
        req: request dict; reads the mesh keys and Params keys.

    Returns:
        Dict: ``edges`` (E, 2) vertex pairs, ``measured`` and ``exact`` lengths (E,),
        ``seconds``.
    """
    not_morph(req, "Edge readouts")
    too_slow(req, "edges")
    verts, faces = mesh_of(req)[:2]
    t0 = time.time()
    E, measured, exact = bridge.read_edges(verts, faces, params_of(req))
    return {"edges": E.tolist(), "measured": rounded(measured), "exact": rounded(exact),
            "seconds": time.time() - t0}


def resources(req):
    """The "Why Tessa" resource table for the request's encoding.

    Args:
        req: request dict; reads the mesh keys and Params keys.

    Returns:
        Dict: ``rows`` (bridge.resources), ``scaled`` (bridge.scaled_resources),
        ``stats`` (Encoded.stats), ``payload``, ``mode``.
    """
    verts, faces, logs, full = mesh_of(req)
    p = params_of(req)
    e = encode_of(req, verts, faces, logs, full, p)[0]  # interference: its sweep encoding
    return {"rows": bridge.resources(e), "scaled": bridge.scaled_resources(e, p),
            "stats": e.stats, "payload": e.payload, "mode": e.mode}


def curves(req):
    """Fidelity curves (bridge.fidelity_curves) for the request's mesh and settings.

    Args:
        req: request dict; reads the mesh keys and Params keys.

    Returns:
        Dict: ``shots`` [(shots, error)], ``mesh`` and ``naive`` [(compression %,
        depth, error)], ``same`` (both layouts place values identically).
    """
    not_morph(req, "Curves")
    too_slow(req, "curves")
    verts, faces = mesh_of(req)[:2]
    return bridge.fidelity_curves(verts, faces, spectral_capped(req, params_of(req), len(verts)))


ACTIONS = {"mesh": mesh, "run": run, "edges": edges, "resources": resources, "curves": curves}


def _json_default(x):
    """JSON fallback: numpy scalars to Python numbers, anything else to str.

    Args:
        x: object json cannot encode.

    Returns:
        A JSON-encodable value.
    """
    return x.item() if isinstance(x, np.generic) else str(x)


class handler(BaseHTTPRequestHandler):
    """Vercel's request handler for /api/tessa."""
    def _send(self, code, body):
        """Send a JSON response.

        Args:
            code: HTTP status code.
            body: JSON-encodable response body.
        """
        data = json.dumps(body, default=_json_default).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        """Run the action the JSON body names and send its result.

        Body: ``{"action": "mesh" | "run" | "edges" | "resources" | "curves", ...}`` plus the
        action's keys. Sends 200 with the action's dict, 400 with ``{"error": ...}``
        for an unknown action or a ValueError, KeyError or TypeError from it, and 500
        with ``{"error": ...}`` for anything else.
        """
        try:
            global _cold
            arrived = time.time()
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            # the work must end before Vercel's limit, less the reply's time and, on a
            # cold start, the imports this process has just done
            req["_deadline"] = arrived + FUNCTION_LIMIT - REPLY_MARGIN - (_IMPORT_SECONDS if _cold else 0.0)
            _cold = False
            action = ACTIONS.get(req.get("action"))
            if action is None:
                return self._send(400, {"error": f"unknown action; use one of {sorted(ACTIONS)}"})
            self._send(200, action(req))
        except (ValueError, KeyError, TypeError) as err:
            self._send(400, {"error": str(err)})
        except Exception as err:  # still JSON, so the page shows the cause
            self._send(500, {"error": f"{type(err).__name__}: {err}"})

    def do_GET(self):
        """Health check: sends ``ok``, the ``actions``, ``shapes``, ``backends`` and ``drr`` methods."""
        self._send(200, {"ok": True, "actions": sorted(ACTIONS), "shapes": sorted({*PRESETS, *OBJ_FILES}),
                         "backends": list(bridge.BACKENDS), "drr": list(bridge.DRR_METHODS)})
