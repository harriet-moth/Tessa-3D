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
from dataclasses import replace
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tessa_ripple import bridge  # noqa: E402
from tessa_ripple import interfere as interf  # noqa: E402
from tessa_ripple.mesh import PRESETS, load_decimated, write_obj  # noqa: E402
from tessa_ripple.morph import correspond  # noqa: E402

# OBJ shapes that ship with the app, by name.
OBJ_FILES = {"suzanne": ROOT / "tessa_ripple" / "suzanne.obj",
             "apple": ROOT / "tessa_ripple" / "apple_bitten.obj"}
# The Params fields the web app sets, with their allowed values (None: any number).
# Morph's ``target`` (a file path) is never taken from a request: B comes from
# ``shape_b`` / ``obj_b`` (:func:`mesh_path`).
CHOICES = {"payload": ("radial", "xyz", "spherical"), "layout": ("mesh", "naive"),
           "mode": ("theta", "split", "morph"), "backend": bridge.BACKENDS, "placement": bridge.PLACEMENTS,
           "clamp": None, "clamp_b": None, "compression": None, "quant_bits": None,
           "headroom": None, "shots": None, "noise_strength": None, "seed": None}
MAX_FRAMES = 30
TIME_BUDGET = 45.0  # seconds of decoding per request, under the 60 s function limit
MAX_OBJ = 3_000_000  # characters of uploaded OBJ text (Vercel caps request bodies at 4.5 MB)
UPLOADS = Path(tempfile.gettempdir()) / "tessa_ripple_obj"  # Vercel's writable /tmp


def resolution_of(req) -> int:
    """The request's ``resolution`` (0 low, 1 mid, 2 high; default 0), clamped.

    Args:
        req: request dict.

    Returns:
        int in [0, 2].
    """
    return int(np.clip(int(req.get("resolution", 0)), 0, 2))


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
    k = min(resolution_of(req) if not suffix else len(presets) - 1, len(presets) - 1)
    path = UPLOADS / f"{name}-{k}.obj"
    if not path.exists():
        write_obj(path, *presets[k]())
    return path


def mesh_of(req):
    """The mesh a request names, with its decimation logs.

    Presets: ``resolution`` picks one of the shape's presets. OBJ meshes (shipped
    or uploaded ``obj``): resolution 2 is the file as loaded, 1 and 0 decimate it
    once and twice (each pass keeps about half the vertices).

    Args:
        req: request dict; reads ``obj`` or ``shape`` (default "blob") and
            ``resolution`` (0..2, default 0).

    Returns:
        (verts (N, 3), faces, logs, full_count): logs are the decimation passes and
        full_count the vertex count before them, for :func:`morph.correspond`.

    Raises:
        KeyError: unknown shape.
        ValueError: an unreadable OBJ.
    """
    name = req.get("shape", "blob")
    if not req.get("obj") and name in PRESETS:
        presets = list(PRESETS[name].values())
        v, f = presets[min(resolution_of(req), len(presets) - 1)]()
        return v, f, (), len(v)
    path = mesh_path(req)
    v, f, logs = load_decimated(path, 2 - resolution_of(req))
    return np.array(v), f, logs, len(load_decimated(path, 0)[0])


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
        ``interfere.Interfered`` (interference only), else None.

    Raises:
        ValueError: a morph without B, a shared topology that does not match, or a
            mesh that does not fit.
    """
    if p.mode != "morph":
        return bridge.encode_mesh(verts, faces, p), None, None
    if not (req.get("obj_b") or req.get("shape_b")):
        raise ValueError("Morph: drop a second mesh (B) first")
    opts = morph_options(req)
    match = correspond(verts, faces, mesh_path(req, "_b"), opts["shared_topology"], logs, full)
    if opts["interference"]:
        ie = interf.encode_interfere(verts, faces, p, match)
        return ie.e, match, ie
    return bridge.encode_mesh(verts, faces, p, target=match), match, None


def params_of(req) -> bridge.Params:
    """The bridge.Params a request sets.

    Only the non-null CHOICES keys are read (never morph's file path); named
    choices are checked, and numbers clamped: shots to [256, 65536], compression to
    [0, 100], quant_bits to [1, 16], headroom and noise strength to [0, 10].

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
    for k in ("shots", "quant_bits", "seed"):
        if k in kw:
            kw[k] = int(kw[k])
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


def morph_frames(e, ie, p, want: int, drr: str, t_one: float, opts: dict, t0: float):
    """Read a morph A -> B.

    Frames run along t = 0 -> 1, or all at ``t_one`` with ``hold_t`` (one readout
    reads at ``t_one`` either way). The basis sweep shares one alpha = 0 readout over
    its frames, as animate.run_frames does; the interference readout reads each frame
    on its own (interfere.read). The first frame times the readout; the frame count is
    then cut so the rest fits in TIME_BUDGET.

    Args:
        e: A morph encoding.
        ie: Its ``interfere.Interfered`` for the interference readout, else None.
        p: Readout settings.
        want: Frames asked for.
        drr: DRR method.
        t_one: The t of a single readout, and of every frame with ``hold_t``.
        opts: :func:`morph_options`.
        t0: Request start time (time.time()).

    Returns:
        (frames, refs, ts, kept): decoded (N, 3) positions; the noiseless morph at each
        t (bridge.morph_expected, or the interference ideal); each frame's t; and, for
        interference, the share of shots each frame was decoded from (else None).
    """
    hold, a_dirs = opts["hold_t"], opts["a_dirs"]
    start = time.time()
    reference = None if ie else bridge.tilted_angles(e, p, bridge.reference_seed(p.seed), 0.0)
    ts, frames, kept = ([t_one] if want == 1 or hold else [0.0]), [], []
    while len(frames) < len(ts):
        k, t = len(frames), float(ts[len(frames)])
        if ie:
            raw, share = interf.read(ie, p, t, p.seed + k)
            frames.append(interf.frame(ie, raw, t, drr, a_dirs))
            kept.append(float(share))
        else:
            raw = bridge.decode_raw(e, replace(p, morph_t=t), p.seed + k, reference)
            frames.append(bridge.morph_frame(e, raw, t, drr, a_dirs))
        if want > 1 and k == 0:
            per = max(time.time() - start, 1e-3) / (1 if ie else 2)  # the sweep's reference cost about one readout
            n = int(np.clip(1 + (TIME_BUDGET - (time.time() - t0)) // per, 2, want))
            ts = [t_one] * n if hold else list(np.linspace(0.0, 1.0, n))
    refs = {}
    for t in ts:
        if t not in refs:
            refs[t] = (interf.frame(ie, interf.ideal(ie, t), t, "none", a_dirs) if ie
                       else bridge.morph_expected(e, t, a_dirs))
    return frames, [refs[t] for t in ts], [float(t) for t in ts], (kept if ie else None)


def run(req):
    """Encode once, decode up to ``frames`` readouts (seeds seed, seed + 1, ...).

    Decoding stops early after TIME_BUDGET seconds (at least one frame always runs).
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
        each frame's reference); ``seconds``. A morph adds ``refs`` (noiseless
        positions per frame), ``ts`` (t per frame), ``match`` (B's matching:
        method, coverage, mean distance), ``options`` (:func:`morph_options`),
        ``clamp`` (the one clamp both fields use, in units of π) and, for the
        interference readout, ``kept`` (share of shots each frame was decoded from).
    """
    verts, faces, logs, full = mesh_of(req)
    p, t0 = params_of(req), time.time()
    e, match, ie = encode_of(req, verts, faces, logs, full, p)
    want, frames, drr = int(np.clip(req.get("frames", 1), 1, MAX_FRAMES)), [], req.get("drr", "none")
    extra = {}
    if match is not None:
        t_one = float(np.clip(float(req.get("morph_t", 0.5)), 0.0, 1.0))
        opts = morph_options(req)
        frames, refs, ts, kept = morph_frames(e, ie, p, want, drr, t_one, opts, t0)
        extra = {"refs": [rounded(r) for r in refs], "ts": ts, "options": opts,
                 "match": {k: match[k] for k in ("method", "coverage", "distance")},
                 "clamp": bridge.morph_clamp(p) / np.pi, **({"kept": kept} if ie else {})}
    else:
        refs = [verts]
        while len(frames) < want and (not frames or time.time() - t0 < TIME_BUDGET):
            raw = bridge.decode_raw(e, p, p.seed + len(frames))
            frames.append(bridge.restore(e, raw, drr))
        refs = refs * len(frames)
    return {"verts": rounded(verts), "faces": [list(map(int, f)) for f in faces],
            "frames": [rounded(f) for f in frames], "requested": want,
            "qubits": bridge.vertex_qubits(e).tolist(), "map": bridge.qubit_map(e), "stats": e.stats,
            "split": e.straddles,
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
        Dict: ``rows`` (bridge.resources), ``stats`` (Encoded.stats), ``payload``,
        ``mode``.
    """
    verts, faces, logs, full = mesh_of(req)
    e = encode_of(req, verts, faces, logs, full, params_of(req))[0]  # interference: its sweep encoding
    return {"rows": bridge.resources(e), "stats": e.stats, "payload": e.payload, "mode": e.mode}


def curves(req):
    """Fidelity curves (bridge.fidelity_curves) for the request's mesh and settings.

    Args:
        req: request dict; reads the mesh keys and Params keys.

    Returns:
        Dict: ``shots`` [(shots, error)], ``mesh`` and ``naive`` [(compression %,
        depth, error)], ``same`` (both layouts place values identically).
    """
    not_morph(req, "Curves")
    verts, faces = mesh_of(req)[:2]
    return bridge.fidelity_curves(verts, faces, params_of(req))


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
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
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
