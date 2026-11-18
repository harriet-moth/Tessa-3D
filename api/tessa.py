"""Vercel Python function: every Tessa computation the web app needs, at POST /api/tessa.

The request body is JSON with an ``action`` (run / edges / resources / curves) plus
the mesh and Params settings. All quantum work goes through tessa_ripple.bridge,
the same code the desktop app uses.
"""

import json
import sys
import time
from dataclasses import fields
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tessa_ripple import bridge  # noqa: E402
from tessa_ripple.mesh import PRESETS, load_obj  # noqa: E402

SHAPES = {**PRESETS, "suzanne": {"(OBJ)": lambda: load_obj(ROOT / "tessa_ripple" / "suzanne.obj")}}
MAX_FRAMES = 30
TIME_BUDGET = 45.0  # seconds of decoding per request, under the 60 s function limit


def mesh_of(req):
    presets = list(SHAPES[req.get("shape", "blob")].values())
    return presets[min(int(req.get("resolution", 0)), len(presets) - 1)]()


def params_of(req) -> bridge.Params:
    kw = {f.name: req[f.name] for f in fields(bridge.Params) if req.get(f.name) is not None}
    if kw.get("backend", "noiseless") not in bridge.BACKENDS:
        raise ValueError(f"unknown backend {kw['backend']!r}")
    kw["shots"] = int(np.clip(kw.get("shots", 4096), 256, 65536))
    return bridge.Params(**kw)


def rounded(a, digits=5):
    return np.round(np.asarray(a, dtype=float), digits).tolist()


def run(req):
    """Encode once, decode up to ``frames`` readouts (seed, seed+1, ...)."""
    verts, faces = mesh_of(req)
    p, t0 = params_of(req), time.time()
    e = bridge.encode_mesh(verts, faces, p)
    want, frames, colours = int(np.clip(req.get("frames", 1), 1, MAX_FRAMES)), [], []
    while len(frames) < want and (not frames or time.time() - t0 < TIME_BUDGET):
        raw, colour = bridge.decode_raw(e, p, p.seed + len(frames))
        frames.append(bridge.restore(e, raw, req.get("drr", "none")))
        colours.append(colour)
    return {"verts": rounded(verts), "faces": [list(map(int, f)) for f in faces],
            "frames": [rounded(f) for f in frames], "requested": want,
            "colours": None if colours[0] is None else rounded(colours, 4),
            "colour": None if e.colour is None else rounded(e.colour, 4),
            "qubits": bridge.vertex_qubits(e).tolist(), "stats": e.stats,
            "split": bool((np.diff(np.searchsorted(bridge.cells(e.graph)[:, 1], e.slots, side="right"), axis=1) != 0).any()),
            "error": float(np.mean([bridge.mesh_error(verts, f) for f in frames])),
            "seconds": time.time() - t0}


def edges(req):
    verts, faces = mesh_of(req)
    t0 = time.time()
    E, measured, exact = bridge.read_edges(verts, faces, params_of(req))
    return {"edges": E.tolist(), "measured": rounded(measured), "exact": rounded(exact),
            "seconds": time.time() - t0}


def resources(req):
    verts, faces = mesh_of(req)
    e = bridge.encode_mesh(verts, faces, params_of(req))
    return {"rows": bridge.resources(e), "stats": e.stats, "payload": e.payload, "mode": e.enc.mode}


def curves(req):
    verts, faces = mesh_of(req)
    return bridge.fidelity_curves(verts, faces, params_of(req))


ACTIONS = {"run": run, "edges": edges, "resources": resources, "curves": curves}


def _json_default(x):
    return x.item() if isinstance(x, np.generic) else str(x)


class handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        data = json.dumps(body, default=_json_default).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            action = ACTIONS.get(req.get("action"))
            if action is None:
                return self._send(400, {"error": f"unknown action; use one of {sorted(ACTIONS)}"})
            self._send(200, action(req))
        except (ValueError, KeyError, TypeError) as err:
            self._send(400, {"error": str(err)})

    def do_GET(self):
        self._send(200, {"ok": True, "actions": sorted(ACTIONS), "shapes": sorted(SHAPES),
                         "backends": list(bridge.BACKENDS), "drr": list(bridge.DRR_METHODS)})
