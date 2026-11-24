"""Multi-run frames and the classical layer on top: blend, interpolate, export."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from .bridge import Encoded, Params, decode_raw, reference_seed, tilted_angles
from .mesh import write_obj


def run_frames(e: Encoded, p: Params, n: int, progress=None, ts=None):
    """``n`` independent readouts of one encoding, with seeds seed, seed + 1, ...

    ``ts`` (morph) gives each frame its own t: the basis sweep reads each frame at that
    t's angle, and every frame shares one Z (alpha = 0) readout, as Tessa's
    transition does. That readout is its own run, seeded by
    :func:`~tessa_ripple.bridge.reference_seed`.

    Args:
        e: the encoded mesh.
        p: readout settings; ``p.seed`` is frame 0's seed.
        n: number of frames.
        progress: optional callback ``progress(i, n)`` before frame i.
        ts: optional per-frame morph t (length n); None uses ``p.morph_t``.

    Returns:
        np.ndarray: (n, N*comps) raw decoded values.
    """
    values = []
    reference = None
    if e.mode == "morph":
        # one Z (alpha = 0) readout serves every frame, as in Tessa's transition; a
        # frame (or a front's unreached part) at alpha = 0 reads A from its own readout
        reference = tilted_angles(e, p, reference_seed(p.seed), 0.0)
    for i in range(n):
        if progress:
            progress(i, n)
        q = p if ts is None else replace(p, morph_t=float(ts[i]))
        values.append(decode_raw(e, q, p.seed + i, reference))
    return np.array(values)


def blend(orig, decoded, gain: float = 1.0) -> np.ndarray:
    """Scale each vertex's deviation from the original: orig + gain * (decoded - orig).

    Args:
        orig: (N, 3) reference positions.
        decoded: (N, 3) decoded positions.
        gain: deviation multiplier (1 = true decoded positions).

    Returns:
        (N, 3) blended positions.
    """
    return orig + gain * (np.asarray(decoded) - orig)


def smoothstep(t):
    """Ease-in-out curve 3t² - 2t³.

    Args:
        t: weight(s) in [0, 1], scalar or array.

    Returns:
        Eased weight(s), same shape as ``t``.
    """
    return t * t * (3 - 2 * t)


def interpolate(a, b, t: float) -> np.ndarray:
    """Eased (smoothstep) interpolation from ``a`` to ``b``.

    Args:
        a: positions at t = 0, e.g. (N, 3).
        b: positions at t = 1, same shape.
        t: weight in [0, 1].

    Returns:
        Positions, same shape as ``a``.
    """
    return a + (b - a) * smoothstep(t)


def timeline(keys, steps: int = 0, loop: bool = False) -> list[np.ndarray]:
    """Positions gliding through ``keys`` with eased in-betweens.

    Args:
        keys: sequence of key positions, each (N, 3).
        steps: in-between frames inserted between each pair of keys.
        loop: also glide from the last key back to the first (the first is not
            repeated at the end).

    Returns:
        List of (N, 3) positions: every key, with ``steps`` in-betweens after each.
    """
    keys = list(keys) + ([keys[0]] if loop and len(keys) > 1 else [])
    out = []
    for a, b in zip(keys[:-1], keys[1:]):
        out += [interpolate(a, b, k / (steps + 1)) for k in range(steps + 1)]
    if not loop or len(keys) == 1:
        out.append(keys[-1])
    return out


def export_obj_sequence(folder, positions, faces) -> list[Path]:
    """Write each frame as ``frame_000.obj``, ``frame_001.obj``, ... in ``folder``.

    Args:
        folder: output directory (created if missing).
        positions: sequence of (N, 3) vertex positions, one per frame.
        faces: polygons shared by every frame.

    Returns:
        List of written Paths, in frame order.
    """
    return [write_obj(Path(folder) / f"frame_{i:03d}.obj", v, faces)
            for i, v in enumerate(positions)]
