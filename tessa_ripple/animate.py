"""Multi-run frames and the classical layer on top: blend, mask, interpolate, export."""

from pathlib import Path

import numpy as np

from dataclasses import replace

from .bridge import Encoded, Params, decode_raw, tilted_angles
from .mesh import write_obj


def run_frames(e: Encoded, p: Params, n: int, progress=None, ts=None):
    """``n`` independent readouts of one encoding, with seeds seed, seed + 1, ...

    ``ts`` (morph) gives each frame its own t: the basis sweep reads each frame at that
    t's angle. A transition-decoded sweep shares one Z
    (alpha = 0) readout across all frames, as Tessa's transition does.

    Args:
        e: the encoded mesh.
        p: readout settings; ``p.seed`` is frame 0's seed.
        n: number of frames.
        progress: optional callback ``progress(i, n)`` before frame i.
        ts: optional per-frame morph t (length n); None uses ``p.morph_t``.

    Returns:
        (values, second): values (n, N*comps) raw decoded values; second (n, ...)
        stacked second outputs of :func:`~tessa_ripple.bridge.decode_raw` (colour
        (n, N) in colour mode), else None.
    """
    values, colours = [], []
    reference = None
    if e.mode == "morph" and p.sweep_decode == "transition":
        # one Z (alpha = 0) readout serves every frame, as in Tessa's transition: at
        # t = 0 the first frame is exactly this readout
        reference = tilted_angles(e, p, p.seed, 0.0)
    for i in range(n):
        if progress:
            progress(i, n)
        q = p if ts is None else replace(p, morph_t=float(ts[i]))
        v, c = decode_raw(e, q, p.seed + i, reference)
        values.append(v)
        colours.append(c)
    return np.array(values), (None if colours[0] is None else np.array(colours))


def random_mask(n_verts: int, fraction: float, seed: int = 0) -> np.ndarray:
    """A seeded random 0/1 mask over vertices.

    Args:
        n_verts: number of vertices N.
        fraction: share of vertices set to 1 (clipped to [0, 1], rounded).
        seed: RNG seed.

    Returns:
        (N,) float array, 1.0 on the chosen vertices and 0.0 elsewhere.
    """
    k = int(round(np.clip(fraction, 0, 1) * n_verts))
    mask = np.zeros(n_verts)
    mask[np.random.default_rng(seed).choice(n_verts, k, replace=False)] = 1.0
    return mask


def blend(orig, decoded, gain: float = 1.0, mask=None) -> np.ndarray:
    """Scale each vertex's deviation from the original: orig + gain * mask * (decoded - orig).

    Args:
        orig: (N, 3) reference positions.
        decoded: (N, 3) decoded positions.
        gain: deviation multiplier (1 = true decoded positions).
        mask: optional (N,) per-vertex weight; None means all ones.

    Returns:
        (N, 3) blended positions.
    """
    mask = np.ones(len(orig)) if mask is None else np.asarray(mask)
    return orig + gain * mask[:, None] * (np.asarray(decoded) - orig)


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
