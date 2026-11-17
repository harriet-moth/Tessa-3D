"""Multi-run frames and the classical layer on top: blend, mask, interpolate, export."""

from pathlib import Path

import numpy as np

from .bridge import Encoded, Params, decode_raw
from .mesh import write_obj


def run_frames(e: Encoded, p: Params, n: int, progress=None) -> np.ndarray:
    """``n`` independent readouts with seeds seed, seed+1, ...: raw (n, 3N) values."""
    frames = []
    for i in range(n):
        if progress:
            progress(i, n)
        frames.append(decode_raw(e, p, p.seed + i))
    return np.array(frames)


def random_mask(n_verts: int, fraction: float, seed: int = 0) -> np.ndarray:
    """1.0 for a seeded random ``fraction`` of vertices, 0.0 elsewhere."""
    k = int(round(np.clip(fraction, 0, 1) * n_verts))
    mask = np.zeros(n_verts)
    mask[np.random.default_rng(seed).choice(n_verts, k, replace=False)] = 1.0
    return mask


def blend(orig, decoded, gain: float = 1.0, mask=None) -> np.ndarray:
    """out = orig + gain * mask * (decoded - orig)."""
    mask = np.ones(len(orig)) if mask is None else np.asarray(mask)
    return orig + gain * mask[:, None] * (np.asarray(decoded) - orig)


def smoothstep(t):
    return t * t * (3 - 2 * t)


def interpolate(a, b, t: float) -> np.ndarray:
    return a + (b - a) * smoothstep(t)


def timeline(keys, steps: int = 0, loop: bool = False) -> list[np.ndarray]:
    """Positions gliding through ``keys`` with ``steps`` eased in-betweens per pair."""
    keys = list(keys) + ([keys[0]] if loop and len(keys) > 1 else [])
    out = []
    for a, b in zip(keys[:-1], keys[1:]):
        out += [interpolate(a, b, k / (steps + 1)) for k in range(steps + 1)]
    if not loop or len(keys) == 1:
        out.append(keys[-1])
    return out


def export_obj_sequence(folder, positions, faces) -> list[Path]:
    return [write_obj(Path(folder) / f"frame_{i:03d}.obj", v, faces)
            for i, v in enumerate(positions)]
