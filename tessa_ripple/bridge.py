"""Every Tessa call lives here: fit a lattice, encode a mesh once, decode it per seed.

Pure functions, numpy in and numpy out, so a Blender add-on can reuse them.
"""

from dataclasses import dataclass

import numpy as np
from tessa import Lattice, decode, encode, prepare_field, restore_field
from tessa.backends import aer_sampler, noisy_sampler

MAX_SIDE = 8  # V1 cap: 8x8 lattice, 64 qubits
DRR_METHODS = ("none", "min_max", "mean_std", "percentile", "abs_max", "min_avg_max")
# Typical IBM-like machine averages (tessa.backend_rates fallbacks), so the
# qiskit-ibm-runtime extra is never needed.
IBM_LIKE_RATES = {"sx_error": 3e-4, "cz_error": 8e-3, "readout_error": 1e-2,
                  "sx_duration": 3.2e-8, "cz_duration": 1.6e-7, "t1": 3e-4, "t2": 2.5e-4}


@dataclass(frozen=True)
class Params:
    # encoder (changing any of these re-encodes)
    mode: str = "single"           # "single" | "pair"
    clamp: float | None = None     # theta clamp in radians; None = Tessa default
    clamp_b: float | None = None   # phi clamp (pair only)
    compression: float = 0.0       # percent (0-100) of multiplexer coefficients dropped
    quant_bits: int | None = None
    headroom: float = 0.1          # padding on the symmetric value_range
    # readout (changing these re-decodes)
    shots: int = 4096
    noisy: bool = False
    noise_strength: float = 1.0
    seed: int = 0

    @property
    def encoder_key(self):
        return (self.mode, self.clamp, self.clamp_b, self.compression, self.quant_bits,
                self.headroom)

    @property
    def readout_key(self):
        return (self.shots, self.noisy, self.noise_strength, self.seed)


@dataclass(frozen=True)
class Encoded:
    verts: np.ndarray   # original (N, 3)
    lattice: Lattice
    circuit: object
    enc: object         # tessa Encoding
    prep: object        # tessa FieldPrep for DRR / un-quantising

    @property
    def stats(self) -> dict:
        n_values = self.verts.size
        return {"lattice": f"{self.lattice.rows}x{self.lattice.cols}",
                "qubits": self.lattice.n_qubits, "capacity": self.lattice.capacity,
                "used": n_values / (2 if self.enc.mode == "pair" else 1),
                "depth": self.circuit.depth()}


def fit_lattice(n_values: int) -> Lattice:
    """Smallest near-square Lattice (cols - rows <= 2) holding ``n_values``."""
    shapes = sorted(((r, c) for r in range(2, MAX_SIDE + 1) for c in range(r, min(r + 2, MAX_SIDE) + 1)),
                    key=lambda rc: (rc[0] * rc[1], rc[1] - rc[0]))
    for r, c in shapes:
        lat = Lattice(r, c)
        if lat.capacity >= n_values:
            return lat
    raise ValueError(f"{n_values} values exceed the {MAX_SIDE}x{MAX_SIDE} lattice capacity "
                     f"({Lattice(MAX_SIDE, MAX_SIDE).capacity}); pick a smaller mesh or pair mode")


def encode_mesh(verts: np.ndarray, p: Params) -> Encoded:
    """Encode xyz axis-major (all x, all y, all z) on one symmetric value_range."""
    verts = np.asarray(verts, dtype=float)
    flat, prep = prepare_field(verts.T.ravel(), quant_bits=p.quant_bits)
    m = float(np.abs(flat).max() or 1.0) * (1 + p.headroom)
    kw = dict(value_range=(-m, m), compression=p.compression, clamp=p.clamp)
    if p.mode == "pair":
        half = -(-flat.size // 2)
        a, b = flat[:half], np.pad(flat[half:], (0, 2 * half - flat.size))
        lattice = fit_lattice(half)
        circuit, enc = encode(lattice, a, b, clamp_b=p.clamp_b, value_range_b=(-m, m), **kw)
    else:
        lattice = fit_lattice(flat.size)
        circuit, enc = encode(lattice, flat, **kw)
    return Encoded(verts, lattice, circuit, enc, prep)


def sampler(lattice: Lattice, p: Params, seed: int):
    if p.noisy and p.noise_strength > 0:
        s = p.noise_strength
        rates = {k: (v / s if k in ("t1", "t2") else v * s if k.endswith("error") else v)
                 for k, v in IBM_LIKE_RATES.items()}
        return noisy_sampler(lattice, rates=rates, seed=seed)
    return aer_sampler(lattice, seed=seed)


def decode_raw(e: Encoded, p: Params, seed: int | None = None) -> np.ndarray:
    """One readout of the unchanged circuit: flat axis-major values, before DRR."""
    seed = p.seed if seed is None else seed
    out = decode(e.circuit, e.enc, sampler(e.lattice, p, seed), shots=p.shots)
    return np.concatenate(out)[: e.verts.size] if isinstance(out, tuple) else out


def restore(e: Encoded, raw: np.ndarray, drr: str = "none") -> np.ndarray:
    """Classical post-processing (DRR, un-quantise) and reshape back to (N, 3)."""
    return restore_field(raw, e.prep, dynamic_range=drr).reshape(3, -1).T


def read_mesh(e: Encoded, p: Params, seed: int | None = None, drr: str = "none") -> np.ndarray:
    return restore(e, decode_raw(e, p, seed), drr)
