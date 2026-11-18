"""Every Tessa call lives here: place a mesh on a lattice or chip, encode it once,
decode it per seed, compare resources, and read edge lengths from coherences.

Pure functions, numpy in and numpy out, so a Blender add-on can reuse them.
"""

from dataclasses import dataclass, replace
from functools import cache

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, UCRYGate
from tessa import (Lattice, build_local_noise_model, decode, encode, graph_from_coupling,
                   prepare_field, restore_field)
from tessa.backends import DeviceSampler, aer_sampler, counts_of, device_of, noisy_sampler
from tessa.hardware import devices

from .mesh import bfs_order, edges

MAX_ROWS, MAX_COLS = 10, 12  # largest lattice: the whole 10x12 ibm_phoenix chip
DEVICES = ("phoenix", "fez")  # Tessa's offline calibration snapshots
BACKENDS = ("noiseless", "flat noise", *DEVICES)
DRR_METHODS = ("none", "min_max", "mean_std", "percentile", "abs_max", "min_avg_max")
# Typical IBM-like machine averages (tessa.backend_rates fallbacks) for "flat noise".
IBM_LIKE_RATES = {"sx_error": 3e-4, "cz_error": 8e-3, "readout_error": 1e-2,
                  "sx_duration": 3.2e-8, "cz_duration": 1.6e-7, "t1": 3e-4, "t2": 2.5e-4}


@dataclass(frozen=True)
class Params:
    # encoder (changing any of these re-encodes)
    payload: str = "radial"        # "radial": one radius per vertex | "xyz": three coordinates
    layout: str = "mesh"           # "mesh": BFS neighbourhoods per cell | "naive": axis-major
    mode: str = "single"           # "single" | "pair" (shape in theta, vertex colour in phi)
    clamp: float | None = None     # theta clamp in radians; None = Tessa default
    clamp_b: float | None = None   # phi clamp (pair only)
    compression: float = 0.0       # percent (0-100) of multiplexer coefficients dropped
    quant_bits: int | None = None
    headroom: float = 0.1          # padding on the value_range
    # readout (changing these re-decodes; a chip backend also changes the graph)
    backend: str = "noiseless"     # one of BACKENDS
    shots: int = 4096
    noise_strength: float = 1.0    # scales "flat noise" only
    seed: int = 0

    @property
    def encoder_key(self):
        graph = self.backend if self.backend in DEVICES else "lattice"
        return (self.payload, self.layout, self.mode, self.clamp, self.clamp_b,
                self.compression, self.quant_bits, self.headroom, graph)

    @property
    def readout_key(self):
        return (self.shots, self.backend, self.noise_strength, self.seed)


@dataclass(frozen=True)
class Encoded:
    verts: np.ndarray       # original (N, 3)
    faces: list
    graph: object           # tessa Lattice or a chip's TessaGraph
    name: str
    circuit: object
    enc: object             # tessa Encoding
    prep: object            # tessa FieldPrep (quantisation, DRR reference)
    slots: np.ndarray       # (N, comps): capacity slot of every encoded value
    payload: str
    centre: np.ndarray | None
    dirs: np.ndarray | None
    colour: np.ndarray | None   # original colour field in [0, 1] (pair mode)
    single_twoq: int            # two-qubit gates of the theta-only circuit

    @property
    def pair(self) -> bool:
        return self.enc.mode == "pair"

    @property
    def stats(self) -> dict:
        n = self.slots.size * (2 if self.pair else 1)
        return {"graph": self.name, "qubits": self.graph.n_qubits, "values": n,
                "capacity": self.graph.capacity * (2 if self.pair else 1),
                "ratio": n / self.graph.n_qubits, "depth": self.circuit.depth(),
                "twoq": two_qubit(self.circuit), "twoq_single": self.single_twoq}


def two_qubit(circuit) -> int:
    ops = circuit.count_ops()
    return sum(ops.get(g, 0) for g in ("cx", "cz", "ecr"))


# --- what is encoded -------------------------------------------------------------
def to_payload(verts, payload):
    """(values (N, comps), centre, unit directions). Radial: one radius per vertex
    from the centroid, so noise moves vertices along their direction from the centre."""
    if payload == "xyz":
        return verts.copy(), None, None
    centre = verts.mean(0)
    d = verts - centre
    r = np.linalg.norm(d, axis=1)
    return r[:, None], centre, d / np.where(r > 0, r, 1)[:, None]


def from_payload(e: Encoded, vals) -> np.ndarray:
    return vals if e.payload == "xyz" else e.centre + vals[:, :1] * e.dirs


def hue(verts) -> np.ndarray:
    """Vertex colour field for pair mode: hue from the angle around the vertical axis."""
    c = verts - verts.mean(0)
    return (np.arctan2(c[:, 1], c[:, 0]) / (2 * np.pi)) % 1.0


# --- where it goes on the chip ---------------------------------------------------
def cells(graph) -> np.ndarray:
    """(start, stop) capacity slots of each data qubit, in graph.groups order."""
    stops = np.cumsum([g.n_values for g in graph.groups])
    return np.column_stack([stops - [g.n_values for g in graph.groups], stops])


def place(graph, n, comps, order=None) -> np.ndarray:
    """Capacity slot of every value, shape (n, comps).

    ``order=None`` is the naive axis-major layout (all x, then all y, then all z).
    Otherwise each cell takes the next mesh-neighbouring vertices from ``order``
    and stores them as [x1..xm, y1..ym, z1..zm], so a vertex's components share a
    cell and every cell holds a smooth patch of the surface.
    """
    if order is None:
        if n * comps > graph.capacity:
            raise ValueError(f"{n * comps} values exceed capacity {graph.capacity}")
        return np.arange(comps) * n + np.arange(n)[:, None]
    slots, k = np.empty((n, comps), int), 0
    for lo, hi in cells(graph):
        m = min((hi - lo) // comps, n - k)
        if m > 0:
            slots[order[k:k + m]] = lo + np.arange(comps) * m + np.arange(m)[:, None]
            k += m
        if k == n:
            return slots
    raise ValueError(f"{n} vertices x {comps} values do not fit {graph.capacity} slots; "
                     "try the radial payload (one value per vertex)")


def fill(graph, slots, vals) -> np.ndarray:
    """Capacity-length payload: values at their slots, unused slots padded with their
    cell's mean so padding keeps each cell smooth (and compressible)."""
    arr = np.full(graph.capacity, np.nan)
    arr[slots] = vals
    for lo, hi in cells(graph):
        cell = arr[lo:hi]
        if np.isnan(cell).any() and not np.isnan(cell).all():
            cell[np.isnan(cell)] = np.nanmean(cell)
    arr[np.isnan(arr)] = np.nanmean(arr)
    return arr


def fit_lattice(n: int, comps: int = 1, mesh: bool = True) -> Lattice:
    """Smallest near-square Lattice (cols - rows <= 2, up to 10x12) holding the mesh."""
    shapes = sorted(((r, c) for r in range(2, MAX_ROWS + 1)
                     for c in range(r, min(r + 2, MAX_COLS) + 1)), key=lambda rc: (rc[0] * rc[1], rc[1] - rc[0]))
    for r, c in shapes:
        lat = Lattice(r, c)
        usable = sum((hi - lo) // comps for lo, hi in cells(lat)) if mesh else lat.capacity // comps
        if usable >= n:
            return lat
    raise ValueError(f"{n} vertices x {comps} values exceed the {MAX_ROWS}x{MAX_COLS} lattice; "
                     "try the radial payload (one value per vertex)")


def vertex_qubits(e: Encoded) -> np.ndarray:
    """Data qubit (physical index on a chip) holding each vertex's first value."""
    cell = np.searchsorted(cells(e.graph)[:, 1], e.slots[:, 0], side="right")
    return np.array([e.graph.groups[i].data_qubit for i in cell])


# --- encode / decode -------------------------------------------------------------
def encode_mesh(verts, faces, p: Params) -> Encoded:
    verts = np.asarray(verts, dtype=float)
    vals, centre, dirs = to_payload(verts, p.payload)
    n, comps = vals.shape
    flat, prep = prepare_field(vals.ravel(), quant_bits=p.quant_bits)
    top = float(np.abs(flat).max() or 1.0) * (1 + p.headroom)
    rng = (-top, top) if p.payload == "xyz" else (0.0, top)
    order = bfs_order(n, faces) if p.layout == "mesh" else None
    if p.backend in DEVICES:
        graph, name = devices.graph(p.backend), f"ibm_{p.backend} (whole chip)"
    else:
        graph = fit_lattice(n, comps, order is not None)
        name = f"{graph.rows}x{graph.cols} lattice"
    slots = place(graph, n, comps, order)
    a = fill(graph, slots, flat.reshape(n, comps))
    kw = dict(value_range=rng, compression=p.compression)
    single, _ = encode(graph, a, clamp=p.clamp, **kw)
    colour = None
    if p.mode == "pair":
        colour = hue(verts)
        b = fill(graph, slots, np.repeat(colour[:, None], comps, 1))
        circuit, enc = encode(graph, a, b, clamp=p.clamp, clamp_b=p.clamp_b,
                              value_range_b=(0.0, 1.0), **kw)
    else:
        circuit, enc = single, _
    return Encoded(verts, faces, graph, name, circuit, enc, prep, slots, p.payload, centre,
                   dirs, colour, two_qubit(single))


@cache
def _chip(name):
    """A chip's snapshot backend and per-qubit noise model, built once."""
    be, graph = devices.snapshot_backend(name), devices.graph(name)
    return be, build_local_noise_model(graph, be, phys_map=list(range(graph.n_qubits)))


def sampler(graph, p: Params, seed: int):
    if p.backend in DEVICES:  # same as tessa.calibrated_sampler, reusing the noise model
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2

        be, nm = _chip(p.backend)
        return DeviceSampler(SamplerV2.from_backend(AerSimulator(noise_model=nm), seed=seed),
                             be, noise_model=nm)
    if p.backend == "flat noise" and p.noise_strength > 0:
        s = p.noise_strength
        rates = {k: (v / s if k in ("t1", "t2") else v * s if k.endswith("error") else v)
                 for k, v in IBM_LIKE_RATES.items()}
        return noisy_sampler(graph, rates=rates, seed=seed)
    return aer_sampler(graph, seed=seed)


def decode_raw(e: Encoded, p: Params, seed: int | None = None):
    """One readout of the unchanged circuit: (values (N*comps,), colour (N,) or None)."""
    seed = p.seed if seed is None else seed
    out = decode(e.circuit, e.enc, sampler(e.graph, p, seed), shots=p.shots)
    a, b = out if isinstance(out, tuple) else (out, None)
    return a[e.slots].ravel(), (None if b is None else b[e.slots].mean(1))


def restore(e: Encoded, raw, drr: str = "none") -> np.ndarray:
    """Classical post-processing (DRR) and back to vertex positions (N, 3)."""
    vals = restore_field(raw, e.prep, dynamic_range=drr).reshape(e.slots.shape)
    return from_payload(e, vals)


def read_mesh(e: Encoded, p: Params, seed: int | None = None, drr: str = "none") -> np.ndarray:
    return restore(e, decode_raw(e, p, seed)[0], drr)


def mesh_error(verts, positions) -> float:
    """Mean absolute coordinate error as a fraction of the mesh's bounding box."""
    return float(np.abs(positions - verts).mean() / np.ptp(verts))


def fidelity_curves(verts, faces, p: Params, progress=None) -> dict:
    """Error vs shots, and error + depth vs compression for both layouts."""
    shots, comps, out, k = (256, 1024, 4096, 16384, 65536), (0, 25, 50, 75, 90), {}, 0
    total = len(shots) + 2 * len(comps)
    e = encode_mesh(verts, faces, p)
    out["shots"] = []
    for s in shots:
        progress and progress(k := k + 1, total)
        out["shots"].append((s, mesh_error(verts, read_mesh(e, replace(p, shots=s)))))
    for layout in ("mesh", "naive"):
        out[layout] = []
        for c in comps:
            progress and progress(k := k + 1, total)
            q = replace(p, layout=layout, compression=c)
            ee = encode_mesh(verts, faces, q)
            out[layout].append((c, ee.circuit.depth(), mesh_error(verts, read_mesh(ee, q))))
    return out


# --- "Why Tessa": the same values under other encodings, on a real chip graph -----
def resources(e: Encoded) -> list[dict]:
    """Qubits, two-qubit gates, depth, SWAPs and readout cost of ``e``'s values under
    Tessa and four baselines, transpiled onto the ibm_phoenix snapshot (Tessa on a
    whole-chip graph is transpiled onto its own chip)."""
    phoenix = devices.snapshot_backend("phoenix")
    n = e.slots.size * (2 if e.pair else 1)
    rng = np.random.default_rng(0)

    def routed(qc, backend=phoenix, layout=None):
        tq = transpile(qc, backend, initial_layout=layout, optimization_level=1, seed_transpiler=0)
        free = transpile(qc, basis_gates=["cz", "sx", "rz", "x"], optimization_level=1, seed_transpiler=0)
        cz = two_qubit(tq)
        return {"twoq": cz, "depth": tq.depth(), "swaps": max(0, round((cz - two_qubit(free)) / 3))}

    # generic random data for every row, so no encoding gets credit for this mesh
    rand = [rng.random(e.graph.capacity) for _ in range(2 if e.pair else 1)]
    generic, _ = encode(e.graph, *rand)
    if isinstance(e.graph, Lattice):  # lattice site (r, c) -> ibm_phoenix qubit (10 per row)
        r, c = np.divmod(np.arange(e.graph.n_qubits), e.graph.cols)
        tessa = routed(generic, layout=list(r * 10 + c if e.graph.cols <= 10 else c * 10 + r))
    else:
        chip = devices.snapshot_backend(e.name.split()[0].removeprefix("ibm_"))
        tessa = routed(generic, chip, list(range(e.graph.n_qubits)))
    k = int(np.ceil(np.log2(n)))
    amp = QuantumCircuit(k)
    vec = rng.random(2 ** k)
    amp.append(StatePreparation(vec / np.linalg.norm(vec)), range(k))
    frqi = QuantumCircuit(k + 1)
    frqi.h(range(k))
    frqi.append(UCRYGate(list(rng.random(2 ** k))), [k, *range(k)])
    a = int(np.argmin([j + np.ceil(n / 2 ** j) for j in range(12)]))  # fewest qubits
    m = int(np.ceil(n / 2 ** a))
    qcrank = QuantumCircuit(a + m)
    qcrank.h(range(a))
    for d in range(m):
        qcrank.append(UCRYGate(list(rng.random(2 ** a))), [a + d, *range(a)])
    max_deg = max(g.degree for g in e.graph.groups)
    rows = [
        dict(name="Tessa", qubits=e.graph.n_qubits, settings=2 if e.pair else 1, configs=2 ** max_deg, **tessa),
        dict(name="Angle (1 value/qubit)", qubits=n, settings=1, configs=1, twoq=0, depth=1, swaps=0),
        dict(name="QCrank (parallel QPIXL)", qubits=a + m, settings=1, configs=2 ** a, **routed(qcrank)),
        dict(name="FRQI", qubits=k + 1, settings=1, configs=2 ** k, **routed(frqi)),
        dict(name="Amplitude", qubits=k, settings="1 (no signs)", configs=2 ** k, **routed(amp)),
    ]
    for row in rows:
        row["fits"] = row["qubits"] <= phoenix.num_qubits
    rows[0]["fits"] = True  # Tessa's graph is always a whole chip or a sub-lattice of Phoenix
    return rows


# --- coherence readout: edge lengths straight from the encoded state --------------
PAIR = graph_from_coupling([(0, 1)])  # one data + one address qubit (degree one)


def read_edges(verts, faces, p: Params, seed: int | None = None):
    """Each mesh edge (i, j) and axis is encoded as a degree-one Tessa cell holding
    (v_i, v_j); a Hadamard on its address qubit makes K = 2 P(0) - 1 = cos(pi dv / 2),
    so |v_i - v_j| is read from the state's coherence, never decoded per vertex. This
    is the paper's coherence experiment B (isolated pairs on separate couplers, plus
    reference circuits), not the full-lattice version.
    Returns (edges (E, 2), measured lengths, exact lengths)."""
    seed = p.seed if seed is None else seed
    E = edges(faces)
    a, b = verts[E[:, 0]].ravel(), verts[E[:, 1]].ravel()  # (edge, axis) pairs
    # Every pair shares one zoomed scale: the range spans the largest component
    # difference, so K uses its whole [-1, 1] range instead of sitting near 1.
    half = float(np.abs(a - b).max() or 1.0) * (1 + p.headroom) / 2
    addr = PAIR.groups[0].address_qubits[0]

    def pair_circuit(vi, vj):
        mid = (vi + vj) / 2
        qc, _ = encode(PAIR, [vi, vj], value_range=(mid - half, mid + half))
        qc.h(addr)
        qc.add_register(ClassicalRegister(1))
        qc.measure(addr, 0)
        return qc

    circuits = [pair_circuit(vi, vj) for vi, vj in zip(a, b)]
    if p.backend in DEVICES:  # spread the pairs over the chip's lowest-error couplers
        be, _ = _chip(p.backend)
        snap = be.snapshot
        couplers = [c for _, c in sorted(zip([cz[0] or 1.0 for cz in snap["cz"]], snap["edges"]))]
        layouts = [couplers[k % len(couplers)] for k in range(len(circuits))]
        used = couplers[: min(len(couplers), len(circuits))]
        smp = sampler(None, p, seed)
        run = [transpile(qc, be, initial_layout=lay, optimization_level=1, seed_transpiler=0)
               for qc, lay in zip(circuits + [pair_circuit(0.0, 0.0)] * len(used), layouts + used)]
        ref_of = [k % len(couplers) for k in range(len(circuits))]
    else:
        smp = sampler(PAIR, p, seed)
        run = transpile(circuits + [pair_circuit(0.0, 0.0)], device_of(smp),
                        optimization_level=1, seed_transpiler=0)
        ref_of = [0] * len(circuits)
    result = smp.run(run, shots=p.shots).result()
    K = np.array([2 * counts_of(result, k).get("0", 0) / p.shots - 1 for k in range(len(run))])
    # Reference pairs (v_i = v_j, exact K = 1) measure each coupler's contrast loss;
    # dividing by it removes the noise's shrink of K, as in the paper's reference circuit.
    K = K[: len(circuits)] / np.maximum(K[len(circuits):][ref_of], 1e-3)
    dv = 2 * np.arccos(np.clip(K, -1, 1)) / np.pi * (2 * half)  # |v_i - v_j| per axis
    measured = np.sqrt((dv.reshape(-1, 3) ** 2).sum(1))
    return E, measured, np.linalg.norm(verts[E[:, 0]] - verts[E[:, 1]], axis=1)
