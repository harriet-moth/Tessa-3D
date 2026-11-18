"""Every Tessa call lives here: place a mesh on a lattice or chip, encode it once,
decode it per seed, compare resources, and read edge lengths from coherences.

Pure functions, numpy in and numpy out, so a Blender add-on can reuse them.
"""

import json
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, UCRYGate
from tessa import (Lattice, build_local_noise_model, decode, encode, graph_from_coupling,
                   prepare_field, probe_qubits, restore_field, slot_order)
from tessa.backends import DeviceSampler, aer_sampler, counts_of, device_of, noisy_sampler
from tessa.hardware import devices

from .mesh import bfs_order, edges, grow_patches, importance

MAX_ROWS, MAX_COLS = 10, 12  # largest lattice: the whole 10x12 ibm_phoenix chip
DEVICES = ("phoenix", "fez")  # Tessa's offline calibration snapshots
BACKENDS = ("noiseless", "flat noise", *DEVICES)
# chip backends: Tessa's group order | best data qubits first | best first, most curved vertices
# on the very best (see place)
PLACEMENTS = ("graph", "quality", "importance")
QUALITY_FILE = Path(__file__).with_name("quality.json")  # probe_qubits scores per chip snapshot
# What each slot's qubit holds. One value per slot, read in one basis: in theta (read
# in Z, Tessa's "single" mode) or in phi with theta fixed at pi/2 (read in X, the
# paper pipeline's "x" basis). Two values per slot (Tessa's "pair" mode): a colour
# field in phi, or more geometry (twice the vertices fit).
MODES = ("theta", "phi", "colour", "split")
MODE_ALIASES = {"single": "theta", "pair": "colour"}  # the names before the split mode
DRR_METHODS = ("none", "min_max", "mean_std", "percentile", "abs_max", "min_avg_max")
# Typical IBM-like machine averages (tessa.backend_rates fallbacks) for "flat noise".
IBM_LIKE_RATES = {"sx_error": 3e-4, "cz_error": 8e-3, "readout_error": 1e-2,
                  "sx_duration": 3.2e-8, "cz_duration": 1.6e-7, "t1": 3e-4, "t2": 2.5e-4}


@dataclass(frozen=True)
class Params:
    # encoder (changing any of these re-encodes)
    payload: str = "radial"        # "radial": one radius per vertex | "xyz": three coordinates
    layout: str = "mesh"           # "mesh": BFS neighbourhoods per cell | "naive": axis-major
    mode: str = "theta"            # one of MODES: θ (Z) | φ (X) | θ + colour φ | shape over θ and φ
    clamp: float | None = None     # theta clamp in radians; None = Tessa default
    clamp_b: float | None = None   # phi clamp (pair only)
    compression: float = 0.0       # percent (0-100) of multiplexer coefficients dropped
    quant_bits: int | None = None
    headroom: float = 0.1          # padding on the value_range
    placement: str = "quality"     # chips only: one of PLACEMENTS
    # readout (changing these re-decodes; a chip backend also changes the graph)
    backend: str = "noiseless"     # one of BACKENDS
    shots: int = 4096
    noise_strength: float = 1.0    # scales "flat noise" only
    seed: int = 0

    @property
    def encoder_key(self):
        graph = self.backend if self.backend in DEVICES else "lattice"
        return (self.payload, self.layout, self.mode, self.clamp, self.clamp_b,
                self.compression, self.quant_bits, self.headroom, graph,
                self.placement if graph != "lattice" else "graph")

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
    quality: np.ndarray | None = None  # chip only: probe_qubits score per data qubit
    field: np.ndarray | None = None    # (N, comps): 0 = value in θ, 1 = in φ (split mode)
    mode: str = "theta"

    @property
    def fields(self) -> np.ndarray:
        return np.zeros(self.slots.shape, int) if self.field is None else self.field

    @property
    def pair(self) -> bool:
        return self.enc.mode == "pair"

    @property
    def used_groups(self) -> np.ndarray:
        """Indices (graph.groups order) of the data qubits holding mesh values."""
        return np.unique(np.searchsorted(cells(self.graph)[:, 1], self.slots, side="right"))

    @property
    def used_qubits(self) -> list[int]:
        """The data qubits holding mesh values plus their address qubits. The rest of a
        chip only holds constant padding: one rotation each, no two-qubit gates."""
        gs = [self.graph.groups[i] for i in self.used_groups]
        return sorted({q for g in gs for q in (g.data_qubit, *g.address_qubits)})

    @property
    def stats(self) -> dict:
        n = self.slots.size * (2 if self.mode == "colour" else 1)
        q = None if self.quality is None else float(self.quality[self.used_groups].mean())
        used = len(self.used_qubits)
        return {"graph": self.name, "quality": q, "qubits": used, "graph_qubits": self.graph.n_qubits,
                "values": n, "capacity": self.graph.capacity * (2 if self.pair else 1),
                "ratio": n / used, "depth": self.circuit.depth(),
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


def place(graph, n, comps, order=None, quality=None, weight=None, per_slot=1, grow=None):
    """Capacity slot and field (0 = θ, 1 = φ) of every value, each shape (n, comps).

    ``order=None`` is the naive axis-major layout (all x, then all y, then all z).
    Otherwise each cell takes the next mesh-neighbouring vertices from ``order``
    and stores them as [x1..xm, y1..ym, z1..zm], so a vertex's components share a
    cell and every cell holds a smooth patch of the surface.

    Cells are filled in graph order, or, given ``quality`` (one score per data
    qubit), in Tessa's ``slot_order``: best data qubit first, so a mesh smaller
    than the chip lands only on its best qubits.

    ``weight`` (one importance score per vertex) also matches the most important
    vertices to the best of those qubits. Naive layout: vertices fill the ranked
    slots most important first. Mesh layout: the neighbourhood patches are cut
    as usual, then within each patch size the patches with the highest mean
    importance swap onto the best data qubits, so every patch stays whole.

    ``per_slot=2`` (split mode) gives every slot two positions, θ then φ: a data
    qubit with C slots takes 2C values, its θ field filled first and its φ field
    after, so each field holds runs of one coordinate and stays smooth.
    """
    ranked = np.arange(graph.capacity) if quality is None else slot_order(graph, quality)
    cap = graph.capacity
    if order is None:
        if n * comps > per_slot * cap:
            raise ValueError(f"{n * comps} values exceed capacity {per_slot * cap}")
        pos = np.empty((n, comps), int)  # position t: slot ranked[t % cap], field t // cap
        first = np.arange(n) if weight is None else np.argsort(-weight, kind="stable")
        pos[first] = np.arange(comps) * n + np.arange(n)[:, None]
        return ranked[pos % cap], pos // cap
    bounds = cells(graph)
    cell_of = np.searchsorted(bounds[:, 1], ranked, side="right")
    seq = cell_of[np.r_[True, cell_of[1:] != cell_of[:-1]]]  # cells in ranked order
    if n * comps > per_slot * cap:
        raise ValueError(f"{n} vertices x {comps} values do not fit {per_slot * cap} positions "
                         f"({cap} slots x {per_slot}); try the radial payload"
                         + ("" if per_slot > 1 else ", or Mode 'shape θ+φ' (two values per slot)"))
    plan, k = [], 0  # (cell, whole vertices it takes), best cell first
    for c in seq:
        m = min(per_slot * (bounds[c, 1] - bounds[c, 0]) // comps, n - k)
        if m > 0:
            plan.append((c, m))
            k += m
        if k == n:
            break
    if grow is None:  # consecutive runs of ``order``
        patches = np.split(order[:k], np.cumsum([m for _, m in plan])[:-1])
        rest = order[k:]
    else:  # grow(sizes) -> (patches, leftover vertices), e.g. compact mesh patches
        patches, rest = grow([m for _, m in plan])
    if weight is not None:  # same-size patches: most important onto the best cells
        for m in {m for _, m in plan}:
            idx = [i for i, (_, mm) in enumerate(plan) if mm == m]  # best cell first
            mine = sorted((patches[i] for i in idx), key=lambda p: -weight[p].mean())
            for i, p in zip(idx, mine):
                patches[i] = p
    slots, field = np.empty((n, comps), int), np.empty((n, comps), int)
    for (c, m), p in zip(plan, patches):
        lo, size = bounds[c, 0], bounds[c, 1] - bounds[c, 0]
        pos = np.arange(comps) * m + np.arange(m)[:, None]  # [x1..xm, y1..ym, z1..zm]
        slots[p], field[p] = lo + pos % size, pos // size
    if k < n:  # whole vertices ran out: the rest fill the cells' spare positions, best
        used = dict(plan)  # cell first, so a few vertices straddle two data qubits
        spare = [(bounds[c, 0] + q % size, q // size) for c in seq
                 for size in [bounds[c, 1] - bounds[c, 0]]
                 for q in range(comps * used.get(c, 0), per_slot * size)]
        spill = np.array(spare[: (n - k) * comps]).reshape(n - k, comps, 2)
        slots[rest], field[rest] = spill[..., 0], spill[..., 1]
    return slots, field


def fill(graph, slots, vals) -> np.ndarray:
    """Capacity-length payload: values at their slots, unused slots padded with their
    cell's mean so padding keeps each cell smooth (and compressible)."""
    arr = np.full(graph.capacity, np.nan)
    arr[slots] = vals
    for lo, hi in cells(graph):
        cell = arr[lo:hi]
        if np.isnan(cell).any() and not np.isnan(cell).all():
            cell[np.isnan(cell)] = np.nanmean(cell)
    arr[np.isnan(arr)] = np.nanmean(arr) if len(vals) else 0.0
    return arr


def fit_lattice(n: int, comps: int = 1, mesh: bool = True, per_slot: int = 1) -> Lattice:
    """Smallest near-square Lattice (cols - rows <= 2, up to 10x12) holding the mesh."""
    shapes = sorted(((r, c) for r in range(2, MAX_ROWS + 1)
                     for c in range(r, min(r + 2, MAX_COLS) + 1)), key=lambda rc: (rc[0] * rc[1], rc[1] - rc[0]))
    for whole in (True, False):  # prefer every vertex whole on one data qubit
        for r, c in shapes:
            lat = Lattice(r, c)
            usable = (sum(per_slot * (hi - lo) // comps for lo, hi in cells(lat)) if mesh and whole
                      else per_slot * lat.capacity // comps)
            if usable >= n:
                return lat
    raise ValueError(f"{n} vertices x {comps} values exceed the {MAX_ROWS}x{MAX_COLS} lattice; "
                     "try the radial payload, or Mode 'shape θ+φ' (two values per slot)")


def vertex_qubits(e: Encoded) -> np.ndarray:
    """Data qubit (physical index on a chip) holding each vertex's first value."""
    cell = np.searchsorted(cells(e.graph)[:, 1], e.slots[:, 0], side="right")
    return np.array([e.graph.groups[i].data_qubit for i in cell])


def qubit_map(e: Encoded) -> dict:
    """Where every value sits on the qubits, for drawing the lattice or chip.

    ``coords``/``edges`` draw the qubits and couplers (the chip's own drawing for a
    device, rows and columns for a lattice). ``groups`` lists each data qubit with its
    address qubits. ``vertices[i][c] = [group, address, field]``: value ``c`` of
    vertex ``i`` is in that group's slot whose address qubit ``groups[g].addresses[j]``
    reads bit ``j`` of ``address`` (the bit order Tessa's decoder uses), in ``field``
    0 (θ) or 1 (φ, split mode only). In colour mode the same slot also holds the
    vertex colour in φ.
    """
    g = e.graph
    if isinstance(g, Lattice):
        coords = [list(g.site_of_qubit(q))[::-1] for q in range(g.n_qubits)]
        couplers = [(g.qubit_index(r, c), g.qubit_index(r + dr, c + dc))
                    for r in range(g.rows) for c in range(g.cols)
                    for dr, dc in ((0, 1), (1, 0)) if r + dr < g.rows and c + dc < g.cols]
    else:
        coords = snapshot_json(e.name.split()[0].removeprefix("ibm_"))["coords"]
        couplers = sorted(tuple(sorted(c)) for c in g.edges)
    bounds = cells(g)
    cell = np.searchsorted(bounds[:, 1], e.slots, side="right")
    return {"coords": coords, "edges": [list(map(int, c)) for c in couplers],
            "groups": [{"data": grp.data_qubit, "addresses": list(grp.address_qubits)}
                       for grp in g.groups],
            "vertices": np.stack([cell, e.slots - bounds[cell, 0], e.fields], -1).tolist()}


# --- encode / decode -------------------------------------------------------------
def encode_mesh(verts, faces, p: Params) -> Encoded:
    mode = MODE_ALIASES.get(p.mode, p.mode)
    if mode not in MODES:
        raise ValueError(f"unknown mode {p.mode!r}; use one of {MODES}")
    per = 2 if mode == "split" else 1
    verts = np.asarray(verts, dtype=float)
    vals, centre, dirs = to_payload(verts, p.payload)
    n, comps = vals.shape
    flat, prep = prepare_field(vals.ravel(), quant_bits=p.quant_bits)
    top = float(np.abs(flat).max() or 1.0) * (1 + p.headroom)
    rng = (-top, top) if p.payload == "xyz" else (0.0, top)
    order = bfs_order(n, faces) if p.layout == "mesh" else None
    quality = None
    if p.backend in DEVICES:
        graph, name = devices.graph(p.backend), f"ibm_{p.backend} (whole chip)"
        quality = chip_quality(p.backend)
    else:
        graph = fit_lattice(n, comps, order is not None, per)
        name = f"{graph.rows}x{graph.cols} lattice"
    ranked = p.placement in ("quality", "importance")
    weight = importance(verts, faces) if p.placement == "importance" and quality is not None else None
    grow = (lambda sizes: grow_patches(verts, faces, sizes)) if order is not None else None
    slots, field = place(graph, n, comps, order, quality if ranked else None, weight, per, grow)
    vals = flat.reshape(n, comps)
    a = fill(graph, slots[field == 0], vals[field == 0])
    kw = dict(value_range=rng, compression=p.compression)
    single, _ = encode(graph, a, clamp=p.clamp, **kw)
    colour = None
    if mode == "theta":
        circuit, enc = single, _
    elif mode == "phi":  # θ = π/2 everywhere (0.5 is mid-range at any clamp), value in φ
        c_b = p.clamp_b or 0.0
        circuit, enc = encode(graph, np.full(graph.capacity, 0.5), a, clamp_b=c_b,
                              value_range=(0.0, 1.0), value_range_b=rng,
                              compression=p.compression)
        circuit = circuit.copy()
        circuit.ry(-np.pi / 2, sorted({g.data_qubit for g in graph.groups}))  # Z now reads X
        # <X> = cos φ, so the tilted circuit decodes as one field: φ = arccos(<Z>)
        enc = replace(enc, mode="single", clamps=(c_b,), ranges=(rng,))
    else:  # Tessa pair mode: a second field in φ, read in the X basis
        if mode == "colour":
            colour = hue(verts)
            b, range_b = fill(graph, slots, np.repeat(colour[:, None], comps, 1)), (0.0, 1.0)
        else:
            b, range_b = fill(graph, slots[field == 1], vals[field == 1]), rng
        circuit, enc = encode(graph, a, b, clamp=p.clamp, clamp_b=p.clamp_b,
                              value_range_b=range_b, **kw)
    return Encoded(verts, faces, graph, name, circuit, enc, prep, slots, p.payload, centre,
                   dirs, colour, two_qubit(single), quality,
                   field if mode == "split" else None, mode)


@cache
def snapshot_json(device: str) -> dict:
    """Tessa's stored calibration snapshot of ``device`` ("phoenix" or "fez")."""
    return json.loads((devices.SNAPSHOT_DIR / f"ibm_{device}.json").read_text())


def probe_chip(device: str) -> np.ndarray:
    """Tessa's probe_qubits on the calibrated simulator: per data qubit, the fraction
    of test symbols read back right (about 8 s per chip)."""
    graph = devices.graph(device)
    return probe_qubits(graph, sampler(graph, Params(backend=device), 0))


@cache
def chip_quality(device: str) -> np.ndarray:
    """Quality per data qubit, from QUALITY_FILE when it matches the snapshot's
    calibration date, otherwise probed now."""
    try:
        stored = json.loads(QUALITY_FILE.read_text())[device]
        if stored["calibrated_at"] == snapshot_json(device)["calibrated_at"]:
            return np.array(stored["quality"])
    except (FileNotFoundError, KeyError):
        pass
    return probe_chip(device)


def write_quality():
    """Regenerate QUALITY_FILE: ``python -m tessa_ripple.bridge``."""
    QUALITY_FILE.write_text(json.dumps({d: {"calibrated_at": snapshot_json(d)["calibrated_at"],
                                            "quality": probe_chip(d).round(4).tolist()}
                                        for d in DEVICES}, indent=1) + "\n")


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
    if e.mode == "split":  # each value from its own field
        return np.where(e.fields == 1, b[e.slots], a[e.slots]).ravel(), None
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
    placed = {lay: encode_mesh(verts, faces, replace(p, layout=lay)).slots for lay in ("mesh", "naive")}
    out["same"] = bool(np.array_equal(placed["mesh"], placed["naive"]))  # e.g. radial on a ring-ordered mesh
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
    n = e.stats["values"]
    rng = np.random.default_rng(0)

    def routed(qc, backend=phoenix, layout=None):
        tq = transpile(qc, backend, initial_layout=layout, optimization_level=1, seed_transpiler=0)
        free = transpile(qc, basis_gates=["cz", "sx", "rz", "x"], optimization_level=1, seed_transpiler=0)
        cz = two_qubit(tq)
        return {"twoq": cz, "depth": tq.depth(), "swaps": max(0, round((cz - two_qubit(free)) / 3))}

    # generic random data for every row, so no encoding gets credit for this mesh; Tessa's
    # goes in the mesh's own slots, the rest of the graph padded as in encode_mesh
    if e.mode == "colour":
        where = [e.slots, e.slots]
    else:
        where = [e.slots[e.fields == f] for f in ((0, 1) if e.pair else (0,))]
    rand = [fill(e.graph, w, rng.random(w.shape)) for w in where]
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
        dict(name="Tessa", qubits=len(e.used_qubits), settings=2 if e.pair else 1, configs=2 ** max_deg, **tessa),
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


if __name__ == "__main__":
    write_quality()
