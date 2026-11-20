"""Every Tessa call lives here: place a mesh on a lattice or chip, encode it once,
decode it per seed, compare resources, and read edge lengths from coherences.

Pure functions, numpy in and numpy out, so a Blender add-on can reuse them.

Wherever Tessa already implements a step, this module calls it rather than
re-deriving it: ``encode``/``decode`` and ``prepare_field``/``restore_field`` for the
codec; ``slot_order``, ``probe_qubits`` and ``_slot_ranges`` for placement; the
samplers (``aer_sampler``, ``noisy_sampler``, ``backends.aer._aer_sampler``) with
``build_local_noise_model`` and ``backend_rates``; ``hardware.devices`` for chip
snapshots; ``hardware.pipeline._tilted``, ``_angle_readout``, ``_transition_frames``
and ``ideal_frames`` for the morph basis sweep; ``rectangular_layout`` and
``square_lattice_edges`` for lattice geometry; and ``DYNAMIC_RANGE_METHODS``.
"""

import json
from dataclasses import dataclass, field as dc_field, replace
from types import SimpleNamespace
from functools import cache, cached_property
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, UCRYGate
from tessa import (DYNAMIC_RANGE_METHODS, Lattice, backend_rates, build_local_noise_model, decode,
                   encode, graph_from_coupling, prepare_field, probe_qubits, rectangular_layout,
                   restore_field, slot_order, square_lattice_edges)
from tessa.backends import aer_sampler, counts_of, device_of, noisy_sampler
from tessa.backends.aer import _TWO_QUBIT_GATES, _aer_sampler
from tessa.codec import _normalise
from tessa.core.mode import DEFAULT_CLAMP, get_mode
from tessa.hardware import devices
from tessa.hardware.pipeline import _angle_readout, _tilted, _transition_frames, ideal_frames
from tessa.symbols import _slot_ranges

from .mesh import bfs_order, edge_lengths, edges, grow_patches, importance
from .morph import correspond, slerp

MAX_ROWS, MAX_COLS = 10, 12  # largest lattice: the whole ibm_phoenix chip, turned on its side
PHOENIX_ROWS, PHOENIX_COLS = 12, 10  # ibm_phoenix: 12 rows of 10 qubits, row-major
DEVICES = ("phoenix", "fez")  # Tessa's offline calibration snapshots
BACKENDS = ("noiseless", "flat noise", *DEVICES)
# chip backends: Tessa's group order | best data qubits first | best first, most curved vertices
# on the very best (see place)
PLACEMENTS = ("graph", "quality", "importance")
QUALITY_FILE = Path(__file__).with_name("quality.json")  # probe_qubits scores per chip snapshot
# What each slot's qubit holds. One value per slot, read in one basis: in theta (read
# in Z, Tessa's "single" mode) or in phi with theta fixed at pi/2 (read in X, the
# paper pipeline's "x" basis). Two values per slot (Tessa's "pair" mode): a colour
# field in phi, or more geometry (twice the vertices fit). Morph: mesh A in theta, a
# second mesh B (one matched point per A vertex) in phi, in the same slots.
MODES = ("theta", "phi", "colour", "split", "morph")
# morph meshes too big for one run are read in batches; on a chip, later batches start
# on the next-best data qubits (spreading the best ones) or reuse the same ones
BATCH_PLACEMENTS = ("next-best", "same")
BATCH_SEED_STEP = 1000  # batch k of a readout with seed s uses seed s + 1000 k
# Morph is read by a basis sweep: at t the data qubits are measured along
# cos(alpha) Z + sin(alpha) X, alpha = t * pi/2. How that readout becomes values:
# "transition" (default) divides out sin(theta) with an alpha = 0 (Z) readout, as Tessa's
# transition and InterwovenQPIXL's tilted_readout.py do (exactly A at t = 0, exactly B at
# t = 1); "raw" maps arccos(E) through A's map ("B seen through A" at t = 1)
SWEEP_DECODES = ("transition", "raw")
MODE_ALIASES = {"single": "theta", "pair": "colour"}  # the names before the split mode
DRR_METHODS = DYNAMIC_RANGE_METHODS  # Tessa's dynamic range recovery methods


class _NoCalibration(dict):
    """A backend ``Target`` with no entries, so ``tessa.backend_rates`` returns its fallbacks.

    Attributes:
        qubit_properties: Empty, so no qubit reports T1 or T2.
    """
    qubit_properties = ()


# Typical IBM-like machine averages (tessa.backend_rates fallbacks) for "flat noise".
IBM_LIKE_RATES = backend_rates(SimpleNamespace(target=_NoCalibration()))


@dataclass(frozen=True)
class Params:
    """Every setting of one encode + readout.

    Encoder settings (changing any re-encodes):

    Attributes:
        payload: "radial" (one radius per vertex from the centroid) or "xyz" (three
            coordinates per vertex).
        layout: "mesh" (BFS neighbourhood patches per cell) or "naive" (axis-major).
        mode: One of MODES (or a MODE_ALIASES name): θ (Z), φ (X), θ + colour φ,
            shape over θ and φ, or morph A→B.
        clamp: θ clamp in radians; None for Tessa's default.
        clamp_b: φ clamp in radians (pair modes); None for Tessa's default.
        compression: Percent (0-100) of multiplexer coefficients dropped.
        quant_bits: Quantisation bits for ``prepare_field``; None for none.
        headroom: Fractional padding on the value range.
        placement: Chips only: one of PLACEMENTS.
        target: Morph: mesh B's OBJ path.
        shared_topology: Morph: pair B with A by vertex index (else closest point).
        batch_placement: Chips with several runs only: one of BATCH_PLACEMENTS.

    Readout settings (changing these re-decodes; a chip backend also changes the graph):

    Attributes:
        backend: One of BACKENDS.
        shots: Shots per measurement setting.
        noise_strength: Scales the "flat noise" rates only.
        seed: Sampler seed of the first readout.
        morph_t: Morph: the basis sweep reads at alpha = t * pi/2 (0 = A, 1 = B).
        sweep_decode: Morph: one of SWEEP_DECODES.
    """
    # encoder (changing any of these re-encodes)
    payload: str = "radial"
    layout: str = "mesh"
    mode: str = "theta"
    clamp: float | None = None
    clamp_b: float | None = None
    compression: float = 0.0
    quant_bits: int | None = None
    headroom: float = 0.1
    placement: str = "quality"
    target: str | None = None
    shared_topology: bool = False
    batch_placement: str = "next-best"
    # readout (changing these re-decodes; a chip backend also changes the graph)
    backend: str = "noiseless"
    shots: int = 4096
    noise_strength: float = 1.0
    seed: int = 0
    morph_t: float = 0.0
    sweep_decode: str = "transition"

    @property
    def canonical_mode(self) -> str:
        """``mode`` with old names (MODE_ALIASES) mapped to their MODES name.

        Returns:
            str: One of MODES (or an unknown name unchanged).
        """
        return MODE_ALIASES.get(self.mode, self.mode)

    @property
    def encoder_key(self):
        """Cache key of everything that changes the encoded circuit.

        Morph-only and chip-only settings join the key only when they apply, so
        toggling an unused setting never re-encodes.

        Returns:
            tuple: Hashable key; equal keys give the same ``encode_mesh`` result.
        """
        graph = self.backend if self.backend in DEVICES else "lattice"
        morph = self.canonical_mode == "morph"
        return (self.payload, self.layout, self.mode, self.clamp, self.clamp_b,
                self.compression, self.quant_bits, self.headroom, graph,
                self.placement if graph != "lattice" else "graph",
                *((self.target, self.shared_topology) if morph else ()),
                *((self.batch_placement,) if morph and graph != "lattice" else ()))

    @property
    def readout_key(self):
        """Cache key of everything that changes a readout of one encoding.

        Morph keys on ``morph_t`` and ``sweep_decode``: the sweep reads in a basis set by t.

        Returns:
            tuple: Hashable key; equal keys give the same decoded values.
        """
        morph = self.canonical_mode == "morph"
        return (self.shots, self.backend, self.noise_strength, self.seed,
                *((self.morph_t, self.sweep_decode) if morph else ()))


@dataclass(frozen=True)
class Encoded:
    """A mesh encoded once, with everything needed to decode it and map it back.

    Attributes:
        verts: (N, 3) original vertex positions.
        faces: Polygon vertex-index lists.
        graph: Tessa ``Lattice`` or a chip's ``TessaGraph``.
        name: Graph label, e.g. "4x5 lattice" or "ibm_fez (whole chip)".
        circuit: The (first run's) encoding circuit.
        enc: The (first run's) Tessa ``Encoding``.
        prep: Tessa ``FieldPrep`` of the main field (quantisation, DRR reference).
        slots: (N, comps) capacity slot of every encoded value.
        payload: "radial" or "xyz".
        centre: (3,) centre radial values (and morph xyz) are measured from, or None.
        dirs: (N, 3) unit directions from ``centre`` (radial), or None.
        colour: (N,) original colour field in [0, 1] (colour mode), or None.
        single_twoq: Two-qubit gates of the θ-only circuit.
        quality: Chip only: ``probe_qubits`` score per data qubit, else None.
        field: (N, comps) 0 = value in θ, 1 = in φ (split mode), else None.
        mode: One of MODES.
        parts: Several runs: one (circuit, enc) per batch, parts[0] being
            (circuit, enc); empty for one run.
        batch: Several runs: (N,) each vertex's batch, else None.
        target: Morph: (N, 3) B's matched point per A vertex (aligned), else None.
        dirs_b: Radial morph: (N, 3) unit directions of ``target`` from ``centre``.
        prep_b: Morph: Tessa ``FieldPrep`` of B's field.
        match: Morph: the :func:`morph.correspond` dict (method, coverage, B mesh).
        _sweeps: Cache of :func:`sweep_circuit` results per (part, alpha).
    """
    verts: np.ndarray
    faces: list
    graph: object
    name: str
    circuit: object
    enc: object
    prep: object
    slots: np.ndarray
    payload: str
    centre: np.ndarray | None
    dirs: np.ndarray | None
    colour: np.ndarray | None
    single_twoq: int
    quality: np.ndarray | None = None
    field: np.ndarray | None = None
    mode: str = "theta"
    parts: tuple = ()
    batch: np.ndarray | None = None
    target: np.ndarray | None = None
    dirs_b: np.ndarray | None = None
    prep_b: object = None
    match: dict | None = None
    _sweeps: dict = dc_field(default_factory=dict, compare=False, repr=False)

    @property
    def fields(self) -> np.ndarray:
        """Field of every value, all 0 (θ) outside split mode.

        Returns:
            np.ndarray: (N, comps) int, 0 = θ, 1 = φ.
        """
        return np.zeros(self.slots.shape, int) if self.field is None else self.field

    @property
    def runs(self) -> int:
        """Number of circuits one readout runs (1 unless the mesh was batched).

        Returns:
            int: At least 1.
        """
        return max(1, len(self.parts))

    @property
    def batches(self) -> np.ndarray:
        """Each vertex's batch (all 0 for one run).

        Returns:
            np.ndarray: (N,) int batch index.
        """
        return np.zeros(len(self.slots), int) if self.batch is None else self.batch

    @property
    def target_r(self) -> np.ndarray | None:
        """Morph: distance of each matched B point from ``centre``.

        Returns:
            np.ndarray | None: (N,) radii, or None outside morph.
        """
        return None if self.target is None else np.linalg.norm(self.target - self.centre, axis=1)

    @property
    def pair(self) -> bool:
        """Whether the encoding uses Tessa's pair mode (a second field in φ).

        Returns:
            bool: True for colour, split and morph.
        """
        return self.enc.mode == "pair"

    @property
    def device(self) -> str | None:
        """The chip snapshot of a whole-chip graph.

        Returns:
            str | None: "phoenix" or "fez", or None for a lattice.
        """
        return None if isinstance(self.graph, Lattice) else self.name.split()[0].removeprefix("ibm_")

    @cached_property
    def groups_of(self) -> np.ndarray:
        """The data qubit holding every encoded value.

        Returns:
            np.ndarray: (N, comps) int, an index into ``graph.groups``.
        """
        return slot_groups(self.graph, self.slots)

    @property
    def straddles(self) -> bool:
        """Whether some vertex's values sit on more than one data qubit (e.g. naive xyz).

        Returns:
            bool: True if any vertex is split across data qubits.
        """
        return bool((self.groups_of != self.groups_of[:, :1]).any())

    @property
    def used_groups(self) -> np.ndarray:
        """Data qubits holding mesh values.

        Returns:
            np.ndarray: Sorted unique indices into ``graph.groups``.
        """
        return np.unique(self.groups_of)

    @cached_property
    def used_qubits(self) -> list[int]:
        """The data qubits holding mesh values plus their address qubits.

        The rest of a chip only holds constant padding: one rotation each, no
        two-qubit gates.

        Returns:
            list[int]: Sorted qubit indices (physical on a chip).
        """
        gs = [self.graph.groups[i] for i in self.used_groups]
        return sorted({q for g in gs for q in (g.data_qubit, *g.address_qubits)})

    @cached_property
    def stats(self) -> dict:
        """Resource summary for the info panel and the web API.

        Colour mode counts its N distinct hues as extra values (one per vertex,
        repeated in every slot of that vertex); morph counts one B value in φ beside
        every A value in θ.

        Returns:
            dict: ``graph`` (name), ``quality`` (mean score of used data qubits or
            None), ``qubits`` (used), ``graph_qubits``, ``values``, ``capacity``,
            ``ratio`` (values per used qubit), ``depth``, ``twoq``, ``twoq_single``
            (θ field alone), ``runs`` and ``values_per_run`` (list).
        """
        # colour mode repeats one hue per vertex in all its slots: N distinct extra values;
        # morph holds one B value in φ beside every A value in θ
        n = self.slots.size * (2 if self.mode == "morph" else 1) + (len(self.slots) if self.mode == "colour" else 0)
        q = None if self.quality is None else float(self.quality[self.used_groups].mean())
        used = len(self.used_qubits)
        per_run = np.bincount(self.batches, minlength=self.runs) * (n // len(self.slots))
        return {"graph": self.name, "quality": q, "qubits": used, "graph_qubits": self.graph.n_qubits,
                "values": n, "capacity": self.graph.capacity * (2 if self.pair else 1),
                "ratio": n / used, "depth": self.circuit.depth(),
                "twoq": two_qubit(self.circuit), "twoq_single": self.single_twoq,
                "runs": self.runs, "values_per_run": per_run.tolist()}


def two_qubit(circuit) -> int:
    """Count two-qubit gates, using Tessa's own set of two-qubit gate names.

    Args:
        circuit: A qiskit ``QuantumCircuit``.

    Returns:
        int: Number of two-qubit gates.
    """
    return sum(n for g, n in circuit.count_ops().items() if g in _TWO_QUBIT_GATES)


# --- what is encoded -------------------------------------------------------------
def radial(points, centre):
    """Radii and unit directions of ``points`` from ``centre``.

    Args:
        points: (N, 3) positions.
        centre: (3,) origin.

    Returns:
        tuple: ``(r, dirs)``: (N, 1) radii and (N, 3) unit directions (zero for a
        point on the centre).
    """
    d = points - centre
    r = np.linalg.norm(d, axis=1)
    return r[:, None], d / np.where(r > 0, r, 1)[:, None]


def to_payload(verts, payload):
    """The values to encode for a mesh.

    Radial: one radius per vertex from the centroid, so noise moves vertices along
    their direction from the centre.

    Args:
        verts: (N, 3) vertex positions.
        payload: "radial" or "xyz".

    Returns:
        tuple: ``(values, centre, dirs)``: values (N, comps) (comps 1 radial, 3 xyz),
        the centroid (3,) and unit directions (N, 3); centre and dirs are None for xyz.
    """
    if payload == "xyz":
        return verts.copy(), None, None
    centre = verts.mean(0)
    r, dirs = radial(verts, centre)
    return r, centre, dirs


def from_payload(e: Encoded, vals, dirs=None) -> np.ndarray:
    """Vertex positions from (decoded) payload values.

    Args:
        e: The encoding (payload, centre, A's directions).
        vals: (N, comps) values in payload units.
        dirs: (N, 3) unit directions for radial values; None uses ``e.dirs``.

    Returns:
        np.ndarray: (N, 3) positions. Xyz is offset by ``e.centre`` when set (morph
        centres xyz on A's centroid; the other modes do not).
    """
    if e.payload == "xyz":  # morph centres xyz on A's centroid; the other modes do not
        return vals if e.centre is None else e.centre + vals
    return e.centre + vals[:, :1] * (e.dirs if dirs is None else dirs)


def hue(verts) -> np.ndarray:
    """Vertex colour field for colour mode: hue from the angle around the vertical axis.

    Args:
        verts: (N, 3) vertex positions.

    Returns:
        np.ndarray: (N,) hues in [0, 1).
    """
    c = verts - verts.mean(0)
    return (np.arctan2(c[:, 1], c[:, 0]) / (2 * np.pi)) % 1.0


# --- where it goes on the chip ---------------------------------------------------
def cells(graph) -> np.ndarray:
    """Capacity slot range of each data qubit (Tessa's ``_slot_ranges``).

    Args:
        graph: Tessa ``Lattice`` or ``TessaGraph``.

    Returns:
        np.ndarray: (groups, 2) int ``(start, stop)`` per data qubit, in
        ``graph.groups`` order.
    """
    return np.array(_slot_ranges(graph), int).reshape(-1, 2)


def slot_groups(graph, slots) -> np.ndarray:
    """The data qubit holding each capacity slot.

    Args:
        graph: Tessa ``Lattice`` or ``TessaGraph``.
        slots: Int array of capacity slot indices, any shape.

    Returns:
        np.ndarray: Same shape as ``slots``: indices into ``graph.groups``.
    """
    return np.searchsorted(cells(graph)[:, 1], slots, side="right")


def place(graph, n, comps, order=None, quality=None, weight=None, per_slot=1, grow=None):
    """Capacity slot and field (0 = θ, 1 = φ) of every value.

    ``order=None`` is the naive axis-major layout (all x, then all y, then all z).
    Otherwise each cell takes the next mesh-neighbouring vertices from ``order``
    and stores them as [x1..xm, y1..ym, z1..zm], so a vertex's components share a
    cell and every cell holds a smooth patch of the surface.

    Cells are filled in graph order, or, given ``quality``, in Tessa's
    ``slot_order``: best data qubit first, so a mesh smaller than the chip lands
    only on its best qubits.

    ``weight`` also matches the most important vertices to the best of those
    qubits. Naive layout: vertices fill the ranked slots most important first. Mesh
    layout: the neighbourhood patches are cut as usual, then within each patch size
    the patches with the highest mean importance swap onto the best data qubits, so
    every patch stays whole.

    ``per_slot=2`` (split mode) gives every slot two positions, θ then φ: a data
    qubit with C slots takes 2C values, its θ field filled first and its φ field
    after, so each field holds runs of one coordinate and stays smooth.

    Args:
        graph: Tessa ``Lattice`` or ``TessaGraph``.
        n: Number of vertices.
        comps: Values per vertex (1 radial, 3 xyz).
        order: (n,) vertex order for the mesh layout (e.g. :func:`mesh.bfs_order`),
            or None for the naive layout.
        quality: One score per data qubit (higher is better), or None for graph order.
        weight: (n,) importance score per vertex, or None.
        per_slot: Positions per slot: 1, or 2 for split mode.
        grow: Optional ``grow(sizes) -> (patches, leftover vertices)`` that cuts the
            mesh-layout patches (e.g. compact :func:`mesh.grow_patches`); None takes
            consecutive runs of ``order``.

    Returns:
        tuple: ``(slots, field)``, each (n, comps) int. When whole vertices run out of
        room, the rest fill spare positions, best cell first, so a few straddle two
        data qubits.

    Raises:
        ValueError: If ``n * comps`` exceeds ``per_slot * graph.capacity``.
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
    cell_of = slot_groups(graph, ranked)
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
    """Capacity-length payload: values at their slots, the rest padded.

    Unused slots take their cell's mean, so padding keeps each cell smooth (and
    compressible); wholly empty cells take the overall mean (0 if no values).

    Args:
        graph: Tessa ``Lattice`` or ``TessaGraph``.
        slots: Int array of capacity slots, any shape.
        vals: Values, same shape as ``slots``.

    Returns:
        np.ndarray: (graph.capacity,) float array for ``tessa.encode``.
    """
    arr = np.full(graph.capacity, np.nan)
    arr[slots] = vals
    for lo, hi in cells(graph):
        cell = arr[lo:hi]
        if np.isnan(cell).any() and not np.isnan(cell).all():
            cell[np.isnan(cell)] = np.nanmean(cell)
    arr[np.isnan(arr)] = np.nanmean(arr) if len(vals) else 0.0
    return arr


def fit_lattice(n: int, comps: int = 1, mesh: bool = True, per_slot: int = 1) -> Lattice:
    """Smallest near-square Lattice (cols - rows <= 2, up to 10x12) holding the mesh.

    Prefers a lattice where every vertex fits whole on one data qubit; only if none
    does, one that merely has the capacity.

    Args:
        n: Number of vertices.
        comps: Values per vertex.
        mesh: Mesh layout (whole vertices per cell) rather than naive.
        per_slot: Positions per slot: 1, or 2 for split mode.

    Returns:
        Lattice: The smallest fitting Tessa lattice.

    Raises:
        ValueError: If even the 10x12 lattice is too small.
    """
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
    """Data qubit holding each vertex's first value.

    Args:
        e: The encoding.

    Returns:
        np.ndarray: (N,) qubit indices (physical on a chip).
    """
    return np.array([e.graph.groups[i].data_qubit for i in e.groups_of[:, 0]])


def qubit_map(e: Encoded) -> dict:
    """Where every value sits on the qubits, for drawing the lattice or chip.

    Args:
        e: The encoding.

    Returns:
        dict: ``coords``/``edges`` draw the qubits and couplers (the chip's own
        drawing for a device, rows and columns for a lattice). ``groups`` lists each
        data qubit with its address qubits. ``vertices[i][c] = [group, address,
        field]``: value ``c`` of vertex ``i`` is in that group's slot whose address
        qubit ``groups[g].addresses[j]`` reads bit ``j`` of ``address`` (the bit order
        Tessa's decoder uses), in ``field`` 0 (θ) or 1 (φ, split mode only). In colour
        mode the same slot also holds the vertex colour in φ.
    """
    g = e.graph
    if e.device is None:
        coords = [list(g.site_of_qubit(q))[::-1] for q in range(g.n_qubits)]
        couplers, _ = square_lattice_edges(g.rows, g.cols)
    else:
        coords = snapshot_json(e.device)["coords"]
        couplers = sorted(g.edges)  # TessaGraph keeps each edge as (low, high)
    bounds, cell = cells(g), e.groups_of
    return {"coords": coords, "edges": [list(map(int, c)) for c in couplers],
            "groups": [{"data": grp.data_qubit, "addresses": list(grp.address_qubits)}
                       for grp in g.groups],
            "vertices": np.stack([cell, e.slots - bounds[cell, 0], e.fields], -1).tolist()}


# --- encode / decode -------------------------------------------------------------
def morph_clamp(p: Params) -> float:
    """Morph's one clamp for both fields: θ's, or Tessa's pair default.

    φ follows θ so equal geometry gets equal angles in both fields.

    Args:
        p: Settings (``p.clamp``).

    Returns:
        float: Clamp in radians.
    """
    return p.clamp or DEFAULT_CLAMP


def split_batches(verts, faces, n_batches: int) -> list[np.ndarray]:
    """Connected surface patches of whole vertices, as even as possible.

    Args:
        verts: (N, 3) vertex positions.
        faces: Polygon vertex-index lists.
        n_batches: Number of patches.

    Returns:
        list[np.ndarray]: ``n_batches`` vertex-index arrays, each sorted.
    """
    sizes = [len(a) for a in np.array_split(np.arange(len(verts)), n_batches)]
    patches, rest = grow_patches(verts, faces, sizes)
    assert not len(rest)
    return [np.sort(q) for q in patches]


def batches_needed(n: int, comps: int, capacity: int) -> int:
    """Runs needed to fit a mesh, splitting whole vertices evenly.

    Args:
        n: Number of vertices.
        comps: Values per vertex.
        capacity: Positions per run.

    Returns:
        int: Smallest k with ceil(n / k) * comps <= capacity.

    Raises:
        ValueError: If one vertex alone exceeds ``capacity``.
    """
    if comps > capacity:
        raise ValueError(f"one vertex ({comps} values) exceeds capacity {capacity}")
    k = max(1, -(-n * comps // capacity))
    while -(-n // k) * comps > capacity:
        k += 1
    return k


def submesh(faces, idx):
    """Faces lying wholly inside a vertex subset, reindexed to that subset.

    Args:
        faces: Polygon vertex-index lists.
        idx: Sorted vertex indices of the subset.

    Returns:
        list[list[int]]: Faces in subset indices (position in ``idx``).
    """
    new = np.full(max((v for f in faces for v in f), default=-1) + 1, -1)
    new[idx] = np.arange(len(idx))
    return [[int(new[v]) for v in f] for f in faces if all(new[v] >= 0 for v in f)]


def rotated_quality(quality, skip: int) -> np.ndarray:
    """Scores ranked like ``quality``, rotated so the ``skip`` best data qubits come last.

    Batch k starts on the next-best qubits after those the earlier batches filled
    first (``skip`` wraps around the chip).

    Args:
        quality: One score per data qubit.
        skip: Number of best data qubits to move to the end of the ranking.

    Returns:
        np.ndarray: (groups,) scores whose descending order is the rotated ranking.
    """
    best = np.argsort(-np.asarray(quality, float), kind="stable")
    out = np.empty(len(best))
    out[np.roll(best, -skip)] = len(best) - np.arange(len(best))
    return out


def encode_mesh(verts, faces, p: Params, target=None) -> Encoded:
    """Encode a mesh with Tessa: pick the graph, place the values, build the circuit(s).

    Morph meshes too large for one run are split into connected batches, one circuit
    each (see BATCH_PLACEMENTS).

    Args:
        verts: (N, 3) vertex positions.
        faces: Polygon vertex-index lists.
        p: Encoder settings.
        target: Morph only: the dict :func:`morph.correspond` returns, or None to
            build it from ``p.target``.

    Returns:
        Encoded: The encoding and everything needed to decode it.

    Raises:
        ValueError: For an unknown mode, a morph without B, or a mesh that does not fit.
    """
    mode = p.canonical_mode
    if mode not in MODES:
        raise ValueError(f"unknown mode {p.mode!r}; use one of {MODES}")
    per = 2 if mode == "split" else 1
    verts = np.asarray(verts, dtype=float)
    vals, centre, dirs = to_payload(verts, p.payload)
    vals_b = dirs_b = prep_b = match = None
    clamp, clamp_b = p.clamp, p.clamp_b
    if mode == "morph":
        if target is None:
            if p.target is None:
                raise ValueError("Morph: upload a B mesh")
            target = correspond(verts, faces, p.target, p.shared_topology)
        match = target
        tgt = np.asarray(match["target"], float)
        centre = verts.mean(0)
        if p.payload == "xyz":
            vals, vals_b = verts - centre, tgt - centre
        else:
            vals_b, dirs_b = radial(tgt, centre)
        clamp = clamp_b = morph_clamp(p)
    n, comps = vals.shape
    flat, prep = prepare_field(vals.ravel(), quant_bits=p.quant_bits)
    top = float(np.abs(flat).max() or 1.0)
    if vals_b is not None:
        flat_b, prep_b = prepare_field(vals_b.ravel(), quant_bits=p.quant_bits)
        top = max(top, float(np.abs(flat_b).max()))
    top *= 1 + p.headroom
    rng = (-top, top) if p.payload == "xyz" else (0.0, top)
    if mode == "morph" and p.payload == "radial":
        # Centred on A's mean radius: the sweep pulls every value toward mid-range, so
        # with (0, r_max) the mesh shrinks as t grows. Here A's radii sit near θ = π/2,
        # where the sweep leaves them almost unchanged and reads B's nearly as they are.
        mid = float(flat.mean())
        half = max(float(np.abs(flat - mid).max()), float(np.abs(flat_b - mid).max())) or mid or 1.0
        half *= 1 + p.headroom
        rng = (mid - half, mid + half)
    mesh_layout = p.layout == "mesh"
    can_batch = mode == "morph"  # generic below; only morph switches it on for now
    quality = None
    if p.backend in DEVICES:
        graph, name = devices.graph(p.backend), f"ibm_{p.backend} (whole chip)"
        quality = chip_quality(p.backend)
    else:
        try:
            graph = fit_lattice(n, comps, mesh_layout, per)
        except ValueError:
            if not can_batch:
                raise
            graph = Lattice(MAX_ROWS, MAX_COLS)
        name = f"{graph.rows}x{graph.cols} lattice"
    k_runs = batches_needed(n, comps, per * graph.capacity) if can_batch else 1
    ranked = p.placement in ("quality", "importance")
    weight = importance(verts, faces) if p.placement == "importance" and quality is not None else None
    groups = split_batches(verts, faces, k_runs) if k_runs > 1 else [np.arange(n)]
    slots, field = np.empty((n, comps), int), np.empty((n, comps), int)
    batch = np.zeros(n, int)
    vals = flat.reshape(n, comps)
    vals_b = None if vals_b is None else flat_b.reshape(n, comps)
    colour = hue(verts) if mode == "colour" else None
    parts, skip = [], 0
    for k, idx in enumerate(groups):
        if k_runs == 1:
            v_k, f_k = verts, faces
        else:
            v_k, f_k = verts[idx], submesh(faces, idx)
        order = bfs_order(len(idx), f_k) if mesh_layout else None
        grow = (lambda sizes, v=v_k, f=f_k, o=order: grow_patches(v, f, sizes, o)) if mesh_layout else None
        q_k = None
        if ranked and quality is not None:
            q_k = rotated_quality(quality, skip) if p.batch_placement == "next-best" and skip else quality
        s_k, fld_k = place(graph, len(idx), comps, order, q_k, None if weight is None else weight[idx],
                           per, grow)
        slots[idx], field[idx], batch[idx] = s_k, fld_k, k
        if q_k is not None:  # the best data qubits whose capacity this batch takes first
            sizes = np.diff(cells(graph), axis=1)[np.argsort(-q_k, kind="stable"), 0] * per
            skip += int(np.searchsorted(np.cumsum(sizes), len(idx) * comps)) + 1
        parts.append(_circuit(graph, mode, p, s_k, fld_k, vals[idx],
                              None if vals_b is None else vals_b[idx], None if colour is None else colour[idx],
                              rng, clamp, clamp_b))
    circuit, enc, single_twoq = parts[0]
    return Encoded(verts, faces, graph, name, circuit, enc, prep, slots, p.payload, centre,
                   dirs, colour, single_twoq, quality,
                   field if mode == "split" else None, mode,
                   parts=tuple((c, e) for c, e, _ in parts) if k_runs > 1 else (),
                   batch=batch if k_runs > 1 else None,
                   target=None if match is None else tgt,
                   dirs_b=dirs_b, prep_b=prep_b, match=match)


def _circuit(graph, mode, p: Params, slots, field, vals, vals_b, colour, rng, clamp, clamp_b):
    """One run's circuit via ``tessa.encode``.

    Phi mode fixes θ = π/2 and tilts the circuit with Tessa's ``_tilted`` so Z reads
    X (<X> = cos φ), then decodes it as one field. Colour, split and morph use Tessa's
    pair mode, a second field in φ.

    Args:
        graph: Tessa ``Lattice`` or ``TessaGraph``.
        mode: One of MODES.
        p: Settings (compression, φ clamp in phi mode).
        slots: (n, comps) capacity slots of this run's values.
        field: (n, comps) 0 = θ, 1 = φ.
        vals: (n, comps) prepared values.
        vals_b: Morph: (n, comps) B's prepared values, else None.
        colour: Colour mode: (n,) hues in [0, 1], else None.
        rng: ``(lo, hi)`` value range shared by the geometry fields.
        clamp: θ clamp (ignored for morph's θ-only reference circuit).
        clamp_b: φ clamp.

    Returns:
        tuple: ``(circuit, enc, single_twoq)``: the circuit, its Tessa ``Encoding``, and
        the two-qubit gates of the θ field alone.
    """
    a = fill(graph, slots[field == 0], vals[field == 0])
    kw = dict(value_range=rng, compression=p.compression)
    single, enc = encode(graph, a, clamp=None if mode == "morph" else clamp, **kw)
    single_twoq = two_qubit(single)
    if mode == "theta":
        return single, enc, single_twoq
    if mode == "phi":  # θ = π/2 everywhere (0.5 is mid-range at any clamp), value in φ
        c_b = p.clamp_b or 0.0
        circuit, enc = encode(graph, np.full(graph.capacity, 0.5), a, clamp_b=c_b,
                              value_range=(0.0, 1.0), value_range_b=rng,
                              compression=p.compression)
        circuit = _tilted(circuit, enc, np.pi / 2)  # Z now reads X (Tessa's pipeline "x" basis)
        # <X> = cos φ, so the tilted circuit decodes as one field: φ = arccos(<Z>)
        return circuit, replace(enc, mode="single", clamps=(c_b,), ranges=(rng,)), single_twoq
    # Tessa pair mode: a second field in φ, read in the X basis
    if mode == "colour":
        b, range_b = fill(graph, slots, np.repeat(colour[:, None], vals.shape[1], 1)), (0.0, 1.0)
    elif mode == "morph":  # B's matching value in the same slot, on the same scale
        b, range_b = fill(graph, slots, vals_b), rng
    else:
        b, range_b = fill(graph, slots[field == 1], vals[field == 1]), rng
    circuit, enc = encode(graph, a, b, clamp=clamp, clamp_b=clamp_b, value_range_b=range_b, **kw)
    return circuit, enc, single_twoq


def snapshot_json(device: str) -> dict:
    """Tessa's stored calibration snapshot of a chip.

    Args:
        device: "phoenix" or "fez".

    Returns:
        dict: The snapshot (name, calibrated_at, edges, coords, qubits, cz).
    """
    return devices.snapshot_backend(device).snapshot


def probe_chip(device: str) -> np.ndarray:
    """Tessa's ``probe_qubits`` on the calibrated simulator (about 8 s per chip).

    Args:
        device: "phoenix" or "fez".

    Returns:
        np.ndarray: Per data qubit, the fraction of test symbols read back right.
    """
    graph = devices.graph(device)
    return probe_qubits(graph, sampler(graph, Params(backend=device), 0))


@cache
def chip_quality(device: str) -> np.ndarray:
    """Quality per data qubit: from QUALITY_FILE if current, otherwise probed now.

    Args:
        device: "phoenix" or "fez".

    Returns:
        np.ndarray: One score per data qubit (graph.groups order), higher is better.
    """
    try:
        stored = json.loads(QUALITY_FILE.read_text())[device]
        if stored["calibrated_at"] == snapshot_json(device)["calibrated_at"]:
            return np.array(stored["quality"])
    except (FileNotFoundError, KeyError):
        pass
    return probe_chip(device)


def write_quality():
    """Regenerate QUALITY_FILE for every DEVICES chip: ``python -m tessa_ripple.bridge``.

    Returns:
        None. Writes the file.
    """
    QUALITY_FILE.write_text(json.dumps({d: {"calibrated_at": snapshot_json(d)["calibrated_at"],
                                            "quality": probe_chip(d).round(4).tolist()}
                                        for d in DEVICES}, indent=1) + "\n")


@cache
def _chip(name):
    """A chip's snapshot backend and per-qubit noise model, built once.

    Args:
        name: "phoenix" or "fez".

    Returns:
        tuple: ``(backend, noise_model)``: Tessa's ``SnapshotBackend`` and Tessa's
        ``build_local_noise_model`` over the whole chip.
    """
    be, graph = devices.snapshot_backend(name), devices.graph(name)
    return be, build_local_noise_model(graph, be, phys_map=list(range(graph.n_qubits)))


def sampler(graph, p: Params, seed: int):
    """The Tessa sampler for ``p.backend``.

    Chips: Tessa's ``calibrated_sampler`` construction (``_aer_sampler`` on the
    snapshot backend) with the cached noise model. "flat noise": ``noisy_sampler``
    with IBM_LIKE_RATES scaled by ``p.noise_strength`` (errors times it, T1/T2
    divided by it). Otherwise the noiseless ``aer_sampler``.

    Args:
        graph: Tessa ``Lattice`` or ``TessaGraph`` (unused for chips).
        p: Settings (backend, noise_strength).
        seed: Aer seed.

    Returns:
        DeviceSampler: A Tessa sampler for ``tessa.decode`` or ``run``.
    """
    if p.backend in DEVICES:  # tessa.calibrated_sampler, reusing the noise model across seeds
        be, nm = _chip(p.backend)
        return _aer_sampler(be, noise_model=nm, seed=seed)
    if p.backend == "flat noise" and p.noise_strength > 0:
        s = p.noise_strength
        rates = {k: (v / s if k in ("t1", "t2") else v * s if k.endswith("error") else v)
                 for k, v in IBM_LIKE_RATES.items()}
        return noisy_sampler(graph, rates=rates, seed=seed)
    return aer_sampler(graph, seed=seed)


def batch_seed(seed: int, k: int) -> int:
    """Seed of batch ``k`` in a readout, so batches never share a noise draw.

    Args:
        seed: The readout's seed.
        k: Batch index.

    Returns:
        int: ``seed + BATCH_SEED_STEP * k``.
    """
    return seed + BATCH_SEED_STEP * k


def sweep_circuit(e: Encoded, alpha, part: int = 0):
    """Morph's basis sweep: ``part``'s pair circuit with RY(-alpha) on every data qubit.

    The address qubits are never turned, so Z reads cos(alpha) Z + sin(alpha) X on
    every cell. One angle for every qubit is Tessa's own
    ``tessa.hardware.pipeline._tilted``; Tessa has no per-qubit form, so differing
    angles are applied here. The result decodes as one field with A's clamp and
    range (the phi mode's trick at alpha = pi/2). Cached per (part, alpha) on ``e``.

    Args:
        e: A morph encoding.
        alpha: Scalar angle, or one per data qubit (graph.groups order).
        part: Run (batch) index.

    Returns:
        tuple: ``(circuit, enc)``: the tilted circuit and its single-field Encoding.

    Raises:
        ValueError: If ``alpha`` is neither a scalar nor one angle per data qubit.
    """
    groups = e.graph.groups
    alpha = np.asarray(alpha, float)
    if alpha.ndim == 0:
        alpha = np.full(len(groups), float(alpha))
    if alpha.shape != (len(groups),):
        raise ValueError(f"alpha needs one angle per data qubit ({len(groups)}), got shape {alpha.shape}")
    key = (part, alpha.tobytes())
    if key not in e._sweeps:
        circuit, enc = e.parts[part] if e.parts else (e.circuit, e.enc)
        if (alpha == alpha[0]).all():
            qc = _tilted(circuit, enc, float(alpha[0]))
        else:
            qc = circuit.copy()
            for g, a in zip(groups, alpha):
                qc.ry(-a, g.data_qubit)
        # decoded as one field with A's clamp and range (the phi mode's trick at alpha = pi/2)
        e._sweeps[key] = (qc, replace(enc, mode="single", clamps=enc.clamps[:1], ranges=enc.ranges[:1]))
    return e._sweeps[key]


def _read(e: Encoded, p: Params, seed: int, runs):
    """``tessa.decode`` every run with its own seed and gather each vertex's values.

    Args:
        e: The encoding (slots, batches).
        p: Readout settings (backend, shots).
        seed: The readout's seed; run k uses :func:`batch_seed`.
        runs: One ``(circuit, enc)`` per batch.

    Returns:
        tuple: ``(a, b)``: (N, comps) first-field values, and (N, comps) second-field
        values for a pair encoding, else None.
    """
    a = b = None
    for k, (circuit, enc) in enumerate(runs):
        out = decode(circuit, enc, sampler(e.graph, p, batch_seed(seed, k)), shots=p.shots)
        oa, ob = out if isinstance(out, tuple) else (out, None)
        m = e.batches == k
        if a is None:
            a = np.empty(e.slots.shape)
            b = None if ob is None else np.empty(e.slots.shape)
        a[m] = oa[e.slots[m]]
        if ob is not None:
            b[m] = ob[e.slots[m]]
    return a, b


def _per_value(e: Encoded, alpha) -> np.ndarray:
    """A sweep angle as one angle per encoded value.

    Args:
        e: The encoding.
        alpha: Scalar, or one angle per data qubit (graph.groups order).

    Returns:
        np.ndarray: (N*comps,) angles.
    """
    alpha = np.asarray(alpha, float)
    if not alpha.ndim:
        return np.full(e.slots.size, float(alpha))
    return alpha[e.groups_of].ravel()


def tilted_angles(e: Encoded, p: Params, seed: int, alpha) -> np.ndarray:
    """arccos E(alpha), E = <cos(alpha) Z + sin(alpha) X>, per value.

    One readout of the tilted circuit(s), decoded with Tessa's ``_angle_readout``:
    what Tessa's transition decodes from.

    Args:
        e: A morph encoding.
        p: Readout settings.
        seed: The readout's seed.
        alpha: Scalar, or one angle per data qubit.

    Returns:
        np.ndarray: (N*comps,) angles in radians.
    """
    runs = [sweep_circuit(e, alpha, k) for k in range(e.runs)]
    runs = [(qc, _angle_readout(enc)) for qc, enc in runs]  # raw angle, as Tessa's transition
    return _read(e, p, seed, runs)[0].ravel()


def _transition_plan(e: Encoded, alphas, inputs=None):
    """The part of a ``tessa.hardware.pipeline`` Plan its transition functions read.

    Args:
        e: The encoding (its clamps).
        alphas: List of sweep angles (scalars or per-value arrays), one per frame.
        inputs: ``(a01, b01)`` noiseless inputs in [0, 1], for ``ideal_frames`` only.

    Returns:
        SimpleNamespace: A stand-in Plan with ``extra`` = clamps, alphas[, inputs].
    """
    extra = {"clamps": e.enc.clamps, "alphas": alphas}
    if inputs is not None:
        extra["inputs"] = inputs
    return SimpleNamespace(extra=extra)


def transition_values(e: Encoded, angle_alpha, angle_z, alpha) -> np.ndarray:
    """Tessa's transition decode (``tessa.hardware.pipeline._transition_frames``).

    The Z readout's sin(theta) is divided out of the X part of E(alpha). Noiseless,
    the frame is exactly A at alpha = 0 and exactly B at pi/2.

    Args:
        e: A morph encoding.
        angle_alpha: (N*comps,) :func:`tilted_angles` at ``alpha``.
        angle_z: (N*comps,) :func:`tilted_angles` at alpha = 0.
        alpha: Scalar, or one angle per data qubit.

    Returns:
        np.ndarray: (N*comps,) values in A's units.
    """
    frame = _transition_frames(_transition_plan(e, [0.0, _per_value(e, alpha)]), [angle_z, angle_alpha])[1]
    lo, hi = e.enc.ranges[0]
    return lo + (hi - lo) * frame


def decode_raw(e: Encoded, p: Params, seed: int | None = None, reference=None):
    """One readout of the encoding, before DRR.

    Morph is a basis sweep: it reads one field at alpha = p.morph_t * pi/2; with
    ``p.sweep_decode == "transition"`` it is turned into values with an alpha = 0
    readout (:func:`tilted_angles`).

    Args:
        e: The encoding.
        p: Readout settings.
        seed: Sampler seed; None uses ``p.seed``.
        reference: Transition sweep only: (N*comps,) alpha = 0 angles to reuse
            (run_frames shares one per run); None reads it now with the same seed.

    Returns:
        tuple: ``(values, second)``: values (N*comps,), and the colour (N,) in colour
        mode, else None.
    """
    seed = p.seed if seed is None else seed
    runs = e.parts or ((e.circuit, e.enc),)
    if e.mode == "morph":
        alpha = p.morph_t * np.pi / 2
        if p.sweep_decode == "transition":
            angle = tilted_angles(e, p, seed, alpha)
            if reference is None:
                reference = angle if alpha == 0 else tilted_angles(e, p, seed, 0.0)
            return transition_values(e, angle, reference, alpha), None
        runs = [sweep_circuit(e, alpha, k) for k in range(len(runs))]
    a, b = _read(e, p, seed, runs)
    if e.mode == "split":  # each value from its own field
        return np.where(e.fields == 1, b, a).ravel(), None
    if e.mode == "morph":  # the sweep circuits read one field
        return a.ravel(), None
    return a.ravel(), (None if b is None else b.mean(1))


def _values(e: Encoded, raw, prep, drr: str) -> np.ndarray:
    """Tessa's ``restore_field`` (DRR, un-quantise) on raw decoded values.

    Args:
        e: The encoding (for the shape).
        raw: (N*comps,) decoded values.
        prep: The field's Tessa ``FieldPrep``.
        drr: One of DRR_METHODS.

    Returns:
        np.ndarray: (N, comps) values in payload units.
    """
    return restore_field(raw, prep, dynamic_range=drr).reshape(e.slots.shape)


def restore(e: Encoded, raw, drr: str = "none", dirs=None) -> np.ndarray:
    """Classical post-processing (DRR) and back to vertex positions.

    Args:
        e: The encoding.
        raw: (N*comps,) decoded values.
        drr: One of DRR_METHODS.
        dirs: (N, 3) radial directions; None uses A's.

    Returns:
        np.ndarray: (N, 3) positions.
    """
    return from_payload(e, _values(e, raw, e.prep, drr), dirs)


def morph_dirs(e: Encoded, t: float, a_dirs_only: bool = False):
    """Radial morph: each vertex's direction at t.

    The turning from A's to B's directions (slerp) is classical; only the radius is
    read from the state.

    Args:
        e: A morph encoding.
        t: Morph parameter in [0, 1].
        a_dirs_only: Keep A's directions throughout.

    Returns:
        np.ndarray | None: (N, 3) unit directions, or None for xyz.
    """
    if e.payload == "xyz":
        return None
    return e.dirs if a_dirs_only else slerp(e.dirs, e.dirs_b, t)


def morph_frame(e: Encoded, raw, t: float, drr: str = "none", a_dirs_only: bool = False):
    """One decoded morph frame at t.

    The basis-sweep readout already holds the morph; it only needs its direction at
    t (radial payload).

    Args:
        e: A morph encoding.
        raw: (N*comps,) the sweep's decoded values at t.
        t: Morph parameter in [0, 1].
        drr: One of DRR_METHODS.
        a_dirs_only: Radial: keep A's directions.

    Returns:
        np.ndarray: (N, 3) positions.
    """
    return restore(e, raw, drr, morph_dirs(e, t, a_dirs_only))


def sweep_values(e: Encoded, alpha, decode: str = "transition") -> np.ndarray:
    """Noiseless sweep readout in value units.

    Raw: angle = arccos(cos a cos θ + sin a sin θ cos φ), mapped back through A's
    angle -> value map (Tessa's pair ``Mode``). Transition: Tessa's
    ``tessa.hardware.pipeline.ideal_frames``. Inputs are normalised with Tessa's
    ``codec._normalise``, as ``encode`` does.

    Args:
        e: A morph encoding.
        alpha: Scalar, or one angle per data qubit.
        decode: One of SWEEP_DECODES.

    Returns:
        np.ndarray: (N, comps) values.
    """
    m = get_mode("pair")
    lo, hi = e.enc.ranges[0]
    a01, b01 = (_normalise(prep.reference, (lo, hi))[0] for prep in (e.prep, e.prep_b))  # as encode
    th, ph = m.angles(a01, e.enc.clamps), m.phases(b01, e.enc.clamps)
    alpha = _per_value(e, alpha)
    if decode == "transition":  # Tessa's noiseless transition frame
        frame = ideal_frames(_transition_plan(e, [alpha], (a01, b01)))[0]
        return (lo + (hi - lo) * frame).reshape(e.slots.shape)
    ang = np.arccos(np.clip(np.cos(alpha) * np.cos(th) + np.sin(alpha) * np.sin(th) * np.cos(ph), -1, 1))
    return (lo + (hi - lo) * m.values_from_angles(ang, e.enc.clamps)).reshape(e.slots.shape)


def morph_expected(e: Encoded, t: float, a_dirs_only: bool = False, sweep_decode: str = "transition"):
    """Noiseless reference positions of a morph frame at t, for error colouring.

    Args:
        e: A morph encoding.
        t: Morph parameter in [0, 1].
        a_dirs_only: Radial: keep A's directions.
        sweep_decode: One of SWEEP_DECODES.

    Returns:
        np.ndarray: (N, 3) positions.
    """
    return from_payload(e, sweep_values(e, t * np.pi / 2, sweep_decode), morph_dirs(e, t, a_dirs_only))


def read_mesh(e: Encoded, p: Params, seed: int | None = None, drr: str = "none") -> np.ndarray:
    """Decode and restore one readout to vertex positions.

    Args:
        e: The encoding.
        p: Readout settings (morph: ``p.morph_t`` picks the frame).
        seed: Sampler seed; None uses ``p.seed``.
        drr: One of DRR_METHODS.

    Returns:
        np.ndarray: (N, 3) decoded positions.
    """
    raw, _ = decode_raw(e, p, seed)
    if e.mode == "morph":
        return morph_frame(e, raw, p.morph_t, drr)
    return restore(e, raw, drr)


def mesh_error(verts, positions) -> float:
    """Mean absolute coordinate error as a fraction of the mesh's bounding box.

    Args:
        verts: (N, 3) reference positions.
        positions: (N, 3) decoded positions.

    Returns:
        float: mean |positions - verts| / ptp(verts).
    """
    return float(np.abs(positions - verts).mean() / np.ptp(verts))


def fidelity_curves(verts, faces, p: Params, progress=None) -> dict:
    """Error vs shots, and error + depth vs compression for both layouts.

    Args:
        verts: (N, 3) vertex positions.
        faces: Polygon vertex-index lists.
        p: Base settings.
        progress: Optional ``progress(step, total)`` callback.

    Returns:
        dict: ``shots``: [(shots, error)]; ``mesh``/``naive``: [(compression %,
        depth, error)]; ``same``: True if both layouts place every value identically.
    """
    shots, comps, out, k = (256, 1024, 4096, 16384, 65536), (0, 25, 50, 75, 90), {}, 0
    total = len(shots) + 2 * len(comps)
    e = encode_mesh(verts, faces, p)
    out["shots"] = []
    for s in shots:
        progress and progress(k := k + 1, total)
        out["shots"].append((s, mesh_error(verts, read_mesh(e, replace(p, shots=s)))))
    placed = {}  # each layout's slots (placement does not depend on compression)
    for layout in ("mesh", "naive"):
        out[layout] = []
        for c in comps:
            progress and progress(k := k + 1, total)
            q = replace(p, layout=layout, compression=c)
            ee = encode_mesh(verts, faces, q)
            placed.setdefault(layout, ee.slots)
            out[layout].append((c, ee.circuit.depth(), mesh_error(verts, read_mesh(ee, q))))
    out["same"] = bool(np.array_equal(placed["mesh"], placed["naive"]))  # e.g. radial on a ring-ordered mesh
    return out


# --- "Why Tessa": the same values under other encodings, on a real chip graph -----
def phoenix_layout(lat: Lattice) -> list[int]:
    """Tessa's ``rectangular_layout`` of a lattice on ibm_phoenix.

    Every lattice coupler is a chip coupler. A lattice wider than the chip (up to
    12 columns) goes on turned on its side: site (r, c) on qubit c * 10 + r.

    Args:
        lat: Tessa ``Lattice`` of at most 10x12 or 12x10.

    Returns:
        list[int]: ``initial_layout``: physical qubit per lattice qubit.
    """
    if lat.cols <= PHOENIX_COLS:
        return rectangular_layout(lat, PHOENIX_ROWS, PHOENIX_COLS)
    turned = Lattice(lat.cols, lat.rows)
    layout = rectangular_layout(turned, PHOENIX_ROWS, PHOENIX_COLS)
    return [layout[turned.qubit_index(c, r)] for r, c in map(lat.site_of_qubit, range(lat.n_qubits))]


def resources(e: Encoded) -> list[dict]:
    """Cost of ``e``'s values under Tessa and four baselines, transpiled onto a chip.

    Every row encodes generic random data (so no encoding gets credit for this
    mesh) and is transpiled onto the ibm_phoenix snapshot; Tessa on a whole-chip
    graph is transpiled onto its own chip. SWAPs are estimated as the extra two-qubit
    gates over an unrouted transpile, divided by 3.

    Args:
        e: The encoding.

    Returns:
        list[dict]: One row per encoding (Tessa, angle, QCrank, FRQI, amplitude) with
        ``name``, ``qubits``, ``settings`` (measurement settings), ``configs``
        (address outcomes per cell), ``twoq``, ``depth``, ``swaps`` and ``fits``.
    """
    phoenix = devices.snapshot_backend("phoenix")
    n = e.stats["values"]
    rng = np.random.default_rng(0)

    def routed(qc, backend=phoenix, layout=None):
        """Two-qubit gates, depth and estimated SWAPs of ``qc`` transpiled onto ``backend``."""
        tq = transpile(qc, backend, initial_layout=layout, optimization_level=1, seed_transpiler=0)
        free = transpile(qc, basis_gates=["cz", "sx", "rz", "x"], optimization_level=1, seed_transpiler=0)
        cz = two_qubit(tq)
        return {"twoq": cz, "depth": tq.depth(), "swaps": max(0, round((cz - two_qubit(free)) / 3))}

    # generic random data for every row, so no encoding gets credit for this mesh; Tessa's
    # goes in the mesh's own slots, the rest of the graph padded as in encode_mesh
    if e.mode in ("colour", "morph"):
        where = [e.slots, e.slots]
    else:
        where = [e.slots[e.fields == f] for f in ((0, 1) if e.pair else (0,))]
    rand = [fill(e.graph, w, rng.random(w.shape)) for w in where]
    generic, _ = encode(e.graph, *rand)
    if e.device is None:
        tessa = routed(generic, layout=phoenix_layout(e.graph))
    else:
        tessa = routed(generic, devices.snapshot_backend(e.device), list(range(e.graph.n_qubits)))
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
    """Edge lengths read from the encoded state's coherences.

    Each mesh edge (i, j) and axis is encoded as a degree-one Tessa cell holding
    (v_i, v_j); a Hadamard on its address qubit makes K = 2 P(0) - 1 =
    cos(pi dv / 2), so |v_i - v_j| is read from the state's coherence, never decoded
    per vertex. This is the paper's coherence experiment B (isolated pairs on
    separate couplers, plus reference circuits), not the full-lattice version. On a
    chip the pairs spread over its lowest-error couplers.

    Args:
        verts: (N, 3) vertex positions.
        faces: Polygon vertex-index lists.
        p: Settings (backend, shots, noise, headroom).
        seed: Sampler seed; None uses ``p.seed``.

    Returns:
        tuple: ``(E, measured, exact)``: edges (E, 2), measured lengths (E,) and
        exact lengths (E,).
    """
    seed = p.seed if seed is None else seed
    E = edges(faces)
    a, b = verts[E[:, 0]].ravel(), verts[E[:, 1]].ravel()  # (edge, axis) pairs
    # Every pair shares one zoomed scale: the range spans the largest component
    # difference, so K uses its whole [-1, 1] range instead of sitting near 1.
    half = float(np.abs(a - b).max() or 1.0) * (1 + p.headroom) / 2
    addr = PAIR.groups[0].address_qubits[0]

    def pair_circuit(vi, vj):
        """The (vi, vj) cell with a Hadamard and measurement on its address qubit."""
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
    return E, measured, edge_lengths(verts, E)


if __name__ == "__main__":
    write_quality()
