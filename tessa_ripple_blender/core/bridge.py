"""Every Tessa call lives here: place a mesh on a lattice or chip, encode it once,
decode it per seed, compare resources, and read edge lengths from coherences.

Pure functions, numpy in and numpy out, so a Blender add-on can reuse them.

Wherever Tessa already implements a step, this module calls it rather than
re-deriving it: ``encode``, ``decode``'s own steps (``cell_halo``, ``cell_circuit``,
``codec._measured``, ``codec._trace_halo``, ``decode_from_cell_counts``; built once per
circuit, not per readout) and ``prepare_field``/``restore_field`` for the codec; ``slot_order``, ``probe_qubits`` and ``_slot_ranges`` for placement; the
samplers (``aer_sampler``, ``backends.aer._aer_sampler`` with ``topology_device``) with
``build_local_noise_model``, ``build_average_noise_model`` and ``backend_rates``;
``hardware.devices`` for chip snapshots; ``hardware.pipeline._tilted``, ``_angle_readout``, ``_transition_frames``
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
from ..tessa import (DYNAMIC_RANGE_METHODS, Lattice, backend_rates, build_average_noise_model,
                   build_local_noise_model, cell_circuit, cell_halo, decode_from_cell_counts,
                   decode_from_counts, encode, encode_symbols, graph_from_coupling, measurement_circuits,
                   prepare_field, probe_qubits, rectangular_layout, restore_field, slot_order,
                   square_lattice_edges, symbols_from_counts, transpile_to_backend)
from ..tessa.backends import device_of
from ..tessa.backends.aer import _TWO_QUBIT_GATES, _aer_sampler, topology_device
from ..tessa.codec import _check_traceable, _measured, _normalise, _trace_halo
from ..tessa.core.mode import DEFAULT_CLAMP, get_mode
from ..tessa.hardware import devices
from ..tessa.hardware.pipeline import _angle_readout, _tilted, _transition_frames, ideal_frames
from ..tessa.symbols import _slot_ranges

from . import spectral
from .mesh import bfs_order, edge_lengths, edges, grow_patches
from .morph import correspond, front_alpha, geodesic, slerp

MAX_ROWS, MAX_COLS = 10, 12  # largest lattice: the whole ibm_phoenix chip, turned on its side
PAYLOADS = ("radial", "xyz", "spherical", "spectral")
PHOENIX_ROWS, PHOENIX_COLS = 12, 10  # ibm_phoenix: 12 rows of 10 qubits, row-major
DEVICES = ("phoenix", "fez")  # Tessa's offline calibration snapshots
BACKENDS = ("noiseless", "flat noise", *DEVICES)
# chip backends: Tessa's group order | best data qubits first (Tessa's slot_order)
PLACEMENTS = ("graph", "quality")
QUALITY_FILE = Path(__file__).with_name("quality.json")  # probe_qubits scores per chip snapshot
# What each slot's qubit holds. One value per slot in theta, read in Z (Tessa's "single"
# mode; READOUTS for other ways to read it). Split: two values per slot, the second in
# phi on the full circle, read in Z, X and Y (Tessa's "pair_full" mode): more geometry
# (twice the vertices fit). Morph: mesh A in theta, a second mesh B (one matched point per A vertex) in phi,
# in the same slots, read by a basis sweep: at t the data qubits are measured along
# cos(alpha) Z + sin(alpha) X, alpha = t * pi/2, and Tessa's transition decode divides
# out sin(theta) with an alpha = 0 (Z) readout (exactly A at t = 0, exactly B at t = 1).
MODES = ("theta", "split", "morph")
# How a one-field (Mode θ) encoding is read. "z": θ in Z, one setting. "bloch": the same
# circuit (φ = 0) read in Z, X and Y, θ = atan2(|<X>, <Y>|, <Z>): depolarising noise
# shrinks the Bloch vector, not its direction, so the shrink cancels without any
# statistics of the input. "snap": each value rounded to one of SNAP_LEVELS points spread
# over the Bloch sphere (tessa.symbols' constellation), read in Z, X and Y and snapped to
# the nearest: exact while the noise keeps each point nearest its own.
READOUTS = ("z", "bloch", "snap")
SNAP_LEVELS = 64
# Pilots: known values in every spare slot of each run, read with the mesh; the line
# taking their readings back to their values corrects the whole run (a DRR measured on
# the chip, not taken from the input). On the mesh's own data qubits they add no gates; a
# data qubit holding only pilots gains its multiplexer (blob on Phoenix: 344 → 404 CZ),
# which pays: pilots on the mesh's qubits alone fit worse than no correction there. Each
# run keeps room for at least PILOT_ROOM of them.
PILOT_ROOM = 2
MIN_SHOTS, MAX_SHOTS = 256, 65536
# meshes too big for one run are read in batches; on a chip each later batch starts
# on the next-best data qubits, so the best ones are spread across the runs
BATCH_SEED_STEP = 1000  # batch k of a readout with seed s uses seed s + 1000 k
# morph's alpha = 0 (Z) reference is a separate run on the chip, so it gets its own noise
# draw: seed s + 2**20, clear of every frame seed (s + i) and batch seed (s + 1000 k)
REFERENCE_SEED_OFFSET = 1 << 20
DRR_METHODS = DYNAMIC_RANGE_METHODS  # Tessa's dynamic range recovery methods
DRR_CHOICES = (*DRR_METHODS, "pilots")  # the apps' DRR menu: "pilots" sets Params.pilots
SPECTRAL_FRONT = ("The morph front needs one row per vertex; the spectral payload's qubits hold "
                  "modes of the whole surface, not patches of it. Use the uniform sweep or the "
                  "interference readout.")


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
        payload: "radial" (one radius per vertex from the centroid; directions kept
            classically), "xyz" (three coordinates per vertex), "spherical" (direction
            as θ, φ of one slot read in Z, X and Y, radius in a second slot; Mode θ only)
            or "spectral" (xyz coefficients of A's first k Laplacian modes, one row per
            mode; always one run; no morph front).
        layout: "mesh" (BFS neighbourhood patches per cell) or "naive" (axis-major).
        mode: One of MODES: θ (Z), shape over θ and φ, or morph A→B.
        clamp: θ clamp in radians; None for Tessa's default.
        clamp_b: φ clamp in radians (pair modes); None for Tessa's default.
        compression: Percent (0-100) of multiplexer coefficients dropped.
        quant_bits: Quantisation bits for ``prepare_field``; None for none.
        headroom: Fractional padding on the value range.
        placement: Chips only: one of PLACEMENTS.
        target: Morph: mesh B's OBJ path.
        shared_topology: Morph: pair B with A by vertex index (else closest point).
        spectral_k: Spectral: modes to encode; None picks the smallest k whose
            truncation error meets ``spectral_tol`` for A (and B), capped at one run.
        spectral_tol: Spectral: largest truncation error (RMS over the bounding-box
            diagonal) for the automatic k.
        spectral_band: Spectral: scale each mode to fill [-1, 1] (on by default; off
            shares one range across modes, for the experiments only).
        readout: One of READOUTS: how a Mode θ encoding is read (not spherical).
        pilots: Keep known pilot values in spare slots and correct each run by them
            (Mode θ, not spherical or snap): the DRR menu's "pilots".

    Readout settings (changing these re-decodes; a chip backend also changes the graph):

    Attributes:
        backend: One of BACKENDS.
        shots: Shots per measurement setting.
        noise_strength: Scales the "flat noise" rates only.
        seed: Sampler seed of the first readout.
        morph_t: Morph: the basis sweep reads at alpha = t * pi/2 (0 = A, 1 = B).
        morph_front: Morph: give each (batch, data qubit) its own alpha, a front
            spreading from ``front_source`` over the surface (:func:`morph_alpha`).
        front_width: Morph front: softness w of the ramp, in normalised distance.
        front_source: Morph front: source vertex index; None = auto
            (:func:`front_source`, the largest |B - A|).
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
    spectral_k: int | None = None
    spectral_tol: float = 0.01
    spectral_band: bool = True
    readout: str = "z"
    pilots: bool = False
    # readout (changing these re-decodes; a chip backend also changes the graph)
    backend: str = "noiseless"
    shots: int = 4096
    noise_strength: float = 1.0
    seed: int = 0
    morph_t: float = 0.0
    morph_front: bool = False
    front_width: float = 0.15
    front_source: int | None = None

    @property
    def encoder_key(self):
        """Cache key of everything that changes the encoded circuit.

        Morph-only, chip-only and spectral-only settings join the key only when they
        apply, so toggling an unused setting never re-encodes.

        Returns:
            tuple: Hashable key; equal keys give the same ``encode_mesh`` result.
        """
        graph = self.backend if self.backend in DEVICES else "lattice"
        morph = self.mode == "morph"
        spec = self.payload == "spectral"
        return (self.payload, self.layout, self.mode, self.clamp, self.clamp_b,
                self.compression, self.quant_bits, self.headroom, graph,
                self.placement if graph != "lattice" else "graph",
                *((self.target, self.shared_topology) if morph else ()),
                *((self.spectral_k, self.spectral_tol, self.spectral_band) if spec else ()),
                self.readout, self.pilots)

    @property
    def readout_key(self):
        """Cache key of everything that changes a readout of one encoding.

        Morph keys on ``morph_t``: the sweep reads in a basis set by t. The front's
        width and source join only while the front is on.

        Returns:
            tuple: Hashable key; equal keys give the same decoded values.
        """
        morph = self.mode == "morph"
        front = (self.morph_front, *((self.front_width, self.front_source) if self.morph_front else ()))
        return (self.shots, self.backend, self.noise_strength, self.seed,
                *((self.morph_t, *front) if morph else ()))

    @property
    def front(self):
        """The morph front's settings, or None for the uniform sweep.

        Returns:
            tuple | None: ``(width, source)`` when ``mode`` is morph and
            ``morph_front`` is on, else None.
        """
        return (self.front_width, self.front_source) if self.mode == "morph" and self.morph_front else None


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
        slots: (N, comps) capacity slot of every encoded value. Spectral: (rows, 3),
            one row per mode, not per vertex (so are ``groups_of``, ``fields`` and
            every other per-row array).
        payload: One of PAYLOADS.
        centre: (3,) centre radial, spherical and spectral values (and morph xyz) are
            measured from, or None.
        dirs: (N, 3) unit directions from ``centre`` (radial), or None.
        single_twoq: Two-qubit gates of the θ-only circuit.
        quality: Chip only: ``probe_qubits`` score per data qubit, else None.
        field: (N, comps) 0 = value in θ, 1 = in φ (split mode, spherical payload),
            else None.
        mode: One of MODES.
        parts: Several runs: one (circuit, enc) per batch, parts[0] being
            (circuit, enc); empty for one run.
        batch: Several runs: (N,) each vertex's batch, else None.
        target: Morph: (N, 3) B's matched point per A vertex (aligned), else None.
        dirs_b: Radial morph: (N, 3) unit directions of ``target`` from ``centre``.
        prep_b: Morph: Tessa ``FieldPrep`` of B's field.
        match: Morph: the :func:`morph.correspond` dict (method, coverage, B mesh).
        scales: Spherical: (3,) units of one encoded fraction (π polar, 2π azimuth,
            the radius range). Spectral: (rows,) each mode's band scale. Else None.
        basis: Spectral: (N, rows) A's Laplacian modes (the constant first in a
            morph), else None.
        spectral: Spectral: ``k`` (non-constant modes), ``rows`` (encoded modes),
            ``error_a``/``error_b`` (truncation error over the bounding-box diagonal;
            ``error_b`` None outside morph), else None.
        readout: One of READOUTS.
        pilots: With ``Params.pilots``: per run, ``(slots, values)`` of its pilots; else ().
        _sweeps: Cache of :func:`sweep_circuit` results per (part, alpha).
        _fronts: Cache of :func:`front_distances` per source vertex.
        _jobs: Cache of :func:`_cell_jobs` per (circuit, device).
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
    scales: np.ndarray | None = None
    basis: np.ndarray | None = None
    spectral: dict | None = None
    readout: str = "z"
    pilots: tuple = ()
    _sweeps: dict = dc_field(default_factory=dict, compare=False, repr=False)
    _fronts: dict = dc_field(default_factory=dict, compare=False, repr=False)
    _jobs: dict = dc_field(default_factory=dict, compare=False, repr=False)

    @property
    def fields(self) -> np.ndarray:
        """Field of every value, all 0 (θ) outside split mode and the spherical payload.

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
    def run_parts(self) -> tuple:
        """Every run's ``(circuit, enc)``, in batch order (one for an unbatched mesh).

        Returns:
            tuple: ``parts``, or ``((circuit, enc),)``.
        """
        return self.parts or ((self.circuit, self.enc),)

    @property
    def batches(self) -> np.ndarray:
        """Each vertex's batch (all 0 for one run).

        Returns:
            np.ndarray: (N,) int batch index.
        """
        return np.zeros(len(self.slots), int) if self.batch is None else self.batch

    @property
    def pair(self) -> bool:
        """Whether the encoding uses a Tessa pair mode (a second field in φ).

        Returns:
            bool: True for split, morph and the spherical payload. (A Bloch or snap
            readout reads one field in Tessa's ``pair_full`` mode: not a pair.)
        """
        return self.mode in ("split", "morph") or self.payload == "spherical"

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

        Morph counts one B value in φ beside every A value in θ.

        Returns:
            dict: ``graph`` (name), ``quality`` (mean score of used data qubits or
            None), ``qubits`` (used), ``graph_qubits``, ``values``, ``capacity``,
            ``ratio`` (values per used qubit), ``depth``, ``twoq``, ``twoq_single``
            (θ field alone), ``runs`` and ``values_per_run`` (list).
        """
        n = self.slots.size * (2 if self.mode == "morph" else 1)  # morph: B in φ beside A in θ
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
    their direction from the centre; the directions stay classical. Spherical: the
    same radius plus the direction as polar angle (from +z) and azimuth, all three
    encoded, so only the centroid stays classical.

    Args:
        verts: (N, 3) vertex positions.
        payload: "radial", "xyz" or "spherical" (spectral needs faces and a capacity:
            :func:`spectral_payload`).

    Returns:
        tuple: ``(values, centre, dirs)``: values (N, comps) (comps 1 radial, 3 xyz,
        3 spherical: polar, azimuth in [0, 2π), radius), the centroid (3,) and unit
        directions (N, 3); centre is None for xyz, dirs None for xyz and spherical.

    Raises:
        ValueError: For the spectral payload.
    """
    if payload == "spectral":
        raise ValueError("The spectral payload needs the mesh's faces: use spectral_payload")
    if payload == "xyz":
        return verts.copy(), None, None
    centre = verts.mean(0)
    r, dirs = radial(verts, centre)
    if payload == "spherical":  # the angles tessa.symbols uses for a point on the Bloch sphere
        polar = np.arccos(np.clip(dirs[:, 2], -1.0, 1.0))
        azimuth = np.mod(np.arctan2(dirs[:, 1], dirs[:, 0]), 2 * np.pi)
        return np.column_stack([polar, azimuth, r[:, 0]]), centre, None
    return r, centre, dirs


def from_payload(e: Encoded, vals, dirs=None) -> np.ndarray:
    """Vertex positions from (decoded) payload values.

    Args:
        e: The encoding (payload, centre, A's directions, spectral basis).
        vals: (N, comps) values in payload units; spectral: (rows, 3) band-scaled
            coefficients.
        dirs: (N, 3) unit directions for radial values; None uses ``e.dirs``.
            Ignored for spherical and spectral, whose directions are decoded.

    Returns:
        np.ndarray: (N, 3) positions. Xyz is offset by ``e.centre`` when set (morph
        centres xyz on A's centroid; the other modes do not).
    """
    if e.payload == "spectral":  # V = centre + Φ (s ⊙ C')
        return spectral.reconstruct(np.asarray(vals, float) * e.scales[:, None], e.centre, e.basis)
    if e.payload == "spherical":  # fractions of (π, 2π, radius range)
        polar, azimuth, r = (vals * e.scales).T
        s = np.sin(polar)
        return e.centre + r[:, None] * np.column_stack([s * np.cos(azimuth), s * np.sin(azimuth), np.cos(polar)])
    if e.payload == "xyz":  # morph centres xyz on A's centroid; the other modes do not
        return vals if e.centre is None else e.centre + vals
    return e.centre + vals[:, :1] * (e.dirs if dirs is None else dirs)


def run_positions(p: Params, per_slot: int = 1) -> int:
    """Values one run can hold: the chip's whole capacity, or the largest lattice's.

    Args:
        p: Settings (backend).
        per_slot: Positions per slot: 1, or 2 for split mode.

    Returns:
        int: Capacity of one run in values (less the pilots' room).
    """
    graph = devices.graph(p.backend) if p.backend in DEVICES else Lattice(MAX_ROWS, MAX_COLS)
    return per_slot * graph.capacity - (PILOT_ROOM if p.pilots else 0)


def spectral_cap(n_verts: int, p: Params, positions: int | None = None) -> int:
    """The most non-constant modes a spectral encoding can hold: one run, and N - 1.

    Args:
        n_verts: Vertices of A.
        p: Settings (backend, mode; a morph spends one row on the constant mode).
        positions: Values one run holds; None uses :func:`run_positions` (the
            interference morph passes its half capacity, ``interfere.spectral_positions``).

    Returns:
        int: The cap on ``Params.spectral_k``.
    """
    if positions is None:
        positions = run_positions(p, 2 if p.mode == "split" else 1)
    return int(min(positions // 3 - (p.mode == "morph"), n_verts - 1))


def cap_spectral(p: Params, n_verts: int, interference: bool = False) -> Params:
    """``p`` with ``spectral_k`` capped to what one run holds (a slider may ask for more).

    One place for the desktop app, the web API and the Blender add-on: a k past the
    cap encodes the cap.

    Args:
        p: Settings.
        n_verts: Vertices of A.
        interference: The interference morph readout, whose run holds half as many
            values (``interfere.spectral_positions``).

    Returns:
        Params: ``p``, with ``spectral_k`` at most :func:`spectral_cap` (unchanged
        outside the spectral payload, or with k automatic).
    """
    if p.payload != "spectral" or p.spectral_k is None:
        return p
    positions = None
    if interference:
        from .interfere import spectral_positions  # interfere builds on this module
        positions = spectral_positions(p)
    return replace(p, spectral_k=min(p.spectral_k, spectral_cap(n_verts, p, positions)))


def spectral_payload(verts, faces, p: Params, positions: int, target=None) -> dict:
    """The spectral payload's values: A's (and B's) band-scaled coefficients.

    k is ``p.spectral_k`` or :func:`spectral.choose_k` at ``p.spectral_tol``, capped
    so every mode fits one run. A morph adds the constant mode as row 0 (B's centroid
    shift; A's coefficient there is 0).

    Args:
        verts: (N, 3) A's vertices.
        faces: A's polygons.
        p: Settings (spectral_k, spectral_tol, spectral_band, headroom).
        positions: Values one run holds (:func:`run_positions`, or the interference
            morph's half capacity).
        target: (N, 3) B on A's vertices for a morph, or None.

    Returns:
        dict: ``vals`` (rows, 3) A's coefficients over ``scales`` (in [-1, 1]),
        ``vals_b`` likewise for B or None, ``centre`` (3,), ``basis`` (N, rows),
        ``scales`` (rows,) and ``info`` (``Encoded.spectral``).

    Raises:
        ValueError: If ``p.spectral_k`` does not fit one run or the mesh.
    """
    morph = target is not None
    k_cap = min(positions // 3 - morph, len(verts) - 1)  # spectral_cap, for whatever target says
    if p.spectral_k is None:
        k = spectral.choose_k(verts, target, faces, p.spectral_tol, k_cap)
    else:
        k = int(p.spectral_k)
        if k > k_cap:
            raise ValueError(f"Spectral k = {k} does not fit: one run holds {positions} values, "
                             f"so k ≤ {k_cap} here")
    phi, _, mass = spectral.basis(verts, faces, k, constant=morph)
    centre = spectral.centroid(verts, mass)
    ca = spectral.coefficients(verts, centre, phi, mass)
    cb = None if target is None else spectral.coefficients(target, centre, phi, mass)
    if p.spectral_band:
        s = spectral.band_scales(ca, cb, p.headroom)
    else:  # one shared range for every mode (experiments only)
        top = max(np.abs(ca).max(), 0.0 if cb is None else np.abs(cb).max()) or 1.0
        s = np.full(len(ca), top * (1 + p.headroom))
    diag = spectral.diagonal(verts)

    def err(v, c):
        """RMS distance of the truncated shape from ``v``, over A's bounding-box diagonal."""
        return float(np.sqrt(((spectral.reconstruct(c, centre, phi) - v) ** 2).sum(1).mean()) / diag)

    info = {"k": k, "rows": len(ca), "error_a": err(verts, ca),
            "error_b": None if cb is None else err(target, cb)}
    return {"vals": ca / s[:, None], "vals_b": None if cb is None else cb / s[:, None],
            "centre": centre, "basis": phi, "scales": s, "info": info}


def encoded_shape(e: Encoded, which: str = "a") -> np.ndarray:
    """The positions a noiseless readout returns: A (or B) as encoded.

    The original vertices for every payload except spectral, where it is the mesh
    truncated to the encoded modes (and quantised, with ``quant_bits``).

    Args:
        e: The encoding.
        which: "a", or "b" for a morph's B.

    Returns:
        np.ndarray: (N, 3) positions.
    """
    if e.payload != "spectral":
        return e.verts if which == "a" else e.target
    prep = e.prep if which == "a" else e.prep_b
    return from_payload(e, prep.reference.reshape(e.slots.shape))


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


def place(graph, n, comps, order=None, quality=None, per_slot=1, grow=None):
    """Capacity slot and field (0 = θ, 1 = φ) of every value.

    ``order=None`` is the naive axis-major layout (all x, then all y, then all z).
    Otherwise each cell takes the next mesh-neighbouring vertices from ``order``
    and stores them as [x1..xm, y1..ym, z1..zm], so a vertex's components share a
    cell and every cell holds a smooth patch of the surface.

    Cells are filled in graph order, or, given ``quality``, in Tessa's
    ``slot_order``: best data qubit first, so a mesh smaller than the chip lands
    only on its best qubits.

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
        pos = np.arange(comps) * n + np.arange(n)[:, None]  # position t: slot ranked[t % cap], field t // cap
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

    Spectral: each mode's (its x coefficient), since a vertex is spread over every
    mode; rows then are modes, not vertices.

    Args:
        e: The encoding.

    Returns:
        np.ndarray: (rows,) qubit indices (physical on a chip); rows = N vertices, or
        the modes of a spectral encoding.
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
        Tessa's decoder uses), in ``field`` 0 (θ) or 1 (φ: split mode, spherical payload).
        Spectral: ``i`` is a mode (low frequency first), ``c`` its x, y or z coefficient.
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
        float: Clamp in radians. A clamp of 0 (or None) is Tessa's default too: the
        transition decode divides by sin θ, which a θ of 0 would make 0.
    """
    return p.clamp or DEFAULT_CLAMP


def morph_range(flat, flat_b, payload: str, headroom: float) -> tuple[float, float]:
    """The value range A and B share in a morph (θ and φ use the same scale).

    Xyz: symmetric about 0, out to the largest |coordinate| of either mesh. Radial:
    centred on A's mean radius, a narrower range than (0, r_max), so the same readout
    noise is a smaller share of each radius. Spectral: (-1, 1), the band scales already
    hold the headroom.

    Args:
        flat: A's prepared values (``prepare_field`` output), flat.
        flat_b: B's prepared values, same length.
        payload: "radial", "xyz" or "spectral".
        headroom: Fractional padding on the range.

    Returns:
        tuple: ``(lo, hi)`` for ``tessa.encode``'s ``value_range`` and ``value_range_b``.
    """
    if payload == "spectral":
        return (-1.0, 1.0)
    if payload == "xyz":
        top = max(float(np.abs(flat).max() or 1.0), float(np.abs(flat_b).max())) * (1 + headroom)
        return (-top, top)
    mid = float(flat.mean())
    half = max(float(np.abs(flat - mid).max()), float(np.abs(flat_b - mid).max())) or mid or 1.0
    half *= 1 + headroom
    return (mid - half, mid + half)


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

    Meshes too large for one run (on the largest lattice, or on a chip) are split into
    connected batches, one circuit each; on a chip each later batch starts on the
    next-best data qubits. The spectral payload is always one run: its rows are modes,
    low frequency first, each mode's x, y, z on one data qubit, so with quality
    placement the modes that move the shape most land on the best qubits.

    Args:
        verts: (N, 3) vertex positions.
        faces: Polygon vertex-index lists.
        p: Encoder settings.
        target: Morph only: the dict :func:`morph.correspond` returns, or None to
            build it from ``p.target``.

    Returns:
        Encoded: The encoding and everything needed to decode it.

    Raises:
        ValueError: For an unknown mode or payload, a morph without B, a spectral
            morph front, or a mesh that does not fit.
    """
    mode = p.mode
    if mode not in MODES:
        raise ValueError(f"unknown mode {p.mode!r}; use one of {MODES}")
    if p.payload not in PAYLOADS:
        raise ValueError(f"unknown payload {p.payload!r}; use one of {PAYLOADS}")
    spherical, spec = p.payload == "spherical", p.payload == "spectral"
    if spherical and mode != "theta":
        raise ValueError("The spherical payload fills θ and φ itself; use Mode θ (Z)")
    if spec and p.front is not None:
        raise ValueError(SPECTRAL_FRONT)
    if p.readout not in READOUTS:
        raise ValueError(f"unknown readout {p.readout!r}; use one of {READOUTS}")
    if (p.readout != "z" or p.pilots) and (mode != "theta" or spherical):
        what = "pilot DRR" if p.readout == "z" else f"The {p.readout} readout"
        raise ValueError(f"{what} reads one θ field: use Mode θ and a payload other than spherical "
                         "(spherical already reads its direction in Z, X and Y)")
    if p.pilots and p.readout == "snap":
        raise ValueError("Snap readout snaps to its own levels: it needs no pilots (choose another DRR)")
    room = PILOT_ROOM if p.pilots else 0
    per = 2 if mode == "split" else 1
    verts = np.asarray(verts, dtype=float)
    vals_b = dirs_b = prep_b = match = tgt = sp = None
    clamp, clamp_b = p.clamp, p.clamp_b
    if mode == "morph":
        if target is None:
            if p.target is None:
                raise ValueError("Morph: upload a B mesh")
            target = correspond(verts, faces, p.target, p.shared_topology)
        match = target
        tgt = np.asarray(match["target"], float)
        clamp = clamp_b = morph_clamp(p)
    if spec:  # one row per mode; the centre (A's M-weighted centroid) stays classical
        sp = spectral_payload(verts, faces, p, run_positions(p, per), tgt)
        vals, vals_b, centre, dirs = sp["vals"], sp["vals_b"], sp["centre"], None
    else:
        vals, centre, dirs = to_payload(verts, p.payload)
        if mode == "morph":
            centre = verts.mean(0)
            if p.payload == "xyz":
                vals, vals_b = verts - centre, tgt - centre
            else:
                vals_b, dirs_b = radial(tgt, centre)
    scales = None if sp is None else sp["scales"]
    if spherical:  # encode fractions: polar of π, azimuth of 2π, radius of its padded max
        scales = np.array([np.pi, 2 * np.pi, (vals[:, 2].max() or 1.0) * (1 + p.headroom)])
        vals = vals / scales
    n, comps = vals.shape
    slot_comps = 2 if spherical else comps  # spherical: a direction slot and a radius slot
    flat, prep = prepare_field(vals.ravel(), quant_bits=p.quant_bits)
    if vals_b is not None:
        flat_b, prep_b = prepare_field(vals_b.ravel(), quant_bits=p.quant_bits)
        rng = morph_range(flat, flat_b, p.payload, p.headroom)
    elif spec:  # the band scales already hold the headroom
        rng = (-1.0, 1.0)
    else:
        top = float(np.abs(flat).max() or 1.0) * (1 + p.headroom)
        rng = (-top, top) if p.payload == "xyz" else (0.0, top)
    mesh_layout = p.layout == "mesh" or spec  # spectral: whole modes per cell, in mode order
    quality = None
    if p.backend in DEVICES:
        graph, name = devices.graph(p.backend), f"ibm_{p.backend} (whole chip)"
        quality = chip_quality(p.backend)
    else:
        try:
            graph = fit_lattice(n + room, slot_comps, mesh_layout, per)  # room: pilots' spare slots
        except ValueError:  # too big for the largest lattice: read it in several runs
            graph = Lattice(MAX_ROWS, MAX_COLS)
        name = f"{graph.rows}x{graph.cols} lattice"
    k_runs = batches_needed(n, slot_comps, per * graph.capacity - room)
    groups = split_batches(verts, faces, k_runs) if k_runs > 1 else [np.arange(n)]
    slots, field = np.empty((n, comps), int), np.empty((n, comps), int)
    batch = np.zeros(n, int)
    vals = flat.reshape(n, comps)
    vals_b = None if vals_b is None else flat_b.reshape(n, comps)
    parts, skip, pilots = [], 0, []
    for k, idx in enumerate(groups):
        if k_runs == 1:
            v_k, f_k = verts, faces
        else:
            v_k, f_k = verts[idx], submesh(faces, idx)
        if spec:  # modes low first, a mode's x, y, z on one data qubit (BFS patches mean nothing here)
            order, grow = np.arange(len(idx)), None
        else:
            order = bfs_order(len(idx), f_k) if mesh_layout else None
            grow = (lambda sizes, v=v_k, f=f_k, o=order: grow_patches(v, f, sizes, o)) if mesh_layout else None
        q_k = None
        if p.placement == "quality" and quality is not None:  # later batches: next-best qubits
            q_k = rotated_quality(quality, skip) if skip else quality
        s_k, fld_k = place(graph, len(idx), slot_comps, order, q_k, per, grow)
        if spherical:  # polar in θ and azimuth in φ of the direction slot, radius in θ of its own
            s_k, fld_k = s_k[:, [0, 0, 1]], np.broadcast_to([0, 1, 0], (len(idx), 3)).copy()
        slots[idx], field[idx], batch[idx] = s_k, fld_k, k
        if q_k is not None:  # the best data qubits whose capacity this batch takes first
            # Counted as if its values packed densely. place() puts whole vertices in a
            # cell, so radial matches it exactly; xyz and split batches (sized to the
            # chip) overflow whole-vertex packing and spill over every data qubit, so
            # all cells are used either way, and counting every one would rotate the
            # ranking a full turn (each batch back on the best qubits). Measured on
            # Suzanne (fez, phoenix): the dense count is the better spread.
            sizes = np.diff(cells(graph), axis=1)[np.argsort(-q_k, kind="stable"), 0] * per
            skip += int(np.searchsorted(np.cumsum(sizes), len(idx) * slot_comps)) + 1
        if spherical:
            parts.append(_spherical_circuit(graph, p, s_k, fld_k, vals[idx]))
            continue
        pil = _pilots(graph, s_k, rng) if p.pilots else None
        pilots.append(pil)
        parts.append(_circuit(graph, mode, p, s_k, fld_k, vals[idx],
                              None if vals_b is None else vals_b[idx], rng, clamp, clamp_b, pil))
    circuit, enc, single_twoq = parts[0]
    return Encoded(verts, faces, graph, name, circuit, enc, prep, slots, p.payload, centre,
                   dirs, single_twoq, quality,
                   field if mode == "split" or spherical else None, mode,
                   parts=tuple((c, e) for c, e, _ in parts) if k_runs > 1 else (),
                   batch=batch if k_runs > 1 else None,
                   target=None if match is None else tgt,
                   dirs_b=dirs_b, prep_b=prep_b, match=match, scales=scales,
                   basis=None if sp is None else sp["basis"], spectral=None if sp is None else sp["info"],
                   readout=p.readout, pilots=tuple(pilots) if p.pilots else ())


def _spherical_circuit(graph, p: Params, slots, field, vals):
    """One run's spherical-payload circuit via ``tessa.encode`` in ``pair_full`` mode.

    Each vertex's direction slot holds the polar angle in θ and the azimuth in φ, so
    its Bloch vector points along the vertex's own direction (squeezed into the θ
    clamp); its radius slot holds the radius in θ. Tessa reads ``pair_full`` in Z, X
    and Y, so φ covers the whole circle. φ has no clamp: azimuth 0 and 2π are the
    same direction, so the seam needs no gap.

    Args:
        graph: Tessa ``Lattice`` or ``TessaGraph``.
        p: Settings (θ clamp, compression).
        slots: (n, 3) capacity slots of polar, azimuth and radius.
        field: (n, 3) 0 = θ, 1 = φ (``[0, 1, 0]`` per vertex).
        vals: (n, 3) prepared fractions in [0, 1].

    Returns:
        tuple: ``(circuit, enc, single_twoq)``: the circuit, its Tessa ``Encoding``,
        and the two-qubit gates of the θ field alone.
    """
    a = fill(graph, slots[field == 0], vals[field == 0])
    b = fill(graph, slots[field == 1], vals[field == 1])
    kw = dict(value_range=(0.0, 1.0), compression=p.compression)
    single, _ = encode(graph, a, **kw)
    circuit, enc = encode(graph, a, b, mode="pair_full", clamp=p.clamp, clamp_b=0.0,
                          value_range_b=(0.0, 1.0), **kw)
    return circuit, enc, two_qubit(single)


def _circuit(graph, mode, p: Params, slots, field, vals, vals_b, rng, clamp, clamp_b, pilots=None):
    """One run's circuit via ``tessa.encode``.

    Split uses Tessa's ``pair_full`` mode, a second field in φ on the full circle, read
    in Z, X and Y (twice the half circle's room: about half the error under noise);
    morph its ``pair`` mode, φ on the half circle, which the basis sweep reads between
    Z and X. Mode θ is read as ``p.readout`` says (READOUTS).

    Args:
        graph: Tessa ``Lattice`` or ``TessaGraph``.
        mode: One of MODES.
        p: Settings (compression).
        slots: (n, comps) capacity slots of this run's values.
        field: (n, comps) 0 = θ, 1 = φ.
        vals: (n, comps) prepared values.
        vals_b: Morph: (n, comps) B's prepared values, else None.
        rng: ``(lo, hi)`` value range shared by the geometry fields.
        clamp: θ clamp (ignored for morph's θ-only reference circuit).
        clamp_b: φ clamp.
        pilots: ``(slots, values)`` of known values for spare slots (:func:`_pilots`), or None.

    Returns:
        tuple: ``(circuit, enc, single_twoq)``: the circuit, its Tessa ``Encoding`` (a
        :class:`SnapEncoding` for the snap readout), and the two-qubit gates of the θ
        field alone.
    """
    if mode == "theta" and p.readout == "snap":
        circuit, enc = _snap_circuit(graph, slots, vals, rng)
        return circuit, enc, two_qubit(circuit)
    s0, v0 = slots[field == 0], vals[field == 0]
    if pilots is not None:
        s0, v0 = np.r_[s0, pilots[0]], np.r_[v0, pilots[1]]
    a = fill(graph, s0, v0)
    kw = dict(value_range=rng, compression=p.compression)
    single, enc = encode(graph, a, clamp=None if mode == "morph" else clamp, **kw)
    single_twoq = two_qubit(single)
    if mode == "theta":
        if p.readout == "bloch":  # the same circuit (φ = 0), decoded from Z, X and Y
            enc = replace(enc, mode="pair_full", clamps=(enc.clamps[0], 0.0), ranges=(enc.ranges[0], (0.0, 1.0)))
        return single, enc, single_twoq
    # a second field in φ on the same scale
    if mode == "morph":  # B's matching value in the same slot
        b = fill(graph, slots, vals_b)
        circuit, enc = encode(graph, a, b, clamp=clamp, clamp_b=clamp_b, value_range_b=rng, **kw)
    else:  # split: the values placed in φ, on the full circle
        b = fill(graph, slots[field == 1], vals[field == 1])
        circuit, enc = encode(graph, a, b, mode="pair_full", clamp=clamp, clamp_b=clamp_b, value_range_b=rng, **kw)
    return circuit, enc, single_twoq


@dataclass(frozen=True)
class SnapEncoding:
    """A snap readout's Tessa ``SymbolEncoding``, with the value range its levels span.

    Attributes:
        symbols: Tessa's ``SymbolEncoding`` (constellation, one symbol per slot).
        ranges: ``((lo, hi),)``: level 0 is ``lo``, level SNAP_LEVELS - 1 is ``hi``.
    """
    symbols: object
    ranges: tuple

    @property
    def graph(self):
        """The graph the symbols sit on.

        Returns:
            Tessa ``Lattice`` or ``TessaGraph``.
        """
        return self.symbols.graph

    @property
    def mode(self) -> str:
        """Tessa's mode underneath (``pair_full``: read in Z, X and Y).

        Returns:
            str: The mode name.
        """
        return self.symbols.mode

    def values(self, counts) -> np.ndarray:
        """The snapped values, from per-cell counts.

        Args:
            counts: ``{site: {basis: counts}}``, as ``tessa.decode(return_counts=True)``.

        Returns:
            np.ndarray: (capacity,) values in the encoded units at their slots (0 elsewhere).
        """
        lo, hi = self.ranges[0]
        sym = np.asarray(symbols_from_counts(counts, self.symbols), float).ravel()
        out = np.zeros(self.graph.capacity)
        out[self.symbols.slot_index] = lo + (hi - lo) * sym / (SNAP_LEVELS - 1)
        return out


def _snap_circuit(graph, slots, vals, rng):
    """The snap readout's circuit: Tessa's ``encode_symbols`` constellation at our slots.

    Args:
        graph: Tessa ``Lattice`` or ``TessaGraph``.
        slots: (n, comps) capacity slots (one value each).
        vals: (n, comps) prepared values.
        rng: ``(lo, hi)`` the levels span.

    Returns:
        tuple: ``(circuit, SnapEncoding)``.
    """
    lo, hi = rng
    sym = np.rint(np.clip((np.ravel(vals) - lo) / (hi - lo), 0, 1) * (SNAP_LEVELS - 1)).astype(int)
    circuit, enc = encode_symbols(graph, sym, SNAP_LEVELS, scheme="constellation", per_slot=1,
                                  slot_order=np.ravel(slots))
    return circuit, SnapEncoding(enc, (tuple(rng),))


def _pilots(graph, slots, rng) -> tuple:
    """Known values for every spare slot of a run: alternately 5 % above its range's
    bottom and 5 % below its top. More pilots fit the line better (blob on Phoenix:
    8 pilots, or the 12 on the mesh's own qubits, add error; all 460 match classical
    DRR), worth the gates of the data qubits that hold only pilots.

    Args:
        graph: Tessa ``Lattice`` or ``TessaGraph``.
        slots: (n, comps) the run's capacity slots.
        rng: ``(lo, hi)`` value range.

    Returns:
        tuple: ``(slots, values)``.

    Raises:
        ValueError: Fewer than two spare slots.
    """
    spare = np.setdiff1d(np.arange(graph.capacity), slots)
    if len(spare) < 2:
        raise ValueError("Pilot DRR needs two spare slots in every run, and this mesh fills them")
    lo, hi = rng
    return spare, np.where(np.arange(len(spare)) % 2, hi - 0.05 * (hi - lo), lo + 0.05 * (hi - lo))


def _calibrate(out, pilots) -> np.ndarray:
    """A run's readout corrected by its pilots: the straight line (least squares) that
    takes the pilots' readings back to their known values, applied to every slot.

    Args:
        out: (capacity,) decoded values of one run.
        pilots: ``(slots, values)`` (:func:`_pilots`).

    Returns:
        np.ndarray: (capacity,) corrected values.
    """
    slots, values = pilots
    gain, offset = np.polyfit(out[slots], values, 1)
    return gain * out + offset


def gain_shots(shots: int, gain: float) -> int:
    """Shots whose shot noise is ``gain`` times that of ``shots``.

    Shot noise falls as 1/√shots, so a readout at shots / gain² moves ``gain`` times as
    far: Material's exaggeration made by the readout itself, not by scaling it. Noise
    from the chip's gates does not scale this way; only its shot noise does.

    Args:
        shots: The readout's shots.
        gain: Wanted noise multiple (0 or less: the most shots).

    Returns:
        int: Shots, in [MIN_SHOTS, MAX_SHOTS].
    """
    if gain <= 0:
        return MAX_SHOTS
    return int(np.clip(round(shots / gain ** 2), MIN_SHOTS, MAX_SHOTS))


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
    snapshot backend) with the cached noise model. "flat noise": ``noisy_sampler``'s
    construction, IBM_LIKE_RATES scaled by ``p.noise_strength`` (errors times it,
    T1/T2 divided by it). Otherwise the noiseless ``aer_sampler``'s. Lattice devices
    and noise models are built once (:func:`_lattice`).

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
    device, models = _lattice(graph)
    if p.backend == "flat noise" and p.noise_strength > 0:  # tessa.noisy_sampler, its model built once
        s = p.noise_strength
        if s not in models:
            rates = {k: (v / s if k in ("t1", "t2") else v * s if k.endswith("error") else v)
                     for k, v in IBM_LIKE_RATES.items()}
            models[s] = build_average_noise_model(graph, rates)
        return _aer_sampler(device, noise_model=models[s], seed=seed)
    return _aer_sampler(device, seed=seed)  # tessa.aer_sampler


_LATTICES = {}  # graph key -> (graph, device, {noise strength: flat noise model})


def _lattice(graph):
    """A graph's simulator device and its "flat noise" models, built once.

    ``tessa.aer_sampler`` and ``noisy_sampler`` build a ``topology_device`` (and the
    noisy one its noise model, about 1 s) on every call, and a run reads many seeds;
    the device staying the same object also lets :func:`_cell_jobs` keep its
    transpiles. A ``Lattice`` (not hashable) is keyed by its shape; any other graph
    (the edge readout's fixed PAIR) by identity, kept alive by the cache.

    Args:
        graph: Tessa ``Lattice`` or ``TessaGraph``.

    Returns:
        tuple: ``(device, models)``: Tessa's ``topology_device`` and a dict of
        ``build_average_noise_model`` results by noise strength, filled by :func:`sampler`.
    """
    key = (graph.rows, graph.cols) if isinstance(graph, Lattice) else id(graph)
    if key not in _LATTICES:
        _LATTICES[key] = graph, topology_device(graph), {}
    return _LATTICES[key][1:]


def batch_seed(seed: int, k: int) -> int:
    """Seed of batch ``k`` in a readout, so batches never share a noise draw.

    Args:
        seed: The readout's seed.
        k: Batch index.

    Returns:
        int: ``seed + BATCH_SEED_STEP * k``.
    """
    return seed + BATCH_SEED_STEP * k


def reference_seed(seed: int) -> int:
    """Seed of the morph's alpha = 0 (Z) reference readout for a readout seeded ``seed``.

    The reference and the tilted readout are two runs on the chip. Read with one
    seed, Aer would draw the same random numbers for both, so at small alpha their
    shot noise would largely cancel in the transition decode: quieter than a chip.

    Args:
        seed: The readout's (or first frame's) seed.

    Returns:
        int: ``seed + REFERENCE_SEED_OFFSET``.
    """
    return seed + REFERENCE_SEED_OFFSET


def sweep_circuit(e: Encoded, alpha, part: int = 0):
    """Morph's basis sweep: ``part``'s pair circuit with RY(-alpha) on every data qubit.

    The address qubits are never turned, so Z reads cos(alpha) Z + sin(alpha) X on
    every cell. One angle for every qubit is Tessa's own
    ``tessa.hardware.pipeline._tilted``; Tessa has no per-qubit form, so differing
    angles are applied here. The result decodes as one field with A's clamp and
    range. Cached per (part, alpha) on ``e``.

    A batched mesh puts different vertices on the same data qubit in each run, so
    ``alpha`` may also be (runs, groups): run ``part`` uses row ``part``.

    Args:
        e: A morph encoding.
        alpha: Scalar angle, one per data qubit (graph.groups order), or
            (e.runs, groups) one per (batch, data qubit).
        part: Run (batch) index.

    Returns:
        tuple: ``(circuit, enc)``: the tilted circuit and its single-field Encoding.

    Raises:
        ValueError: If ``alpha`` has none of those shapes.
    """
    groups = e.graph.groups
    alpha = _alpha_grid(e, alpha)
    if alpha.ndim == 2:
        alpha = alpha[part]
    if alpha.ndim == 0:
        alpha = np.full(len(groups), float(alpha))
    if alpha.shape != (len(groups),):
        raise ValueError(f"alpha needs one angle per data qubit ({len(groups)}), got shape {alpha.shape}")
    key = (part, alpha.tobytes())
    if key not in e._sweeps:
        circuit, enc = e.run_parts[part]
        if (alpha == alpha[0]).all():
            qc = _tilted(circuit, enc, float(alpha[0]))
        else:
            qc = circuit.copy()
            for g, a in zip(groups, alpha):
                qc.ry(-a, g.data_qubit)
        # decoded as one field with A's clamp and range
        e._sweeps[key] = (qc, replace(enc, mode="single", clamps=enc.clamps[:1], ranges=enc.ranges[:1]))
    return e._sweeps[key]


def _cell_jobs(e: Encoded, circuit, enc, device) -> tuple[list, list]:
    """``tessa.decode``'s per-cell jobs for ``circuit``, transpiled once and kept on ``e``.

    The same jobs as ``decode(..., per_group=True, halo="traced")`` builds, from
    Tessa's own steps: each cell with its diamond halo (``cell_halo``,
    ``cell_circuit``), measured in each basis its mode reads (``codec._measured``),
    transpiled onto its own qubits, and the halo traced out (``codec._trace_halo``).
    ``decode`` rebuilds them on every call; a readout here only samples them.

    Args:
        e: The encoding that owns ``circuit`` (holds the cache).
        circuit: Unmeasured Tessa circuit (a run's, or a sweep's tilted one).
        enc: Its Tessa ``Encoding``.
        device: The device to transpile for (a cached object: part of the key).

    Returns:
        tuple: ``(keys, pubs)``: ``(site, basis)`` per job, and the transpiled jobs.
    """
    key = (id(circuit), id(device))
    if key not in e._jobs:
        graph, keys, pubs = enc.graph, [], []
        for g in graph.groups:
            halo = cell_halo(graph, g, "diamond")
            cell = cell_circuit(circuit, g, halo)
            for basis in get_mode(enc.mode).bases:
                tqc = transpile(_measured(cell, [0], basis), device, optimization_level=1, seed_transpiler=0,
                                initial_layout=[g.data_qubit, *g.address_qubits, *halo])
                keys.append((g.site, basis))
                pubs.append(_trace_halo(tqc, halo))
        e._jobs[key] = (circuit, device, keys, pubs)  # circuit and device kept, so their ids stay theirs
    return e._jobs[key][2:]


MAX_JOB_SHOTS = 1 << 19  # shots per circuit in one job of several frames (bounds the result's memory)


def _decode_frames(e: Encoded, circuit, enc, inner, frames: int, shots: int) -> list:
    """``frames`` readouts of one circuit from one simulator job, decoded by Tessa.

    The cell jobs (:func:`_cell_jobs`) are sampled frames × shots at once; each frame
    is its own ``shots`` of them, decoded with ``tessa.decode_from_cell_counts``. So it
    is distributed exactly as a separately seeded readout, and one frame is exactly
    ``tessa.decode(circuit, enc, inner, shots=shots)``.

    Args:
        e: The encoding that owns ``circuit``.
        circuit: Unmeasured Tessa circuit.
        enc: Its Tessa ``Encoding``.
        inner: Tessa sampler (:func:`sampler`).
        frames: Readouts wanted.
        shots: Shots per readout.

    Returns:
        list: One decoded result per frame (an array, or a tuple of arrays for a
        pair encoding).

    Raises:
        ValueError: ``inner`` cannot stand in for the traced halo (Tessa's check).
    """
    _check_traceable(inner)
    keys, pubs = _cell_jobs(e, circuit, enc, device_of(inner))
    result = inner.run(pubs, shots=shots * frames).result()
    per_job = [_frame_counts(r.data.meas, shots, frames) for r in result]
    out = []
    for k in range(frames):
        counts = {}
        for (site, basis), by_frame in zip(keys, per_job):
            counts.setdefault(site, {})[basis] = by_frame[k]
        out.append(enc.values(counts) if isinstance(enc, SnapEncoding) else decode_from_cell_counts(counts, enc))
    return out


def _frame_counts(bits, shots: int, frames: int) -> list[dict]:
    """``BitArray.get_counts()`` of each frame's ``shots``, counted by outcome.

    ``get_counts`` turns every shot into a string; a cell has only a few distinct
    outcomes, so each shot becomes an integer and only the distinct ones are
    formatted. The dicts are equal to ``get_counts``'s.

    Args:
        bits: A SamplerV2 ``BitArray`` of ``shots * frames`` shots (no extra shape).
        shots: Shots per frame.
        frames: Frames, in shot order.

    Returns:
        list[dict]: Per frame, ``{bitstring: count}``.
    """
    if bits.num_bits > 62:  # too wide for one integer: qiskit's own count
        return [bits.slice_shots(range(k * shots, (k + 1) * shots)).get_counts() for k in range(frames)]
    raw = bits.array.reshape(-1, bits.array.shape[-1]).astype(np.int64)
    codes = raw @ (256 ** np.arange(raw.shape[1] - 1, -1, -1, dtype=np.int64))  # big-endian bytes
    out = []
    for k in range(frames):
        values, n = np.unique(codes[k * shots:(k + 1) * shots], return_counts=True)
        out.append({format(int(v), f"0{bits.num_bits}b"): int(c) for v, c in zip(values, n)})
    return out


def _read(e: Encoded, p: Params, seed: int, runs, frames: int = 1, pilots=()):
    """``tessa.decode`` every run with its own seed and gather each vertex's values.

    Several frames share each run's cached jobs and one simulator job (:func:`_decode_frames`),
    in jobs of at most MAX_JOB_SHOTS shots per circuit: the job holding frames i..j is
    seeded seed + i (run k: :func:`batch_seed` of that), so with one frame per job frame
    i is exactly a readout seeded seed + i.

    Args:
        e: The encoding (slots, batches).
        p: Readout settings (backend, shots).
        seed: The readout's seed; run k uses :func:`batch_seed`.
        runs: One ``(circuit, enc)`` per batch.
        frames: Readouts to take.
        pilots: Per run, ``(slots, values)`` to correct it by (:func:`_calibrate`), or ().

    Returns:
        tuple: ``(a, b)``: (frames, N, comps) first-field values, and (frames, N, comps)
        second-field values for a pair encoding, else None.
    """
    per_job = max(1, MAX_JOB_SHOTS // p.shots)
    a = np.empty((frames, *e.slots.shape))
    b = None
    for k, (circuit, enc) in enumerate(runs):
        m = e.batches == k
        for i0 in range(0, frames, per_job):
            n = min(per_job, frames - i0)
            inner = sampler(e.graph, p, batch_seed(seed + i0, k))
            for i, out in enumerate(_decode_frames(e, circuit, enc, inner, n, p.shots), start=i0):
                oa, ob = _fields_of(out, pilots[k] if pilots else None)
                a[i, m] = oa[e.slots[m]]
                if ob is not None:
                    if b is None:
                        b = np.empty_like(a)
                    b[i, m] = ob[e.slots[m]]
    return a, b


def _fields_of(out, pilots=None) -> tuple:
    """A run's decoded capacity array(s) as ``(first field, second field or None)``,
    the first corrected by the run's pilots.

    Args:
        out: What Tessa's decode returns for one run: an array, or a tuple of two.
        pilots: The run's ``(slots, values)`` (:func:`_pilots`), or None.

    Returns:
        tuple: ``(a, b)``: (capacity,) arrays; ``b`` None for one field.
    """
    a, b = out if isinstance(out, tuple) else (out, None)
    return (a if pilots is None else _calibrate(a, pilots)), b


def _join(e: Encoded, a, b) -> np.ndarray:
    """Per-value raw readouts: split mode and the spherical payload take each value from its own field.

    Args:
        e: The encoding (``fields``).
        a: (..., N, comps) first-field values.
        b: (..., N, comps) second-field values, or None.

    Returns:
        np.ndarray: (..., N*comps) values.
    """
    if e.field is not None:
        a = np.where(e.fields == 1, b, a)
    return a.reshape(*a.shape[:-2], -1)


# --- real hardware: Tessa's whole-chip path ----------------------------------------
def hardware_circuits(e: Encoded) -> list[tuple]:
    """Every circuit one readout runs on real hardware: per run, Tessa's
    ``measurement_circuits`` (the whole chip measured in each basis its mode reads).

    ``tessa.decode(per_group=True)`` (every simulated readout here) cuts the chip into
    cells; on hardware the whole circuit runs at once, ``per_group=False``, as here.

    Args:
        e: The encoding (not a morph: its basis sweep reads one t per circuit set).

    Returns:
        list: ``(run, basis, circuit)`` per circuit, unmeasured qubits on the graph's own
        indices (a chip graph's are the chip's physical qubits).

    Raises:
        ValueError: For a morph.
    """
    if e.mode == "morph":
        raise ValueError("Hardware runs read a fixed mesh; a morph's basis sweep is simulated only")
    return [(k, basis, qc) for k, (circuit, enc) in enumerate(e.run_parts)
            for basis, qc in measurement_circuits(circuit, enc.symbols.encoding if isinstance(enc, SnapEncoding)
                                                  else enc).items()]


def values_from_counts(e: Encoded, counts) -> np.ndarray:
    """One readout's raw values from whole-chip counts, as :func:`decode_raw` returns them.

    Tessa's ``decode_from_counts`` (or, for a snap readout, ``symbols_from_counts``),
    then each run's pilots, gathered by vertex. Counts measured on a chip today can be
    decoded at any time later against the same encoding (``encode_mesh`` is
    deterministic for the same mesh and settings).

    Args:
        e: The encoding the circuits came from.
        counts: Per run, ``{basis: counts}`` (as :func:`read_hardware` returns).

    Returns:
        np.ndarray: (N*comps,) raw values, for :func:`restore`.
    """
    a, b = np.empty(e.slots.shape), None
    for k, ((_, enc), by_basis) in enumerate(zip(e.run_parts, counts, strict=True)):
        out = enc.values(by_basis) if isinstance(enc, SnapEncoding) else decode_from_counts(by_basis, enc)
        oa, ob = _fields_of(out, e.pilots[k] if e.pilots else None)
        m = e.batches == k
        a[m] = oa[e.slots[m]]
        if ob is not None:
            b = np.empty_like(a) if b is None else b
            b[m] = ob[e.slots[m]]
    return _join(e, a, b)


HARDWARE_SHOTS = 100_000  # most shots per circuit in one hardware job (as Tessa's pipeline)


def submit_hardware(e: Encoded, sampler, shots: int, frames: int = 1, device=None, optimization_level: int = 1):
    """Submit one job of every :func:`hardware_circuits` circuit, transpiled onto the
    device with the identity layout (Tessa's ``transpile_to_backend``: no SWAPs), each
    sampled ``frames`` × ``shots`` times: :func:`hardware_frames` cuts it into frames.

    Nothing here decides to spend hardware time: the caller passes the sampler (e.g.
    ``tessa.ibm_sampler``), and this call submits the job.

    Args:
        e: The encoding (a chip graph for a chip; a lattice runs on a simulator).
        sampler: A SamplerV2-like ``run(circuits, shots=...)``.
        shots: Shots per circuit per frame.
        frames: Frames (frames × shots at most HARDWARE_SHOTS).
        device: The backend to transpile for; None asks ``sampler`` (Tessa's ``device_of``).
        optimization_level: Transpiler optimisation.

    Returns:
        tuple: ``(job, reports)``: the submitted job and Tessa's transpile report per
        circuit (depths, SWAPs, two-qubit gates).

    Raises:
        ValueError: frames × shots over HARDWARE_SHOTS.
    """
    if frames * shots > HARDWARE_SHOTS:
        raise ValueError(f"{frames} frames × {shots:,} shots is over {HARDWARE_SHOTS:,} shots per circuit")
    target = sampler if device is None else device
    transpiled, reports = zip(*(transpile_to_backend(qc, target, optimization_level)
                                for _, _, qc in hardware_circuits(e)))
    return sampler.run(list(transpiled), shots=shots * frames), list(reports)


def hardware_counts(e: Encoded, result, shots: int, frames: int = 1) -> list:
    """A finished job's counts, per frame and run: ``[frame][run][basis] -> counts``.

    Args:
        e: The encoding the job was submitted for (:func:`submit_hardware`).
        result: The job's SamplerV2 result, one pub per :func:`hardware_circuits` circuit.
        shots: Shots per frame.
        frames: Frames in the job.

    Returns:
        list: Per frame, per run, ``{basis: counts}`` (:func:`values_from_counts`'s input).
    """
    jobs = hardware_circuits(e)
    per_pub = [_frame_counts(result[i].data.meas, shots, frames) for i in range(len(jobs))]
    out = [[{} for _ in range(e.runs)] for _ in range(frames)]
    for (k, basis, _), by_frame in zip(jobs, per_pub, strict=True):
        for f in range(frames):
            out[f][k][basis] = by_frame[f]
    return out


def read_hardware(e: Encoded, sampler, shots: int, device=None, optimization_level: int = 1):
    """One readout on a device, waited for: :func:`submit_hardware`, then
    :func:`values_from_counts` on its counts.

    Args:
        e: The encoding.
        sampler: A SamplerV2-like ``run(circuits, shots=...)``.
        shots: Shots per circuit.
        device: The backend to transpile for; None asks ``sampler``.
        optimization_level: Transpiler optimisation.

    Returns:
        tuple: ``(raw, counts, reports)``: (N*comps,) values, per run ``{basis: counts}``,
        and Tessa's transpile report per circuit.
    """
    job, reports = submit_hardware(e, sampler, shots, 1, device, optimization_level)
    counts = hardware_counts(e, job.result(), shots)[0]
    return values_from_counts(e, counts), counts, reports


def _alpha_grid(e: Encoded, alpha) -> np.ndarray:
    """Check a sweep angle's shape.

    Args:
        e: The encoding (runs, graph).
        alpha: Scalar, (groups,) one per data qubit, or (e.runs, groups) one per
            (batch, data qubit).

    Returns:
        np.ndarray: ``alpha`` as a float array, shape unchanged.

    Raises:
        ValueError: For any other shape.
    """
    alpha = np.asarray(alpha, float)
    g = len(e.graph.groups)
    if alpha.ndim == 0 or alpha.shape in ((g,), (e.runs, g)):
        return alpha
    raise ValueError(f"alpha needs one angle per data qubit ({g}) or per (batch, data qubit) "
                     f"({e.runs}, {g}), got shape {alpha.shape}")


def _per_value(e: Encoded, alpha) -> np.ndarray:
    """A sweep angle as one angle per encoded value.

    ``e.groups_of`` is per batch (each batch's slots index the same graph), so a
    (runs, groups) angle is looked up by each vertex's batch and data qubit.

    Args:
        e: The encoding.
        alpha: Scalar, (groups,) one per data qubit (graph.groups order), or
            (e.runs, groups) one per (batch, data qubit).

    Returns:
        np.ndarray: (N*comps,) angles.
    """
    alpha = _alpha_grid(e, alpha)
    if not alpha.ndim:
        return np.full(e.slots.size, float(alpha))
    if alpha.ndim == 2:
        return alpha[e.batches[:, None], e.groups_of].ravel()
    return alpha[e.groups_of].ravel()


def tilted_angles(e: Encoded, p: Params, seed: int, alpha, frames: int | None = None) -> np.ndarray:
    """arccos E(alpha), E = <cos(alpha) Z + sin(alpha) X>, per value.

    Readouts of the tilted circuit(s), decoded with Tessa's ``_angle_readout``: what
    Tessa's transition decodes from.

    Args:
        e: A morph encoding.
        p: Readout settings.
        seed: The readout's seed.
        alpha: Scalar, one angle per data qubit, or (runs, groups).
        frames: None for one readout; else that many, sharing each circuit's simulator
            job (:func:`_read`).

    Returns:
        np.ndarray: (N*comps,) angles in radians, or (frames, N*comps).
    """
    runs = [sweep_circuit(e, alpha, k) for k in range(e.runs)]
    runs = [(qc, _angle_readout(enc)) for qc, enc in runs]  # raw angle, as Tessa's transition
    a = _read(e, p, seed, runs, frames or 1)[0].reshape(frames or 1, -1)
    return a if frames else a[0]


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

    A value read at alpha = 0 (a frame at t = 0, or the part of the mesh a front has
    not reached) is already a Z readout, so it is decoded from that readout alone.
    Against a separate Z reference the transition would subtract two independent Z
    readouts and divide by sin(theta), making A noisier than A.

    Args:
        e: A morph encoding.
        angle_alpha: (N*comps,) :func:`tilted_angles` at ``alpha``.
        angle_z: (N*comps,) :func:`tilted_angles` at alpha = 0.
        alpha: Scalar, one angle per data qubit, or (runs, groups).

    Returns:
        np.ndarray: (N*comps,) values in A's units.
    """
    per = _per_value(e, alpha)
    angle_z = np.where(per == 0, angle_alpha, angle_z)  # alpha = 0: its own Z readout
    frame = _transition_frames(_transition_plan(e, [0.0, per]), [angle_z, angle_alpha])[1]
    lo, hi = e.enc.ranges[0]
    return lo + (hi - lo) * frame


def decode_raw(e: Encoded, p: Params, seed: int | None = None, reference=None):
    """One readout of the encoding, before DRR.

    Morph is a basis sweep: it reads one field at :func:`morph_alpha` (alpha =
    p.morph_t * pi/2, or the front's angle per (batch, data qubit)) and turns it into
    values with Tessa's transition decode and an alpha = 0 readout
    (:func:`tilted_angles`).

    Args:
        e: The encoding.
        p: Readout settings.
        seed: Sampler seed; None uses ``p.seed``.
        reference: Morph only: (N*comps,) alpha = 0 angles to reuse (run_frames
            shares one per run); None reads it now, seeded by :func:`reference_seed`.

    Returns:
        np.ndarray: (N*comps,) decoded values.
    """
    return decode_frames(e, p, 1, seed, reference)[0]


def decode_frames(e: Encoded, p: Params, n: int, seed: int | None = None, reference=None) -> np.ndarray:
    """``n`` independent readouts of the encoding at one setting, before DRR.

    Every circuit is transpiled once and sampled n × shots in one simulator job
    (:func:`_read`), each frame being its own ``p.shots`` of them: far cheaper than n
    :func:`decode_raw` calls (3-5× with noise), and distributed the same. A morph
    reads every frame at ``p.morph_t`` against one alpha = 0 readout.

    Args:
        e: The encoding.
        p: Readout settings.
        n: Frames.
        seed: Sampler seed; None uses ``p.seed``.
        reference: Morph only: as :func:`decode_raw`.

    Returns:
        np.ndarray: (n, N*comps) decoded values.
    """
    seed = p.seed if seed is None else seed
    if e.mode == "morph":
        alpha = morph_alpha(e, p.morph_t, p.front)
        angles = tilted_angles(e, p, seed, alpha, n)
        if reference is None and not np.all(alpha == 0):
            reference = tilted_angles(e, p, reference_seed(seed), 0.0)
        # alpha = 0 everywhere: each frame is its own Z readout (transition_values)
        return np.array([transition_values(e, a, a if reference is None else reference, alpha) for a in angles])
    return _join(e, *_read(e, p, seed, e.run_parts, n, e.pilots))


def _values(e: Encoded, raw, prep, drr: str) -> np.ndarray:
    """Tessa's ``restore_field`` (DRR, un-quantise) on raw decoded values.

    ``clip``: a stretch then clips to the reference's range, as Tessa recommends: every
    true value lies inside it, so no value moves further from its target (1-3 % less
    error on the blob and Suzanne, never more).

    The spherical payload restores polar, azimuth and radius each against its own
    reference, and leaves the azimuth out of DRR: it is a phase round the whole
    circle, which noise turns but does not shrink (Tessa's pipeline likewise keeps
    hue out of DRR).

    Args:
        e: The encoding (for the shape and payload).
        raw: (N*comps,) decoded values.
        prep: The field's Tessa ``FieldPrep``.
        drr: One of DRR_METHODS.

    Returns:
        np.ndarray: (N, comps) values in payload units.
    """
    if e.payload == "spherical" and drr != "none":
        raw, ref = np.reshape(raw, e.slots.shape), prep.reference.reshape(e.slots.shape)
        return np.stack([restore_field(raw[:, c], replace(prep, reference=ref[:, c]),
                                       dynamic_range="none" if c == 1 else drr, clip=True)
                         for c in range(raw.shape[1])], axis=1)
    return restore_field(raw, prep, dynamic_range=drr, clip=True).reshape(e.slots.shape)


def restore(e: Encoded, raw, drr: str = "none", dirs=None, reference=None) -> np.ndarray:
    """Classical post-processing (DRR) and back to vertex positions.

    Args:
        e: The encoding.
        raw: (N*comps,) decoded values.
        drr: One of DRR_CHOICES ("pilots": already corrected while reading).
        dirs: (N, 3) radial directions; None uses A's.
        reference: (N*comps,) what DRR stretches the readout towards, in the same
            units as ``raw``; None uses A's input. A morph frame passes its noiseless
            frame (:func:`morph_frame`): A's input would pull a frame near B back to A.

    Returns:
        np.ndarray: (N, 3) positions.
    """
    prep = e.prep if reference is None else replace(e.prep, reference=np.ravel(reference))
    drr = "none" if drr == "pilots" else drr  # the pilots corrected the readout itself (_read)
    return from_payload(e, _values(e, raw, prep, drr), dirs)


# --- morph front: one alpha per (batch, data qubit), spreading over the surface -----
def front_source(e: Encoded) -> int:
    """The front's automatic source: the A vertex that moves most (largest |B - A|).

    The maximum of the *morph Δ* colour; for apple -> bitten apple it lands in the bite.

    Args:
        e: A morph encoding.

    Returns:
        int: Vertex index.
    """
    return int(np.argmax(np.linalg.norm(e.target - e.verts, axis=1)))


def qubit_distance(e: Encoded, d_vertex) -> np.ndarray:
    """Distance of each (batch, data qubit): the median over the vertices it holds.

    In xyz a vertex's components may sit on different data qubits; each qubit counts
    the vertex once. A (batch, data qubit) holding no values gets 1, so it turns last.

    Args:
        e: The encoding (batches, ``groups_of``).
        d_vertex: (N,) per-vertex distances in [0, 1].

    Returns:
        np.ndarray: (e.runs, groups) distances.
    """
    d_vertex = np.asarray(d_vertex, float)
    g = len(e.graph.groups)
    out = np.ones((e.runs, g))
    n, comps = e.groups_of.shape
    cell = e.batches[:, None] * g + e.groups_of  # flat (batch, data qubit) of every value
    pairs = np.unique(np.column_stack([cell.ravel(), np.repeat(np.arange(n), comps)]), axis=0)
    order = np.argsort(pairs[:, 0], kind="stable")
    pairs = pairs[order]
    keys, starts = np.unique(pairs[:, 0], return_index=True)
    for key, vs in zip(keys, np.split(d_vertex[pairs[:, 1]], starts[1:])):
        out.flat[key] = np.median(vs)
    return out


def front_distances(e: Encoded, source: int | None = None):
    """Per-vertex and per-(batch, data qubit) surface distances from the front's source.

    Cached on ``e`` per source.

    Args:
        e: A morph encoding.
        source: Source vertex; None = :func:`front_source`.

    Returns:
        tuple: ``(source, d_vertex, d_qubit)``: the source used, (N,) distances
        (:func:`morph.geodesic`) and (e.runs, groups) medians (:func:`qubit_distance`).
    """
    source = front_source(e) if source is None else int(source)
    if source not in e._fronts:
        d = geodesic(e.verts, e.faces, source)
        e._fronts[source] = (source, d, qubit_distance(e, d))
    return e._fronts[source]


def morph_alpha(e: Encoded, t: float, front=None):
    """The sweep angle of a morph frame at t.

    Args:
        e: A morph encoding.
        t: Morph parameter in [0, 1].
        front: None for the uniform sweep, or ``(width, source)`` (``Params.front``).

    Returns:
        float | np.ndarray: ``t * pi/2``, or the front's (e.runs, groups) angles
        (:func:`morph.front_alpha` of :func:`front_distances`).

    Raises:
        ValueError: For a front on the spectral payload.
    """
    if front is None:
        return t * np.pi / 2
    if e.payload == "spectral":
        raise ValueError(SPECTRAL_FRONT)
    width, source = front
    return front_alpha(front_distances(e, source)[2], t, width)


def per_vertex(e: Encoded, values) -> np.ndarray:
    """A per-value array as one number per vertex: the mean over its components.

    Args:
        e: The encoding (its ``slots`` shape).
        values: (N*comps,) one number per encoded value.

    Returns:
        np.ndarray: (N,) means (an xyz vertex's values may sit on several qubits).
    """
    return np.asarray(values, float).reshape(e.slots.shape).mean(1)


def front_weights(e: Encoded, t: float, front, smooth: bool = False) -> np.ndarray:
    """How far each vertex has turned from A (0) to B (1) at t, under a morph front.

    Args:
        e: A morph encoding.
        t: Morph parameter.
        front: ``(width, source)`` (``Params.front``; source None = auto).
        smooth: The smooth reference (:func:`front_smooth_alpha`: each vertex's own
            distance, which no circuit can make) instead of its data qubits' angle.

    Returns:
        np.ndarray: (N,) weights in [0, 1]: :func:`per_vertex` of the sweep angle / (π/2).
    """
    alpha = front_smooth_alpha(e, t, front) if smooth else _per_value(e, morph_alpha(e, t, front))
    return per_vertex(e, alpha) / (np.pi / 2)


def front_smooth_alpha(e: Encoded, t: float, front) -> np.ndarray:
    """The smooth reference: the front with each vertex's own distance, per value.

    No circuit can do this (it needs a different angle per slot); it shows what the
    chip's patch quantisation adds.

    Args:
        e: A morph encoding.
        t: Morph parameter in [0, 1].
        front: ``(width, source)``.

    Returns:
        np.ndarray: (N*comps,) angles, one per encoded value.
    """
    width, source = front
    d = front_distances(e, source)[1]
    return np.repeat(front_alpha(d, t, width), e.slots.shape[1])


def morph_dirs(e: Encoded, t: float, a_dirs_only: bool = False, alpha=None):
    """Radial morph: each vertex's direction at t.

    The turning from A's to B's directions (slerp) is classical; only the radius is
    read from the state. With a front (``alpha`` per value), each vertex turns by its
    own weight alpha / (pi/2), so the direction moves with the front, not ahead of it.

    Args:
        e: A morph encoding.
        t: Morph parameter in [0, 1].
        a_dirs_only: Keep A's directions throughout.
        alpha: None for the uniform sweep, or (N*comps,) angles per value.

    Returns:
        np.ndarray | None: (N, 3) unit directions, or None for every payload but radial.
    """
    if e.payload != "radial":
        return None
    if a_dirs_only:
        return e.dirs
    if alpha is not None:
        t = np.asarray(alpha, float).reshape(e.slots.shape)[:, :1] / (np.pi / 2)
    return slerp(e.dirs, e.dirs_b, t)


def _front_value_alpha(e: Encoded, t: float, front):
    """Per-value angles of a front frame, for :func:`morph_dirs`.

    Args:
        e: A morph encoding.
        t: Morph parameter.
        front: None or ``(width, source)``.

    Returns:
        np.ndarray | None: (N*comps,) angles, or None without a front.
    """
    return None if front is None else _per_value(e, morph_alpha(e, t, front))


def morph_frame(e: Encoded, raw, t: float, drr: str = "none", a_dirs_only: bool = False, front=None):
    """One decoded morph frame at t.

    The basis-sweep readout already holds the morph; it only needs its direction at
    t (radial payload).

    Args:
        e: A morph encoding.
        raw: (N*comps,) the sweep's decoded values at t.
        t: Morph parameter in [0, 1].
        drr: One of DRR_METHODS.
        a_dirs_only: Radial: keep A's directions.
        front: None, or ``(width, source)`` the frame was read with.

    Returns:
        np.ndarray: (N, 3) positions.
    """
    # DRR's reference is the noiseless frame at t, as Tessa's pipeline gives transition
    # frames their ideal_frames (A's input would stretch a frame near B back towards A)
    reference = None if drr == "none" else sweep_values(e, morph_alpha(e, t, front))
    return restore(e, raw, drr, morph_dirs(e, t, a_dirs_only, _front_value_alpha(e, t, front)), reference)


def sweep_values(e: Encoded, alpha, per_value: bool = False) -> np.ndarray:
    """Noiseless sweep readout in value units: Tessa's ``tessa.hardware.pipeline.ideal_frames``.

    Inputs are normalised with Tessa's ``codec._normalise``, as ``encode`` does.

    Args:
        e: A morph encoding.
        alpha: Scalar, one angle per data qubit, (runs, groups), or with
            ``per_value`` (N*comps,) one angle per value.
        per_value: ``alpha`` is already one angle per value (e.g. the smooth
            reference, which no circuit can produce).

    Returns:
        np.ndarray: (N, comps) values.
    """
    lo, hi = e.enc.ranges[0]
    a01, b01 = (_normalise(prep.reference, (lo, hi))[0] for prep in (e.prep, e.prep_b))  # as encode
    alpha = np.asarray(alpha, float) if per_value else _per_value(e, alpha)
    frame = ideal_frames(_transition_plan(e, [alpha], (a01, b01)))[0]
    return (lo + (hi - lo) * frame).reshape(e.slots.shape)


def morph_expected(e: Encoded, t: float, a_dirs_only: bool = False, front=None):
    """Noiseless reference positions of a morph frame at t, for error colouring.

    Args:
        e: A morph encoding.
        t: Morph parameter in [0, 1].
        a_dirs_only: Radial: keep A's directions.
        front: None, or ``(width, source)``: the front's per-qubit angles.

    Returns:
        np.ndarray: (N, 3) positions.
    """
    alpha = morph_alpha(e, t, front)
    return from_payload(e, sweep_values(e, alpha),
                        morph_dirs(e, t, a_dirs_only, _front_value_alpha(e, t, front)))


def front_reference(e: Encoded, t: float, front, a_dirs_only: bool = False):
    """Noiseless positions of the smooth front (per-vertex distance, no chip).

    Args:
        e: A morph encoding.
        t: Morph parameter in [0, 1].
        front: ``(width, source)``.
        a_dirs_only: Radial: keep A's directions.

    Returns:
        np.ndarray: (N, 3) positions.
    """
    alpha = front_smooth_alpha(e, t, front)
    return from_payload(e, sweep_values(e, alpha, per_value=True), morph_dirs(e, t, a_dirs_only, alpha))


def read_mesh(e: Encoded, p: Params, seed: int | None = None, drr: str = "none") -> np.ndarray:
    """Decode and restore one readout to vertex positions.

    Args:
        e: The encoding.
        p: Readout settings (morph: ``p.morph_t`` picks the frame, ``p.front`` the front).
        seed: Sampler seed; None uses ``p.seed``.
        drr: One of DRR_METHODS.

    Returns:
        np.ndarray: (N, 3) decoded positions.
    """
    raw = decode_raw(e, p, seed)
    if e.mode == "morph":
        return morph_frame(e, raw, p.morph_t, drr, front=p.front)
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


SCALE_UP = 10  # "Why Tessa"'s scaled-up line: this many times the mesh's values


def baseline_sizes(n: int) -> tuple[int, int, int]:
    """Register sizes of the compressing baselines for ``n`` values.

    Args:
        n: Number of values.

    Returns:
        tuple: ``(k, a, m)``: amplitude/FRQI address qubits ``k = ceil(log2 n)``, and
        QCrank's ``a`` address and ``m`` data qubits, chosen for the fewest qubits.
    """
    k = int(np.ceil(np.log2(max(n, 2))))
    a = int(np.argmin([j + np.ceil(n / 2 ** j) for j in range(k + 1)]))  # past k only adds qubits
    return k, a, int(np.ceil(n / 2 ** a))


def scaled_resources(e: Encoded, p: Params, factor: int = SCALE_UP) -> dict | None:
    """What Tessa and the baselines would need for a mesh ``factor`` times larger.

    Counts only what follows from the sizes, with no circuits built: Tessa's graph and
    runs come from :func:`fit_lattice` (or ``e``'s chip) and :func:`batches_needed`, as
    :func:`encode_mesh` would pick them; the baselines' registers from
    :func:`baseline_sizes`.

    Args:
        e: The encoding.
        p: Its settings (layout).
        factor: Size multiple.

    Returns:
        dict | None: ``factor``, ``values``, ``chip_qubits`` and ``rows``, one per
        encoding (as in :func:`resources`) with ``name``, ``qubits``, ``graph``,
        ``runs``, ``configs`` and ``fits``. None for the spectral payload, whose values
        are a fixed number of modes, not one per vertex.
    """
    if e.payload == "spectral":
        return None
    chip = PHOENIX_ROWS * PHOENIX_COLS
    n = e.stats["values"] * factor
    rows_n = len(e.slots) * factor  # vertices
    comps = 2 if e.payload == "spherical" else e.slots.shape[1]
    per = 2 if e.mode == "split" else 1
    if e.device is not None:
        graph, name = e.graph, f"ibm_{e.device}"
    else:
        try:
            graph = fit_lattice(rows_n, comps, p.layout == "mesh", per)
        except ValueError:  # too big for the largest lattice: several runs
            graph = Lattice(MAX_ROWS, MAX_COLS)
        name = f"{graph.rows}x{graph.cols} lattice"
    k, a, m = baseline_sizes(n)
    rows = [
        dict(name="Tessa", qubits=graph.n_qubits, graph=name,
             runs=batches_needed(rows_n, comps, per * graph.capacity),
             configs=2 ** max(g.degree for g in graph.groups), fits=True),
        dict(name="Angle (1 value/qubit)", qubits=n, configs=1),
        dict(name="QCrank (parallel QPIXL)", qubits=a + m, configs=2 ** a),
        dict(name="FRQI", qubits=k + 1, configs=2 ** k),
        dict(name="Amplitude", qubits=k, configs=2 ** k),
    ]
    for row in rows[1:]:
        row.update(graph=None, runs=1, fits=row["qubits"] <= chip)
    return {"factor": factor, "values": n, "chip_qubits": chip, "rows": rows}


def resources(e: Encoded, where=None) -> list[dict]:
    """Cost of ``e``'s values under Tessa and four baselines, transpiled onto a chip.

    Every row encodes generic random data (so no encoding gets credit for this
    mesh) and is transpiled onto the ibm_phoenix snapshot; Tessa on a whole-chip
    graph is transpiled onto its own chip. SWAPs are estimated as the extra two-qubit
    gates over an unrouted transpile, divided by 3.

    Args:
        e: The encoding.
        where: Optional list of capacity-slot arrays, one per Tessa field, holding
            the mesh's values; None derives it from ``e`` (the interference morph
            passes ``[A's and B's slots]``: one field, A and B in separate slots).

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
    if where is None:  # a mesh read in several runs: each run's circuit holds one batch
        first = e.batches == 0
        slots, fields = e.slots[first], e.fields[first]
        where = [slots, slots] if e.mode == "morph" else [slots[fields == f] for f in ((0, 1) if e.pair else (0,))]
    rand = [fill(e.graph, w, rng.random(w.shape)) for w in where]
    mode = get_mode(e.enc.mode)  # split, spherical and a snap readout: pair_full; Bloch reads θ's own circuit
    if isinstance(e.enc, SnapEncoding):
        rand.append(rng.random(e.graph.capacity))  # every constellation point has a phase
    generic, _ = encode(e.graph, *rand, **({"mode": mode.name} if len(rand) > 1 else {}))
    if e.device is None:
        tessa = routed(generic, layout=phoenix_layout(e.graph))
    else:
        tessa = routed(generic, devices.snapshot_backend(e.device), list(range(e.graph.n_qubits)))
    k, a, m = baseline_sizes(n)
    amp = QuantumCircuit(k)
    vec = rng.random(2 ** k)
    amp.append(StatePreparation(vec / np.linalg.norm(vec)), range(k))
    frqi = QuantumCircuit(k + 1)
    frqi.h(range(k))
    frqi.append(UCRYGate(list(rng.random(2 ** k))), [k, *range(k)])
    qcrank = QuantumCircuit(a + m)
    qcrank.h(range(a))
    for d in range(m):
        qcrank.append(UCRYGate(list(rng.random(2 ** a))), [a + d, *range(a)])
    max_deg = max(g.degree for g in e.graph.groups)
    rows = [
        dict(name="Tessa", qubits=len(e.used_qubits), settings=len(get_mode(e.enc.mode).bases) * e.runs,
             configs=2 ** max_deg, **tessa),
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
    if not len(E):  # no faces: nothing to read
        return E, np.zeros(0), np.zeros(0)
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
        qc.add_register(ClassicalRegister(1, "c"))
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
        # one transpile per coupler (its pairs and its reference together), not per circuit
        every = circuits + [pair_circuit(0.0, 0.0)] * len(used)
        on = {}
        for k, lay in enumerate(layouts + used):
            on.setdefault(tuple(lay), []).append(k)
        run = [None] * len(every)
        for lay, ks in on.items():
            tqc = transpile([every[k] for k in ks], be, initial_layout=list(lay), optimization_level=1, seed_transpiler=0)
            for k, qc in zip(ks, tqc):
                run[k] = qc
        ref_of = [k % len(couplers) for k in range(len(circuits))]
    else:
        smp = sampler(PAIR, p, seed)
        run = transpile(circuits + [pair_circuit(0.0, 0.0)], device_of(smp),
                        optimization_level=1, seed_transpiler=0)
        ref_of = [0] * len(circuits)
    result = smp.run(run, shots=p.shots).result()
    K = np.array([1 - 2 * result[k].data.c.bitcount().mean() for k in range(len(run))])  # 2 P(0) - 1
    # Reference pairs (v_i = v_j, exact K = 1) measure each coupler's contrast loss;
    # dividing by it removes the noise's shrink of K, as in the paper's reference circuit.
    K = K[: len(circuits)] / np.maximum(K[len(circuits):][ref_of], 1e-3)
    dv = 2 * np.arccos(np.clip(K, -1, 1)) / np.pi * (2 * half)  # |v_i - v_j| per axis
    measured = np.sqrt((dv.reshape(-1, 3) ** 2).sum(1))
    return E, measured, edge_lengths(verts, E)


if __name__ == "__main__":
    write_quality()
