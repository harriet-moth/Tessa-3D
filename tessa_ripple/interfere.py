"""Interference morph: A and B on the two values of an address qubit, read by interference.

The basis-sweep morph (``bridge``, mode "morph") reads each slot along
cos(α) Z + sin(α) X. Noiseless it is a cross-fade in cosine space, so the only
quantum part a viewer sees is the noise. This module puts the morph in the
*address register* instead, after the paper's Appendix D (reading coherences
between cells by turning address qubits before measuring):

- Some address qubits are **selectors**. Every data qubit has at most one
  selector among its address neighbours (exactly one where the graph packs
  perfectly: 6x6, 10x12, Phoenix, Fez); the selector's data neighbours form its
  **star**. Each slot pair (selector bit 0, selector bit 1) holds A's value and
  B's value, so the morph keeps half of Tessa's capacity.
- Encoding is plain ``tessa.encode`` (single mode). With the selector in |+⟩ a
  star holds (|0⟩|A-patch⟩ + |1⟩|B-patch⟩)/√2: a superposition of the two meshes'
  patches (a cat state when the patches differ).
- A frame at β turns the selectors by RY(-β) and measures everything in Z.
  Tessa's own decode of a selector-0 slot then reads the data qubit after
  projecting the star onto cos(β/2)|A⟩ + sin(β/2)|B⟩: exactly A at β = 0, exactly
  B at β = π, and in between a *coherent* interpolation of the qubit states.

What only coherence produces: for an isolated data qubit the frame follows the
great circle between the two Bloch vectors (the half-angles add as vectors),
while a dephased selector gives the classical mixture, cos θ = c² cos θ_A +
s² cos θ_B, which is Tessa's own cross-fade (``ideal_frames``). In a star the
other members' overlaps ⟨A_j|B_j⟩ weight the interference term, so how far a
vertex moves at mid-morph depends on its patch's neighbours: an entanglement
effect, not a per-vertex blend.

Simulation reads each star as one cluster circuit (``tessa.cell_circuit``: the
star's data qubits and all their address qubits, the neighbouring data qubits as
Tessa's traced halo), since turning a shared address qubit breaks Tessa's
one-cell-at-a-time readout. Every Tessa piece is reused: ``encode``, the
cluster extraction, ``codec._run`` (transpile, trace the halo, sample),
``decode_cells`` per data qubit, ``ideal_frames`` for the incoherent reference,
and ``restore_field`` (via ``bridge.restore``) for DRR.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
from qiskit.quantum_info import Statevector
from tessa import DataGroup, Lattice, cell_circuit, cell_halo, encode, prepare_field
from tessa.codec import _check_traceable, _measured, _normalise, _run
from tessa.core.decode import _bit_table, decode_cells
from tessa.core.mode import get_mode
from tessa.hardware import devices
from tessa.hardware.pipeline import ideal_frames

from . import bridge
from .bridge import (DEVICES, Encoded, Params, batch_seed, batches_needed, cells, chip_quality, fill,
                     fit_lattice, morph_clamp, morph_dirs, morph_range, place, radial, restore,
                     rotated_quality, slot_groups, spectral_payload, split_batches, submesh, two_qubit)
from .mesh import bfs_order, grow_patches
from .morph import correspond


@dataclass(frozen=True)
class Stars:
    """Which address qubits select between A and B, and the half graph left for values.

    Attributes:
        graph: The full Tessa ``Lattice`` or ``TessaGraph``.
        selectors: Selector address qubits, one per star.
        star: (groups,) the star (index into ``selectors``) of each data qubit, -1 if
            it has no selector neighbour (it then holds nothing).
        bit: (groups,) position of the selector in each data qubit's
            ``address_qubits`` (its bit of the local address), -1 if none.
        half: Duck-typed Tessa graph of the covered data qubits with the selector
            removed from each address register: what placement fills.
        full_of: (len(half.groups),) the ``graph.groups`` index of each half group.
    """
    graph: object
    selectors: tuple
    star: np.ndarray
    bit: np.ndarray
    half: object
    full_of: np.ndarray

    @property
    def capacity(self) -> int:
        """Values (A, B pairs) one run holds.

        Returns:
            int: ``half.capacity``, half of the covered data qubits' slots.
        """
        return self.half.capacity


class HalfGraph:
    """The value-carrying part of a graph once the selectors are taken out.

    Duck-types what ``bridge.place``, ``bridge.cells`` and Tessa's ``slot_order``
    read: ``groups`` (each with ``n_values``) and ``capacity``.

    Args:
        groups: ``DataGroup`` per covered data qubit, selector removed.
        n_qubits: The full graph's qubit count.
    """

    def __init__(self, groups, n_qubits: int):
        self.groups = list(groups)
        self.n_qubits = n_qubits

    @property
    def capacity(self) -> int:
        """Total slots.

        Returns:
            int: Sum of ``n_values`` over the groups.
        """
        return sum(g.n_values for g in self.groups)


def choose_selectors(graph) -> Stars:
    """Pick selector address qubits so each data qubit has at most one, keeping most values.

    An integer program (``scipy.optimize.milp``): choose address qubits x_a in {0, 1}
    with at most one chosen neighbour per data qubit (stars never overlap, so each
    star is read on its own), maximising the values kept, Σ_a x_a Σ_{d ~ a} 2^(m_d - 1).
    Some graphs pack perfectly (6x6, 10x12, Phoenix, Fez): every data qubit is
    covered and the morph keeps exactly half of Tessa's capacity. Others (3x3, 5x5,
    5x7, ...) leave a few data qubits uncovered; those hold padding only.

    Args:
        graph: Tessa ``Lattice`` or ``TessaGraph``.

    Returns:
        Stars: The selectors, each data qubit's star and selector bit, and the half graph.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    groups = graph.groups
    addr = sorted({a for g in groups for a in g.address_qubits})
    col = {a: i for i, a in enumerate(addr)}
    gain = np.zeros(len(addr))
    adj = np.zeros((len(groups), len(addr)))
    for k, g in enumerate(groups):
        for a in g.address_qubits:
            adj[k, col[a]] = 1
            gain[col[a]] += g.n_values // 2
    res = milp(-gain, constraints=LinearConstraint(adj, 0, 1), integrality=np.ones(len(addr)),
               bounds=Bounds(0, 1))
    chosen = [addr[i] for i in np.flatnonzero(np.round(res.x) > 0)]
    star, bit = np.full(len(groups), -1), np.full(len(groups), -1)
    half, full_of = [], []
    for k, g in enumerate(groups):
        for s, a in enumerate(chosen):
            if a in g.address_qubits:
                star[k], bit[k] = s, g.address_qubits.index(a)
                rest = tuple(q for q in g.address_qubits if q != a)
                sites = tuple(x for q, x in zip(g.address_qubits, g.address_sites) if q != a)
                half.append(DataGroup(site=g.site, data_qubit=g.data_qubit, address_qubits=rest,
                                      address_sites=sites))
                full_of.append(k)
    return Stars(graph, tuple(chosen), star, bit, HalfGraph(half, graph.n_qubits), np.array(full_of, int))


def pair_slots(stars: Stars, half_slots):
    """Full-graph slots of A and B for half-graph slots.

    Half slot x of a data qubit is its local address with the selector bit removed;
    A sits at the address with that bit 0, B with it 1.

    Args:
        stars: From :func:`choose_selectors`.
        half_slots: Int array of half-graph slots, any shape.

    Returns:
        tuple: ``(a_slots, b_slots)``, same shape, full-graph capacity slots.
    """
    half_slots = np.asarray(half_slots, int)
    h = slot_groups(stars.half, half_slots)
    local = half_slots - cells(stars.half)[h, 0]
    full = stars.full_of[h]
    j = stars.bit[full]
    low = local & ((1 << j) - 1)
    a = cells(stars.graph)[full, 0] + (low | ((local >> j) << (j + 1)))
    return a, a + (1 << j)


@dataclass(frozen=True)
class Interfered:
    """A morph encoded for the interference readout.

    Attributes:
        e: A ``bridge.Encoded`` (mode "morph") whose ``slots`` are A's full-graph
            slots, so ``bridge.restore`` and ``morph_dirs`` work unchanged.
        stars: The selectors (:func:`choose_selectors`).
        b_slots: (N, comps) B's full-graph slots.
        a01: (N*comps,) A's values normalised to [0, 1] (as ``encode`` sees them).
        b01: (N*comps,) B's, likewise.
        clamp: θ clamp of both meshes.
        ideals: :func:`ideal`'s frames by t, filled as they are asked for.
    """
    e: Encoded
    stars: Stars
    b_slots: np.ndarray
    a01: np.ndarray
    b01: np.ndarray
    clamp: float
    ideals: dict = field(default_factory=dict, compare=False, repr=False)


def spectral_positions(p: Params) -> int:
    """Values (A/B pairs) one interference run holds: the graph's half capacity.

    Args:
        p: Settings (backend).

    Returns:
        int: ``choose_selectors(graph).capacity`` of the chip, or of the largest lattice.
    """
    graph = devices.graph(p.backend) if p.backend in DEVICES else Lattice(bridge.MAX_ROWS, bridge.MAX_COLS)
    return choose_selectors(graph).capacity


def _payload(verts, faces, p: Params, target):
    """A's and B's prepared values and their shared range, as ``bridge.encode_mesh`` (morph).

    Spectral: one row per mode (the constant first), k capped so every A/B pair fits
    one run's half capacity.

    Args:
        verts: (N, 3) A's vertices.
        faces: A's polygons.
        p: Settings (payload, quant_bits, headroom, target, shared_topology, backend,
            spectral settings).
        target: :func:`morph.correspond` dict or None.

    Returns:
        dict: ``vals``, ``vals_b`` (rows, comps), ``prep``, ``prep_b``, ``rng``,
        ``centre``, ``dirs``, ``dirs_b``, ``match``, ``target`` and, for spectral,
        ``spectral`` (:func:`bridge.spectral_payload`'s dict), else None.

    Raises:
        ValueError: For the spherical payload or a morph without B.
    """
    if p.payload not in ("radial", "xyz", "spectral"):
        raise ValueError("The interference morph takes the radial, xyz or spectral payload")
    if target is None:
        if p.target is None:
            raise ValueError("Morph: upload a B mesh")
        target = correspond(verts, faces, p.target, p.shared_topology)
    tgt = np.asarray(target["target"], float)
    centre = verts.mean(0)
    dirs = dirs_b = sp = None
    if p.payload == "spectral":
        sp = spectral_payload(verts, faces, p, spectral_positions(p), tgt)
        vals, vals_b, centre = sp["vals"], sp["vals_b"], sp["centre"]
    elif p.payload == "xyz":
        vals, vals_b = verts - centre, tgt - centre
    else:
        vals, dirs = radial(verts, centre)
        vals_b, dirs_b = radial(tgt, centre)
    flat, prep = prepare_field(vals.ravel(), quant_bits=p.quant_bits)
    flat_b, prep_b = prepare_field(vals_b.ravel(), quant_bits=p.quant_bits)
    rng = morph_range(flat, flat_b, p.payload, p.headroom)  # as encode_mesh's morph
    n = len(vals)
    return dict(vals=flat.reshape(n, -1), vals_b=flat_b.reshape(n, -1), prep=prep, prep_b=prep_b,
                rng=rng, centre=centre, dirs=dirs, dirs_b=dirs_b, match=target, target=tgt,
                spectral=sp)


def encode_interfere(verts, faces, p: Params, target=None) -> Interfered:
    """Encode a morph A → B for the interference readout.

    Placement, batching and quality ranking follow ``bridge.encode_mesh`` on the
    half graph (one A/B pair per position); each run is one ``tessa.encode``
    (single mode) of the full graph.

    Args:
        verts: (N, 3) A's vertices.
        faces: A's polygons.
        p: Settings (payload radial, xyz or spectral, layout, backend, placement,
            clamp, compression, target).
        target: :func:`morph.correspond` dict, or None to build it from ``p.target``.

    Returns:
        Interfered: The encoding.
    """
    verts = np.asarray(verts, float)
    d = _payload(verts, faces, p, target)
    sp = d["spectral"]
    n, comps = d["vals"].shape  # n: vertices, or spectral modes
    quality = None
    if p.backend in DEVICES:
        graph, name = devices.graph(p.backend), f"ibm_{p.backend} (whole chip)"
        quality = chip_quality(p.backend)
    else:
        graph = _fit(n, comps, p.layout == "mesh" or sp is not None)
        name = f"{graph.rows}x{graph.cols} lattice"
    stars = choose_selectors(graph)
    half = stars.half
    hq = None if quality is None or p.placement != "quality" else quality[stars.full_of]
    k_runs = batches_needed(n, comps, half.capacity)
    groups = split_batches(verts, faces, k_runs) if k_runs > 1 else [np.arange(n)]
    clamp = morph_clamp(p)
    a_slots, b_slots = np.empty((n, comps), int), np.empty((n, comps), int)
    batch, parts, skip = np.zeros(n, int), [], 0
    for k, idx in enumerate(groups):
        v_k, f_k = (verts, faces) if k_runs == 1 else (verts[idx], submesh(faces, idx))
        mesh = p.layout == "mesh"
        if sp is not None:  # modes low first, whole modes per data qubit (one run by construction)
            order, grow = np.arange(len(idx)), None
        else:
            order = bfs_order(len(idx), f_k) if mesh else None
            grow = (lambda sizes, v=v_k, f=f_k, o=order: grow_patches(v, f, sizes, o)) if mesh else None
        q_k = None if hq is None else (rotated_quality(hq, skip) if skip else hq)
        s_k, _ = place(half, len(idx), comps, order, q_k, 1, grow)
        if q_k is not None:  # later batches start on the next-best data qubits
            sizes = np.diff(cells(half), axis=1)[np.argsort(-q_k, kind="stable"), 0]
            skip += int(np.searchsorted(np.cumsum(sizes), len(idx) * comps)) + 1
        a_k, b_k = pair_slots(stars, s_k)
        a_slots[idx], b_slots[idx], batch[idx] = a_k, b_k, k
        # every half slot gets a pair; empty pairs hold A's padding twice (overlap 1)
        pad_a = fill(half, s_k, d["vals"][idx])
        pad_b = pad_a.copy()
        pad_b[s_k] = d["vals_b"][idx]
        all_a, all_b = pair_slots(stars, np.arange(half.capacity))
        full = np.full(graph.capacity, np.nan)
        full[all_a], full[all_b] = pad_a, pad_b
        full[np.isnan(full)] = float(np.nanmean(full))  # data qubits outside every star
        parts.append(encode(graph, full, value_range=d["rng"], clamp=clamp, compression=p.compression))
    circuit, enc = parts[0]
    e = Encoded(verts, faces, graph, name, circuit, enc, d["prep"], a_slots, p.payload, d["centre"],
                d["dirs"], two_qubit(circuit), quality, None, "morph",
                parts=tuple(parts) if k_runs > 1 else (), batch=batch if k_runs > 1 else None,
                target=d["target"], dirs_b=d["dirs_b"], prep_b=d["prep_b"], match=d["match"],
                **({} if sp is None else dict(scales=sp["scales"], basis=sp["basis"], spectral=sp["info"])))
    a01, b01 = (_normalise(prep.reference, d["rng"])[0] for prep in (d["prep"], d["prep_b"]))
    return Interfered(e, stars, b_slots, a01, b01, clamp)


def _fit(n: int, comps: int, mesh: bool):
    """Smallest lattice whose half graph holds the mesh (``bridge.fit_lattice`` at twice the size).

    Args:
        n: Vertices.
        comps: Values per vertex.
        mesh: Mesh layout.

    Returns:
        Lattice: The lattice; the largest one if even that needs batches.
    """
    for m in range(n, 4 * n + 1, max(1, n // 8)):  # half capacity is about half, not exactly
        try:
            lat = fit_lattice(m, comps, mesh)
        except ValueError:
            return Lattice(bridge.MAX_ROWS, bridge.MAX_COLS)
        if choose_selectors(lat).capacity >= n * comps:
            return lat
    return Lattice(bridge.MAX_ROWS, bridge.MAX_COLS)


def beta_of(t: float) -> float:
    """Selector angle of morph parameter t: β = t·π (A at 0, B at π).

    Args:
        t: Morph parameter in [0, 1].

    Returns:
        float: β in radians.
    """
    return float(t) * np.pi


def _clusters(ie: Interfered, part: int):
    """The stars holding mesh values in one run, with their cluster qubits.

    Args:
        ie: The encoding.
        part: Run index.

    Returns:
        list[tuple]: ``(selector, members, qubits, halo)`` per star: member indices
        into ``graph.groups``, the cluster's qubits (data qubits first, then address
        qubits) and the traced halo (neighbouring data qubits).
    """
    g, st = ie.e.graph, ie.stars
    used = np.unique(ie.e.groups_of[ie.e.batches == part])
    out = []
    for s in np.unique(st.star[used]):
        members = np.flatnonzero(st.star == s)
        data = [g.groups[k].data_qubit for k in members]
        addr = sorted({a for k in members for a in g.groups[k].address_qubits})
        halo = sorted(set().union(*(cell_halo(g, g.groups[k]) for k in members)) - set(data))  # Tessa's diamond
        out.append((st.selectors[s], members, data + addr, halo))
    return out


def _cluster_circuit(circuit, qubits, halo, selector, beta: float, flip: bool = False):
    """One star's part of the run's circuit, its selector turned by RY(-β).

    ``tessa.cell_circuit`` keeps the gates inside the given qubits; a DataGroup
    whose "address" register is the rest of the cluster says which ones.

    Args:
        circuit: The run's full encoding circuit.
        qubits: Cluster qubits (data first).
        halo: Halo data qubits, kept last.
        selector: The star's selector qubit.
        beta: Selector angle.
        flip: Apply Z to the selector first (half of a dephasing control: summing
            the counts with and without it removes the A/B coherence).

    Returns:
        QuantumCircuit: The cluster circuit, register ``qubits + halo``.
    """
    group = DataGroup(site=-1, data_qubit=qubits[0], address_qubits=tuple(qubits[1:]),
                      address_sites=tuple(qubits[1:]))
    qc = cell_circuit(circuit, group, tuple(halo))
    if flip:
        qc.z(qubits.index(selector))
    if beta:
        qc.ry(-beta, qubits.index(selector))
    return qc


def _decode_star(ie: Interfered, members, register, counts):
    """Tessa's ``decode_cells`` for every member of a star, from the cluster's counts.

    Args:
        ie: The encoding.
        members: Indices into ``graph.groups``.
        register: The cluster circuit's qubits, in register order.
        counts: Counts (or probabilities) over the register.

    Returns:
        dict: ``{site: values in [0, 1]}`` per member.
    """
    pos = {q: i for i, q in enumerate(register)}
    groups = [ie.e.graph.groups[k] for k in members]
    local = [DataGroup(site=g.site, data_qubit=pos[g.data_qubit],
                       address_qubits=tuple(pos[a] for a in g.address_qubits),
                       address_sites=g.address_sites) for g in groups]
    return decode_cells({"z": counts}, local, len(register), "single", ie.e.enc.clamps)


def _selector_zero(register, selector, counts) -> tuple[float, float]:
    """Shots in which the selector read 0, and all shots.

    Args:
        register: Cluster qubits in register order.
        selector: The selector qubit.
        counts: Counts dict (Qiskit bit order).

    Returns:
        tuple: ``(zero, total)`` weights.
    """
    bits, w = _bit_table(counts, len(register))  # Tessa's parse: bits[:, q] is register[q]
    return float(w[bits[:, register.index(selector)] == 0].sum()), float(w.sum())


def _merge(counts_list) -> dict:
    """Sum counts dicts.

    Args:
        counts_list: Counts dicts over the same register.

    Returns:
        dict: Their sum.
    """
    out = {}
    for c in counts_list:
        for k, n in c.items():
            out[k] = out.get(k, 0) + n
    return out


def _gather(ie: Interfered, packed_by_run) -> np.ndarray:
    """Each value's selector-0 (A-slot) reading, in A's units.

    Args:
        ie: The encoding.
        packed_by_run: One ``{site: values in [0, 1]}`` per run.

    Returns:
        np.ndarray: (N*comps,) values.
    """
    g, e = ie.e.graph, ie.e
    lo, hi = e.enc.ranges[0]
    out = np.empty(e.slots.shape)
    for k, packed in enumerate(packed_by_run):  # a cell no star read stays 0
        full = g.unpack({grp.site: packed.get(grp.site, np.zeros(grp.n_values)) for grp in g.groups})
        m = e.batches == k
        out[m] = full[e.slots[m]]
    return (lo + (hi - lo) * out).ravel()


def read(ie: Interfered, p: Params, t: float, seed: int | None = None, dephase: bool = False):
    """One noisy (or shot-noise) readout of the frame at t.

    Every star is one cluster job, all submitted together through Tessa's
    ``codec._run`` (transpile on the backend, trace the halo, sample).

    Args:
        ie: The encoding.
        p: Readout settings (backend, shots, noise_strength).
        t: Morph parameter.
        seed: Sampler seed; None uses ``p.seed``.
        dephase: Control: half the shots with Z on every selector before it turns,
            summed, so the selectors are fully dephased and the frame should read
            :func:`incoherent` whatever the chip's noise.

    Returns:
        tuple: ``(raw, kept)``: (N*comps,) values in A's units, and the share of
        shots whose selector read 0 (the shots a frame is decoded from).
    """
    seed = p.seed if seed is None else seed
    _check_traceable(bridge.sampler(ie.e.graph, p, seed))  # the halo is traced, as Tessa's decode checks
    beta = beta_of(t)
    packed_by_run, zero, total = [], 0.0, 0.0
    for k in range(ie.e.runs):
        circuit = (ie.e.parts[k] if ie.e.parts else (ie.e.circuit, ie.e.enc))[0]
        stars = _clusters(ie, k)
        flips = (False, True) if dephase else (False,)
        jobs = []
        for sel, _, qubits, halo in stars:
            for flip in flips:
                qc = _cluster_circuit(circuit, qubits, halo, sel, beta, flip)
                jobs.append((_measured(qc, [], "z"), qubits + halo, tuple(halo)))
        shots = p.shots // len(flips)
        every = _run(bridge.sampler(ie.e.graph, p, batch_seed(seed, k)), jobs, shots, 1)
        counts = [_merge(every[i:i + len(flips)]) for i in range(0, len(every), len(flips))]
        packed = {}
        for (sel, members, qubits, halo), c in zip(stars, counts):
            packed.update(_decode_star(ie, members, qubits + halo, c))
            z, n = _selector_zero(qubits + halo, sel, c)
            zero, total = zero + z, total + n
        packed_by_run.append(packed)
    return _gather(ie, packed_by_run), zero / (total or 1.0)


def run_frames(ie: Interfered, p: Params, ts, progress=None):
    """One :func:`read` per frame, frame i at ``ts[i]`` with seed ``p.seed + i``.

    The interference counterpart of ``animate.run_frames``: every frame is its own
    readout (no shared reference, since each frame is one setting of the selectors).

    Args:
        ie: The encoding.
        p: Readout settings; ``p.seed`` is frame 0's seed.
        ts: Morph t of each frame.
        progress: Optional callback ``progress(i, n)`` before frame i.

    Returns:
        tuple: ``(raw, kept)``: (n, N*comps) values in A's units, and (n,) the share
        of shots each frame was decoded from (selector read 0).
    """
    raw, kept = [], []
    for i, t in enumerate(ts):
        if progress:
            progress(i, len(ts))
        r, k = read(ie, p, float(t), p.seed + i)
        raw.append(r)
        kept.append(k)
    return np.array(raw), np.array(kept)


def ideal(ie: Interfered, t: float) -> np.ndarray:
    """The exact noiseless frame at t (infinite shots), entanglement within stars included.

    Each cluster's statevector (``qiskit.quantum_info.Statevector``) gives exact
    probabilities, decoded by Tessa's ``decode_cells`` like counts. A frame is computed
    once per t and kept on ``ie`` (every redraw of a reference asks again).

    Args:
        ie: The encoding.
        t: Morph parameter.

    Returns:
        np.ndarray: (N*comps,) values in A's units (read-only: it is shared).
    """
    t = float(t)
    if t not in ie.ideals:
        ie.ideals[t] = out = _ideal(ie, t)
        out.setflags(write=False)
    return ie.ideals[t]


def _ideal(ie: Interfered, t: float) -> np.ndarray:
    """:func:`ideal`, computed.

    Args:
        ie: The encoding.
        t: Morph parameter.

    Returns:
        np.ndarray: (N*comps,) values in A's units.
    """
    beta = beta_of(t)
    packed_by_run = []
    for k in range(ie.e.runs):
        circuit = (ie.e.parts[k] if ie.e.parts else (ie.e.circuit, ie.e.enc))[0]
        packed = {}
        for sel, members, qubits, _ in _clusters(ie, k):
            qc = _cluster_circuit(circuit, qubits, (), sel, beta)
            probs = Statevector(qc).probabilities_dict()
            packed.update(_decode_star(ie, members, qubits, probs))
        packed_by_run.append(packed)
    return _gather(ie, packed_by_run)


def incoherent(ie: Interfered, t: float) -> np.ndarray:
    """The frame with every selector dephased: the classical mixture of A and B.

    The selector-0 slot then holds cos²(β/2) ρ_A + sin²(β/2) ρ_B, so
    cos θ = w cos θ_A + (1 - w) cos θ_B with w = cos²(β/2): Tessa's cross-fade
    (``ideal_frames``) at tan α = tan²(β/2).

    Args:
        ie: The encoding.
        t: Morph parameter.

    Returns:
        np.ndarray: (N*comps,) values in A's units.
    """
    beta = beta_of(t)
    alpha = np.arctan2(np.sin(beta / 2) ** 2, np.cos(beta / 2) ** 2)
    plan = SimpleNamespace(extra={"clamps": (ie.clamp, ie.clamp), "alphas": [alpha],
                                  "inputs": (ie.a01, ie.b01)})
    lo, hi = ie.e.enc.ranges[0]
    return lo + (hi - lo) * ideal_frames(plan)[0]


def pure(ie: Interfered, t: float) -> np.ndarray:
    """The frame if every data qubit were alone in its star: the coherent superposition.

    cos(β/2)|ψ_A⟩ + sin(β/2)|ψ_B⟩ normalised. Both are real, so the half-angles add
    as vectors: θ/2 = atan2(c sin(θ_A/2) + s sin(θ_B/2), c cos(θ_A/2) + s cos(θ_B/2)),
    the great circle between the two Bloch vectors.

    Args:
        ie: The encoding.
        t: Morph parameter.

    Returns:
        np.ndarray: (N*comps,) values in A's units.
    """
    beta, k = beta_of(t), ie.clamp
    c, s = np.cos(beta / 2), np.sin(beta / 2)
    single = get_mode("single")  # Tessa's value <-> RY angle map, with the clamp
    ha, hb = (single.angles(x, k) / 2 for x in (ie.a01, ie.b01))
    theta = 2 * np.arctan2(c * np.sin(ha) + s * np.sin(hb), c * np.cos(ha) + s * np.cos(hb))
    lo, hi = ie.e.enc.ranges[0]
    return lo + (hi - lo) * single.values_from_angles(theta, k)


def coherence(frame, coherent, mixed) -> float:
    """How much of the interference survives: the least-squares λ in
    frame ≈ mixed + λ (coherent - mixed).

    1 is the noiseless frame, 0 a fully dephased selector.

    Args:
        frame: (M,) a read frame.
        coherent: (M,) :func:`ideal` at the same t.
        mixed: (M,) :func:`incoherent` at the same t.

    Returns:
        float: λ (nan when coherent and mixed agree, e.g. at t = 0 and 1, where only
        rounding separates them and λ would be noise divided by ~1e-30).
    """
    coherent, mixed = np.asarray(coherent, float), np.asarray(mixed, float)
    d = coherent - mixed
    den = float(d @ d)
    scale = float(np.abs(coherent).max(initial=0.0)) or 1.0
    if den <= d.size * (1e-9 * scale) ** 2:  # no interference term to measure
        return float("nan")
    return float((np.asarray(frame, float) - mixed) @ d / den)


def frame(ie: Interfered, raw, t: float, drr: str = "none", a_dirs_only: bool = False) -> np.ndarray:
    """Positions of a frame: DRR and payload back to vertices (``bridge.restore``).

    Radial directions turn with ``bridge.morph_dirs`` (slerp at t), as the sweep morph;
    xyz and spectral read every part of the shape from the state. DRR stretches towards
    the noiseless frame at t (:func:`ideal`), not A's input.

    Args:
        ie: The encoding.
        raw: (N*comps,) values from :func:`read`, :func:`ideal`, etc.
        t: Morph parameter.
        drr: One of ``bridge.DRR_METHODS``.
        a_dirs_only: Radial: keep A's directions.

    Returns:
        np.ndarray: (N, 3) positions.
    """
    reference = None if drr == "none" else ideal(ie, t)
    return restore(ie.e, raw, drr, morph_dirs(ie.e, t, a_dirs_only), reference)


def cost(ie: Interfered) -> dict:
    """Two-qubit gates and depth of one frame, summed over runs (the RY on selectors adds none).

    Args:
        ie: The encoding.

    Returns:
        dict: ``twoq``, ``depth``, ``runs``, ``stars`` (in the first run) and
        ``capacity`` (A/B pairs per run).
    """
    qcs = [c for c, _ in (ie.e.parts or ((ie.e.circuit, ie.e.enc),))]
    return {"twoq": sum(two_qubit(q) for q in qcs), "depth": sum(q.depth() for q in qcs),
            "runs": ie.e.runs, "stars": len(_clusters(ie, 0)), "capacity": ie.stars.capacity}
