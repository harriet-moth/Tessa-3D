"""Meshes as (verts (N, 3), faces): presets, OBJ input/output, topology helpers and decimation.

Faces are lists of polygons, each a list of vertex indices wound outward. Nothing
here calls Tessa; :mod:`tessa_ripple.bridge` uses these helpers to lay a mesh out
on qubits.
"""

import heapq
from collections import deque
from functools import cache
from pathlib import Path

import numpy as np


def cube(n: int = 3):
    """Surface grid of a [-1, 1] cube with shared vertices.

    Args:
        n: vertices per cube edge (>= 2).

    Returns:
        (verts, faces): verts (6n² - 12n + 8, 3) float array; faces a list of quads,
        each a list of 4 vertex indices wound outward.
    """
    t = np.linspace(-1, 1, n)
    index, verts, faces = {}, [], []

    def vid(p):
        """Index of point ``p``, adding it to ``verts`` the first time it is seen.

        Args:
            p: (3,) point; matched after rounding to 9 decimals.

        Returns:
            int vertex index.
        """
        key = tuple(np.round(p, 9))
        if key not in index:
            index[key] = len(verts)
            verts.append(p)
        return index[key]

    for axis in range(3):
        for side in (-1.0, 1.0):
            u_ax, v_ax = [a for a in range(3) if a != axis]
            grid = np.empty((n, n), dtype=int)
            for i, u in enumerate(t):
                for j, v in enumerate(t):
                    p = np.zeros(3)
                    p[axis], p[u_ax], p[v_ax] = side, u, v
                    grid[i, j] = vid(p)
            flip = (side > 0) == (axis == 1)  # (u, v, axis) is right-handed except on axis 1
            for i in range(n - 1):
                for j in range(n - 1):
                    quad = [grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]]
                    faces.append(quad[::-1] if flip else quad)  # every face winds outward
    return np.array(verts), faces


def uv_sphere(segments: int = 8, rings: int = 6):
    """Unit UV sphere: quads plus triangles at the poles.

    Args:
        segments: vertices around each ring (longitude divisions).
        rings: latitude bands; rings - 1 rings of vertices sit between the poles.

    Returns:
        (verts, faces): verts (segments * (rings - 1) + 2, 3) float array; faces a
        list of polygons (lists of vertex indices) wound outward.
    """
    verts = [[0.0, 0.0, 1.0]]
    for r in range(1, rings):
        th = np.pi * r / rings
        for s in range(segments):
            ph = 2 * np.pi * s / segments
            verts.append([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    verts.append([0.0, 0.0, -1.0])
    ring = lambda r, s: 1 + (r - 1) * segments + s % segments  # noqa: E731
    bottom = len(verts) - 1
    faces = [[0, ring(1, s), ring(1, s + 1)] for s in range(segments)]
    for r in range(1, rings - 1):
        faces += [[ring(r, s), ring(r + 1, s), ring(r + 1, s + 1), ring(r, s + 1)]
                  for s in range(segments)]
    faces += [[ring(rings - 1, s + 1), ring(rings - 1, s), bottom] for s in range(segments)]
    return np.array(verts), faces


def blob(segments: int = 12, rings: int = 8):
    """A soft five-lobed cushion, like the Tessa3D landing page's blob.

    A UV sphere pushed out into five uneven lobes round its equator (fading towards
    the poles), flattened to about two thirds of its height, and tipped 50° so its top
    faces the default view. Star-shaped, so one radius per vertex describes it, and (unlike a
    sphere) the radii vary smoothly, so there is real data to encode.

    Args:
        segments: vertices around each ring (five lobes want 30 or more).
        rings: latitude bands.

    Returns:
        (verts, faces) as :func:`uv_sphere`.
    """
    v, f = uv_sphere(segments, rings)
    az = np.arctan2(v[:, 1], v[:, 0])
    z = v[:, 2]
    side = np.sqrt(np.clip(1 - z ** 2, 0, 1))  # 1 at the equator, 0 at the poles
    # five round lobes with softly pinched valleys, unevenly spaced: broad near +1,
    # dipping to -1 only between lobes
    phase = 2.5 * az + 0.22 * np.sin(2 * az + 0.6)
    lobes = 1 - 2 * np.abs(np.sin(phase)) ** 2.4
    r = 1 + (0.17 * lobes + 0.06 * np.cos(2 * az + 1.1)) * side ** 2
    out = v * r[:, None]
    out[:, 2] *= 0.62 * (1 + 0.1 * lobes * side)  # a cushion: flat, puffier over the lobes
    # tip the top towards the default view (from -y, a little above), as in the design
    tilt = np.radians(50)
    y, zz = out[:, 1].copy(), out[:, 2].copy()
    out[:, 1], out[:, 2] = y * np.cos(tilt) - zz * np.sin(tilt), y * np.sin(tilt) + zz * np.cos(tilt)
    return out, f


PRESETS = {
    "blob": {"30x12": lambda: blob(30, 12), "40x16": lambda: blob(40, 16),
             "50x20": lambda: blob(50, 20)},
    "sphere": {"8x6": lambda: uv_sphere(8, 6), "12x8": lambda: uv_sphere(12, 8),
               "16x8": lambda: uv_sphere(16, 8)},
    "cube": {"3/edge": lambda: cube(3), "4/edge": lambda: cube(4), "5/edge": lambda: cube(5)},
}


def edges(faces) -> np.ndarray:
    """Unique undirected mesh edges.

    Args:
        faces: list of polygons, each a list of vertex indices.

    Returns:
        (E, 2) int array of (low, high) vertex pairs, sorted ((0, 2) with no faces).
    """
    pairs = {tuple(sorted((f[i], f[(i + 1) % len(f)]))) for f in faces for i in range(len(f))}
    return np.array(sorted(pairs), int).reshape(-1, 2)


def edge_lengths(verts, E) -> np.ndarray:
    """Length of each edge.

    Args:
        verts: (N, 3) vertex positions.
        E: (E, 2) int array of vertex index pairs, e.g. from :func:`edges`.

    Returns:
        (E,) float array of Euclidean lengths.
    """
    return np.linalg.norm(verts[E[:, 0]] - verts[E[:, 1]], axis=1)


def adjacency(n_verts: int, faces) -> list[list[int]]:
    """Each vertex's neighbours over mesh edges.

    Args:
        n_verts: number of vertices N.
        faces: list of polygons, each a list of vertex indices.

    Returns:
        List of N lists; entry i holds the neighbours of vertex i in ascending order.
    """
    adj = [[] for _ in range(n_verts)]
    for a, b in edges(faces):  # sorted pairs, so every list comes out sorted
        adj[a].append(int(b))
        adj[b].append(int(a))
    return [sorted(a) for a in adj]


def grow_patches(verts, faces, sizes, order=None):
    """Split the vertices into compact patches of the given sizes, plus the leftovers.

    Each patch starts at the first free vertex of the breadth-first order, so patches
    sweep across the mesh and leave no stranded pockets. It then grows over mesh
    edges, always taking the free neighbour nearest its seed, so it becomes a round
    blob rather than an arc; if it runs out of free neighbours it continues from the
    next free vertex. Each patch is returned as a nearest-neighbour path from its
    seed, so consecutive values in a Tessa cell are close on the surface, which helps
    compression.

    Args:
        verts: (N, 3) vertex positions.
        faces: list of polygons, each a list of vertex indices.
        sizes: patch sizes, filled in order; their sum must not exceed N.
        order: (N,) breadth-first vertex order to seed from; None computes
            :func:`bfs_order` (pass it when the caller already has it).

    Returns:
        (patches, rest): patches a list of int arrays, one per size, each a
        nearest-neighbour path; rest the unused vertices, in ``order``.

    Raises:
        StopIteration: if ``sizes`` asks for more vertices than the mesh has.
    """
    verts = np.asarray(verts, float)
    n = len(verts)
    adj = adjacency(n, faces)
    free = np.ones(n, bool)
    order = bfs_order(n, faces) if order is None else order
    sweep = iter(order)
    patches = []
    for m in sizes:
        patch, heap, seed = [], [], None
        while len(patch) < m:
            if not heap:
                seed = next(s for s in sweep if free[s])
                heap = [(0.0, seed)]
            _, v = heapq.heappop(heap)
            if not free[v]:
                continue
            free[v] = False
            patch.append(v)
            for w in adj[v]:
                if free[w]:
                    heapq.heappush(heap, (float(np.linalg.norm(verts[w] - verts[seed])), w))
        path, left = [patch[0]], patch[1:]
        while left:  # nearest-neighbour walk through the patch
            j = int(np.argmin(np.linalg.norm(verts[left] - verts[path[-1]], axis=1)))
            path.append(left.pop(j))
        patches.append(np.array(path, int))
    return patches, order[free[order]]


def bfs_order(n_verts: int, faces) -> np.ndarray:
    """Vertices in breadth-first order over mesh edges.

    Consecutive vertices are mesh neighbours, so the mesh-aware layout fills each
    Tessa cell from this order. Each connected component is walked in turn, starting
    from its lowest vertex index.

    Args:
        n_verts: number of vertices N.
        faces: list of polygons, each a list of vertex indices.

    Returns:
        (N,) int array, a permutation of range(N).
    """
    adj = adjacency(n_verts, faces)
    seen, order = np.zeros(n_verts, bool), []
    for start in range(n_verts):  # one BFS per connected component
        if seen[start]:
            continue
        seen[start], queue = True, deque([start])
        while queue:
            v = queue.popleft()
            order.append(v)
            for w in adj[v]:
                if not seen[w]:
                    seen[w] = True
                    queue.append(w)
    return np.array(order)


def _normal(p):
    """Newell normal of a polygon (unnormalised).

    Args:
        p: (k, 3) corner positions in winding order.

    Returns:
        (3,) normal vector; its length is twice the polygon's area.
    """
    q = np.roll(p, -1, 0)
    return np.array([((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])).sum(),
                     ((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])).sum(),
                     ((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])).sum()])


def decimate(verts, faces, keep: float = 0.5, return_log: bool = False):
    """Greedy shortest-edge collapse to about ``keep`` of the vertices.

    Each collapse merges an edge's two ends at its midpoint. It is skipped if it would
    tear the surface (the link condition), flip a face, touch an open boundary (eye
    holes, cut edges), or merge corners that are not neighbours in some face. Quads
    that lose a corner become triangles.

    Args:
        verts: (N, 3) vertex positions (not modified).
        faces: list of polygons, each a list of vertex indices (not modified).
        keep: target fraction of vertices to keep (at least 4 stay).
        return_log: also return the ordered collapses, so :func:`replay_decimation`
            can apply them to another mesh with the same topology.

    Returns:
        (verts, faces), or (verts, faces, log) with ``return_log``: verts (M, 3), faces
        reindexed to them, log a list of (a, b) original-index pairs, b merged into a.
        Fewer than ``keep`` may go if no allowed collapse remains.
    """
    V = np.asarray(verts, float).copy()
    F = [list(f) for f in faces]
    n = len(V)
    at = _faces_at(n, F)
    E = edges(F)
    open_edges = [e for e in E if sum(1 for k in at[e[0]] & at[e[1]]) == 1]
    boundary = {int(v) for e in open_edges for v in e}

    def nbrs(v):
        """Vertices sharing a face edge with ``v``.

        Args:
            v: vertex index.

        Returns:
            Set of neighbouring vertex indices in the current faces.
        """
        return {F[k][(F[k].index(v) + d) % len(F[k])] for k in at[v] for d in (-1, 1)}

    length = lambda a, b: float(np.linalg.norm(V[a] - V[b]))  # noqa: E731
    heap = [(length(a, b), int(a), int(b)) for a, b in E]
    heapq.heapify(heap)
    alive, target = n, max(4, int(round(n * keep)))
    dead, log = np.zeros(n, bool), []
    while heap and alive > target:
        d, a, b = heapq.heappop(heap)
        if dead[a] or dead[b] or b not in nbrs(a):
            continue
        if abs(d - length(a, b)) > 1e-12:  # stale: re-queue at its current length
            heapq.heappush(heap, (length(a, b), a, b))
            continue
        if a in boundary or b in boundary:
            continue
        shared = at[a] & at[b]
        if any(abs(F[k].index(a) - F[k].index(b)) not in (1, len(F[k]) - 1) for k in shared):
            continue  # a and b are opposite corners of a quad
        if len(nbrs(a) & nbrs(b)) != sum(len(F[k]) == 3 for k in shared):
            continue  # link condition: collapsing would make the surface non-manifold
        p = (V[a] + V[b]) / 2
        flips = False
        for k in (at[a] | at[b]) - shared:  # faces that survive must not turn over
            before = V[F[k]]
            after = np.array([p if v in (a, b) else V[v] for v in F[k]])
            if _normal(before) @ _normal(after) <= 0:
                flips = True
                break
        if flips:
            continue
        V[a], dead[b], alive = p, True, alive - 1
        log.append((a, b))
        _merge(F, at, a, b)
        for c in nbrs(a):
            heapq.heappush(heap, (length(a, c), a, c))
    out = _compact(V, F, dead)
    return (*out, log) if return_log else out


def _faces_at(n_verts: int, F) -> list[set[int]]:
    """The faces around each vertex.

    Args:
        n_verts: number of vertices N.
        F: list of polygons, each a list of vertex indices.

    Returns:
        List of N sets of face indices into ``F``.
    """
    at = [set() for _ in range(n_verts)]
    for k, f in enumerate(F):
        for v in f:
            at[v].add(k)
    return at


def _merge(F, at, a, b):
    """Rewrite the faces around ``b`` onto ``a`` (one edge collapse), in place.

    Faces left with fewer than 3 corners become empty lists.

    Args:
        F: list of polygons (mutated).
        at: faces around each vertex, as :func:`_faces_at` (mutated).
        a: vertex that survives.
        b: vertex merged into ``a``.
    """
    for k in list(at[b]):
        f = [a if v == b else v for v in F[k]]
        f = [v for i, v in enumerate(f) if v != f[i - 1]]  # drop the merged duplicate
        if len(f) < 3:
            for v in F[k]:
                at[v].discard(k)
            F[k] = []
        else:
            F[k] = f
            at[a].add(k)
    at[b] = set()


def _compact(V, F, dead):
    """Drop dead vertices and empty faces, reindexing the rest.

    Args:
        V: (N, 3) vertex positions.
        F: list of polygons, empty lists for removed faces.
        dead: (N,) bool mask of removed vertices.

    Returns:
        (verts, faces): verts (M, 3) of the live vertices; faces reindexed to them.
    """
    keep_v = np.flatnonzero(~dead)
    new = np.full(len(V), -1)
    new[keep_v] = np.arange(len(keep_v))
    return V[keep_v], [[int(new[v]) for v in f] for f in F if f]


def replay_decimation(verts, faces, log):
    """Apply a :func:`decimate` collapse log to another mesh with the same topology.

    The same pairs merge, each at the midpoint of *this* mesh's two vertices. On the
    mesh the log came from, it reproduces :func:`decimate` exactly.

    Args:
        verts: (N, 3) vertex positions (not modified).
        faces: list of polygons with the same topology as the logged mesh.
        log: list of (a, b) merges from ``decimate(..., return_log=True)``.

    Returns:
        (verts, faces) after the collapses, as :func:`decimate`.
    """
    V = np.asarray(verts, float).copy()
    F = [list(f) for f in faces]
    at = _faces_at(len(V), F)
    dead = np.zeros(len(V), bool)
    for a, b in log:
        V[a] = (V[a] + V[b]) / 2
        dead[b] = True
        _merge(F, at, a, b)
    return _compact(V, F, dead)


@cache
def load_decimated(path, times: int = 0):
    """An OBJ decimated ``times`` times (each pass keeps about half the vertices).

    Cached, and built on the ``times - 1`` result, so each pass runs once per file;
    the returned vertex array is read-only, since every caller shares it.

    Args:
        path: OBJ file path (str or Path; the cache keys on it as given).
        times: number of :func:`decimate` passes; 0 is the file as loaded.

    Returns:
        (verts, faces, logs): read-only (M, 3) vertices, faces, and a tuple of one
        collapse log per pass, for :func:`replay_decimation`.
    """
    if times == 0:
        v, f = load_obj(path)
        v.setflags(write=False)
        return v, f, ()
    v, f, logs = load_decimated(path, times - 1)
    v, f, log = decimate(v, f, return_log=True)
    v.setflags(write=False)
    return v, f, (*logs, log)


def load_obj(path):
    """Vertices and faces of an OBJ file (e.g. Blender's Suzanne), normalised.

    Centred on the bounding-box centre and scaled so the largest coordinate is 1.
    OBJ is Y-up and the app Z-up, so (x, y, z) becomes (x, -z, y). Texture and
    normal indices are ignored; negative (relative) face indices are resolved.
    Any text encoding is accepted (only numbers are read: UTF-8, else Latin-1),
    as are ``\\`` line continuations. Faces with fewer than three corners are skipped.

    Args:
        path: OBJ file path.

    Returns:
        (verts, faces): verts (N, 3) float array; faces a list of polygons with
        0-based vertex indices.

    Raises:
        ValueError: The file has no vertices, an unreadable number, or a face corner
            that names a missing vertex (the message names the file and line).
    """
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:  # e.g. a Latin-1 comment or object name; numbers are ASCII anyway
        text = raw.decode("latin-1")
    text = text.replace("\\\r\n", " ").replace("\\\n", " ")  # line continuations
    name = Path(path).name
    verts, faces = [], []
    for n, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        try:
            if parts and parts[0] == "v":
                if len(parts) < 4:
                    raise ValueError("a vertex needs x, y and z")
                verts.append([float(x) for x in parts[1:4]])
            elif parts and parts[0] == "f" and len(parts) >= 4:
                idx = [int(p.split("/")[0]) for p in parts[1:]]
                face = [i - 1 if i > 0 else len(verts) + i for i in idx]
                if min(face) < 0 or 0 in idx:
                    raise ValueError("a face corner names a vertex before the first")
                faces.append(face)
        except ValueError as err:
            raise ValueError(f"{name}, line {n}: {err} ({line.strip()[:40]!r})") from None
    if not verts:
        raise ValueError(f"{name} has no vertices ('v' lines)")
    if faces and max(max(f) for f in faces) >= len(verts):
        raise ValueError(f"{name}: a face names vertex {max(max(f) for f in faces) + 1}, "
                         f"but the file has only {len(verts)}")
    v = np.array(verts)[:, [0, 2, 1]] * [1, -1, 1]
    v -= (v.min(0) + v.max(0)) / 2
    return v / (np.abs(v).max() or 1.0), faces


def write_obj(path, verts, faces) -> Path:
    """Write a mesh as an OBJ file, converting the app's Z-up to OBJ's Y-up.

    (x, y, z) is written as (x, z, -y), the inverse of :func:`load_obj`'s turn.
    Parent folders are created as needed.

    Args:
        path: output file path.
        verts: (N, 3) vertex positions.
        faces: list of polygons, each a list of 0-based vertex indices.

    Returns:
        The written Path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # the app is Z-up, OBJ (and Blender's importer) Y-up: (x, y, z) -> (x, z, -y)
    lines = [f"v {x:.6f} {z:.6f} {-y:.6f}" for x, y, z in verts]
    lines += ["f " + " ".join(str(i + 1) for i in f) for f in faces]
    path.write_text("\n".join(lines) + "\n")
    return path
