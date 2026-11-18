"""Cube and UV-sphere meshes as (verts (N, 3), faces) plus OBJ export."""

from pathlib import Path

import numpy as np


def cube(n: int = 3):
    """Surface grid of a [-1, 1] cube, n vertices per edge, shared vertices."""
    t = np.linspace(-1, 1, n)
    index, verts, faces = {}, [], []

    def vid(p):
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
            for i in range(n - 1):
                for j in range(n - 1):
                    faces.append([grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]])
    return np.array(verts), faces


def uv_sphere(segments: int = 8, rings: int = 6):
    """Unit UV sphere: segments * (rings - 1) + 2 vertices, quads plus pole triangles."""
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
    """A UV sphere with smooth bumps: star-shaped, so one radius per vertex describes
    it, but (unlike a sphere) the radii vary smoothly, so there is real data to encode."""
    v, f = uv_sphere(segments, rings)
    az = np.arctan2(v[:, 1], v[:, 0])
    r = 1 + 0.2 * np.cos(3 * az) * (1 - v[:, 2] ** 2) + 0.15 * v[:, 2] ** 3
    return v * r[:, None], f


PRESETS = {
    "blob": {"12x8": lambda: blob(12, 8), "16x10": lambda: blob(16, 10),
             "24x12": lambda: blob(24, 12)},
    "sphere": {"8x6": lambda: uv_sphere(8, 6), "12x8": lambda: uv_sphere(12, 8),
               "16x8": lambda: uv_sphere(16, 8)},
    "cube": {"3/edge": lambda: cube(3), "4/edge": lambda: cube(4), "5/edge": lambda: cube(5)},
}


def edges(faces) -> np.ndarray:
    """Unique undirected mesh edges, shape (E, 2)."""
    pairs = {tuple(sorted((f[i], f[(i + 1) % len(f)]))) for f in faces for i in range(len(f))}
    return np.array(sorted(pairs))


def importance(verts, faces) -> np.ndarray:
    """Per-vertex curvature, scale-free: how far a vertex sits from the mean of its
    neighbours, over its mean edge length. Flat and gently curved regions score
    low; tips, ridges and creases (where noise shows most) score high."""
    verts, E = np.asarray(verts, float), edges(faces)
    n = len(verts)
    deg = np.bincount(E.ravel(), minlength=n).astype(float)
    nsum, elen = np.zeros_like(verts), np.zeros(n)
    length = np.linalg.norm(verts[E[:, 0]] - verts[E[:, 1]], axis=1)
    for a, b in ((E[:, 0], E[:, 1]), (E[:, 1], E[:, 0])):
        np.add.at(nsum, a, verts[b])
        np.add.at(elen, a, length)
    deg = np.maximum(deg, 1)
    return np.linalg.norm(verts - nsum / deg[:, None], axis=1) / np.maximum(elen / deg, 1e-12)


def bfs_order(n_verts: int, faces) -> np.ndarray:
    """Vertices in breadth-first order over mesh edges, so consecutive vertices are
    mesh neighbours (the mesh-aware layout fills each Tessa cell from this order)."""
    adj = [[] for _ in range(n_verts)]
    for a, b in edges(faces):
        adj[a].append(b)
        adj[b].append(a)
    seen, order = np.zeros(n_verts, bool), []
    for start in range(n_verts):  # one BFS per connected component
        if seen[start]:
            continue
        seen[start], queue = True, [start]
        while queue:
            v = queue.pop(0)
            order.append(v)
            for w in sorted(adj[v]):
                if not seen[w]:
                    seen[w] = True
                    queue.append(w)
    return np.array(order)


def subdivided_size(n_verts: int, faces) -> int:
    """Vertex count after :func:`subdivide`, without doing it: one new vertex per
    edge, plus a centre for every face with four or more corners."""
    return n_verts + len(edges(faces)) + sum(len(f) >= 4 for f in faces)


def subdivide(verts, faces):
    """Linear subdivision: every edge gets its midpoint; a triangle becomes four
    triangles, a quad (or n-gon) becomes one quad per corner around its centre. The
    shape is unchanged; there are just more vertices to encode."""
    verts = np.asarray(verts, float)
    E = edges(faces)
    mid = {(int(a), int(b)): len(verts) + k for k, (a, b) in enumerate(E)}
    out_v, out_f = [verts, verts[E].mean(1)], []
    m = lambda a, b: mid[(min(a, b), max(a, b))]  # noqa: E731
    nxt = len(verts) + len(E)
    for f in faces:
        k = len(f)
        if k == 3:
            a, b, c = f
            out_f += [[a, m(a, b), m(c, a)], [b, m(b, c), m(a, b)],
                      [c, m(c, a), m(b, c)], [m(a, b), m(b, c), m(c, a)]]
        else:
            out_v.append(verts[f].mean(0, keepdims=True))
            out_f += [[f[i], m(f[i], f[(i + 1) % k]), nxt, m(f[i - 1], f[i])] for i in range(k)]
            nxt += 1
    return np.concatenate(out_v), out_f


def _normal(p):
    """Newell normal of a polygon (unnormalised)."""
    q = np.roll(p, -1, 0)
    return np.array([((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])).sum(),
                     ((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])).sum(),
                     ((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])).sum()])


def decimate(verts, faces, keep: float = 0.5):
    """Greedy shortest-edge collapse to about ``keep`` of the vertices. Each collapse
    merges an edge's two ends at its midpoint. It is skipped if it would tear the
    surface (the link condition), flip a face, touch an open boundary (eye holes,
    cut edges), or merge corners that are not neighbours in some face. Quads that
    lose a corner become triangles."""
    import heapq

    V = np.asarray(verts, float).copy()
    F = [list(f) for f in faces]
    n = len(V)
    at = [set() for _ in range(n)]  # faces around each vertex
    for k, f in enumerate(F):
        for v in f:
            at[v].add(k)
    E = edges(F)
    open_edges = [e for e in E if sum(1 for k in at[e[0]] & at[e[1]]) == 1]
    boundary = {int(v) for e in open_edges for v in e}

    def nbrs(v):
        return {F[k][(F[k].index(v) + d) % len(F[k])] for k in at[v] for d in (-1, 1)}

    length = lambda a, b: float(np.linalg.norm(V[a] - V[b]))  # noqa: E731
    heap = [(length(a, b), int(a), int(b)) for a, b in E]
    heapq.heapify(heap)
    alive, target = n, max(4, int(round(n * keep)))
    dead = np.zeros(n, bool)
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
        for c in nbrs(a):
            heapq.heappush(heap, (length(a, c), a, c))
    keep_v = np.flatnonzero(~dead)
    new = np.full(n, -1)
    new[keep_v] = np.arange(len(keep_v))
    return V[keep_v], [[int(new[v]) for v in f] for f in F if f]


def load_obj(path):
    """Vertices and faces of an OBJ file (e.g. Blender's Suzanne), centred and scaled
    so the largest coordinate is 1. OBJ is Y-up; the app is Z-up, so (x, y, z) becomes
    (x, -z, y). Texture/normal indices are ignored."""
    verts, faces = [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "v":
            verts.append([float(x) for x in parts[1:4]])
        elif parts and parts[0] == "f":
            idx = [int(p.split("/")[0]) for p in parts[1:]]
            faces.append([i - 1 if i > 0 else len(verts) + i for i in idx])
    v = np.array(verts)[:, [0, 2, 1]] * [1, -1, 1]
    v -= (v.min(0) + v.max(0)) / 2
    return v / np.abs(v).max(), faces


def write_obj(path, verts, faces) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # the app is Z-up, OBJ (and Blender's importer) Y-up: (x, y, z) -> (x, z, -y)
    lines = [f"v {x:.6f} {z:.6f} {-y:.6f}" for x, y, z in verts]
    lines += ["f " + " ".join(str(i + 1) for i in f) for f in faces]
    path.write_text("\n".join(lines) + "\n")
    return path
