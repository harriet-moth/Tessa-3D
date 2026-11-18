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
