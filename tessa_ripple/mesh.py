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


PRESETS = {
    "cube": {"3/edge": lambda: cube(3), "4/edge": lambda: cube(4), "5/edge": lambda: cube(5)},
    "sphere": {"8x6": lambda: uv_sphere(8, 6), "12x8": lambda: uv_sphere(12, 8),
               "16x8": lambda: uv_sphere(16, 8)},
}


def write_obj(path, verts, faces) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += ["f " + " ".join(str(i + 1) for i in f) for f in faces]
    path.write_text("\n".join(lines) + "\n")
    return path
