"""Morph: a second mesh B carried in the phase of the qubits that hold mesh A.

Classical geometry only (no Tessa calls): align B to A, find one B point per A
vertex, measure how much of B those points cover, and the surface distances that
drive the morph front. bridge.encode_mesh does the encoding, with ``mode="morph"``.
"""

from pathlib import Path

import numpy as np

from .mesh import edge_lengths, edges, load_decimated, replay_decimation

def align(b_verts, a_verts) -> np.ndarray:
    """B moved onto A: centroid on A's, uniformly scaled to A's RMS radius.

    No rotation is fitted: B keeps its own orientation.

    Args:
        b_verts: (M, 3) vertices of B.
        a_verts: (N, 3) vertices of A.

    Returns:
        (M, 3) aligned B vertices.
    """
    a, b = np.asarray(a_verts, float), np.asarray(b_verts, float)
    ca, cb = a.mean(0), b.mean(0)
    ra = np.sqrt(((a - ca) ** 2).sum(1).mean())
    rb = np.sqrt(((b - cb) ** 2).sum(1).mean()) or 1.0
    return ca + (b - cb) * (ra / rb)


def triangles(faces) -> np.ndarray:
    """Fan-triangulate polygons.

    Args:
        faces: list of polygons, each a list of vertex indices.

    Returns:
        (T, 3) int array of triangle vertex indices.
    """
    return np.array([[f[0], f[i], f[i + 1]] for f in faces for i in range(1, len(f) - 1)], int)


def closest_on_triangles(p, a, b, c) -> np.ndarray:
    """Closest point on triangle (a, b, c) to p, vectorised.

    Ericson, Real-Time Collision Detection, 5.1.5.

    Args:
        p: (..., 3) query points.
        a, b, c: (..., 3) triangle corners, broadcastable with ``p``.

    Returns:
        (..., 3) closest points on the triangles.
    """
    ab, ac, ap = b - a, c - a, p - a
    dot = lambda u, v: (u * v).sum(-1)  # noqa: E731
    d1, d2 = dot(ab, ap), dot(ac, ap)
    bp = p - b
    d3, d4 = dot(ab, bp), dot(ac, bp)
    cp = p - c
    d5, d6 = dot(ab, cp), dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        v_in, w_in = vb / denom, vc / denom
        t_ab = d1 / (d1 - d3)
        t_ac = d2 / (d2 - d6)
        t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    out = a + v_in[..., None] * ab + w_in[..., None] * ac  # inside the face
    # Ericson tests A, B, AB, C, AC, BC in turn and returns the first hit; here later
    # overrides win, so apply them in reverse
    regions = [
        ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), b + t_bc[..., None] * (c - b)),  # edge BC
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + t_ac[..., None] * ac),                  # edge AC
        ((d6 >= 0) & (d5 <= d6), c),                                                    # vertex C
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + t_ab[..., None] * ab),                  # edge AB
        ((d3 >= 0) & (d4 <= d3), b),                                                    # vertex B
        ((d1 <= 0) & (d2 <= 0), a),                                                     # vertex A
    ]
    for mask, q in regions:
        out = np.where(mask[..., None], q, out)
    bad = ~np.isfinite(out).all(-1)  # zero-area triangle: the closest point on its edges
    if bad.any():
        p, a, b, c = np.broadcast_arrays(p, a, b, c)
        sides = np.stack([_closest_on_segments(p, u, v) for u, v in ((a, b), (b, c), (c, a))])
        best = np.linalg.norm(sides - p, axis=-1).argmin(0)
        out = np.where(bad[..., None], np.take_along_axis(sides, best[None, ..., None], 0)[0], out)
    return out


def _closest_on_segments(p, a, b) -> np.ndarray:
    """Closest point on segment ab to p, vectorised (a point if a == b).

    Args:
        p: (..., 3) query points.
        a, b: (..., 3) segment ends, same shape as ``p``.

    Returns:
        (..., 3) closest points on the segments.
    """
    ab = b - a
    den = (ab * ab).sum(-1)
    t = np.clip(((p - a) * ab).sum(-1) / np.where(den > 0, den, 1.0), 0.0, 1.0)
    return a + t[..., None] * ab


def closest_points(points, verts, faces, k: int = 16):
    """For each point, the closest point on a mesh surface.

    Only the ``k`` triangles whose centroids are nearest each point are searched.

    Args:
        points: (P, 3) query points.
        verts: (N, 3) mesh vertices.
        faces: list of polygons (fan-triangulated here).
        k: candidate triangles per point (capped at the triangle count).

    Returns:
        (closest, tri): closest (P, 3) surface points; tri (P,) index of the
        triangle each lies on, into :func:`triangles` of ``faces``.
    """
    from scipy.spatial import cKDTree

    points, verts = np.asarray(points, float), np.asarray(verts, float)
    tri = triangles(faces)
    corners = verts[tri]  # (T, 3, 3)
    k = min(k, len(tri))
    _, near = cKDTree(corners.mean(1)).query(points, k=k)
    near = near.reshape(len(points), k)
    a, b, c = (corners[near][:, :, i] for i in range(3))
    q = closest_on_triangles(points[:, None], a, b, c)  # (P, k, 3)
    best = np.linalg.norm(q - points[:, None], axis=2).argmin(1)
    rows = np.arange(len(points))
    return q[rows, best], near[rows, best]


def coverage(matched, b_verts, b_faces, radius: float) -> float:
    """Share of B's surface area within ``radius`` of some matched point.

    Low when closest-point sampling piles A's vertices onto one part of a very
    different B.

    Args:
        matched: (P, 3) matched points on B.
        b_verts: (M, 3) B vertices.
        b_faces: B's polygons.
        radius: distance from a triangle centroid to its nearest matched point that
            counts as covered.

    Returns:
        Fraction in [0, 1] of B's triangle area counted as covered.
    """
    from scipy.spatial import cKDTree

    tri = triangles(b_faces)
    corners = np.asarray(b_verts, float)[tri]
    area = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1) / 2
    d, _ = cKDTree(matched).query(corners.mean(1))
    return float(area[d <= radius].sum() / (area.sum() or 1.0))


def mean_edge(verts, faces) -> float:
    """Mean edge length of a mesh.

    Args:
        verts: (N, 3) vertex positions.
        faces: list of polygons, each a list of vertex indices.

    Returns:
        Mean length over the unique edges.
    """
    return float(edge_lengths(verts, edges(faces)).mean())


def correspond(a_verts, a_faces, b_path, shared_topology: bool = False, a_logs=(),
               a_full_count: int | None = None) -> dict:
    """One point of B (aligned to A) per vertex of A.

    Shared topology: B's vertex i is A's vertex i, so the counts must match. When A
    was decimated (``a_logs``), B is loaded at full size and the same collapses are
    replayed on it, since decimating B on its own would pick different edges.
    Otherwise: the closest point on B's full-resolution surface to each A vertex.

    Args:
        a_verts: (N, 3) vertices of A (as encoded, possibly decimated).
        a_faces: A's polygons.
        b_path: OBJ path of mesh B.
        shared_topology: pair by vertex index instead of closest point.
        a_logs: A's collapse logs, one per decimation pass, from
            :func:`~tessa_ripple.mesh.load_decimated`.
        a_full_count: A's vertex count before decimation (None: ``len(a_verts)``).

    Returns:
        Dict with ``target`` (N, 3) matched B points, ``method`` (str),
        ``distance`` (mean |B - A|), ``coverage`` (share of B's area near a matched
        point; 1.0 for shared topology), and B's own aligned ``verts``/``faces``.

    Raises:
        ValueError: shared topology with differing vertex counts or faces.
    """
    a_verts = np.asarray(a_verts, float)
    b_verts, b_faces, _ = load_decimated(b_path)
    full = len(a_verts) if a_full_count is None else a_full_count
    if shared_topology:
        if len(b_verts) != full:
            raise ValueError(f"Shared topology: A has {full} vertices, B has {len(b_verts)}. "
                             "Untick Shared topology to sample by closest point.")
        for log in a_logs:
            b_verts, b_faces = replay_decimation(b_verts, b_faces, log)
        if [list(f) for f in b_faces] != [list(f) for f in a_faces]:
            # same count, other faces: vertex i of B is not vertex i of A (e.g. two exports
            # with their own vertex order), so pairing by index would scramble the shape
            raise ValueError(f"Shared topology: {Path(b_path).name} has A's vertex count but "
                             "different faces, so its vertices are not A's. Untick Shared "
                             "topology to sample by closest point.")
        b_verts = target = align(b_verts, a_verts)
        method, cov = "by index" + (f", replayed ×{len(a_logs)}" if a_logs else ""), 1.0
    else:
        if not len(triangles(b_faces)):
            raise ValueError(f"{Path(b_path).name} has no faces: closest-point matching needs B's "
                             "surface. Tick Shared topology if B has A's vertex count.")
        b_verts = align(b_verts, a_verts)
        target, _ = closest_points(a_verts, b_verts, b_faces)
        method = "closest point"
        cov = coverage(target, b_verts, b_faces, mean_edge(a_verts, a_faces))
    return {"target": target, "method": method, "coverage": cov,
            "distance": float(np.linalg.norm(target - a_verts, axis=1).mean()),
            "verts": b_verts, "faces": b_faces}


def geodesic(verts, faces, source: int) -> np.ndarray:
    """Surface distance of every vertex from ``source``, normalised to [0, 1].

    Dijkstra (``scipy.sparse.csgraph.dijkstra``) over the mesh edges, weighted by
    their lengths. Divided by the largest finite distance; vertices in another
    connected part (unreachable) get 1.

    Args:
        verts: (N, 3) vertex positions.
        faces: list of polygons, each a list of vertex indices.
        source: index of the source vertex.

    Returns:
        (N,) distances: 0 at ``source``, 1 at the farthest reachable vertex.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    verts = np.asarray(verts, float)
    E = edges(faces)
    n = len(verts)
    w = edge_lengths(verts, E)
    graph = coo_matrix((w, (E[:, 0], E[:, 1])), shape=(n, n)).tocsr()
    d = dijkstra(graph, directed=False, indices=int(source))
    finite = np.isfinite(d)
    top = d[finite].max() if finite.any() else 0.0
    return np.where(finite, d / (top or 1.0), 1.0)


def front_alpha(d, t: float, width: float = 0.15):
    """Sweep angle of a morph front at t, per entry of ``d``.

    α = (π/2) · clip((t·(1 + w) − d) / w, 0, 1): a ramp of width ``w`` (in
    normalised distance) that leaves the source at t = 0 and passes the farthest
    point (d = 1) at t = 1. So α = 0 everywhere at t = 0 (exactly A) and π/2
    everywhere at t = 1 (exactly B), whatever ``d`` holds.

    Args:
        d: Distances in [0, 1], any shape (per vertex, or per (batch, data qubit)).
        t: Morph parameter in [0, 1].
        width: Softness w > 0 of the front, as a share of the normalised distance.

    Returns:
        np.ndarray: Angles in [0, π/2], same shape as ``d``.

    Raises:
        ValueError: If ``width`` is not positive.
    """
    if width <= 0:
        raise ValueError(f"front width must be positive, got {width}")
    d = np.asarray(d, float)
    ramp = np.clip((t * (1 + width) - d) / width, 0.0, 1.0)
    # (1 + w - 1) / w can round just below 1: pin t = 1 so B's end is exactly pi/2
    return (np.pi / 2) * (np.ones_like(d) if t >= 1 else ramp)


def slerp(d0, d1, t) -> np.ndarray:
    """Per-row spherical interpolation between unit vectors.

    (Anti)parallel rows fall back to a renormalised lerp.

    Args:
        d0: (N, 3) unit vectors at t = 0.
        d1: (N, 3) unit vectors at t = 1.
        t: interpolation weight in [0, 1]: a scalar, or (N, 1) one per row.

    Returns:
        (N, 3) unit vectors.
    """
    d0, d1 = np.asarray(d0, float), np.asarray(d1, float)
    cos = np.clip((d0 * d1).sum(-1, keepdims=True), -1, 1)
    om = np.arccos(cos)
    s = np.sin(om)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (np.sin((1 - t) * om) * d0 + np.sin(t * om) * d1) / s
    lin = (1 - t) * d0 + t * d1  # (anti)parallel rows: plain lerp, renormalised
    out = np.where(s > 1e-6, out, lin)
    n = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.where(n > 0, n, 1)
