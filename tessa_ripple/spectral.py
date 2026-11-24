"""Spectral payload: a mesh as coefficients of its own Laplacian eigenvectors.

Classical geometry only (no Tessa calls). ``bridge.encode_mesh`` encodes the
coefficients with ``payload="spectral"``.

A mesh's low-frequency "harmonics" (eigenvectors of the cotangent Laplacian, as in
spectral mesh compression) carry most of its shape in a few values. Encoding the
coefficients of the first k of them, instead of xyz per vertex:

- needs 3k values instead of 3N, so a smaller circuit, always one run;
- turns readout noise into smooth swells of the whole surface (a sum of low modes)
  instead of per-vertex grain.

The basis is mesh A's. Morphs put B on A's vertices (``morph.correspond``), so both
meshes share it and the endpoints are A and B truncated to k modes.

Conventions: L is the positive semi-definite cotangent Laplacian, M the lumped
(barycentric) mass, and the basis solves L φ = λ M φ with Φᵀ M Φ = I. The centre
stays classical, as in every payload: it is A's M-weighted centroid, so A has no
constant part and the constant mode (λ = 0) is dropped. B's matched points generally
do have one (its centroid shift against A: 3 % of the bounding-box diagonal for
Suzanne → sphere), so a morph keeps the constant as an extra encoded mode
(``constant=True``), in which A's coefficient is exactly 0.
"""

from collections import OrderedDict

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from .morph import triangles

DENSE_MAX = 1500  # up to this many vertices the dense generalised eigensolver is fastest
_CACHE_SIZE = 8
_cache: "OrderedDict[tuple, tuple]" = OrderedDict()


def cotan_laplacian(verts, faces):
    """Cotangent Laplacian and lumped mass of a mesh (polygons fan-triangulated).

    Edge (i, j) has weight (cot α + cot β) / 2, α and β the angles opposite it.
    Degenerate (zero-area) triangles add nothing. A vertex in no triangle gets a tiny
    mass so M stays positive definite (it becomes its own zero mode).

    Args:
        verts: (N, 3) vertex positions.
        faces: Polygon vertex-index lists.

    Returns:
        tuple: ``(L, mass)``: (N, N) sparse CSR positive semi-definite Laplacian (rows
        sum to 0) and (N,) lumped vertex areas.
    """
    v = np.asarray(verts, float)
    n = len(v)
    t = triangles(faces)
    if not len(t):
        return sparse.csr_matrix((n, n)), np.ones(n)
    i, j, k = t.T
    cross = np.cross(v[j] - v[i], v[k] - v[i])
    double_area = np.linalg.norm(cross, axis=1)
    scale = float(np.ptp(v, axis=0).max() or 1.0)
    ok = double_area > 1e-12 * scale ** 2

    def cot(a, b, c):
        """Cotangent of the angle at corner a of triangle (a, b, c), 0 where degenerate."""
        u, w = v[b] - v[a], v[c] - v[a]
        return np.where(ok, (u * w).sum(1) / np.where(ok, double_area, 1.0), 0.0)

    rows = np.concatenate([j, k, i])  # the edge opposite each corner
    cols = np.concatenate([k, i, j])
    w = np.concatenate([cot(i, j, k), cot(j, k, i), cot(k, i, j)]) / 2
    W = sparse.coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    W = W + W.T
    L = sparse.diags(np.asarray(W.sum(1)).ravel()) - W
    mass = np.bincount(t.ravel(), np.repeat(double_area / 6, 3), minlength=n)
    floor = 1e-9 * (mass.mean() or 1.0)
    return L.tocsr(), np.where(mass > floor, mass, floor)


def _solve(L, mass, m: int):
    """The m smallest eigenpairs of L φ = λ M φ, M-orthonormal.

    Dense ``scipy.linalg.eigh`` for small meshes; shift-invert ``eigsh`` otherwise.

    Args:
        L: (N, N) sparse Laplacian.
        mass: (N,) lumped mass.
        m: Number of eigenpairs (at most N).

    Returns:
        tuple: ``(lam, phi)``: (m,) ascending eigenvalues and (N, m) eigenvectors.
    """
    n = L.shape[0]
    if n <= DENSE_MAX or m >= n - 1:
        lam, phi = eigh(L.toarray(), np.diag(mass), subset_by_index=[0, m - 1])
    else:
        lam, phi = eigsh(L, m, sparse.diags(mass), sigma=-1e-8, which="LM")
        order = np.argsort(lam)
        lam, phi = lam[order], phi[:, order]
        phi /= np.sqrt((phi ** 2 * mass[:, None]).sum(0))  # M-normalise (ARPACK's already are, near enough)
    return lam, phi


def _drop_constant(lam, phi, mass):
    """Remove the constant function from the zero-eigenvalue block.

    For a connected mesh this is mode 0. A mesh in several pieces has one zero mode
    per piece in an arbitrary basis; that block is rotated so its first vector is the
    constant, which is then dropped, leaving the pieces' relative offsets.

    Args:
        lam: (m,) ascending eigenvalues.
        phi: (N, m) M-orthonormal eigenvectors.
        mass: (N,) lumped mass.

    Returns:
        tuple: ``(lam, phi)`` with m - 1 modes.
    """
    tol = 1e-8 * max(float(np.abs(lam).max()), 1.0)
    z = int(max(1, (lam <= tol).sum()))
    const = np.ones(len(mass)) / np.sqrt(mass.sum())
    a = phi[:, :z].T @ (mass * const)  # the constant in the zero block's basis
    q, _ = np.linalg.qr(np.column_stack([a, np.eye(z)]))  # first column ∥ a
    zero = phi[:, :z] @ q[:, :z]
    return np.r_[lam[:z][1:], lam[z:]], np.column_stack([zero[:, 1:], phi[:, z:]])


def basis(verts, faces, k: int, constant: bool = False):
    """A's first k non-constant Laplacian eigenvectors (cached per mesh and k).

    Args:
        verts: (N, 3) vertex positions.
        faces: Polygon vertex-index lists.
        k: Number of non-constant modes, 1 <= k <= N - 1.
        constant: Put the constant mode first (k + 1 modes): for a morph, where B's
            centroid shift has to be encoded too.

    Returns:
        tuple: ``(phi, lam, mass)``: (N, k [+ 1]) M-orthonormal modes, low frequency
        first; their eigenvalues; (N,) lumped mass.

    Raises:
        ValueError: If k is outside [1, N - 1].
    """
    v = np.ascontiguousarray(verts, float)
    n = len(v)
    if not 1 <= k <= n - 1:
        raise ValueError(f"spectral k must be between 1 and {n - 1} (vertices - 1), got {k}")
    key = (v.shape, hash(v.tobytes()), hash(tuple(map(tuple, faces))))
    hit = _cache.get(key)
    if hit is None or hit[0].shape[1] < k:  # a cached larger basis serves any smaller k
        L, mass = cotan_laplacian(v, faces)
        lam, phi = _drop_constant(*_solve(L, mass, k + 1), mass)
        hit = (phi, lam, mass)
        _cache[key] = hit
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    _cache.move_to_end(key)
    phi, lam, mass = hit
    phi, lam = phi[:, :k], lam[:k]
    if constant:
        phi = np.column_stack([np.full(n, 1 / np.sqrt(mass.sum())), phi])
        lam = np.r_[0.0, lam]
    return phi, lam, mass


def centroid(verts, mass) -> np.ndarray:
    """M-weighted centroid: the centre whose offset the dropped constant mode would hold.

    Args:
        verts: (N, 3) positions.
        mass: (N,) lumped mass.

    Returns:
        np.ndarray: (3,) centre.
    """
    return (np.asarray(mass, float)[:, None] * verts).sum(0) / mass.sum()


def coefficients(V, centre, phi, mass) -> np.ndarray:
    """Spectral coefficients C = Φᵀ M (V - centre).

    Args:
        V: (N, 3) positions on A's vertices.
        centre: (3,) centre.
        phi: (N, k) modes.
        mass: (N,) lumped mass.

    Returns:
        np.ndarray: (k, 3) coefficients, one row per mode.
    """
    return phi.T @ (np.asarray(mass, float)[:, None] * (np.asarray(V, float) - centre))


def reconstruct(C, centre, phi) -> np.ndarray:
    """Positions from coefficients: centre + Φ C.

    Args:
        C: (k, 3) coefficients.
        centre: (3,) centre.
        phi: (N, k) modes (only the first ``len(C)`` are used).

    Returns:
        np.ndarray: (N, 3) positions.
    """
    C = np.asarray(C, float)
    return centre + phi[:, :len(C)] @ C


def diagonal(verts) -> float:
    """Length of the bounding box's diagonal.

    Args:
        verts: (N, 3) positions.

    Returns:
        float: The diagonal (1 for a single point).
    """
    return float(np.linalg.norm(np.ptp(verts, axis=0)) or 1.0)


def truncation_errors(V, centre, phi, mass, diag: float | None = None) -> np.ndarray:
    """Truncation error for every k from 1 to the basis size.

    Error = RMS over vertices of |V - (centre + Φ_k C_k)|, over the bounding-box diagonal.

    Args:
        V: (N, 3) positions on A's vertices.
        centre: (3,) centre.
        phi: (N, K) modes.
        mass: (N,) lumped mass.
        diag: Length to divide by; None uses ``V``'s bounding-box diagonal.

    Returns:
        np.ndarray: (K,) error with the first 1, 2, ..., K modes.
    """
    V = np.asarray(V, float)
    C = coefficients(V, centre, phi, mass)
    resid = V - centre
    out = np.empty(phi.shape[1])
    for j in range(phi.shape[1]):  # peel one mode at a time: O(N K) per step, cheap at these sizes
        resid = resid - np.outer(phi[:, j], C[j])
        out[j] = np.sqrt((resid ** 2).sum(1).mean())
    return out / (diagonal(V) if diag is None else diag)


def choose_k(a, b, faces, tol: float, k_max: int) -> int:
    """Smallest k whose truncation error is at most ``tol`` for A and for B.

    With B the basis keeps the constant mode (``basis(..., constant=True)``), so B's
    centroid shift is not counted as truncation.

    Args:
        a: (N, 3) A's vertices (the basis is A's).
        b: (N, 3) B on A's vertices, or None (A alone).
        faces: A's polygons.
        tol: Largest truncation error, as a fraction of the bounding-box diagonal.
        k_max: Cap on the non-constant modes (e.g. what one run holds); also capped
            at N - 1.

    Returns:
        int: k in [1, min(k_max, N - 1)]; the cap if no k meets ``tol``.

    Raises:
        ValueError: If the cap is below 1.
    """
    k_max = int(min(k_max, len(a) - 1))
    lead = 0 if b is None else 1
    phi, _, mass = basis(a, faces, k_max, constant=b is not None)
    centre = centroid(a, mass)
    err = truncation_errors(a, centre, phi, mass)[lead:]
    if b is not None:
        err = np.maximum(err, truncation_errors(b, centre, phi, mass)[lead:])
    hit = np.flatnonzero(err <= tol)
    return int(hit[0] + 1) if len(hit) else k_max


def band_scales(Ca, Cb=None, headroom: float = 0.1) -> np.ndarray:
    """One scale per mode, so every mode's coefficients fill [-1, 1].

    Energy piles into the low modes, so one shared range would be set by the largest
    coefficient (about √N times the coordinate scale) and leave the high modes a
    sliver of it. Scaling each mode by its own largest |coefficient| (over xyz, A and
    B) keeps the per-vertex noise near xyz's level.

    Args:
        Ca: (k, 3) A's coefficients.
        Cb: (k, 3) B's coefficients, or None.
        headroom: Fractional padding.

    Returns:
        np.ndarray: (k,) positive scales s; C / s[:, None] lies in [-1, 1].
    """
    top = np.abs(Ca).max(1)
    if Cb is not None:
        top = np.maximum(top, np.abs(Cb).max(1))
    return np.where(top > 0, top, 1.0) * (1 + headroom)


def high_share(E, phi_full, mass, k: int) -> float:
    """Share of an error field's energy above mode k (its "grain").

    Args:
        E: (N, 3) error field on A's vertices.
        phi_full: (N, K) modes, K > k (the more, the closer to the whole spectrum).
        mass: (N,) lumped mass.
        k: Modes counted as low.

    Returns:
        float: 1 - |low part|²_M / |E - mean|²_M, in [0, 1].
    """
    E = np.asarray(E, float)
    E = E - centroid(E, mass)  # the constant (a shift of the whole mesh) is neither low nor high
    total = float((mass[:, None] * E ** 2).sum())
    if total <= 0:
        return 0.0
    C = coefficients(E, 0.0, phi_full[:, :k], mass)
    return float(max(0.0, 1 - (C ** 2).sum() / total))
