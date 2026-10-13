from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def mask_laplacian_eigenchannels(
    mask: np.ndarray,
    modes: int,
    *,
    tol: float = 1e-4,
    maxiter: int | None = None,
) -> np.ndarray:
    """Return low-frequency graph Laplacian channels for a binary Cartesian mask.

    The active pixels form a four-neighbor graph. The graph Laplacian has a
    natural Neumann-like behavior at mask boundaries because edges crossing the
    inactive boundary are omitted. The first constant mode is discarded; on
    disconnected masks, subsequent zero modes encode component structure.
    """
    if modes < 0:
        raise ValueError("modes must be non-negative")
    mask_bool = np.asarray(mask > 0.5, dtype=bool)
    if mask_bool.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {mask_bool.shape}")
    h, w = mask_bool.shape
    out = np.zeros((modes, h, w), dtype=np.float32)
    if modes == 0:
        return out

    active = np.argwhere(mask_bool)
    n_active = int(active.shape[0])
    if n_active <= 1:
        return out

    index = -np.ones((h, w), dtype=np.int64)
    index[mask_bool] = np.arange(n_active, dtype=np.int64)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    degree = np.zeros(n_active, dtype=np.float64)
    for y, x in active:
        i = int(index[y, x])
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            yy = int(y + dy)
            xx = int(x + dx)
            if 0 <= yy < h and 0 <= xx < w and mask_bool[yy, xx]:
                j = int(index[yy, xx])
                rows.append(i)
                cols.append(j)
                vals.append(-1.0)
                degree[i] += 1.0
    rows.extend(range(n_active))
    cols.extend(range(n_active))
    vals.extend(degree.tolist())
    lap = sp.coo_matrix((vals, (rows, cols)), shape=(n_active, n_active), dtype=np.float64).tocsr()

    requested = min(modes + 1, n_active)
    try:
        if requested >= n_active:
            eigvals, eigvecs = np.linalg.eigh(lap.toarray())
            order = np.argsort(eigvals)[:requested]
            eigvecs = eigvecs[:, order]
        else:
            eigvals, eigvecs = spla.eigsh(lap, k=requested, which="SM", tol=tol, maxiter=maxiter)
            order = np.argsort(eigvals)
            eigvecs = eigvecs[:, order]
    except spla.ArpackNoConvergence as exc:
        if exc.eigenvectors is None or exc.eigenvectors.shape[1] < 2:
            raise
        eigvecs = exc.eigenvectors

    available = max(0, eigvecs.shape[1] - 1)
    count = min(modes, available)
    coords = active.astype(np.float64)
    y_center = (h - 1) * 0.5
    x_center = (w - 1) * 0.5
    canonical_axis = (coords[:, 1] - x_center) + 0.37 * (coords[:, 0] - y_center)
    for k in range(count):
        vec = np.asarray(eigvecs[:, k + 1], dtype=np.float64)
        vec = vec - vec.mean()
        rms = float(np.sqrt(np.mean(vec * vec)))
        if rms > 1e-12:
            vec = vec / rms
        if float(np.dot(vec, canonical_axis)) < 0.0:
            vec = -vec
        channel = np.zeros((h, w), dtype=np.float32)
        channel[mask_bool] = vec.astype(np.float32)
        out[k] = channel
    return out
