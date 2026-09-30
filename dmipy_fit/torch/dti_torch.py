"""The diffusion tensor fitted to every voxel at once in PyTorch: the log-linear model

    log S = log S0 - b g' D g

solved by least squares for the seven unknowns (the six tensor elements and ``log S0``) with one design matrix
shared by all voxels. ``'OLS'`` is one product with the design's pseudo-inverse; ``'WLS'`` (the default, dipy's
``TensorModel`` default) reweights every measurement by the ordinary fit's predicted signal and solves the batch of
7 x 7 normal equations. The eigen-decomposition gives the eigenvalues (clipped from below at ``1e-6 / b_max``, dipy's
floor), the eigenvectors, FA and MD.

The measurements used are the PGSE ones (OGSE measurements carry another b-value scaling) with ``b <= b_max``;
``b_max=None`` uses them all. Units are SI: b in s/m^2 in the scheme, the tensor, the eigenvalues and MD in m^2/s.
The design is built in float64 and cast to the device dtype (float32 by default); every product runs with TF32 off
(:func:`~dmipy_fit.torch.csd_tournier_torch.full_precision`). The setup is cached by value: the b-values, the
directions, the OGSE flags, ``b_max``, the method, the dtype and the device.
"""
import functools
import hashlib
from typing import NamedTuple

import numpy as np

from .csd_tournier_torch import full_precision

__all__ = ['DTIFit', 'build_dti_fitter_torch']

EIGH_CHUNK = 16384        # cuSOLVER's batched 3 x 3 eigh refuses batches above 31,650 (L40S, torch 2.11 + CUDA 12.8)


class DTIFit(NamedTuple):
    """The tensor fit of ``N`` voxels, torch tensors on the fit's device."""
    S0: object           # (N,)        the fitted non-weighted signal
    tensor: object       # (N, 3, 3)   m^2/s
    evals: object        # (N, 3)      descending, m^2/s
    evecs: object        # (N, 3, 3)   evecs[:, :, k] is the eigenvector of evals[:, k]
    fa: object           # (N,)
    md: object           # (N,)        m^2/s

    @property
    def principal_direction(self):
        """``(N, 3)``: the eigenvector of the largest eigenvalue."""
        return self.evecs[:, :, 0]


def _design(bvalues, directions, b_max):
    """``(X (n, 7), sel (n,))``: the log-linear design in s/mm^2 over the selected measurements, columns
    ``[Dxx, Dyy, Dzz, Dxy, Dxz, Dyz, log S0]``."""
    b = np.asarray(bvalues, float) * 1e-6
    g = np.asarray(directions, float)
    sel = np.ones(len(b), bool) if b_max is None else b <= b_max * 1e-6
    b, g = b[sel], g[sel]
    X = np.stack([-b * g[:, 0] ** 2, -b * g[:, 1] ** 2, -b * g[:, 2] ** 2,
                  -2 * b * g[:, 0] * g[:, 1], -2 * b * g[:, 0] * g[:, 2], -2 * b * g[:, 1] * g[:, 2],
                  np.ones_like(b)], axis=1)
    return X, sel


@functools.lru_cache(maxsize=16)
def _setup(key, bvalues, directions, pgse, b_max, method, dtype_name, device_name):
    """The device tensors of one scheme: selection indices, design, pseudo-inverse, the symmetric outer-product
    table of the design's rows for the weighted normal equations, and the eigenvalue floor."""
    import torch
    bvals = np.frombuffer(bvalues, float)
    dirs = np.frombuffer(directions, float).reshape(-1, 3)
    pg = np.frombuffer(pgse, bool)
    X, sel = _design(bvals[pg], dirs[pg], b_max)
    idx = np.flatnonzero(pg)[sel]
    if np.count_nonzero(X[:, 0] != 0) < 6:
        raise ValueError("a tensor fit needs at least six diffusion-weighted PGSE measurements with b <= b_max")
    dt, dev = getattr(torch, dtype_name), torch.device(device_name)
    put = lambda a: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
    iu = np.triu_indices(7)
    XX = (X[:, :, None] * X[:, None, :])[:, iu[0], iu[1]]                       # (n, 28)
    return dict(idx=torch.as_tensor(idx, device=dev), X=put(X), pinv=put(np.linalg.pinv(X)), XX=put(XX),
                iu=(torch.as_tensor(iu[0], device=dev), torch.as_tensor(iu[1], device=dev)),
                floor=1e-6 / float(-X[:, :6].min()))


def _eigh(T):
    """Batched symmetric eigen-decomposition in chunks of :data:`EIGH_CHUNK`, ascending as torch returns it."""
    import torch
    if T.shape[0] <= EIGH_CHUNK:
        return torch.linalg.eigh(T)
    parts = [torch.linalg.eigh(T[i:i + EIGH_CHUNK]) for i in range(0, T.shape[0], EIGH_CHUNK)]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


def build_dti_fitter_torch(acquisition_scheme, *, b_max=None, method='WLS', dtype=None, device=None,
                           min_signal=1e-4):
    """The tensor fitter of one scheme: ``fit(data) -> DTIFit`` for ``data (N, N_meas)`` (numpy or torch; raw or
    S0-normalised signal), every voxel at once on ``device`` (the current CUDA device when one exists, else the CPU).

    Parameters
    ----------
    acquisition_scheme : DmipyAcquisitionScheme
    b_max : float or None
        The largest b-value (s/m^2) the fit uses; None uses every PGSE measurement.
    method : 'WLS' or 'OLS'
        Weighted (by the ordinary fit's predicted signal, as dipy's default) or ordinary least squares.
    dtype : torch dtype or None
        Default ``torch.float32``.
    min_signal : float
        The signal is clipped from below at this value before the logarithm (dipy's ``MIN_POSITIVE_SIGNAL``).
    """
    import torch
    if method not in ('WLS', 'OLS'):
        raise ValueError("method must be 'WLS' or 'OLS', got {!r}".format(method))
    dtype = torch.float32 if dtype is None else dtype
    dev = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    bvals = np.ascontiguousarray(acquisition_scheme.bvalues, float)
    dirs = np.ascontiguousarray(acquisition_scheme.gradient_directions, float)
    pgse = ~np.asarray(getattr(acquisition_scheme, 'is_ogse', np.zeros(len(bvals), bool)), bool)
    raw = (bvals.tobytes(), dirs.tobytes(), np.ascontiguousarray(pgse).tobytes())
    key = hashlib.sha1(b''.join(raw)).hexdigest()
    st = _setup(key, *raw, None if b_max is None else float(b_max), method, str(dtype).split('.')[-1], str(dev))
    return functools.partial(_fit, st=st, weighted=(method == 'WLS'), dtype=dtype, device=dev,
                             min_signal=float(min_signal))


def _fit(data, *, st, weighted, dtype, device, min_signal):
    import torch
    with full_precision(), torch.no_grad():
        d = torch.as_tensor(data, device=device)
        y = torch.log(torch.clamp(d[:, st['idx']].to(dtype), min=min_signal))          # (N, n)
        beta = y @ st['pinv'].T                                                        # (N, 7) ordinary fit
        if weighted:
            w2 = torch.exp(2.0 * (beta @ st['X'].T))                                   # (N, n): predicted S^2
            n = beta.shape[0]
            N_ = torch.zeros((n, 7, 7), dtype=dtype, device=device)
            N_[:, st['iu'][0], st['iu'][1]] = w2 @ st['XX']
            N_ = N_ + torch.triu(N_, 1).transpose(1, 2)
            beta = torch.linalg.solve(N_, ((w2 * y) @ st['X'])[:, :, None])[:, :, 0]
        dxx, dyy, dzz, dxy, dxz, dyz = (beta[:, k] for k in range(6))
        T = torch.stack([torch.stack([dxx, dxy, dxz], -1), torch.stack([dxy, dyy, dyz], -1),
                         torch.stack([dxz, dyz, dzz], -1)], -2)                        # (N, 3, 3) mm^2/s
        ev, vec = _eigh(T)
        ev, vec = torch.flip(ev, [-1]), torch.flip(vec, [-1])                          # descending
        ev = torch.clamp(ev, min=st['floor'])
        all_zero = (ev == 0).all(-1)
        fa = torch.sqrt(0.5 * ((ev[:, 0] - ev[:, 1]) ** 2 + (ev[:, 1] - ev[:, 2]) ** 2 + (ev[:, 2] - ev[:, 0]) ** 2)
                        / ((ev * ev).sum(-1) + all_zero))
        return DTIFit(S0=torch.exp(beta[:, 6]), tensor=T * 1e-6, evals=ev * 1e-6, evecs=vec, fa=fa,
                      md=ev.mean(-1) * 1e-6)
