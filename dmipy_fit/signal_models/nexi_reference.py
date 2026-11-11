"""Opt-in reference signal for Jelescu et al. (2022), NeuroImage 256:119277.

This module does not replace the existing ``reference_models.nexi`` factory.
"""

import numpy as np

__all__ = ["nexi_jelescu2022"]


def nexi_jelescu2022(bvalues, diffusion_time, *, f, di, de, t_ex):
    r"""Normalised powder signal of published NEXI, Eqs. (3)--(6).

    ``bvalues`` (s/m²) and ``diffusion_time`` (s) broadcast together, with
    b = q² t in the narrow-pulse approximation. The caller supplies t
    explicitly; no finite-pulse timing correction is inferred. ``di`` is
    the stick axial diffusivity and ``de`` the independent isotropic
    extra-neurite diffusivity (m²/s). ``f`` is the intra-neurite signal
    fraction. ``t_ex`` is 1/(r_ie + r_ei), in seconds, with detailed balance
    r_ie = (1-f)/t_ex and r_ei = f/t_ex. All inputs are required.

    Uses 64-point Gauss-Legendre integration on cos(theta) in [0, 1]. A
    symmetric similarity transform of the two-site generator gives the
    same two exponentials as Eqs. (3)--(5), including their degeneracies.
    ``t_ex=np.inf`` means no exchange; ``t_ex=0`` explicitly requests the
    fast-exchange limit of each orientation kernel, averaged afterwards.
    This is a CPU reference predictor, without a fitting interface or S0.

    doi:10.1016/j.neuroimage.2022.119277
    """
    b, t = np.broadcast_arrays(np.asarray(bvalues, dtype=float),
                               np.asarray(diffusion_time, dtype=float))
    if np.any(~np.isfinite(b)) or np.any(b < 0):
        raise ValueError("bvalues must be finite and nonnegative")
    if np.any(~np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("diffusion_time must be finite and positive")
    if not np.isfinite(f) or not 0 < f < 1:
        raise ValueError("f must be strictly between zero and one")
    if not np.isfinite(di) or not np.isfinite(de) or di < 0 or de < 0:
        raise ValueError("di and de must be finite and nonnegative")
    if np.isnan(t_ex) or t_ex < 0:
        raise ValueError("t_ex must be nonnegative or positive infinity")

    nodes, weights = np.polynomial.legendre.leggauss(64)
    u, weights = (nodes + 1) / 2, weights / 2
    d_intra = di * u ** 2
    if t_ex == 0:
        return np.exp(-b[..., None] * (f * d_intra + (1-f) * de)) @ weights

    # diag(sqrt(fractions))^-1 K diag(sqrt(fractions)) is symmetric.
    root = np.sqrt([f, 1-f])
    generator = np.array([[-(1-f), np.sqrt(f * (1-f))],
                          [np.sqrt(f * (1-f)), -f]]) / t_ex
    matrices = np.broadcast_to(t[..., None, None, None] * generator,
                               b.shape + (len(u), 2, 2)).copy()
    matrices[..., 0, 0] -= b[..., None] * d_intra
    matrices[..., 1, 1] -= b[..., None] * de
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    amplitudes = np.einsum("...ij,i->...j", eigenvectors, root) ** 2
    kernels = np.sum(amplitudes * np.exp(eigenvalues), axis=-1)
    return kernels @ weights
