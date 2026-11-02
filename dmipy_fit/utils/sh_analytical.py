# -*- coding: utf-8 -*-
"""
Exact analytical spherical harmonics of the Gaussian (Zeppelin/Stick) diffusion kernels.

The Watson and Bingham orientation distributions' coefficients and the harmonic rotation blocks are
``dmipy_replay.math.sh_analytical`` and ``dmipy_replay.replay.so3.wigner_blocks``, the ecosystem's one implementation.

Gaussian kernel (Zeppelin / Stick)
-----------------------------------
For E(theta) = exp(-b*lambda_perp) * exp(kappa*cos²θ)  with kappa = -b*(lambda_par - lambda_perp),
the m=0 rotational harmonic coefficient at order l is:

    kernel_rh[l//2] = 2*pi * sqrt((2l+1)/(4*pi)) * exp(-b*lambda_perp) * J_l(kappa)

The kernels use the recurrence for J_l(kappa) = ∫₋₁¹ P_l(t) exp(kappa t²) dt:

    J_0     = sqrt(pi) * erfi(sqrt(kappa)) / sqrt(kappa)  kappa > 0  (Watson)
              sqrt(pi) * erf(sqrt(|kappa|)) / sqrt(|kappa|)  kappa < 0  (Gaussian)
    K_1     = exp(kappa)/kappa - J_0 / (2 kappa)
    J_{l+2} = [(2l+3) K_{l+1} - (l+1) J_l] / (l+2)       l even
    K_{l+3} = K_{l+1} - (2l+5) J_{l+2} / (2 kappa)

O(l_max), numerically stable. For kappa > 700 the asymptotic
r_l ≈ 1 − l(l+1)/(4·kappa) is used to avoid float64 overflow.

References
----------
Kaden E, Knosche TR, Anwander A (2007). Parametric spherical deconvolution.
NeuroImage 37(2):474-488.
"""

import math

import numpy as np
from scipy.special import erfi, erf
from dmipy_replay.replay.so3 import wigner_blocks

__all__ = [
    'gaussian_signal_sh',
    'gaussian_signal_lm',
    'gaussian_J_l',
    'gaussian_kernel_rh',
]


def gaussian_signal_sh(kappa, beta, l_max=8):
    r"""Unnormalized SH coefficients of exp(kappa·n_x² + beta·n_y²).

    The canonical Bingham coefficients of ``dmipy_replay.math.sh_analytical.bingham_canonical_sh``, not divided by
    the partition function Z.  Used to represent the angular factor of a Gaussian
    diffusion kernel (Zeppelin, Stick, or B-tensor with traceless l=2 part) as
    a set of SH coefficients before rotating to the lab frame.

    Physical parameters (kappa ≥ beta ≥ 0) come from eigendecomposing the
    traceless part of the diffusion phase matrix:

        kappa = a₃ − a₁,  beta = a₃ − a₂   (a₁ ≤ a₂ ≤ a₃)

    where the a_i are eigenvalues of the traceless symmetric phase matrix M.
    The full signal SH is then ``pre_factor · exp(−a₃) · rotate(gaussian_signal_sh(kappa, beta))``.

    Parameters
    ----------
    kappa : float
        Primary Bingham parameter (≥ 0).  kappa = beta for axially symmetric case.
    beta : float
        Secondary Bingham parameter (0 ≤ beta ≤ kappa).
    l_max : int, optional
        Maximum SH order (must be even).  Default 8.

    Returns
    -------
    sh : ndarray, shape ((l_max+1)*(l_max+2)//2,)
        Tournier real SH coefficients.  Non-zero only at even l, even m ≥ 0.
        c_0^0 = Z / (2√π) where Z is the Bingham partition function.
    """
    from numpy.polynomial.legendre import leggauss
    from scipy.special import lpmv, iv as bessel_iv

    kappa = float(np.asarray(kappa).flat[0])
    beta = float(np.asarray(beta).flat[0])

    # Adaptive GL order: the integrand exp((kappa+beta)/2*(1−t²)) is peaked near
    # t=0 with width ~1/√K where K=(kappa+beta)/2.  Accurate integration needs
    # N_GL >> √K nodes.  32 is sufficient for ODF concentrations (kappa < ~50)
    # but signal kernels can have kappa ~ O(100) at high b-values.
    N_GL = max(32, int(6.0 * math.sqrt(kappa + beta)) + 16)
    nodes, weights = leggauss(N_GL)
    s2 = 1.0 - nodes ** 2

    A = (kappa + beta) / 2.0 * s2
    B = (kappa - beta) / 2.0 * s2
    exp_A = np.exp(A)

    n_coef = (l_max + 1) * (l_max + 2) // 2
    sh = np.zeros(n_coef)

    counter = 0
    for l in range(0, l_max + 1, 2):
        n_in_order = 2 * l + 1
        for m_block in range(n_in_order):
            m = m_block - l
            coef_idx = counter + m_block

            if m < 0 or m % 2 != 0:
                sh[coef_idx] = 0.0
                continue

            q = m // 2
            Iq = bessel_iv(q, B)
            Plm = lpmv(m, l, nodes)

            if m == 0:
                N_lm = math.sqrt((2 * l + 1) / (4.0 * math.pi))
            else:
                fact_ratio = (math.factorial(l - m)
                              / float(math.factorial(l + m)))
                N_lm = math.sqrt(2.0 * (2 * l + 1) / (4.0 * math.pi)
                                 * fact_ratio)

            # No division by Z — unnormalized projection
            sh[coef_idx] = (N_lm * 2.0 * np.pi
                            * float(np.dot(weights, exp_A * Iq * Plm)))

        counter += n_in_order

    return sh


def _l2sh_to_matrix(phi2m):
    r"""Convert 5 Tournier l=2 real SH coefficients to a 3×3 traceless matrix.

    Inverts the identity:  n̂ᵀ M n̂ = Σ_{m=-2}^{2} c_m Y₂^m(n̂)

    using the Cartesian forms of the Tournier real SH (legacy=False, no
    Condon-Shortley phase):

        Y₂^{-2} = f15 · 2 n_x n_y
        Y₂^{-1} = f15 · 2 n_y n_z
        Y₂^0    = f5  · (2n_z² − n_x² − n_y²)
        Y₂^{+1} = f15 · 2 n_x n_z
        Y₂^{+2} = f15 · (n_x² − n_y²)

    where f5 = √(5/16π),  f15 = √(15/16π).

    Parameters
    ----------
    phi2m : array_like, shape (5,)
        SH coefficients [c_{-2}, c_{-1}, c_0, c_{+1}, c_{+2}].

    Returns
    -------
    M : ndarray, shape (3, 3)
        Symmetric traceless matrix such that n̂ᵀMn̂ = Σ c_m Y₂^m(n̂).
    """
    phi2m = np.asarray(phi2m, dtype=np.float64).ravel()
    c_m2, c_m1, c_0, c_p1, c_p2 = phi2m

    f5 = math.sqrt(5.0 / (16.0 * math.pi))
    f15 = math.sqrt(15.0 / (16.0 * math.pi))

    M = np.zeros((3, 3), dtype=np.float64)
    M[0, 0] = -f5 * c_0 + f15 * c_p2
    M[1, 1] = -f5 * c_0 - f15 * c_p2
    M[2, 2] = 2.0 * f5 * c_0
    M[0, 1] = M[1, 0] = f15 * c_m2
    M[0, 2] = M[2, 0] = f15 * c_p1
    M[1, 2] = M[2, 1] = f15 * c_m1
    return M


def gaussian_signal_lm(M_l2, pre_factor, l_max=8):
    r"""SH coefficients of a Gaussian signal kernel with arbitrary B-tensor shape.

    Computes the exact SH expansion of ``E(n̂) = pre_factor · exp(−n̂ᵀ M_l2 n̂)``
    where M_l2 is the traceless-symmetric phase matrix (the l=2 component of the
    full diffusion phase φ(n̂)).

    Algorithm:

    1. Eigendecompose M_l2 = R diag(a₁, a₂, a₃) Rᵀ, a₁ ≤ a₂ ≤ a₃.
    2. Set kappa = a₃ − a₁ ≥ 0, beta = a₃ − a₂ ≥ 0.
    3. Compute unnormalized canonical SH = :func:`gaussian_signal_sh(kappa, beta)`.
    4. Scale by ``pre_factor · exp(−a₃)``.
    5. Rotate to lab frame via Wigner-D using R = [ê₁|ê₂|ê₃].

    For a Zeppelin with PGSE (b, g):
        ``M_l2 = (λ_par − λ_perp) · (b·g⊗g − trace(b·g⊗g)/3·I)``
        ``pre_factor = exp(−b·(λ_par + 2λ_perp)/3)``

    For a Stick with PGSE (b, g):
        ``M_l2 = λ_par · (b·g⊗g − trace(b·g⊗g)/3·I)``
        ``pre_factor = exp(−b·λ_par/3)``

    Parameters
    ----------
    M_l2 : ndarray, shape (3, 3)
        Symmetric traceless matrix (the l=2 phase matrix).
    pre_factor : float
        Multiplicative scalar from the l=0 part of the phase (= exp(−φ₀)).
    l_max : int, optional
        Maximum SH order (must be even).  Default 8.

    Returns
    -------
    sh : ndarray, shape ((l_max+1)*(l_max+2)//2,)
        Tournier real SH coefficients of E(n̂).
    """
    M_l2 = np.asarray(M_l2, dtype=np.float64)

    # Eigendecompose: vals ascending, vecs are columns of rotation matrix R
    vals, vecs = np.linalg.eigh(M_l2)   # a₁ ≤ a₂ ≤ a₃
    # Ensure proper rotation (det = +1)
    if np.linalg.det(vecs) < 0:
        vecs[:, 2] = -vecs[:, 2]

    kappa = float(vals[2] - vals[0])    # ≥ 0
    beta = float(vals[2] - vals[1])     # ≥ 0, kappa ≥ beta

    scalar = float(pre_factor) * math.exp(-float(vals[2]))

    sh_canon = gaussian_signal_sh(kappa, beta, l_max)

    D_blocks = wigner_blocks(l_max, vecs[None])

    n_coef = (l_max + 1) * (l_max + 2) // 2
    sh = np.zeros(n_coef)
    counter = 0
    for l_idx, l in enumerate(range(0, l_max + 1, 2)):
        n_in_l = 2 * l + 1
        sh[counter:counter + n_in_l] = D_blocks[l][0] @ sh_canon[counter:counter + n_in_l]
        counter += n_in_l

    return scalar * sh


# ============================================================================
# Gaussian diffusion kernels (Zeppelin, Stick, TemporalZeppelin)
# ============================================================================

def gaussian_J_l(kappa, l_max=8):
    r"""Compute J_l(kappa) = ∫₋₁¹ P_l(t) exp(kappa·t²) dt for even l.

    Works for any real kappa.  For Gaussian diffusion kernels the physical
    range is kappa = −b·(λ_∥ − λ_⊥) ≤ 0 (no overflow risk).  For the Watson
    distribution use kappa > 0 (handled by :func:`watson_zonal_ratios`).

    Parameters
    ----------
    kappa : float
        Exponent coefficient. kappa < 0 uses erf; kappa > 0 uses erfi.
    l_max : int, optional
        Maximum SH order (must be even). Default 8.

    Returns
    -------
    J : ndarray, shape (l_max//2 + 1,)
        J[m] = J_{2m}(kappa) for m = 0, 1, ..., l_max//2.
    """
    kappa = float(np.asarray(kappa).flat[0])
    n_levels = l_max // 2 + 1
    J = np.zeros(n_levels)

    if abs(kappa) < 1e-12:
        J[0] = 2.0   # J_0(0) = int_{-1}^{1} 1 dt = 2; J_l(0) = 0 for l>0
        return J

    if kappa > 0:
        sqrt_abs = np.sqrt(kappa)
        J0 = np.sqrt(np.pi) * erfi(sqrt_abs) / sqrt_abs
    else:
        sqrt_abs = np.sqrt(-kappa)
        J0 = np.sqrt(np.pi) * erf(sqrt_abs) / sqrt_abs

    J[0] = J0
    if n_levels == 1:
        return J

    exp_k = np.exp(kappa)    # < 1 for kappa < 0, no overflow
    K_curr = exp_k / kappa - J0 / (2.0 * kappa)   # K_1

    for m in range(n_levels - 1):
        J[m + 1] = ((4 * m + 3) * K_curr - (2 * m + 1) * J[m]) / (2 * m + 2)
        K_curr = K_curr - (4 * m + 5) * J[m + 1] / (2.0 * kappa)

    return J


def gaussian_kernel_rh(b, lambda_par, lambda_perp, sh_order=8):
    r"""Analytical RH coefficients for a z-aligned Gaussian kernel.

    For the signal E(θ) = exp(−b·λ_⊥) · exp(κ·cos²θ) with κ = −b·(λ_∥ − λ_⊥),
    the m=0 rotational harmonic coefficient at SH order l is

        kernel_rh[l//2] = 2π · √((2l+1)/(4π)) · exp(−b·λ_⊥) · J_l(κ)

    This replaces the 10-point angular-sampling approximation used in
    ``AnisotropicSignalModelProperties.rotational_harmonics_representation``.

    Parameters
    ----------
    b : float
        B-value in s/m².
    lambda_par : float
        Parallel diffusivity in m²/s.
    lambda_perp : float
        Perpendicular diffusivity in m²/s.  Pass 0.0 for C1Stick.
    sh_order : int, optional
        Maximum SH order (even). Default 8.

    Returns
    -------
    rh : ndarray, shape (sh_order//2 + 1,)
        Rotational harmonic coefficients.  Directly usable as
        ``kernel_rh`` in :func:`~dmipy.utils.spherical_convolution.sh_convolution`.
    """
    b = float(np.asarray(b).flat[0])
    lambda_par = float(np.asarray(lambda_par).flat[0])
    lambda_perp = float(np.asarray(lambda_perp).flat[0])
    kappa = -b * (lambda_par - lambda_perp)
    J = gaussian_J_l(kappa, l_max=sh_order)

    n_levels = sh_order // 2 + 1
    rh = np.empty(n_levels)
    exp_factor = float(np.exp(-b * lambda_perp))
    l_vals = np.arange(0, 2 * n_levels, 2, dtype=float)     # 0, 2, 4, ...
    rh = (2.0 * np.pi * np.sqrt((2 * l_vals + 1) / (4.0 * np.pi))
          * exp_factor * J)
    return rh
