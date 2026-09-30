# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import eval_legendre

from .sh_basis import optional_module

numba, have_numba = optional_module("numba")

__all__ = [
    'real_sym_rh_basis',
    'sh_convolution'
]


def real_sym_rh_basis(sh_order, theta):
    r"""The real symmetric rotational harmonics ``Y_l^0`` of even degree up to ``sh_order`` at polar angles
    ``theta``: ``(n, sh_order // 2 + 1)``, ``Y_l^0(theta) = sqrt((2l + 1) / (4 pi)) P_l(cos theta)`` with ``P_l``
    the Legendre polynomial (the orthonormal basis; axially symmetric, so the azimuth does not enter).
    """
    x = np.cos(np.reshape(np.asarray(theta, float), [-1]))
    degrees = np.arange(0, int(sh_order) + 1, 2)
    return np.stack([np.sqrt((2 * l + 1) / (4 * np.pi)) * eval_legendre(l, x) for l in degrees], axis=1)


def sh_convolution(f_distribution_sh, kernel_rh):
    r"""Spherical convolution between a fiber distribution (f) in spherical
    harmonics and a kernel in terms of rotational harmonics (oriented along the
    z-axis).

    Parameters
    ----------
    f_distribution_sh : array, shape (sh_coef)
        spherical harmonic coefficients of a fiber distribution.
    kernel_rh : array, shape (sh_coef),
        rotational harmonic coefficients of the convolution kernel. In our case
        this is the spherical signal of one micro-environment at one b-value.

    Returns
    -------
    f_kernel_convolved : array, shape (sh_coef)
        spherical harmonic coefficients of the convolved kernel and
        distribution.
    """
    sh_order_rh = 2 * (len(kernel_rh) - 1)
    number_coef_sh = len(f_distribution_sh)
    sh_order_sh = int(-3 + np.sqrt(9 - 4 * (2 - 2 * number_coef_sh))) // 2

    sh_order_used = min(sh_order_rh, sh_order_sh)
    number_coef_used = int((sh_order_used + 2) * (sh_order_used + 1) // 2)

    f_kernel_convolved = np.zeros(number_coef_used)

    counter = 0
    for n_ in range(0, sh_order_used + 1, 2):
        coef_in_order = 2 * n_ + 1
        f_kernel_convolved[counter: counter + coef_in_order] = (
            f_distribution_sh[counter: counter + coef_in_order] *
            kernel_rh[n_ // 2] * np.sqrt((4 * np.pi) / (2 * n_ + 1))
        )
        counter += coef_in_order
    return f_kernel_convolved


if have_numba:
    sh_convolution = numba.njit()(sh_convolution)
