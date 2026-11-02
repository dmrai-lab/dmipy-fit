"""The extra-axonal short-time Mitra surface-relaxation form of the coherence-pathway paper (eq:b_hat_ext)::

      B_EA = exp(-(kappa_ext/2) integral chi_perp sqrt(D_inf/(pi t)) dt)
           = exp(-kappa_ext sqrt(D_inf TE/pi))   for PGSE chi_perp=1.

The long-time, Gamma-averaged and exterior surface-to-volume forms are :mod:`dmipy_analytic.relaxation`.
"""
from __future__ import annotations

import numpy as np


def b_hat_ea_short(rho_ext, S_ext_over_V, D_inf, TE):
    """Extra-axonal short-time Mitra surface attenuation (eq:b_hat_ext), PGSE chi=1.

    kappa_ext = 2 rho_ext S_ext/V;  exponent = (kappa_ext/2) * 2 sqrt(D_inf TE/pi).
    """
    kappa_ext = 2.0 * rho_ext * S_ext_over_V
    return np.exp(-kappa_ext * np.sqrt(D_inf * TE / np.pi))
