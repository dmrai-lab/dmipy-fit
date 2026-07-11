"""Analytical surface-relaxivity factors from the coherence-pathway paper.

Implements the B_hat_{P,i} factors of eq:mc_general (paper sections/theory.tex,
Sec. theory:surface):

  intra-pore (Brownstein-Tarr, Gamma-averaged), eq:b_hat_int_dist:
      B_IA = 2 (beta c)^(alpha/2) / Gamma(alpha) * K_alpha(2 sqrt(beta c)),
      c = 4 rho_int tau_perp,  beta = gamma RATE = 1/scale_diameter.
      Weak limit: exp(-rho_int <4/d> tau_perp), <4/d> = 4 beta/(alpha-1).

  extra-axonal long-time / clinical PGSE (eq:b_hat_ext_long):
      B_EA = exp(-rho_ext (S_ext/V_EA) tau_perp).
  extra-axonal short-time Mitra (eq:b_hat_ext), for OGSE/short-TE:
      B_EA = exp(-(kappa_ext/2) integral chi_perp sqrt(D_inf/(pi t)) dt)
           = exp(-kappa_ext sqrt(D_inf TE/pi))   for PGSE chi_perp=1.

VOLUME WEIGHTING.  The signal is the spin (water) average, and water content per
cylinder scales as the cross-sectional area d^2.  The paper's eq:b_hat_int_dist
averages exp(-rho 4/d tau) over the NUMBER distribution P(d); to match an MC whose
walkers populate cylinders proportional to area, the VOLUME-weighted distribution
d^2 P(d) must be used -- a Gamma(alpha+2, beta).  ``volume_weighted=True`` (default)
applies this (shape -> alpha+2); set False for the paper's as-written number form.
"""
from __future__ import annotations

import numpy as np
from scipy.special import kv, gamma as gamma_fn


def b_hat_ia(alpha, scale_diameter, rho_int, tau_perp, volume_weighted=True):
    """Intra-pore Gamma-averaged surface attenuation (eq:b_hat_int_dist).

    Parameters
    ----------
    alpha : float            Gamma shape over diameter.
    scale_diameter : float   Gamma scale beta_d (m); rate beta = 1/scale.
    rho_int : float          Interior surface relaxivity (m/s).
    tau_perp : float or array  Transverse occupancy time (s).
    volume_weighted : bool   Use d^2 P(d) (shape alpha+2) to match spin/area
                             weighting.  False = paper's number-weighted form.
    """
    a = alpha + 2.0 if volume_weighted else alpha
    beta = 1.0 / scale_diameter                       # rate (1/m)
    tau_perp = np.asarray(tau_perp, dtype=float)
    c = 4.0 * rho_int * tau_perp                       # m
    z = 2.0 * np.sqrt(beta * c)                         # dimensionless
    # B = 2 (beta c)^(a/2) / Gamma(a) * K_a(z); stable for z>0
    with np.errstate(over='ignore', invalid='ignore'):
        B = 2.0 * (beta * c) ** (a / 2.0) / gamma_fn(a) * kv(a, z)
    # c -> 0 limit is 1 (no relaxation)
    B = np.where(c <= 0, 1.0, B)
    return B


def mean_inv_diameter_4(alpha, scale_diameter, volume_weighted=True):
    """<4/d> for the (volume- or number-weighted) Gamma distribution.

    number:  4 beta/(alpha-1);   volume (d^2 P): 4 beta/(alpha+1).
    """
    a = alpha + 2.0 if volume_weighted else alpha
    beta = 1.0 / scale_diameter
    return 4.0 * beta / (a - 1.0)


def b_hat_ea_long(rho_ext, S_ext_over_V_EA, tau_perp):
    """Extra-axonal long-time (clinical PGSE) surface attenuation (eq:b_hat_ext_long)."""
    return np.exp(-rho_ext * S_ext_over_V_EA * np.asarray(tau_perp, float))


def b_hat_ea_short(rho_ext, S_ext_over_V, D_inf, TE):
    """Extra-axonal short-time Mitra surface attenuation (eq:b_hat_ext), PGSE chi=1.

    kappa_ext = 2 rho_ext S_ext/V;  exponent = (kappa_ext/2) * 2 sqrt(D_inf TE/pi).
    """
    kappa_ext = 2.0 * rho_ext * S_ext_over_V
    return np.exp(-kappa_ext * np.sqrt(D_inf * TE / np.pi))


def exterior_surface_to_volume(f_axon, gamma_shape, gamma_scale_outer_diameter):
    r"""Extra-axonal exterior surface-to-volume ratio ``S_ext/V_EA`` (1/m) from the
    axon **outer** (fibre) diameter Gamma distribution — the same closed form the
    Monte-Carlo substrate uses (its inner-scale form
    ``4 f g / ((1-f)(alpha+1) beta_inner)`` re-expressed on the outer scale
    ``beta_outer = beta_inner / g``, where the g-ratio cancels):

    .. math::
        \frac{S_{\mathrm{ext}}}{V_{\mathrm{EA}}}
        = \frac{4\, f_{\mathrm{axon}}}{(1-f_{\mathrm{axon}})\,(\alpha+1)\,\beta_{\mathrm{outer}}}

    Parameters
    ----------
    f_axon : float
        Fibre (OUTER) volume fraction -- the total axon+myelin packing fraction, so
        ``1 - f_axon`` is the extra-axonal volume.  (Not the lumen fraction.)
    gamma_shape : float
        Shape ``alpha`` of the axon diameter Gamma distribution (same for inner/outer).
    gamma_scale_outer_diameter : float
        Scale ``beta`` (m) of the **outer** (fibre) diameter Gamma distribution
        (mean outer diameter = ``alpha * beta``).
    """
    return (4.0 * f_axon
            / ((1.0 - f_axon) * (gamma_shape + 1.0) * gamma_scale_outer_diameter))
