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


def _b_hat_gamma_closed_form(a, beta, c):
    r"""Gamma-averaged surface attenuation ``E_d[exp(-c/d)]`` for
    ``d ~ Gamma(shape a, rate beta)`` in closed form:

    .. math:: B = \frac{2 (\beta c)^{a/2}}{\Gamma(a)} K_a(2\sqrt{\beta c}),

    the Bessel-K result of ``\int_0^\infty x^{a-1} e^{-\beta x - c/x} dx``.
    ``c -> 0`` gives 1 (no relaxation).  Shared by the cylinder
    (:func:`b_hat_ia`) and sphere (:func:`b_hat_sphere`) forms, which differ
    only in ``a`` (volume-weight shift) and ``c`` (S/V coefficient)."""
    c = np.asarray(c, dtype=float)
    z = 2.0 * np.sqrt(beta * c)
    with np.errstate(over='ignore', invalid='ignore'):
        B = 2.0 * (beta * c) ** (a / 2.0) / gamma_fn(a) * kv(a, z)
    return np.where(c <= 0, 1.0, B)


def b_hat_ia(alpha, scale_diameter, rho_int, tau_perp, volume_weighted=True):
    """Intra-pore (cylinder) Gamma-averaged surface attenuation (eq:b_hat_int_dist).

    ``E_d[exp(-rho (4/d) tau)]`` for a cylinder lumen ``d ~ Gamma``: S/V = 4/d,
    water content per cylinder ~ cross-sectional area d^2.

    Parameters
    ----------
    alpha : float            Gamma shape over diameter.
    scale_diameter : float   Gamma scale beta_d (m); rate beta = 1/scale.
    rho_int : float          Interior surface relaxivity (m/s).
    tau_perp : float or array  Transverse occupancy time (s).
    volume_weighted : bool   Use d^2 P(d) (shape alpha+2) to match spin/area
                             weighting.  False = paper's number-weighted form.
    """
    a = alpha + 2.0 if volume_weighted else alpha      # cylinder: water ~ d^2
    beta = 1.0 / scale_diameter                        # rate (1/m)
    c = 4.0 * rho_int * np.asarray(tau_perp, dtype=float)   # S/V = 4/d
    return _b_hat_gamma_closed_form(a, beta, c)


def b_hat_sphere(alpha, scale_diameter, rho_int, tau_perp, volume_weighted=True):
    """Intra-sphere Gamma-averaged surface attenuation -- the sphere analog of
    :func:`b_hat_ia`.

    ``E_d[exp(-rho (6/d) tau)]`` for a sphere diameter ``d ~ Gamma``: a sphere
    has S/V = 3/R = 6/d, and its water content ~ volume d^3, so the spin-average
    uses the volume-weighted distribution d^3 P(d) -> Gamma(alpha+3).  Same
    closed form as the cylinder with the coefficient 4->6 and the weight
    shift +2->+3.

    Parameters
    ----------
    alpha : float            Gamma shape over diameter.
    scale_diameter : float   Gamma scale beta_d (m); rate beta = 1/scale.
    rho_int : float          Interior surface relaxivity (m/s).
    tau_perp : float or array  Transverse occupancy time (s).
    volume_weighted : bool   Use d^3 P(d) (shape alpha+3) to match spin/volume
                             weighting.  False = number-weighted form.
    """
    a = alpha + 3.0 if volume_weighted else alpha      # sphere: water ~ d^3
    beta = 1.0 / scale_diameter                        # rate (1/m)
    c = 6.0 * rho_int * np.asarray(tau_perp, dtype=float)   # S/V = 6/d
    return _b_hat_gamma_closed_form(a, beta, c)


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


# Exterior surface-to-volume of a packed cell population, per cell geometry:
# (coeff, volume-weight power m).  Cell S/V = coeff/d (sphere 6, cylinder 4,
# plane 2); water content ~ d^m (sphere d^3, cylinder d^2, plane d^1). The
# volume-weighted <coeff/d> = coeff / (scale (alpha + m - 1)).
_EXTERIOR_GEOMETRY = {'sphere': (6.0, 3), 'cylinder': (4.0, 2), 'plane': (2.0, 1)}


def exterior_surface_to_volume(f_cell, gamma_shape, gamma_scale_outer_diameter,
                               *, geometry):
    r"""Exterior surface-to-volume ratio ``S_ext/V_EA`` (1/m) of the extra-cellular
    space around a packed population of cells with **outer**-diameter Gamma
    distribution.

    .. math::
        \frac{S_{\mathrm{ext}}}{V_{\mathrm{EA}}}
        = \frac{k\, f_{\mathrm{cell}}}{(1-f_{\mathrm{cell}})\,(\alpha+m-1)\,\beta_{\mathrm{outer}}}

    with ``(k, m)`` set by the cell geometry: cylinder (axons) ``(4, 2)`` → ``(α+1)``,
    sphere (somas) ``(6, 3)`` → ``(α+2)``, plane ``(2, 1)`` → ``α``.  ``geometry``
    is **required** — the exterior S/V depends on cell shape and cannot be inferred
    from the extra-cellular compartment, so there is deliberately no default (a
    silent cylinder value would be wrong by 1.5× for a soma/sphere population).

    Parameters
    ----------
    f_cell : float
        Cell (OUTER) volume fraction; ``1 - f_cell`` is the extra-cellular volume.
        For myelinated axons this is the fibre (axon+myelin) packing fraction.
    gamma_shape : float
        Shape ``alpha`` of the cell outer-diameter Gamma distribution.
    gamma_scale_outer_diameter : float
        Scale ``beta`` (m) of the outer-diameter Gamma distribution
        (mean outer diameter = ``alpha * beta``).
    geometry : {'sphere', 'cylinder', 'plane'}
        Cell shape (required).
    """
    try:
        coeff, m = _EXTERIOR_GEOMETRY[geometry]
    except KeyError:
        raise ValueError(
            "geometry must be one of {}; got {!r}".format(
                sorted(_EXTERIOR_GEOMETRY), geometry))
    return (coeff * f_cell
            / ((1.0 - f_cell) * (gamma_shape + m - 1.0)
               * gamma_scale_outer_diameter))
