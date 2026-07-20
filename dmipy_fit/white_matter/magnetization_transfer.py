"""Two-pool magnetization-transfer (MT) kernel for the white-matter model.

The analytical counterpart of the emergent MT in ``dmipy_sim`` (the surface-binding
walk + vector-Bloch bound pool).  It closes the "three observables of one wall"
picture: the SAME wall geometry that drives time-dependent diffusion and surface
relaxivity also drives MT, through the forward exchange rate

    k_f = kappa_MT * (S/V)                                                    (1)

which shares the identical ``S/V`` closed forms as surface relaxivity
(:mod:`dmipy_fit.white_matter.surface`) -- interior Gamma-averaged ``<4/d>`` and
exterior ``S_ext/V_EA`` -- so the MT geometry is obtained by ``rho -> kappa_MT``.

Two regimes, one geometric rate:

* **Degenerate (diffusion / plain-T2) limit** -- no off-resonance saturation,
  bound-pool ``T2b -> 0``, fast exchange.  MT acts as an extra transverse rate
  ``k_f`` on the free water, i.e. exactly the Brownstein--Tarr surface factor with
  ``rho -> kappa_MT`` (:func:`mt_transverse_attenuation`).  This is why MT is
  degenerate with surface relaxivity in the diffusion signal (Zheng et al., MRM
  2026, model MT precisely here: T1=inf, single TR, no saturation), and why a
  plain multi-echo experiment sees only the SUM ``(rho + kappa_MT)(S/V)``.

* **Distinguishing (qMT) regime** -- an off-resonance saturation pulse saturates the
  broad-linewidth bound pool and transfers to the free water via exchange, giving a
  Z-spectrum / MTR (:func:`z_spectrum`, :func:`mtr`).  A pure surface-relaxivity
  (transverse) sink produces NO off-resonance MTR; the bound pool does.  This is the
  observable that lifts the degeneracy.  The steady state is the two-pool
  Bloch--McConnell longitudinal solution (Henkelman 1993; Sled & Pike 2000/2001),
  parameterised by the SAME geometric ``k_f`` of Eq. (1).

Lineshape.  The bound-pool absorption uses a **Lorentzian** in ``T2b`` -- the exact
steady-state response of a short-``T2b`` Bloch pool -- so this kernel matches the
``dmipy_sim`` Monte-Carlo bound pool (an isotropic short-``T2b`` spin), not the
super-Lorentzian of a restricted macromolecular powder.  Pass a different lineshape
for quantitative-MT of real myelin (Morrison & Henkelman 1995); the exchange
algebra is lineshape-agnostic.

References
----------
Henkelman RM et al. Magn Reson Med 1993;29:759 (two-pool MT steady state).
Sled JG, Pike GB. J Magn Reson 2000; Magn Reson Med 2001 (pulsed/CW qMT).
Morrison C, Henkelman RM. Magn Reson Med 1995 (super-Lorentzian lineshape).
Zheng Z et al. Magn Reson Med 2026, 10.1002/mrm.70378 (MC MT as a surface process).
"""
from __future__ import annotations

import numpy as np

from .surface import b_hat_ia, b_hat_ea_long, b_hat_ea_short, mean_inv_diameter_4

__all__ = [
    "forward_rate_interior", "forward_rate_exterior",
    "mt_transverse_attenuation",
    "lorentzian_lineshape", "saturation_rate",
    "two_pool_steady_state", "z_spectrum", "mtr",
]


# ── the shared geometry: k_f = kappa_MT * (S/V) (reuses surface.py) ─────────────
def forward_rate_interior(alpha, scale_diameter, kappa_MT, volume_weighted=True):
    """Interior (intra-axonal) MT forward rate ``k_f = kappa_MT * <4/d>`` (s^-1).

    Same volume-weighted Gamma moment as the interior surface-relaxivity rate --
    only ``rho -> kappa_MT``.
    """
    return float(kappa_MT) * mean_inv_diameter_4(alpha, scale_diameter,
                                                 volume_weighted=volume_weighted)


def forward_rate_exterior(kappa_MT, S_ext_over_V_EA):
    """Exterior (extra-axonal) MT forward rate ``k_f = kappa_MT * S_ext/V_EA`` (s^-1).

    The exterior ``S_ext/V_EA`` is the same random-packing (Burcaw/Novikov) geometry
    that sets the exterior surface-relaxivity rate and the time-dependent diffusion.
    """
    return float(kappa_MT) * float(S_ext_over_V_EA)


# ── degenerate limit: MT == surface relaxivity with rho -> kappa_MT ─────────────
def mt_transverse_attenuation(compartment, tau_perp, *, kappa_MT, alpha=None,
                              scale_diameter=None, S_ext_over_V_EA=None,
                              D_inf=None, TE=None, regime="long"):
    """Degenerate-limit MT attenuation of the FREE-water transverse signal.

    In the fast-exchange, ``T2b -> 0``, saturation-free limit an MT bound pool
    removes free-water transverse magnetisation at the wall exactly as surface
    relaxivity does, so the attenuation is the Brownstein--Tarr surface factor with
    ``rho -> kappa_MT``.  This function simply forwards to
    :mod:`dmipy_fit.white_matter.surface` -- it exists to make the degeneracy
    explicit and to feed the diffusion-signal model.

    compartment : 'intra' -> Gamma-averaged interior Bessel factor (needs alpha,
        scale_diameter); 'extra' -> exterior factor (needs S_ext_over_V_EA;
        regime='long' clinical PGSE, or 'short' Mitra sqrt(TE) with D_inf, TE).
    """
    if compartment == "intra":
        if alpha is None or scale_diameter is None:
            raise ValueError("intra needs alpha and scale_diameter")
        return b_hat_ia(alpha, scale_diameter, kappa_MT, tau_perp)
    if compartment == "extra":
        if S_ext_over_V_EA is None:
            raise ValueError("extra needs S_ext_over_V_EA")
        if regime == "short":
            if D_inf is None or TE is None:
                raise ValueError("regime='short' needs D_inf and TE")
            return b_hat_ea_short(kappa_MT, S_ext_over_V_EA, D_inf, TE)
        return b_hat_ea_long(kappa_MT, S_ext_over_V_EA, tau_perp)
    raise ValueError(f"compartment must be 'intra' or 'extra', got {compartment!r}")


# ── qMT: two-pool off-resonance saturation (the degeneracy-lifting observable) ──
def lorentzian_lineshape(offset_hz, T2):
    """Absorption lineshape g(Delta) = T2 / (1 + (2 pi Delta T2)^2)  [s].

    The steady-state response of a Bloch pool of transverse time ``T2`` (matches the
    dmipy-sim MC bound pool).  On resonance g(0) = T2.
    """
    off = np.asarray(offset_hz, dtype=float)
    return T2 / (1.0 + (2.0 * np.pi * off * T2) ** 2)


def saturation_rate(offset_hz, w1_hz, T2):
    """RF saturation rate W(Delta) = omega1^2 g(Delta)  [s^-1], omega1 = 2 pi w1_hz."""
    w1 = 2.0 * np.pi * float(w1_hz)                       # rad/s
    return w1 ** 2 * lorentzian_lineshape(offset_hz, T2)


def two_pool_steady_state(k_f, k_r, R1a, R1b, W_a, W_b, M0a=1.0, M0b=None):
    """Free-pool longitudinal steady state Mz_a / M0a under CW saturation.

    Solves the two-pool Bloch--McConnell longitudinal system
        0 = R1a(M0a - Mza) - k_f Mza + k_r Mzb - W_a Mza
        0 = R1b(M0b - Mzb) - k_r Mzb + k_f Mza - W_b Mzb
    for Mza/M0a.  ``M0b`` defaults to detailed balance ``M0a * k_f / k_r``.
    All rates in s^-1; W_a, W_b may be arrays (over offset).  With W_a=W_b=0 this
    returns exactly 1 (equilibrium).

    This is the longitudinal quasi-steady-state (the Sled--Pike/Henkelman qMT
    standard): the RF enters only through the per-pool saturation rates W, i.e. the
    transverse magnetisation is assumed slaved.  It matches the full two-pool Bloch
    steady state (e.g. dmipy_sim.mt) in the qMT regime -- modest B1 and offsets well
    outside the narrow free-pool line, where the bound pool dominates and free-pool
    direct saturation is small.  A strong B1 near the free resonance stresses the
    approximation (the omitted free-pool transverse becomes non-negligible).
    """
    if M0b is None:
        M0b = M0a * k_f / k_r
    W_a = np.asarray(W_a, dtype=float)
    W_b = np.asarray(W_b, dtype=float)
    Db = R1b + k_r + W_b
    num = R1a * M0a + k_r * R1b * M0b / Db
    den = R1a + k_f + W_a - k_f * k_r / Db
    return (num / den) / M0a


def z_spectrum(offsets_hz, w1_hz, *, k_f, k_r, T1a, T1b, T2a, T2b, M0a=1.0, M0b=None):
    """Free-pool Z-spectrum: normalised Mz_a vs saturation offset (Hz).

    The bound pool (short ``T2b``) saturates over a broad offset range and transfers
    to the free pool via exchange -> the MT dip; the narrow free pool (long ``T2a``)
    saturates only near resonance.  ``k_f = kappa_MT * (S/V)`` ties the dip depth to
    the wall geometry.
    """
    W_a = saturation_rate(offsets_hz, w1_hz, T2a)
    W_b = saturation_rate(offsets_hz, w1_hz, T2b)
    return two_pool_steady_state(k_f, k_r, 1.0 / T1a, 1.0 / T1b, W_a, W_b,
                                 M0a=M0a, M0b=M0b)


def mtr(offset_hz, w1_hz, *, k_f, k_r, T1a, T1b, T2a, T2b,
        reference_offset_hz=1e6, M0a=1.0, M0b=None):
    """Magnetization-transfer ratio MTR = 1 - Z(offset) / Z(reference_offset).

    The reference (default far off-resonance) is the unsaturated level.  A pure
    surface-relaxivity mechanism (no bound pool: k_f=0) gives MTR=0 at any offset.
    """
    z_on = z_spectrum(offset_hz, w1_hz, k_f=k_f, k_r=k_r, T1a=T1a, T1b=T1b,
                      T2a=T2a, T2b=T2b, M0a=M0a, M0b=M0b)
    z_ref = z_spectrum(reference_offset_hz, w1_hz, k_f=k_f, k_r=k_r, T1a=T1a,
                       T1b=T1b, T2a=T2a, T2b=T2b, M0a=M0a, M0b=M0b)
    return 1.0 - z_on / z_ref
