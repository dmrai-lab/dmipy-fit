"""Compartment-wise T1/T2 in the Karger exchange model, PGSE vs PGSTE.

X0GeneralizedKarger propagates the two-compartment magnetisation with a
matrix exponential once any per-compartment relaxation time is supplied.  These
tests exercise that path end-to-end for both encodings and pin the timing of the
relaxation factors down to a hand computation:

* PGSE (spin echo, no mixing time): the transverse (T2) factor accrues over the
  whole echo and T1 never enters -- the signal is independent of T1.
* PGSTE (stimulated echo, built with ``AcquisitionScheme.from_pgste``): during
  the mixing time TM the magnetisation is stored longitudinally, so the mixing
  window carries the longitudinal ``exp(-TM/T1)`` weighting (and exchange) but no
  transverse relaxation -- T2 is gated to the encoding lobes.

The hand computations below re-derive the model's documented propagators
(eq. Karger SE / STE) from scratch in the idealised instantaneous-pulse limit
(all RF durations zero) so that a match validates the wiring, not the algebra.
"""
import numpy as np
import numpy.testing as npt
from scipy.linalg import expm

from dmipy_fit.signal_models.exchange_models import X0GeneralizedKarger
from dmipy_fit.signal_models.gaussian_models import G1Ball
from dmipy_fit.core.acquisition_scheme import AcquisitionScheme


# ---------------------------------------------------------------------------
# Fixture parameters (distinct per-compartment D, T2, T1)
# ---------------------------------------------------------------------------
D1, D2 = 1.0e-9, 2.0e-9       # m^2/s  — intra / extra diffusivity
T2_1, T2_2 = 0.05, 0.08       # s      — intra / extra transverse relaxation
T1_1, T1_2 = 1.0, 1.5         # s      — intra / extra longitudinal relaxation
F = 0.4                        # intra volume fraction
KAPPA = 5.0                    # s^-1   — intra->extra exchange rate
DELTA = 6e-3                   # s      — encoding lobe duration
TM = 40e-3                     # s      — mixing (storage) time
INF = 1e10                     # s      — "no relaxation" sentinel used by the model

B = np.array([0.0, 1e9, 2e9])  # s/m^2
BVECS = np.tile([1.0, 0.0, 0.0], (len(B), 1))


def _model():
    return X0GeneralizedKarger(G1Ball(), G1Ball())


def _kwargs(T1_1_=T1_1, T1_2_=T1_2, T2_1_=T2_1, T2_2_=T2_2):
    """Full combined-namespace kwargs; per-compartment T1/T2 are read by the
    exchange model from the ``G1Ball_<i>_T{1,2}`` keys."""
    return dict(
        G1Ball_1_lambda_iso=D1, G1Ball_2_lambda_iso=D2,
        f=F, kappa=KAPPA,
        G1Ball_1_T2=T2_1_, G1Ball_2_T2=T2_2_,
        G1Ball_1_T1=T1_1_, G1Ball_2_T1=T1_2_,
    )


def _Kmat():
    """2x2 exchange matrix with detailed balance kee = kappa*f/(1-f)."""
    kee = KAPPA * F / (1.0 - F)
    return np.array([[-KAPPA, kee], [KAPPA, -kee]], dtype=float)


def _hand_se(b, dt1, dt2, T1_1_, T1_2_, T2_1_, T2_2_):
    """Instantaneous-pulse SE Karger propagator, per measurement.

    Free precession splits into two intervals dt1, dt2 straddling the 180; each
    carries transverse relaxation and half the b-weighting.  T1 is absent from a
    spin echo (no storage window)."""
    K = _Kmat()
    RT2 = np.diag([1.0 / T2_1_, 1.0 / T2_2_])
    RD = np.diag([D1, D2])
    M0 = np.array([F, 1.0 - F])
    out = []
    for bv in b:
        if bv < 1e3:
            out.append(1.0)
            continue
        B1 = B2 = bv / 2.0
        P1 = expm((K - RT2) * dt1 - B1 * RD)
        P2 = expm((K - RT2) * dt2 - B2 * RD)
        out.append(float(np.sum(P2 @ P1 @ M0)))
    return np.array(out)


def _hand_ste(b, delta, tm, dt6, T1_1_, T1_2_, T2_1_, T2_2_,
              t2_in_mixing=False):
    """Instantaneous-pulse STE Karger propagator, per measurement.

    Chronology: encode (delta, transverse), store, mix (TM, longitudinal),
    recall, encode (dt6, transverse).  The mixing window carries T1 and exchange
    only; set ``t2_in_mixing=True`` to (incorrectly) let T2 accrue during TM, used
    to show the model gates it out.  Includes the 0.5 stimulated-echo factor."""
    K = _Kmat()
    RT2 = np.diag([1.0 / T2_1_, 1.0 / T2_2_])
    RT1 = np.diag([1.0 / T1_1_, 1.0 / T1_2_])
    RD = np.diag([D1, D2])
    M0 = np.array([F, 1.0 - F])
    R_mix = (K - RT1 - RT2) if t2_in_mixing else (K - RT1)
    out = []
    for bv in b:
        if bv < 1e3:
            out.append(1.0)
            continue
        B1 = B2 = bv / 2.0
        P_enc1 = expm((K - RT2) * delta - B1 * RD)
        P_mix = expm(R_mix * tm)
        P_enc2 = expm((K - RT2) * dt6 - B2 * RD)
        out.append(float(0.5 * np.sum(P_enc2 @ P_mix @ P_enc1 @ M0)))
    return np.array(out)


# ---------------------------------------------------------------------------
# PGSE (spin echo): T2 over the full echo, no T1 dependence
# ---------------------------------------------------------------------------

def test_pgse_karger_matches_hand_se_propagator():
    """PGSE two-compartment signal reproduces the hand SE propagator with
    distinct per-compartment T2 (and T1, which must not matter)."""
    TE = 2.0 * DELTA + TM
    scheme = AcquisitionScheme.from_pgse(
        B, BVECS, delta=DELTA, Delta=DELTA + TM, TE=TE)
    E = np.asarray(_model()(scheme, **_kwargs()))
    # 180 sits at TE/2; instantaneous pulses -> dt1 = dt2 = TE/2.
    expected = _hand_se(B, TE / 2.0, TE / 2.0,
                        T1_1, T1_2, T2_1, T2_2)
    npt.assert_allclose(E, expected, atol=1e-9)


def test_pgse_karger_independent_of_T1():
    """No mixing time -> T1 is inert: two wildly different T1 pairs give the
    identical PGSE signal."""
    scheme = AcquisitionScheme.from_pgse(
        B, BVECS, delta=DELTA, Delta=DELTA + TM, TE=2.0 * DELTA + TM)
    m = _model()
    E_a = np.asarray(m(scheme, **_kwargs(T1_1_=0.3, T1_2_=0.9)))
    E_b = np.asarray(m(scheme, **_kwargs(T1_1_=INF, T1_2_=INF)))
    npt.assert_allclose(E_a, E_b, atol=1e-12)


# ---------------------------------------------------------------------------
# PGSTE (stimulated echo): T1 over TM, T2 gated to the encoding lobes
# ---------------------------------------------------------------------------

def test_pgste_karger_matches_hand_ste_propagator():
    """PGSTE signal (both encoding lobes transverse) reproduces the hand STE
    propagator with distinct per-compartment T1 and T2.

    A plain ``from_pgste`` scheme is now correct out of the box: the second encoding
    lobe carries its own transverse time dt6 = delta (the geometry), not a value
    reconstructed from TE.
    """
    scheme = AcquisitionScheme.from_pgste(B, BVECS, delta=DELTA, TM=TM)
    npt.assert_allclose(scheme.TM, TM)
    E = np.asarray(_model()(scheme, **_kwargs()))
    dt6 = DELTA                                 # second encoding lobe = delta
    expected = _hand_ste(B, DELTA, TM, dt6, T1_1, T1_2, T2_1, T2_2)
    npt.assert_allclose(E, expected, atol=1e-9)


def test_pgste_default_te_encodes_both_lobes():
    """With the corrected from_pgste convention the echo time is the full
    stimulated-echo history TE = 2*delta + TM and the transverse occupancy is
    tau_perp = 2*delta, so the propagator's second encoding lobe carries its own
    transverse time dt6 = delta (both lobes are encoded), not a clamped-to-zero
    single lobe as under the old TE = 2*delta convention."""
    scheme = AcquisitionScheme.from_pgste(B, BVECS, delta=DELTA, TM=TM)
    npt.assert_allclose(scheme.TE, 2.0 * DELTA + TM)
    npt.assert_allclose(scheme.tau_perp, 2.0 * DELTA)
    E = np.asarray(_model()(scheme, **_kwargs()))
    expected = _hand_ste(B, DELTA, TM, DELTA, T1_1, T1_2, T2_1, T2_2)
    npt.assert_allclose(E, expected, atol=1e-9)


def test_pgste_longitudinal_weighting_over_mixing_time():
    """Equal per-compartment T1 factors out of the mixing propagator as the
    scalar exp(-TM/T1) (it commutes with the exchange matrix), so PGSTE applies
    exactly that longitudinal weighting over the mixing time.

    Checked on the diffusion-weighted measurements; the exchange model forces the
    b0 signal to 1.0 (it operates on normalised attenuation), so b0 is excluded.
    """
    scheme = AcquisitionScheme.from_pgste(B, BVECS, delta=DELTA, TM=TM)
    m = _model()
    T1 = 1.1
    E_T1 = np.asarray(m(scheme, **_kwargs(T1_1_=T1, T1_2_=T1)))
    E_noT1 = np.asarray(m(scheme, **_kwargs(T1_1_=INF, T1_2_=INF)))
    dw = B > 1e3
    npt.assert_allclose(E_T1[dw], np.exp(-TM / T1) * E_noT1[dw], atol=1e-12)
    # T1 -> inf leaves no longitudinal loss (only exchange during the window).
    assert np.all(E_noT1[dw] > E_T1[dw])


def test_pgste_transverse_relaxation_gated_to_encoding():
    """The mixing window carries T1 and exchange but NO transverse relaxation:
    the model matches the hand propagator whose mixing factor excludes T2, and is
    strictly larger than the (counterfactual) variant that lets T2 accrue over TM.
    """
    scheme = AcquisitionScheme.from_pgste(B, BVECS, delta=DELTA, TM=TM)
    # T1 -> inf isolates the transverse behaviour of the mixing window.
    E = np.asarray(_model()(scheme, **_kwargs(T1_1_=INF, T1_2_=INF)))
    dt6 = DELTA                                 # second encoding lobe = delta
    gated = _hand_ste(B, DELTA, TM, dt6, INF, INF, T2_1, T2_2,
                      t2_in_mixing=False)
    ungated = _hand_ste(B, DELTA, TM, dt6, INF, INF, T2_1, T2_2,
                        t2_in_mixing=True)
    npt.assert_allclose(E, gated, atol=1e-9)
    dw = B > 1e3
    assert np.all(E[dw] > ungated[dw])
