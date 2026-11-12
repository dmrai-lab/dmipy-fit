"""Independent CPU checks for the reference-catalogue audit, batch 1.

The audit IDs (B1-B5, C1-C4) and the published equations are in
docs/catalog_audit.md. No network, external files or reference checkouts are
used. Every reference signal below is written here from the paper's equations
with NumPy/SciPy only. Predictions and tolerances are stated in each test's
docstring. A defect the audit found is fixed and tested against the paper's
equations. A documented design difference (NODDI/Bingham-NODDI signal-averaged
extracellular compartment, the Zeppelin tissue of free_water_elimination, the
dispersion-free special case of stick_tortuous_zeppelin) is asserted as such,
with its measured size.

Conventions: SI units (b in s/m^2, diffusivities in m^2/s). Fractions are
signal fractions.
"""

import numpy as np
import pytest
from scipy.optimize import least_squares
from scipy.special import dawsn, erf

from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy_fit.custom_optimizers import reference_models as rm
from dmipy_fit.distributions.distributions import (
    SD2Bingham, kappa2odi, odi2kappa,
)

DA = 1.7e-9
DISO = 3.0e-9


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _scheme(b=(1e9, 2e9, 3e9), per_shell=20, seed=0):
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(per_shell * len(b), 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    bvals = np.repeat(np.asarray(b, dtype=float), per_shell)
    return acquisition_scheme_from_bvalues(
        bvals, dirs, delta=0.0106, Delta=0.0431)


def _unit(mu):
    theta, phi = mu
    return np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi), np.cos(theta)])


def _sphere_grid(n=96):
    """Gauss-Legendre in cos(theta) x uniform phi: nodes (N,3), weights sum 4pi."""
    x, wx = np.polynomial.legendre.leggauss(n)
    ph = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    xx, pp = np.meshgrid(x, ph, indexing="ij")
    s = np.sqrt(1 - xx ** 2)
    nodes = np.stack([s * np.cos(pp), s * np.sin(pp), xx], -1).reshape(-1, 3)
    w = (wx[:, None] * np.ones_like(pp)).reshape(-1) * (2 * np.pi / (2 * n))
    return nodes, w


def _watson_weights(nodes, w, kappa, mu_c):
    """Watson ODF (Zhang 2012 Eq. 3) as normalised quadrature weights."""
    dens = np.exp(kappa * (nodes @ mu_c) ** 2) * w
    return dens / dens.sum()


def _tau1(kappa):
    """Zhang 2012 Eq. 8: <cos^2 psi> over the Watson ODF."""
    return -1 / (2 * kappa) + 1 / (2 * dawsn(np.sqrt(kappa)) * np.sqrt(kappa))


def _stick_dispersed(scheme, nodes, weights, d):
    """Zhang 2012 Eq. 2 / Tariq 2016 Eq. 3: ODF-average of stick attenuation."""
    c = scheme.gradient_directions @ nodes.T
    return (np.exp(-scheme.bvalues[:, None] * d * c ** 2) * weights).sum(1)


def _zeppelin_signal_average(scheme, nodes, weights, dpar, dperp):
    """ODF average of the per-orientation zeppelin SIGNAL (what the factory does)."""
    c = scheme.gradient_directions @ nodes.T
    b = scheme.bvalues[:, None]
    return (np.exp(-b * (dperp + (dpar - dperp) * c ** 2)) * weights).sum(1)


def _zeppelin_tensor_average(scheme, nodes, weights, dpar, dperp):
    """Zhang 2012 Eq. 4 / Tariq 2016 Eqs. 4-5 / Jelescu 2016 Eq. 2:
    exp(-b g^T <D> g) with <D> the ODF average of the TENSOR."""
    nn = (weights[:, None, None] * nodes[:, :, None] * nodes[:, None, :]).sum(0)
    dmean = dperp * np.eye(3) + (dpar - dperp) * nn
    g = scheme.gradient_directions
    return np.exp(-scheme.bvalues * np.einsum("ij,jk,ik->i", g, dmean, g))


# ---------------------------------------------------------------------------
# B1  ball_and_stick  (Behrens 2003, MRM 50, Eq. 12-13)
# ---------------------------------------------------------------------------

def test_b1_matches_behrens_eq12():
    """Eq. 12: S = (1-f) exp(-b d) + f exp(-b d (g.n)^2), one shared d.

    Prediction: the factory equals Eq. 12 to rtol 1e-12 (both closed forms in
    float64), with d the single diffusivity parameter.
    """
    scheme = _scheme()
    f, d, mu = 0.4, 1.3e-9, [0.8, -0.5]
    model = rm.ball_and_stick()
    e = model(scheme, C1Stick_1_mu=mu, C1Stick_1_lambda_par=d,
              partial_volume_0=f, partial_volume_1=1 - f)
    c = scheme.gradient_directions @ _unit(mu)
    ref = (1 - f) * np.exp(-scheme.bvalues * d) \
        + f * np.exp(-scheme.bvalues * d * c ** 2)
    np.testing.assert_allclose(e, ref, rtol=1e-12, atol=1e-14)


def test_b1_has_a_single_shared_diffusivity():
    """Behrens 2003 Eq. 12 has ONE diffusivity d shared by the stick and the
    isotropic part.

    Prediction: the only diffusivity parameter is C1Stick_1_lambda_par.
    """
    names = rm.ball_and_stick().parameter_names
    assert [n for n in names if "lambda" in n] == ["C1Stick_1_lambda_par"]


# ---------------------------------------------------------------------------
# B2  ball_and_zeppelin  (Panagiotaki 2012; full text not read)
# ---------------------------------------------------------------------------

def test_b2_is_a_zeppelin_plus_ball_sum():
    """Two Gaussian compartments: f_z exp(-b g^T D g) + (1-f_z) exp(-b d_iso).

    Prediction: the factory equals this sum to rtol 1e-12. This checks the
    composition arithmetic only; whether the Panagiotaki 2012 taxonomy
    contains this pair is NOT verified (see docs/catalog_audit.md).
    """
    scheme = _scheme()
    fz, lp, lq, diso, mu = 0.65, 1.8e-9, 0.6e-9, 2.1e-9, [1.1, 0.3]
    e = rm.ball_and_zeppelin()(
        scheme, G2Zeppelin_1_mu=mu, G2Zeppelin_1_lambda_par=lp,
        G2Zeppelin_1_lambda_perp=lq, G1Ball_1_lambda_iso=diso,
        partial_volume_0=fz, partial_volume_1=1 - fz)
    c = scheme.gradient_directions @ _unit(mu)
    b = scheme.bvalues
    ref = fz * np.exp(-b * (lq + (lp - lq) * c ** 2)) \
        + (1 - fz) * np.exp(-b * diso)
    np.testing.assert_allclose(e, ref, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# B3  stick_tortuous_zeppelin  (Novikov 2019, Eqs. 3.4-3.5)
# ---------------------------------------------------------------------------

def test_b3_equals_standard_model_kernel_under_the_tortuosity_constraint():
    """Eq. 3.5 with f_CSF=0, De_par=Da, De_perp=Da(1-f), delta-function ODF.

    Prediction: factory == kernel to rtol 1e-12.
    """
    scheme = _scheme()
    f, da, mu = 0.55, 2.0e-9, [0.6, 1.0]
    e = rm.stick_tortuous_zeppelin()(
        scheme, C1Stick_1_mu=mu, C1Stick_1_lambda_par=da,
        partial_volume_0=f, partial_volume_1=1 - f)
    xi = scheme.gradient_directions @ _unit(mu)
    b = scheme.bvalues
    ref = f * np.exp(-b * da * xi ** 2) \
        + (1 - f) * np.exp(-b * da * (1 - f) - b * (da - da * (1 - f)) * xi ** 2)
    np.testing.assert_allclose(e, ref, rtol=1e-12, atol=1e-14)


def test_b3_is_the_dispersion_free_tortuosity_constrained_special_case():
    """Documented design: the Standard Model kernel (Eq. 3.5) has four free
    scalars and an ODF; the factory is its special case with De_par = Da,
    De_perp = Da(1-f) and a delta-function ODF, and says so.

    Prediction: one diffusivity parameter, no ODF/dispersion parameter, and the
    docstring names it a special case, not the Standard Model.
    """
    model = rm.stick_tortuous_zeppelin()
    names = model.parameter_names
    assert [n for n in names if "lambda" in n] == ["C1Stick_1_lambda_par"]
    assert not any(k in n for n in names for k in ("odi", "kappa", "sh_coeff"))
    doc = " ".join(rm.stick_tortuous_zeppelin.__doc__.split())
    assert "special case of the Standard Model" in doc
    assert "not the Standard Model itself" in doc


# ---------------------------------------------------------------------------
# B4  free_water_elimination  (Pasternak 2009; abstract + secondary only)
# ---------------------------------------------------------------------------

def test_b4_free_water_ball_is_fixed_at_3e_minus_3_mm2_per_s():
    """Free-water diffusivity 3.0e-3 mm^2/s, not a free parameter.

    Prediction: a Zeppelin with equal eigenvalues D plus fraction f of free
    water gives (1-f) exp(-b D) + f exp(-b 3e-9) to rtol 1e-12, and no
    diffusivity of the Ball appears among the parameters.
    """
    scheme = _scheme()
    f_tissue, d = 0.7, 0.9e-9
    model = rm.free_water_elimination()
    assert "G1Ball_1_lambda_iso" not in model.parameter_names
    e = model(scheme, G2Zeppelin_1_mu=[0.3, 0.2],
              G2Zeppelin_1_lambda_par=d, G2Zeppelin_1_lambda_perp=d,
              partial_volume_0=f_tissue, partial_volume_1=1 - f_tissue)
    ref = f_tissue * np.exp(-scheme.bvalues * d) \
        + (1 - f_tissue) * np.exp(-scheme.bvalues * DISO)
    np.testing.assert_allclose(e, ref, rtol=1e-12, atol=1e-14)


def test_b4_zeppelin_tissue_cannot_represent_a_general_tensor():
    """Documented difference (the package has no full-tensor Gaussian
    compartment): Pasternak 2009's tissue compartment is a full diffusion
    tensor; the factory's is an axially symmetric Zeppelin.

    Generate free-water-plus-full-tensor data and refit the factory.

    Tensor eigenvalues (1.7, 0.9, 0.4)e-9 m^2/s (distinct), free-water
    fraction 0.2, shells b = 1, 2, 3e9. A Zeppelin cannot represent two
    different radial eigenvalues.

    Prediction: the best Zeppelin fit leaves an rms residual above 1e-2
    (measured 4.4e-2): the difference the factory's docstring states.
    """
    scheme = _scheme(per_shell=30)
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    tensor = q @ np.diag([1.7e-9, 0.9e-9, 0.4e-9]) @ q.T
    g = scheme.gradient_directions
    f_tissue = 0.8
    data = f_tissue * np.exp(-scheme.bvalues * np.einsum(
        "ij,jk,ik->i", g, tensor, g)) \
        + (1 - f_tissue) * np.exp(-scheme.bvalues * DISO)
    model = rm.free_water_elimination()
    axis = q[:, 0]
    theta0, phi0 = np.arccos(axis[2]), np.arctan2(axis[1], axis[0])

    def residual(p):
        e = model(scheme, G2Zeppelin_1_mu=[p[0], p[1]],
                  G2Zeppelin_1_lambda_par=p[2] * 1e-9,
                  G2Zeppelin_1_lambda_perp=p[3] * 1e-9,
                  partial_volume_0=p[4], partial_volume_1=1 - p[4])
        return e - data

    best = np.inf
    for lp0, lq0 in ((1.7, 0.65), (1.5, 0.9), (1.2, 0.4)):
        sol = least_squares(
            residual, [theta0, phi0, lp0, lq0, 0.8],
            bounds=([0, -np.pi, 0.1, 0.1, 0.01], [np.pi, np.pi, 3, 3, 0.99]))
        best = min(best, np.sqrt(np.mean(sol.fun ** 2)))
    assert best > 1e-2
    assert "full diffusion tensor" in " ".join(
        rm.free_water_elimination.__doc__.split())


# ---------------------------------------------------------------------------
# B5  ivim  (Le Bihan 1988; primary text not read)
# ---------------------------------------------------------------------------

def test_b5_is_a_biexponential_with_fixed_pseudodiffusion():
    """S/S0 = (1-f) exp(-b D) + f exp(-b D*) with D* = 7e-9 fixed.

    The biexponential form is the standard IVIM statement; it was not
    verified against the 1988 text. Prediction: the factory equals it to
    rtol 1e-12; the only free diffusivity is the tissue Ball with bounds
    [0.5e-9, 6e-9]; the blood Ball diffusivity is not a parameter.
    """
    scheme = _scheme(b=(0, 50e6, 200e6, 800e6), per_shell=3)
    f, d = 0.12, 0.8e-9
    model = rm.ivim()
    assert [n for n in model.parameter_names if "lambda" in n] == [
        "G1Ball_1_lambda_iso"]
    lo, hi = np.asarray(model.parameter_ranges["G1Ball_1_lambda_iso"]) \
        * model.parameter_scales["G1Ball_1_lambda_iso"]
    np.testing.assert_allclose([lo, hi], [0.5e-9, 6e-9], rtol=1e-12)
    e = model(scheme, G1Ball_1_lambda_iso=d,
              partial_volume_0=1 - f, partial_volume_1=f)
    ref = (1 - f) * np.exp(-scheme.bvalues * d) \
        + f * np.exp(-scheme.bvalues * 7e-9)
    np.testing.assert_allclose(e, ref, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# C1  noddi  (Zhang 2012, Eqs. 1-9)
# ---------------------------------------------------------------------------

_W = "SD1WatsonDistributed_1_"


def _noddi_params(odi, nu, fiso, mu):
    return {_W + "SD1Watson_1_mu": mu, _W + "SD1Watson_1_odi": odi,
            _W + "partial_volume_0": nu,
            "partial_volume_0": 1 - fiso, "partial_volume_1": fiso}


def test_c1_odi_is_zhang_eq9():
    """OD = (2/pi) arctan(1/kappa) (Eq. 9); kappa = 1/tan(OD pi/2).

    Prediction: round trip to 1e-12; kappa(0.5) = 1.
    """
    odi = np.array([0.05, 0.3, 0.5, 0.9])
    np.testing.assert_allclose(kappa2odi(odi2kappa(odi)), odi, rtol=1e-12)
    np.testing.assert_allclose(odi2kappa(0.5), 1.0, rtol=1e-12)


def test_c1_tau1_closed_form_is_the_watson_mean_cos2():
    """Eq. 7/8 reference self-check: tau1 equals the Watson <cos^2>.

    Prediction: agreement to 1e-9 against 96x192 quadrature; tau1(0)->1/3.
    """
    nodes, w = _sphere_grid()
    for kappa in (0.3, 2.0, 16.0):
        wt = _watson_weights(nodes, w, kappa, np.array([0, 0, 1.0]))
        np.testing.assert_allclose(_tau1(kappa), wt @ nodes[:, 2] ** 2,
                                   rtol=1e-9)
    assert abs(_tau1(1e-4) - 1 / 3) < 1e-4


@pytest.mark.parametrize("odi", [0.1, 0.3, 0.7])
def test_c1_composition_matches_eq1_with_signal_averaged_extracellular(odi):
    """Eq. 1 with d_par = 1.7e-9 and d_iso = 3e-9 fixed, d_perp = d_par(1-nu),
    intracellular Eq. 2 by quadrature, and the extracellular compartment
    written as the ODF average of the per-orientation zeppelin SIGNAL.

    Prediction: factory agrees with this reference to atol 1.5e-3 (spherical
    harmonic truncation at the lowest shell order, measured <= 3e-4).
    """
    scheme = _scheme()
    nu, fiso, mu = 0.6, 0.2, [0.7, 0.4]
    nodes, w = _sphere_grid()
    wt = _watson_weights(nodes, w, odi2kappa(odi), _unit(mu))
    a_in = _stick_dispersed(scheme, nodes, wt, DA)
    a_ec = _zeppelin_signal_average(scheme, nodes, wt, DA, DA * (1 - nu))
    ref = (1 - fiso) * (nu * a_in + (1 - nu) * a_ec) \
        + fiso * np.exp(-scheme.bvalues * DISO)
    e = rm.noddi()(scheme, **_noddi_params(odi, nu, fiso, mu))
    np.testing.assert_allclose(e, ref, atol=1.5e-3)


@pytest.mark.parametrize("odi,size", [(0.1, 4.5e-3), (0.3, 6.9e-3),
                                      (0.7, 7.3e-3)])
def test_c1_extracellular_differs_from_zhang_eq4_by_the_documented_size(
        odi, size):
    """Design difference (documented in noddi()): Zhang 2012 Eq. 4 is the
    exponential of the ODF-averaged TENSOR (apparent d'_par, d'_perp, Eqs.
    5-8); the factory averages the per-orientation zeppelin SIGNALS, for the
    optimizer's nested volume fractions.

    Full NODDI signal per Eqs. 1, 2, 4 (tensor average via quadrature).
    Prediction: the maximum deviation is 4.5e-3, 6.9e-3, 7.3e-3 of S0 at ODI
    0.1, 0.3, 0.7 (rtol 5 %), far above the 1.5e-3 agreement of the
    signal-averaged reference.
    """
    scheme = _scheme()
    nu, fiso, mu = 0.6, 0.2, [0.7, 0.4]
    nodes, w = _sphere_grid()
    wt = _watson_weights(nodes, w, odi2kappa(odi), _unit(mu))
    a_in = _stick_dispersed(scheme, nodes, wt, DA)
    a_ec = _zeppelin_tensor_average(scheme, nodes, wt, DA, DA * (1 - nu))
    ref = (1 - fiso) * (nu * a_in + (1 - nu) * a_ec) \
        + fiso * np.exp(-scheme.bvalues * DISO)
    e = rm.noddi()(scheme, **_noddi_params(odi, nu, fiso, mu))
    np.testing.assert_allclose(np.abs(e - ref).max(), size, rtol=0.05)
    doc = " ".join(rm.noddi.__doc__.split())
    assert "Zhang et al. 2012 Eq. 4" in doc and "4.5e-3 to 7.3e-3" in doc


def test_c1_free_parameters_are_the_published_set():
    """Published free set: nu_ic, kappa(ODI), mu (2), nu_iso; d_par, d_iso fixed.

    Prediction: factory parameter names are exactly mu, odi, the bundle
    stick fraction and the two top-level fractions (nu_iso = 1 - tissue
    fraction); no diffusivity is free.
    """
    names = set(rm.noddi().parameter_names)
    assert names == {
        _W + "SD1Watson_1_mu", _W + "SD1Watson_1_odi",
        _W + "partial_volume_0", "partial_volume_0", "partial_volume_1"}


# ---------------------------------------------------------------------------
# C2  bingham_noddi  (Tariq 2016, Eqs. 3-5, 10, 14-16)
# ---------------------------------------------------------------------------

_B = "SD2BinghamDistributed_1_"


def _bingham_params(odi, beta_fraction, psi, nu, fiso, mu):
    return {_B + "SD2Bingham_1_mu": mu, _B + "SD2Bingham_1_psi": psi,
            _B + "SD2Bingham_1_odi": odi,
            _B + "SD2Bingham_1_beta_fraction": beta_fraction,
            _B + "partial_volume_0": nu,
            "partial_volume_0": 1 - fiso, "partial_volume_1": fiso}


def _bingham_weights(nodes, w, odi, beta_fraction, psi, mu):
    """The factory's own Bingham density (Eq. 10), renormalised on the grid.

    Using the factory's density isolates the compartment equations from the
    (unverified) psi/axis convention of the angle parametrisation.
    """
    dens = SD2Bingham()(nodes, mu=mu, psi=psi, odi=odi,
                        beta_fraction=beta_fraction) * w
    return dens / dens.sum()


def test_c2_reduces_to_watson_noddi_when_beta_is_zero():
    """Tariq 2016: Watson is the beta = 0 special case.

    Prediction: bingham_noddi(beta_fraction=0) equals noddi() to atol 1.5e-3.
    """
    scheme = _scheme()
    odi, nu, fiso, mu = 0.3, 0.6, 0.2, [0.7, 0.4]
    ew = rm.noddi()(scheme, **_noddi_params(odi, nu, fiso, mu))
    eb = rm.bingham_noddi()(scheme, **_bingham_params(
        odi, 0.0, 0.0, nu, fiso, mu))
    np.testing.assert_allclose(eb, ew, atol=1.5e-3)


@pytest.mark.parametrize("beta_fraction", [0.5, 0.9])
def test_c2_composition_matches_eq1_with_signal_averaged_extracellular(
        beta_fraction):
    """Eq. 1/3 composition with the extracellular zeppelin signal averaged.

    Prediction: factory agrees with the quadrature reference to atol 2e-3
    (Bingham SH truncation is larger than Watson's; unmeasured, stated here
    before running).
    """
    scheme = _scheme()
    odi, psi, nu, fiso, mu = 0.3, 0.4, 0.6, 0.2, [0.7, 0.4]
    nodes, w = _sphere_grid()
    wt = _bingham_weights(nodes, w, odi, beta_fraction, psi, mu)
    a_in = _stick_dispersed(scheme, nodes, wt, DA)
    a_ec = _zeppelin_signal_average(scheme, nodes, wt, DA, DA * (1 - nu))
    ref = (1 - fiso) * (nu * a_in + (1 - nu) * a_ec) \
        + fiso * np.exp(-scheme.bvalues * DISO)
    e = rm.bingham_noddi()(scheme, **_bingham_params(
        odi, beta_fraction, psi, nu, fiso, mu))
    np.testing.assert_allclose(e, ref, atol=2e-3)


@pytest.mark.parametrize("beta_fraction,size", [(0.5, 7.3e-3), (0.9, 7.4e-3)])
def test_c2_extracellular_differs_from_tariq_eq4_by_the_documented_size(
        beta_fraction, size):
    """Design difference (documented in bingham_noddi()): Tariq 2016 Eqs. 4-5
    define the extra-neurite signal as exp(-b g^T D_en g) with D_en the
    ODF-averaged tensor; the factory averages the per-orientation zeppelin
    signals (as noddi()).

    Full Bingham-NODDI signal per Eqs. 1, 3, 4, 5.
    Prediction: the maximum deviation is 7.3e-3 and 7.4e-3 of S0 (rtol 5 %).
    """
    scheme = _scheme()
    odi, psi, nu, fiso, mu = 0.3, 0.4, 0.6, 0.2, [0.7, 0.4]
    nodes, w = _sphere_grid()
    wt = _bingham_weights(nodes, w, odi, beta_fraction, psi, mu)
    a_in = _stick_dispersed(scheme, nodes, wt, DA)
    a_ec = _zeppelin_tensor_average(scheme, nodes, wt, DA, DA * (1 - nu))
    ref = (1 - fiso) * (nu * a_in + (1 - nu) * a_ec) \
        + fiso * np.exp(-scheme.bvalues * DISO)
    e = rm.bingham_noddi()(scheme, **_bingham_params(
        odi, beta_fraction, psi, nu, fiso, mu))
    np.testing.assert_allclose(np.abs(e - ref).max(), size, rtol=0.05)
    doc = " ".join(rm.bingham_noddi.__doc__.split())
    assert "Tariq et al. 2016 Eqs. 4-5" in doc and "7.3e-3 and 7.4e-3" in doc


def test_c2_kappa_upper_bound_is_64():
    """Tariq 2016, Model fitting procedure: kappa, beta <= 64.

    Prediction: the lower odi bound maps to kappa = 64 (rtol 1e-9; beta <=
    kappa by construction of beta_fraction), and the signal at that bound
    equals the quadrature reference with signal-averaged extracellular to
    atol 1e-3 (measured 8e-4).
    """
    name = _B + "SD2Bingham_1_odi"
    model = rm.bingham_noddi()
    lo = np.asarray(model.parameter_ranges[name])[0] \
        * model.parameter_scales[name]
    np.testing.assert_allclose(odi2kappa(lo), 64.0, rtol=1e-9)
    scheme = _scheme()
    psi, nu, fiso, mu = 0.4, 0.6, 0.2, [0.7, 0.4]
    nodes, w = _sphere_grid()
    wt = _bingham_weights(nodes, w, lo, 0.0, psi, mu)
    ref = (1 - fiso) * (nu * _stick_dispersed(scheme, nodes, wt, DA)
                        + (1 - nu) * _zeppelin_signal_average(
                            scheme, nodes, wt, DA, DA * (1 - nu))) \
        + fiso * np.exp(-scheme.bvalues * DISO)
    e = model(scheme, **_bingham_params(lo, 0.0, psi, nu, fiso, mu))
    np.testing.assert_allclose(e, ref, atol=1e-3)


# ---------------------------------------------------------------------------
# C3  noddida  (Jelescu 2016, NMR Biomed 29:33, Eqs. 2-5)
# ---------------------------------------------------------------------------

_N = "SD1WatsonDistributed_1_"


def _noddida_params(f, da, dpar, dperp, odi, mu):
    return {_N + "C1Stick_1_lambda_par": da,
            _N + "G2Zeppelin_1_lambda_par": dpar,
            _N + "G2Zeppelin_1_lambda_perp": dperp,
            _N + "SD1Watson_1_mu": mu, _N + "SD1Watson_1_odi": odi,
            _N + "partial_volume_0": f}


def _noddida_reference(scheme, odi, f, da, dpar, dperp, mu, extracellular):
    nodes, w = _sphere_grid()
    wt = _watson_weights(nodes, w, odi2kappa(odi), _unit(mu))
    return f * _stick_dispersed(scheme, nodes, wt, da) \
        + (1 - f) * extracellular(scheme, nodes, wt, dpar, dperp)


def test_c3_has_the_published_free_parameters_and_bounds():
    """Jelescu 2016 Eq. 2: free f, Da, De_par, De_perp, kappa and mu; no CSF
    compartment; no tortuosity link; diffusivity bounds [0, 4] um^2/ms.

    Prediction: exactly those parameters, no Ball, and each diffusivity bound
    equals [0, 4e-9] m^2/s.
    """
    model = rm.noddida()
    assert set(model.parameter_names) == {
        _N + "C1Stick_1_lambda_par", _N + "G2Zeppelin_1_lambda_par",
        _N + "G2Zeppelin_1_lambda_perp", _N + "SD1Watson_1_mu",
        _N + "SD1Watson_1_odi", _N + "partial_volume_0"}
    assert not any("G1Ball" in n for n in model.parameter_names)
    for name in (_N + "C1Stick_1_lambda_par", _N + "G2Zeppelin_1_lambda_par",
                 _N + "G2Zeppelin_1_lambda_perp"):
        bounds = np.asarray(model.parameter_ranges[name]) \
            * model.parameter_scales[name]
        np.testing.assert_allclose(bounds, [0.0, 4e-9], atol=1e-20)


@pytest.mark.parametrize("odi", [0.1, 0.3, 0.7])
def test_c3_matches_eq2_with_signal_averaged_extracellular(odi):
    """Eq. 2 with independent Da, De_par, De_perp and the Watson ODF by
    quadrature, the extracellular compartment written as the ODF average of
    the per-orientation zeppelin SIGNAL (the bundle design of noddi()).

    Prediction: the factory agrees to atol 1.5e-3 (measured <= 7e-5).
    """
    scheme = _scheme()
    f, da, dpar, dperp, mu = 0.5, 2.1e-9, 1.4e-9, 0.6e-9, [0.7, 0.4]
    ref = _noddida_reference(scheme, odi, f, da, dpar, dperp, mu,
                             _zeppelin_signal_average)
    e = rm.noddida()(scheme, **_noddida_params(f, da, dpar, dperp, odi, mu))
    np.testing.assert_allclose(e, ref, atol=1.5e-3)


@pytest.mark.parametrize("odi,size", [(0.1, 6.0e-3), (0.3, 9.3e-3),
                                      (0.7, 9.8e-3)])
def test_c3_extracellular_differs_from_eq2_by_the_documented_size(odi, size):
    """Design difference (documented in noddida() and noddi()): Eq. 2 is the
    exponential of the ODF-averaged tensor; the factory averages zeppelin
    signals.

    Prediction: for f 0.5, Da 2.1, De_par 1.4, De_perp 0.6 um^2/ms the maximum
    deviation is 6.0e-3, 9.3e-3, 9.8e-3 of S0 at ODI 0.1, 0.3, 0.7 (rtol 5 %).
    """
    scheme = _scheme()
    f, da, dpar, dperp, mu = 0.5, 2.1e-9, 1.4e-9, 0.6e-9, [0.7, 0.4]
    ref = _noddida_reference(scheme, odi, f, da, dpar, dperp, mu,
                             _zeppelin_tensor_average)
    e = rm.noddida()(scheme, **_noddida_params(f, da, dpar, dperp, odi, mu))
    np.testing.assert_allclose(np.abs(e - ref).max(), size, rtol=0.05)


def test_c3_zero_diffusivities_are_valid():
    """The paper's lower bound is 0: the signal at Da = 0 is finite and the
    stick is then unattenuated.

    Prediction: finite values; with f = 1 the signal is 1 to 1e-9.
    """
    scheme = _scheme()
    e = rm.noddida()(scheme, **_noddida_params(
        1.0, 0.0, 1.4e-9, 0.6e-9, 0.3, [0.7, 0.4]))
    assert np.isfinite(e).all()
    np.testing.assert_allclose(e, 1.0, atol=1e-9)


def test_c3b_tortuosity_with_a_third_compartment():
    """Szafer/Bruggeman: lambda_perp = lambda_par * vf_extra / (vf_intra +
    vf_extra) (the helper's own docstring and unit test intend this).

    Prediction: with (0.4, 0.35) the helper returns 0.4667 lambda_par (it
    returned 0.3962 lambda_par before the fix of the normalisation).
    """
    from dmipy_fit.utils.utils import T1_tortuosity
    lpar = 1.9e-9
    got = T1_tortuosity()(lpar, 0.4, 0.35)
    np.testing.assert_allclose(got, lpar * 0.35 / 0.75, rtol=1e-9)


# ---------------------------------------------------------------------------
# C4  mcsmt  (Kaden 2016, NeuroImage 139, Eqs. 4-8)
# ---------------------------------------------------------------------------

def _smt_reference(b, v, lam):
    """Kaden 2016 Eqs. 6-8 with lambda_perp = (1 - v) lambda."""
    lam_perp = (1 - v) * lam
    e_int = np.sqrt(np.pi) * erf(np.sqrt(b * lam)) / (2 * np.sqrt(b * lam))
    x = b * (lam - lam_perp)
    e_ext = np.exp(-b * lam_perp) * np.sqrt(np.pi) * erf(np.sqrt(x)) \
        / (2 * np.sqrt(x))
    return v * e_int + (1 - v) * e_ext


@pytest.mark.parametrize("v,lam", [(0.3, 1.2e-9), (0.6, 2.0e-9),
                                   (0.85, 2.8e-9)])
def test_c4_matches_kaden_eqs_6_to_8(v, lam):
    """Spherical-mean intra + tortuous extra signal.

    Prediction: factory == Eq. 6 with Eqs. 7-8 per shell to rtol 1e-9
    (erf closed forms; tolerance allows a quadrature-based implementation).
    """
    scheme = _scheme(b=(0.5e9, 1e9, 2e9, 3e9, 5e9), per_shell=10)
    e = rm.mcsmt()(scheme, C1Stick_1_lambda_par=lam,
                   partial_volume_0=v, partial_volume_1=1 - v)
    shells = np.unique(scheme.bvalues)
    ref = np.array([_smt_reference(bb, v, lam) for bb in shells])
    np.testing.assert_allclose(np.ravel(e)[:len(shells)], ref, rtol=1e-9)


def test_c4_free_parameters_are_v_int_and_lambda():
    """Published free set: v_int and lambda (Kaden 2016, Mean diffusion-signal
    model); lambda_perp and lambda_par of the extra space are linked.

    Prediction: names are C1Stick_1_lambda_par and the two partial volumes.
    """
    assert set(rm.mcsmt().parameter_names) == {
        "C1Stick_1_lambda_par", "partial_volume_0", "partial_volume_1"}


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

def test_citations_are_corrected():
    """Le Bihan 1988 is Radiology 168; Kaden 2016's authors are Kaden, Kelm,
    Carson, Does & Alexander; NODDIDA is Jelescu, Veraart, Fieremans &
    Novikov, NMR Biomed 29 (2016), in the docstrings and the catalogue table.
    """
    for text in (rm.__doc__, rm.ivim.__doc__):
        assert "Radiology 168" in text and "Radiology 161" not in text
    for text in (rm.__doc__ + rm.mcsmt.__doc__,):
        assert "Kaden, Kelm, Carson, Does & Alexander" in text \
            or "Kaden et al. 2016" in text
        assert "McKinstry" not in text and "Rueckert" not in text
    for text in (rm.noddida.__doc__,):
        assert "Jelescu" in text and "2016" in text
        assert "Adisetiyo" not in text and "2015" not in text
    assert "Jelescu et al. 2016, NMR Biomed 29" in rm.__doc__
