"""Independent CPU checks for the reference-catalogue audit, batch 1.

The audit IDs (B1-B5, C1-C4) and the published equations are in
docs/catalog_audit.md. No network, external files or reference checkouts are
used. Every reference signal below is written here from the paper's equations
with NumPy/SciPy only. Predictions and tolerances are stated in each test's
docstring. A test that asserts the *published* behaviour and fails on the
factory is a strict xfail naming the finding; the factory is not changed here.

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

def test_b1_matches_behrens_eq12_when_diffusivities_are_equal():
    """Eq. 12: S = (1-f) exp(-b d) + f exp(-b d (g.n)^2), one shared d.

    Prediction: with both factory diffusivities set to the same d the factory
    equals Eq. 12 to rtol 1e-12 (both closed forms in float64).
    """
    scheme = _scheme()
    f, d, mu = 0.4, 1.3e-9, [0.8, -0.5]
    model = rm.ball_and_stick()
    e = model(scheme, C1Stick_1_mu=mu, C1Stick_1_lambda_par=d,
              G1Ball_1_lambda_iso=d, partial_volume_0=f,
              partial_volume_1=1 - f)
    c = scheme.gradient_directions @ _unit(mu)
    ref = (1 - f) * np.exp(-scheme.bvalues * d) \
        + f * np.exp(-scheme.bvalues * d * c ** 2)
    np.testing.assert_allclose(e, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.xfail(strict=True, reason=(
    "CATALOG-AUDIT-B1: Behrens 2003 Eq. 12 has ONE diffusivity d shared by "
    "the stick and the isotropic part (six parameters: sigma, S0, d, f, "
    "theta, phi). ball_and_stick() frees the stick and ball diffusivities "
    "separately."))
def test_b1_has_a_single_shared_diffusivity():
    """Published model: exactly one diffusivity parameter.

    Prediction: fails; the factory exposes C1Stick_1_lambda_par and
    G1Ball_1_lambda_iso (two).
    """
    names = rm.ball_and_stick().parameter_names
    diffusivities = [n for n in names if "lambda" in n]
    assert len(diffusivities) == 1


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


@pytest.mark.xfail(strict=True, reason=(
    "CATALOG-AUDIT-B3: the Standard Model kernel (Eq. 3.5) has four "
    "independent scalars f, Da, De_par, De_perp; Novikov 2019 Sec. 3.4 "
    "treats tortuosity/equal-diffusivity as optional constraints. The "
    "factory imposes both (De_par=Da, De_perp=Da(1-f)), leaving one "
    "diffusivity."))
def test_b3_has_three_independent_diffusivities():
    """Published model: Da, De_par, De_perp all free.

    Prediction: fails; the factory exposes one diffusivity parameter.
    """
    names = rm.stick_tortuous_zeppelin().parameter_names
    assert len([n for n in names if "lambda" in n]) == 3


@pytest.mark.xfail(strict=True, reason=(
    "CATALOG-AUDIT-B3: the Standard Model convolves the kernel with an ODF "
    "P(n) (Eq. 3.4); the factory has a single orientation (delta ODF) and "
    "no dispersion or ODF parameter."))
def test_b3_has_an_orientation_distribution():
    """Published model: signal = integral P(n) K(b, g.n) dn.

    Prediction: fails; no parameter name mentions an ODF/dispersion
    (odi, kappa, ...).
    """
    names = rm.stick_tortuous_zeppelin().parameter_names
    assert any(k in n for n in names for k in ("odi", "kappa", "sh_coeff"))


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


@pytest.mark.xfail(strict=True, reason=(
    "CATALOG-AUDIT-B4: Pasternak 2009 fits a bi-tensor model whose tissue "
    "compartment is a full diffusion tensor (six parameters). The factory "
    "tissue compartment is an axially symmetric Zeppelin (two "
    "diffusivities, one axis)."))
def test_b4_tissue_compartment_represents_a_general_tensor():
    """Generate free-water-plus-full-tensor data and refit the factory.

    Tensor eigenvalues (1.7, 0.9, 0.4)e-9 m^2/s (distinct), free-water
    fraction 0.2, shells b = 1, 2, 3e9. A Zeppelin cannot represent two
    different radial eigenvalues.

    Prediction: the best Zeppelin fit leaves an rms residual above 1e-3 (my
    estimate 1e-2), so asserting rms < 1e-6 fails.
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
    assert best < 1e-6


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


@pytest.mark.xfail(strict=True, reason=(
    "CATALOG-AUDIT-C1: Zhang 2012 Eq. 4 defines the extracellular signal as "
    "exp(-b g^T (int f(n) D(n) dn) g), the exponential of the ODF-averaged "
    "TENSOR (apparent d'_par, d'_perp, Eqs. 5-8). The factory averages the "
    "per-orientation zeppelin SIGNALS. The two differ by up to ~0.02 of S0."))
@pytest.mark.parametrize("odi", [0.1, 0.3, 0.7])
def test_c1_extracellular_is_exponential_of_averaged_tensor(odi):
    """Full NODDI signal per Eqs. 1, 2, 4 (tensor average via quadrature).

    Prediction: fails at atol 1.5e-3; the largest deviation measured in a
    pre-run of the zeppelin alone was 8e-3 (ODI 0.05), 2e-2 (ODI 0.3, 0.7)
    before weighting by (1-f_iso)(1-nu) = 0.32.
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
    np.testing.assert_allclose(e, ref, atol=1.5e-3)


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


@pytest.mark.xfail(strict=True, reason=(
    "CATALOG-AUDIT-C2: Tariq 2016 Eqs. 4-5 define the extra-neurite signal "
    "as exp(-b g^T D_en g) with D_en the ODF-averaged tensor; the factory "
    "averages the per-orientation zeppelin signals (same defect as C1)."))
@pytest.mark.parametrize("beta_fraction", [0.5, 0.9])
def test_c2_extracellular_is_exponential_of_averaged_tensor(beta_fraction):
    """Full Bingham-NODDI signal per Eqs. 1, 3, 4, 5.

    Prediction: fails at atol 2e-3; deviation of the same order as C1
    (~1e-2 of S0 after weighting).
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
    np.testing.assert_allclose(e, ref, atol=2e-3)


@pytest.mark.xfail(strict=True, reason=(
    "CATALOG-AUDIT-C2: the Bingham-NODDI fit constrains kappa, beta <= 64 "
    "(Tariq 2016, Model fitting procedure). The factory's odi >= 0.02 caps "
    "kappa at about 31.8."))
def test_c2_kappa_upper_bound_is_64():
    """Published bound kappa <= 64 reachable through the odi range.

    Prediction: fails; odi2kappa(0.02) = 31.8 < 64.
    """
    lo = rm.bingham_noddi().parameter_ranges[_B + "SD2Bingham_1_odi"][0]
    assert odi2kappa(lo) >= 64


# ---------------------------------------------------------------------------
# C3  noddida  (Jelescu 2016, NMR Biomed 29:33, Eqs. 2-5)
# ---------------------------------------------------------------------------

def _noddida_closed_form(scheme, f, da, diso, mu, f_in):
    """f1 stick + f2 zeppelin(Da, De_perp = Da (1 - f_in)) + f3 ball."""
    xi = scheme.gradient_directions @ _unit(mu)
    b = scheme.bvalues
    stick = np.exp(-b * da * xi ** 2)
    zep = np.exp(-b * da * (1 - f_in) - b * da * f_in * xi ** 2)
    return f[0] * stick + f[1] * zep + f[2] * np.exp(-b * diso)


def test_c3_factory_is_the_aligned_special_case_without_the_ball():
    """Aligned (kappa -> inf) limit of Eq. 2 with Da = De_par, De_perp =
    Da(1-f), the ball fraction set to zero (fractions sum over the two
    tissue compartments only).

    Prediction: factory == closed form to rtol 1e-12. This documents what the
    factory is, not what NODDIDA is.
    """
    scheme = _scheme()
    f = np.array([0.45, 0.55, 0.0])
    da, diso, mu = 1.9e-9, 2.6e-9, [0.9, -1.2]
    e = rm.noddida()(
        scheme, C1Stick_1_mu=mu, C1Stick_1_lambda_par=da,
        G1Ball_1_lambda_iso=diso, partial_volume_0=f[0],
        partial_volume_1=f[1], partial_volume_2=f[2])
    ref = _noddida_closed_form(scheme, f, da, diso, mu, f[0] / (f[0] + f[1]))
    np.testing.assert_allclose(e, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.xfail(strict=True, reason=(
    "CATALOG-AUDIT-C3b: T1_tortuosity.__call__ normalises vf_intra by "
    "(vf_intra + vf_extra) but then uses the UNnormalised vf_extra, so for "
    "vf_intra + vf_extra != 1 it returns the wrong lambda_perp. Hit by "
    "noddida() whenever the ball fraction is non-zero. The existing unit "
    "test compares values of order 1e-9 at 7 decimals and cannot see it."))
def test_c3b_tortuosity_with_a_third_compartment():
    """Szafer/Bruggeman: lambda_perp = lambda_par * vf_extra / (vf_intra +
    vf_extra) (the helper's own docstring and unit test intend this).

    Prediction: fails; with (0.4, 0.35) the helper returns 0.3962 lambda_par
    instead of 0.4667 lambda_par (measured in the audit by inverting the
    factory signal; computed here from the helper's code path).
    """
    from dmipy_fit.utils.utils import T1_tortuosity
    lpar = 1.9e-9
    got = T1_tortuosity()(lpar, 0.4, 0.35)
    np.testing.assert_allclose(got, lpar * 0.35 / 0.75, rtol=1e-9)


@pytest.mark.xfail(strict=True, reason=(
    "CATALOG-AUDIT-C3b: consequence for noddida(): with a non-zero ball "
    "fraction the zeppelin's perpendicular diffusivity is Da * 0.396 instead "
    "of Da * 0.467 at fractions (0.4, 0.35, 0.25)."))
def test_c3b_noddida_signal_with_ball_fraction():
    """Closed form with De_perp = Da f_ec/(f_ic + f_ec).

    Prediction: fails; the signal differs by up to ~2e-2 (b <= 3e9).
    """
    scheme = _scheme()
    f = np.array([0.4, 0.35, 0.25])
    da, diso, mu = 1.9e-9, 2.6e-9, [0.9, -1.2]
    e = rm.noddida()(
        scheme, C1Stick_1_mu=mu, C1Stick_1_lambda_par=da,
        G1Ball_1_lambda_iso=diso, partial_volume_0=f[0],
        partial_volume_1=f[1], partial_volume_2=f[2])
    ref = _noddida_closed_form(scheme, f, da, diso, mu, f[0] / (f[0] + f[1]))
    np.testing.assert_allclose(e, ref, rtol=1e-9)


@pytest.mark.xfail(strict=True, reason=(
    "CATALOG-AUDIT-C3: NODDIDA releases Da, De_par and De_perp as three "
    "independent diffusivities (Jelescu 2016 Eq. 2, Methods). The factory "
    "ties De_par = Da and De_perp = Da(1-f) and so has one."))
def test_c3_has_three_independent_tissue_diffusivities():
    """Prediction: fails; the tissue diffusivity parameters are Da plus the
    free Ball's d_iso (two names, only one of them tissue)."""
    names = rm.noddida().parameter_names
    tissue = [n for n in names if "lambda" in n and "G1Ball" not in n]
    assert len(tissue) == 3


@pytest.mark.xfail(strict=True, reason=(
    "CATALOG-AUDIT-C3: NODDIDA keeps NODDI's Watson-dispersed neurite "
    "orientation distribution (concentration kappa, Eqs. 2-3). The factory "
    "has no dispersion: all neurites are parallel."))
def test_c3_has_a_watson_dispersion_parameter():
    """Prediction: fails; no odi/kappa parameter exists."""
    names = rm.noddida().parameter_names
    assert any(k in n for n in names for k in ("odi", "kappa"))


@pytest.mark.xfail(strict=True, reason=(
    "CATALOG-AUDIT-C3: the NODDIDA of Jelescu 2016 has no CSF compartment "
    "(f_iso = 0). The factory adds a free-diffusivity isotropic Ball."))
def test_c3_has_no_isotropic_compartment():
    """Prediction: fails; the factory contains G1Ball."""
    names = rm.noddida().parameter_names
    assert not any("G1Ball" in n for n in names)


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
