"""Independent CPU checks for the reference-catalogue audit, batch 2.

The audit IDs (E1-E3, F1-F3) and the published
equations are in docs/catalog_audit.md. No network, external files or reference
checkouts are used. Every reference signal below is written here from the
paper's equations with NumPy/SciPy only. Predictions and tolerances are stated
in each test's docstring; they were written before the factories were changed.
A test that asserts the *published* behaviour and fails on the factory is a
strict xfail naming the finding.

Conventions: SI units (b in s/m^2, diffusivities in m^2/s, lengths in m).
Fractions are signal fractions.
"""

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import erf, gamma as gamma_fn, jn_zeros, jnp_zeros, jv, jvp
from scipy.stats import gamma as gamma_dist

from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy_fit.custom_optimizers import reference_models as rm

DA = 1.7e-9
DCSF = 3.0e-9
DIN = 1.58e-9


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _unit(mu):
    theta, phi = mu
    return np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi), np.cos(theta)])


def _directions(n, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1)[:, None]


def _scheme(b=(1e9, 2e9, 3e9), per_shell=20, delta=0.0106, Delta=0.0431,
            TE=None, b0=False):
    dirs = _directions(per_shell * len(b))
    bvals = np.repeat(np.asarray(b, dtype=float), per_shell)
    if b0:
        dirs = np.vstack([[[1.0, 0.0, 0.0]], dirs])
        bvals = np.r_[0.0, bvals]
    return acquisition_scheme_from_bvalues(
        bvals, dirs, delta=delta, Delta=Delta, TE=TE)


def _stick_oriented(scheme, mu, d):
    return np.exp(-scheme.bvalues * d * (scheme.gradient_directions @ mu) ** 2)


def _gpd_sphere(q, delta, Delta, diameter, D, n_roots=60):
    """Gaussian-phase signal of a sphere, SANDI Eq. 8 (Palombo 2020):
    exp{-2 ((gamma g)^2 / D) sum_m a_m^-4 / (a_m^2 r^2 - 2) *
        [2 delta - (2 + e^{-a^2 D (Delta-delta)} - 2 e^{-a^2 D delta}
                    - 2 e^{-a^2 D Delta} + e^{-a^2 D (Delta+delta)})/(a^2 D)]}
    with a_m r the roots of J_{3/2}(x)/x = J_{5/2}(x), and gamma g = 2 pi q /
    delta for the 1/m q-value of a pulse of duration delta."""
    r = diameter / 2.0
    f = lambda x: jv(1.5, x) / x - jv(2.5, x)  # noqa: E731
    xs = np.linspace(1e-3, 3.0 * n_roots + 10, 400000)
    fv = f(xs)
    idx = np.where(np.sign(fv[:-1]) != np.sign(fv[1:]))[0][:n_roots]
    roots = np.array([brentq(f, xs[i], xs[i + 1]) for i in idx])
    a2 = (roots / r) ** 2
    gg2 = (2 * np.pi * np.asarray(q) / delta) ** 2
    bracket = (2 * delta - (2 + np.exp(-a2 * D * (Delta - delta))
                            - 2 * np.exp(-a2 * D * delta)
                            - 2 * np.exp(-a2 * D * Delta)
                            + np.exp(-a2 * D * (Delta + delta))) / (a2 * D))
    s = (a2 ** -2 / (a2 * r ** 2 - 2) * bracket).sum()
    return np.exp(-2 * gg2 / D * s)


def _callaghan_cylinder_perp(q, radius, D, Delta, n_max=30, k_max=60):
    """Perpendicular restricted signal of a cylinder, short-pulse limit,
    Assaf 2008 Eq. 3. The bracket of the printed equation is a square (the
    printed form does not reach 1 at q -> 0; the squared Callaghan form
    does)."""
    x = 2 * np.pi * q * radius
    # n = 0: beta_00 = 0 (free disc term), then the roots of J_0' = -J_1
    b0 = np.concatenate([[0.0], jnp_zeros(0, k_max)])
    e = 0.0
    for beta in b0:
        e += 4 * np.exp(-beta ** 2 * D * Delta / radius ** 2) \
            * (x * jvp(0, x) / (x ** 2 - beta ** 2)) ** 2
    for n in range(1, n_max + 1):
        for beta in jnp_zeros(n, k_max):
            e += 8 * np.exp(-beta ** 2 * D * Delta / radius ** 2) \
                * beta ** 2 / (beta ** 2 - n ** 2) \
                * (x * jvp(n, x) / (x ** 2 - beta ** 2)) ** 2
    return e


# ---------------------------------------------------------------------------
# E1  charmed  (Assaf & Basser 2005; Eqs. 1-3 as restated by Assaf 2008)
# ---------------------------------------------------------------------------

def _perp_scheme(delta=5e-4, Delta=0.03, b=(1e9, 3e9, 6e9), per_shell=6):
    """Gradients perpendicular to z (mu = [0, 0]), small delta so that the
    short-pulse limit of Eq. 3 applies."""
    rng = np.random.default_rng(3)
    ang = rng.uniform(0, 2 * np.pi, per_shell * len(b))
    dirs = np.stack([np.cos(ang), np.sin(ang), 0 * ang], 1)
    bvals = np.repeat(np.asarray(b, dtype=float), per_shell)
    return acquisition_scheme_from_bvalues(
        bvals, dirs, delta=delta, Delta=Delta)


def test_e1_restricted_part_equals_callaghan_eq3():
    """Assaf 2008 Eq. 3 (short-pulse cylinder, perpendicular gradient), the
    sum over (n, k) written independently, against the factory's cylinder
    alone (fraction 1 restricted, perpendicular gradients).

    Prediction: agreement to atol 1e-9 (the factory's diffusion time tau is
    Delta, as in Eq. 3; measured 6e-16).
    """
    scheme = _perp_scheme()
    diam = 4e-6
    q = np.asarray(scheme.qvalues)
    ref = np.array([_callaghan_cylinder_perp(qq, diam / 2, DA, 0.03)
                    for qq in q])
    e = rm.charmed()(scheme, **{
        "C3CylinderCallaghanApproximation_1_mu": [0.0, 0.0],
        "C3CylinderCallaghanApproximation_1_lambda_par": 1.5e-9,
        "C3CylinderCallaghanApproximation_1_diameter": diam,
        "partial_volume_0": 1.0, "partial_volume_1": 0.0})
    np.testing.assert_allclose(e, ref, atol=1e-9)


def test_e1_composite_is_population_weighted_hindered_plus_restricted():
    """Assaf 2008 Eq. 1: E = f_h E_h + f_r E_r, with E_h Gaussian and
    E_r the cylinder of Eq. 3. For perpendicular gradients E_h = exp(-b D_h)
    with D_h = lambda_perp = lambda_par (1 - f_r) (the factory's tortuosity).

    Prediction: agreement to atol 1e-9 (measured 3e-16).
    """
    scheme = _perp_scheme()
    diam, lpar, fr = 4e-6, 1.5e-9, 0.6
    q = np.asarray(scheme.qvalues)
    er = np.array([_callaghan_cylinder_perp(qq, diam / 2, DA, 0.03)
                   for qq in q])
    eh = np.exp(-scheme.bvalues * lpar * (1 - fr))
    e = rm.charmed()(scheme, **{
        "C3CylinderCallaghanApproximation_1_mu": [0.0, 0.0],
        "C3CylinderCallaghanApproximation_1_lambda_par": lpar,
        "C3CylinderCallaghanApproximation_1_diameter": diam,
        "partial_volume_0": fr, "partial_volume_1": 1 - fr})
    np.testing.assert_allclose(e, fr * er + (1 - fr) * eh, atol=1e-9)


def test_e1_eq3_reference_is_normalised_at_zero_q():
    """The reference itself: Eq. 3 (squared Callaghan form) is 1 at q -> 0.

    Prediction: 1 to 1e-6 at q = 1e-3 m^-1.
    """
    np.testing.assert_allclose(
        _callaghan_cylinder_perp(1e-3, 2e-6, DA, 0.03), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# E2  axcaliber  (Assaf 2008, Eqs. 1-5)
# ---------------------------------------------------------------------------

_G = "DD1GammaDistributed_1_"


def _gamma_cylinder_reference(scheme, mu, lpar, alpha, beta_radius, Delta,
                              D=DA, n_radii=80):
    """Eq. 4 for any gradient direction: the Gamma of Eq. 5 on the RADIUS a
    (the printed variable), weighted by the cylinder area pi a^2 (spins), by
    quadrature on a fine radius grid; each cylinder is the Eq. 3 perpendicular
    signal at q sin(theta) times the Gaussian parallel decay."""
    mean = alpha * beta_radius
    sd = np.sqrt(alpha) * beta_radius
    a = np.linspace(max(1e-8, mean - 5 * sd), mean + 8 * sd, n_radii)
    wts = gamma_dist.pdf(a, alpha, scale=beta_radius) * np.pi * a ** 2
    wts /= wts.sum()
    cos = scheme.gradient_directions @ _unit(mu)
    sin = np.sqrt(1 - cos ** 2)
    out = np.empty(len(cos))
    for i, (qq, s) in enumerate(zip(scheme.qvalues, sin)):
        e = np.array([_callaghan_cylinder_perp(qq * s, ai, D, Delta, n_max=8,
                                               k_max=20) for ai in a])
        out[i] = (wts * e).sum()
    return out * np.exp(-scheme.bvalues * lpar * cos ** 2)


def test_e2_gamma_distributed_restricted_signal_matches_eq4_eq5():
    """Any gradient direction, f_r = 1: the factory's Gamma-distributed
    cylinders (alpha, beta) against Eq. 4 with the Eq. 5 Gamma of the same
    (alpha, beta) on the cylinder RADIUS a, area-weighted. (The factory draws
    the radius from the Gamma and sets the cylinder diameter to twice the
    draw, so the mean diameter is 2 alpha beta.)

    Prediction: agreement to atol 1e-2 (the factory's 30-step quadrature of
    the Gamma; measured 2.1e-3).
    """
    scheme = _scheme(b=(1e9, 3e9), per_shell=30, delta=5e-4, Delta=0.03)
    alpha, beta, mu = 4.0, 0.6e-6, [0.6, 0.4]
    ref = _gamma_cylinder_reference(scheme, mu, 1.5e-9, alpha, beta, 0.03)
    e = rm.axcaliber()(scheme, **{
        _G + "C3CylinderCallaghanApproximation_1_mu": mu,
        _G + "C3CylinderCallaghanApproximation_1_lambda_par": 1.5e-9,
        _G + "DD1Gamma_1_alpha": alpha, _G + "DD1Gamma_1_beta": beta,
        "G2Zeppelin_1_lambda_par": 1.5e-9, "G2Zeppelin_1_lambda_perp": 0.8e-9,
        "partial_volume_0": 1.0, "partial_volume_1": 0.0})
    np.testing.assert_allclose(e, ref, atol=1e-2)


def test_e2_hindered_part_is_stejskal_tanner_with_a_free_dh():
    """Eq. 2: E_h = exp(-gamma^2 delta^2 g^2 D_h (Delta - delta/3)) = exp(-b
    D_h) for the perpendicular gradients of the AxCaliber protocol, D_h free.

    Prediction: with f_r -> 0 the signal is exp(-b lambda_perp) to 1e-9, and
    lambda_perp is a free parameter of the factory.
    """
    scheme = _perp_scheme()
    assert "G2Zeppelin_1_lambda_perp" in rm.axcaliber().parameter_names
    e = rm.axcaliber()(scheme, **{
        _G + "C3CylinderCallaghanApproximation_1_mu": [0.0, 0.0],
        _G + "C3CylinderCallaghanApproximation_1_lambda_par": 1.5e-9,
        _G + "DD1Gamma_1_alpha": 4.0, _G + "DD1Gamma_1_beta": 0.6e-6,
        "G2Zeppelin_1_mu": [0.0, 0.0],
        "G2Zeppelin_1_lambda_par": 1.5e-9, "G2Zeppelin_1_lambda_perp": 0.8e-9,
        "partial_volume_0": 0.0, "partial_volume_1": 1.0})
    np.testing.assert_allclose(e, np.exp(-scheme.bvalues * 0.8e-9),
                               atol=1e-9)


# ---------------------------------------------------------------------------
# E3  active_ax  (Alexander 2010; model per Dyrby 2013 / Sepehrband 2016)
# ---------------------------------------------------------------------------

_C = "C3CylinderCallaghanApproximation_1_"


def test_e3_restricted_and_hindered_share_one_orientation():
    """The minimal model of white matter diffusion has one fibre direction:
    the intra-axonal cylinders and the extra-axonal hindered tensor are
    coaxial.

    Prediction: the factory has a single orientation parameter.
    """
    mus = [n for n in rm.active_ax().parameter_names if n.endswith("_mu")]
    assert len(mus) == 1, mus


def test_e3_extra_axonal_perpendicular_diffusivity_is_tortuous():
    """Extra-axonal hindrance perpendicular to the axons by tortuosity
    (Dyrby 2013, Sepehrband 2016: d_perp = d_par v_ec / (v_ic + v_ec)), and
    the same d_par for both compartments.

    Prediction: the free parameters contain neither a zeppelin d_perp nor a
    second parallel diffusivity; and for perpendicular gradients with
    fractions (0.4, 0.4, 0.2) the signal is 0.4 E_cyl (Eq. 3) + 0.4 exp(-b
    d_par 0.5) + 0.2 exp(-b 3e-9) to atol 1e-9 (the cylinder is Eq. 3 as in E1).
    """
    names = rm.active_ax().parameter_names
    assert "G2Zeppelin_1_lambda_perp" not in names
    assert "G2Zeppelin_1_lambda_par" not in names
    scheme = _perp_scheme(b=(3e9,), per_shell=4)
    lpar, diam = 1.5e-9, 3e-6
    e = rm.active_ax()(scheme, **{
        _C + "mu": [0.0, 0.0], _C + "lambda_par": lpar, _C + "diameter": diam,
        "partial_volume_0": 0.4, "partial_volume_1": 0.4,
        "partial_volume_2": 0.2})
    ecyl = np.array([_callaghan_cylinder_perp(qq, diam / 2, DA, 0.03)
                     for qq in scheme.qvalues])
    ref = 0.4 * ecyl + 0.4 * np.exp(-scheme.bvalues * lpar * 0.5) \
        + 0.2 * np.exp(-scheme.bvalues * DCSF)
    np.testing.assert_allclose(e, ref, atol=1e-9)


def test_e3_csf_ball_is_fixed_at_free_water_diffusivity():
    """The CSF compartment has a fixed diffusivity (3e-9 m^2/s).

    Prediction: no free ball diffusivity.
    """
    assert "G1Ball_1_lambda_iso" not in rm.active_ax().parameter_names


# ---------------------------------------------------------------------------
# F1  verdict  (Panagiotaki 2014; abstract only)
# ---------------------------------------------------------------------------

def test_f1_is_sphere_plus_stick_plus_ball():
    """Composition arithmetic (the abstract describes intracellular,
    extracellular and vascular compartments): S = f_s E_sphere + f_v E_stick +
    f_e exp(-b D_e), the sphere by the Gaussian-phase form of Eq. 8 in the
    package's sphere diffusivity.

    Prediction: agreement to atol 1e-4 with a PGSE scheme.
    """
    scheme = _scheme(b=(0.5e9, 1.5e9, 3e9), per_shell=8)
    model = rm.verdict()
    d_s = model.models[0].diffusion_constant
    diam, dv, de, mu = 12e-6, 6e-9, 1.8e-9, [0.8, 1.1]
    fs, fv, fe = 0.3, 0.2, 0.5
    q = np.asarray(scheme.qvalues)
    es = np.array([_gpd_sphere(qq, scheme.delta[0], scheme.Delta[0], diam, d_s)
                   for qq in q])
    e = model(scheme, **{
        "S4SphereGaussianPhaseApproximation_1_diameter": diam,
        "C1Stick_1_mu": mu, "C1Stick_1_lambda_par": dv,
        "G1Ball_1_lambda_iso": de,
        "partial_volume_0": fs, "partial_volume_1": fv,
        "partial_volume_2": fe})
    ref = fs * es + fv * _stick_oriented(scheme, _unit(mu), dv) \
        + fe * np.exp(-scheme.bvalues * de)
    np.testing.assert_allclose(e, ref, atol=1e-4)


# ---------------------------------------------------------------------------
# F2  sandi  (Palombo 2020, Eqs. 4-10)
# ---------------------------------------------------------------------------

def _sandi_shell_reference(scheme, d_in, d_ec, r_diam, f_in, f_ec, d_is):
    """Eq. 10 on the shell-averaged signal: (1 - f_ec) (f_in A_in + (1 - f_in)
    A_is) + f_ec exp(-b D_ec); A_in the powder-averaged stick (Eq. 7), A_is
    the Gaussian-phase sphere (Eq. 8)."""
    out = []
    for bb in np.unique(scheme.bvalues):
        i = np.where(scheme.bvalues == bb)[0][0]
        x = bb * d_in
        a_in = np.sqrt(np.pi) * erf(np.sqrt(x)) / (2 * np.sqrt(x))
        a_is = _gpd_sphere(scheme.qvalues[i], scheme.delta[i],
                           scheme.Delta[i], r_diam, d_is)
        out.append((1 - f_ec) * (f_in * a_in + (1 - f_in) * a_is)
                   + f_ec * np.exp(-bb * d_ec))
    return np.array(out)


def _sandi_scheme():
    return _scheme(b=(0.5e9, 2e9, 4e9, 6e9), per_shell=12, delta=0.006,
                   Delta=0.020)


def _sandi_call(model, scheme, d_in, d_ec, r_diam, f_in, f_ec):
    f_is = 1 - f_in
    return model(scheme, **{
        "S4SphereGaussianPhaseApproximation_1_diameter": r_diam,
        "C1Stick_1_lambda_par": d_in, "G1Ball_1_lambda_iso": d_ec,
        "partial_volume_0": (1 - f_ec) * f_is,
        "partial_volume_1": (1 - f_ec) * f_in, "partial_volume_2": f_ec})


def test_f2_has_the_published_free_parameters():
    """Eq. 10 with D_is fixed: free f_in, f_ec, D_in, D_ec and r_s; no
    orientation (the model is the direction-averaged signal).

    Prediction: the soma diameter, the neurite diffusivity, the extracellular
    diffusivity and three signal fractions; no orientation parameter.
    """
    names = set(rm.sandi().parameter_names)
    assert names == {
        "S4SphereGaussianPhaseApproximation_1_diameter",
        "C1Stick_1_lambda_par", "G1Ball_1_lambda_iso",
        "partial_volume_0", "partial_volume_1", "partial_volume_2"}


def test_f2_matches_eq10_with_soma_diffusivity_3_um2_per_ms():
    """Eq. 10 per shell with D_is fixed to the free-water value ~3 um^2/ms
    (Sec. 2.2.3 last paragraph), D_in free, soma diameter 12 um.

    Prediction: agreement to atol 1e-4.
    """
    scheme = _sandi_scheme()
    args = (1.2e-9, 1.1e-9, 12e-6, 0.45, 0.35)
    ref = _sandi_shell_reference(scheme, *args, d_is=3.0e-9)
    e = _sandi_call(rm.sandi(), scheme, *args)
    np.testing.assert_allclose(e, ref, atol=1e-4)


def test_f2_jax_spherical_mean_forward_matches_the_numpy_model():
    """The JAX forward of the spherical-mean sandi() (what solver="jax" fits)
    equals its NumPy evaluation per shell.

    Prediction: agreement to rtol 1e-6 in float64.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from dmipy_fit.jax.multicompartment_jax import build_mc_sm_forward_fn
    with jax.enable_x64(True):
        model, scheme = rm.sandi(), _sandi_scheme()
        params = {
            "S4SphereGaussianPhaseApproximation_1_diameter": 12e-6,
            "C1Stick_1_lambda_par": 1.2e-9, "G1Ball_1_lambda_iso": 1.1e-9,
            "partial_volume_0": 0.33, "partial_volume_1": 0.32,
            "partial_volume_2": 0.35}
        fn = build_mc_sm_forward_fn(model, scheme, broadcast=False)
        got = np.asarray(fn(jnp.asarray(
            model.parameters_to_parameter_vector(**params))))
    np.testing.assert_allclose(got, model(scheme, **params), rtol=1e-6)


# ---------------------------------------------------------------------------
# F3  impulsed  (Xu 2019, Eq. 1 and Sec. "IMPULSED outcomes")
# ---------------------------------------------------------------------------

def test_f3_is_v_in_sphere_plus_ball_with_din_1_58():
    """Eq. 1: S = v_in S_in + (1 - v_in) S_ex, S_ex = exp(-b D_ex), D_ex free;
    cells impermeable spheres with D_in fixed to 1.58 um^2/ms. For PGSE the
    sphere is the Gaussian-phase form (the OGSE forms are in the supplement,
    not read).

    Prediction: agreement to atol 1e-4; no free D_in.
    """
    scheme = _scheme(b=(0.25e9, 1e9, 1.8e9), per_shell=6, delta=0.012,
                     Delta=0.074)
    model = rm.impulsed()
    assert not any("diffusion" in n or "lambda_par" in n
                   for n in model.parameter_names)
    diam, dex, vin = 14e-6, 2.0e-9, 0.55
    q = np.asarray(scheme.qvalues)
    es = np.array([_gpd_sphere(qq, scheme.delta[0], scheme.Delta[0], diam,
                               DIN) for qq in q])
    e = model(scheme, **{
        "S4SphereGaussianPhaseApproximation_1_diameter": diam,
        "G1Ball_1_lambda_iso": dex,
        "partial_volume_0": vin, "partial_volume_1": 1 - vin})
    ref = vin * es + (1 - vin) * np.exp(-scheme.bvalues * dex)
    np.testing.assert_allclose(e, ref, atol=1e-4)
