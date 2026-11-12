"""Batch 3 predictions, written before changing the factories.

Independent SciPy/NumPy equations; SI units. Sources/access and finding IDs
are in docs/catalog_audit.md. Strict xfails retain published assertions for
unfixed findings. No network, git or external data is used by these tests.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import erf, spherical_jn

from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy_fit.custom_optimizers import reference_models as rm


def scheme(b, delta, big_delta, te):
    rng = np.random.default_rng(3)
    dirs = rng.normal(size=(12 * len(b), 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    return acquisition_scheme_from_bvalues(
        np.repeat(b, 12), dirs, delta=delta, Delta=big_delta, TE=te)


def sphere(sch, diameter, diffusivity):
    """Murday-Cotts GPA sphere (Palombo Eq. 8), independent Neumann roots."""
    roots = np.array([brentq(lambda x: spherical_jn(1, x, derivative=True),
                            k * np.pi + 0.1, (k + 1) * np.pi)
                      for k in range(60)])
    a2 = (roots / (diameter / 2)) ** 2
    d, t = sch.delta[:, None], sch.Delta[:, None]
    z = a2 * diffusivity
    bracket = (2*d - (2 + np.exp(-z*(t-d)) - 2*np.exp(-z*d)
                      - 2*np.exp(-z*t) + np.exp(-z*(t+d))) / z)
    gg2 = sch.bvalues / (sch.delta**2 * (sch.Delta-sch.delta/3))
    return np.exp(-2 * gg2 / diffusivity *
                  np.sum(bracket / a2**2 / (roots**2 - 2), axis=1))


def karger(exponent1, exponent2, f, rate, time):
    """Olesen Eqs. 2, 4-6: detailed balance, equilibrium initial state."""
    reverse = rate * f / (1-f)
    return np.array([np.ones(2) @ expm(np.array([
        [-r1-rate*time, reverse*time],
        [rate*time, -r2-reverse*time]])) @ [f, 1-f]
        for r1, r2 in zip(exponent1, exponent2)])


def test_g2_matches_independent_karger_matrix():
    """Prediction: two balls equal the constant-q matrix exponential to 1e-12.

    Kärger 1985 abstract only; equation verified in Olesen 2022 Eqs. 2,4-6.
    kappa is pool-1 efflux; reverse rate is kappa*f/(1-f).
    """
    sch = scheme([0., 1e9, 3e9], .001, .04, None)
    p = {'X0GeneralizedKarger_1_G1Ball_1_lambda_iso': .5e-9,
         'X0GeneralizedKarger_1_G1Ball_2_lambda_iso': 2e-9,
         'X0GeneralizedKarger_1_f': .6, 'X0GeneralizedKarger_1_kappa': 12.}
    expected = karger(sch.bvalues*.5e-9, sch.bvalues*2e-9, .6, 12., .04-.001/3)
    np.testing.assert_allclose(rm.karger_two_compartment()(sch, **p), expected,
                               rtol=1e-12, atol=1e-12)


def sandix_params():
    return {'S4SphereGaussianPhaseApproximation_1_diameter': 12e-6,
            'X1NEXI_1_D_i': 1.2e-9, 'X1NEXI_1_D_e': 1e-9,
            'X1NEXI_1_f': .6, 'X1NEXI_1_t_ex': .004,
            'partial_volume_0': .2, 'partial_volume_1': .78,
            'partial_volume_2': .02}


def sandix_reference(sch, finite_pulse):
    """Olesen Sec. 2.1: soma + exchanging stick/ball + immobile offset."""
    nodes, w = np.polynomial.legendre.leggauss(64)
    u, w = (nodes+1)/2, w/2
    b, d, t = sch.bvalues, sch.delta[0], sch.Delta[0]
    f, tex = .6, .004
    k = (1-f)/tex
    if finite_pulse:
        # Integrate Eq. 3 over each ramp/plateau separately, avoiding a step
        # across a gradient breakpoint. q(t)^2 is in rad^2/m^2.
        exchange = np.array([[-k, k*f/(1-f)], [k, -k*f/(1-f)]])
        kernels = []
        for bb in b:
            gg2 = bb / (d*d*(t-d/3))
            states = np.tile([f, 1-f], (len(u), 1))
            for start, end, qtime in (
                    (0., d, lambda x: x), (d, t, lambda x: d),
                    (t, t+d, lambda x: t+d-x)):
                def rhs(x, y):
                    m = y.reshape(-1, 2)
                    decay = gg2*qtime(x)**2 * np.column_stack(
                        [1.2e-9*u*u, np.full(len(u), 1e-9)])
                    return (m @ exchange.T - decay*m).ravel()
                sol = solve_ivp(rhs, (start, end), states.ravel(),
                                rtol=1e-10, atol=1e-12)
                assert sol.success
                states = sol.y[:, -1].reshape(-1, 2)
            kernels.append(states.sum(axis=1) @ w)
        pair = np.array(kernels)
    else:
        pair = sum(ww * karger(b*1.2e-9*uu**2, b*1e-9, f, k, t-d/3)
                   for uu, ww in zip(u, w))
    return .2*sphere(sch, 12e-6, 2e-9) + .78*pair + .02


def test_g4_composition_and_narrow_pulse_kernel():
    """Prediction: no orientation; soma D0=2; free Dn; immobile offset.

    Olesen Sec. 2.1, Eqs. 1,3-6; coupled angular average to atol 1e-8.
    """
    sch = scheme([1e9, 3e9], .0005, .025, None)
    model = rm.sandix()
    assert set(model.parameter_names) == set(sandix_params())
    actual = model(sch, **sandix_params())
    ref = sandix_reference(sch, False)[::12]
    np.testing.assert_allclose(actual, ref, atol=1e-8, rtol=1e-8)


@pytest.mark.xfail(strict=True, reason='G4-finite-pulse: SANDIX still uses effective-time narrow-pulse exchange')
def test_g4_matches_published_finite_pulse_rate_equations():
    """Prediction: SANDIX equals the finite-pulse ODE, atol 1e-7 (Eq. 3)."""
    sch = scheme([1e9, 3e9], .012, .020, None)
    expected = sandix_reference(sch, True)[::12]
    np.testing.assert_allclose(rm.sandix()(sch, **sandix_params()), expected,
                               atol=1e-7, rtol=1e-7)


def impulsed_params(rate):
    return {'X0GeneralizedKarger_1_S4SphereGaussianPhaseApproximation_1_diameter': 14e-6,
            'X0GeneralizedKarger_1_G1Ball_1_lambda_iso': 2e-9,
            'X0GeneralizedKarger_1_f': .55, 'X0GeneralizedKarger_1_kappa': rate}


def test_g5_no_exchange_limit():
    """Prediction: k=0 mixture, Din=1.56 um2/ms (Shi Sec. 2.2), atol 1e-8.

    Updated from the preprint-era 1.58 assumption after reading the final
    2025 manuscript, before changing the factory's diffusion constant.
    """
    sch = scheme([.25e9, 1e9, 1.8e9], .012, .074, None)
    ref = .55*sphere(sch, 14e-6, 1.56e-9)+.45*np.exp(-sch.bvalues*2e-9)
    np.testing.assert_allclose(rm.exchange_impulsed()(sch, **impulsed_params(0.)),
                               ref, atol=1e-8, rtol=1e-8)


@pytest.mark.xfail(strict=True, reason='G5-EXCHANGE: missing Shi two-mode diffusion, edge correction and T=delta+Delta')
def test_g5_matches_shi_pgse_exchange():
    """Prediction: Shi 2025 Eqs. 1-5 for PGSE agree to atol 1e-7.

    Test is at nonzero b; Din=1.56 um2/ms, kin=10/s, d=14 um.
    """
    sch = scheme([.25e9, 1e9, 1.8e9], .012, .074, None)
    b, din, d, f, k = sch.bvalues, 1.56e-9, 14e-6, .55, 10.
    sr = sphere(sch, d, din)
    r = d/2
    tbar, tstar = (3*r/4)**2/(2*din), (4*r/3)**2/(2*din)
    p = tstar/(tstar + 1/k - tbar)
    eh = sr**f * np.exp(-b*(1-f)*2e-9)
    r1 = -np.log((1-p)*sr+p*eh)
    km = k*(1+2.39*(d*d*k/din)**.83*f**2.88)
    kr = k*f/(1-f)
    duration = .012+.074
    expected = np.array([np.ones(2) @ expm(np.array([
        [-a-km*duration, kr*duration],
        [km*duration, -bb*2e-9-kr*duration]])) @ [f, 1-f]
        for a, bb in zip(r1, b)])
    np.testing.assert_allclose(rm.exchange_impulsed()(sch, **impulsed_params(k)),
                               expected, atol=1e-7, rtol=1e-7)


def test_h1_perpendicular_log_time_dependence_and_csf():
    """Prediction: G3 changes Dperp, not Dpar; plus fixed 3e-9 CSF (1e-12).

    Novikov Sec. 2.4, Eq. 2.34; finite-pulse form Burcaw 2015 Eq. 24.
    """
    sch = scheme([1e9, 2e9], .006, .04, None)
    dp = .4e-9 + 2e-12*(np.log(.04/.006)+1.5)/(.04-.006/3)
    cos = sch.gradient_directions[:, 2]
    ref = .8*np.exp(-sch.bvalues*(1.7e-9*cos*cos+dp*(1-cos*cos)))
    ref += .2*np.exp(-sch.bvalues*3e-9)
    got = rm.temporal_zeppelin_model()(sch, **{
        'G3TemporalZeppelin_1_mu': [0., 0.],
        'G3TemporalZeppelin_1_lambda_par': 1.7e-9,
        'G3TemporalZeppelin_1_lambda_inf': .4e-9,
        'G3TemporalZeppelin_1_A': 2e-12,
        'partial_volume_0': .8, 'partial_volume_1': .2})
    np.testing.assert_allclose(got, ref, atol=1e-12, rtol=1e-12)


def test_i1_generic_two_pool_relaxation():
    """Prediction: generic stick+ball weights are f0*exp(-TE/T2) (1e-12).

    Gong Eq. 8 supports the relaxation algebra, not this named composition.
    """
    sch = scheme([0., 1e9], .006, .02, .08)
    p = {'OccupancyGatedModel_1_mu': [0., 0.],
         'OccupancyGatedModel_1_lambda_par': 1.7e-9,
         'OccupancyGatedModel_1_T2': .09,
         'OccupancyGatedModel_2_lambda_iso': 1e-9,
         'OccupancyGatedModel_2_T2': .05,
         'partial_volume_0': .6, 'partial_volume_1': .4}
    ref = .6*np.exp(-.08/.09-sch.bvalues*1.7e-9*sch.gradient_directions[:, 2]**2)
    ref += .4*np.exp(-.08/.05-sch.bvalues*1e-9)
    np.testing.assert_allclose(rm.mte_ball_stick()(sch, **p), ref, atol=1e-12)


def noddi_params():
    pre = 'SD1WatsonDistributed_1_'
    return {pre+'OccupancyGatedModel_1_T2': .09,
            pre+'OccupancyGatedModel_2_T2': .05,
            pre+'SD1Watson_1_mu': [0., 0.],
            pre+'SD1Watson_1_odi': .3, pre+'partial_volume_0': .6,
            'OccupancyGatedModel_1_T2': 1.,
            'partial_volume_0': .8, 'partial_volume_1': .2}


def noddi_reference(sch, averaged_tensor):
    # Independent quadrature of Watson p(n), axis z. Normalise by integral.
    u, w = np.polynomial.legendre.leggauss(80)
    phi = np.arange(160)*2*np.pi/160
    n = np.stack(np.broadcast_arrays(np.sqrt(1-u[:, None]**2)*np.cos(phi),
                 np.sqrt(1-u[:, None]**2)*np.sin(phi), u[:, None]), axis=-1).reshape(-1, 3)
    k = 1/np.tan(.3*np.pi/2)
    wt = np.repeat(w*np.exp(k*u*u), len(phi)); wt /= wt.sum()
    along = (sch.gradient_directions @ n.T)**2
    stick = np.exp(-sch.bvalues[:, None]*1.7e-9*along) @ wt
    diffusivity = 1.7e-9*(.4+.6*along)
    extra = (np.exp(-sch.bvalues*(diffusivity @ wt)) if averaged_tensor
             else np.exp(-sch.bvalues[:, None]*diffusivity) @ wt)
    return (.8*(.6*np.exp(-.08/.09)*stick+.4*np.exp(-.08/.05)*extra)
            + .2*np.exp(-.08/1.-sch.bvalues*3e-9))


def test_i2_preserves_documented_signal_average_and_t2_weights():
    """Prediction: C1 signal-average convention plus Gong Eq. 8, atol 2e-4."""
    sch = scheme([0., 1e9, 3e9], .006, .02, .08)
    np.testing.assert_allclose(rm.mte_noddi()(sch, **noddi_params()),
                               noddi_reference(sch, False), atol=2e-4, rtol=0)


@pytest.mark.xfail(strict=True, reason='I2-EA (C1 design choice): signal average retained instead of averaged-tensor Gaussian')
def test_i2_published_extracellular_gaussian():
    """Prediction: published NODDI extracellular Gaussian, atol 2e-4."""
    sch = scheme([0., 1e9, 3e9], .006, .02, .08)
    np.testing.assert_allclose(rm.mte_noddi()(sch, **noddi_params()),
                               noddi_reference(sch, True), atol=2e-4, rtol=0)


def sandi_params():
    return {'OccupancyGatedModel_1_diameter': 12e-6,
            'OccupancyGatedModel_1_T2': .047,
            'OccupancyGatedModel_2_lambda_par': 1.2e-9,
            'OccupancyGatedModel_2_T2': .088,
            'OccupancyGatedModel_3_lambda_iso': 1e-9,
            'OccupancyGatedModel_3_T2': .058,
            'partial_volume_0': .3, 'partial_volume_1': .4,
            'partial_volume_2': .3}


def test_i3_direction_average_free_dn_and_compartment_t2():
    """Prediction: SANDI base with 3 free T2, no mu, free Dn, Ds=3e-9.

    Full ISMRM abstract #0766 Methods: Si(0,TE)=Si0 exp(-TE/T2i).
    Palombo Eq. 10 direction-average, agreement to atol 2e-5.
    """
    sch = scheme([0., 1e9, 3e9], .008, .022, .063)
    model = rm.mte_sandi()
    assert set(model.parameter_names) == set(sandi_params())
    x = sch.bvalues*1.2e-9
    stick = np.ones_like(x)
    nz = x > 0
    stick[nz] = np.sqrt(np.pi)*erf(np.sqrt(x[nz]))/(2*np.sqrt(x[nz]))
    ref = .3*np.exp(-.063/.047)*sphere(sch, 12e-6, 3e-9)
    ref += .4*np.exp(-.063/.088)*stick
    ref += .3*np.exp(-.063/.058-sch.bvalues*1e-9)
    np.testing.assert_allclose(model(sch, **sandi_params()), ref[::12], atol=2e-5)


@pytest.mark.parametrize('factory,params', [(rm.sandix, sandix_params),
                                          (rm.mte_sandi, sandi_params)])
@pytest.mark.parametrize('broadcast', [False, True])
def test_spherical_mean_jax_matches_numpy(factory, params, broadcast):
    """Prediction: corrected spherical means remain JAX-fittable, atol 2e-5."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from dmipy_fit.jax.multicompartment_jax import build_mc_sm_forward_fn
    sch = scheme([0., 1e9, 3e9], .008, .022, .063)
    with jax.enable_x64(True):
        model = factory()
        forward = build_mc_sm_forward_fn(model, sch, broadcast=broadcast)
        got = np.asarray(forward(jnp.asarray(model.parameters_to_parameter_vector(**params()))))
    expected = model(sch, **params())
    if broadcast:
        expected = expected[sch.shell_indices]
    np.testing.assert_allclose(got, expected, atol=2e-5)


def test_i3_multiple_te_b0_is_compartment_relaxation_sum():
    """Prediction: at two TEs, both b0 shells obey the abstract's sum (1e-12).

    JAX and NumPy must also agree on nonzero b shells (2e-5).
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from dmipy_fit.jax.multicompartment_jax import build_mc_sm_forward_fn
    sch = scheme([0., 1e9, 3e9, 0., 1e9, 3e9], .008, .022,
                 np.repeat([.047, .047, .047, .127, .127, .127], 12))
    model = rm.mte_sandi()
    with jax.enable_x64(True):
        fn = build_mc_sm_forward_fn(model, sch, broadcast=False)
        got = np.asarray(fn(jnp.asarray(model.parameters_to_parameter_vector(**sandi_params()))))
    mask = sch.shell_b0_mask
    assert mask.sum() == 2
    te = sch.spherical_mean_scheme.TE[mask]
    expected = (.3*np.exp(-te/.047)+.4*np.exp(-te/.088)+.3*np.exp(-te/.058))
    np.testing.assert_allclose(got[mask], expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(got, model(sch, **sandi_params()), atol=2e-5)
