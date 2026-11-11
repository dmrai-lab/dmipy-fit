"""Independent CPU checks for the exchange-model correctness audit.

No network, external files, optional accelerators, or reference checkouts are
used by these tests. Predictions and tolerances are stated on each test.
Known failures retain the physical assertions (strict xfail); the audit IDs
are defined in docs/exchange_model_audit.md.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm
from scipy.special import erf

from dmipy_analytic import exchange
from dmipy_analytic.constants import GAMMA
from dmipy_fit.core.acquisition_scheme import (
    AcquisitionScheme, acquisition_scheme_from_bvalues,
)
from dmipy_fit.custom_optimizers.reference_models import fexi, nexi, stick_zeppelin_karger
from dmipy_fit.signal_models.filter_exchange import (
    FEXIApparentExchange, fexi_measurements,
)
from dmipy_fit.signal_models.nexi_reference import nexi_jelescu2022


def _generator(f, kappa):
    """Independent column-conserving two-site generator, s^-1."""
    back = kappa * f / (1-f)
    return np.array([[-kappa, back], [kappa, -back]])


def _pgse(b, directions, delta, big_delta):
    return acquisition_scheme_from_bvalues(
        np.asarray(b, dtype=float), np.asarray(directions, dtype=float),
        delta=delta, Delta=big_delta,
    )


def _factory_case(name, b, directions, delta, big_delta, f, kappa):
    scheme = _pgse(b, directions, delta, big_delta)
    prefix = "X0GeneralizedKarger_1_"
    parameters = {prefix + "f": f, prefix + "kappa": kappa}
    if name == "stick_zeppelin_karger":
        di, de = 2.1e-9, 1.4e-9
        parameters.update({prefix + "C1Stick_1_lambda_par": di,
                           prefix + "G2Zeppelin_1_lambda_par": de,
                           prefix + "mu": [0., 0.]})
        c2 = scheme.gradient_directions[:, 2] ** 2
        diffusivities = np.column_stack([di * c2, de * (c2 + (1-f) * (1-c2))])
        model = stick_zeppelin_karger()
    elif name == "fexi":
        parameters[prefix + "G1Ball_1_lambda_iso"] = 0.3e-9
        diffusivities = np.tile([0.3e-9, 2.e-9], (len(b), 1))
        model = fexi()
    else:
        raise ValueError(name)
    return model, scheme, parameters, diffusivities


def _finite_pgse_signal(b, delta, big_delta, diffusivities, f, kappa, steps):
    """Midpoint expm products for q ramp / plateau / ramp, independent of analytic code."""
    q_peak = np.sqrt(b / (big_delta - delta / 3))
    k = _generator(f, kappa)
    d = np.diag(diffusivities)
    m = np.array([f, 1-f])
    h = delta / steps
    for q in q_peak * (np.arange(steps) + 0.5) / steps:
        m = expm((k - q*q*d) * h) @ m
    m = expm((k - q_peak*q_peak*d) * (big_delta - delta)) @ m
    for q in q_peak * (1 - (np.arange(steps) + 0.5) / steps):
        m = expm((k - q*q*d) * h) @ m
    return m.sum()


@pytest.mark.parametrize("name", ["stick_zeppelin_karger", "fexi"])
def test_factories_no_exchange(name):
    """At kappa=0, E=f E1+(1-f) E2, including b0; rtol=2e-12, atol=2e-14."""
    model, scheme, p, d = _factory_case(
        name, [0., 0.6e9, 2.e9, 4.e9],
        [[0, 0, 1], [1, 0, 0], [0, 0.6, 0.8], [0, 0, 1]],
        0.004, 0.04, 0.37, 0.,
    )
    expected = np.exp(-scheme.bvalues[:, None] * d) @ [0.37, 0.63]
    assert_allclose(model(scheme, **p), expected, rtol=2e-12, atol=2e-14)


@pytest.mark.parametrize("name", ["stick_zeppelin_karger", "fexi"])
def test_factories_fast_exchange_at_each_orientation(name):
    """At kappa=1e8/s, E approaches exp[-b(f D1+(1-f)D2)] per direction;
    rtol=5e-6, atol=2e-8 allows finite-rate correction and eigenvalue cancellation.
    This is a single Gaussian tensor per kernel, not a powder-averaged Gaussian.
    """
    model, scheme, p, d = _factory_case(
        name, [0., 0.6e9, 2.e9, 4.e9],
        [[0, 0, 1], [1, 0, 0], [0, 0.6, 0.8], [0, 0, 1]],
        0.004, 0.04, 0.37, 1.e8,
    )
    expected = np.exp(-scheme.bvalues * (d @ [0.37, 0.63]))
    assert_allclose(model(scheme, **p), expected, rtol=5e-6, atol=2e-8)


def test_published_nexi_no_exchange_and_stick_power_law():
    r"""At t_ex=inf the powder signal is f sqrt(pi/(4 b Di)) erf(sqrt(b Di))
    +(1-f) exp(-b De), to rtol=2e-12, atol=2e-14 through b Di=100.
    At b Di>=25 the stick term equals its b^-1/2 asymptote to rtol=2e-12.
    """
    f, di, de = 0.6, 2.5e-9, 1.2e-9
    b = np.array([0., 0.1, 1., 10., 25., 100.]) / di
    stick = np.ones_like(b)
    stick[1:] = np.sqrt(np.pi / (4*b[1:]*di)) * erf(np.sqrt(b[1:]*di))
    signal = nexi_jelescu2022(b, 0.04, f=f, di=di, de=de, t_ex=np.inf)
    assert_allclose(signal, f*stick + (1-f)*np.exp(-b*de), rtol=2e-12, atol=2e-14)
    recovered_stick = (signal[-2:] - (1-f)*np.exp(-b[-2:]*de)) / f
    assert_allclose(recovered_stick, np.sqrt(np.pi/(4*b[-2:]*di)), rtol=2e-12, atol=0.)


@pytest.mark.parametrize("t_ex", [0., 1.e-8])
def test_published_nexi_fast_exchange_powder_limit(t_ex):
    r"""The fast limit is exp[-b(1-f)De] sqrt(pi/(4 b f Di)) erf(sqrt(b f Di)),
    averaged AFTER local exchange; rtol=2e-6, atol=2e-9 at t_ex=1e-8 s or 0.
    At b=4e9 it differs from exp[-b(f Di/3+(1-f)De)] by >0.01 absolute.
    """
    f, di, de = 0.6, 2.4e-9, 1.4e-9
    b = np.array([0., 0.5e9, 2.e9, 4.e9])
    x = b[1:] * f * di
    expected = np.r_[1., np.exp(-b[1:]*(1-f)*de) * np.sqrt(np.pi/(4*x)) * erf(np.sqrt(x))]
    got = nexi_jelescu2022(b, 0.04, f=f, di=di, de=de, t_ex=t_ex)
    assert_allclose(got, expected, rtol=2e-6, atol=2e-9)
    assert abs(got[-1] - np.exp(-b[-1]*(f*di/3 + (1-f)*de))) > 0.01


@pytest.mark.parametrize("f,t_ex", [(0.25, 0.012), (0.7, 0.08)])
def test_published_nexi_against_paper_eigenvalues(f, t_ex):
    """Eqs. (3)--(6), independently using Eq. (4) eigen-diffusivities and Eq. (5)
    weights with adaptive quadrature, equal the reference signal to rtol=2e-12,
    atol=2e-14. Broadcasting b and t is checked on a 2x3 array.
    """
    from scipy.integrate import quad
    di, de = 2.3e-9, 1.1e-9
    b = np.array([[0.4e9, 2.e9, 6.e9]])
    t = np.array([[0.015], [0.06]])
    expected = np.empty((2, 3))
    for i, time in enumerate(t[:, 0]):
        for j, bv in enumerate(b[0]):
            q2 = bv / time

            def kernel(u):
                d1 = di*u*u
                rate_d = 1/(q2*t_ex)
                gap = np.sqrt((de-d1+(2*f-1)*rate_d)**2 + 4*f*(1-f)*rate_d**2)
                dminus, dplus = (d1+de+rate_d-gap)/2, (d1+de+rate_d+gap)/2
                weight = (f*d1+(1-f)*de-dplus) / (dminus-dplus)
                return weight*np.exp(-bv*dminus) + (1-weight)*np.exp(-bv*dplus)

            expected[i, j] = quad(kernel, 0., 1., epsabs=1e-13, epsrel=1e-13)[0]
    got = nexi_jelescu2022(b, t, f=f, di=di, de=de, t_ex=t_ex)
    assert_allclose(got, expected, rtol=2e-12, atol=2e-14)


@pytest.mark.parametrize("f,kappa", [(0.2, 0.), (0.35, 17.), (0.8, 120.)])
def test_karger_closed_form_against_narrow_pgse_matrix_exponential(f, kappa):
    """For ideal narrow-pulse PGSE, exp[(K-q² diag(D))Delta] acting on [f,1-f]
    equals the closed form through b=6e9, including b0, to rtol=2e-12, atol=2e-14.
    The generator and matrix exponential are independent of analytic.karger.
    """
    b, t = np.array([0., 0.5e9, 2.e9, 6.e9]), 0.04
    d = np.array([0.2e-9, 2.e-9])
    expected = [np.sum(expm(_generator(f, kappa)*t - np.diag(bv*d)) @ [f, 1-f]) for bv in b]
    got = exchange.karger(b*d[0], b*d[1], kappa, f, t)
    assert_allclose(got, expected, rtol=2e-12, atol=2e-14)


@pytest.mark.parametrize("name", ["stick_zeppelin_karger", "fexi"])
def test_factories_against_finite_pgse_in_narrow_pulse_limit(name):
    """At delta=1e-6 s, Delta=0.04 s, the effective-time formula approaches the
    Bloch-Torrey ODE to rtol=1e-5, atol=1e-10 (512 midpoint expm steps per lobe).
    """
    f, kappa, delta, big_delta = 0.35, 17., 1.e-6, 0.04
    model, scheme, p, d = _factory_case(name, [0., 2.4e9], [[0, 0, 1]]*2, delta, big_delta, f, kappa)
    expected = _finite_pgse_signal(scheme.bvalues[1], delta, big_delta, d[1], f, kappa, 512)
    assert_allclose(model(scheme, **p)[1], expected, rtol=1e-5, atol=1e-10)


@pytest.mark.xfail(strict=True, reason="EXCHANGE-AUDIT-2: finite PGSE uses a constant-q effective-time approximation")
@pytest.mark.parametrize("name", ["stick_zeppelin_karger", "fexi"])
def test_factories_against_finite_pgse_ode(name):
    """For delta=18 ms, Delta=40 ms, a claimed exact Gaussian PGSE model must match
    the time-ordered Bloch-Torrey solution to rtol=1e-5, atol=1e-10. The oracle's
    256/512-step convergence must itself be <2e-7 absolute. The physical assertion
    is retained as strict xfail until finite-pulse exchange is implemented here.
    """
    f, kappa, delta, big_delta = 0.35, 17., 0.018, 0.04
    model, scheme, p, d = _factory_case(name, [0., 2.4e9], [[0, 0, 1]]*2, delta, big_delta, f, kappa)
    coarse = _finite_pgse_signal(scheme.bvalues[1], delta, big_delta, d[1], f, kappa, 256)
    expected = _finite_pgse_signal(scheme.bvalues[1], delta, big_delta, d[1], f, kappa, 512)
    assert abs(expected-coarse) < 2e-7
    assert_allclose(model(scheme, **p)[1], expected, rtol=1e-5, atol=1e-10)


def test_waveform_karger_against_finite_pgse_ode():
    """The waveform propagator follows the finite rectangular PGSE ODE at delta=18 ms,
    Delta=40 ms, b=2.4e9, to rtol=2e-6, atol=1e-10; midpoint oracle convergence
    256 -> 512 steps is <2e-7 absolute. This distinguishes waveform and closed paths.
    """
    b, delta, big_delta, f, kappa = 2.4e9, 0.018, 0.04, 0.35, 17.
    d, dt = np.array([0.3e-9, 2.e-9]), 1.e-4
    g = np.sqrt(b/(big_delta-delta/3)) / (GAMMA*delta)
    gradient = np.r_[np.full(180, g), np.zeros(220), np.full(180, -g), 0.]
    coarse = _finite_pgse_signal(b, delta, big_delta, d, f, kappa, 256)
    expected = _finite_pgse_signal(b, delta, big_delta, d, f, kappa, 512)
    assert abs(expected-coarse) < 2e-7
    got = exchange.karger_waveform(gradient, dt, d, _generator(f, kappa), [f, 1-f])
    assert_allclose(got, expected, rtol=2e-6, atol=1e-10)


def test_nexi_factory_matches_published_nexi():
    """Eq. (6) at equivalent tissue parameters, for perpendicular and parallel
    gradients alike, to the original rtol=2e-6, atol=1e-9 audit tolerance.
    """
    f, t_ex = 0.37, 0.03
    model = nexi()
    scheme = _pgse([0., 2.e9, 2.e9],
                   [[0, 0, 1], [1, 0, 0], [0, 0, 1]], 0.004, 0.04)
    p = {"X1NEXI_1_f": f, "X1NEXI_1_D_i": 2.1e-9,
         "X1NEXI_1_D_e": 1.4e-9, "X1NEXI_1_t_ex": t_ex}
    expected = nexi_jelescu2022(scheme.bvalues, scheme.Delta-scheme.delta/3,
                               f=f, di=2.1e-9, de=1.4e-9, t_ex=t_ex)
    got = model(scheme, **p)[scheme.shell_indices]
    assert_allclose(got, expected, rtol=2e-6, atol=1e-9)


@pytest.mark.xfail(strict=True, reason="EXCHANGE-AUDIT-3: compartment signals are clipped at 1e-10 before taking logs")
@pytest.mark.parametrize("name", ["stick_zeppelin_karger", "fexi"])
def test_factories_no_exchange_at_high_attenuation(name):
    """Even below 1e-10, no exchange must give f E1+(1-f) E2 to rtol=2e-12,
    atol=0. At b=1e11 s/m² both kernels expose the current log-signal floor.
    """
    model, scheme, p, d = _factory_case(name, [0., 1.e11], [[0, 0, 1]]*2,
                                      0.004, 0.04, 0.37, 0.)
    expected = np.exp(-scheme.bvalues[:, None]*d) @ [0.37, 0.63]
    assert_allclose(model(scheme, **p), expected, rtol=2e-12, atol=0.)


def _fexi_scheme():
    return AcquisitionScheme.from_fexi(
        [1., 0., 0.], 0.004, 0.01, [0.01, 0.03, 0.07, 0.15, 0.3, 0.6],
        b_f=1.2e9, bvalues=[0., 5.e6], dt=2.e-4, slew_rate=np.inf,
        timing=None,
    )


@pytest.mark.xfail(strict=True, reason="EXCHANGE-AUDIT-4: reported FEXI b and continuous waveform integral disagree")
def test_fexi_filter_and_no_exchange_limit():
    """A filter actually attenuates each pool: at kappa=0 the forward factory signal
    is f exp[-(bf+bd)Ds]+(1-f) exp[-(bf+bd)Df], to rtol=2e-6, atol=1e-9.
    bf=1.2e9 must reduce the slow-pool ADC relative to its unfiltered equilibrium.
    """
    scheme, f, ds, df = _fexi_scheme(), 0.35, 0.3e-9, 2.e-9
    m = fexi_measurements(scheme)
    model = fexi()
    prefix = "X0GeneralizedKarger_1_"
    got = model(scheme, **{prefix + "G1Ball_1_lambda_iso": ds,
                           prefix + "f": f, prefix + "kappa": 0.})
    b = m.b_f + m.b_d
    expected = f*np.exp(-b*ds) + (1-f)*np.exp(-b*df)
    assert_allclose(got, expected, rtol=2e-6, atol=1e-9)
    weights = np.array([f, 1-f])*np.exp(-1.2e9*np.array([ds, df]))
    assert weights @ [ds, df] / weights.sum() < f*ds+(1-f)*df


@pytest.mark.xfail(strict=True, reason="EXCHANGE-AUDIT-4: reported FEXI b and continuous waveform integral disagree")
def test_fexi_filter_fast_exchange_limit():
    """At kappa=1e6/s the two-pool FEXI waveform is the single averaged Gaussian
    exp[-(bf+bd)(f Ds+(1-f)Df)] to rtol=2e-4, atol=1e-8. Mixing time adds no
    attenuation without relaxation, and the diffusion filter has zero contrast.
    """
    scheme, f, ds, df = _fexi_scheme(), 0.35, 0.3e-9, 2.e-9
    m = fexi_measurements(scheme)
    prefix = "X0GeneralizedKarger_1_"
    got = fexi()(scheme, **{prefix + "G1Ball_1_lambda_iso": ds,
                           prefix + "f": f, prefix + "kappa": 1.e6})
    expected = np.exp(-(m.b_f+m.b_d)*(f*ds+(1-f)*df))
    assert_allclose(got, expected, rtol=2e-4, atol=1e-8)


@pytest.mark.parametrize("f,axr", [(0.3, 3.), (0.65, 12.)])
def test_axr_fit_recovers_independent_two_site_exchange(f, axr):
    """Synthetic two-site filter/mix/detect matrix-exponential data at bd=5e6:
    the AXR fit recovers k12+k21 (not k12) within 1%, ADC_eq within 0.5%, and
    sigma within 1%. bD<=0.01 makes higher detection cumulants small; filter
    and detection exchange are included with 10 ms constant-q PGSE blocks.
    """
    scheme = _fexi_scheme()
    m = fexi_measurements(scheme)
    d, td, kappa = np.array([0.3e-9, 2.e-9]), 0.01, axr*(1-f)
    k, m0 = _generator(f, kappa), np.array([f, 1-f])
    y = np.empty(len(m.b_d))
    for i, (bf, bd, tm) in enumerate(zip(m.b_f, m.b_d, m.t_m)):
        filtered = expm(k*td - np.diag(bf*d)) @ m0
        mixed = expm(k*tm) @ filtered
        y[i] = np.sum(expm(k*td - np.diag(bd*d)) @ mixed)
    fitted = FEXIApparentExchange().fit(scheme, y, starting_axr=(1., 10.))
    p = fitted.fitted_parameters
    adc_eq = m0 @ d
    filtered = expm(k*td - np.diag(1.2e9*d)) @ m0
    sigma = (1 - (filtered @ d)/(filtered.sum()*adc_eq)) * (-np.expm1(-axr*td)/(axr*td))
    assert_allclose(p["axr"], axr, rtol=0.01, atol=0.)
    assert_allclose(p["adc_eq"], adc_eq, rtol=0.005, atol=0.)
    assert_allclose(p["sigma"], sigma, rtol=0.01, atol=0.)
    assert abs(float(p["axr"])-kappa) > 0.2*axr


def test_axr_fit_recovers_fexi_factory_signal():
    """On the actual finite-pulse FEXI acquisition, the two-pool forward factory
    (f=0.35, k12=5.2/s, Df=2e-9) gives AXR=8/s within 1% at bd=5e6.
    Independent matrix-exponential recovery is checked separately above.
    """
    scheme, prefix = _fexi_scheme(), "X0GeneralizedKarger_1_"
    y = fexi()(scheme, **{prefix + "G1Ball_1_lambda_iso": 0.3e-9,
                          prefix + "f": 0.35, prefix + "kappa": 5.2})
    p = FEXIApparentExchange().fit(scheme, y, starting_axr=(1., 10.)).fitted_parameters
    assert_allclose(p["axr"], 8., rtol=0.01, atol=0.)
