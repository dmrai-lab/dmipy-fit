"""Filter-exchange imaging on the acquisition object: the Karger models on a FEXI protocol are the generalised Karger
equation along each sequence's own effective gradient (dmipy_analytic.exchange.karger_waveform), and the AXR analysis
recovers the two-site rate ``k_12 + k_21`` from that signal.

The AXR model is exact for two Gaussian pools of equal ``T1`` in the limit of a small detection b
(dmipy_analytic.exchange.fexi_adc). At a detection b of ``b_d`` every series' two-point ADC lies within ``[mu_1 - b_d
dD^2 / 8, mu_1]`` of its initial slope ``mu_1`` (the log of a Laplace transform over diffusivities in ``[D_1, D_2]``,
Hoeffding's lemma), so a fitted parameter ``p`` is within ``sum_s |dp / dADC_s| b_d dD^2 / 8`` of its limit, the
sensitivities those of the fit itself at the limit (one series' ADC moved at a time). The limit's ``sigma`` is the
two-site signal's own: ``(1 - ADC'(t_m) / ADC_eq) exp(AXR t_m)`` at a detection b of ``1e5`` s/m^2.
"""
import numpy as np
import pytest

import dmipy_replay as dr
from dmipy_analytic import exchange
from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.custom_optimizers.reference_models import stick_zeppelin_karger
from dmipy_fit.signal_models.exchange_models import X0GeneralizedKarger
from dmipy_fit.signal_models.filter_exchange import FEXIApparentExchange, fexi_measurements
from dmipy_fit.signal_models.gaussian_models import G1Ball

F, D1, D2, B_F = 0.5, 0.2e-9, 1.5e-9, 1.0e9
DELTA, BIG_DELTA, DT = 4e-3, 10e-3, 1e-4
AXR_RANGE = (1.0, 2.0, 5.0, 10.0, 15.0)


def _t_m(axr):
    """Eight mixing times from about 0.1 / AXR to 3.5 / AXR, on the grid."""
    return np.round(np.geomspace(0.1, 3.5, 8) / axr / DT) * DT


def _scheme(axr, b_d, **kw):
    return AcquisitionScheme.from_fexi([1.0, 0.0, 0.0], DELTA, BIG_DELTA, _t_m(axr), b_f=B_F, bvalues=[0.0, b_d], dt=DT,
                                       slew_rate=np.inf, **kw)


def _two_site(scheme, axr):
    kappa = axr * (1.0 - F)
    return X0GeneralizedKarger(G1Ball(), G1Ball())(scheme, G1Ball_1_lambda_iso=D1, G1Ball_2_lambda_iso=D2, f=F,
                                                   kappa=kappa)


def test_the_measurements_are_read_off_the_sequences():
    sch = AcquisitionScheme.from_fexi([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], DELTA, BIG_DELTA, [0.02, 0.2], b_f=B_F,
                                      bvalues=[0.0, 0.6e9], dt=DT, slew_rate=np.inf)
    m = fexi_measurements(sch)
    assert m.n_series == 4 and m.series.tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
    assert m.filtered.tolist() == ([False] * 4 + [True] * 4) * 2
    assert m.t_m == pytest.approx(np.repeat([0.02, 0.2], 8), rel=1e-12)
    assert m.b_f == pytest.approx(np.tile(np.repeat([0.0, B_F], 4), 2), rel=1e-6, abs=0.0)
    assert m.b_d == pytest.approx(np.tile(np.repeat([0.0, 0.6e9], 2), 4), rel=1e-6, abs=0.0)
    assert sch.bvalues == pytest.approx(m.b_f + m.b_d, rel=1e-6)
    with pytest.raises(ValueError, match="store / recall"):
        fexi_measurements(AcquisitionScheme.from_pgse([0.0, 1e9], [[1, 0, 0], [1, 0, 0]], DELTA, BIG_DELTA))


def test_the_karger_model_on_a_fexi_protocol_is_the_waveform_form():
    sch = _scheme(5.0, 0.8e9, timing=None)
    kappa = 2.5
    E = _two_site(sch, 5.0)
    K, M0 = exchange.exchange_matrix(kappa, F), np.array([F, 1.0 - F])
    for seq, rows in zip(sch.protocol, sch.protocol.rows):
        stored = ~seq.rf.transverse_gate(seq.n_t, seq.dt).astype(bool)
        want = exchange.karger_waveform(np.asarray(seq.G_eff), seq.dt, [D1, D2], K, M0, stored=stored)
        assert E[rows] == pytest.approx(want, rel=1e-12)
    assert np.all(E[~fexi_measurements(sch).filtered & (fexi_measurements(sch).b_d == 0)] == pytest.approx(1.0))


def test_stick_zeppelin_karger_on_a_fexi_protocol_reads_each_pool_along_the_row_direction():
    """A stick and a tortuous zeppelin: along a row's direction ``n`` their diffusivities are ``D_i (n.mu)^2`` and
    ``D_e [(n.mu)^2 + (1 - f)(1 - (n.mu)^2)]``, Gaussian pools under a colinear encoding, so the waveform form with
    those two numbers is the model to the float32 rounding of the directions and b-values the scheme holds (the gradient
    is stored in float32): ``4 U32`` relative in each exponent, which the signal carries times the exponent."""
    dirs = [[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]]
    sch = AcquisitionScheme.from_fexi(dirs, DELTA, BIG_DELTA, [0.03, 0.15], b_f=B_F, bvalues=[0.0, 0.7e9], dt=DT,
                                      slew_rate=np.inf)
    pre = "X0GeneralizedKarger_1_"
    Di, De, f, kappa, mu = 2.1e-9, 1.7e-9, 0.6, 8.0, np.array([0.5, 0.3])
    E = stick_zeppelin_karger()(sch, **{pre + "C1Stick_1_lambda_par": Di, pre + "G2Zeppelin_1_lambda_par": De, pre + "mu": mu,
                       pre + "f": f, pre + "kappa": kappa})
    axis = np.array([np.sin(mu[0]) * np.cos(mu[1]), np.sin(mu[0]) * np.sin(mu[1]), np.cos(mu[0])])
    K, M0 = exchange.exchange_matrix(kappa, f), np.array([f, 1.0 - f])
    for seq, rows in zip(sch.protocol, sch.protocol.rows):
        stored = ~seq.rf.transverse_gate(seq.n_t, seq.dt).astype(bool)
        for k, m in enumerate(rows):
            c2 = float(seq.encoding.gradient_directions[k] @ axis) ** 2
            D = [Di * c2, De * (c2 + (1.0 - f) * (1.0 - c2))]
            want = exchange.karger_waveform(np.asarray(seq.G_eff)[k], seq.dt, D, K, M0, stored=stored)
            assert E[m] == pytest.approx(want, rel=4 * 2.0 ** -24 * max(1.0, -np.log(want)))


def _sensitivities(model, sch, limit, b_d):
    """``dp / dADC_s`` of the fit at the limit, one series' ADC moved by ``eps`` at a time."""
    m = fexi_measurements(sch)
    base = model.predict(sch, *limit, S0=np.ones(m.n_series))
    eps = 1e-3 * limit[0]
    J = np.empty((3, m.n_series))
    for s in range(m.n_series):
        y = base * np.where(m.series == s, np.exp(-m.b_d * eps), 1.0)
        p = model.fit(sch, y).fitted_parameters
        J[:, s] = (np.array([float(p[k]) for k in model.parameter_names]) - np.asarray(limit)) / eps
    return J


def _limit(axr):
    """``(ADC_eq, sigma, AXR)`` of the two-site signal in the small-b limit: the slope at ``1e5`` s/m^2."""
    eps = 1e5
    sch = _scheme(axr, eps)
    E = _two_site(sch, axr)
    m = fexi_measurements(sch)
    adc = {}
    for s in range(m.n_series):
        rows = np.flatnonzero(m.series == s)
        adc[s] = -np.log(E[rows[m.b_d[rows] > 0]][0] / E[rows[m.b_d[rows] == 0]][0]) / eps
    off = [adc[s] for s in range(m.n_series) if not m.filtered[m.series == s][0]]
    s1 = next(s for s in range(m.n_series) if m.filtered[m.series == s][0])
    adc_eq = float(np.mean(off))
    sigma = (1.0 - adc[s1] / adc_eq) * np.exp(axr * _t_m(axr)[0])            # the first filtered series: t_m[0]
    return adc_eq, float(sigma), axr


def _fit(model, axr, b_d):
    sch = _scheme(axr, b_d)
    p = model.fit(sch, _two_site(sch, axr)).fitted_parameters
    return sch, np.array([float(p[k]) for k in model.parameter_names])


@pytest.mark.parametrize("axr", AXR_RANGE)
def test_the_axr_fit_recovers_the_two_site_rate(axr):
    """``AXR = k_12 + k_21`` and the limit's ``ADC_eq`` and ``sigma``: (i) at detection b-values ``b`` and ``5 b`` the
    fit is within ``c max(sum J+, sum |J-|)`` of the limit, ``c = b dD^2 / 8`` and ``J`` the sensitivities (every
    series' ADC errs low, so the linear form's extremes over the box ``[-c, 0]`` are its positive and its negative
    parts); (ii) the error is first order in ``b``, as the cumulant expansion has it: the extrapolation ``(5 p(b) -
    p(5 b)) / 4`` to ``b = 0`` is within a fifth of the error at ``b``. A target of ``kappa`` alone, the rate out of the
    first pool, fails (ii)."""
    model, b = FEXIApparentExchange(), 2e7
    limit = np.array(_limit(axr))
    got = {}
    for b_d in (b, 5 * b):
        sch, got[b_d] = _fit(model, axr, b_d)
        J = _sensitivities(model, sch, limit, b_d)
        bound = np.maximum(np.clip(J, 0, None).sum(axis=1), -np.clip(J, None, 0).sum(axis=1)) * b_d * (D2 - D1) ** 2 / 8
        assert np.all(np.abs(got[b_d] - limit) <= bound), (b_d, got[b_d], limit, bound)
    p0 = (5.0 * got[b] - got[5 * b]) / 4.0
    assert np.all(np.abs(p0 - limit) <= np.abs(got[b] - limit) / 5.0), (p0, got[b], limit)
    wrong = limit.copy()
    wrong[2] = axr * (1.0 - F)
    assert abs(p0[2] - wrong[2]) > abs(got[b][2] - wrong[2]) / 5.0


def _multi_delta_scheme():
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(10, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    b = np.r_[0.0, np.repeat([1e9, 2.5e9, 4e9], 10)]
    d = np.r_[[[0.0, 0.0, 1.0]], np.tile(dirs, (3, 1))]
    return AcquisitionScheme(dr.Protocol([dr.pgse(d, 6e-3, Delta, bvalues=b, slew_rate=np.inf)
                                          for Delta in (0.015, 0.025, 0.04)]))


def _assert_stick_zeppelin_karger_recovers_truth(kappa, **fit_kwargs):
    """The noise-free Stick–Zeppelin Karger signal on the multi-Delta protocol, fitted with ``fit_kwargs``, returns the generating
    parameters (the axis up to its sign) to what its residual allows: in the linear regime a parameter's error is at
    most ``sqrt(rss (J^T J)^-1_ii)``, ``J`` the model's Jacobian at the solution."""
    sch = _multi_delta_scheme()
    assert sorted(set(np.round(sch.Delta, 6))) == [0.015, 0.025, 0.04]
    model, pre = stick_zeppelin_karger(), "X0GeneralizedKarger_1_"
    truth = {pre + "C1Stick_1_lambda_par": 2.2e-9, pre + "G2Zeppelin_1_lambda_par": 1.6e-9, pre + "mu": [0.6, 1.0],
             pre + "f": 0.55, pre + "kappa": kappa}
    E = model(sch, **truth)
    fit = {k: np.squeeze(v) for k, v in model.fit(sch, E, **fit_kwargs).fitted_parameters.items()}
    rss = float(np.sum((model(sch, **fit) - E) ** 2))
    # The global minimum of a noise-free signal has a residual at the floating-point floor; a residual of order
    # ``r`` (relative to the signal energy) fixes a parameter to ``sqrt(r)`` relative, so ``r <= 1e-12`` is a
    # parameter precision of 1e-6, and a local minimum (residual ~1e-3) fails it.
    assert rss <= 1e-12 * float(np.sum(E ** 2)), rss / float(np.sum(E ** 2))
    names = [pre + "C1Stick_1_lambda_par", pre + "G2Zeppelin_1_lambda_par", pre + "f", pre + "kappa", "theta", "phi"]
    scale = np.array([1e-9, 1e-9, 1.0, 1.0, 1.0, 1.0])
    theta, phi = fit[pre + "mu"]
    if np.cos(theta) * np.cos(truth[pre + "mu"][0]) < 0:                 # the same axis, written antipodally
        truth = dict(truth, **{pre + "mu": [np.pi - truth[pre + "mu"][0], truth[pre + "mu"][1] + np.pi]})
    want = [truth[k] for k in names[:4]] + [truth[pre + "mu"][0], np.angle(np.exp(1j * (truth[pre + "mu"][1] - phi))) + phi]
    got = [fit[k] for k in names[:4]] + [theta, phi]

    def at(x):
        p = dict(fit, **{k: v for k, v in zip(names[:4], x[:4])})
        p[pre + "mu"] = [x[4], x[5]]
        return model(sch, **p)

    J = []
    for i, sc in enumerate(scale):
        up, dn = np.array(got, float), np.array(got, float)
        up[i] += 1e-6 * sc
        dn[i] -= 1e-6 * sc
        J.append((at(up) - at(dn)) / (2e-6))
    J = np.array(J).T
    bound = np.sqrt(rss * np.diag(np.linalg.inv(J.T @ J))) * scale
    for k, g, w, bnd in zip(names, got, want, bound):
        assert abs(g - w) <= max(bnd, 4 * np.finfo(float).eps * abs(w)), (k, g, w, bnd)


def test_stick_zeppelin_karger_fits_on_a_protocol_of_multi_delta_pgse_sequences():
    """Three PGSE sequences of one ``delta`` and three ``Delta`` (one echo time each) in one Protocol: every row reads
    its own sequence's diffusion time, and the MIX fit of the noise-free Stick–Zeppelin Karger signal returns the generating
    parameters."""
    _assert_stick_zeppelin_karger_recovers_truth(20.0, solver="mix")


@pytest.mark.parametrize("kappa", [5.0, 20.0, 40.0, 77.0, 150.0])
def test_stick_zeppelin_karger_default_solver_reaches_the_global_minimum_on_multi_delta_pgse(kappa):
    """The default solver recovers the generating Stick–Zeppelin Karger parameters of a noise-free multi-Delta protocol at exchange
    rates from below ``1 / (Delta_max - delta / 3)`` = 26 s^-1 to above ``1 / (Delta_min - delta / 3)`` = 77 s^-1: its
    local stage runs until the residual stops decreasing, wherever the coarse grid (spacing 40 s^-1 in ``kappa``)
    left it."""
    _assert_stick_zeppelin_karger_recovers_truth(kappa)
