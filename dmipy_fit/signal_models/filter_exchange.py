# -*- coding: utf-8 -*-
"""Filter-exchange imaging (FEXI): the apparent-exchange-rate analysis on a FEXI acquisition.

A FEXI measurement is a row of a filter-exchange sequence (:func:`dmipy_replay.sequences.fexi`): a filter block of
b-value ``b_f``, the magnetisation stored over a mixing time ``t_m``, a detection block of b-value ``b_d``. Rows that
share a sequence (one mixing time) and a filter state form a SERIES, each with its own unweighted signal. The AXR
model reads every series as ``S0_series exp(-b_d ADC)``, ``ADC = ADC_eq`` unfiltered and ``ADC'(t_m) = ADC_eq [1 -
sigma exp(-AXR t_m)]`` filtered (Lasic et al. 2011; md-dmri ``fexi11``); its equation is
:func:`dmipy_analytic.exchange.fexi_signal`.

Everything a row means is read off the acquisition object: ``b_f`` and ``b_d`` by
:func:`dmipy_replay.sequences.fexi_bvalues`, ``t_m`` as the sequence's storage time ``ScannerSequence.TM``. A row is
filtered when its filter plays (``b_f > 0``).
"""
from dataclasses import dataclass

import numpy as np
from dmipy_analytic import exchange

__all__ = ["FEXIMeasurements", "fexi_measurements", "FEXIApparentExchange", "FittedFEXIApparentExchange"]


@dataclass(frozen=True)
class FEXIMeasurements:
    """What each measurement of a FEXI acquisition is, in acquisition order: ``b_f`` and ``b_d`` (s/m^2), ``t_m`` (s),
    ``filtered`` (bool) and ``series`` (int, ``0 .. n_series - 1``, one per sequence and filter state)."""
    b_f: np.ndarray
    b_d: np.ndarray
    t_m: np.ndarray
    filtered: np.ndarray
    series: np.ndarray

    @property
    def n_series(self):
        return int(self.series.max()) + 1


def fexi_measurements(acquisition_scheme):
    """:class:`FEXIMeasurements` of an :class:`~dmipy_fit.core.acquisition_scheme.AcquisitionScheme` whose sequences are
    filter-exchange sequences; any other sequence is refused by :func:`dmipy_replay.sequences.fexi_bvalues`."""
    from dmipy_replay.sequences import fexi_bvalues
    protocol = getattr(acquisition_scheme, "protocol", None)
    if protocol is None:
        raise TypeError("the FEXI analysis reads the acquisition object: an AcquisitionScheme over dmipy-replay's "
                        "fexi sequences (AcquisitionScheme.from_fexi, or AcquisitionScheme(fexi_protocol(...)))")
    n = protocol.n_meas
    b_f, b_d, t_m = np.empty(n), np.empty(n), np.empty(n)
    series = np.empty(n, dtype=int)
    filtered = np.empty(n, dtype=bool)
    k = 0
    for seq, rows in zip(protocol, protocol.rows):
        bf, bd = fexi_bvalues(seq)
        on = bf > 0.0
        b_f[rows], b_d[rows], t_m[rows], filtered[rows] = bf, bd, float(seq.TM), on
        for state in (False, True):
            hit = rows[on == state]
            if hit.size:
                series[hit] = k
                k += 1
    return FEXIMeasurements(b_f, b_d, t_m, filtered, series)


class FEXIApparentExchange:
    """The apparent-exchange-rate (AXR) model of FEXI: ``ADC_eq``, the filter efficiency ``sigma`` and ``AXR``, with
    one unweighted signal per series.

    Its equation is :func:`dmipy_analytic.exchange.fexi_signal`, read on the measurements of a FEXI acquisition
    (:func:`fexi_measurements`). For two exchanging Gaussian pools of equal ``T1`` it is exact in the limit of a small
    detection b-value, with ``AXR = k_12 + k_21`` (:func:`dmipy_analytic.exchange.fexi_adc`). At finite detection b
    the two-pool signal has higher cumulants: a monoexponential series fit estimates an effective ADC, which can
    bias AXR. The fitted ADC approaches the first cumulant as detection b tends to zero.
    The fit is least squares in the signal, the series' unweighted signals
    eliminated in closed form for each ``(ADC_eq, sigma, AXR)`` (variable projection), from several starting rates.

    References: Lasic S, Nilsson M, Latt J, Stahlberg F, Topgaard D (2011), Magn. Reson. Med. 66:356-365; md-dmri
    ``methods/fexi11`` (Nilsson M, Topgaard D et al.).
    """
    parameter_names = ("adc_eq", "sigma", "axr")
    parameter_ranges = {"adc_eq": (0.01e-9, 4.0e-9), "sigma": (0.0, 1.0), "axr": (1e-3, 1e3)}
    _scale = np.array([1e-9, 1.0, 1.0])

    def __call__(self, acquisition_scheme, adc_eq, sigma, axr):
        """The model's signal per measurement, each series at 1 at no detection weighting."""
        m = fexi_measurements(acquisition_scheme)
        return exchange.fexi_signal(m.b_d, m.t_m, m.filtered, adc_eq, sigma, axr)

    def predict(self, acquisition_scheme, adc_eq, sigma, axr, S0):
        """The signal per measurement with the series' unweighted signals ``S0`` (``(n_series,)``)."""
        m = fexi_measurements(acquisition_scheme)
        return np.asarray(S0, dtype=float)[m.series] * self(acquisition_scheme, adc_eq, sigma, axr)

    def _unfiltered_slope(self, m, y):
        """The starting ``ADC_eq``: the median over the unfiltered rows of the slope of ``-ln S`` from their series'
        least-weighted row, clipped into the parameter's range (``1e-9`` when there is no such pair)."""
        slopes = []
        for k in np.unique(m.series[~m.filtered]):
            rows = np.flatnonzero(m.series == k)
            ref = rows[np.argmin(m.b_d[rows])]
            for i in rows:
                if m.b_d[i] > m.b_d[ref] and y[i] > 0 and y[ref] > 0:
                    slopes.append(-np.log(y[i] / y[ref]) / (m.b_d[i] - m.b_d[ref]))
        adc0 = float(np.median(slopes)) if slopes else 1e-9
        return float(np.clip(adc0, *self.parameter_ranges["adc_eq"]))

    def fit(self, acquisition_scheme, data, mask=None, starting_axr=(0.3, 1.0, 3.0, 10.0, 30.0, 100.0)):
        """Fit every voxel of ``data`` (``(..., n_meas)``, any scale): a :class:`FittedFEXIApparentExchange`.

        ``mask`` (``data.shape[:-1]``, bool) selects the voxels fitted; the others are NaN. Each voxel is fitted from
        every rate in ``starting_axr`` (s^-1) and the best is kept, ``ADC_eq`` starting at the unfiltered series'
        slope and ``sigma`` at 0.3."""
        from scipy.optimize import least_squares
        m = fexi_measurements(acquisition_scheme)
        data = np.asarray(data, dtype=float)
        shape = data.shape[:-1]
        Y = data.reshape(-1, data.shape[-1])
        sel = np.ones(Y.shape[0], bool) if mask is None else np.asarray(mask, bool).reshape(-1)
        lo = np.array([self.parameter_ranges[k][0] for k in self.parameter_names]) / self._scale
        hi = np.array([self.parameter_ranges[k][1] for k in self.parameter_names]) / self._scale
        onehot = np.eye(m.n_series)[m.series]                      # (n_meas, n_series)

        def model(x):
            return exchange.fexi_signal(m.b_d, m.t_m, m.filtered, *(x * self._scale))

        def s0_of(mu, y):
            return (onehot.T @ (y * mu)) / np.maximum(onehot.T @ (mu * mu), 1e-300)

        params = np.full((Y.shape[0], 3), np.nan)
        S0 = np.full((Y.shape[0], m.n_series), np.nan)
        cost = np.full(Y.shape[0], np.nan)
        for v in np.flatnonzero(sel):
            y = Y[v]
            adc0 = self._unfiltered_slope(m, y)
            best = None
            for r in starting_axr:
                x0 = np.clip(np.array([adc0, 0.3, r]) / self._scale, lo * (1 + 1e-9), hi * (1 - 1e-9))

                def resid(x):
                    mu = model(x)
                    return y - s0_of(mu, y)[m.series] * mu

                sol = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", xtol=1e-15, ftol=1e-15, gtol=1e-15,
                                    max_nfev=2000)
                if best is None or sol.cost < best.cost:
                    best = sol
            params[v] = best.x * self._scale
            S0[v] = s0_of(model(best.x), y)
            cost[v] = best.cost
        fitted = {k: params[:, i].reshape(shape) for i, k in enumerate(self.parameter_names)}
        return FittedFEXIApparentExchange(self, acquisition_scheme, fitted, S0.reshape(shape + (m.n_series,)),
                                          cost.reshape(shape))


@dataclass
class FittedFEXIApparentExchange:
    """The result of :meth:`FEXIApparentExchange.fit`: ``fitted_parameters`` (``adc_eq``, ``sigma``, ``axr``, each of
    the data's voxel shape), ``S0`` (the series' unweighted signals, ``(..., n_series)``) and ``cost`` (half the sum of
    squared residuals per voxel)."""
    model: FEXIApparentExchange
    acquisition_scheme: object
    fitted_parameters: dict
    S0: np.ndarray
    cost: np.ndarray

    def predict(self):
        """The fitted signal, ``(..., n_meas)``."""
        p = self.fitted_parameters
        shape = p["adc_eq"].shape
        flat = [np.ravel(p[k]) for k in self.model.parameter_names]
        S0 = self.S0.reshape(-1, self.S0.shape[-1])
        out = np.array([self.model.predict(self.acquisition_scheme, a, s, x, s0)
                        for a, s, x, s0 in zip(*flat, S0)])
        return out.reshape(shape + (out.shape[-1],))
