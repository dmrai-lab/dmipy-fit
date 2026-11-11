"""Continuous integrals of held effective gradients.

Sample k acts on [k dt, (k+1) dt); the last sample is readout only.
F(t) = integral_0^t G_eff(s) ds - integral_0^T G_eff(s) ds and B = gamma**2 integral F F.T dt:
the moment to the readout, which is the net moment of an unrefocused waveform removed
(identically zero for a refocused one), as dmipy-sim defines b and the B-tensor.
The analytic exchange propagator integrates the same waveform.
"""
import numpy as np
from dmipy_analytic.constants import GAMMA


def waveform_btensor(gradient, dt):
    """Exact B (s/m²) of a piecewise constant effective gradient (..., n_t, 3).

    The moment is anchored at the readout, as in dmipy-sim.
    ``dt`` is a positive scalar or one duration per interval.
    """
    g = np.asarray(gradient, dtype=float)
    h = np.asarray(dt, dtype=float)
    if g.ndim < 2 or g.shape[-1] != 3 or g.shape[-2] < 2:
        raise ValueError('waveform_btensor: gradient must have shape (..., n_t>=2, 3)')
    if h.ndim > 1 or (h.ndim == 1 and h.shape != (g.shape[-2]-1,)):
        raise ValueError('waveform_btensor: dt must be scalar or one duration per interval')
    if not np.all(np.isfinite(h)) or np.any(h <= 0) or not np.all(np.isfinite(g)):
        raise ValueError('waveform_btensor: finite gradient and positive finite dt required')
    v = GAMMA * g[..., :-1, :]
    q1 = np.cumsum(v * h[..., None], axis=-2)
    q0 = q1 - v * h[..., None] - q1[..., -1:, :]      # the moment before each interval, to the readout
    outer = lambda a, b: a[..., :, None] * b[..., None, :]
    terms = (outer(q0, q0) * h[..., None, None]
             + (outer(q0, v) + outer(v, q0)) * h[..., None, None]**2 / 2
             + outer(v, v) * h[..., None, None]**3 / 3)
    return np.sum(terms, axis=-3)


def waveform_bvalues(gradient, dt):
    """Trace of :func:`waveform_btensor`, b = gamma² integral |F|² dt."""
    return np.trace(waveform_btensor(gradient, dt), axis1=-2, axis2=-1)


class PGSETimingDomainError(ValueError):
    """PGSE timings cannot be represented on the propagator's uniform grid."""


def pgse_waveform_blocks(scheme):
    """(G_eff, dt, stored, rows) on exact, boundary-aligned PGSE grids.

    The analytic propagator accepts uniform intervals. Ratios are rationalised
    only when reconstructed timings agree within four float64 rounding units;
    otherwise PGSETimingDomainError requests a waveform-backed acquisition.
    At most one million intervals are allowed per block. Pulse steps <= 0.1 ms
    resolve the fourth-order Magnus solver; zero-gradient plateaus are exact.
    No timing is rounded to a sampling raster.
    """
    from fractions import Fraction
    if scheme.delta is None or scheme.Delta is None or scheme.gradient_strengths is None:
        raise PGSETimingDomainError('PGSE requires delta, Delta and gradient_strengths')
    timings = np.column_stack([scheme.delta, scheme.Delta])
    blocks = []
    for delta, big_delta in np.unique(timings, axis=0):
        if not np.isfinite(delta + big_delta) or not 0 < delta <= big_delta:
            raise PGSETimingDomainError('PGSE requires finite 0 < delta <= Delta')
        ratio = Fraction(float(big_delta / delta)).limit_denominator(100000)
        dt = delta / ratio.denominator
        if abs(float(ratio.numerator * dt) - big_delta) > 4*np.finfo(float).eps*big_delta:
            raise PGSETimingDomainError(
                'PGSE timings do not share a practical uniform grid; use AcquisitionScheme.from_pgse')
        refine = max(1, int(np.ceil(dt / 1e-4)))
        pulse = ratio.denominator * refine
        separation = ratio.numerator * refine
        dt /= refine
        n = separation + pulse + 1
        if n > 1000001:
            raise PGSETimingDomainError(
                'PGSE uniform grid exceeds one million intervals; use AcquisitionScheme.from_pgse')
        rows = np.flatnonzero(np.all(timings == [delta, big_delta], axis=1))
        g = np.zeros((len(rows), n, 3))
        amplitude = scheme.gradient_strengths[rows, None] * scheme.gradient_directions[rows]
        g[:, :pulse] = amplitude[:, None]
        g[:, separation:separation+pulse] = -amplitude[:, None]
        blocks.append((g, dt, np.zeros(n, dtype=bool), rows))
    return blocks


def acquisition_waveform_blocks(scheme):
    """Public sequence waveform blocks, or exact rectangular PGSE blocks."""
    protocol = getattr(scheme, 'protocol', None)
    if protocol is None:
        return pgse_waveform_blocks(scheme)
    from fractions import Fraction
    from math import lcm
    from dmipy_replay.acquisition.rf import role_of
    blocks = []
    for seq, rows in zip(protocol, protocol.rows):
        refine = 1
        for event in seq.rf:
            if role_of(event) not in ('store', 'recall'):
                continue
            ratio = Fraction(float(event.t_s / seq.dt)).limit_denominator(100000)
            if abs(float(ratio)*seq.dt - event.t_s) > 4*np.finfo(float).eps*seq.T:
                raise PGSETimingDomainError('storage boundary cannot be represented on a practical uniform grid')
            refine = lcm(refine, ratio.denominator)
        if (seq.n_t-1)*refine > 1000000:
            raise PGSETimingDomainError('storage-aligned waveform exceeds one million intervals')
        gradient = np.asarray(seq.G_eff, dtype=float)
        if refine > 1:
            gradient = np.concatenate([np.repeat(gradient[:, :-1], refine, axis=1), gradient[:, -1:]], axis=1)
        dt = float(seq.dt) / refine
        stored = ~np.asarray(seq.rf.transverse_gate(gradient.shape[1], dt), dtype=bool)
        blocks.append((gradient, dt, stored, np.asarray(rows)))
    return blocks
