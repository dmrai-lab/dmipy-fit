"""Standard myelin-water-fraction (MWF) from a multi-echo T2 decay.

Classic regularised non-negative T2-spectrum fit (Whittall & MacKay 1989;
Prasloski 2012; Doucette 2020, DECAES) with ideal ``exp(-t/T2)`` bases (the
instantaneous-pulse assumption). Pure relaxometry — no susceptibility, no
orientation/FOD, no finite-pulse/EPG modelling. Takes a measured or simulated
multi-echo decay + its echo times and returns the myelin-water fraction as the
spectral weight below a short-T2 cutoff.

    from dmipy_fit.white_matter.mwf import t2_spectrum_mwf
    mwf, T2_grid, spectrum = t2_spectrum_mwf(signal, echo_times)
"""
import numpy as np

__all__ = ['t2_spectrum_mwf']


def _nnls_x2(A, y, factor=1.02, L=None, mu_max=10.0):
    r"""Zero-order regularised NNLS with the chi-square (X2) weight criterion.

    The literature-standard myelin-water solver (Whittall & MacKay 1989;
    Prasloski 2012; Doucette 2020, DECAES). The regularisation weight ``mu`` is
    chosen **per signal** so that the regularised misfit is ``factor`` times the
    unregularised NNLS minimum --- i.e. the smoothest spectrum whose data fit is
    still within ~2 % of the best possible fit (``factor`` in [1.02, 1.025]).
    Far more noise-robust than any fixed weight: at high SNR ``mu`` shrinks
    toward zero, at low SNR it grows to suppress spurious spectral splitting.

    ``L`` is the regularisation operator (default: identity = zero-order
    Tikhonov, the conventional Mackay choice). Returns ``(spectrum, mu)``.
    """
    from scipy.optimize import nnls, fminbound
    n = A.shape[1]
    if L is None:
        L = np.eye(n)
    y = np.asarray(y, float)
    f0, _ = nnls(A, y)                      # unregularised reference fit
    sse0 = float(np.sum((A @ f0 - y) ** 2)) # minimum achievable misfit
    if sse0 <= 0.0:                         # perfect/degenerate fit: no reg
        return f0, 0.0
    y_aug = np.r_[y, np.zeros(n)]

    def obj(mu):
        Aaug = np.vstack([A, np.sqrt(mu) * L])
        f, _ = nnls(Aaug, y_aug)
        sser = float(np.sum((A @ f - y) ** 2))   # misfit on DATA rows only
        return abs(sser - factor * sse0) / sse0

    mu = float(fminbound(obj, 0.0, mu_max, xtol=1e-5, maxfun=300))
    f, _ = nnls(np.vstack([A, np.sqrt(mu) * L]), y_aug)
    return f, mu


def t2_spectrum_mwf(signal, echo_times, T2_grid=None, cutoff=0.025, reg='x2',
                    x2_factor=1.02):
    """Classic regularised NNLS T2-spectrum MWF (the standard analysis).

    Fits a non-negative $T_2$ spectrum to the multi-echo decay assuming **ideal**
    ``exp(-t/T2)`` bases (the instantaneous-pulse assumption), and returns the myelin
    water fraction = spectral weight below ``cutoff`` (s). Returns
    ``(mwf, T2_grid, spectrum)``.

    ``reg`` selects the regularisation: ``'x2'`` (default) chooses the weight
    per signal by the chi-square criterion (Whittall--MacKay / Prasloski /
    DECAES standard; smoothest fit within ``x2_factor`` of the unregularised
    misfit), which is the literature-trusted, noise-robust choice. A float
    instead applies that fixed zero-order Tikhonov weight (legacy behaviour).
    """
    from scipy.optimize import nnls
    if T2_grid is None:
        T2_grid = np.logspace(np.log10(5e-3), np.log10(2.0), 60)
    A = np.exp(-np.asarray(echo_times)[:, None] / T2_grid[None, :])
    if reg == 'x2':
        spec, _ = _nnls_x2(A, np.asarray(signal, float), factor=x2_factor)
    else:
        Ar = np.vstack([A, reg * np.eye(len(T2_grid))])
        yr = np.r_[np.asarray(signal, float), np.zeros(len(T2_grid))]
        spec, _ = nnls(Ar, yr)
    mwf = spec[T2_grid < cutoff].sum() / spec.sum() if spec.sum() > 0 else 0.0
    return float(mwf), T2_grid, spec
