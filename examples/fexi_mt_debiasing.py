"""Paper 2 companion: MT biases FEXI's apparent exchange rate; qMT removes it.

FEXI measures the apparent exchange rate (AXR) from the mixing-time (t_m) recovery of a
diffusion filter.  t_m is a LONGITUDINAL storage period, so MT acts during it -- and for
a 3-pool longitudinal model (intra + extra free water + a shared bound pool) the filter
imbalance is an EXACT eigenmode with decay rate AXR_water + k_f (free water leaks to the
bound sink at k_f; the bound returns symmetrically and cannot restore the imbalance).
So FEXI reads AXR_water + k_f: a naive AXR is biased up by exactly the MT forward rate.
The off-resonance qMT Z-spectrum measures k_f, and subtracting it recovers AXR_water.

Panel A: the FEXI t_m recovery (one SNR=40 realisation); the fitted recovery sits at
AXR_water + k_f, well above the true water-only recovery (rate AXR_water) -- the gap is MT.
Panel B: recovered AXR_water vs SNR -- biased (PGSTE/FEXI alone) stays at AXR_water + k_f
regardless of SNR; the joint FEXI + Z-spectrum fit is unbiased.

This directly answers Kiselev & Li 2026 (arXiv:2601.20657) -- geometric exchange is one
FEXI confound; MT is a second, independent one, and here it is corrected.

Run (fit venv): python examples/fexi_mt_debiasing.py  ->  examples/fexi_mt_debiasing.png
Local figure -- nothing uploaded.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from dmipy_fit.white_matter.magnetization_transfer import z_spectrum

TM = np.linspace(0.02, 1.0, 12)
OFFS = np.array([1000., 2000., 4000., 8000., 16000., 32000.])
AXR_TRUE, KF_TRUE, K_R = 1.0, 1.5, 8.0
T1A, T1B, T2A, T2B, W1 = 1.0, 1.0, 0.05, 1e-5, 300.0
ADCEQ, SIGMA = 0.8, 0.5
BIASED = dict(color="#e0651a")
JOINT = dict(color="#2563eb")


def fexi(axr_water, k_f):                         # exact 3-pool imbalance eigenrate = AXR_w + k_f
    return ADCEQ * (1.0 - SIGMA * np.exp(-(axr_water + k_f) * TM))


def zspec(k_f):
    return z_spectrum(OFFS, W1, k_f=k_f, k_r=K_R, T1a=T1A, T1b=T1B, T2a=T2A, T2b=T2B)


def fit_biased(f):
    return least_squares(lambda a: fexi(a[0], 0.0) - f, [1.0], bounds=(0.0, 50.0)).x[0]


def fit_joint(f, z):
    def resid(x):
        return np.concatenate([fexi(x[0], x[1]) - f, zspec(x[1]) - z])
    return least_squares(resid, [0.5, 0.5], bounds=([0., 0.], [50., 10.])).x


def main():
    f_true, z_true = fexi(AXR_TRUE, KF_TRUE), zspec(KF_TRUE)
    rng = np.random.RandomState(1)

    snrs = [20, 40, 80, 160]
    N = 40
    kb_m, kb_s, kj_m, kj_s = [], [], [], []
    for snr in snrs:
        kb, kj = [], []
        for _ in range(N):
            f = f_true + rng.randn(len(TM)) * ADCEQ / snr
            z = z_true + rng.randn(len(OFFS)) / snr
            kb.append(fit_biased(f))
            kj.append(fit_joint(f, z)[0])
        kb_m.append(np.mean(kb)); kb_s.append(np.std(kb))
        kj_m.append(np.mean(kj)); kj_s.append(np.std(kj))

    f = f_true + rng.randn(len(TM)) * ADCEQ / 40.0
    kb1 = fit_biased(f)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.3))
    tm = TM * 1e3

    axA.plot(tm, f, "ko", ms=5, label="FEXI data (SNR 40)")
    axA.plot(tm, fexi(kb1, 0.0), "-", color=BIASED["color"], lw=2,
             label=f"fitted recovery  AXR$_{{app}}$={kb1:.2f} = AXR$_w$+k$_f$")
    axA.plot(tm, fexi(AXR_TRUE, 0.0), "--", color=JOINT["color"], lw=2,
             label=f"true water-only  AXR$_w$={AXR_TRUE:.2f}")
    axA.set_xlabel("mixing time t$_m$ (ms)")
    axA.set_ylabel("apparent ADC  (a.u.)")
    axA.legend(frameon=False, fontsize=8.5, loc="lower right")
    axA.set_title("A  FEXI recovery sits at AXR$_w$+k$_f$; the gap is MT", fontsize=11, loc="left")

    axB.axhline(AXR_TRUE, color="0.6", ls=":", lw=1.2, label=f"true AXR$_w$ = {AXR_TRUE:.1f}")
    axB.axhline(AXR_TRUE + KF_TRUE, color=BIASED["color"], ls="--", lw=1,
                label=f"AXR$_w$+k$_f$ = {AXR_TRUE+KF_TRUE:.1f}")
    x = np.arange(len(snrs))
    axB.errorbar(x - 0.06, kb_m, yerr=kb_s, fmt="s-", capsize=4, lw=2, **BIASED,
                 label="FEXI only  (biased)")
    axB.errorbar(x + 0.06, kj_m, yerr=kj_s, fmt="o-", capsize=4, lw=2, **JOINT,
                 label="joint + Z-spectrum")
    axB.set_xticks(x); axB.set_xticklabels([str(s) for s in snrs])
    axB.set_xlabel("SNR"); axB.set_ylabel("recovered water AXR (s$^{-1}$)")
    axB.legend(frameon=False, fontsize=8.5, loc="center right")
    axB.set_title("B  Joint fit recovers AXR$_w$ across SNR", fontsize=11, loc="left")

    fig.suptitle("MT confounds FEXI's exchange rate (AXR = AXR$_w$ + k$_f$); the qMT Z-spectrum de-biases it",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(os.path.dirname(__file__), "fexi_mt_debiasing.png")
    fig.savefig(out, dpi=140)
    for s, bm, bs, jm, js in zip(snrs, kb_m, kb_s, kj_m, kj_s):
        print(f"SNR {s:>3}: biased AXR={bm:.2f}±{bs:.2f}  joint AXR_w={jm:.2f}±{js:.2f}  "
              f"(true AXR_w={AXR_TRUE}, AXR_w+k_f={AXR_TRUE+KF_TRUE})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
