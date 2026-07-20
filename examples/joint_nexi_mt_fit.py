"""Paper 2, quantitative: joint NEXI + MT + Z-spectrum fit de-biases permeability.

A single synthetic dataset = PGSTE mixing-time measurements (Karger exchange x the
longitudinal MT factor exp(-k_f*TM)) + a qMT Z-spectrum (which constrains k_f).  We add
Gaussian noise across a range of SNR and fit two ways per realisation:
  - biased: exchange rate from the PGSTE data alone (no MT term);
  - joint:  (kappa_exchange, k_f) from PGSTE + Z-spectrum together.

Panel A: PGSTE data (one SNR=40 realisation) with both fits.
Panel B: recovered exchange rate vs SNR (mean +/- SD over realisations) -- the bias
does not wash out with SNR, while the joint fit is unbiased with shrinking error bars.

Run (fit venv): python examples/joint_nexi_mt_fit.py  ->  examples/joint_nexi_mt_fit.png
Local figure -- nothing uploaded.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.signal_models.exchange_models import X0GeneralizedKarger
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.signal_models.gaussian_models import G2Zeppelin
from dmipy_fit.white_matter.magnetization_transfer import z_spectrum

MODEL = X0GeneralizedKarger(C1Stick(), G2Zeppelin())
TMS = np.linspace(0.02, 0.30, 8)
OFFS = np.array([1000., 2000., 4000., 8000., 16000., 32000.])
KAPPA_TRUE, KF_TRUE, K_R = 20.0, 0.4, 8.0
T1A, T1B, T2A, T2B, W1 = 1.0, 1.0, 0.05, 1e-5, 300.0
_KW = dict(f=0.5, mu=[0., 0.], C1Stick_1_lambda_par=1.7e-9,
           G2Zeppelin_1_lambda_par=1.7e-9, G2Zeppelin_1_lambda_perp=0.6e-9)
_SCHEMES = [AcquisitionScheme.from_pgste(np.array([2e9]), np.array([[1., 0., 0.]]),
                                         delta=5e-3, TM=float(t), TE=0.06, n_t=6000)
            for t in TMS]
_S0 = np.array([float(MODEL(s, kappa=0.0, **_KW)[0]) for s in _SCHEMES])
BIASED = dict(color="#e0651a")
JOINT = dict(color="#2563eb")


def exch(kappa):
    return np.array([float(MODEL(s, kappa=kappa, **_KW)[0]) for s in _SCHEMES]) / _S0


def zspec(k_f):
    return z_spectrum(OFFS, W1, k_f=k_f, k_r=K_R, T1a=T1A, T1b=T1B, T2a=T2A, T2b=T2B)


def fit_biased(pg):
    return least_squares(lambda k: exch(k[0]) - pg, [10.0], bounds=(0.0, 200.0)).x[0]


def fit_joint(pg, z):
    def resid(x):
        return np.concatenate([exch(x[0]) * np.exp(-x[1] * TMS) - pg, zspec(x[1]) - z])
    return least_squares(resid, [10.0, 0.1], bounds=([0.0, 0.0], [200.0, 5.0])).x


def main():
    pg_true = exch(KAPPA_TRUE) * np.exp(-KF_TRUE * TMS)
    z_true = zspec(KF_TRUE)
    rng = np.random.RandomState(1)

    snrs = [20, 40, 80, 160]
    N = 40
    kb_m, kb_s, kj_m, kj_s = [], [], [], []
    for snr in snrs:
        kb, kj = [], []
        for _ in range(N):
            pg = pg_true + rng.randn(len(TMS)) / snr
            z = z_true + rng.randn(len(OFFS)) / snr
            kb.append(fit_biased(pg))
            kj.append(fit_joint(pg, z)[0])
        kb_m.append(np.mean(kb)); kb_s.append(np.std(kb))
        kj_m.append(np.mean(kj)); kj_s.append(np.std(kj))

    # Panel A example (one SNR=40 realisation)
    pg = pg_true + rng.randn(len(TMS)) / 40.0
    z = z_true + rng.randn(len(OFFS)) / 40.0
    kb1 = fit_biased(pg)
    kj1 = fit_joint(pg, z)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.3))
    tm = TMS * 1e3

    axA.plot(tm, pg, "ko", ms=5, label="PGSTE data (SNR 40)")
    axA.plot(tm, exch(kb1), "--", color=BIASED["color"], lw=2, label=f"biased fit  κ={kb1:.0f}")
    axA.plot(tm, exch(kj1[0]) * np.exp(-kj1[1] * TMS), "-", color=JOINT["color"], lw=2,
             label=f"joint fit  κ={kj1[0]:.0f}, k$_f$={kj1[1]:.2f}")
    axA.set_xlabel("mixing time TM (ms)")
    axA.set_ylabel("exchange attenuation  $S(\\kappa)/S(0)$")
    axA.legend(frameon=False, fontsize=8.5, loc="upper right")
    axA.set_title("A  One noisy PGSTE realisation + both fits", fontsize=11, loc="left")

    axB.axhline(KAPPA_TRUE, color="0.6", ls=":", lw=1.2, label=f"true κ = {KAPPA_TRUE:.0f}")
    x = np.arange(len(snrs))
    axB.errorbar(x - 0.06, kb_m, yerr=kb_s, fmt="s-", capsize=4, lw=2, **BIASED,
                 label="PGSTE only  (biased)")
    axB.errorbar(x + 0.06, kj_m, yerr=kj_s, fmt="o-", capsize=4, lw=2, **JOINT,
                 label="joint + Z-spectrum")
    axB.set_xticks(x); axB.set_xticklabels([str(s) for s in snrs])
    axB.set_xlabel("SNR"); axB.set_ylabel("recovered exchange rate κ (s$^{-1}$)")
    axB.legend(frameon=False, fontsize=9, loc="center right")
    axB.set_title("B  Joint fit is unbiased across SNR", fontsize=11, loc="left")

    fig.suptitle("Joint NEXI + MT + Z-spectrum fit removes the PGSTE permeability bias",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(os.path.dirname(__file__), "joint_nexi_mt_fit.png")
    fig.savefig(out, dpi=140)
    for s, bm, bs, jm, js in zip(snrs, kb_m, kb_s, kj_m, kj_s):
        print(f"SNR {s:>3}: biased κ={bm:5.1f}±{bs:4.1f}   joint κ={jm:5.1f}±{js:4.1f}  (true 20)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
