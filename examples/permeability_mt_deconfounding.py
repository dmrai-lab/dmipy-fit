"""Paper 2 headline: MT biases PGSTE permeability, the qMT Z-spectrum removes it.

PGSTE encodes membrane permeability (grey-matter water exchange) in the mixing-time
(TM) signal decay -- and, unlike PGSE, the transverse confounds (susceptibility,
surface relaxivity) are gated off during longitudinal storage.  But MT is the one
effect still active during TM: it adds exp(-k_f*TM) over the SAME mixing time.

Panel A -- decomposition: the measured exchange-weighted decay is the true exchange
attenuation TIMES the MT factor, so MT masquerades as extra exchange.
Panel B: fitting the exchange rate WITHOUT an MT term over-estimates it more and more
with k_f; supplying k_f (measured independently by the qMT Z-spectrum) recovers the
truth, flat.  The Z-spectrum breaks the tie -- the same lever that lifts the rho/kappa
degeneracy in paper 1.

We use the exchange ATTENUATION ratio S(kappa)/S(kappa=0) (the standard exchange
observable): it isolates exchange from the diffusion baseline.  n_t=6000 keeps the
per-scheme gradient calibration smooth.

Run (fit venv): python examples/permeability_mt_deconfounding.py
Writes examples/permeability_mt_deconfounding.png.  Local figure -- nothing uploaded.
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

MODEL = X0GeneralizedKarger(C1Stick(), G2Zeppelin())
TMS = np.linspace(0.02, 0.30, 12)
F, DPAR, DPERP, B, KAPPA_TRUE = 0.5, 1.7e-9, 0.6e-9, 2e9, 20.0
BIASED = dict(color="#e0651a")
CORR = dict(color="#2563eb")

# build one PGSTE scheme per TM ONCE (n_t high -> smooth b-calibration), + the
# no-exchange baseline S(kappa=0) for the exchange attenuation ratio.
_SCHEMES = [AcquisitionScheme.from_pgste(np.array([B]), np.array([[1., 0., 0.]]),
                                         delta=5e-3, TM=float(TM), TE=0.06, n_t=6000)
            for TM in TMS]


def _karger(sch, kappa):
    return float(MODEL(sch, kappa=kappa, f=F, mu=[0., 0.], C1Stick_1_lambda_par=DPAR,
                       G2Zeppelin_1_lambda_par=DPAR, G2Zeppelin_1_lambda_perp=DPERP)[0])


_S0 = np.array([_karger(s, 0.0) for s in _SCHEMES])            # no-exchange baseline


def exchange_ratio(kappa):
    return np.array([_karger(s, kappa) for s in _SCHEMES]) / _S0


def fit_kappa(resid):
    return float(least_squares(resid, x0=[10.0], bounds=(0.0, 200.0)).x[0])


def main():
    r_true = exchange_ratio(KAPPA_TRUE)

    # Panel A: decomposition at one representative k_f
    kf_A = 0.4
    mt_A = np.exp(-kf_A * TMS)
    measured = r_true * mt_A

    # Panel B: sweep k_f -> biased (no MT) vs corrected (MT known) exchange rate
    kfs = np.linspace(0.0, 0.6, 13)
    kb, kc = [], []
    for kf in kfs:
        mt = np.exp(-kf * TMS)
        m = r_true * mt
        kb.append(fit_kappa(lambda k: exchange_ratio(k[0]) - m))
        kc.append(fit_kappa(lambda k: exchange_ratio(k[0]) * mt - m))
    kb, kc = np.array(kb), np.array(kc)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.3))
    tm = TMS * 1e3

    axA.plot(tm, r_true, "-", color=CORR["color"], lw=2, label=f"exchange only (true κ={KAPPA_TRUE:.0f})")
    axA.plot(tm, mt_A, "-", color="0.55", lw=2, label=f"MT factor  exp(−k$_f$·TM), k$_f$={kf_A}")
    axA.plot(tm, measured, "ko-", ms=4, lw=1.5, label="measured = exchange × MT")
    axA.set_xlabel("mixing time TM (ms)")
    axA.set_ylabel("exchange attenuation  $S(\\kappa)/S(0)$")
    axA.set_ylim(0.7, 1.005)
    axA.legend(frameon=False, fontsize=8.5, loc="lower left")
    axA.set_title("A  MT adds a TM-decay that mimics exchange", fontsize=11, loc="left")

    axB.axhline(KAPPA_TRUE, color="0.6", ls=":", lw=1.2, label=f"true κ = {KAPPA_TRUE:.0f}")
    axB.plot(kfs, kb, "s-", **BIASED, lw=2, ms=5, label="fit WITHOUT MT  (biased)")
    axB.plot(kfs, kc, "o-", **CORR, lw=2, ms=5, label="fit WITH MT  (Z-spectrum k$_f$)")
    axB.set_xlabel("MT forward rate k$_f$ (s$^{-1}$)")
    axB.set_ylabel("recovered exchange rate κ (s$^{-1}$)")
    axB.legend(frameon=False, fontsize=9, loc="upper left")
    axB.set_title("B  Z-spectrum k$_f$ removes the bias", fontsize=11, loc="left")

    fig.suptitle("MT confounds PGSTE permeability; the qMT Z-spectrum de-biases it", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(os.path.dirname(__file__), "permeability_mt_deconfounding.png")
    fig.savefig(out, dpi=140)
    print(f"Panel B: biased κ {kb.min():.0f}–{kb.max():.0f}, corrected κ {kc.min():.2f}–{kc.max():.2f} "
          f"(true {KAPPA_TRUE:.0f})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
