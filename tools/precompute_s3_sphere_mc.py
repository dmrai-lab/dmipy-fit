#!/usr/bin/env python
"""Regenerate the reflecting-sphere Monte Carlo reference for _S3SphereCallaghan.

The finite-time Callaghan sphere (``_S3SphereCallaghanApproximation``) is an SGP
(narrow-pulse) series.  Its excited-mode decay — not just the tau->0 / tau->inf
limits — is validated here against a dmipy-sim reflecting-sphere PGSE Monte-Carlo
in the *strongly attenuated* regime (E from ~0.72 down to ~0.09).  As the pulse
width delta -> 0 the finite-pulse (SGP) bias vanishes and the Monte-Carlo signal
converges monotonically onto the analytic series.

This script bakes that seed-averaged MC curve into a small offline fixture so the
test suite can assert the convergence without re-running any random walk.  It
needs a CUDA GPU and the ``dmipy-sim`` package.

    python tools/precompute_s3_sphere_mc.py

Fixture written to dmipy_fit/core/tests/fixtures/:
    s3_sphere_callaghan_mc.npz   arrays: q, deltas, E_mc[delta, q], E_s3[q]
    s3_sphere_callaghan_mc.yaml  human-readable provenance
"""
from __future__ import annotations

import os
import numpy as np

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

from dmipy_sim import simulate, Sphere
from dmipy_sim.waveforms import pgse
from dmipy_sim.constants import GAMMA
from dmipy_fit.signal_models.sphere_models import _S3SphereCallaghanApproximation

# --- physics (kept in sync with TestS3SphereCallaghan) -----------------------
D = 1.7e-9                                  # m^2/s, water-in-axons default
R = 5e-6                                    # sphere radius (m)
DIAMETER = 2 * R
TAU = 40e-3                                 # diffusion time Delta - delta/3 (s)
Q_TARGETS = np.array([4e4, 6e4, 8e4, 1.0e5])   # 1/m; E spans 0.72 -> 0.09
DELTAS = np.array([0.2e-3, 0.1e-3, 0.05e-3])   # narrow-pulse sweep (<< R^2/D=14.7ms)
N_WALKERS = 500_000
N_T = 12_000                                # dt=4us -> step ~ R/25 (sub-stepped)
SEEDS = [11, 22, 33, 44]                    # seed-averaged: eff noise ~7e-4

OUT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..',
    'dmipy_fit', 'core', 'tests', 'fixtures'))


def main():
    s3 = _S3SphereCallaghanApproximation(diffusion_constant=D)
    E_s3 = s3.sphere_attenuation(Q_TARGETS, np.full_like(Q_TARGETS, TAU), DIAMETER)
    bvecs = np.tile([1., 0., 0.], (len(Q_TARGETS), 1))

    E_mc = np.zeros((len(DELTAS), len(Q_TARGETS)))
    for i, delta in enumerate(DELTAS):
        DELTA = TAU + delta / 3.0                  # so tau = Delta - delta/3
        G = 2 * np.pi * Q_TARGETS / (GAMMA * delta)   # square pulses -> exact delta
        wf = pgse(delta=delta, DELTA=DELTA, G_magnitude=G, bvecs=bvecs,
                  n_t=N_T, slew_rate=np.inf)
        runs = [simulate(N_WALKERS, D, wf, Sphere(R), seed=s, require_gpu=True)
                for s in SEEDS]
        E_mc[i] = np.mean(runs, axis=0)
        print(f"delta={delta*1e3:.3f} ms  max|E_MC-E_S3|={np.max(np.abs(E_mc[i]-E_s3)):.6f}")

    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, 's3_sphere_callaghan_mc.npz'),
             q=Q_TARGETS, deltas=DELTAS, tau=TAU, radius=R, D=D,
             E_mc=E_mc, E_s3=E_s3)

    with open(os.path.join(OUT, 's3_sphere_callaghan_mc.yaml'), 'w') as f:
        f.write(
            "# Reflecting-sphere PGSE Monte-Carlo reference for _S3SphereCallaghan.\n"
            "# Regenerate: python tools/precompute_s3_sphere_mc.py  (needs a CUDA GPU).\n"
            "geometry: reflecting Sphere, radius 5e-6 m (impermeable)\n"
            f"diffusion_constant_m2_s: {D}\n"
            f"tau_s: {TAU}          # Delta - delta/3\n"
            f"q_targets_per_m: {Q_TARGETS.tolist()}\n"
            f"deltas_s: {DELTAS.tolist()}   # narrow-pulse sweep, square lobes\n"
            f"n_walkers: {N_WALKERS}\n"
            f"n_t: {N_T}\n"
            f"seeds: {SEEDS}       # seed-averaged; effective MC noise ~7e-4\n"
            "engine: dmipy-sim (GPU Bloch-Torrey random walk), GAMMA=267.513e6\n"
            "claim: E_MC -> E_S3 monotonically as delta -> 0 (finite-pulse SGP bias)\n"
        )
    print(f"wrote fixture to {OUT}")


if __name__ == '__main__':
    main()
