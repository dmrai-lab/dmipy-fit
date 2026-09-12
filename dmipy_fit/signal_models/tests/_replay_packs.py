"""Tiny replay packs for the fit's replay tests, built with the public dmipy-sim producer (a walk on an analytic
pore, `build_replay_pack`), so the tests run wherever dmipy-sim is installed."""
import numpy as np


def build_public_pack(shape, diameter_m, D0, *, n_t=150, n_walkers=1500, seed=7, K=48, blt_temporal_K=32,
                      T_max=60e-3):
    import dmipy_sim as d
    from dmipy_sim.replay.bank import build_replay_pack
    r = float(diameter_m) / 2
    g = {"sphere": lambda: d.Sphere(radius=r),
         "cylinder": lambda: d.Cylinder(radius=r, orientation=(0, 0, 1))}[shape]()
    dt = float(T_max) / (int(n_t) - 1)
    walk = d.simulate_trajectories(int(n_walkers), float(D0), g, float(T_max), dt, seed=int(seed), require_gpu=False)
    return build_replay_pack(walk, id=f"test/{shape}/d{diameter_m * 1e6:05.2f}um", license="CC0", citation="test",
                             K=int(K), blt_temporal_K=int(blt_temporal_K),
                             provenance={"diameter_m": float(diameter_m), "shape": shape})
