"""Tiny replay packs for the fit's replay tests: a walk on an analytic pore kept as its bands
(`dmipy_sim.simulate_trajectories`, the band deriving the block) and packed by the public producer
(`build_replay_pack`), so the tests run wherever dmipy-sim is installed. A pack is built once per process."""
_BUILT = {}


def build_public_pack(shape, diameter_m, D0, *, n_walkers=1500, seed=7, K=48, blt_temporal_K=32, T_max=60e-3):
    key = (shape, float(diameter_m), float(D0), int(n_walkers), int(seed), int(K), int(blt_temporal_K), float(T_max))
    if key not in _BUILT:
        import dmipy_sim as d
        from dmipy_sim.replay.bank import build_replay_pack
        r = float(diameter_m) / 2
        g = {"sphere": lambda: d.Sphere(radius=r),
             "cylinder": lambda: d.Cylinder(radius=r, orientation=(0, 0, 1)),
             "plane": lambda: d.Box1D(length=float(diameter_m))}[shape]()
        walk = d.simulate_trajectories(int(n_walkers), float(D0), g, float(T_max), seed=int(seed), K=int(K),
                                       require_gpu=False)
        _BUILT[key] = build_replay_pack(walk, id=f"test/{shape}/d{diameter_m * 1e6:05.2f}um", license="CC0",
                                        citation="test", K=int(K), blt_temporal_K=int(blt_temporal_K),
                                        provenance={"diameter_m": float(diameter_m), "shape": shape})
    return _BUILT[key]
