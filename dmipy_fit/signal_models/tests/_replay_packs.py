"""Tiny replay packs for the fit's replay tests: a walk on an analytic pore kept as its bands
(`dmipy_sim.walk`, the band deriving the block) and packed by the public producer
(`build_replay_pack`), so the tests run wherever dmipy-sim is installed. A pack is built once per process."""
_BUILT = {}
#: the scanner class the test walks are walked for: its band sizes the walk's step and block
WALK_CLASS = "connectome_2"
#: the envelope the test packs state (a pack replays only acquisitions inside it): the analytical models are compared
#: on idealised square lobes of up to a few hundred mT/m, so its band and amplitude are the tests' own, beyond every
#: catalogue scanner
TEST_BAND_HZ = 1e9
TEST_G_MAX = 10.0


def walk_envelope():
    """The envelope a test walk is walked for: the ``WALK_CLASS`` preset (:func:`dmipy_replay.envelope_of`) with the
    contact tier up to 1e-4 m/s and no field tier."""
    from dmipy_replay import envelope_of
    return envelope_of(WALK_CLASS, B0_list=(3.0,), rho2_max=1e-4)


def stated_envelope():
    """The tests' :class:`dmipy_replay.Envelope`: a band wider than any test pack stores (a pack states the band it
    stores), ``TEST_G_MAX`` T/m per axis, the contact tier up to 1e-4 m/s, no field tier (the test packs carry none)."""
    from dmipy_replay import Envelope
    return Envelope(name="dmipy-fit tests", band_hz=TEST_BAND_HZ, G_max=TEST_G_MAX, B0_list=(3.0,), chi_max=None,
                    rho2_max=1e-4, max_refocusing_pulses=None)


def build_public_pack(shape, diameter_m, D0, *, n_walkers=1500, seed=7, K=48, blt_temporal_K=32, T_max=60e-3):
    key = (shape, float(diameter_m), float(D0), int(n_walkers), int(seed), int(K), int(blt_temporal_K), float(T_max))
    if key not in _BUILT:
        import dmipy_sim as d
        from dmipy_sim.replay.bank import build_replay_pack
        r = float(diameter_m) / 2
        g = {"sphere": lambda: d.Sphere(radius=r),
             "cylinder": lambda: d.Cylinder(radius=r, orientation=(0, 0, 1)),
             "plane": lambda: d.Box1D(length=float(diameter_m))}[shape]()
        walk = d.walk(d.spec.spec_of(g), int(n_walkers), float(T_max), diffusivity=float(D0), seed=int(seed),
                      K=int(K), require_gpu=False, envelope=walk_envelope())
        _BUILT[key] = build_replay_pack(walk, id=f"test/{shape}/d{diameter_m * 1e6:05.2f}um", license="CC0",
                                        citation="test", K=int(K), blt_temporal_K=int(blt_temporal_K),
                                        envelope=stated_envelope(),
                                        provenance={"diameter_m": float(diameter_m), "shape": shape})
    return _BUILT[key]
