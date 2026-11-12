"""The parity record of the signal forms of dmipy-fit: every signal model's output on fixed acquisition grids and
parameters, written bit for bit, for NumPy and for JAX (float32 and float64) on the CPU.

``record/`` holds the outputs of dmipy-fit's own equations (``numpy.npz``, ``jax_f32.npz``, ``jax_f64.npz``: one array
per case) and ``record/manifest.json``: the sha256 of every array (its dtype and shape included), whether the case is
bit-reproducible, the tolerance a case that is not is held to, and the environment the record was made in. The cases
call dmipy-fit's public model classes (``__call__``, ``spherical_mean``, ``rotational_harmonics_representation``,
``signal_lm``, ``use_jax=True``), its JAX forward factories and the standalone forms it exports, so the record holds
whichever module implements the equation. ``FORMS`` names where each standalone form lives; it is the one place a
move of a form is written down.

Commands (the interpreter needs dmipy-fit, dmipy-sim, dmipy-replay, numpy, scipy and jax; numba must be absent, since
its jit of a form is a second implementation of it):

* ``python tests/parity_analytic/harness.py check``: emit every case in a fresh process under the pinned environment
  (:data:`PINNED_ENV`) and compare with ``record/``; exit status 1 if any case differs. The record is bit for bit on
  the machine that made it; another x86-64 machine can differ in the last bits of the libm and XLA routines the
  forms call (a GitHub runner differs from the maker's laptop in about a third of the cases, by one ulp in a few
  elements), so CI compares against a reference recorded on the runner itself, as follows.
* ``python tests/parity_analytic/harness.py check --reference DIR [--base-manifest FILE]``: compare with the outputs
  that ``record`` wrote to ``DIR`` on this machine (CI records the base revision of the pull request, in the
  environment that revision pins, with ``record`` and then checks the revision under test). Every case must be bit
  for bit equal, a ``host_dependent`` case within its derived ``atol``, except the cases whose entry in
  ``record/manifest.json`` differs from the one in ``FILE`` (the base revision's committed manifest) or exists in
  only one of them: those are the deliberate changes of the pull request, and their measured differences are
  printed.
* ``python tests/parity_analytic/harness.py record``: emit every case twice, write ``record/``; a case whose two runs
  differ is recorded as not bit-reproducible with its derived tolerance.
* ``python tests/parity_analytic/harness.py rebaseline PREFIX...``: emit every case once and replace the record's
  entries of the cases whose id starts with a prefix, drop the entries of cases that are no longer registered, and
  print how far each replaced entry moved (max abs, max relative, max ulp). Every other entry is left as it was; one
  that differs without a prefix is reported and the record is not written.
* ``python tests/parity_analytic/harness.py rebaseline --only KEY...``: emit only the named cases (a KEY is a case id or
  an ``fnmatch`` pattern such as ``jax_f32/C1Stick/*``; each must match a registered case or a recorded entry) and
  replace exactly those entries and arrays; a matching recorded entry whose case is no longer registered is dropped
  (entry and array). Nothing else is read for drift, so the environment drift of the other cases (another Python,
  NumPy or JAX than the record's) does not matter; the manifest's ``environment`` block stays as it was, and the
  manifest's ``rewrites`` list records which keys were written in which environment.
* ``python tests/parity_analytic/harness.py rename-case OLD NEW``: rename the keys of case ``OLD`` (the id, and every id
  below ``OLD/``) to ``NEW`` in the manifest and the ``.npz`` files, array bytes untouched, and verify that every
  renamed array has its old sha256 under its new name. ``OLD`` and ``NEW`` without a section prefix apply to every
  section (``C1Stick/legacy`` renames ``numpy/``, ``jax_f32/`` and ``jax_f64/`` ids alike).
* ``python tests/parity_analytic/harness.py inputs``: write ``inputs.npz``, the fixed arrays the cases read.
* ``python tests/parity_analytic/harness.py list``: the case ids.

Tolerance of a case that is not bit-reproducible: ``ops`` is the number of rounding operations between the inputs and
the output, declared at the case, and ``eps`` the machine epsilon of the output dtype. A case whose two runs differ is
held to ``rtol = ops * eps * max(1, -log(min E))`` (the ``-log E`` factor is the rounding of the exponent that ``exp``
carries to the output). A case that declares ``host_dependent`` is held to ``atol = ops * eps``: XLA compiles the
complex matrix products of the JAX matrix method for the host CPU, so their rounding differs between x86-64 machines
(bit-identical on the laptop, different at the 1e-13 level on a GitHub runner), and the propagation is a contraction
of a unit state, so its rounding is absolute.
"""
import hashlib
import importlib
import fnmatch
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD = os.path.join(HERE, "record")
SECTIONS = ("numpy", "jax_f32", "jax_f64")

# The environment the cases are emitted under: one BLAS thread (``import dmipy_sim`` caps BLAS at eight unless
# ``DMIPY_REPLAY_CPU_THREADS`` says otherwise), the AVX-512 code paths of numpy off (the baseline
# AVX2 paths are the same on every x86-64 machine that runs CI), OpenBLAS's Haswell kernels, XLA's CPU backend with
# single-threaded Eigen contractions and no instruction set above AVX2.
PINNED_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "DMIPY_REPLAY_CPU_THREADS": "1",
    "OPENBLAS_CORETYPE": "Haswell",
    "NPY_DISABLE_CPU_FEATURES": "AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL "
                                "AVX512_ICL AVX512_SPR",
    "JAX_PLATFORMS": "cpu",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_cpu_max_isa=AVX2",
    "PYTHONHASHSEED": "0",
}

# ---------------------------------------------------------------------------------------------------------------------
# The standalone forms the record calls: id -> (module, attribute). A form that moves is re-pointed here (with an
# adapter lambda where its arguments changed); the record does not change.
# ---------------------------------------------------------------------------------------------------------------------
FORMS = {
    "surface.b_hat_ia": ("dmipy_analytic.relaxation", "surface_relaxation_gamma_cylinders"),
    "surface.b_hat_sphere": ("dmipy_analytic.relaxation", "surface_relaxation_gamma_spheres"),
    "surface.b_hat_ea_long": ("dmipy_analytic.relaxation", "surface_relaxation"),
    "surface.exterior_surface_to_volume": ("dmipy_analytic.relaxation", "exterior_surface_to_volume"),
    "distributions.odi2kappa": ("dmipy_fit.distributions.distributions", "odi2kappa"),
    "distributions.kappa2odi": ("dmipy_fit.distributions.distributions", "kappa2odi"),
    "sh_analytical.gaussian_signal_sh": ("dmipy_fit.utils.sh_analytical", "gaussian_signal_sh"),
    "sh_analytical.gaussian_signal_lm": ("dmipy_fit.utils.sh_analytical", "gaussian_signal_lm"),
    "sh_analytical.gaussian_J_l": ("dmipy_fit.utils.sh_analytical", "gaussian_J_l"),
    "sh_analytical.gaussian_kernel_rh": ("dmipy_fit.utils.sh_analytical", "gaussian_kernel_rh"),
    "convolution.real_sym_rh_basis": ("dmipy_fit.utils.spherical_convolution", "real_sym_rh_basis"),
    "convolution.sh_convolution": ("dmipy_fit.utils.spherical_convolution", "sh_convolution"),
    "cylinder.E_lm_from_exponent_coeffs": ("dmipy_fit.signal_models.cylinder_models", "E_lm_from_exponent_coeffs"),
    "cylinder.dispersed_signal_from_E_lm": ("dmipy_fit.signal_models.cylinder_models", "dispersed_signal_from_E_lm"),
    "cylinder.watson_odf_lm": ("dmipy_analytic.orientation", "watson_sh"),
}


def form(form_id):
    """The callable of a standalone form, imported on first use."""
    module, name = FORMS[form_id]
    return getattr(importlib.import_module(module), name)


# ---------------------------------------------------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------------------------------------------------
REGISTRY = {}     # case id -> Case


class Case:
    """One recorded output: ``thunk()`` returns an array. ``section`` is ``numpy``, ``jax_f32`` or ``jax_f64``;
    ``ops`` the rounding operations of the output (the tolerance of a case that is not bit-reproducible);
    ``trivial`` admits a constant output (a b = 0 acquisition or the dot); ``host_dependent`` marks a case whose
    rounding depends on the host CPU (held to its derived tolerance, not to the digest)."""

    def __init__(self, cid, section, thunk, ops=8, trivial=False, host_dependent=False):
        self.cid, self.section, self.thunk, self.ops, self.trivial = cid, section, thunk, ops, trivial
        self.host_dependent = host_dependent


def add(section, cid, thunk, **kw):
    if cid in REGISTRY:
        raise ValueError("duplicate case " + cid)
    REGISTRY[cid] = Case(cid, section, thunk, **kw)


# ---------------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------------
INPUTS_PATH = os.path.join(HERE, "inputs.npz")
_inputs_cache = {}


def make_inputs(path=INPUTS_PATH):
    """Write the fixed arrays the cases read: unit gradient directions (the z and x axes first in every shell, so
    the parallel and perpendicular branches of every oriented form are hit), a spherical-harmonic coefficient
    vector and a rotational-harmonic kernel."""
    import numpy as np
    rng = np.random.default_rng(834)

    def unit(n):
        d = rng.normal(size=(n, 3))
        return d / np.linalg.norm(d, axis=1)[:, None]
    z, x = np.array([0., 0., 1.]), np.array([1., 0., 0.])
    dirs = np.vstack([np.vstack([z, z])] + [np.vstack([z, x, unit(4)]) for _ in range(5)])
    dirs_ogse = np.vstack([z, z, z, x, unit(1), z, x, unit(1)])
    sh_coeff = rng.normal(size=45) * 0.1
    sh_coeff[0] = 1.0 / (2.0 * np.sqrt(np.pi))
    np.savez(path, dirs=dirs, dirs_ogse=dirs_ogse, sh_coeff=sh_coeff, rh_kernel=rng.uniform(0.1, 1.0, size=5),
             unit_vectors=unit(11))


def inputs():
    if not _inputs_cache:
        import numpy as np
        with np.load(INPUTS_PATH) as f:
            _inputs_cache.update({k: f[k] for k in f.files})
    return _inputs_cache


# ---------------------------------------------------------------------------------------------------------------------
# Acquisition grids
# ---------------------------------------------------------------------------------------------------------------------
N_T = 1000                                                                           # samples per waveform
B_PGSE = [0., 0.] + [5e8] * 6 + [1e9] * 6 + [2e9] * 6 + [3e9] * 6 + [5e9] * 6      # s/m^2
B_OGSE = [0., 0., 3e8, 3e8, 3e8, 6e8, 6e8, 6e8]
F_OGSE = [0., 0., 50., 50., 50., 100., 100., 100.]                                   # Hz
_scheme_cache = {}


def _timing():
    """A spin-echo timing budget with finite RF pulses (2 ms excitation, 6 ms refocusing window, 4 ms readout)."""
    from dmipy_replay.acquisition.timing import SequenceTiming
    return SequenceTiming(t_excite=0.002, t_refocus=0.006, t_readout_pre_echo=0.004)


def scheme(name):
    """The acquisition scheme ``name``: ``pgse``/``pgse2`` (one / two timings; the waveform-first scheme),
    ``legacy``/``legacy2`` (the waveform-free scheme), ``ogse`` (cosine), ``ogse_trap`` (trapezoid), ``mixed``
    (PGSE and OGSE measurements), ``ste``, ``pte`` (b-tensor encodings), ``pgste`` (stimulated echo), ``pgse_rf`` and
    ``pgste_rf`` (the same with finite RF pulses)."""
    if name not in _scheme_cache:
        import numpy as np
        from dmipy_fit.core.acquisition_scheme import AcquisitionScheme, acquisition_scheme_from_bvalues
        I = inputs()
        b, dirs, n = np.array(B_PGSE), I["dirs"], len(B_PGSE)
        even = np.arange(n) % 2 == 0
        delta2, Delta2 = np.where(even, 0.010, 0.016), np.where(even, 0.030, 0.040)
        bo, do, fo = np.array(B_OGSE), I["dirs_ogse"], np.array(F_OGSE)
        build = {
            "pgse": lambda: AcquisitionScheme.from_pgse(b, dirs, 0.012, 0.032, TE=0.075),
            "pgse2": lambda: AcquisitionScheme.from_pgse(b, dirs, delta2, Delta2, TE=0.09),
            "legacy": lambda: acquisition_scheme_from_bvalues(b, dirs, 0.012, 0.032, TE=0.075),
            "legacy2": lambda: acquisition_scheme_from_bvalues(b, dirs, delta2, Delta2, TE=0.09),
            "ogse": lambda: AcquisitionScheme.from_ogse(bo, do, fo, 0.02, shape="cosine"),
            "ogse_trap": lambda: AcquisitionScheme.from_ogse(
                bo, do, np.array([0., 0., 50., 50., 50., 50., 50., 50.]), 0.02, shape="trapezoid", slew_rate=300.),
            "mixed": lambda: scheme("pgse") + scheme("ogse"),
            "ste": lambda: AcquisitionScheme.from_btensor_ste(np.array([0., 1e9, 2e9, 3e9]), 0.03),
            "pte": lambda: AcquisitionScheme.from_btensor_pte(np.array([0., 1e9, 2e9, 3e9]), [0., 0., 1.], 0.03),
            "pgste": lambda: AcquisitionScheme.from_pgste(b, dirs, 0.01, 0.1, TE=0.13),
            "pgse_rf": lambda: AcquisitionScheme.from_pgse(b, dirs, 0.010, 0.025, TE=0.08, timing=_timing()),
            "pgste_rf": lambda: AcquisitionScheme.from_pgste(b, dirs, 0.01, 0.1, TE=0.13, timing=_timing()),
        }
        _scheme_cache[name] = build[name]()
    return _scheme_cache[name]


# ---------------------------------------------------------------------------------------------------------------------
# Parameters (SI)
# ---------------------------------------------------------------------------------------------------------------------
PI = 3.141592653589793
MU = {"A": [0.7, 1.1], "B": [0.0, 0.0], "C": [PI / 2, 0.3]}


def _params(kind):
    mu = MU
    if kind == "stick":
        return {"A": dict(mu=mu["A"], lambda_par=1.7e-9), "B": dict(mu=mu["B"], lambda_par=2.5e-9),
                "C": dict(mu=mu["C"], lambda_par=1.0e-9)}
    if kind == "zeppelin":
        return {"A": dict(mu=mu["A"], lambda_par=1.7e-9, lambda_perp=0.6e-9),
                "B": dict(mu=mu["B"], lambda_par=2.5e-9, lambda_perp=1.2e-9),
                "C": dict(mu=mu["C"], lambda_par=1.0e-9, lambda_perp=0.3e-9)}
    if kind == "temporal_zeppelin":
        return {"A": dict(mu=mu["A"], lambda_par=1.7e-9, lambda_inf=0.6e-9, A=2e-12),
                "B": dict(mu=mu["B"], lambda_par=2.5e-9, lambda_inf=1.2e-9, A=0.5e-12)}
    if kind == "ball":
        return {"A": dict(lambda_iso=3e-9), "B": dict(lambda_iso=0.7e-9)}
    if kind == "cylinder":
        return {"A": dict(mu=mu["A"], lambda_par=1.7e-9, diameter=4e-6),
                "B": dict(mu=mu["B"], lambda_par=2.5e-9, diameter=0.8e-6),
                "C": dict(mu=mu["C"], lambda_par=1.0e-9, diameter=9e-6)}
    if kind in ("sphere", "plane"):
        return {"A": dict(diameter=10e-6), "B": dict(diameter=3e-6), "C": dict(diameter=18e-6)}
    if kind == "capped":
        return {"A": dict(mu=mu["A"], diameter=4e-6, length=20e-6), "B": dict(mu=mu["B"], diameter=1e-6, length=8e-6)}
    raise KeyError(kind)


def _import(path):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


SM = "dmipy_fit.signal_models."
# model key -> (class path, constructor kwargs, parameter kind, schemes of the NumPy call)
NUMPY_MODELS = [
    ("G1Ball", SM + "gaussian_models.G1Ball", {}, "ball", ["pgse", "legacy", "ste", "pgste"]),
    ("G2Zeppelin", SM + "gaussian_models.G2Zeppelin", {}, "zeppelin", ["pgse", "legacy", "ste", "pte", "ogse"]),
    ("G3TemporalZeppelin", SM + "gaussian_models.G3TemporalZeppelin", {}, "temporal_zeppelin",
     ["pgse", "pgse2", "legacy2"]),
    ("C1Stick", SM + "cylinder_models.C1Stick", {}, "stick", ["pgse", "legacy", "ste", "pte"]),
    ("C2", SM + "cylinder_models.C2CylinderStejskalTannerApproximation", {}, "cylinder",
     ["pgse", "legacy", "pgse2"]),
    ("C3", SM + "cylinder_models.C3CylinderCallaghanApproximation", {}, "cylinder", ["pgse", "legacy", "pgse2"]),
    ("C3x", SM + "cylinder_models.C3CylinderCallaghanApproximation",
     dict(diffusion_perpendicular=1.2e-9, number_of_roots=12, number_of_functions=20), "cylinder", ["legacy2"]),
    ("C4", SM + "cylinder_models.C4CylinderGaussianPhaseApproximation", {}, "cylinder",
     ["pgse", "pgse2", "legacy", "legacy2", "ogse", "ogse_trap", "mixed", "ste", "pte"]),
    ("C4x", SM + "cylinder_models.C4CylinderGaussianPhaseApproximation", dict(diffusion_perpendicular=1.2e-9),
     "cylinder", ["pgse2"]),
    ("C5", SM + "cylinder_models.C5CylinderMatrixMethod", {}, "cylinder", ["pgse", "legacy", "ogse"]),
    ("S1Dot", SM + "sphere_models.S1Dot", {}, None, ["pgse", "legacy"]),
    ("S2", SM + "sphere_models.S2SphereStejskalTannerApproximation", {}, "sphere", ["pgse", "legacy"]),
    ("S3", SM + "sphere_models.S3SphereCallaghanApproximation", {}, "sphere", ["pgse", "legacy", "pgse2"]),
    ("S4", SM + "sphere_models.S4SphereGaussianPhaseApproximation", {}, "sphere",
     ["pgse", "pgse2", "legacy", "legacy2", "ogse", "ogse_trap", "mixed", "ste", "pte"]),
    ("S4x", SM + "sphere_models.S4SphereGaussianPhaseApproximation", dict(diffusion_constant=1.58e-9), "sphere",
     ["pgse2"]),
    ("S5", SM + "sphere_models.S5SphereMatrixMethod", {}, "sphere", ["pgse", "legacy", "ogse"]),
    ("P2", SM + "plane_models.P2PlaneStejskalTannerApproximation", {}, "plane", ["pgse", "legacy"]),
    ("P3", SM + "plane_models.P3PlaneCallaghanApproximation", {}, "plane", ["pgse", "legacy", "pgse2"]),
    ("P4", SM + "plane_models.P4PlaneGaussianPhaseApproximation", {}, "plane", ["pgse", "legacy", "pgse2"]),
    ("P5", SM + "plane_models.P5PlaneMatrixMethod", {}, "plane", ["pgse", "legacy", "ogse"]),
    ("CC2", SM + "capped_cylinder_models.CC2CappedCylinderStejskalTannerApproximation", {}, "capped",
     ["pgse", "legacy"]),
    ("CC3", SM + "capped_cylinder_models.CC3CappedCylinderCallaghanApproximation", {}, "capped",
     ["pgse", "legacy", "pgse2"]),
]
# the models of NUMPY_MODELS that need float64 on JAX (the Bessel / Van Gelderen sums overflow float32)
FLOAT64_ONLY = {"C2", "C3", "C3x", "C4", "C4x"}


def _call(cls_path, kwargs, sch, params, **call_kw):
    import numpy as np
    return np.asarray(_import(cls_path)(**kwargs)(scheme(sch), **params, **call_kw))


def _numpy_model_cases():
    for key, cls_path, kwargs, kind, schemes in NUMPY_MODELS:
        sets = _params(kind) if kind else {"A": {}}
        for sch in schemes:
            for pk, params in sets.items():
                if pk != "A" and key.endswith("x"):
                    continue
                add("numpy", f"numpy/{key}/{sch}/{pk}",
                    lambda c=cls_path, k=kwargs, s=sch, p=params: _call(c, k, s, p),
                    trivial=(kind is None or (key == "C1Stick" and sch == "pte" and pk == "B")))


# method outputs of the Gaussian and the cylinder models other than __call__
def _method_cases():
    import numpy as np
    for key, kind, schemes, methods in [
        ("G2Zeppelin", "zeppelin", ["pgse", "legacy"], ["spherical_mean", "rotational_harmonics_representation"]),
        ("G2Zeppelin", "zeppelin", ["pgse", "ste", "pte"], ["signal_lm"]),
        ("G3TemporalZeppelin", "temporal_zeppelin", ["pgse2", "legacy2"],
         ["spherical_mean", "rotational_harmonics_representation"]),
        ("C1Stick", "stick", ["pgse", "legacy"], ["spherical_mean", "rotational_harmonics_representation"]),
        ("C1Stick", "stick", ["pgse", "ste", "pte"], ["signal_lm"]),
        ("C4", "cylinder", ["pgse", "ste", "pte"], ["signal_lm", "signal_from_gamma_lm"]),
    ]:
        entry = next(m for m in NUMPY_MODELS if m[0] == key)
        for sch in schemes:
            for meth in methods:
                for pk, params in _params(kind).items():
                    if pk == "C":
                        continue
                    # the oriented methods of the Gaussian models take the parameters, not the orientation
                    p = {k: v for k, v in params.items() if not (k == "mu" and meth in (
                        "spherical_mean", "rotational_harmonics_representation"))}
                    add("numpy", f"numpy/{key}.{meth}/{sch}/{pk}",
                        lambda c=entry[1], k=entry[2], s=sch, m=meth, pp=p:
                        np.asarray(getattr(_import(c)(**k), m)(scheme(s), **pp)))


def _karger_rh_cases():
    import numpy as np
    from dmipy_fit.signal_models import cylinder_models as cm, gaussian_models as gm
    from dmipy_fit.signal_models.exchange_models import X0GeneralizedKarger
    from dmipy_fit.core.modeling_framework import MultiCompartmentModel

    def rh(sch):
        karger = X0GeneralizedKarger(cm.C1Stick(), gm.G2Zeppelin())
        mcm = MultiCompartmentModel([karger])
        params = mcm.parameter_vector_to_parameters(midpoint_vector(mcm))
        params = {k[len(mcm.model_names[0]):]: v for k, v in params.items()}
        return np.asarray(karger.rotational_harmonics_representation(scheme(sch), **params))
    for sch in ("pgse2", "legacy2"):
        add("numpy", f"numpy/X0GeneralizedKarger.rotational_harmonics_representation/{sch}", lambda s=sch: rh(s))


def _sm_forward_cases():
    """The spherical-mean forward of the JAX fit: per shell and broadcast to every measurement. A zeppelin is prolate
    (``lambda_par > lambda_perp``: the closed form with ``erf``) by narrowed bounds and by the tortuosity link of MC-SMT,
    and oblate (``lambda_par < lambda_perp``: the Gauss-Legendre rule of dmipy-analytic) by narrowed bounds."""
    from dmipy_fit.core.modeling_framework import MultiCompartmentSphericalMeanModel
    from dmipy_fit.custom_optimizers.reference_models import mcsmt, sandi
    from dmipy_fit.signal_models import cylinder_models as cm, gaussian_models as gm

    def compartments(*classes):
        return lambda: MultiCompartmentSphericalMeanModel([x() for x in classes])

    def stick_zeppelin(par_bounds, perp_bounds):
        def build():
            model = compartments(cm.C1Stick, gm.G2Zeppelin)()
            model.set_parameter_optimization_bounds("G2Zeppelin_1_lambda_par", [par_bounds])
            model.set_parameter_optimization_bounds("G2Zeppelin_1_lambda_perp", [perp_bounds])
            return model
        return build

    for cid, build in (("ball", compartments(gm.G1Ball)),
                       ("stick_zeppelin", stick_zeppelin([1.0e-9, 3.0e-9], [0.1e-9, 1.0e-9])),
                       ("stick_zeppelin_oblate", stick_zeppelin([0.1e-9, 1.0e-9], [1.0e-9, 3.0e-9])),
                       ("stick_ball", compartments(cm.C1Stick, gm.G1Ball)), ("mcsmt", mcsmt),
                       ("sandi", sandi)):
        for broadcast in (False, True):
            for sect in ("jax_f32", "jax_f64"):
                add(sect, f"{sect}/sm_forward/{cid}/{'meas' if broadcast else 'shell'}",
                    lambda b=build, bc=broadcast: _sm_forward(b(), scheme("pgse"), bc))


def _sm_forward(model, sch, broadcast):
    import jax.numpy as jnp
    import numpy as np
    from dmipy_fit.jax.multicompartment_jax import build_mc_sm_forward_fn
    fn = build_mc_sm_forward_fn(model, sch, broadcast=broadcast)
    return np.asarray(fn(jnp.asarray(midpoint_vector(model))))


# ---------------------------------------------------------------------------------------------------------------------
# Standalone forms
# ---------------------------------------------------------------------------------------------------------------------
def _standalone_cases():
    import numpy as np
    I = inputs()
    tau = np.array([0.0, 0.02, 0.05, 0.12])
    add("numpy", "numpy/surface.b_hat_ia", lambda: np.stack([
        form("surface.b_hat_ia")(2.0, 0.3e-6, rho, tau, vw) for rho in (2e-6, 15e-6) for vw in (True, False)]))
    add("numpy", "numpy/surface.b_hat_sphere", lambda: np.stack([
        form("surface.b_hat_sphere")(2.0, 1.0e-6, rho, tau, vw) for rho in (2e-6, 15e-6) for vw in (True, False)]))
    add("numpy", "numpy/surface.b_hat_ea_long", lambda: np.stack([
        form("surface.b_hat_ea_long")(rho, sv, tau) for rho in (2e-6, 15e-6) for sv in (0.5e6, 2.5e6)]))
    add("numpy", "numpy/surface.exterior_surface_to_volume", lambda: np.array([
        form("surface.exterior_surface_to_volume")(f, 2.0, 0.3e-6, geometry=g)
        for f in (0.3, 0.7) for g in ("sphere", "cylinder", "plane")]))
    odi = np.array([0.02, 0.1, 0.3, 0.5, 0.8, 0.99])
    add("numpy", "numpy/distributions.odi2kappa", lambda: form("distributions.odi2kappa")(odi))
    add("numpy", "numpy/distributions.kappa2odi",
        lambda: form("distributions.kappa2odi")(np.array([0.1, 1.0, 4.0, 16.0, 64.0])))
    kappas = [(0.0, 0.0), (1.5, 0.5), (8.0, 3.0), (-2.0, -1.0), (30.0, 12.0)]
    add("numpy", "numpy/sh_analytical.gaussian_signal_sh", lambda: np.concatenate([
        np.ravel(form("sh_analytical.gaussian_signal_sh")(k, b, l_max=8)) for k, b in kappas if k >= b >= 0]))
    add("numpy", "numpy/sh_analytical.gaussian_J_l", lambda: np.concatenate([
        np.ravel(form("sh_analytical.gaussian_J_l")(k, l_max=8)) for k in (-20.0, -1.5, 1e-9, 0.7, 6.0, 40.0,
                                                                        600.0)]))
    add("numpy", "numpy/sh_analytical.gaussian_kernel_rh", lambda: np.stack([
        form("sh_analytical.gaussian_kernel_rh")(b, lp, lq, sh_order=8)
        for b in (5e8, 2e9, 5e9) for lp, lq in ((1.7e-9, 0.6e-9), (1.7e-9, 0.0), (2.5e-9, 2.5e-9))]))

    def signal_lm():
        M = np.array([[0.3, 0.1, -0.2], [0.1, -0.5, 0.15], [-0.2, 0.15, 0.2]])
        return np.stack([form("sh_analytical.gaussian_signal_lm")(M * s, pre, l_max=8)
                         for s, pre in ((0.5, 0.7), (2.0, 0.2))])
    add("numpy", "numpy/sh_analytical.gaussian_signal_lm", signal_lm)
    thetas = np.linspace(0.0, PI, 7)
    add("numpy", "numpy/convolution.real_sym_rh_basis", lambda: form("convolution.real_sym_rh_basis")(8, thetas))
    add("numpy", "numpy/convolution.sh_convolution", lambda: np.concatenate([
        form("convolution.sh_convolution")(I["sh_coeff"][:int((o + 2) * (o + 1) // 2)], I["rh_kernel"][:o // 2 + 1])
        for o in (2, 4, 8)]))
    add("numpy", "numpy/cylinder.E_lm_from_exponent_coeffs", lambda: form("cylinder.E_lm_from_exponent_coeffs")(
        np.array([0.2, 1.1, 2.0]), np.array([[0.1, -0.2, 0.3, 0.05, -0.1]] * 3) * np.array([[1.], [2.], [0.5]]),
        l_max=4))
    add("numpy", "numpy/cylinder.dispersed_signal_from_E_lm", lambda: form("cylinder.dispersed_signal_from_E_lm")(
        np.stack([I["sh_coeff"][:15] * 3.0, I["sh_coeff"][:15] - 0.1]), I["sh_coeff"][:15]))
    add("numpy", "numpy/cylinder.watson_odf_lm", lambda: np.concatenate([
        form("cylinder.watson_odf_lm")(np.array(m) / np.linalg.norm(m), k, l_max=8)
        for m, k in (([0., 0., 1.], 0.0), ([0.3, 0.2, 0.9], 2.5), ([1., 1., 0.], 12.0))]))
    from dmipy_fit.utils.utils import T1_tortuosity
    add("numpy", "numpy/utils.T1_tortuosity", lambda: np.array([
        T1_tortuosity(s0i, s0e)(lp, vi, ve) for s0i, s0e in ((1., 1.), (0.8, 1.3)) for lp in (1.7e-9, 2.5e-9)
        for vi, ve in ((0.3, None), (0.3, 0.5), (0.6, 0.2))]))
    from dmipy_fit.distributions import distributions as dist
    nn = I["unit_vectors"]
    add("numpy", "numpy/distributions.SD1Watson", lambda: np.concatenate([
        dist.SD1Watson()(nn, mu=m, odi=o) for m, o in (([0.7, 1.1], 0.1), ([0., 0.], 0.5), ([1.5, -0.4], 0.9))]))
    add("numpy", "numpy/distributions.SD1Watson.sh", lambda: np.concatenate([
        dist.SD1Watson().spherical_harmonics_representation(mu=m, odi=o, sh_order=so)
        for m, o, so in (([0.7, 1.1], 0.1, 8), ([0., 0.], 0.5, 6), ([1.5, -0.4], 0.9, 4), ([0.3, 0.2], 0.05, None))]))
    add("numpy", "numpy/distributions.SD2Bingham", lambda: np.concatenate([
        dist.SD2Bingham()(nn, mu=m, odi=o, beta_fraction=bf, psi=ps)
        for m, o, bf, ps in (([0.7, 1.1], 0.1, 0.4, 0.3), ([0., 0.], 0.5, 0.8, 1.2), ([1.5, -0.4], 0.9, 0.0, 0.0))]))
    add("numpy", "numpy/distributions.SD2Bingham.sh", lambda: np.concatenate([
        dist.SD2Bingham().spherical_harmonics_representation(mu=m, odi=o, beta_fraction=bf, psi=ps, sh_order=so)
        for m, o, bf, ps, so in (([0.7, 1.1], 0.1, 0.4, 0.3, 8), ([0.2, 0.5], 0.5, 0.8, 1.2, 6),
                                 ([1.5, -0.4], 0.9, 0.2, 0.7, None))]))
    for norm in ("standard", "plane", "cylinder", "sphere"):
        add("numpy", f"numpy/distributions.DD1Gamma/{norm}", lambda n=norm: np.stack([np.concatenate(
            dist.DD1Gamma(normalization=n)(alpha=a, beta=b)) for a, b in ((2.0, 0.3e-6), (6.0, 0.6e-6),
                                                                         (3.0, 1.0e-6))]))
        add("numpy", f"numpy/distributions.DD2Poisson/{norm}", lambda n=norm: np.stack([np.concatenate(
            dist.DD2Poisson(normalization=n)(mean_diameter=m)) for m in (2e-6, 5e-6, 11e-6)]))


# ---------------------------------------------------------------------------------------------------------------------
# Composite models: the signal of a multi-compartment model at the mid-range of its parameters
# ---------------------------------------------------------------------------------------------------------------------
def midpoint_vector(model, frac=0.37, overrides=None):
    """The parameter vector of ``model`` (SI) at ``frac`` of every parameter's range (orientation and the
    relaxation times follow their ranges); ``overrides`` sets a parameter outright (one without a finite range)."""
    import numpy as np
    out = dict(overrides or {})
    for name in model.parameter_names:
        if name in out:
            continue
        rng = np.atleast_2d(np.array(model.parameter_ranges[name], dtype=float))
        scale = np.atleast_1d(model.parameter_scales[name]).astype(float)
        val = (rng[:, 0] + frac * (rng[:, 1] - rng[:, 0])) * scale
        out[name] = val if model.parameter_cardinality[name] > 1 else val[0]
    return model.parameters_to_parameter_vector(**out)


def _composites():
    """(id, builder, schemes): builders return a MultiCompartmentModel (or a compartment, wrapped)."""
    from dmipy_fit.core.modeling_framework import MultiCompartmentModel
    from dmipy_fit.custom_optimizers import reference_models as rm
    from dmipy_fit.signal_models import attenuation as att
    from dmipy_fit.signal_models import cylinder_models as cm, gaussian_models as gm, sphere_models as sm
    from dmipy_fit.signal_models import plane_models as pm
    from dmipy_fit.signal_models.exchange_models import X0GeneralizedKarger
    from dmipy_fit.distributions import distribute_models as dm
    from dmipy_fit.utils.utils import T1_tortuosity
    T2, T1 = att.TransverseRelaxation, att.LongitudinalRelaxation
    SR = att.SurfaceRelaxivity

    def wrapped(model):
        return lambda: MultiCompartmentModel([model()])

    def gated(base, factors):
        return lambda: MultiCompartmentModel([att.OccupancyGatedModel(base(), [f() for f in factors])])

    def karger(a, b, gate_a=(), gate_b=()):
        def build():
            ia = att.OccupancyGatedModel(a(), [f() for f in gate_a]) if gate_a else a()
            eb = att.OccupancyGatedModel(b(), [f() for f in gate_b]) if gate_b else b()
            return MultiCompartmentModel([X0GeneralizedKarger(ia, eb)])
        return build

    ref = {n: getattr(rm, n) for n in (
        "ball", "zeppelin", "temporal_zeppelin", "ball_and_stick", "ball_and_zeppelin", "stick_tortuous_zeppelin",
        "free_water_elimination", "ivim", "noddi", "bingham_noddi", "noddida", "charmed",
        "axcaliber", "active_ax", "verdict", "sandi", "impulsed", "karger_two_compartment", "fexi",
        "sandix", "exchange_impulsed", "temporal_zeppelin_model")}
    comp = []
    for n, schemes in [
        ("ball", ["pgse", "legacy"]), ("zeppelin", ["pgse", "ste"]), ("temporal_zeppelin", ["pgse2"]),
        ("ball_and_stick", ["pgse", "legacy"]), ("ball_and_zeppelin", ["pgse"]),
        ("stick_tortuous_zeppelin", ["pgse", "ste"]), ("free_water_elimination", ["pgse"]), ("ivim", ["pgse"]),
        ("noddi", ["pgse", "legacy", "ste"]), ("bingham_noddi", ["pgse", "legacy"]), ("noddida", ["pgse"]),
        ("charmed", ["pgse2", "legacy2"]), ("axcaliber", ["pgse2", "legacy2"]),
        ("active_ax", ["pgse2"]), ("verdict", ["pgse2", "ogse"]), ("sandi", ["pgse2", "mixed"]),
        ("impulsed", ["mixed", "ogse"]), ("karger_two_compartment", ["pgse2"]),
        ("fexi", ["pgse2"]), ("sandix", ["pgse2"]), ("exchange_impulsed", ["pgse2"]),
        ("temporal_zeppelin_model", ["pgse2"])]:
        comp.append(("ref." + n, ref[n], schemes))

    def stick_zeppelin_tortuous():
        stick, zeppelin = cm.C1Stick(), gm.G2Zeppelin()
        return MultiCompartmentModel([X0GeneralizedKarger(
            stick, zeppelin,
            parameter_links=[(zeppelin, 'lambda_perp', T1_tortuosity(),
                              [(zeppelin, 'lambda_par'), (None, 'f')])])])

    comp += [
        ("karger.stick_zeppelin_tortuous", stick_zeppelin_tortuous, ["pgse2", "legacy2"]),
        ("gated.zeppelin_T2_SR", gated(gm.G2Zeppelin, [T2, lambda: SR(surface_to_volume=2e6)]), ["pgse", "ste"]),
        ("gated.stick_T2_SR_ext", gated(cm.C1Stick, [T2, lambda: att.ExteriorSurfaceRelaxivity(1.5e6)]), ["pgse"]),
        ("gated.stick_T2_intrapore", gated(cm.C1Stick, [T2, att.IntraPoreSurfaceRelaxivity]), ["pgse"]),
        ("gated.sphere_T2_intrasphere", gated(sm.S2SphereStejskalTannerApproximation,
                                              [T2, att.IntraSphereSurfaceRelaxivity]), ["pgse"]),
        ("gated.cylinder_T2_SR", gated(cm.C4CylinderGaussianPhaseApproximation, [T2, SR]), ["pgse"]),
        ("gated.plane_SR", gated(pm.P2PlaneStejskalTannerApproximation, [SR]), ["pgse"]),
        ("gated.zeppelin_T1_T2", gated(gm.G2Zeppelin, [T2, T1]), ["pgste"]),
        ("karger.ball_ball_T2", karger(gm.G1Ball, gm.G1Ball, [T2], [T2]), ["pgse2"]),
        ("karger.stick_zeppelin_T2_T1", karger(cm.C1Stick, gm.G2Zeppelin, [T2, T1], [T2, T1]), ["pgste"]),
        ("karger.stick_zeppelin_T2", karger(cm.C1Stick, gm.G2Zeppelin, [T2], [T2]), ["pgse2"]),
        ("karger.sphere_ball", karger(sm.S4SphereGaussianPhaseApproximation, gm.G1Ball), ["pgse2"]),
        ("karger.ball_ball", karger(gm.G1Ball, gm.G1Ball), ["pgse_rf"]),
        ("karger.stick_zeppelin_T2_rf", karger(cm.C1Stick, gm.G2Zeppelin, [T2], [T2]), ["pgse_rf"]),
        ("karger.ball_ball_T2_T1_rf", karger(gm.G1Ball, gm.G1Ball, [T2, T1], [T2, T1]), ["pgste_rf"]),
        ("single.g3", wrapped(gm.G3TemporalZeppelin), ["pgse2", "legacy2"]),
        ("single.dot", wrapped(sm.S1Dot), ["pgse"]),
        ("single.c2", wrapped(cm.C2CylinderStejskalTannerApproximation), ["pgse", "legacy"]),
        ("single.c3", wrapped(cm.C3CylinderCallaghanApproximation), ["pgse2", "legacy2"]),
        ("single.c4", wrapped(cm.C4CylinderGaussianPhaseApproximation), ["pgse2", "legacy2", "ogse"]),
        ("single.s2", wrapped(sm.S2SphereStejskalTannerApproximation), ["pgse", "legacy"]),
        ("single.s4", wrapped(sm.S4SphereGaussianPhaseApproximation), ["pgse2", "legacy2", "ogse", "ste"]),
        ("single.p2", wrapped(pm.P2PlaneStejskalTannerApproximation), ["pgse", "legacy"]),
        ("single.p3", wrapped(pm.P3PlaneCallaghanApproximation), ["pgse2", "legacy2"]),
        ("dist.watson_stick_zeppelin", lambda: MultiCompartmentModel([dm.SD1WatsonDistributed(
            [cm.C1Stick(), gm.G2Zeppelin()])]), ["pgse", "legacy", "ste", "pte"]),
        ("dist.bingham_stick_zeppelin", lambda: MultiCompartmentModel([dm.SD2BinghamDistributed(
            [cm.C1Stick(), gm.G2Zeppelin()])]), ["pgse", "ste"]),
        ("dist.watson_cylinder_zeppelin", lambda: MultiCompartmentModel([dm.SD1WatsonDistributed(
            [cm.C4CylinderGaussianPhaseApproximation(), gm.G2Zeppelin()])]), ["pgse", "legacy"]),
        ("dist.watson_c2", lambda: MultiCompartmentModel([dm.SD1WatsonDistributed(
            [cm.C2CylinderStejskalTannerApproximation()])]), ["legacy"]),
        ("dist.watson_stick_T2", lambda: MultiCompartmentModel([dm.SD1WatsonDistributed(
            [att.OccupancyGatedModel(cm.C1Stick(), [T2()]), att.OccupancyGatedModel(gm.G2Zeppelin(), [T2()])])]),
         ["legacy"]),
        ("dist.sh_stick", lambda: MultiCompartmentModel([dm.SD3SphericalHarmonicsDistributed(
            [cm.C1Stick()], sh_order=4)]), ["pgse"]),
        ("dist.gamma_c3", wrapped(lambda: dm.DD1GammaDistributed([cm.C3CylinderCallaghanApproximation()])),
         ["pgse2", "legacy2"]),
        ("dist.gamma_c4", wrapped(lambda: dm.DD1GammaDistributed([cm.C4CylinderGaussianPhaseApproximation()])),
         ["pgse2", "legacy2"]),
        ("dist.gamma_s4", wrapped(lambda: dm.DD1GammaDistributed([sm.S4SphereGaussianPhaseApproximation()])),
         ["pgse2"]),
        ("dist.poisson_c3", wrapped(lambda: dm.DD2PoissonDistributed([cm.C3CylinderCallaghanApproximation()])),
         ["pgse2"]),
        ("dist.poisson_c4", wrapped(lambda: dm.DD2PoissonDistributed([cm.C4CylinderGaussianPhaseApproximation()])),
         ["pgse2"]),
        ("dist.gamma_c2", wrapped(lambda: dm.DD1GammaDistributed([cm.C2CylinderStejskalTannerApproximation()])),
         ["pgse"]),
        ("dist.bundle", lambda: MultiCompartmentModel([dm.BundleModel([cm.C1Stick(), gm.G2Zeppelin()])]),
         ["pgse", "ste"]),
    ]
    return comp


TRIVIAL_COMPOSITES = {"single.dot"}          # the dot does not attenuate


def _composite_cases():
    for cid, builder, schemes in _composites():
        for sch in schemes:
            add("numpy", f"numpy/composite/{cid}/{sch}",
                lambda b=builder, s=sch, c=cid: _simulate(b(), scheme(s), c), trivial=cid in TRIVIAL_COMPOSITES)


OVERRIDES = {"dist.sh_stick": lambda: {
    "SD3SphericalHarmonicsDistributed_1_SD3SphericalHarmonics_1_sh_coeff": inputs()["sh_coeff"][:15]}}


def _simulate(model, sch, cid=None):
    import numpy as np
    over = OVERRIDES.get(cid, lambda: None)()
    return np.asarray(model.simulate_signal(sch, midpoint_vector(model, overrides=over)))


# ---------------------------------------------------------------------------------------------------------------------
# JAX
# ---------------------------------------------------------------------------------------------------------------------
JAX_DIRECT = [       # class key, schemes, (the models of NUMPY_MODELS with a use_jax path)
    ("G1Ball", ["pgse", "legacy"]), ("G2Zeppelin", ["pgse", "legacy"]), ("C1Stick", ["pgse", "legacy"]),
    ("C2", ["pgse", "legacy"]), ("C3", ["pgse", "legacy"]), ("C4", ["pgse", "legacy", "pgse2"]),
    ("S2", ["pgse", "legacy"]), ("C5", ["pgse", "ogse"]), ("S5", ["pgse", "ogse"]), ("P5", ["pgse", "ogse"]),
]


def _jax_direct_cases():
    for key, schemes in JAX_DIRECT:
        entry = next(m for m in NUMPY_MODELS if m[0] == key)
        _, cls_path, kwargs, kind, _ = entry
        for sch in schemes:
            for pk, params in _params(kind).items():
                matrix = key in ("C5", "S5", "P5")
                n_modes = 24 if key == "P5" else 16
                for sect in (("jax_f64",) if key in FLOAT64_ONLY or matrix else ("jax_f32", "jax_f64")):
                    add(sect, f"{sect}/{key}/{sch}/{pk}",
                        lambda c=cls_path, k=kwargs, s=sch, p=params: _call(c, k, s, p, use_jax=True),
                        ops=N_T * (2 * n_modes + 4) if matrix else 8, host_dependent=matrix)


# The composites build_mc_forward_fn refuses, by id: the forward factories have no twin for them (a refusal is the
# behaviour that is held).
JAX_UNSUPPORTED = {
    "dist.sh_stick": "no JAX forward for the spherical-harmonic distribution",
    "dist.gamma_s4": "the diameter integral has C3 and C4 cylinders only",
    "dist.gamma_c2": "the diameter integral has C3 and C4 cylinders only",
    "dist.poisson_c4": "the Poisson diameter integral has the C3 cylinder only",
    "dist.bundle": "no JAX forward for the bundle model",
    "ref.sandi": "a spherical-mean model: its JAX forward is the sm_forward/sandi case",
}


# schemes the JAX forwards do not evaluate: the finite-RF schemes (the relaxation path refuses them and the closed
# Karger path would ignore the pulses) and the schemes without delta for the models that read it
JAX_SKIP_SCHEMES = {"pgse_rf", "pgste_rf"}
JAX_SKIP_CASES = {("single.c4", "ogse")}


def _jax_composite_cases():
    cases = []
    for cid, builder, schemes in _composites():
        if cid in JAX_UNSUPPORTED:
            continue
        for sch in schemes:
            if sch not in JAX_SKIP_SCHEMES and (cid, sch) not in JAX_SKIP_CASES:
                cases.append((cid, builder, sch))
    for cid, builder, sch in cases:
        add("jax_f64", f"jax_f64/forward/{cid}/{sch}",
            lambda b=builder, s=sch, c=cid: _forward_jax(b(), scheme(s), c), trivial=cid in TRIVIAL_COMPOSITES)
        if not _needs_float64(cid):
            add("jax_f32", f"jax_f32/forward/{cid}/{sch}",
                lambda b=builder, s=sch, c=cid: _forward_jax(b(), scheme(s), c), trivial=cid in TRIVIAL_COMPOSITES)


F64_COMPOSITES = ("cylinder", "c2", "c3", "c4", "charmed", "axcaliber", "active_ax", "gamma", "poisson")


def _needs_float64(cid):
    """Whether the forward model has a Bessel / Van Gelderen sum that overflows float32."""
    return any(t in cid for t in F64_COMPOSITES)


def _forward_jax(model, sch, cid=None):
    import jax.numpy as jnp
    import numpy as np
    from dmipy_fit.jax.multicompartment_jax import build_mc_forward_fn
    fn = build_mc_forward_fn(model, sch)
    over = OVERRIDES.get(cid, lambda: None)()
    return np.asarray(fn(jnp.asarray(midpoint_vector(model, overrides=over))))


# ---------------------------------------------------------------------------------------------------------------------
# Building the registry
# ---------------------------------------------------------------------------------------------------------------------
def build_registry():
    if REGISTRY:
        return REGISTRY
    _numpy_model_cases()
    _method_cases()
    _karger_rh_cases()
    _standalone_cases()
    _composite_cases()
    _jax_direct_cases()
    _jax_composite_cases()
    _sm_forward_cases()
    return REGISTRY


# ---------------------------------------------------------------------------------------------------------------------
# Emission, hashing, comparison
# ---------------------------------------------------------------------------------------------------------------------
def digest(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def environment():
    """The versions the record was made with (the dmipy packages by the commit they were installed from)."""
    import numpy
    from importlib import metadata
    info = {"python": platform.python_version(), "machine": platform.machine(), "numpy": numpy.__version__}
    for dist in ("scipy", "jax", "jaxlib", "dmipy-sim", "dmipy-replay", "dmipy-analytic"):
        try:
            info[dist] = metadata.version(dist)
            direct = metadata.distribution(dist).read_text("direct_url.json")
            commit = json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None
            if commit:
                info[dist] += "@" + commit
        except metadata.PackageNotFoundError:
            info[dist] = None
    info["pinned_env"] = PINNED_ENV
    return info


def matches(cid, patterns):
    """Whether case id ``cid`` is one of ``patterns`` (exact ids or ``fnmatch`` patterns)."""
    return any(fnmatch.fnmatchcase(cid, p) for p in patterns)


def emit(section, out_dir, only=None):
    """Run every case of ``section`` (only those matching ``only``, when given) and write ``out_dir/<section>.npz``."""
    import warnings
    import numpy as np
    if importlib.util.find_spec("numba") is not None:
        raise SystemExit("numba is installed: its jit of a form is a second implementation; the record is made "
                         "without it")
    if section.startswith("jax"):
        import jax
        want = section == "jax_f64"
        if bool(jax.config.jax_enable_x64) != want:
            raise SystemExit(f"{section} needs JAX_ENABLE_X64={int(want)}")
    build_registry()
    out, errors = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            for cid, case in REGISTRY.items():
                if case.section != section or (only is not None and not matches(cid, only)):
                    continue
                try:
                    a = np.ascontiguousarray(np.asarray(case.thunk()))
                    if section.startswith("jax"):
                        import jax
                        if bool(jax.config.jax_enable_x64) != (section == "jax_f64"):
                            raise RuntimeError("the x64 flag changed during the run")
                    if not np.isfinite(a).all():
                        raise RuntimeError("non-finite output")
                    if not case.trivial and a.size > 1 and np.ptp(a) == 0:
                        raise RuntimeError("constant output")
                    out[cid.replace("/", "__")] = a
                except Exception as exc:                      # every failing case is reported, not the first
                    errors[cid] = f"{type(exc).__name__}: {str(exc)[:160]}"
    if errors:
        for cid, msg in errors.items():
            print(f"FAILED {cid}: {msg}", file=sys.stderr)
        raise SystemExit(f"{len(errors)} cases failed in {section}")
    np.savez(os.path.join(out_dir, section + ".npz"), **out)
    return len(out)


def run_emit(section, out_dir, only=None):
    """``emit`` in a fresh interpreter under the pinned environment."""
    env = dict(os.environ)
    env.update(PINNED_ENV)
    env["JAX_ENABLE_X64"] = "1" if section == "jax_f64" else "0"
    subprocess.run([sys.executable, os.path.abspath(__file__), "emit", section, out_dir, *(only or [])],
                   check=True, env=env)


def ulps(a, b):
    import numpy as np
    return np.abs(a - b) / np.spacing(np.maximum(np.abs(a), np.abs(b)).astype(a.dtype))


def derived_rtol(case_ops, a):
    import numpy as np
    eps = float(np.finfo(a.dtype).eps)
    with np.errstate(all="ignore"):
        exponent = max(1.0, float(np.max(-np.log(np.clip(a, np.finfo(a.dtype).tiny, None)))))
    return case_ops * eps * exponent


def load(path):
    import numpy as np
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def record():
    import numpy as np
    build_registry()
    os.makedirs(RECORD, exist_ok=True)
    manifest = {"format": 1, "environment": None, "sections": {}}
    for section in SECTIONS:
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            run_emit(section, d1)
            run_emit(section, d2)
            first, second = load(os.path.join(d1, section + ".npz")), load(os.path.join(d2, section + ".npz"))
            entries = {}
            for key, a in first.items():
                b = second[key]
                same = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
                cid = key.replace("__", "/")
                host = REGISTRY[cid].host_dependent
                entry = {"sha256": digest(a), "shape": list(a.shape), "dtype": a.dtype.str,
                         "bitwise": bool(same) and not host}
                if host:
                    entry["reason"] = "host_dependent"
                    entry["atol"] = REGISTRY[cid].ops * float(np.finfo(a.dtype).eps)
                    entry["ops"] = REGISTRY[cid].ops
                elif not entry["bitwise"]:
                    entry["rtol"] = derived_rtol(REGISTRY[cid].ops, a)
                    entry["ops"] = REGISTRY[cid].ops
                entries[cid] = entry
            manifest["sections"][section] = dict(sorted(entries.items()))
            np.savez(os.path.join(RECORD, section + ".npz"), **first)
            print(f"{section}: {len(entries)} cases, "
                  f"{sum(not e['bitwise'] for e in entries.values())} not bit-reproducible")
    manifest["environment"] = environment()
    with open(os.path.join(RECORD, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=False)
        f.write("\n")


def rebaselined(base_manifest_path):
    """The case ids whose entry in ``record/manifest.json`` differs from the one in ``base_manifest_path`` or exists
    in only one of them: the outputs a change deliberately re-baselines."""
    with open(base_manifest_path) as f:
        base = json.load(f)["sections"]
    with open(os.path.join(RECORD, "manifest.json")) as f:
        head = json.load(f)["sections"]
    changed = set()
    for section in SECTIONS:
        a, b = base.get(section, {}), head.get(section, {})
        for cid in set(a) | set(b):
            if cid not in a or cid not in b or a[cid]["sha256"] != b[cid]["sha256"]:
                changed.add(cid)
    return changed


def check(reference=None, base_manifest=None):
    """Compare this code's outputs with ``record/``, or with the outputs ``record`` wrote to ``reference`` on this
    machine (the cases of ``rebaselined(base_manifest)`` excepted: their differences are printed, not failed)."""
    import numpy as np
    build_registry()
    record_dir = reference or RECORD
    with open(os.path.join(record_dir, "manifest.json")) as f:
        manifest = json.load(f)
    allowed = rebaselined(base_manifest) if base_manifest else set()
    failures, notes, total = [], [], 0
    now_env = environment()
    for key, value in manifest["environment"].items():
        if key != "pinned_env" and now_env.get(key) != value:
            print(f"note: {key} is {now_env.get(key)}, the record was made with {value}")
    for section in SECTIONS:
        entries = manifest["sections"][section]
        recorded = load(os.path.join(record_dir, section + ".npz"))
        with tempfile.TemporaryDirectory() as d:
            run_emit(section, d)
            now = load(os.path.join(d, section + ".npz"))
        want = {cid for cid, c in REGISTRY.items() if c.section == section}
        missing, extra = sorted(want - set(entries) - allowed), sorted(set(entries) - want - allowed)
        if missing or extra:
            failures.append(f"{section}: cases differ from the record: missing {missing}, extra {extra}")
        for cid in sorted(allowed & want - set(entries)):
            notes.append(f"{cid}: added")
        for cid in sorted(allowed & set(entries) - want):
            notes.append(f"{cid}: removed")
        for cid, entry in entries.items():
            if cid not in want:
                continue
            total += 1
            key = cid.replace("/", "__")
            if key not in now:
                failures.append(f"{cid}: not emitted")
                continue
            a, r = now[key], recorded[key]
            if digest(r) != entry["sha256"]:
                failures.append(f"{cid}: the stored array does not match its sha256")
            if a.dtype != r.dtype or a.shape != r.shape:
                (notes if cid in allowed else failures).append(f"{cid}: dtype/shape {a.dtype}{a.shape} != {r.dtype}{r.shape}")
            elif cid in allowed:
                if not np.array_equal(a, r):
                    d = np.abs(a - r)
                    rel = d / np.maximum(np.abs(r), np.finfo(r.dtype).tiny)
                    notes.append(f"{cid}: re-baselined, max abs {d.max():.3e}, max rel {rel.max():.3e}, "
                                 f"max {ulps(a, r).max():.1f} ulp, {int((a != r).sum())}/{a.size} elements")
            elif entry["bitwise"]:
                if not np.array_equal(a, r) or digest(a) != entry["sha256"]:
                    d = np.abs(a - r)
                    failures.append(f"{cid}: differs: max abs {d.max():.3e}, max {ulps(a, r).max():.1f} ulp, "
                                    f"{int((a != r).sum())}/{a.size} elements")
            elif "atol" in entry:
                err = np.abs(a - r)
                if not (err <= entry["atol"]).all():
                    failures.append(f"{cid}: host dependent, beyond its derived atol {entry['atol']:.2e}: "
                                    f"max abs {err.max():.3e}")
                elif err.max() > 0:
                    notes.append(f"{cid}: host dependent, max abs {err.max():.3e} within its derived atol "
                                 f"{entry['atol']:.2e}")
            else:
                tol = entry["rtol"]
                err = np.abs(a - r) / np.maximum(np.abs(r), np.finfo(r.dtype).tiny)
                if not (err <= tol).all():
                    failures.append(f"{cid}: not bit-reproducible, beyond its derived rtol {tol:.2e}: "
                                    f"max rel {err.max():.3e}")
    print(f"{total} cases compared, {len(failures)} differ")
    for line in notes:
        print("  note: " + line)
    for line in failures:
        print("  " + line)
    return 1 if failures else 0


def rebaseline(prefixes):
    """Replace the record's entries of the cases starting with one of ``prefixes`` by this code's outputs and drop the
    entries of cases that are no longer registered; refuse (and write nothing) when any other case differs."""
    import numpy as np
    build_registry()
    with open(os.path.join(RECORD, "manifest.json")) as f:
        manifest = json.load(f)
    unexpected, report = [], []
    new_sections = {}
    for section in SECTIONS:
        entries = manifest["sections"][section]
        recorded = load(os.path.join(RECORD, section + ".npz"))
        with tempfile.TemporaryDirectory() as d:
            run_emit(section, d)
            now = load(os.path.join(d, section + ".npz"))
        registered = {cid for cid, c in REGISTRY.items() if c.section == section}
        kept_entries, kept_arrays = {}, {}
        for cid in sorted(registered | set(entries)):
            key = cid.replace("/", "__")
            wanted = cid.startswith(tuple(prefixes))
            if cid not in registered:                                    # the case is gone
                report.append(f"{cid}: removed")
                continue
            a = now[key]
            old = recorded.get(key)
            same = old is not None and old.dtype == a.dtype and old.shape == a.shape and np.array_equal(old, a)
            if cid in entries and not wanted:
                if not same:
                    ent = entries[cid]
                    tolerated = "atol" in ent or "rtol" in ent
                    if not tolerated:
                        unexpected.append(cid)
                kept_entries[cid], kept_arrays[key] = entries[cid], old
                continue
            entry = {"sha256": digest(a), "shape": list(a.shape), "dtype": a.dtype.str, "bitwise": True}
            if cid in entries and not entries[cid]["bitwise"]:
                entry = dict(entries[cid], sha256=digest(a))
            kept_entries[cid], kept_arrays[key] = entry, a
            if old is None:
                report.append(f"{cid}: added")
            elif not same:
                d_abs = np.abs(a - old)
                rel = d_abs / np.maximum(np.abs(old), np.finfo(old.dtype).tiny)
                report.append(f"{cid}: moved, max abs {d_abs.max():.3e}, max rel {rel.max():.3e}, "
                              f"max {ulps(a, old).max():.1f} ulp, {int((a != old).sum())}/{a.size} elements")
        new_sections[section] = (kept_entries, kept_arrays)
    if unexpected:
        print("differ without being named; nothing written:")
        for cid in unexpected:
            print("  " + cid)
        sys.exit(1)
    for section, (kept_entries, kept_arrays) in new_sections.items():
        manifest["sections"][section] = dict(sorted(kept_entries.items()))
        np.savez(os.path.join(RECORD, section + ".npz"), **kept_arrays)
    manifest["environment"] = environment()
    with open(os.path.join(RECORD, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=False)
        f.write("\n")
    for line in report:
        print(line)
    print(f"{len(report)} entries changed")


def _write_manifest(manifest, record_dir):
    with open(os.path.join(record_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=False)
        f.write("\n")


def rebaseline_only(patterns, record_dir=RECORD):
    """Replace the record's entries and arrays of the registered cases matching ``patterns`` by this code's outputs
    and nothing else: the other entries are not compared, the manifest's ``environment`` block is kept, and the
    ``rewrites`` list of the manifest gains the written keys under the environment they were written in."""
    import numpy as np
    registry = build_registry()
    with open(os.path.join(record_dir, "manifest.json")) as f:
        manifest = json.load(f)
    known = set(registry) | {cid for s in SECTIONS for cid in manifest["sections"][s]}
    unmatched = [p for p in patterns if not any(fnmatch.fnmatchcase(cid, p) for cid in known)]
    if not patterns or unmatched:
        raise SystemExit(f"--only needs case ids or patterns that each match a registered case or a recorded entry; "
                         f"no match: {unmatched or 'none given'}")
    written, report = {}, []
    for section in SECTIONS:
        names = [cid for cid, c in registry.items() if c.section == section and matches(cid, patterns)]
        entries = manifest["sections"][section]
        stale = [cid for cid in entries if matches(cid, patterns) and cid not in registry]
        if not names and not stale:
            continue
        recorded = load(os.path.join(record_dir, section + ".npz"))
        for cid in stale:
            del entries[cid]
            recorded.pop(cid.replace("/", "__"), None)
            report.append(f"{cid}: removed")
            written.setdefault(section, [])
        if not names:
            manifest["sections"][section] = dict(sorted(entries.items()))
            np.savez(os.path.join(record_dir, section + ".npz"), **recorded)
            continue
        with tempfile.TemporaryDirectory() as d:
            run_emit(section, d, only=patterns)
            now = load(os.path.join(d, section + ".npz"))
        for cid in sorted(names):
            key = cid.replace("/", "__")
            a, old = now[key], recorded.get(key)
            entry = {"sha256": digest(a), "shape": list(a.shape), "dtype": a.dtype.str, "bitwise": True}
            if cid in entries and not entries[cid]["bitwise"]:
                entry = dict(entries[cid], sha256=digest(a))
            entries[cid] = entry
            recorded[key] = a
            written.setdefault(section, []).append(cid)
            if old is None:
                report.append(f"{cid}: added")
            elif old.dtype == a.dtype and old.shape == a.shape and np.array_equal(old, a):
                report.append(f"{cid}: unchanged")
            else:
                d_abs = np.abs(a - old) if old.shape == a.shape else np.array([np.nan])
                rel = d_abs / np.maximum(np.abs(old), np.finfo(old.dtype).tiny) if old.shape == a.shape else d_abs
                report.append(f"{cid}: moved, max abs {np.nanmax(d_abs):.3e}, max rel {np.nanmax(rel):.3e}")
        manifest["sections"][section] = dict(sorted(entries.items()))
        np.savez(os.path.join(record_dir, section + ".npz"), **recorded)
    env = environment()
    env.pop("pinned_env", None)
    rewrites = manifest.setdefault("rewrites", [])
    for rec in rewrites:
        if rec["environment"] == env:
            for section, cids in written.items():
                rec["keys"][section] = sorted(set(rec["keys"].get(section, [])) | set(cids))
            break
    else:
        rewrites.append({"environment": env, "keys": {s: sorted(c) for s, c in written.items()}})
    _write_manifest(manifest, record_dir)
    for line in report:
        print(line)
    print(f"{sum(map(len, written.values()))} keys written; every other key and the environment block untouched")


def rename_case(old, new, record_dir=RECORD):
    """Rename case ``old`` (its id and every id below ``old/``) to ``new`` in the manifest and the ``.npz`` files, array
    bytes untouched; verify that each renamed array has its old sha256 under the new name and that nothing else moved.
    Without a section prefix on ``old`` and ``new`` the rename applies to every section."""
    import numpy as np

    def rename(cid, prefix_old, prefix_new):
        if cid == prefix_old:
            return prefix_new
        if cid.startswith(prefix_old + "/"):
            return prefix_new + cid[len(prefix_old):]
        return None

    sectioned = old.split("/", 1)[0] in SECTIONS
    if sectioned != (new.split("/", 1)[0] in SECTIONS):
        raise SystemExit("OLD and NEW must both carry a section prefix or both not")
    if sectioned and old.split("/", 1)[0] != new.split("/", 1)[0]:
        raise SystemExit("a case cannot move between sections")
    with open(os.path.join(record_dir, "manifest.json")) as f:
        manifest = json.load(f)
    moved, new_arrays = 0, {}
    for section in SECTIONS:
        po, pn = (old, new) if sectioned else (f"{section}/{old}", f"{section}/{new}")
        entries = manifest["sections"][section]
        recorded = load(os.path.join(record_dir, section + ".npz"))
        renames = {cid: rename(cid, po, pn) for cid in entries}
        renames = {a: b for a, b in renames.items() if b is not None}
        if not renames:
            continue
        clash = sorted(set(renames.values()) & (set(entries) - set(renames)))
        if clash:
            raise SystemExit(f"{section}: the new ids already exist: {clash}")
        before = {cid: e["sha256"] for cid, e in entries.items()}
        out_entries, out_arrays = {}, {}
        for cid, entry in entries.items():
            target = renames.get(cid, cid)
            out_entries[target] = entry
            out_arrays[target.replace("/", "__")] = recorded[cid.replace("/", "__")]
        for cid, target in renames.items():             # verification: bytes, digests, key sets
            if digest(out_arrays[target.replace("/", "__")]) != before[cid]:
                raise SystemExit(f"{cid}: the renamed array does not have its old sha256")
        if set(out_arrays) != {c.replace("/", "__") for c in out_entries} or len(out_entries) != len(entries):
            raise SystemExit(f"{section}: manifest and arrays disagree after the rename")
        manifest["sections"][section] = dict(sorted(out_entries.items()))
        new_arrays[section] = out_arrays
        moved += len(renames)
    if not moved:
        raise SystemExit(f"no case matches {old}")
    for rec in manifest.get("rewrites", []):           # the rewrite log follows the rename
        for section, cids in rec["keys"].items():
            po, pn = (old, new) if sectioned else (f"{section}/{old}", f"{section}/{new}")
            rec["keys"][section] = sorted(rename(c, po, pn) or c for c in cids)
    for section, arrays in new_arrays.items():
        np.savez(os.path.join(record_dir, section + ".npz"), **arrays)
    _write_manifest(manifest, record_dir)
    for section in new_arrays:                          # re-read what was written
        entries = json.load(open(os.path.join(record_dir, "manifest.json")))["sections"][section]
        on_disk = load(os.path.join(record_dir, section + ".npz"))
        for cid, entry in entries.items():
            if digest(on_disk[cid.replace("/", "__")]) != entry["sha256"]:
                raise SystemExit(f"{cid}: written array does not match its sha256")
    print(f"{moved} keys renamed {old} -> {new}; arrays and digests verified")


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "check"
    if cmd == "emit":
        print(f"{argv[2]}: {emit(argv[2], argv[3], argv[4:] or None)} cases")
    elif cmd == "record":
        record()
    elif cmd == "check":
        options = dict(zip(argv[2::2], argv[3::2]))
        unknown = set(options) - {"--reference", "--base-manifest"}
        if unknown or len(argv) % 2:
            raise SystemExit(f"check takes --reference DIR and --base-manifest FILE, not {sorted(unknown) or argv[2:]}")
        sys.exit(check(options.get("--reference"), options.get("--base-manifest")))
    elif cmd == "rebaseline":
        if argv[2:3] == ["--only"]:
            rebaseline_only(argv[3:])
        else:
            rebaseline(argv[2:])
    elif cmd == "rename-case":
        if len(argv) != 4:
            raise SystemExit("rename-case takes OLD NEW")
        rename_case(argv[2], argv[3])
    elif cmd == "inputs":
        make_inputs()
    elif cmd == "list":
        for cid, c in build_registry().items():
            print(c.section, cid)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
