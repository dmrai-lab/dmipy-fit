"""Exact matrix-method restricted-diffusion compartments (P5 / C5 / S5).

Physical validation, not just regression: the matrix method must (i) reproduce the Gaussian-phase models
in their valid low-b regime, (ii) reduce to free diffusion on unrestricted axes, (iii) obey restriction
monotonicity and the physical range, (iv) converge in mode count, and (v) depart from the GPA (the exact
model attenuates *more*) precisely where the Gaussian-phase approximation breaks down at high b. These
compartments are acquisition-scheme glue around the shared solver `dmipy_sim.math.matrix_method`; the
pure-numerics tests of that solver itself (unit-geometry convergence, the published narrow-pulse
coefficients, free-diffusion/Mitra limits, ...) live in dmipy-sim's own `tests/math/test_matrix_method.py`.
"""
import numpy as np
import numpy.testing as npt

from dmipy_sim.math.matrix_method import matrix_restricted_signal

from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.core.constants import CONSTANTS
from dmipy_fit.signal_models._matrix_adapter import modes_for
from dmipy_fit.signal_models.cylinder_models import (
    C5CylinderMatrixMethod, C4CylinderGaussianPhaseApproximation)
from dmipy_fit.signal_models.sphere_models import (
    S5SphereMatrixMethod, S4SphereGaussianPhaseApproximation)
from dmipy_fit.signal_models.plane_models import P5PlaneMatrixMethod

DELTA, DELTABIG = 10e-3, 40e-3
LAMBDA_PAR = 1.7e-9
MU = [np.pi / 2, 0.0]                      # cylinder/plane axis along x
D_RESTRICTED = CONSTANTS['water_in_axons_diffusion_constant']


def _scheme(bvalues, direction):
    dirs = np.tile(direction, (len(bvalues), 1))
    return AcquisitionScheme.from_pgse(np.asarray(bvalues, float), dirs, DELTA, DELTABIG)


def test_cylinder_matrix_matches_gpa_low_b():
    "C5 perp reproduces Van Gelderen GPA in its valid low-b regime; diverges (attenuates more) at high b."
    b = np.array([0.0, 5e8, 1e9])
    scheme = _scheme(b, [0, 0, 1.])        # perpendicular to the x-axis
    c5 = C5CylinderMatrixMethod(mu=MU, lambda_par=LAMBDA_PAR, diameter=8e-6)
    c4 = C4CylinderGaussianPhaseApproximation(mu=MU, lambda_par=LAMBDA_PAR, diameter=8e-6)
    npt.assert_allclose(c5(scheme), c4(scheme), atol=3e-4)


def test_sphere_matrix_matches_gpa_low_b():
    "S5 reproduces Murday-Cotts GPA at low b."
    b = np.array([0.0, 5e8, 1e9])
    scheme = _scheme(b, [1., 0, 0])
    s5 = S5SphereMatrixMethod(diameter=8e-6)
    s4 = S4SphereGaussianPhaseApproximation(diameter=8e-6)
    npt.assert_allclose(s5(scheme), s4(scheme), atol=2e-4)


def test_cylinder_parallel_is_free_diffusion():
    "Along the cylinder axis the signal is free Gaussian diffusion exp(-b lambda_par)."
    b = np.array([0.0, 1e9, 3e9])
    scheme = _scheme(b, [1., 0, 0])        # parallel to the x-axis
    c5 = C5CylinderMatrixMethod(mu=MU, lambda_par=LAMBDA_PAR, diameter=8e-6)
    npt.assert_allclose(c5(scheme), np.exp(-b * LAMBDA_PAR), atol=1e-9)


def test_matrix_departs_from_gpa_at_high_b():
    "At high b the GPA breaks down; the exact matrix model attenuates MORE (lower signal), both physical."
    b = np.array([0.0, 1e10])
    scheme = _scheme(b, [0, 0, 1.])
    c5 = C5CylinderMatrixMethod(mu=MU, lambda_par=LAMBDA_PAR, diameter=15e-6)(scheme)
    c4 = C4CylinderGaussianPhaseApproximation(mu=MU, lambda_par=LAMBDA_PAR, diameter=15e-6)(scheme)
    assert c5[1] < c4[1]                    # matrix (exact) below GPA at high b
    assert abs(c5[1] - c4[1]) > 1e-3        # a real departure, not noise
    assert 0.0 <= c5[1] <= 1.0 + 1e-9


def test_b0_is_unity_and_physical_range():
    b = np.array([0.0, 5e8, 2e9, 5e9])
    for E in (C5CylinderMatrixMethod(mu=MU, lambda_par=LAMBDA_PAR, diameter=6e-6)(_scheme(b, [0, 0, 1.])),
              S5SphereMatrixMethod(diameter=6e-6)(_scheme(b, [1., 0, 0])),
              P5PlaneMatrixMethod(diameter=6e-6)(_scheme(b, [1., 0, 0]))):
        npt.assert_allclose(E[0], 1.0)
        assert np.all(E > 0) and np.all(E <= 1.0 + 1e-9)


def test_restriction_monotonicity():
    "Smaller pore -> higher (more restricted) signal, for all three geometries."
    b = np.array([0.0, 2e9])
    for model, scheme in (
            (lambda d: C5CylinderMatrixMethod(mu=MU, lambda_par=LAMBDA_PAR, diameter=d), _scheme(b, [0, 0, 1.])),
            (lambda d: S5SphereMatrixMethod(diameter=d), _scheme(b, [1., 0, 0])),
            (lambda d: P5PlaneMatrixMethod(diameter=d), _scheme(b, [1., 0, 0]))):
        E_small = model(4e-6)(scheme)[1]
        E_large = model(12e-6)(scheme)[1]
        assert E_small > E_large


def test_mode_convergence():
    "Signal is stable once enough eigenmodes are kept (default is converged)."
    b = np.array([0.0, 5e9])
    scheme = _scheme(b, [1., 0, 0])
    E_lo = S5SphereMatrixMethod(diameter=10e-6, n_modes=10)(scheme)[1]
    E_hi = S5SphereMatrixMethod(diameter=10e-6, n_modes=22)(scheme)[1]
    npt.assert_allclose(E_lo, E_hi, atol=1e-4)


def test_matrix_method_matches_sim_for_pgse_slew_and_ogse():
    "C5 (perpendicular axis), S5 and P5 reproduce dmipy_sim.math.matrix_method's own number for a PGSE at"
    "infinite slew, a slew-limited PGSE, and a cosine OGSE -- the model is acquisition-scheme glue around"
    "the shared solver, not a second implementation of it. A small n_modes keeps this a wiring check"
    "(sizing, axis projection, waveform reconstruction), not a convergence study."
    diameter = 8e-6
    n_modes = 8
    b = np.array([0.0, 3e9])
    direction = np.tile([0., 0., 1.], (2, 1))      # perpendicular to C5's x-axis (MU)
    schemes = {
        'pgse_inf_slew': AcquisitionScheme.from_pgse(b, direction, DELTA, DELTABIG),
        'pgse_slew_limited': AcquisitionScheme.from_pgse(b, direction, DELTA, DELTABIG, slew_rate=200.0),
        'ogse_cosine': AcquisitionScheme.from_ogse(
            b, direction, oscillation_frequency=200.0, gradient_duration=DELTA, Delta=DELTABIG),
    }
    c5 = C5CylinderMatrixMethod(mu=MU, lambda_par=LAMBDA_PAR, diameter=diameter, n_modes=n_modes)
    s5 = S5SphereMatrixMethod(diameter=diameter, n_modes=n_modes)
    p5 = P5PlaneMatrixMethod(diameter=diameter, n_modes=n_modes)
    for name, scheme in schemes.items():
        G, dt = np.asarray(scheme._G, np.float64), float(scheme._dt)
        g_axis = G[1, :, 2]            # the sole (z) direction's signed schedule, non-b0 measurement

        E_sim = matrix_restricted_signal(
            'cylinder', g_axis, dt, D_RESTRICTED, diameter, n_modes=modes_for("cylinder", c5.n_modes))
        npt.assert_allclose(c5(scheme)[1], E_sim, atol=1e-10, err_msg=name)

        E_sim = matrix_restricted_signal(
            'sphere', g_axis, dt, D_RESTRICTED, diameter, n_modes=modes_for("sphere", s5.n_modes))
        npt.assert_allclose(s5(scheme)[1], E_sim, atol=1e-10, err_msg=name)

        E_sim = matrix_restricted_signal(
            'plane', g_axis, dt, D_RESTRICTED, diameter, n_modes=modes_for("plane", p5.n_modes))
        npt.assert_allclose(p5(scheme)[1], E_sim, atol=1e-10, err_msg=name)


def test_plane_approaches_free():
    "As the slab thickens the signal descends monotonically toward free diffusion, always from above."
    b = np.array([0.0, 1e9])
    scheme = _scheme(b, [1., 0, 0])
    free = np.exp(-b[1] * P5PlaneMatrixMethod(diameter=6e-6).diffusion_constant)
    # the plane's n_modes is one count of cosine modes (dmipy_sim.math.matrix_method); a wider slab needs more
    # of them to reach the free limit on the same grid
    E = [P5PlaneMatrixMethod(diameter=L, n_modes=nm)(scheme)[1]
         for L, nm in ((40e-6, 64), (100e-6, 96), (300e-6, 160))]
    assert E[0] > E[1] > E[2] >= free - 1e-6      # monotone descent toward free, never below it
    assert E[2] - free < 0.03                     # the 300 um slab is within 0.03 of free
