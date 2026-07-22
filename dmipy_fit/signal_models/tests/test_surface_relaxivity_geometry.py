"""SurfaceRelaxivity derives S/V from the base compartment's pore geometry.

The generic SurfaceRelaxivity factor turns a fitted ``diameter`` into a
surface-to-volume ratio ``coeff / d``. The coefficient is geometry-dependent:
sphere S/V = 6/d, cylinder = 4/d, plane = 2/d. When the factor is attached via
OccupancyGatedModel, the geometry is read from the base compartment's
``diameter`` parameter type automatically. A regression against the previous
behaviour, where the factor hardcoded the cylinder value (4/d) for every
compartment -- silently mis-scaling sphere (and plane) surface relaxivity.
"""
import numpy as np
import pytest

from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy_fit.signal_models import sphere_models, cylinder_models, plane_models
from dmipy_fit.signal_models.attenuation import (
    OccupancyGatedModel, SurfaceRelaxivity)

TE = 0.05
RHO = 20e-6
D = 8e-6


def _scheme():
    b = np.array([0., 2e9, 2e9, 2e9])
    g = np.array([[1., 0, 0], [1., 0, 0], [0, 1., 0], [0, 0, 1.]])
    return acquisition_scheme_from_bvalues(
        b, g, np.full(4, 0.0035), np.full(4, 0.015), TE=TE)


def _implied_sv_times_d(gated):
    """Recover S/V*d from the applied relaxivity factor on a dwi measurement.
    mu/lambda_par are ignored by isotropic bases (sphere/plane) and used by the
    cylinder; relaxivity is multiplicative so the diffusion part cancels."""
    acq = _scheme()
    common = dict(diameter=D, mu=[0., 0.], lambda_par=1.7e-9)
    E0 = np.asarray(gated(acq, surface_relaxivity=0.0, **common))
    E1 = np.asarray(gated(acq, surface_relaxivity=RHO, **common))
    sv = -np.log(E1[1] / E0[1]) / (RHO * TE)   # tau_perp falls back to TE
    return sv * D


@pytest.mark.parametrize('base_cls, coeff', [
    (sphere_models.S4SphereGaussianPhaseApproximation, 6.0),
    (sphere_models.S2SphereStejskalTannerApproximation, 6.0),
    (cylinder_models.C2CylinderStejskalTannerApproximation, 4.0),
    (plane_models.P2PlaneStejskalTannerApproximation, 2.0),
])
def test_geometry_sets_surface_to_volume_coefficient(base_cls, coeff):
    """S/V coefficient follows the base compartment's diameter geometry."""
    f = SurfaceRelaxivity()
    gated = OccupancyGatedModel(base_cls(), [f])
    # geometry was bound from the base compartment
    assert f.geometry == base_cls()._parameter_types['diameter']
    np.testing.assert_allclose(
        _implied_sv_times_d(gated), coeff, rtol=1e-6)


def test_explicit_geometry_overrides_binding():
    """A user-set geometry is not overwritten by the base compartment."""
    f = SurfaceRelaxivity(geometry='sphere')
    OccupancyGatedModel(
        cylinder_models.C2CylinderStejskalTannerApproximation(), [f])
    assert f.geometry == 'sphere'   # not clobbered to 'cylinder'


def test_explicit_surface_to_volume_bypasses_diameter():
    """surface_to_volume overrides any geometry-derived value."""
    sv = 1.0 / 3e-6
    f = SurfaceRelaxivity(surface_to_volume=sv)
    gated = OccupancyGatedModel(
        sphere_models.S4SphereGaussianPhaseApproximation(), [f])
    assert f.geometry is None       # binding skipped when surface_to_volume set
    acq = _scheme()
    E0 = np.asarray(gated(acq, diameter=D, surface_relaxivity=0.0))
    E1 = np.asarray(gated(acq, diameter=D, surface_relaxivity=RHO))
    got = -np.log(E1[1] / E0[1]) / (RHO * TE)
    np.testing.assert_allclose(got, sv, rtol=1e-6)


def test_numpy_jax_parity_sphere_surface_relaxivity():
    """JAX SurfaceRelaxivity factor uses the same geometry coefficient as NumPy."""
    pytest.importorskip("jax")
    from dmipy_fit.jax.attenuation_jax import build_jax_factor

    f = SurfaceRelaxivity()
    gated = OccupancyGatedModel(
        sphere_models.S4SphereGaussianPhaseApproximation(), [f])   # binds sphere
    acq = _scheme()
    E_np = np.asarray(gated(acq, diameter=D, surface_relaxivity=RHO))

    from dmipy_fit.jax.jax_compat import scheme_to_jax
    sj = scheme_to_jax(acq)
    E_base = np.asarray(
        gated.model(acq, diameter=D))              # diffusion-only base
    jax_factor = build_jax_factor(f)
    fac = np.asarray(jax_factor(sj, None, {'surface_relaxivity': RHO,
                                           'diameter': D}))
    np.testing.assert_allclose(E_np, E_base * fac, rtol=1e-5, atol=1e-6)
