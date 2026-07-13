"""JAX forwards for OccupancyGatedModel + attenuation factors.

The gated compartment's JAX signal (base diffusion JAX fn x factor product) must
equal the numpy OccupancyGatedModel.__call__ for the public factors (transverse T2
and surface relaxivity).
"""
import numpy as np
import numpy.testing as npt
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dmipy_fit.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme)
from dmipy_fit.signal_models.gaussian_models import G2Zeppelin
from dmipy_fit.signal_models.attenuation import (
    OccupancyGatedModel, TransverseRelaxation, LongitudinalRelaxation,
    ExteriorSurfaceRelaxivity, IntraPoreSurfaceRelaxivity)
from dmipy_fit.signal_models.cylinder_models import C1Stick as _C1Stick
from dmipy_fit.jax.jax_compat import scheme_to_jax
from dmipy_fit.jax.multicompartment_jax import _make_occupancy_gated_jax_fn

scheme = wu_minn_hcp_acquisition_scheme()
scheme.TE = 0.08


def _jax_params(d):
    return {k: jnp.array(v) for k, v in d.items()}


@pytest.mark.parametrize("factors,params", [
    ([TransverseRelaxation()],
     dict(mu=[1.1, 0.4], lambda_par=1.7e-9, lambda_perp=0.45e-9, T2=0.06)),
    ([ExteriorSurfaceRelaxivity(S_ext_over_V=5e4), TransverseRelaxation()],
     dict(mu=[0.8, 1.2], lambda_par=1.7e-9, lambda_perp=0.4e-9,
          surface_relaxivity=8e-6, T2=0.05)),
])
def test_gated_jax_matches_numpy(factors, params):
    gated = OccupancyGatedModel(G2Zeppelin(), factors)
    E_np = gated(scheme, **params)
    jfn = _make_occupancy_gated_jax_fn(gated)
    E_jax = np.asarray(jfn(scheme_to_jax(scheme), _jax_params(params)))
    npt.assert_allclose(E_jax, E_np, atol=1e-12)


def test_longitudinal_relaxation_jax_matches_numpy():
    """PGSTE longitudinal (T1) factor exp(-TM/T1) reproduced by the JAX builder;
    the transverse T2 factor is gated to the encoding time and T1 to TM.

    Built analytically (b from the b-value, not integrated from a waveform) so the
    parity isolates the attenuation factors, matching the other cases above."""
    pgste = wu_minn_hcp_acquisition_scheme()
    pgste.TE = 0.02                                    # STE transverse (2*delta) time
    pgste.TM = np.full(pgste.number_of_measurements, 0.040)
    gated = OccupancyGatedModel(
        G2Zeppelin(), [TransverseRelaxation(), LongitudinalRelaxation()])
    params = dict(mu=[0.9, 0.3], lambda_par=1.7e-9, lambda_perp=0.4e-9,
                  T2=0.05, T1=1.1)
    E_np = gated(pgste, **params)
    jfn = _make_occupancy_gated_jax_fn(gated)
    sjax = scheme_to_jax(pgste)
    assert 'tau_par' in sjax
    E_jax = np.asarray(jfn(sjax, _jax_params(params)))
    npt.assert_allclose(E_jax, E_np, atol=1e-12)
    # spin echo (no TM): tau_par absent -> the T1 factor is inert
    se = wu_minn_hcp_acquisition_scheme()
    se.TE = 0.08
    E_np_se = gated(se, **params)
    E_jax_se = np.asarray(jfn(scheme_to_jax(se), _jax_params(params)))
    npt.assert_allclose(E_jax_se, E_np_se, atol=1e-12)


@pytest.mark.parametrize("vol_weighted", [True, False])
def test_intrapore_surface_relaxivity_jax_matches_numpy(vol_weighted):
    """Gamma-averaged intra-pore surface attenuation (Bessel-K closed form in
    numpy) reproduced by generalized Gauss-Laguerre quadrature in JAX."""
    intra = OccupancyGatedModel(_C1Stick(), [
        IntraPoreSurfaceRelaxivity(gamma_shape=2.0,
                                   gamma_scale_outer_diameter=0.304e-6,
                                   volume_weighted=vol_weighted),
        TransverseRelaxation()])
    p = dict(mu=[1.1, 0.4], lambda_par=1.7e-9,
             surface_relaxivity=15e-6, g_ratio=0.7, T2=0.06)
    E_np = intra(scheme, **p)
    jfn = _make_occupancy_gated_jax_fn(intra)
    E_jax = np.asarray(jfn(scheme_to_jax(scheme), _jax_params(p)))
    npt.assert_allclose(E_jax, E_np, atol=1e-10)
