"""Batched JAX multi-tissue volume-fraction NNLS == scipy COBYLA reference.

The S0-weighted geometric volume fractions are recovered by a convex NNLS step,
batched on GPU via jax.vmap + jaxopt OSQP (fractions_jax) and, as a CPU
fallback, by per-voxel scipy COBYLA (MultiTissueVolumeFractionOptimizer). These
must agree, for both the spherical-mean and full-signal per-compartment bases.
"""
import numpy as np
import numpy.testing as npt
import pytest

from dmipy_fit.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme)
from dmipy_fit.core.spherical_mean_framework import (
    MultiCompartmentSphericalMeanModel)
from dmipy_fit.core.modeling_framework import MultiCompartmentModel
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.signal_models.gaussian_models import G2Zeppelin, G1Ball
from dmipy_fit.optimizers.multi_tissue_volume_fraction import (
    MultiTissueVolumeFractionOptimizer)

pytest.importorskip("jax")
from dmipy_fit.jax.fractions_jax import fit_multi_tissue_fractions_jax

scheme = wu_minn_hcp_acquisition_scheme()
scheme.TE = 0.08
S0s = [7000., 5000., 11000.]
FRACS = np.array([[0.5, 0.3, 0.2], [0.6, 0.2, 0.2], [0.4, 0.4, 0.2]])
KERNEL = dict(C1Stick_1_lambda_par=1.7e-9, G2Zeppelin_1_lambda_par=1.7e-9,
              G2Zeppelin_1_lambda_perp=0.5e-9, G1Ball_1_lambda_iso=3e-9)


def test_sm_fractions_jax_matches_cobyla():
    m = MultiCompartmentSphericalMeanModel(
        [C1Stick(), G2Zeppelin(), G1Ball()], S0_tissue_responses=S0s)
    xsi, data = [], []
    for f in FRACS:
        gt = dict(KERNEL, partial_volume_0=f[0], partial_volume_1=f[1],
                  partial_volume_2=f[2])
        phi = m(scheme, quantity='stochastic cost function', **gt)   # per-shell
        data.append((phi * np.array(S0s)) @ f)
        xsi.append(m.parameters_to_parameter_vector(**gt))
    xsi, data = np.array(xsi), np.array(data)
    cob = np.array([MultiTissueVolumeFractionOptimizer(scheme, m, S0s)(
        data[i], xsi[i]) for i in range(len(FRACS))])
    jx = fit_multi_tissue_fractions_jax(m, scheme, xsi, data, S0s,
                                        spherical_mean=True)
    npt.assert_allclose(jx, cob, atol=2e-3)
    npt.assert_allclose(jx, FRACS, atol=2e-3)            # and recovers GT


def test_full_signal_fractions_jax_matches_cobyla():
    m = MultiCompartmentModel(
        [C1Stick(), G2Zeppelin(), G1Ball()], S0_tissue_responses=S0s)
    xsi, data = [], []
    for f in FRACS:
        gt = dict(KERNEL, C1Stick_1_mu=[np.pi / 2, np.pi / 2],
                  G2Zeppelin_1_mu=[np.pi / 2, np.pi / 2],
                  partial_volume_0=f[0], partial_volume_1=f[1],
                  partial_volume_2=f[2])
        phi = m(scheme, quantity='stochastic cost function', **gt)   # per-meas
        data.append((phi * np.array(S0s)) @ f)
        xsi.append(m.parameters_to_parameter_vector(**gt))
    xsi, data = np.array(xsi), np.array(data)
    cob = np.array([MultiTissueVolumeFractionOptimizer(scheme, m, S0s)(
        data[i], xsi[i]) for i in range(len(FRACS))])
    jx = fit_multi_tissue_fractions_jax(m, scheme, xsi, data, S0s,
                                        spherical_mean=False)
    npt.assert_allclose(jx, cob, atol=2e-3)
    npt.assert_allclose(jx, FRACS, atol=2e-3)
