"""Phase 5 tests: JAX implementations of complex Bessel-function models."""
import pytest
import numpy as np
from numpy.testing import assert_allclose

jax = pytest.importorskip('jax')
import jax.numpy as jnp

# Cylinder models use bessel_jn and Van Gelderen sums that require float64.
# Enable x64 for this entire test module before any JIT compilation occurs.
jax.config.update("jax_enable_x64", True)

from dmipy_fit.signal_models.sphere_models import S2SphereStejskalTannerApproximation
from dmipy_fit.signal_models.cylinder_models import (
    C2CylinderStejskalTannerApproximation,
    C3CylinderCallaghanApproximation,
    C4CylinderGaussianPhaseApproximation,
)
from dmipy_fit.jax.signal_models_jax import (
    s2sphere_signal,
    c2cylinder_signal,
    c4cylinder_signal,
    build_c3cylinder_jax_fn,
)
from dmipy_fit.jax.jax_compat import scheme_to_jax, unitsphere2cart_1d_jax
from dmipy_fit.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MU = np.array([np.pi / 4, np.pi / 3])
DIAMETER = 4e-6  # 4 µm
LAMBDA_PAR = 1.7e-9


@pytest.fixture(scope='module')
def hcp_scheme():
    return wu_minn_hcp_acquisition_scheme()


@pytest.fixture(scope='module')
def hcp_jax(hcp_scheme):
    return scheme_to_jax(hcp_scheme)


@pytest.fixture(scope='module')
def mu_cart():
    return unitsphere2cart_1d_jax(jnp.array(MU))


# ---------------------------------------------------------------------------
# S2 sphere
# ---------------------------------------------------------------------------

class TestS2SphereJax:
    def test_matches_numpy(self, hcp_scheme, hcp_jax):
        model = S2SphereStejskalTannerApproximation()
        E_np = model(hcp_scheme, mu=MU, diameter=DIAMETER)
        E_jax = s2sphere_signal(hcp_jax['qvalues'], jnp.array(DIAMETER))
        assert_allclose(np.array(E_jax), E_np, rtol=1e-5, atol=1e-8)

    def test_jit_compiles(self, hcp_jax):
        fn = jax.jit(s2sphere_signal)
        E = fn(hcp_jax['qvalues'], jnp.array(DIAMETER))
        assert jnp.isfinite(E).all()

    def test_grad_finite(self, hcp_jax):
        def loss(d):
            return jnp.sum(s2sphere_signal(hcp_jax['qvalues'], d))
        g = jax.grad(loss)(jnp.array(DIAMETER))
        assert jnp.isfinite(g)

    def test_q_zero_gives_one(self):
        q_zero = jnp.zeros(5)
        E = s2sphere_signal(q_zero, jnp.array(DIAMETER))
        assert_allclose(np.array(E), np.ones(5), atol=1e-10)


# ---------------------------------------------------------------------------
# C2 cylinder (Soderman)
# ---------------------------------------------------------------------------

class TestC2CylinderJax:
    def test_matches_numpy(self, hcp_scheme, hcp_jax, mu_cart):
        model = C2CylinderStejskalTannerApproximation()
        E_np = model(hcp_scheme, mu=MU, lambda_par=LAMBDA_PAR, diameter=DIAMETER)
        E_jax = c2cylinder_signal(
            hcp_jax['bvalues'],
            hcp_jax['gradient_directions'],
            hcp_jax['qvalues'],
            mu_cart,
            jnp.array(LAMBDA_PAR),
            jnp.array(DIAMETER),
        )
        assert_allclose(np.array(E_jax), E_np, rtol=1e-5, atol=1e-8)

    def test_jit_compiles(self, hcp_jax, mu_cart):
        fn = jax.jit(c2cylinder_signal)
        E = fn(hcp_jax['bvalues'], hcp_jax['gradient_directions'],
               hcp_jax['qvalues'], mu_cart,
               jnp.array(LAMBDA_PAR), jnp.array(DIAMETER))
        assert jnp.isfinite(E).all()

    def test_grad_finite(self, hcp_jax, mu_cart):
        def loss(d):
            return jnp.sum(c2cylinder_signal(
                hcp_jax['bvalues'], hcp_jax['gradient_directions'],
                hcp_jax['qvalues'], mu_cart,
                jnp.array(LAMBDA_PAR), d))
        g = jax.grad(loss)(jnp.array(DIAMETER))
        assert jnp.isfinite(g)

    def test_signal_between_zero_and_one(self, hcp_jax, mu_cart):
        E = c2cylinder_signal(
            hcp_jax['bvalues'], hcp_jax['gradient_directions'],
            hcp_jax['qvalues'], mu_cart,
            jnp.array(LAMBDA_PAR), jnp.array(DIAMETER))
        assert jnp.all(E >= 0.0)
        assert jnp.all(E <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# C4 cylinder (Gaussian Phase)
# ---------------------------------------------------------------------------

class TestC4CylinderJax:
    def _get_jax_params(self, hcp_scheme):
        model = C4CylinderGaussianPhaseApproximation()
        scheme_jax = scheme_to_jax(hcp_scheme)
        roots_jax = jnp.array(model._CYLINDER_TRASCENDENTAL_ROOTS)
        D = float(model.diffusion_perpendicular)
        gamma = float(model.gyromagnetic_ratio)
        return scheme_jax, roots_jax, D, gamma

    def test_matches_numpy(self, hcp_scheme, mu_cart):
        model = C4CylinderGaussianPhaseApproximation()
        E_np = model(hcp_scheme, mu=MU, lambda_par=LAMBDA_PAR, diameter=DIAMETER)

        scheme_jax, roots_jax, D, gamma = self._get_jax_params(hcp_scheme)
        E_jax = c4cylinder_signal(
            scheme_jax['bvalues'],
            scheme_jax['gradient_directions'],
            scheme_jax['gradient_strengths'],
            scheme_jax['delta'],
            scheme_jax['Delta'],
            mu_cart,
            jnp.array(LAMBDA_PAR),
            jnp.array(DIAMETER),
            D, gamma, roots_jax,
        )
        assert_allclose(np.array(E_jax), E_np, rtol=1e-5, atol=1e-8)

    def test_jit_compiles(self, hcp_scheme, mu_cart):
        scheme_jax, roots_jax, D, gamma = self._get_jax_params(hcp_scheme)
        fn = jax.jit(c4cylinder_signal, static_argnums=())
        E = fn(scheme_jax['bvalues'], scheme_jax['gradient_directions'],
               scheme_jax['gradient_strengths'], scheme_jax['delta'],
               scheme_jax['Delta'], mu_cart,
               jnp.array(LAMBDA_PAR), jnp.array(DIAMETER), D, gamma, roots_jax)
        assert jnp.isfinite(E).all()

    def test_grad_finite(self, hcp_scheme, mu_cart):
        scheme_jax, roots_jax, D, gamma = self._get_jax_params(hcp_scheme)

        def loss(d):
            return jnp.sum(c4cylinder_signal(
                scheme_jax['bvalues'], scheme_jax['gradient_directions'],
                scheme_jax['gradient_strengths'], scheme_jax['delta'],
                scheme_jax['Delta'], mu_cart,
                jnp.array(LAMBDA_PAR), d, D, gamma, roots_jax))
        g = jax.grad(loss)(jnp.array(DIAMETER))
        assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# C3 cylinder (Callaghan) — factory-built function
# ---------------------------------------------------------------------------

class TestC3CylinderJax:
    def test_matches_numpy(self, hcp_scheme, mu_cart):
        model = C3CylinderCallaghanApproximation()
        E_np = model(hcp_scheme, mu=MU, lambda_par=LAMBDA_PAR, diameter=DIAMETER)

        scheme_jax = scheme_to_jax(hcp_scheme)
        fn = build_c3cylinder_jax_fn(model.alpha, model.diffusion_perpendicular)
        E_jax = fn(
            scheme_jax['bvalues'],
            scheme_jax['gradient_directions'],
            scheme_jax['qvalues'],
            scheme_jax['tau'],
            mu_cart,
            jnp.array(LAMBDA_PAR),
            jnp.array(DIAMETER),
        )
        assert_allclose(np.array(E_jax), E_np, rtol=1e-4, atol=1e-6)

    def test_jit_compiles(self, hcp_scheme, mu_cart):
        model = C3CylinderCallaghanApproximation()
        scheme_jax = scheme_to_jax(hcp_scheme)
        fn = jax.jit(
            build_c3cylinder_jax_fn(model.alpha, model.diffusion_perpendicular)
        )
        E = fn(
            scheme_jax['bvalues'],
            scheme_jax['gradient_directions'],
            scheme_jax['qvalues'],
            scheme_jax['tau'],
            mu_cart,
            jnp.array(LAMBDA_PAR),
            jnp.array(DIAMETER),
        )
        assert jnp.isfinite(E).all()

    def test_grad_finite(self, hcp_scheme, mu_cart):
        model = C3CylinderCallaghanApproximation()
        scheme_jax = scheme_to_jax(hcp_scheme)
        inner_fn = build_c3cylinder_jax_fn(model.alpha, model.diffusion_perpendicular)

        def loss(d):
            return jnp.sum(inner_fn(
                scheme_jax['bvalues'],
                scheme_jax['gradient_directions'],
                scheme_jax['qvalues'],
                scheme_jax['tau'],
                mu_cart,
                jnp.array(LAMBDA_PAR),
                d,
            ))
        g = jax.grad(loss)(jnp.array(DIAMETER))
        assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# MultiCompartmentModel dispatch: complex models via solver='jax'
# ---------------------------------------------------------------------------

class TestComplexModelsViaMultiCompartment:
    def test_s2_in_mc_model(self, hcp_scheme):
        from dmipy_fit.core.modeling_framework import MultiCompartmentModel
        s2 = S2SphereStejskalTannerApproximation()
        mc = MultiCompartmentModel(models=[s2])
        gt_params = mc.parameters_to_parameter_vector(
            S2SphereStejskalTannerApproximation_1_diameter=DIAMETER)
        E = mc.simulate_signal(hcp_scheme, gt_params)
        result = mc.fit(hcp_scheme, E[None], solver='jax')
        fitted_d = float(result.fitted_parameters[
            'S2SphereStejskalTannerApproximation_1_diameter'].squeeze())
        assert_allclose(fitted_d, DIAMETER, rtol=0.05)

    def test_c2_in_mc_model(self, hcp_scheme):
        from dmipy_fit.core.modeling_framework import MultiCompartmentModel
        c2 = C2CylinderStejskalTannerApproximation()
        mc = MultiCompartmentModel(models=[c2])
        gt_params = mc.parameters_to_parameter_vector(
            C2CylinderStejskalTannerApproximation_1_mu=MU,
            C2CylinderStejskalTannerApproximation_1_lambda_par=LAMBDA_PAR,
            C2CylinderStejskalTannerApproximation_1_diameter=DIAMETER,
        )
        E = mc.simulate_signal(hcp_scheme, gt_params)
        # Ns=10 needed: default Ns=5 diameter grid is [0.01,5.0,10.0,15.0,20.0]µm;
        # GT=4µm falls between the first two points. Ns=10 adds a 4.5µm point
        # (12% from GT), giving L-BFGS-B a good starting basin.
        result = mc.fit(hcp_scheme, E[None], solver='jax', Ns=10)
        fitted_d = float(result.fitted_parameters[
            'C2CylinderStejskalTannerApproximation_1_diameter'].squeeze())
        assert_allclose(fitted_d, DIAMETER, rtol=0.15)

    def test_c4_in_mc_model(self, hcp_scheme):
        from dmipy_fit.core.modeling_framework import MultiCompartmentModel
        c4 = C4CylinderGaussianPhaseApproximation()
        mc = MultiCompartmentModel(models=[c4])
        # C4 (GPA) has very low signal sensitivity at d=4µm for the HCP scheme
        # (GPA attenuation ∝ r², so small radii produce near-zero perpendicular
        # effect; signal at d=2µm and d=4µm differ by < 0.5%). Use d=8µm where
        # the signal is clearly diameter-dependent. Ns=10 for better grid coverage.
        DIAMETER_C4 = 8e-6
        gt_params = mc.parameters_to_parameter_vector(
            C4CylinderGaussianPhaseApproximation_1_mu=MU,
            C4CylinderGaussianPhaseApproximation_1_lambda_par=LAMBDA_PAR,
            C4CylinderGaussianPhaseApproximation_1_diameter=DIAMETER_C4,
        )
        E = mc.simulate_signal(hcp_scheme, gt_params)
        result = mc.fit(hcp_scheme, E[None], solver='jax', Ns=10)
        fitted_d = float(result.fitted_parameters[
            'C4CylinderGaussianPhaseApproximation_1_diameter'].squeeze())
        assert_allclose(fitted_d, DIAMETER_C4, rtol=0.15)
