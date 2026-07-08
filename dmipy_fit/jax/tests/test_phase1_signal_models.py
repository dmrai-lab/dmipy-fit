"""Phase 1 tests: JAX forward functions for G1Ball, C1Stick, G2Zeppelin."""
import pytest
import numpy as np
from numpy.testing import assert_allclose

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from dmipy_fit.jax.jax_compat import scheme_to_jax, unitsphere2cart_1d_jax
from dmipy_fit.jax.signal_models_jax import (
    g1ball_signal,
    c1stick_signal,
    c1stick_spherical_mean,
    g2zeppelin_signal,
    g2zeppelin_spherical_mean,
)
from dmipy_fit.signal_models.gaussian_models import G1Ball, G2Zeppelin
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme


@pytest.fixture(scope='module')
def scheme():
    return wu_minn_hcp_acquisition_scheme()


@pytest.fixture(scope='module')
def scheme_jax(scheme):
    return scheme_to_jax(scheme)


# ---------------------------------------------------------------------------
# G1Ball
# ---------------------------------------------------------------------------

class TestG1Ball:
    def test_jax_matches_numpy(self, scheme, scheme_jax):
        model = G1Ball(lambda_iso=1.7e-9)
        E_np = model(scheme)
        E_jax = np.array(g1ball_signal(scheme_jax['bvalues'], 1.7e-9))
        assert_allclose(E_jax, E_np, rtol=1e-6)

    def test_use_jax_flag(self, scheme):
        model = G1Ball(lambda_iso=1.7e-9)
        E_np = model(scheme, use_jax=False)
        E_jax = model(scheme, use_jax=True)
        assert_allclose(E_jax, E_np, rtol=1e-6)

    def test_jit_compiles(self, scheme_jax):
        f = jax.jit(g1ball_signal)
        result = f(scheme_jax['bvalues'], 1.7e-9)
        assert result.shape == scheme_jax['bvalues'].shape

    def test_grad_is_finite(self, scheme_jax):
        f = jax.grad(lambda l: jnp.sum(g1ball_signal(scheme_jax['bvalues'], l)))
        g = f(1.7e-9)
        assert jnp.isfinite(g)

    def test_values_in_unit_interval(self, scheme_jax):
        E = g1ball_signal(scheme_jax['bvalues'], 1.7e-9)
        assert jnp.all(E >= 0.0) and jnp.all(E <= 1.0)


# ---------------------------------------------------------------------------
# C1Stick
# ---------------------------------------------------------------------------

class TestC1Stick:
    MU = np.array([np.pi / 4, np.pi / 3])
    LAMBDA_PAR = 1.7e-9

    def test_jax_matches_numpy(self, scheme, scheme_jax):
        model = C1Stick(mu=self.MU, lambda_par=self.LAMBDA_PAR)
        E_np = model(scheme)
        mu_cart = unitsphere2cart_1d_jax(jnp.array(self.MU))
        E_jax = np.array(c1stick_signal(
            scheme_jax['bvalues'], scheme_jax['gradient_directions'],
            mu_cart, self.LAMBDA_PAR))
        assert_allclose(E_jax, E_np, rtol=1e-6)

    def test_use_jax_flag(self, scheme):
        model = C1Stick(mu=self.MU, lambda_par=self.LAMBDA_PAR)
        E_np = model(scheme, use_jax=False)
        E_jax = model(scheme, use_jax=True)
        assert_allclose(E_jax, E_np, rtol=1e-6)

    def test_jit_compiles(self, scheme_jax):
        mu_cart = unitsphere2cart_1d_jax(jnp.array(self.MU))
        f = jax.jit(c1stick_signal)
        result = f(scheme_jax['bvalues'], scheme_jax['gradient_directions'],
                   mu_cart, self.LAMBDA_PAR)
        assert result.shape == scheme_jax['bvalues'].shape

    def test_grad_wrt_lambda_par(self, scheme_jax):
        mu_cart = unitsphere2cart_1d_jax(jnp.array(self.MU))
        f = jax.grad(lambda l: jnp.sum(c1stick_signal(
            scheme_jax['bvalues'], scheme_jax['gradient_directions'],
            mu_cart, l)))
        g = f(self.LAMBDA_PAR)
        assert jnp.isfinite(g)

    def test_spherical_mean_matches_numerical(self, scheme):
        """Compare analytical spherical mean to per-shell numerical average."""
        model = C1Stick(mu=self.MU, lambda_par=self.LAMBDA_PAR)
        scheme_jax_local = scheme_to_jax(scheme)

        for shell_idx, b in enumerate(scheme.shell_bvalues):
            if b == 0:
                continue
            mask = scheme.shell_indices == shell_idx
            E_shell = np.array(c1stick_signal(
                scheme_jax_local['bvalues'][mask],
                scheme_jax_local['gradient_directions'][mask],
                unitsphere2cart_1d_jax(jnp.array(self.MU)),
                self.LAMBDA_PAR))
            numerical_mean = float(E_shell.mean())
            analytical_mean = float(c1stick_spherical_mean(
                jnp.array([b]), self.LAMBDA_PAR)[0])
            # Tolerance is loose; numerical mean over finite samples
            assert_allclose(analytical_mean, numerical_mean, rtol=0.05)


# ---------------------------------------------------------------------------
# G2Zeppelin
# ---------------------------------------------------------------------------

class TestG2Zeppelin:
    MU = np.array([np.pi / 4, np.pi / 3])
    LAMBDA_PAR = 1.7e-9
    LAMBDA_PERP = 0.5e-9

    def test_jax_matches_numpy(self, scheme, scheme_jax):
        model = G2Zeppelin(mu=self.MU, lambda_par=self.LAMBDA_PAR,
                           lambda_perp=self.LAMBDA_PERP)
        E_np = model(scheme)
        mu_cart = unitsphere2cart_1d_jax(jnp.array(self.MU))
        E_jax = np.array(g2zeppelin_signal(
            scheme_jax['bvalues'], scheme_jax['gradient_directions'],
            mu_cart, self.LAMBDA_PAR, self.LAMBDA_PERP))
        assert_allclose(E_jax, E_np, rtol=1e-3)

    def test_use_jax_flag(self, scheme):
        model = G2Zeppelin(mu=self.MU, lambda_par=self.LAMBDA_PAR,
                           lambda_perp=self.LAMBDA_PERP)
        E_np = model(scheme, use_jax=False)
        E_jax = model(scheme, use_jax=True)
        assert_allclose(E_jax, E_np, rtol=1e-3)

    def test_jit_compiles(self, scheme_jax):
        mu_cart = unitsphere2cart_1d_jax(jnp.array(self.MU))
        f = jax.jit(g2zeppelin_signal)
        result = f(scheme_jax['bvalues'], scheme_jax['gradient_directions'],
                   mu_cart, self.LAMBDA_PAR, self.LAMBDA_PERP)
        assert result.shape == scheme_jax['bvalues'].shape

    def test_grad_wrt_lambda_par(self, scheme_jax):
        mu_cart = unitsphere2cart_1d_jax(jnp.array(self.MU))
        f = jax.grad(lambda l: jnp.sum(g2zeppelin_signal(
            scheme_jax['bvalues'], scheme_jax['gradient_directions'],
            mu_cart, l, self.LAMBDA_PERP)))
        g = f(self.LAMBDA_PAR)
        assert jnp.isfinite(g)

    def test_reduces_to_ball_when_isotropic(self, scheme_jax):
        """When lambda_par == lambda_perp, Zeppelin == Ball."""
        mu_cart = unitsphere2cart_1d_jax(jnp.array(self.MU))
        lam = 1.5e-9
        E_zep = g2zeppelin_signal(
            scheme_jax['bvalues'], scheme_jax['gradient_directions'],
            mu_cart, lam, lam)
        E_ball = g1ball_signal(scheme_jax['bvalues'], lam)
        assert_allclose(np.array(E_zep), np.array(E_ball), rtol=3e-3)
