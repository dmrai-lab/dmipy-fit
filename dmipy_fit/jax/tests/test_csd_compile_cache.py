"""Tests for dmipy-fit#32 items 1-2 (value-keyed compile cache, warm-up) and item 4 (x64 pinning).

``AcquisitionScheme``/``PGSEAcquisitionScheme`` have no ``__hash__``/``__eq__`` by design (a
scheme is not made hashable by identity tricks); ``fingerprint()`` is the explicit value identity
these tests check. The compile cache is the module-level ``_compiled_fit_batch`` in
``dmipy_fit.jax.csd_jax``, keyed on (scheme fingerprint, sh_order, unity_constraint, lambda_lb,
maxiter, tol, dtype, batch); ``CsdOsqpOptimizer`` goes through it, and ``warm()`` populates it
ahead of a request.
"""
import os
import time

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jaxopt = pytest.importorskip('jaxopt')

from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy_fit.signal_models.gaussian_models import G1Ball
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.core.modeling_framework import MultiCompartmentSphericalHarmonicsModel
from dmipy_fit.jax import csd_jax
from dmipy_fit.jax.csd_jax import (
    CsdOsqpOptimizer, warm, pin_x64_off, x64_pinned, X64PinnedError,
    compile_cache_info, clear_compile_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_scheme(n_dir=12, seed=0, b_s_mm2=1000.0):
    """A tiny PGSE scheme (4 b0 + n_dir DWI) -- fast to compile, for cache-behaviour tests."""
    rng = np.random.default_rng(seed)
    dirs_dw = rng.normal(size=(n_dir, 3))
    dirs_dw /= np.linalg.norm(dirs_dw, axis=1, keepdims=True)
    bvals = np.concatenate([np.zeros(4), np.full(n_dir, b_s_mm2)]) * 1e6
    dirs = np.concatenate([np.zeros((4, 3)), dirs_dw], axis=0)
    return acquisition_scheme_from_bvalues(bvals, dirs, delta=0.01, Delta=0.03)


def _stick_ball_model(scheme, lambda_par=1.7e-9, lambda_iso=3.0e-9, sh_order=4):
    mc = MultiCompartmentSphericalHarmonicsModel(models=[C1Stick(), G1Ball()], sh_order=sh_order)
    mc.set_fixed_parameter('C1Stick_1_lambda_par', lambda_par)
    mc.set_fixed_parameter('G1Ball_1_lambda_iso', lambda_iso)
    mc.scheme = scheme
    mc._check_if_kernel_parameters_are_fixed()
    mc.S0_responses = np.ones(len(mc.models), dtype=float)
    return mc


@pytest.fixture(autouse=True)
def _clean_cache_and_x64():
    """Every test starts from a clean compile cache and an unpinned, x64-off process state."""
    clear_compile_cache()
    prev_pinned = csd_jax._x64_pinned_off
    prev_x64 = jax.config.jax_enable_x64
    csd_jax._x64_pinned_off = False
    jax.config.update("jax_enable_x64", False)
    yield
    clear_compile_cache()
    csd_jax._x64_pinned_off = prev_pinned
    jax.config.update("jax_enable_x64", prev_x64)


# ---------------------------------------------------------------------------
# 1. fingerprint()
# ---------------------------------------------------------------------------

class TestFingerprint:
    def test_equal_numbers_equal_fingerprint(self):
        s1 = _small_scheme(seed=0)
        s2 = _small_scheme(seed=0)
        assert s1 is not s2
        assert s1.fingerprint() == s2.fingerprint()

    def test_different_directions_different_fingerprint(self):
        s1 = _small_scheme(seed=0)
        s2 = _small_scheme(seed=1)
        assert s1.fingerprint() != s2.fingerprint()

    def test_changed_bvalue_changes_fingerprint(self):
        s1 = _small_scheme(seed=0, b_s_mm2=1000.0)
        s2 = _small_scheme(seed=0, b_s_mm2=1000.001)
        assert s1.fingerprint() != s2.fingerprint()

    def test_fingerprint_is_a_hex_digest(self):
        s = _small_scheme()
        fp = s.fingerprint()
        assert isinstance(fp, str)
        assert len(fp) == 64
        int(fp, 16)  # raises ValueError if not hex

    def test_no_value_hash_or_eq_added(self):
        """fingerprint() is the value identity; __eq__/__hash__ stay the untouched object defaults
        (no identity-hashing trick), so two equal-valued schemes are still not '==' or hash-equal."""
        from dmipy_fit.core.acquisition_scheme import PGSEAcquisitionScheme
        assert PGSEAcquisitionScheme.__eq__ is object.__eq__
        assert PGSEAcquisitionScheme.__hash__ is object.__hash__
        s1, s2 = _small_scheme(seed=0), _small_scheme(seed=0)
        assert s1 != s2
        assert hash(s1) != hash(s2)


# ---------------------------------------------------------------------------
# 2. compile cache
# ---------------------------------------------------------------------------

class TestCompileCache:
    def test_two_optimizers_equal_scheme_share_compiled_kernel(self):
        scheme_a = _small_scheme(seed=0)
        scheme_b = _small_scheme(seed=0)  # different object, equal numbers
        assert scheme_a.fingerprint() == scheme_b.fingerprint()

        mc_a = _stick_ball_model(scheme_a)
        mc_b = _stick_ball_model(scheme_b)
        x0_a = np.reshape(mc_a.parameter_initial_guess_to_parameter_vector(), (1, -1))
        x0_b = np.reshape(mc_b.parameter_initial_guess_to_parameter_vector(), (1, -1))

        opt_a = CsdOsqpOptimizer(scheme_a, mc_a, x0_a, sh_order=4, unity_constraint=True)
        info_after_first = compile_cache_info()
        opt_b = CsdOsqpOptimizer(scheme_b, mc_b, x0_b, sh_order=4, unity_constraint=True)
        info_after_second = compile_cache_info()

        assert opt_a._fit_batch_fn is opt_b._fit_batch_fn
        assert info_after_first.misses == 1
        assert info_after_second.misses == 1
        assert info_after_second.hits == info_after_first.hits + 1

    def test_different_scheme_is_a_cache_miss(self):
        scheme_a = _small_scheme(seed=0)
        scheme_c = _small_scheme(seed=2)
        mc_a = _stick_ball_model(scheme_a)
        mc_c = _stick_ball_model(scheme_c)
        x0_a = np.reshape(mc_a.parameter_initial_guess_to_parameter_vector(), (1, -1))
        x0_c = np.reshape(mc_c.parameter_initial_guess_to_parameter_vector(), (1, -1))

        opt_a = CsdOsqpOptimizer(scheme_a, mc_a, x0_a, sh_order=4)
        opt_c = CsdOsqpOptimizer(scheme_c, mc_c, x0_c, sh_order=4)
        assert opt_a._fit_batch_fn is not opt_c._fit_batch_fn
        assert compile_cache_info().misses == 2

    def test_different_sh_order_is_a_cache_miss(self):
        scheme = _small_scheme(seed=0)
        mc4 = _stick_ball_model(scheme, sh_order=4)
        mc8 = _stick_ball_model(scheme, sh_order=8)
        x0_4 = np.reshape(mc4.parameter_initial_guess_to_parameter_vector(), (1, -1))
        x0_8 = np.reshape(mc8.parameter_initial_guess_to_parameter_vector(), (1, -1))

        opt4 = CsdOsqpOptimizer(scheme, mc4, x0_4, sh_order=4)
        opt8 = CsdOsqpOptimizer(scheme, mc8, x0_8, sh_order=8)
        assert opt4._fit_batch_fn is not opt8._fit_batch_fn
        assert compile_cache_info().misses == 2

    def test_cached_kernel_gives_correct_answer_for_a_different_response(self):
        """The compiled kernel is reused across a DIFFERENT tissue response on an equal scheme --
        it must still solve the right QP for each (Q/AT/G/h are traced arguments, not baked-in
        constants), not silently reuse a stale response's answer."""
        scheme_a = _small_scheme(seed=0)
        scheme_b = _small_scheme(seed=0)

        mc_a = _stick_ball_model(scheme_a, lambda_par=1.7e-9, lambda_iso=3.0e-9)
        mc_b = _stick_ball_model(scheme_b, lambda_par=0.9e-9, lambda_iso=2.0e-9)  # different response
        x0_a = np.reshape(mc_a.parameter_initial_guess_to_parameter_vector(), (1, -1))
        x0_b = np.reshape(mc_b.parameter_initial_guess_to_parameter_vector(), (1, -1))

        opt_a = CsdOsqpOptimizer(scheme_a, mc_a, x0_a, sh_order=4, unity_constraint=False)
        opt_b = CsdOsqpOptimizer(scheme_b, mc_b, x0_b, sh_order=4, unity_constraint=False)
        assert opt_a._fit_batch_fn is opt_b._fit_batch_fn  # same compiled kernel

        mu = [np.pi / 3, np.pi / 5]
        signal_a = 0.7 * C1Stick()(scheme_a, lambda_par=1.7e-9, mu=mu) + 0.3 * G1Ball()(scheme_a, lambda_iso=3.0e-9)
        signal_b = 0.7 * C1Stick()(scheme_b, lambda_par=0.9e-9, mu=mu) + 0.3 * G1Ball()(scheme_b, lambda_iso=2.0e-9)
        signal_a = signal_a / np.mean(signal_a[scheme_a.b0_mask])
        signal_b = signal_b / np.mean(signal_b[scheme_b.b0_mask])

        sol_a = opt_a(signal_a, x0_a[0])
        sol_b = opt_b(signal_b, x0_b[0])
        # Solving each response's own (matching) signal must recover it, not the other's --
        # if the kernel wrongly reused opt_a's closed-over Q/AT this would be a poor fit for b.
        params_a = mc_a.parameter_vector_to_parameters(sol_a)
        params_b = mc_b.parameter_vector_to_parameters(sol_b)
        assert params_a['partial_volume_1'] == pytest.approx(0.3, abs=0.15)
        assert params_b['partial_volume_1'] == pytest.approx(0.3, abs=0.15)

    def test_cache_bounded(self):
        """maxsize=16: pushing more distinct keys through evicts the oldest (functools.lru_cache)."""
        assert compile_cache_info().maxsize == 16


# ---------------------------------------------------------------------------
# 3. warm()
# ---------------------------------------------------------------------------

class TestWarm:
    def test_warm_reports_one_row_per_scheme_and_batch(self):
        scheme = _small_scheme(seed=0)
        report = warm([scheme], sh_order=4, batch=(2, 4))
        assert len(report) == 2
        batches = sorted(r['batch'] for r in report)
        assert batches == [2, 4]
        for r in report:
            assert r['scheme_fingerprint'] == scheme.fingerprint()
            assert r['n_measurements'] == scheme.number_of_measurements
            assert r['seconds'] > 0

    def test_warm_then_fit_hits_the_cache(self):
        scheme_warm = _small_scheme(seed=0)
        warm([scheme_warm], sh_order=4, unity_constraint=True, batch=4)
        assert compile_cache_info().misses == 1

        # A real server fixes DMIPY_CSD_JAX_BATCH for its whole lifetime (batch is part of the
        # cache key); warm() only holds it during its own construction, so match that here.
        prev = os.environ.get("DMIPY_CSD_JAX_BATCH")
        os.environ["DMIPY_CSD_JAX_BATCH"] = "4"
        try:
            scheme_fit = _small_scheme(seed=0)  # different object, equal numbers
            mc = _stick_ball_model(scheme_fit)
            x0 = np.reshape(mc.parameter_initial_guess_to_parameter_vector(), (1, -1))
            opt = CsdOsqpOptimizer(scheme_fit, mc, x0, sh_order=4, unity_constraint=True)
            assert compile_cache_info().misses == 1   # still one miss: warm() already built this key
            assert compile_cache_info().hits >= 1

            data = np.zeros((4, scheme_fit.number_of_measurements))
            x0_all = np.tile(x0, (4, 1))
            opt.fit_batch(data, x0_all)
        finally:
            if prev is None:
                os.environ.pop("DMIPY_CSD_JAX_BATCH", None)
            else:
                os.environ["DMIPY_CSD_JAX_BATCH"] = prev
        assert compile_cache_info().misses == 1  # the actual fit_batch call also paid no new compile

    def test_warm_does_not_leak_the_batch_env_var(self):
        prev = os.environ.get("DMIPY_CSD_JAX_BATCH")
        try:
            os.environ.pop("DMIPY_CSD_JAX_BATCH", None)
            warm([_small_scheme(seed=0)], sh_order=4, batch=4)
            assert "DMIPY_CSD_JAX_BATCH" not in os.environ
        finally:
            if prev is not None:
                os.environ["DMIPY_CSD_JAX_BATCH"] = prev


# ---------------------------------------------------------------------------
# 4. x64 pinning
# ---------------------------------------------------------------------------

class TestX64Pinning:
    def test_unpinned_by_default(self):
        assert x64_pinned() is False

    def test_pin_turns_x64_off(self):
        jax.config.update("jax_enable_x64", True)
        pin_x64_off()
        assert x64_pinned() is True
        assert jax.config.jax_enable_x64 is False

    def test_pinned_and_still_off_fits_normally(self):
        pin_x64_off()
        scheme = _small_scheme(seed=0)
        mc = _stick_ball_model(scheme)
        x0 = np.reshape(mc.parameter_initial_guess_to_parameter_vector(), (1, -1))
        opt = CsdOsqpOptimizer(scheme, mc, x0, sh_order=4)
        signal = np.ones((1, scheme.number_of_measurements))
        result = opt.fit_batch(signal, x0)
        assert result.shape == x0.shape

    def test_pinned_but_x64_flipped_back_on_refuses_by_name(self):
        pin_x64_off()
        scheme = _small_scheme(seed=0)
        mc = _stick_ball_model(scheme)
        x0 = np.reshape(mc.parameter_initial_guess_to_parameter_vector(), (1, -1))
        opt = CsdOsqpOptimizer(scheme, mc, x0, sh_order=4)

        jax.config.update("jax_enable_x64", True)  # simulate a preceding cylinder fit
        signal = np.ones((1, scheme.number_of_measurements))
        with pytest.raises(X64PinnedError):
            opt.fit_batch(signal, x0)

    def test_unpinned_still_saves_and_restores(self):
        """Default (no pin): fit_batch tolerates an ambient x64=True and restores it afterwards."""
        scheme = _small_scheme(seed=0)
        mc = _stick_ball_model(scheme)
        x0 = np.reshape(mc.parameter_initial_guess_to_parameter_vector(), (1, -1))
        opt = CsdOsqpOptimizer(scheme, mc, x0, sh_order=4)

        jax.config.update("jax_enable_x64", True)
        signal = np.ones((1, scheme.number_of_measurements))
        result = opt.fit_batch(signal, x0)
        assert result.shape == x0.shape
        assert jax.config.jax_enable_x64 is True  # restored
