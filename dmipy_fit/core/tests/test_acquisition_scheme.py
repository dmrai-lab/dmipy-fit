import numpy as np
from dmipy_fit.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme)
from dmipy_fit.core.acquisition_scheme import (
    acquisition_scheme_from_bvalues,
    acquisition_scheme_from_qvalues,
    acquisition_scheme_from_gradient_strengths,
    calculate_shell_bvalues_and_indices,
    gtab_dipy2dmipy, gtab_dmipy2dipy)
from dmipy_fit.core.modeling_framework import (
    MultiCompartmentModel)
from dmipy_fit.signal_models.cylinder_models import (
    C4CylinderGaussianPhaseApproximation)
from dipy.core.gradients import gradient_table
from numpy.testing import (
    assert_raises, assert_equal, assert_array_equal)


def test_catch_negative_bvalues(Nsamples=10):
    bvalues = np.tile(-1, Nsamples)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    delta = np.ones(Nsamples)
    Delta = np.ones(Nsamples)
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)


def test_catch_different_length_bvals_bvecs(Nsamples=10):
    bvalues = np.tile(1, Nsamples - 1)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    delta = np.ones(Nsamples)
    Delta = np.ones(Nsamples)
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)


def test_catch_2d_bvals_bvecs(Nsamples=10):
    bvalues = np.ones((Nsamples, 2))
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    delta = np.ones(Nsamples)
    Delta = np.ones(Nsamples)
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)


def test_catch_different_shape_bvals_delta_Delta(Nsamples=10):
    bvalues = np.tile(1, Nsamples)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    delta = np.ones(Nsamples - 1)
    Delta = np.ones(Nsamples)
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)

    delta = np.ones(Nsamples)
    Delta = np.ones(Nsamples - 1)
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)


def test_catch_2d_delta_Delta(Nsamples=10):
    bvalues = np.tile(1, Nsamples)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    delta = np.ones((Nsamples, 2))
    Delta = np.ones(Nsamples)
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)

    delta = np.ones(Nsamples)
    Delta = np.ones((Nsamples, 2))
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)


def test_catch_negative_delta_Delta(Nsamples=10):
    bvalues = np.tile(1, Nsamples)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    delta = -np.ones(Nsamples)
    Delta = np.ones(Nsamples)
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)

    delta = np.ones(Nsamples)
    Delta = -np.ones(Nsamples)
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)


def test_catch_wrong_shape_bvecs(Nsamples=10):
    bvalues = np.tile(1, Nsamples)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1, 1))
    delta = np.ones(Nsamples)
    Delta = np.ones(Nsamples)
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)


def test_catch_non_unit_vector_bvecs(Nsamples=10):
    bvalues = np.tile(1, Nsamples)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1)) + 1.
    delta = np.ones(Nsamples)
    Delta = np.ones(Nsamples)
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)


def test_equivalent_scheme_bvals_and_bvecs(Nsamples=10):
    bvalues = np.tile(1, Nsamples)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    delta = np.ones(Nsamples)
    Delta = np.ones(Nsamples)
    scheme_from_bvals = acquisition_scheme_from_bvalues(
        bvalues, bvecs, delta, Delta)
    qvalues = scheme_from_bvals.qvalues
    scheme_from_qvals = acquisition_scheme_from_qvalues(
        qvalues, bvecs, delta, Delta)
    bvalues_from_qvalues = scheme_from_qvals.bvalues
    assert_array_equal(bvalues, bvalues_from_qvalues)


def test_equivalent_scheme_bvals_and_gradient_strength(Nsamples=10):
    bvalues = np.tile(1, Nsamples)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    delta = np.ones(Nsamples)
    Delta = np.ones(Nsamples)
    scheme_from_bvals = acquisition_scheme_from_bvalues(
        bvalues, bvecs, delta, Delta)
    gradient_strengths = scheme_from_bvals.gradient_strengths
    scheme_from_gradient_strengths = (
        acquisition_scheme_from_gradient_strengths(
            gradient_strengths, bvecs, delta, Delta))
    bvalues_from_gradient_strengths = (
        scheme_from_gradient_strengths.bvalues)
    assert_array_equal(bvalues, bvalues_from_gradient_strengths)


def test_estimate_shell_indices():
    bvalues = np.arange(10)
    max_distance = 1
    shell_indices, shell_bvalues = (
        calculate_shell_bvalues_and_indices(
            bvalues, max_distance=max_distance))
    assert_equal(np.unique(shell_indices), np.array([0]))
    assert_equal(float(np.asarray(shell_bvalues).item()), np.mean(bvalues))

    max_distance = 0.5
    shell_indices, shell_bvalues = (
        calculate_shell_bvalues_and_indices(
            bvalues, max_distance=max_distance))
    assert_array_equal(shell_indices, bvalues)


def test_shell_indices_with_varying_diffusion_times(Nsamples=10):
    # tests whether measurements with the same bvalue but different diffusion
    # time are correctly classified in different shells
    bvalues = np.tile(1e9, Nsamples)
    delta = 0.01
    Delta = np.hstack([np.tile(0.01, len(bvalues) // 2),
                       np.tile(0.03, len(bvalues) // 2)])
    gradient_directions = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    scheme = acquisition_scheme_from_bvalues(
        bvalues, gradient_directions, delta, Delta)
    assert_equal(len(np.unique(scheme.shell_indices)), 2)


def test_dipy2dmipy_acquisition_converter(Nsamples=10):
    bvals = np.tile(1e3, Nsamples)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    big_delta = 0.03
    small_delta = 0.01
    gtab_dipy = gradient_table(
        bvals=bvals, bvecs=bvecs, small_delta=small_delta, big_delta=big_delta)
    gtab_mipy = gtab_dipy2dmipy(gtab_dipy)
    assert_array_equal(gtab_mipy.bvalues / 1e6, gtab_dipy.bvals)
    assert_array_equal(gtab_mipy.gradient_directions, gtab_dipy.bvecs)
    assert_equal(np.unique(gtab_mipy.Delta), gtab_dipy.big_delta)
    assert_equal(np.unique(gtab_mipy.delta), gtab_dipy.small_delta)


def test_dmipy2dipy_acquisition_converter(Nsamples=10):
    bvals = np.tile(1e9, Nsamples)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    big_delta = 0.03
    small_delta = 0.01
    gtab_mipy = acquisition_scheme_from_bvalues(
        bvalues=bvals, gradient_directions=bvecs,
        delta=small_delta, Delta=big_delta)
    gtab_dipy = gtab_dmipy2dipy(gtab_mipy)
    assert_array_equal(gtab_mipy.bvalues / 1e6, gtab_dipy.bvals)
    assert_array_equal(gtab_mipy.gradient_directions, gtab_dipy.bvecs)
    assert_equal(gtab_mipy.Delta, gtab_dipy.big_delta)
    assert_equal(gtab_mipy.delta, gtab_dipy.small_delta)


def test_acquisition_scheme_summary(Nsamples=10):
    bvals = np.tile(1e9, Nsamples)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    big_delta = 0.03
    small_delta = 0.01
    gtab_mipy = acquisition_scheme_from_bvalues(
        bvalues=bvals, gradient_directions=bvecs,
        delta=small_delta, Delta=big_delta)
    gtab_mipy.print_acquisition_info


def test_acquisition_scheme_pruning():
    scheme = wu_minn_hcp_acquisition_scheme()
    test_data = np.random.rand(len(scheme.bvalues))

    scheme_pr, data_pr = scheme.return_pruned_acquisition_scheme(
        [2], test_data)
    assert_array_equal(
        scheme_pr.bvalues,
        scheme.bvalues[scheme.shell_indices == 2])
    assert_array_equal(
        data_pr,
        test_data[scheme.shell_indices == 2])


def test_acq_scheme_without_deltas_model_catch():
    scheme = wu_minn_hcp_acquisition_scheme()
    test_data = np.random.rand(len(scheme.bvalues))
    scheme_clinical = acquisition_scheme_from_bvalues(
        scheme.bvalues, scheme.gradient_directions)
    mc_model = MultiCompartmentModel(
        [C4CylinderGaussianPhaseApproximation()])
    assert_raises(ValueError, mc_model.fit, scheme_clinical, test_data)


# ------------------------------------------------------------------
# to_gradient_array tests
# ------------------------------------------------------------------

def _make_pgse_scheme(n_m=20, delta=20e-3, Delta=40e-3, g_max=300e-3):
    G_mag = np.linspace(10e-3, g_max, n_m)
    bvecs = np.tile(np.r_[1., 0., 0.], (n_m, 1))
    delta_arr = np.full(n_m, delta)
    Delta_arr = np.full(n_m, Delta)
    return acquisition_scheme_from_gradient_strengths(
        G_mag, bvecs, delta_arr, Delta_arr), G_mag


def test_to_gradient_array_shape():
    """to_gradient_array() returns G with shape (n_m, n_t, 3) and scalar dt."""
    n_t = 500
    scheme, _ = _make_pgse_scheme()
    G, dt = scheme.to_gradient_array(n_t=n_t)
    n_m = scheme.number_of_measurements
    assert G.shape == (n_m, n_t, 3), f"Expected ({n_m}, {n_t}, 3), got {G.shape}"
    assert isinstance(dt, float)
    assert dt > 0


def test_to_gradient_array_b_roundtrip():
    """b-values computed from to_gradient_array() match scheme.bvalues within 1%."""
    from dmipy_fit.core.constants import CONSTANTS
    GAMMA = CONSTANTS['water_gyromagnetic_ratio']

    scheme, _ = _make_pgse_scheme()
    G_arr, dt = scheme.to_gradient_array(n_t=1000)

    # Compute b via rectangular q accumulation + trapezoidal integration
    q = np.cumsum(G_arr * dt, axis=1) * GAMMA  # (n_m, n_t, 3)
    q_sq = np.sum(q**2, axis=2)                 # (n_m, n_t)
    b_from_G = np.trapezoid(q_sq, dx=dt, axis=1)      # (n_m,)

    np.testing.assert_allclose(
        b_from_G, scheme.bvalues, rtol=0.01,
        err_msg="b-values from to_gradient_array() deviate >1% from scheme.bvalues")


def test_to_gradient_array_raises_without_timing():
    """to_gradient_array() raises ValueError when delta/Delta are None."""
    scheme = wu_minn_hcp_acquisition_scheme()
    # HCP scheme has no timing — delta/Delta may be None (or schematic only)
    # Create a bvalues-only scheme explicitly
    bvals = np.tile(1e9, 10)
    bvecs = np.tile(np.r_[1., 0., 0.], (10, 1))
    scheme_no_timing = acquisition_scheme_from_bvalues(bvals, bvecs)
    assert_raises(ValueError, scheme_no_timing.to_gradient_array)
