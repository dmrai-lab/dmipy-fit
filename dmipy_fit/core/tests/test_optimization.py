from dmipy_fit.signal_models import (
    cylinder_models, gaussian_models, sphere_models)
from dmipy_fit.core import modeling_framework
from dmipy_fit.utils import utils
from numpy.testing import (
    assert_equal, assert_array_almost_equal, assert_array_equal)
import numpy as np
import pytest
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme

scheme = wu_minn_hcp_acquisition_scheme()

# The fits recover noiseless data from a random start. The optimizer (L-BFGS-B) stops on a relative decrease of its
# objective, which does not bound the error that is left, so the tolerances are the convergence the fits show: over 100
# seeds the largest errors are 6e-4 rad for the axis, 8e-3 relative for a diffusivity and 1.3e-3 for a fraction; the
# tolerances keep a margin of 6 to 11 on them.
AXIS_TOLERANCE = 5e-3  # rad
DIFFUSIVITY_RTOL = 5e-2
FRACTION_ATOL = 1.5e-2


def _axis(mu):
    "The unit vector of the orientation (theta, phi)."
    return utils.sphere2cart(np.r_[1., np.ravel(mu)])


def assert_same_axis(mu, mu_true, tolerance=AXIS_TOLERANCE):
    """The orientations (theta, phi) are the same axis: the angle between the unit vector of one and that of the other
    or its reverse is below ``tolerance``. A stick or a zeppelin is the same for an axis and its reverse, (theta, phi)
    and (pi - theta, phi + pi) are one axis, and phi is undefined at the poles, so the parameters themselves are not
    compared."""
    a, b = _axis(mu), _axis(mu_true)
    angle = np.arctan2(np.linalg.norm(np.cross(a, b)), abs(np.dot(a, b)))
    assert angle <= tolerance, "the axes differ by {:.3e} rad (tolerance {:.1e})".format(angle, tolerance)


def assert_close(value, value_true, rtol=DIFFUSIVITY_RTOL):
    value = float(np.squeeze(value))
    assert abs(value - value_true) <= rtol * abs(value_true), (
        "{:.6e} differs from {:.6e} by more than {:.0e} relative".format(value, value_true, rtol))


def test_axes_are_compared_up_to_sign_and_at_the_poles():
    mu = np.r_[0.7, 0.4]
    assert_same_axis(mu, mu)
    assert_same_axis(np.r_[np.pi - 0.7, 0.4 - np.pi], mu)  # the reverse axis
    assert_same_axis(np.r_[0.7, 0.4 + 1e-4], mu)
    with pytest.raises(AssertionError):
        assert_same_axis(np.r_[0.7, 0.4 + 0.1], mu)
    # phi is undefined at the pole: theta = 1e-3 moves the axis by 1e-3 rad whatever phi is
    assert_same_axis(np.r_[1e-3, 2.5], np.r_[1e-3, -1.0], tolerance=3e-3)
    with pytest.raises(AssertionError):
        assert_same_axis(np.r_[0.1, 2.5], np.r_[0.1, -1.0])


# seeds 34, 53, 126 and 150 draw an axis within 0.011 rad of the pole, where phi is undefined: a fit that recovers the
# axis returns another phi, and a comparison of the parameters (theta, phi) fails on them
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 34, 53, 126, 150])
def test_simple_stick_optimization(seed):
    rng = np.random.default_rng(seed)
    stick = cylinder_models.C1Stick()
    gt_mu = rng.random(2)
    gt_lambda_par = (rng.random() + 1.) * 1e-9

    stick_model = modeling_framework.MultiCompartmentModel(
        models=[stick])

    gt_parameter_vector = stick_model.parameters_to_parameter_vector(
        C1Stick_1_lambda_par=gt_lambda_par, C1Stick_1_mu=gt_mu)

    E = stick_model.simulate_signal(scheme, gt_parameter_vector)

    stick_model.set_initial_guess_parameter('C1Stick_1_lambda_par',
                                            (rng.random() + 1.) * 1e-9)
    stick_model.set_initial_guess_parameter('C1Stick_1_mu', rng.random(2))
    fitted = stick_model.fit(scheme, E).fitted_parameters
    assert_same_axis(fitted['C1Stick_1_mu'], gt_mu)
    assert_close(fitted['C1Stick_1_lambda_par'], gt_lambda_par)


@pytest.mark.parametrize("seed", range(4))
def test_simple_ball_and_stick_optimization(seed):
    rng = np.random.default_rng(seed)
    stick = cylinder_models.C1Stick()
    ball = gaussian_models.G1Ball()

    ball_and_stick = (
        modeling_framework.MultiCompartmentModel(
            models=[ball, stick])
    )
    gt_mu = np.clip(rng.random(2), .3, np.inf)
    gt_lambda_par = (rng.random() + 1.) * 1e-9
    gt_lambda_iso = gt_lambda_par / 2.
    gt_partial_volume = 0.3

    gt_parameter_vector = ball_and_stick.parameters_to_parameter_vector(
        C1Stick_1_lambda_par=gt_lambda_par,
        G1Ball_1_lambda_iso=gt_lambda_iso,
        C1Stick_1_mu=gt_mu,
        partial_volume_0=gt_partial_volume,
        partial_volume_1=1 - gt_partial_volume
    )

    E = ball_and_stick.simulate_signal(
        scheme, gt_parameter_vector)

    vf_rand = rng.random()
    ball_and_stick.set_initial_guess_parameter(
        'C1Stick_1_lambda_par', (rng.random() + 1.) * 1e-9)
    ball_and_stick.set_initial_guess_parameter(
        'G1Ball_1_lambda_iso', gt_lambda_par / 2.)
    ball_and_stick.set_initial_guess_parameter(
        'C1Stick_1_mu', rng.random(2))
    ball_and_stick.set_initial_guess_parameter('partial_volume_0', vf_rand)
    ball_and_stick.set_initial_guess_parameter('partial_volume_1', 1 - vf_rand)

    fitted = ball_and_stick.fit(scheme, E).fitted_parameters
    assert_same_axis(fitted['C1Stick_1_mu'], gt_mu)
    assert_close(fitted['C1Stick_1_lambda_par'], gt_lambda_par)
    assert_close(fitted['G1Ball_1_lambda_iso'], gt_lambda_iso)
    assert abs(float(np.squeeze(fitted['partial_volume_0'])) - gt_partial_volume) <= FRACTION_ATOL


def test_simple_ball_and_stick_optimization_vf_fixed():
    rng = np.random.default_rng(0)
    stick = cylinder_models.C1Stick()
    ball = gaussian_models.G1Ball()

    ball_and_stick = (
        modeling_framework.MultiCompartmentModel(
            models=[ball, stick])
    )
    gt_mu = np.clip(rng.random(2), .3, np.inf)
    gt_lambda_par = (rng.random() + 1.) * 1e-9
    gt_lambda_iso = gt_lambda_par / 2.
    gt_partial_volume = 0.3

    gt_parameter_vector = ball_and_stick.parameters_to_parameter_vector(
        C1Stick_1_lambda_par=gt_lambda_par,
        G1Ball_1_lambda_iso=gt_lambda_iso,
        C1Stick_1_mu=gt_mu,
        partial_volume_0=gt_partial_volume,
        partial_volume_1=1 - gt_partial_volume
    )

    E = ball_and_stick.simulate_signal(
        scheme, gt_parameter_vector)

    E2d = np.array([E, E])

    vf_rand = rng.random()
    ball_and_stick.set_fixed_parameter(
        'partial_volume_0', np.r_[vf_rand, vf_rand])
    ball_and_stick.set_fixed_parameter(
        'partial_volume_1', np.r_[1 - vf_rand, 1 - vf_rand])
    ball_and_stick.set_initial_guess_parameter(
        'C1Stick_1_lambda_par', (rng.random() + 1.) * 1e-9)
    ball_and_stick.set_initial_guess_parameter(
        'G1Ball_1_lambda_iso', gt_lambda_par / 2.)
    ball_and_stick.set_initial_guess_parameter(
        'C1Stick_1_mu', rng.random(2))

    vf_fitted = ball_and_stick.fit(
        scheme, E2d).fitted_parameters['partial_volume_0'][0]
    assert_equal(vf_fitted, vf_rand)


def test_multi_dimensional_x0():
    rng = np.random.default_rng(0)
    stick = cylinder_models.C1Stick()
    ball = gaussian_models.G1Ball()
    ball_and_stick = (
        modeling_framework.MultiCompartmentModel(
            models=[ball, stick],)
    )
    gt_lambda_par = (rng.random() + 1.) * 1e-9
    gt_lambda_iso = gt_lambda_par / 2.
    gt_partial_volume = 0.3
    gt_mu_array = np.empty((10, 10, 2))

    # I'm putting the orientation of the stick all over the sphere.
    for i, mu1 in enumerate(np.linspace(0, np.pi, 10)):
        for j, mu2 in enumerate(np.linspace(-np.pi, np.pi, 10)):
            gt_mu_array[i, j] = np.r_[mu1, mu2]

    gt_parameter_vector = (
        ball_and_stick.parameters_to_parameter_vector(
            C1Stick_1_lambda_par=gt_lambda_par,
            G1Ball_1_lambda_iso=gt_lambda_iso,
            C1Stick_1_mu=gt_mu_array,
            partial_volume_0=gt_partial_volume,
            partial_volume_1=1 - gt_partial_volume)
    )

    E_array = ball_and_stick.simulate_signal(
        scheme, gt_parameter_vector)

    ball_and_stick.set_initial_guess_parameter(
        'C1Stick_1_lambda_par', gt_lambda_par)
    ball_and_stick.set_initial_guess_parameter(
        'G1Ball_1_lambda_iso', gt_lambda_iso)
    ball_and_stick.set_initial_guess_parameter(
        'C1Stick_1_mu', gt_mu_array)
    ball_and_stick.set_initial_guess_parameter(
        'partial_volume_0', gt_partial_volume)
    ball_and_stick.set_initial_guess_parameter(
        'partial_volume_1', 1 - gt_partial_volume)
    # I'm giving a voxel-dependent initial condition with gt_mu_array
    res = ball_and_stick.fit(scheme, E_array).fitted_parameters_vector
    # optimization should stop immediately as I'm giving the ground truth.
    # NaN entries (passive T2 params) are excluded from comparison.
    diff = np.ravel(res - gt_parameter_vector)
    non_nan = ~np.isnan(diff)
    assert_equal(np.all(diff[non_nan] == 0.), True)
    # and the parameter vector dictionaries of the results and x0 should also
    # be the same.
    res_parameters = ball_and_stick.parameter_vector_to_parameters(res)
    x0_parameters = ball_and_stick.parameter_vector_to_parameters(
        gt_parameter_vector)
    for key in res_parameters.keys():
        assert_array_equal(x0_parameters[key], res_parameters[key])


def test_stick_and_tortuous_zeppelin_to_spherical_mean_fit():
    """ this is a more complex test to see if we can generate 3D data using a
    stick and zeppelin model, where we assume the perpendicular diffusivity is
    linked to the parallel diffusivity and volume fraction using tortuosity. We
    then use the spherical mean models of stick and zeppelin with the same
    tortuosity assumption to fit the 3D data (and estimating the spherical mean
    of each shell). The final check is whether the parallel diffusivity and
    volume fraction between the 3D and spherical mean models correspond."""

    rng = np.random.default_rng(0)
    gt_mu = np.clip(rng.random(2), .3, np.inf)
    gt_lambda_par = (rng.random() + 1.) * 1e-9
    gt_partial_volume = 0.3

    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()

    stick_and_zeppelin = (
        modeling_framework.MultiCompartmentModel(
            models=[stick, zeppelin])
    )

    stick_and_zeppelin.set_tortuous_parameter(
        'G2Zeppelin_1_lambda_perp',
        'C1Stick_1_lambda_par',
        'partial_volume_0',
        'partial_volume_1'
    )
    stick_and_zeppelin.set_equal_parameter(
        'C1Stick_1_mu',
        'G2Zeppelin_1_mu'
    )

    stick_and_zeppelin.set_equal_parameter(
        'C1Stick_1_lambda_par',
        'G2Zeppelin_1_lambda_par'
    )

    gt_parameter_vector = (
        stick_and_zeppelin.parameters_to_parameter_vector(
            C1Stick_1_lambda_par=gt_lambda_par,
            C1Stick_1_mu=gt_mu,
            partial_volume_0=gt_partial_volume,
            partial_volume_1=1 - gt_partial_volume)
    )

    E = stick_and_zeppelin.simulate_signal(
        scheme, gt_parameter_vector)

    # now we make the stick and zeppelin spherical mean model and check if the
    # same lambda_par and volume fraction result as the 3D generated data.
    stick_and_tortuous_zeppelin_sm = (
        modeling_framework.MultiCompartmentSphericalMeanModel(
            models=[stick, zeppelin])
    )

    stick_and_tortuous_zeppelin_sm.set_tortuous_parameter(
        'G2Zeppelin_1_lambda_perp',
        'C1Stick_1_lambda_par',
        'partial_volume_0',
        'partial_volume_1')
    stick_and_tortuous_zeppelin_sm.set_equal_parameter(
        'G2Zeppelin_1_lambda_par',
        'C1Stick_1_lambda_par')

    res_sm = stick_and_tortuous_zeppelin_sm.fit(scheme, E).fitted_parameters

    assert_close(res_sm['G2Zeppelin_1_lambda_par'], gt_lambda_par)
    assert abs(float(res_sm['partial_volume_0'].squeeze()) - gt_partial_volume) <= FRACTION_ATOL


def test_fractions_add_up_to_one():
    dot1 = sphere_models.S1Dot()
    dot2 = sphere_models.S1Dot()
    dot3 = sphere_models.S1Dot()
    dot4 = sphere_models.S1Dot()
    dot5 = sphere_models.S1Dot()
    dots = modeling_framework.MultiCompartmentModel(
        models=[dot1, dot2, dot3, dot4, dot5])
    random_fractions = np.random.default_rng(0).random(5)
    random_fractions /= random_fractions.sum()
    parameter_vector = dots.parameters_to_parameter_vector(
        partial_volume_0=random_fractions[0],
        partial_volume_1=random_fractions[1],
        partial_volume_2=random_fractions[2],
        partial_volume_3=random_fractions[3],
        partial_volume_4=random_fractions[4])
    E = dots.simulate_signal(scheme, parameter_vector)
    assert_array_almost_equal(E, np.ones(len(E)))


def test_MIX_fitting_multimodel():
    ball = gaussian_models.G1Ball()
    zeppelin = gaussian_models.G2Zeppelin()
    ball_and_zeppelin = (
        modeling_framework.MultiCompartmentModel(
            models=[ball, zeppelin]))

    parameter_vector = ball_and_zeppelin.parameters_to_parameter_vector(
        G1Ball_1_lambda_iso=2.7e-9,
        partial_volume_0=.2,
        partial_volume_1=.8,
        G2Zeppelin_1_lambda_perp=.5e-9,
        G2Zeppelin_1_mu=(np.pi / 2., np.pi / 2.),
        G2Zeppelin_1_lambda_par=1.7e-9
    )

    E = ball_and_zeppelin.simulate_signal(
        scheme, parameter_vector)
    fit = ball_and_zeppelin.fit(
        scheme,
        E, solver='mix').fitted_parameters_vector
    assert_array_almost_equal(abs(fit).squeeze(), parameter_vector, 2)


def test_MIX_fitting_singlemodel():
    stick = cylinder_models.C1Stick()
    stick_mod = (
        modeling_framework.MultiCompartmentModel(
            models=[stick]))

    parameter_vector = stick_mod.parameters_to_parameter_vector(
        C1Stick_1_mu=(np.pi / 2., np.pi / 2.),
        C1Stick_1_lambda_par=1.7e-9
    )

    E = stick_mod.simulate_signal(
        scheme, parameter_vector)
    fit = stick_mod.fit(
        scheme,
        E, solver='mix').fitted_parameters_vector
    assert_array_almost_equal(abs(fit).squeeze(), parameter_vector, 2)
