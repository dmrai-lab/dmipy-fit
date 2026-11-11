"""Published NEXI factory versus the independent symmetric-generator oracle."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy_fit.core.modeling_framework import MultiCompartmentSphericalMeanModel
from dmipy_fit.custom_optimizers.reference_models import nexi
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.signal_models.gaussian_models import G1Ball
from dmipy_fit.signal_models.nexi_reference import nexi_jelescu2022


@pytest.fixture(scope='module')
def scheme():
    b = np.tile([0., 0.2e9, 1.e9, 3.e9, 6.e9], 3)
    return acquisition_scheme_from_bvalues(
        np.repeat(b, 2), np.tile([[1., 0., 0.], [0., 0., 1.]], (len(b), 1)),
        delta=0.004, Delta=np.repeat([0.015, 0.04, 0.08], 10))


def test_nexi_composition_and_parameters():
    model = nexi()
    assert isinstance(model, MultiCompartmentSphericalMeanModel)
    assert set(model.parameter_names) == {
        'X1NEXI_1_f', 'X1NEXI_1_D_i', 'X1NEXI_1_D_e', 'X1NEXI_1_t_ex'}
    assert set(model.parameter_types.values()) == {'normal'}
    pair = model.models[0].exchange_model
    assert isinstance(pair.model_intra, C1Stick)
    assert isinstance(pair.model_extra, G1Ball)
    assert pair.parameter_links == []
    assert model.parameter_scales['X1NEXI_1_t_ex'] == 1e-3
    assert_allclose(np.array(model.parameter_ranges['X1NEXI_1_t_ex']) * 1e-3,
                    [0.001, 0.5])


@pytest.mark.parametrize('f', [0.2, 0.5, 0.8])
@pytest.mark.parametrize('di', [1.2e-9, 2.8e-9])
@pytest.mark.parametrize('de', [0.6e-9, 1.8e-9])
@pytest.mark.parametrize('t_ex', [0.005, 0.03, 0.2])
def test_nexi_matches_independent_reference(scheme, f, di, de, t_ex, record_property):
    """36 tissue combinations x 15 b/Delta pairs: rtol=2e-12, atol=2e-14.

    b=0--6e9 s/m², Delta=15--80 ms, delta=4 ms. These kernels remain
    above the known Karger floor; the finite-pulse approximation is unchanged.
    """
    model = nexi()
    sm = scheme.spherical_mean_scheme
    expected = nexi_jelescu2022(sm.bvalues, sm.Delta-sm.delta/3,
                               f=f, di=di, de=de, t_ex=t_ex)
    values = {'X1NEXI_1_f': f, 'X1NEXI_1_D_i': di,
              'X1NEXI_1_D_e': de, 'X1NEXI_1_t_ex': t_ex}
    got = model(scheme, **values)
    record_property('max_absolute_error', float(np.max(np.abs(got-expected))))
    assert_allclose(got, expected, rtol=2e-12, atol=2e-14)
    assert_allclose(model.simulate_signal(scheme, values), expected, rtol=2e-12, atol=2e-14)


@pytest.mark.parametrize('missing', ['f', 'D_i', 'D_e', 't_ex'])
def test_nexi_requires_every_parameter(scheme, missing):
    values = {'f': 0.6, 'D_i': 2.4e-9, 'D_e': 0.9e-9, 't_ex': 0.035}
    del values[missing]
    with pytest.raises(KeyError, match=missing):
        nexi().models[0](scheme.spherical_mean_scheme, **values)


def test_nexi_fits_all_four_parameters(scheme):
    """Noise-free multi-Delta fit from a perturbed guess recovers all four
    physical parameters within 0.2%, with normalized signal residual <1e-10.
    The data are generated independently, rather than by the fitted forward.
    """
    model = nexi()
    truth = {'X1NEXI_1_f': 0.6, 'X1NEXI_1_D_i': 2.4e-9,
             'X1NEXI_1_D_e': 0.9e-9, 'X1NEXI_1_t_ex': 0.035}
    for name, value in truth.items():
        model.set_initial_guess_parameter(name, value * 0.95)
    sm = scheme.spherical_mean_scheme
    signal = nexi_jelescu2022(sm.bvalues, sm.Delta-sm.delta/3,
                             f=0.6, di=2.4e-9, de=0.9e-9, t_ex=0.035)
    fitted = model.fit(scheme, signal, solver='brute2fine', maxiter=300)
    values = {name: float(np.squeeze(value))
              for name, value in fitted.fitted_parameters.items()}
    for name, value in truth.items():
        assert_allclose(values[name], value, rtol=0.002, atol=0.)
    assert np.sum((model(scheme, **values)-signal)**2) < 1e-10
