from numpy.testing import assert_almost_equal, assert_

from dmipy_fit.core import modeling_framework
from dmipy_fit.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme)
from dmipy_fit.signal_models import cylinder_models, gaussian_models

scheme = wu_minn_hcp_acquisition_scheme()


def test_multi_tissue_tortuosity():
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    ball = gaussian_models.G1Ball()

    s0s = 3000.
    s0z = 4000.
    s0b = 10000.

    model = modeling_framework.MultiCompartmentModel(
        models=[stick, zeppelin, ball],
        S0_tissue_responses=[s0s, s0z, s0b])
    model.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
                                 'C1Stick_1_lambda_par',
                                 'partial_volume_0',
                                 'partial_volume_1',
                                 True)
    tort = model.parameter_links[0][2]
    s0ic, s0ec = tort.S0_intra, tort.S0_extra
    assert_(s0ic == s0s and s0ec == s0z)


def test_multi_tissue_tortuosity_no_s0():
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    ball = gaussian_models.G1Ball()

    model = modeling_framework.MultiCompartmentModel(
        models=[stick, zeppelin, ball])
    model.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
                                 'C1Stick_1_lambda_par',
                                 'partial_volume_0',
                                 'partial_volume_1',
                                 True)
    tort = model.parameter_links[0][2]
    s0ic, s0ec = tort.S0_intra, tort.S0_extra
    assert_(s0ic == 1 and s0ec == 1)


def test_multi_tissue_tortuosity_no_correction():
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    ball = gaussian_models.G1Ball()

    model = modeling_framework.MultiCompartmentModel(
        models=[stick, zeppelin, ball],
        S0_tissue_responses=[4000, 5000, 10000]
    )
    model.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
                                 'C1Stick_1_lambda_par',
                                 'partial_volume_0',
                                 'partial_volume_1',
                                 False)
    tort = model.parameter_links[0][2]
    s0ic, s0ec = tort.S0_intra, tort.S0_extra
    assert_(s0ic == 1 and s0ec == 1)


def test_multi_tissue_tortuosity_no_s0_no_correction():
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    ball = gaussian_models.G1Ball()

    model = modeling_framework.MultiCompartmentModel(
        models=[stick, zeppelin, ball]
    )
    model.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
                                 'C1Stick_1_lambda_par',
                                 'partial_volume_0',
                                 'partial_volume_1',
                                 False)
    tort = model.parameter_links[0][2]
    s0ic, s0ec = tort.S0_intra, tort.S0_extra
    assert_(s0ic == 1 and s0ec == 1)
