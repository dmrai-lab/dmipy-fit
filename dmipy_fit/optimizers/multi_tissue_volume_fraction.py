import numpy as np
from scipy.optimize import fmin_cobyla


class MultiTissueVolumeFractionOptimizer:
    """
    Secondary optimizer for including S0 tissue response values into the volume
    fraction estimation.

    Following the suggestion by [1]_, when including S0 responses, the
    volume fractions are no longer unity constrained. This means that the
    optimization of linear volume fractions and non-linear parameters is
    independent, and thus this secondary optimization is just a simple convex
    optimization on the volume fractions only.

    Parameters
    ----------
    model: dmipy multi-compartment model instance,
        dmipy initialized mc model.
    S0_tissue_responses: list,
        constains the positive S0 tissue responses that are associated with the
        tissue that each compartment model in the mc-model represents.

    References
    ----------
    .. [1] Dell'Acqua, Flavio, and J-Donald Tournier. "Modelling white matter
           with spherical deconvolution: How and why?." NMR in Biomedicine 32.4
           (2019): e3945.
    """
    _citations = {
        'definition': [
            {'key': 'dellacqua2019', 'authors': "Dell'Acqua F, Tournier J-D",
             'title': 'Modelling white matter with spherical deconvolution: How and why?',
             'journal': 'NMR in Biomedicine',
             'year': 2019, 'doi': '10.1002/nbm.3945'},
        ],
        'default_parameters': {},
    }
    _validity_constraints = []

    def __init__(self, acquisition_scheme, model, S0_tissue_responses):
        self.acquisition_scheme = acquisition_scheme
        self.model = model
        self.S0_tissue_responses = S0_tissue_responses

    def cobyla_cost_function(self, fractions, phi, data):
        "Objective function of linear parameter estimation using COBYLA."
        E_hat = np.dot(phi, fractions)
        diff = data - E_hat
        objective = np.dot(diff, diff)
        return objective * 1e5

    def __call__(self, data, params_si):
        """``params_si``: fitted parameter vector in SI (physical) units -- the
        same convention as :func:`dmipy_fit.jax.fractions_jax.\
fit_multi_tissue_fractions_jax`, so the two solvers are interchangeable."""
        params_dict = self.model.parameter_vector_to_parameters(params_si)
        phi = self.model(self.acquisition_scheme,
                         quantity="stochastic cost function", **params_dict)
        phi *= self.S0_tissue_responses

        if self.model.N_models == 1:
            vf_x0 = [1.]
        else:
            # volume-fraction parameters have unit scale, so their SI values are
            # already in [0, 1] and serve directly as the NNLS initial guess.
            vf_x0 = params_si[-self.model.N_models:]

        vf = fmin_cobyla(self.cobyla_cost_function, x0=vf_x0,
                         cons=[cobyla_positivity_constraint],
                         args=(phi, data),
                         maxfun=2000)
        return vf


def cobyla_positivity_constraint(volume_fractions, *args):
    "COBYLA positivity constraint on volume fractions"
    return volume_fractions - 0.001
