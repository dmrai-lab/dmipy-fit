# -*- coding: utf-8 -*-
import warnings
from time import time

import numpy as np
from dipy.utils.optpkg import optional_package

from .fitted_modeling_framework import FittedMultiCompartmentModel
from ..optimizers.brute2fine import (
    GlobalBruteOptimizer, Brute2FineOptimizer)
from ..optimizers.mix import MixOptimizer
from ..optimizers.multi_tissue_volume_fraction import (
    MultiTissueVolumeFractionOptimizer)
from .model_properties import (
    ModelProperties,
    MultiCompartmentModelProperties,
    ReturnFixedValue,
    homogenize_x0_to_data)
from .spherical_mean_framework import MultiCompartmentSphericalMeanModel
from .spherical_harmonics_framework import MultiCompartmentSphericalHarmonicsModel

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

numba, have_numba, _ = optional_package("numba")

__all__ = [
    'ModelProperties',
    'MultiCompartmentModelProperties',
    'MultiCompartmentModel',
    'MultiCompartmentSphericalMeanModel',
    'MultiCompartmentSphericalHarmonicsModel',
    'homogenize_x0_to_data',
    'ReturnFixedValue',
]

class MultiCompartmentModel(MultiCompartmentModelProperties):
    r'''
    The MultiCompartmentModel class allows to combine any number of
    CompartmentModels and DistributedModels into one combined model that can
    be used to fit and simulate dMRI data.

    Parameters
    ----------
    models : list of N CompartmentModel instances,
        the models to combine into the MultiCompartmentModel.
    parameter_links : list of iterables (model, parameter name, link function,
        argument list),
        deprecated, for testing only.
    '''
    _citations = {
        'definition': [
            {'key': 'panagiotaki2012', 'authors': 'Panagiotaki E, Schneider T, Siow B, Hall MG, Lythgoe MF, Alexander DC',
             'title': 'Compartment models of the diffusion MR signal in brain white matter: a taxonomy and comparison',
             'journal': 'NeuroImage',
             'year': 2012, 'doi': '10.1016/j.neuroimage.2012.01.032'}
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'no_exchange',
         'name': 'No inter-compartment exchange',
         'condition_human': 'Assumes no water exchange between compartments during the diffusion time. Compartment signals are computed independently and summed. Invalid when membrane permeability is high (exchange time << diffusion time).',
         'severity': 'warning',
         'source_key': 'panagiotaki2012'},
    ]

    def __init__(self, models, S0_tissue_responses=None, parameter_links=None,
                 eta=False, S0_global=False):
        self.models = models
        self.N_models = len(models)
        if S0_tissue_responses is not None:
            if len(S0_tissue_responses) != self.N_models:
                msg = 'Number of S0_tissue responses {} must be same as ' \
                      'number of input models {}.'
                raise ValueError(
                    msg.format(len(S0_tissue_responses), self.N_models))
            # Store multi-tissue citation for the citation graph
            self._mt_csd_citations = [
                {'key': 'jeurissen2014',
                 'authors': 'Jeurissen B, Tournier J-D, Dhollander T, Connelly A, Sijbers J',
                 'title': 'Multi-tissue constrained spherical deconvolution for improved analysis of multi-shell diffusion MRI data',
                 'journal': 'NeuroImage', 'year': 2014,
                 'doi': '10.1016/j.neuroimage.2014.07.061'}
            ]
            self._mt_csd_constraints = [
                {'id': 'mt_csd_s0_scaling',
                 'name': 'Multi-tissue S0 scaling',
                 'condition_human': 'Per-tissue S0 response values scale the convolution kernels to account for different b=0 signal intensities across tissue types (WM, GM, CSF). Volume fractions are relative to the tissue with the highest S0 response. Requires accurate S0 estimation from representative tissue voxels.',
                 'severity': 'info',
                 'source_key': 'jeurissen2014'}
            ]
        self.S0_tissue_responses = S0_tissue_responses
        # Pre-compute per-compartment S0 scale factors (normalised so max=1).
        if S0_tissue_responses is not None:
            S0_arr = np.array(S0_tissue_responses, dtype=float)
            self.max_S0_response = float(S0_arr.max())
            self.S0_responses = S0_arr / self.max_S0_response
        else:
            self.max_S0_response = 1.0
            self.S0_responses = np.ones(self.N_models, dtype=float)
        self.parameter_links = parameter_links
        if parameter_links is None:
            self.parameter_links = []

        self._prepare_parameters()
        self._prepare_partial_volumes()
        self._prepare_global_parameters(eta=eta, S0_global=S0_global)
        self._prepare_parameter_links()
        self._prepare_model_properties()
        self._check_for_double_model_class_instances()
        self._prepare_parameters_to_optimize()
        self._check_for_NMR_and_other_models()
        self.x0_parameters = {}

        if not have_numba:
            msg = "We highly recommend installing numba for faster function "
            msg += "execution and model fitting."
            print(msg)

    def _prepare_global_parameters(self, eta=False, S0_global=False):
        """Register model-level global parameters into the parameter dicts.

        All are passive by default (_prepare_parameters_to_optimize marks them
        False); call set_initial_guess_parameter(name, value) to activate.

        Parameters
        ----------
        eta : bool
            If True, add eta (dimensionless, range 0-0.5) to the model.
        S0_global : bool
            If True, add S0_global (dimensionless, range 0.001-2.0) to the
            model. Used in T2 optimisation mode so the model can absorb the
            absolute signal scale per voxel. The forward model output is
            multiplied by S0_global after compartment summation and before eta.
        """
        if eta:
            self.parameter_ranges['eta'] = (0.0, 0.5)
            self.parameter_scales['eta'] = 1.0
            self.parameter_types['eta'] = 'normal'
            self.parameter_cardinality['eta'] = 1
            self._parameter_map['eta'] = (None, 'eta')
            self._inverted_parameter_map[(None, 'eta')] = 'eta'
        if S0_global:
            self.parameter_ranges['S0_global'] = (0.001, 2.0)
            self.parameter_scales['S0_global'] = 1.0
            self.parameter_types['S0_global'] = 'normal'
            self.parameter_cardinality['S0_global'] = 1
            self._parameter_map['S0_global'] = (None, 'S0_global')
            self._inverted_parameter_map[(None, 'S0_global')] = 'S0_global'

    def _check_for_NMR_and_other_models(self):
        model_types = [model._model_type for model in self.models]
        if "NMRModel" in model_types:
            if len(np.unique(model_types)) > 1:
                msg = "Cannot combine 1D-NMR and other 3D model types together"
                msg += " into a MultiCompartmentModel."
                raise ValueError(msg)

    def _check_if_sh_coeff_fixed_if_present(self):
        msg = 'sh_coeff parameter {} must be fixed in standard MC models ' \
              'to estimate the kernel parameters.'
        for name, par_type in self.parameter_types.items():
            if par_type == 'sh_coefficients':
                if self.parameter_optimization_flags[name]:
                    raise ValueError(msg.format(name))

    def fit(self, acquisition_scheme, data,
            mask=None, solver='brute2fine', Ns=5, maxiter=300,
            N_sphere_samples=30, use_parallel_processing=False,
            number_of_processors=None, batch_size=None, loss_fn=None,
            sigma_x0=None, sigma_range=(0.001, 0.5)):
        """ The main data fitting function of a MultiCompartmentModel.

        This function can fit it to an N-dimensional dMRI data set, and returns
        a FittedMultiCompartmentModel instance that contains the fitted
        parameters and other useful functions to study the results.

        No initial guess needs to be given to fit a model, but a partial or
        complete initial guess can be given if the user wants to have a
        solution that is a local minimum close to that guess. The
        parameter_initial_guess input can be created using
        parameter_initial_guess_to_parameter_vector().

        A mask can also be given to exclude voxels from fitting (e.g. voxels
        that are outside the brain). If no mask is given then all voxels are
        included.

        An optimization approach can be chosen as either 'brute2fine' or 'mix'.
        - Choosing brute2fine will first use a brute-force optimization to find
          an initial guess for parameters without one, and will then refine the
          result using gradient-descent-based optimization.

          Note that given no initial guess will make brute2fine precompute an
          global parameter grid that will be re-used for all voxels, which in
          many cases is much faster than giving voxel-varying initial condition
          that requires a grid to be estimated per voxel.

        - Choosing mix will use the recent MIX algorithm based on separation of
          linear and non-linear parameters. MIX first uses a stochastic
          algorithm to find the non-linear parameters (non-volume fractions),
          then estimates the volume fractions while fixing the estimates of the
          non-linear parameters, and then finally refines the solution using
          a gradient-descent-based algorithm.

        The fitting process can be parallelized across voxels using stdlib
        concurrent.futures.  Pass use_parallel_processing=True to enable it.
        The algorithm will automatically use all cores in the machine, unless
        otherwise specified in number_of_processors.

        Data with multiple TE are normalized in separate segments using the
        b0-values according that TE.

        Parameters
        ----------
        acquisition_scheme : PGSEAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        data : N-dimensional array of size (N_x, N_y, ..., N_dwis),
            The measured DWI signal attenuation array of either a single voxel
            or an N-dimensional dataset.
        mask : (N-1)-dimensional integer/boolean array of size (N_x, N_y, ...),
            Optional mask of voxels to be included in the optimization.
        solver : string,
            Selection of optimization algorithm.
            - 'brute2fine' to use brute-force optimization.
            - 'mix' to use Microstructure Imaging of Crossing (MIX)
              optimization.
        Ns : integer,
            for brute optimization, decised how many steps are sampled for
            every parameter.
        maxiter : integer,
            for MIX optimization, how many iterations are allowed.
        N_sphere_samples : integer,
            for brute optimization, how many spherical orientations are sampled
            for 'mu'.
        use_parallel_processing : bool,
            whether or not to use parallel processing (default False).
        number_of_processors : integer,
            number of processors to use for parallel processing. Defaults to
            the number of processors in the computer according to cpu_count().
        sigma_x0 : float or None,
            Initial guess for sigma (noise standard deviation in normalised
            signal units = 1/SNR0). When provided, sigma is jointly optimized
            with the diffusion model parameters (fittable sigma mode).
            Only supported with solver='jax'. Defaults to None (fixed sigma).
        sigma_range : tuple (float, float),
            (lower, upper) bounds for sigma during optimization. Only used
            when sigma_x0 is not None. Default (0.001, 0.5) covers SNR 2–1000.

        Returns
        -------
        FittedCompartmentModel: class instance that contains fitted parameters,
            Can be used to recover parameters themselves or other useful
            functions.
        """
        self._check_tissue_model_acquisition_scheme(acquisition_scheme)
        self._check_model_params_with_acquisition_params(acquisition_scheme)
        self._check_acquisition_scheme_has_b0s(acquisition_scheme)
        self._check_if_volume_fractions_are_fixed()
        self._check_if_sh_coeff_fixed_if_present()

        # Fittable sigma is only supported for the JAX solver.
        if sigma_x0 is not None and solver != 'jax':
            raise ValueError(
                "Fittable sigma (sigma_x0 is not None) is only supported "
                "with solver='jax'. Got solver='{}'.".format(solver)
            )

        # JAX solver is multi-threaded; parallel voxel dispatch unsupported.
        if solver == 'jax':
            use_parallel_processing = False

        # estimate S0
        self.scheme = acquisition_scheme
        data_ = np.atleast_2d(data)
        t2_optimization_active = (
            acquisition_scheme.TE is not None and
            any(v for k, v in self.parameter_optimization_flags.items()
                if k.endswith('_T2'))
        )
        if t2_optimization_active:
            # when fitting T2, use no normalization: model returns raw
            # T2-weighted signal in [0, 1] matching the data scale
            S0 = np.ones_like(data_)
        elif self.S0_tissue_responses is not None:
            # Per-compartment S0 is baked into __call__ via self.S0_responses.
            # Normalise data by global max_S0_response (scalar) so that the
            # model's rho_i * vf_i decomposition matches the data scale.
            S0 = self.max_S0_response * np.ones(data_.shape[:-1])
        elif self.scheme.TE is None or len(np.unique(self.scheme.TE)) == 1:
            S0 = np.mean(data_[..., self.scheme.b0_mask], axis=-1)
        else:  # if multiple TE are in the data
            S0 = np.ones_like(data_)
            for TE_ in self.scheme.shell_TE:
                TE_mask = self.scheme.TE == TE_
                TE_b0_mask = np.all([self.scheme.b0_mask, TE_mask], axis=0)
                S0[..., TE_mask] = np.mean(
                    data_[..., TE_b0_mask], axis=-1)[..., None]

        if mask is None:
            mask = data_[..., 0] > 0
        else:
            mask = np.all([mask, data_[..., 0] > 0], axis=0)
        mask_pos = np.where(mask)

        N_parameters = len(self.bounds_for_optimization)
        N_voxels = np.sum(mask)

        # make starting parameters and data the same size
        x0_ = self.parameter_initial_guess_to_parameter_vector(
            **self.x0_parameters)
        x0_ = homogenize_x0_to_data(
            data_, x0_)
        x0_bool = np.all(
            np.isnan(x0_), axis=tuple(np.arange(x0_.ndim - 1)))
        x0_[..., ~x0_bool] /= self.scales_for_optimization[~x0_bool]

        if use_parallel_processing:
            if number_of_processors is None:
                number_of_processors = cpu_count()
            print('Using parallel processing with {} workers.'.format(
                number_of_processors))
        fitted_parameters_lin = np.empty(
            np.r_[N_voxels, N_parameters], dtype=float)

        start = time()
        if solver == 'brute2fine':
            global_brute = GlobalBruteOptimizer(
                self, self.scheme, x0_, Ns, N_sphere_samples)
            fit_func = Brute2FineOptimizer(self, self.scheme, Ns)
            print('Setup brute2fine optimizer in {} seconds'.format(
                time() - start))
        elif solver == 'mix':
            self._check_for_tortuosity_constraint()
            fit_func = MixOptimizer(self, self.scheme, maxiter)
            print('Setup MIX optimizer in {} seconds'.format(
                time() - start))
        elif solver == 'jax':
            import jax
            try:
                _jax_backend = jax.default_backend()
            except Exception:
                # A CUDA plugin present-but-uninitialisable (no device / OOM /
                # driver mismatch) makes default_backend() RAISE, not return; treat
                # as CPU so the warn+run-on-CPU path below still executes.
                _jax_backend = 'cpu'
            if _jax_backend != 'gpu':
                warnings.warn(
                    "solver='jax' requested but no usable CUDA GPU is available to "
                    "JAX (backend='{}'); running JAX on CPU, which is much slower. "
                    "Pass solver='brute2fine' for the native CPU optimiser to "
                    "silence this.".format(_jax_backend),
                    RuntimeWarning, stacklevel=2)
            from ..jax.optimizers_jax import JaxOptimizer
            from ..jax.multicompartment_jax import build_mc_forward_fn
            fit_func = JaxOptimizer(
                self, self.scheme, maxiter=maxiter,
                Ns=Ns, N_sphere_samples=N_sphere_samples,
                loss_fn=loss_fn,
                fit_sigma=(sigma_x0 is not None),
                sigma_x0=sigma_x0,
                sigma_range=sigma_range)
            print('Setup JAX optimizer in {} seconds'.format(
                time() - start))
        else:
            msg = "Unknown solver name {}".format(solver)
            raise ValueError(msg)
        self.optimizer = fit_func

        start = time()
        if solver == 'jax':
            # --- JAX path: vectorize over all voxels ---
            import jax
            import jax.numpy as jnp
            from ..jax.vmap_fit import vmap_fit
            from ..jax.brute_jax import build_jax_brute_fn
            try:
                from jaxlib.xla_extension import XlaRuntimeError
            except Exception:
                XlaRuntimeError = RuntimeError
            data_masked = data_[mask_pos]                    # (N_voxels, N_meas)
            S0_masked = S0[mask_pos]                         # (N_voxels,) or (N_voxels, N_meas)
            # Normalise data by S0.  S0 can be per-voxel scalar (single TE)
            # or per-measurement (multi-TE / T2 mode / S0_tissue_responses).
            if S0_masked.ndim == 1:
                # Per-voxel scalar S0
                data_norm = (data_masked / S0_masked[:, None]).astype(np.float32)
            else:
                # Per-measurement S0 (multi-TE or T2-mode ones array)
                data_norm = (data_masked / S0_masked).astype(np.float32)

            def _run_jax_fit():
                # JAX brute-force grid search: evaluates all grid points on GPU,
                # then picks best starting point per voxel via one MSE matrix op.
                forward_fn = build_mc_forward_fn(self, self.scheme)
                print('Building JAX brute grid ({} sphere samples, Ns={})...'.format(
                    N_sphere_samples, Ns))
                brute_fn = build_jax_brute_fn(
                    self, self.scheme, forward_fn, fit_func,
                    Ns=Ns, N_sphere_samples=N_sphere_samples)
                x0_nested_all = np.array(
                    brute_fn(jnp.array(data_norm, dtype=jnp.float32)))

                # When fitting sigma, the brute grid covers model params only.
                # Append the (fixed) sigma initial guess as the last column so
                # every voxel starts from sigma_x0 rather than a grid midpoint.
                if fit_func._fit_sigma:
                    sigma_col = np.full(
                        (x0_nested_all.shape[0], 1),
                        fit_func._sigma_x0_norm,
                        dtype=x0_nested_all.dtype)
                    x0_nested_all = np.concatenate(
                        [x0_nested_all, sigma_col], axis=1)

                fitted_nested = vmap_fit(
                    fit_func, data_norm, x0_nested_all,
                    batch_size=batch_size)                   # (N_voxels, N_nested)

                return np.array([
                    fit_func._unnest(fitted_nested[i])
                    for i in range(N_voxels)
                ])                                           # (N_voxels, N_params[+1])

            # Run on the default backend; on any GPU runtime failure (broken CUDA/
            # cuSolver init, or an OOM that survives batch back-off) fall back to CPU
            # with a loud warning rather than dying with a raw XLA traceback.
            try:
                fitted_parameters_lin = _run_jax_fit()
            except (XlaRuntimeError, RuntimeError) as exc:
                if jax.default_backend() == 'cpu':
                    raise
                warnings.warn(
                    "solver='jax' failed on the GPU backend ({}: {}). Retrying on "
                    "CPU -- this is much slower; pass solver='brute2fine' for the "
                    "native CPU optimiser, or free GPU memory / lower batch_size."
                    .format(type(exc).__name__,
                            str(exc).splitlines()[0][:200] if str(exc) else ''),
                    RuntimeWarning, stacklevel=2)
                with jax.default_device(jax.devices('cpu')[0]):
                    fitted_parameters_lin = _run_jax_fit()
        else:
            # --- existing per-voxel loop (brute2fine / mix) ---
            all_args = []
            for idx, pos in enumerate(zip(*mask_pos)):
                voxel_E = data_[pos] / S0[pos]
                voxel_x0_vector = x0_[pos]
                if solver == 'brute2fine':
                    if global_brute.global_optimization_grid is True:
                        voxel_x0_vector = global_brute(voxel_E)
                all_args.append((voxel_E, voxel_x0_vector))

            if use_parallel_processing:
                with ProcessPoolExecutor(number_of_processors) as ex:
                    futures = [ex.submit(fit_func, *args) for args in all_args]
                    for idx, fut in enumerate(futures):
                        fitted_parameters_lin[idx] = fut.result()
            else:
                for idx, args in enumerate(all_args):
                    fitted_parameters_lin[idx] = fit_func(*args)

        fitting_time = time() - start
        print('Fitting of {} voxels complete in {} seconds.'.format(
            len(fitted_parameters_lin), fitting_time))
        print('Average of {} seconds per voxel.'.format(
            fitting_time / N_voxels))

        # When fit_sigma=True, _unnest() appends sigma (in SI units) as the
        # last column of fitted_parameters_lin.  Separate it before applying
        # scales (sigma is already in SI units; scales must not be applied).
        fitted_sigma = None
        if solver == 'jax' and sigma_x0 is not None:
            sigma_lin = fitted_parameters_lin[:, -1]        # (N_voxels,) SI
            fitted_parameters_lin = fitted_parameters_lin[:, :-1]

            # Scatter sigma into a spatial array matching x0_ spatial shape
            fitted_sigma = np.zeros(x0_.shape[:-1], dtype=float)
            fitted_sigma[mask_pos] = sigma_lin

        fitted_parameters = np.zeros_like(x0_, dtype=float)
        fitted_parameters[mask_pos] = (
            fitted_parameters_lin * self.scales_for_optimization)

        # set passive T2 parameters (not being optimized) to NaN
        passive_t2_indices = []
        flat_idx = 0
        for param_name, card in self.parameter_cardinality.items():
            if (param_name.endswith('_T2') and
                    not self.parameter_optimization_flags[param_name]):
                passive_t2_indices.append(flat_idx)
            flat_idx += card
        if passive_t2_indices:
            fitted_parameters[..., passive_t2_indices] = np.nan

        # Store optimizer citations on the model for the citation graph walker
        if hasattr(fit_func, '_citations'):
            self._optimizer_citations = fit_func._citations.get('definition', [])
        self._fit_solver = solver
        self._was_fitted = True

        # Secondary geometric-volume-fraction step (S0-weighted NNLS), shared
        # with the spherical-mean framework. Kept separate from the non-linear
        # fit so the (arbitrary, scanner-dependent) S0 scaling does not perturb
        # the non-linear solver's loss tolerances; recovered here by a convex
        # solve where the optimum is exact regardless of scale.
        fitted_mt_fractions = None
        if self.S0_tissue_responses is not None:
            data_masked = data_[mask_pos]                # (N_voxels, N_meas) raw
            params_si = fitted_parameters[mask_pos]      # SI-unit vectors
            try:
                from ..jax.fractions_jax import (
                    fit_multi_tissue_fractions_jax, supports_jax_fraction_fit)
                use_jax = supports_jax_fraction_fit(self, spherical_mean=False)
            except ImportError:
                use_jax = False
            if use_jax:
                mt = fit_multi_tissue_fractions_jax(
                    self, self.scheme, params_si, data_masked,
                    self.S0_tissue_responses, spherical_mean=False)
            else:
                fit_func = MultiTissueVolumeFractionOptimizer(
                    self.scheme, self, self.S0_tissue_responses)
                mt = np.empty((len(params_si), self.N_models), dtype=float)
                for i in range(len(params_si)):
                    mt[i] = fit_func(data_masked[i], params_si[i])
            fitted_mt_fractions = np.zeros(np.r_[mask.shape, self.N_models])
            fitted_mt_fractions[mask_pos] = mt

        return FittedMultiCompartmentModel(self, S0, mask, fitted_parameters,
                                           fitted_mt_fractions,
                                           fitted_sigma=fitted_sigma)

    def simulate_signal(self, acquisition_scheme, parameters_array_or_dict):
        """
        Function to simulate diffusion data for a given acquisition_scheme
        and model parameters for the MultiCompartmentModel.

        Parameters
        ----------
        acquisition_scheme : PGSEAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy
        model_parameters_array : 1D array of size (N_parameters) or
            N-dimensional array the same size as the data.
            The model parameters of the MultiCompartmentModel model.

        Returns
        -------
        E_simulated: 1D array of size (N_parameters) or N-dimensional
            array the same size as x0.
            The simulated signal of the microstructure model.
        """
        self._check_model_params_with_acquisition_params(acquisition_scheme)

        Ndata = acquisition_scheme.number_of_measurements
        if isinstance(parameters_array_or_dict, np.ndarray):
            x0 = parameters_array_or_dict
        elif isinstance(parameters_array_or_dict, dict):
            x0 = self.parameters_to_parameter_vector(
                **parameters_array_or_dict)

        x0_at_least_2d = np.atleast_2d(x0)
        if x0_at_least_2d.shape[-1] == 0:
            # Zero-parameter model (e.g. a lone S1Dot): the signal is fully
            # determined, so there is a single deterministic measurement vector
            # and no voxel/parameter axis to loop over.
            return np.asarray(self(acquisition_scheme))
        x0_2d = x0_at_least_2d.reshape(-1, x0_at_least_2d.shape[-1])
        E_2d = np.empty(np.r_[x0_2d.shape[:-1], Ndata])
        for i, x0_ in enumerate(x0_2d):
            parameters = self.parameter_vector_to_parameters(x0_)
            E_2d[i] = self(acquisition_scheme, **parameters)
        E_simulated = E_2d.reshape(
            np.r_[x0_at_least_2d.shape[:-1], Ndata])

        if x0.ndim == 1:
            return np.squeeze(E_simulated)
        else:
            return E_simulated

    def __call__(self, acquisition_scheme_or_vertices,
                 quantity="signal", **kwargs):
        """
        The MultiCompartmentModel function call for to generate signal
        attenuation for a given acquisition scheme and model parameters.

        First, the linked parameters are added to the optimized parameters.

        Then, every model in the MultiCompartmentModel is called with the right
        parameters to recover the part of the signal attenuation of that model.
        The resulting values are multiplied with the volume fractions and
        finally the combined signal attenuation is returned.

        Aside from the signal, the function call can also return the Fiber
        Orientation Distributions (FODs) when a dispersed model is used, and
        can also return the stochastic cost function for the MIX algorithm.

        Parameters
        ----------
        acquisition_scheme : PGSEAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        quantity : string
            can be 'signal', 'FOD' or 'stochastic cost function' depending on
            the need of the model.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.
        """
        if quantity == "signal" or quantity == "FOD":
            values = 0
        elif quantity == "stochastic cost function":
            values = np.empty((
                acquisition_scheme_or_vertices.number_of_measurements,
                len(self.models)
            ))
            counter = 0

        kwargs = self.add_linked_parameters_to_parameters(
            kwargs
        )
        if len(self.models) > 1:
            partial_volumes = [
                kwargs[p] for p in self.partial_volume_names
            ]
        else:
            partial_volumes = [1.]

        for model_idx, (model_name, model, partial_volume) in enumerate(zip(
                self.model_names, self.models, partial_volumes
        )):
            parameters = {}
            for parameter in model.parameter_ranges:
                parameter_name = self._inverted_parameter_map[
                    (model, parameter)
                ]
                parameters[parameter] = kwargs.get(
                    # , self.parameter_defaults.get(parameter_name)
                    parameter_name
                )

            if quantity == "signal":
                rho = self.S0_responses[model_idx]
                # partial_volume may arrive as a Python list (e.g. [0.5]); coerce
                # so `rho * partial_volume` (np.float64 * list) doesn't TypeError.
                values = (values + rho * np.asarray(partial_volume, dtype=float)
                          * model(acquisition_scheme_or_vertices, **parameters))
            elif quantity == "FOD":
                try:
                    values = (values + partial_volume * model.fod(
                        acquisition_scheme_or_vertices, **parameters))
                except AttributeError:
                    continue
            elif quantity == "stochastic cost function":
                values[:, counter] = model(acquisition_scheme_or_vertices,
                                           **parameters)
                counter += 1

        if quantity == "signal":
            # T1 relaxation is now an occupancy-gated LongitudinalRelaxation factor
            # on the compartment (signal_models.attenuation), gated by the
            # longitudinal occupancy tau_par; no tissue-level T1 special-casing here.

            # Global S0 multiplier: absorbs absolute per-voxel signal scale.
            S0_global = kwargs.get('S0_global')
            if (S0_global is not None
                    and not np.isnan(np.asarray(S0_global, dtype=float).flat[0])):
                values = values * S0_global

            # Rician noise floor: sqrt(S^2 + eta^2), applied last.
            eta = kwargs.get('eta')
            if (eta is not None
                    and not np.isnan(np.asarray(eta, dtype=float).flat[0])):
                values = np.sqrt(values ** 2 + eta ** 2)

        return values

