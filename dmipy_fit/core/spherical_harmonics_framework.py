# -*- coding: utf-8 -*-
from time import time

import numpy as np
from dipy.utils.optpkg import optional_package

from .fitted_modeling_framework import (
    FittedMultiCompartmentModel,
    FittedMultiCompartmentSphericalHarmonicsModel)
from ..optimizers_fod.csd_cvxpy import CsdCvxpyOptimizer, have_cvxpy
from ..optimizers_fod.csd_tournier import CsdTournierOptimizer
from ..optimizers_fod.csd_plus import CsdPlusOptimizer
from .model_properties import MultiCompartmentModelProperties, homogenize_x0_to_data

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

numba, have_numba, _ = optional_package("numba")

__all__ = ['MultiCompartmentSphericalHarmonicsModel']

class MultiCompartmentSphericalHarmonicsModel(MultiCompartmentModelProperties):
    r'''
    The MultiCompartmentModel class allows to combine any number of
    CompartmentModels and DistributedModels into one combined model that can
    be used to fit and simulate dMRI data.

    Parameters
    ----------
    models : list of N CompartmentModel instances,
        the models to combine into the MultiCompartmentModel.
    '''
    _citations = {
        'definition': [
            {'key': 'jeurissen2014', 'authors': 'Jeurissen B, Tournier J-D, Dhollander T, Connelly A, Sijbers J',
             'title': 'Multi-tissue constrained spherical deconvolution for improved analysis of multi-shell diffusion MRI data',
             'journal': 'NeuroImage',
             'year': 2014, 'doi': '10.1016/j.neuroimage.2014.07.061'}
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'SH_convergence', 'name': 'SH convergence',
         'condition_human': 'max_order must be sufficient for the kernel bandwidth',
         'severity': 'info',
         'source_key': 'jeurissen2014'},
        {'id': 'no_exchange',
         'name': 'No inter-compartment exchange',
         'condition_human': 'Assumes no water exchange between compartments during the diffusion time. Compartment signals are computed independently and summed. Invalid when membrane permeability is high (exchange time << diffusion time).',
         'severity': 'warning',
         'source_key': 'jeurissen2014'},
    ]

    def __init__(self, models, S0_tissue_responses=None, sh_order=8):
        self.models = models
        self.N_models = len(models)
        if S0_tissue_responses is not None:
            self.fit_S0_response = True
            # Store MT-CSD citation for the citation graph
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
            if len(S0_tissue_responses) != self.N_models:
                msg = 'Number of S0_tissue responses {} must be same as ' \
                      'number of input models {}.'
                raise ValueError(
                    msg.format(len(S0_tissue_responses), self.N_models))
        else:
            self.fit_S0_response = False
        self.S0_tissue_responses = S0_tissue_responses
        self.parameter_links = []

        self._check_for_NMR_models()
        self._prepare_parameters()
        self._delete_orientation_parameters()
        self._prepare_partial_volumes()
        self._prepare_parameter_links()
        self._prepare_model_properties()
        self._check_for_double_model_class_instances()
        self._prepare_parameters_to_optimize()
        self._add_spherical_harmonics_parameters(sh_order)
        self._check_that_one_anisotropic_kernel_is_present()
        # self._check_for_tissue_response_models()

        self.x0_parameters = {}
        self.sh_order = sh_order

        if not have_numba:
            msg = "We highly recommend installing numba for faster function "
            msg += "execution and model fitting."
            print(msg)

    def _check_for_NMR_models(self):
        for model in self.models:
            if model._model_type == 'NMRModel':
                msg = "Cannot estimate spherical mean of 1D-NMR models."
                raise ValueError(msg)

    def _delete_orientation_parameters(self):
        """
        Deletes orientation parameters from input models 'mu' since they're not
        needed in spherical mean models.
        """
        "Removes orientation parameters from input models."
        for model in self.models:
            for param_name, param_type in model.parameter_types.items():
                if param_type == 'orientation':
                    appended_param_name = self._inverted_parameter_map[
                        model, param_name]
                    del self.parameter_ranges[appended_param_name]
                    del self.parameter_scales[appended_param_name]
                    del self.parameter_cardinality[appended_param_name]
                    del self.parameter_types[appended_param_name]

    def _add_spherical_harmonics_parameters(self, sh_order):
        N_coef = int((sh_order + 2) * (sh_order + 1) // 2)
        self.parameter_ranges['sh_coeff'] = [
            [-1e3, 1e3] for i in range(N_coef)]
        self.parameter_scales['sh_coeff'] = np.ones(N_coef, dtype=float)
        self.parameter_cardinality['sh_coeff'] = N_coef
        self.parameter_types['sh_coeff'] = 'sh_coefficients'
        self.parameter_optimization_flags['sh_coeff'] = True

    def _check_if_kernel_parameters_are_fixed(self):
        "checks if only volume fraction and sh_coeff parameters are optimized."
        self.volume_fractions_fixed = True
        for name, flag in self.parameter_optimization_flags.items():
            if flag is True:
                if (not name == 'sh_coeff' and
                        not name.startswith('partial_volume') and
                        not name.endswith('_T2')):
                    msg = 'Kernel parameter {} is not fixed.'.format(name)
                    raise ValueError(msg)
                if name.startswith('partial_volume'):
                    self.volume_fractions_fixed = False
        if (not self.volume_fractions_fixed and
                self.multiple_anisotropic_kernels):
            msg = 'Cannot have multiple anisotropic kernels without having '
            msg += 'all volume fractions fixed.'
            raise ValueError(msg)

    def _check_that_one_anisotropic_kernel_is_present(self):
        "checks if one anisotropic kernel is given."
        orientation_counter = 0
        self.multiple_anisotropic_kernels = False
        for model in self.models:
            if 'orientation' in model.parameter_types.values():
                orientation_counter += 1
        if orientation_counter == 0:
            msg = 'MultiCompartmentSphericalHarmonicsModel must at least have '
            msg += 'one anisotropic kernel input model.'
            raise ValueError(msg)
        if orientation_counter > 1:
            self.multiple_anisotropic_kernels = True

    def fit(self, acquisition_scheme, data, mask=None, solver='csd',
            lambda_lb=1e-5, unity_constraint='kernel_dependent',
            use_parallel_processing=False,
            number_of_processors=None, verbose=True, eta=None):
        """ The main data fitting function of a
        MultiCompartmentSphericalHarmonicsModel.

        This function can fit it to an N-dimensional dMRI data set, and returns
        a FittedMultiCompartmentModel instance that contains the fitted
        parameters and other useful functions to study the results.

        A mask can also be given to exclude voxels from fitting (e.g. voxels
        that are outside the brain). If no mask is given then all voxels are
        included.

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
            Can be 'csd', 'csd_tounier07', 'csd_cvxpy' or 'csd_plus', with the
            default being 'csd'. Using 'csd' will make the algorithm
            automatically use the 'tournier07' solver [1]_ if there are no
            volume fractionsto fit or they are fixed. Otherwise, the slower but
            more general cvxpy solver [2]_ is used, which follows the
            formulation of [3]_. Using 'csd_plus' will make the algorithm use
            the global positivity constraints of [5]_.
        lambda_lb: positive float,
            Weight for Laplace-Beltrami regularization to impose smoothness
            into estimated FODs, follows [4]_.
        unity_constraint: String or bool,
            Whether or not to constrain the volume fractions of the FOD to
            unity. The default is set to 'kernel_dependent', meaning it will
            enforce unity if the kernel is voxel-varying or when volume
            fractions are estimated. Otherwise unity_constraint is set to
            False.
        use_parallel_processing : bool,
            Whether or not to use parallel processing (default False).
        number_of_processors : integer,
            Number of processors to use for parallel processing. Defaults to
            the number of processors in the computer according to cpu_count().
        eta : float or None,
            Rician noise floor estimate (in normalised signal units = 1/SNR0).
            When provided and solver='csd_jax', a pre-processing bias
            correction is applied before the QP solve:
            ``data_corrected = sqrt(max(data^2 - eta^2, 0))``.
            Ignored for other solvers.

        Returns
        -------
        FittedCompartmentModel: class instance that contains fitted parameters,
            Can be used to recover parameters themselves or other useful
            functions.

        References
        ----------
        .. [1] Tournier, J-Donald, Fernando Calamante, and Alan Connelly.
            "Robust determination of the fibre orientation distribution in
            diffusion MRI: non-negativity constrained super-resolved spherical
            deconvolution." Neuroimage 35.4 (2007): 1459-1472.
        .. [2] Diamond, Steven, and Stephen Boyd. "CVXPY: A Python-embedded
            modeling language for convex optimization." The Journal of Machine
            Learning Research 17.1 (2016): 2909-2913.
        .. [3] Jeurissen, Ben, et al. "Multi-tissue constrained spherical
            deconvolution for improved analysis of multi-shell diffusion MRI
            data." NeuroImage 103 (2014): 411-426.
        .. [4] Descoteaux, Maxime, et al. "Regularized, fast, and robust
            analytical Q-ball imaging." Magnetic Resonance in Medicine: An
            Official Journal of the International Society for Magnetic
            Resonance in Medicine 58.3 (2007): 497-510.
        .. [5] Dela Haije, Tom, Evren Özarslan, and Aasa Feragen. "Enforcing
            necessary non-negativity constraints for common diffusion MRI models
            using sum of squares programming." NeuroImage 209 (2020): 116405.
        """
        self._check_if_kernel_parameters_are_fixed()
        self._check_tissue_model_acquisition_scheme(acquisition_scheme)
        self._check_acquisition_scheme_has_b0s(acquisition_scheme)
        self._check_model_params_with_acquisition_params(acquisition_scheme)

        self.voxel_varying_kernel = False
        if bool(self.x0_parameters):  # if the dictionary is not empty
            self.voxel_varying_kernel = True

        if unity_constraint == 'kernel_dependent':
            self.unity_constraint = False
            if self.fit_S0_response:
                self.unity_constraint = False
            elif not self.volume_fractions_fixed or self.voxel_varying_kernel:
                self.unity_constraint = True
        else:
            self.unity_constraint = unity_constraint

        if self.fit_S0_response:
            S0_responses = np.array(self.S0_tissue_responses)
            self.max_S0_response = S0_responses.max()
            self.S0_responses = S0_responses / self.max_S0_response
        else:
            self.S0_responses = np.ones(len(self.models), dtype=float)
            self.max_S0_response = 1.

        # estimate S0
        self.scheme = acquisition_scheme
        data_ = np.atleast_2d(data)
        if self.scheme.TE is None or len(np.unique(self.scheme.TE)) == 1:
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

        start = time()
        if solver == 'csd':
            if self.volume_fractions_fixed:
                fit_func = CsdTournierOptimizer(
                    acquisition_scheme, self, x0_, self.sh_order,
                    unity_constraint=self.unity_constraint,
                    lambda_lb=lambda_lb)
                if use_parallel_processing:
                    msg = 'Parallel processing turned off for tournier07'
                    msg += ' optimizer because it does not improve fitting '
                    msg += 'speed.'
                    if verbose:
                        print(msg)
                    use_parallel_processing = False
                if verbose:
                    print(
                        'Setup Tournier07 FOD optimizer in {} seconds'.format(
                            time() - start))
            else:
                if not have_cvxpy:
                    raise ImportError(
                        "cvxpy is required for solver='csd' with non-fixed "
                        "volume fractions. Install it with: "
                        "pip install dmipy[legacy]  or switch to "
                        "solver='csd_jax' (requires jaxopt)."
                    )
                fit_func = CsdCvxpyOptimizer(
                    acquisition_scheme, self, x0_, self.sh_order,
                    unity_constraint=self.unity_constraint,
                    lambda_lb=lambda_lb)
                if verbose:
                    print('Setup CVXPY FOD optimizer in {} seconds'.format(
                        time() - start))
        elif solver == 'csd_tournier07':
            fit_func = CsdTournierOptimizer(
                acquisition_scheme, self, x0_, self.sh_order,
                unity_constraint=self.unity_constraint, lambda_lb=lambda_lb)
            if use_parallel_processing:
                msg = 'Parallel processing turned off for tournier07 optimizer'
                msg += ' because it does not improve fitting speed.'
                if verbose:
                    print(msg)
                use_parallel_processing = False
            if verbose:
                print('Setup Tournier07 FOD optimizer in {} seconds'.format(
                    time() - start))
        elif solver == 'csd_cvxpy':
            if not have_cvxpy:
                raise ImportError(
                    "cvxpy is required for solver='csd_cvxpy'. "
                    "Install it with: pip install dmipy[legacy]  or switch "
                    "to solver='csd_jax' (requires jaxopt)."
                )
            fit_func = CsdCvxpyOptimizer(
                acquisition_scheme, self, x0_, self.sh_order,
                unity_constraint=self.unity_constraint, lambda_lb=lambda_lb)
            if verbose:
                print('Setup CVXPY FOD optimizer in {} seconds'.format(
                    time() - start))
        elif solver == 'csd_plus':
            fit_func = CsdPlusOptimizer(
                acquisition_scheme, self, x0_, self.sh_order,
                unity_constraint=self.unity_constraint)
            if verbose:
                print('Setup CSD-PLUS FOD optimizer in {} seconds'.format(
                    time() - start))
        elif solver == 'csd_jax':
            from ..jax.csd_jax import CsdOsqpOptimizer
            fit_func = CsdOsqpOptimizer(
                acquisition_scheme, self, x0_, self.sh_order,
                unity_constraint=self.unity_constraint, lambda_lb=lambda_lb)
            if verbose:
                print('Setup JAX/OSQP CSD optimizer in {} seconds'.format(
                    time() - start))
        else:
            msg = "Unknown solver name {}".format(solver)
            raise ValueError(msg)

        self.optimizer = fit_func

        # --- JAX batch path: solve all voxels in one vmapped kernel ---------
        if solver == 'csd_jax':
            start = time()
            data_masked = np.zeros((N_voxels,
                                    acquisition_scheme.number_of_measurements),
                                   dtype=float)
            x0_masked = np.zeros((N_voxels, N_parameters), dtype=float)
            for idx, pos in enumerate(zip(*mask_pos)):
                if self.fit_S0_response:
                    data_masked[idx] = data_[pos] / self.max_S0_response
                else:
                    data_masked[idx] = data_[pos] / S0[pos]
                x0_masked[idx] = x0_[pos]

            fitted_parameters_lin = fit_func.fit_batch(
                data_masked, x0_masked, eta=eta)
            fitting_time = time() - start
            if verbose:
                print('JAX/OSQP fitting of {} voxels complete in {} seconds.'.format(
                    N_voxels, fitting_time))
                print('Average of {} seconds per voxel.'.format(
                    fitting_time / N_voxels))

        # --- standard per-voxel loop (all other solvers) --------------------
        else:
            if use_parallel_processing:
                if number_of_processors is None:
                    number_of_processors = cpu_count()
                if verbose:
                    print('Using parallel processing with {} workers.'.format(
                        number_of_processors))
            fitted_parameters_lin = np.empty(
                np.r_[N_voxels, N_parameters], dtype=float)

            start = time()
            all_args = []
            for idx, pos in enumerate(zip(*mask_pos)):
                if self.fit_S0_response:
                    data_to_fit = data_[pos] / self.max_S0_response
                else:
                    data_to_fit = data_[pos] / S0[pos]
                voxel_x0_vector = x0_[pos]
                all_args.append((data_to_fit, voxel_x0_vector))

            if use_parallel_processing:
                with ProcessPoolExecutor(number_of_processors) as ex:
                    futures = [ex.submit(fit_func, *args) for args in all_args]
                    for idx, fut in enumerate(futures):
                        fitted_parameters_lin[idx] = fut.result()
            else:
                for idx, args in enumerate(all_args):
                    fitted_parameters_lin[idx] = fit_func(*args)

        fitted_parameters = np.zeros_like(x0_, dtype=float)
        fitted_parameters[mask_pos] = fitted_parameters_lin

        return FittedMultiCompartmentSphericalHarmonicsModel(
            self, S0, mask, fitted_parameters)

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
        self.volume_fractions_fixed = True
        if self.S0_tissue_responses is None:
            self.S0_responses = np.ones(self.N_models)

        Ndata = acquisition_scheme.number_of_measurements
        if isinstance(parameters_array_or_dict, np.ndarray):
            x0 = parameters_array_or_dict
        elif isinstance(parameters_array_or_dict, dict):
            x0 = self.parameters_to_parameter_vector(
                **parameters_array_or_dict)

        x0_at_least_2d = np.atleast_2d(x0)
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

    def __call__(self, acquisition_scheme, **kwargs):
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
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.
        """
        A = self._construct_convolution_kernel(
            acquisition_scheme=acquisition_scheme, **kwargs)
        self.S0_responses = kwargs.get('S0_responses', self.S0_responses)

        # if vf fixed then just multiply with sh_coeff
        if self.volume_fractions_fixed:
            E = np.dot(A, kwargs['sh_coeff'])
        else:
            sh_coeff = np.zeros(self.optimizer.Ncoef_total)
            sh_coeff[self.optimizer.sh_start:
                     self.optimizer.Ncoef + self.optimizer.sh_start] = kwargs[
                'sh_coeff']
            for i, name in enumerate(self.partial_volume_names):
                sh_coeff[self.optimizer.vf_indices[i]] = (
                    kwargs[name] / (2 * np.sqrt(np.pi)))
            E = np.dot(A, sh_coeff)
        return E


