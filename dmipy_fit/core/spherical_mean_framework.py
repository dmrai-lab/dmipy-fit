# -*- coding: utf-8 -*-
import warnings
from time import time

import numpy as np
from dipy.utils.optpkg import optional_package

from .fitted_modeling_framework import (
    FittedMultiCompartmentModel,
    FittedMultiCompartmentSphericalMeanModel)
from ..optimizers.brute2fine import (
    GlobalBruteOptimizer, Brute2FineOptimizer)
from ..optimizers.mix import MixOptimizer
from ..optimizers.multi_tissue_volume_fraction import (
    MultiTissueVolumeFractionOptimizer)
from ..utils.spherical_mean import estimate_spherical_mean_multi_shell
from .model_properties import MultiCompartmentModelProperties, homogenize_x0_to_data

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

numba, have_numba, _ = optional_package("numba")

__all__ = ['MultiCompartmentSphericalMeanModel', 'compute_sm_w_norms']


def compute_sm_w_norms(acquisition_scheme, sh_order=6):
    """Per-shell L2 norm of the L=0 zonal harmonic weight vector.

    The spherical mean for shell s is  y_s = w_s^T r_s  where
    w_s = pinv(Y_s)[0, :] / (2 sqrt(π))  is the L=0 row of the
    pseudoinverse SH matrix, normalised by the Y_00 integral.
    The effective noise std on y_s is  σ_eff_s = σ · ‖w_s‖₂.

    For b0 shells (averaged directly) the weight is uniform 1/N_dirs,
    so ‖w‖₂ = 1/sqrt(N_dirs).

    Parameters
    ----------
    acquisition_scheme : DmipyAcquisitionScheme
    sh_order : int
        SH expansion order (must match the order used during SM estimation).

    Returns
    -------
    w_norms : np.ndarray, shape (N_shells,)
        ‖w_s‖₂ for each shell in acquisition_scheme.unique_shell_indices,
        in the same order as acquisition_scheme.shell_b_values.
    """
    w_norms = []
    for shell_idx in acquisition_scheme.unique_shell_indices:
        shell_mask = acquisition_scheme.shell_indices == shell_idx
        if acquisition_scheme.shell_b0_mask[shell_idx]:
            N_dirs = int(np.sum(shell_mask))
            w_norms.append(1.0 / np.sqrt(N_dirs))
        else:
            sh_mat = acquisition_scheme.shell_sh_matrices[shell_idx]
            sh_mat_inv = np.linalg.pinv(sh_mat)
            # L=0 row / (2*sqrt(pi)) matches estimate_spherical_mean_shell
            w = sh_mat_inv[0, :] / (2.0 * np.sqrt(np.pi))
            w_norms.append(np.linalg.norm(w))
    return np.array(w_norms)


class MultiCompartmentSphericalMeanModel(MultiCompartmentModelProperties):
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
            {'key': 'kaden2016', 'authors': 'Kaden E, Kruggel F, Alexander DC',
             'title': 'Quantitative mapping of the per-axon diffusion coefficients in brain white matter',
             'journal': 'Magnetic Resonance in Medicine',
             'year': 2016, 'doi': '10.1002/mrm.25734'}
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'no_exchange',
         'name': 'No inter-compartment exchange',
         'condition_human': 'Assumes no water exchange between compartments during the diffusion time. Compartment signals are computed independently and summed. Invalid when membrane permeability is high (exchange time << diffusion time).',
         'severity': 'warning',
         'source_key': 'kaden2016'},
    ]

    def __init__(self, models, S0_tissue_responses=None, parameter_links=None):
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
        self.parameter_links = parameter_links
        if parameter_links is None:
            self.parameter_links = []

        self._check_for_NMR_models()
        self._prepare_parameters()
        self._delete_orientation_parameters()
        self._prepare_partial_volumes()
        self._prepare_parameter_links()
        self._prepare_model_properties()
        self._check_for_double_model_class_instances()
        self._prepare_parameters_to_optimize()
        self.x0_parameters = {}

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

    def fit(self, acquisition_scheme, data,
            mask=None, solver='brute2fine', Ns=5, maxiter=300,
            N_sphere_samples=30, use_parallel_processing=False,
            number_of_processors=None, batch_size=None,
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
            Initial guess for sigma (noise std in normalised signal units =
            1/SNR0). When provided, sigma is jointly optimised with the
            diffusion model parameters using per-shell Rician MLE in the
            spherical mean space.  Only supported with solver='jax'.
            Default None (MSE loss, no noise modelling).
        sigma_range : tuple (float, float),
            (lower, upper) bounds for sigma. Only used when sigma_x0 is not
            None. Default (0.001, 0.5) covers SNR 2–1000.

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
        # Accept either the full per-measurement signal (default; reduced to the
        # spherical mean below) or data that is ALREADY the per-shell spherical
        # mean (last axis == N_shells). The latter lets callers reduce once and
        # reuse it (e.g. isolate_wm_spherical_mean) without re-deriving it, and
        # without ever materialising a full per-measurement intermediate.
        data_is_spherical_mean = (
            data_.shape[-1] == self.scheme.N_shells
            and data_.shape[-1] != self.scheme.number_of_measurements)
        t2_optimization_active = (
            self.scheme.TE is not None and
            any(v for k, v in self.parameter_optimization_flags.items()
                if k.endswith('_T2')))
        if t2_optimization_active:
            # Fitting compartment T2: no b0 normalisation -- compare the raw per-shell
            # signal to the raw (relaxation-carrying) model, exactly as
            # MultiCompartmentModel does. Spherical-mean <-> full-framework parity.
            S0 = np.ones(np.r_[data_.shape[:-1], self.scheme.N_shells])
        elif data_is_spherical_mean:
            # b0 lives in the b0 shell(s) of the per-shell array
            S0 = np.mean(data_[..., self.scheme.shell_b0_mask], axis=-1)
        elif self.scheme.TE is None or len(np.unique(self.scheme.TE)) == 1:
            S0 = np.mean(data_[..., self.scheme.b0_mask], axis=-1)
        else:  # if multiple TE are in the data
            S0 = np.ones(np.r_[data_.shape[:-1],
                               len(acquisition_scheme.shell_TE)])
            for TE_ in self.scheme.shell_TE:
                TE_mask = self.scheme.shell_TE == TE_
                TE_mask_shell = self.scheme.TE == TE_
                TE_b0_mask = np.all([self.scheme.b0_mask, TE_mask_shell],
                                    axis=0)
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

        # reduce the full signal to the per-shell spherical mean -- unless the
        # caller already passed the spherical mean, in which case use it as-is.
        if data_is_spherical_mean:
            data_to_fit = data_
        else:
            data_to_fit = np.zeros(
                np.r_[data_.shape[:-1], self.scheme.N_shells])
            for pos in zip(*mask_pos):
                data_to_fit[pos] = estimate_spherical_mean_multi_shell(
                    data_[pos], self.scheme)

        start = time()
        if solver == 'brute2fine':
            global_brute = GlobalBruteOptimizer(
                self, self.scheme,
                x0_, Ns, N_sphere_samples)
            fit_func = Brute2FineOptimizer(self, self.scheme, Ns)
            print('Setup brute2fine optimizer in {} seconds'.format(
                time() - start))
        elif solver == 'mix':
            self._check_for_tortuosity_constraint()
            fit_func = MixOptimizer(self, self.scheme, maxiter)
            print('Setup MIX optimizer in {} seconds'.format(
                time() - start))
        elif solver == 'jax':
            from ..jax.multicompartment_jax import build_mc_sm_forward_fn
            from ..jax.optimizers_jax import JaxOptimizer
            from ..jax.vmap_fit import vmap_fit
            from ..jax.losses_jax import rician_nll_sm_fittable
            sm_fittable_loss = None
            if sigma_x0 is not None:
                w_norms = compute_sm_w_norms(acquisition_scheme)
                sm_fittable_loss = rician_nll_sm_fittable(w_norms)
            jax_opt = JaxOptimizer(
                self, self.scheme, maxiter=maxiter,
                forward_fn_override=build_mc_sm_forward_fn(
                    self, self.scheme, broadcast=False),  # data is per-shell means
                fit_sigma=(sigma_x0 is not None),
                sigma_x0=sigma_x0,
                sigma_range=sigma_range,
                fittable_loss_override=sm_fittable_loss)
            print('Setup JAX SM optimizer in {} seconds'.format(
                time() - start))
        else:
            msg = "Unknown solver name {}".format(solver)
            raise ValueError(msg)
        if solver != 'jax':
            self.optimizer = fit_func

        start = time()
        if solver == 'jax':
            data_masked  = data_to_fit[mask_pos]          # (N_voxels, N_shells)
            S0_masked    = S0[mask_pos]                    # (N_voxels,) or (N_voxels, N_shells) in multi-TE
            # In multi-TE mode S0 is per-shell; in single-TE it is a per-voxel scalar.
            if S0_masked.ndim == 1:
                E_masked = (data_masked.T / S0_masked).T  # (N_voxels, N_shells)
            else:
                E_masked = data_masked / S0_masked         # element-wise (N_voxels, N_shells)
            x0_masked    = x0_[mask_pos]                  # (N_voxels, N_params)

            # Convert full normalized x0 → nested-VF form expected by optimizer
            x0_nested_all = np.array([
                jax_opt._prepare_x0(x0_masked[i])
                for i in range(N_voxels)
            ])                                             # (N_voxels, N_nested[+1])

            fitted_nested = vmap_fit(
                jax_opt, E_masked, x0_nested_all, batch_size=batch_size)

            # Convert back from nested-VF → full normalized form (+ sigma)
            fitted_parameters_lin = np.array([
                jax_opt._unnest(fitted_nested[i])
                for i in range(N_voxels)
            ])                                             # (N_voxels, N_params[+1])
        else:
            all_args = []
            for idx, pos in enumerate(zip(*mask_pos)):
                voxel_E = data_to_fit[pos] / S0[pos]
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

        fitted_mt_fractions = None
        if self.S0_tissue_responses:
            # secondary geometric-volume-fraction step (S0-weighted NNLS). Kept
            # separate from the non-linear fit so the (arbitrary, scanner-
            # dependent) S0 scaling does not perturb the non-linear solver's
            # absolute/relative loss tolerances.
            print('Starting secondary multi-tissue optimization.')
            start = time()
            data_masked = data_to_fit[mask_pos]          # (N_voxels, N_shells)
            # both fraction solvers take SI-unit parameter vectors
            params_si = fitted_parameters_lin * self.scales_for_optimization
            try:
                from ..jax.fractions_jax import (
                    fit_multi_tissue_fractions_jax, supports_jax_fraction_fit)
                use_jax = supports_jax_fraction_fit(self, spherical_mean=True)
            except ImportError:
                use_jax = False
            if use_jax:
                mt_fractions = fit_multi_tissue_fractions_jax(
                    self, acquisition_scheme, params_si, data_masked,
                    self.S0_tissue_responses, spherical_mean=True)
            else:
                mt_fractions = np.empty(
                    np.r_[N_voxels, self.N_models], dtype=float)
                fit_func = MultiTissueVolumeFractionOptimizer(
                    acquisition_scheme, self, self.S0_tissue_responses)
                for idx in range(len(params_si)):
                    mt_fractions[idx] = fit_func(data_masked[idx],
                                                 params_si[idx])
            fitting_time = time() - start
            msg = 'Multi-tissue fitting of {} voxels complete in {} seconds.'
            print(msg.format(len(mt_fractions), fitting_time))
            fitted_mt_fractions = np.zeros(np.r_[mask.shape, self.N_models])
            fitted_mt_fractions[mask_pos] = mt_fractions

        # When fit_sigma=True, _unnest() appends sigma (SI units) as the last
        # column of fitted_parameters_lin.  Separate it before applying scales.
        fitted_sigma = None
        if solver == 'jax' and sigma_x0 is not None:
            sigma_lin = fitted_parameters_lin[:, -1]        # (N_voxels,) SI
            fitted_parameters_lin = fitted_parameters_lin[:, :-1]
            fitted_sigma = np.zeros(x0_.shape[:-1], dtype=float)
            fitted_sigma[mask_pos] = sigma_lin

        fitted_parameters = np.zeros_like(x0_, dtype=float)
        fitted_parameters[mask_pos] = (
            fitted_parameters_lin * self.scales_for_optimization)

        return FittedMultiCompartmentSphericalMeanModel(
            self, S0, mask, fitted_parameters, fitted_mt_fractions,
            fitted_sigma=fitted_sigma)

    @staticmethod
    def to_spherical_mean(acquisition_scheme, data):
        """Reduce a full per-measurement signal to its per-shell spherical mean.

        The spherical mean over each shell's gradient directions is the
        representation the spherical-mean models actually fit; ``fit`` normally
        computes it internally, but exposing it lets you reduce once and reuse
        it (e.g. feed the result straight back to ``fit``, which accepts
        spherical-mean data directly), or inspect/QC the reduced signal.

        Parameters
        ----------
        acquisition_scheme : AcquisitionScheme instance.
        data : array of size (..., N_measurements).

        Returns
        -------
        spherical_mean : array of size (..., N_shells),
            the b0 shell mean plus one value per diffusion-weighted shell.
        """
        from ..utils.spherical_mean import estimate_spherical_mean_multi_shell
        data = np.asarray(data, dtype=float)
        flat = data.reshape(-1, data.shape[-1])
        out = np.empty((flat.shape[0], acquisition_scheme.N_shells))
        for i in range(flat.shape[0]):
            out[i] = estimate_spherical_mean_multi_shell(
                flat[i], acquisition_scheme)
        return out.reshape(data.shape[:-1] + (acquisition_scheme.N_shells,))

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

        Ndata = acquisition_scheme.shell_indices.max() + 1
        if isinstance(parameters_array_or_dict, np.ndarray):
            x0 = parameters_array_or_dict
        elif isinstance(parameters_array_or_dict, dict):
            x0 = self.parameters_to_parameter_vector(
                **parameters_array_or_dict)

        x0_at_least_2d = np.atleast_2d(x0)
        if x0_at_least_2d.shape[-1] == 0:
            # Zero-parameter model (e.g. a lone S1Dot): fully determined signal,
            # no voxel/parameter axis to loop over.
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
        if quantity == "signal":
            values = 0
        elif quantity == "stochastic cost function":
            values = np.empty((
                len(acquisition_scheme_or_vertices.shell_bvalues),
                len(self.models)
            ))
            counter = 0

        kwargs = self.add_linked_parameters_to_parameters(
            kwargs
        )
        if len(self.models) > 1:
            # cast to float arrays so list-valued partial volumes (e.g.
            # partial_volume_0=[0.5]) multiply with the per-shell spherical mean,
            # matching the MultiCompartmentModel.__call__ behaviour.
            partial_volumes = [
                np.asarray(kwargs[p], dtype=float) for p in self.partial_volume_names
            ]
        else:
            partial_volumes = [1.]

        for model_index, (model_name, model, partial_volume) in enumerate(zip(
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
                sm = model.spherical_mean(
                    acquisition_scheme_or_vertices, **parameters)
                # Relaxation is now an occupancy-gated factor on the compartment
                # (TransverseRelaxation), applied inside model.spherical_mean via
                # OccupancyGatedModel; no special-cased T2 here.
                values = values + partial_volume * sm
            elif quantity == "stochastic cost function":
                values[:, counter] = model.spherical_mean(
                    acquisition_scheme_or_vertices,
                    **parameters)
                counter += 1
        return values


