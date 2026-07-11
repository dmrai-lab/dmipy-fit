# -*- coding: utf-8 -*-
from collections import OrderedDict
from uuid import uuid4

import numpy as np
from dipy.utils.optpkg import optional_package

from ..utils.utils import (
    T1_tortuosity,
    parameter_equality,
    fractional_parameter)

graphviz, have_graphviz, _ = optional_package("graphviz")
if have_graphviz:
    from graphviz import Digraph

__all__ = [
    'ModelProperties',
    'MultiCompartmentModelProperties',
    'ReturnFixedValue',
    'homogenize_x0_to_data',
]

class ModelProperties:
    "Contains various properties for CompartmentModels."

    S0_response = 1.

    @property
    def parameter_ranges(self):
        """Returns the optimization ranges of the model parameters.
        These ranges are given in O(1) scale so optimization algorithms
        don't suffer from large scale differences in optimization parameters.
        """
        return OrderedDict(self._parameter_ranges.copy())

    @property
    def parameter_scales(self):
        """Returns the optimization scales for the model parameters.
        The scales scale the parameter_ranges to their actual size inside
        optimization algorithms.
        """
        return OrderedDict(self._parameter_scales.copy())

    @property
    def parameter_types(self):
        """Returns the optimization scales for the model parameters.
        The scales scale the parameter_ranges to their actual size inside
        optimization algorithms.
        """
        return OrderedDict(self._parameter_types.copy())

    @property
    def parameter_names(self):
        "Returns the names of model parameters."
        return self._parameter_ranges.keys()

    @property
    def parameter_cardinality(self):
        "Returns the cardinality of model parameters"
        return OrderedDict(
            [(k, len(np.atleast_2d(self.parameter_ranges[k]))) for k in
             self.parameter_ranges])


class MultiCompartmentModelProperties:
    "Class that contains various properties of MultiCompartmentModel instance."

    @property
    def parameter_names(self):
        "Returns the names of model parameters."
        return list(self.parameter_ranges.keys())

    def parameter_vector_to_parameters(self, parameter_vector):
        """Returns the model parameters in dictionary format according to their
        parameter_names. Takes parameter_vector as input, which is the same as
        the output of a FittedMultiCompartmentModel.fitted_parameter_vector.

        Parameters
        ----------
        parameter_vector: array of size (Ndata_x, Ndata_y, ..., Nparameters),
            array that contains the linearized model parameters for an ND-array
            of data voxels.

        Returns
        -------
        parameter: dictionary with parameter_names as parameter keys,
            contains the model parameters in dictionary format.
        """
        parameters = {}
        current_pos = 0
        if parameter_vector.ndim == 1:
            for parameter, card in self.parameter_cardinality.items():
                parameters[parameter] = parameter_vector[
                    current_pos: current_pos + card]
                if card == 1:
                    parameters[parameter] = parameters[parameter][0]
                current_pos += card
        else:
            for parameter, card in self.parameter_cardinality.items():
                parameters[parameter] = parameter_vector[
                    ..., current_pos: current_pos + card]
                if card == 1:
                    parameters[parameter] = parameters[parameter][..., 0]
                current_pos += card
        return parameters

    def parameters_to_parameter_vector(self, **parameters):
        """Returns the model parameters in array format. The input is a
        parameters dictionary that has parameter_names as keys. This is also
        the output of a FittedMultiCompartmentModel.fitted_parameters.

        It's possible to give an array of values for one parameter and only a
        float for others. The function will automatically assume that that the
        float parameters are constant in the data set and broadcast them
        accordingly.

        The output parameter_vector can be used in simulate_data() to generate
        data according to the given input parameters.

        Parameters
        ----------
        parameters: keyword arguments of parameter_names.
            Can be given as **parameter_dictionary that contains the model
            parameter values.

        Returns
        -------
        parameter_vector: array of size (Ndata_x, Ndata_y, ..., Nparameters),
            array that contains the linearized model parameters for an ND-array
            of data voxels.
        """
        parameter_vector = []
        parameter_shapes = []
        for parameter, card in self.parameter_cardinality.items():
            value = np.atleast_1d(parameters.get(parameter, np.nan)
                                  if parameter.endswith('_T2')
                                  else parameters[parameter])
            if card == 1 and not np.all(value.shape == np.r_[1]):
                parameter_shapes.append(value.shape)
            elif card > 1 and not np.all(value.shape == np.r_[card]):
                parameter_shapes.append(value.shape[:-1])

        if len(set(parameter_shapes)) > 1:
            msg = "parameter shapes are inconsistent."
            raise ValueError(msg)
        elif len(set(parameter_shapes)) == 0:
            for parameter, card in self.parameter_cardinality.items():
                parameter_vector.append(
                    parameters.get(parameter, np.nan)
                    if parameter.endswith('_T2')
                    else parameters[parameter])
            parameter_vector = np.hstack(parameter_vector)
        elif len(set(parameter_shapes)) == 1:
            for parameter, card in self.parameter_cardinality.items():
                value = np.atleast_1d(
                    parameters.get(parameter, np.nan)
                    if parameter.endswith('_T2')
                    else parameters[parameter])
                if card == 1 and np.all(value.shape == np.r_[1]):
                    parameter_vector.append(
                        np.tile(value[0], np.r_[parameter_shapes[0], 1]))
                elif card == 1 and not np.all(value.shape == np.r_[1]):
                    parameter_vector.append(value[..., None])
                elif card > 1 and np.all(value.shape == np.r_[card]):
                    parameter_vector.append(
                        np.tile(value, np.r_[parameter_shapes[0], 1])
                    )
                else:
                    parameter_vector.append(parameters[parameter])
            parameter_vector = np.concatenate(parameter_vector, axis=-1)
        return parameter_vector

    def parameter_initial_guess_to_parameter_vector(self, **parameters):
        """Function that returns a parameter_vector while allowing for partial
        input of model parameters, setting the ones that were not given to
        'None'. Such an array can be given to the fit() function to provide an
        initial parameter guess when fitting the data to the model.

        Parameters
        ----------
        parameters: keyword arguments of parameter names,
            parameter values of only the parameters you want to give as an
            initial condition for the optimizer.

        Returns
        -------
        parameter_vector: array of size (Ndata_x, Ndata_y, ..., Nparameters),
            array that contains the linearized model parameters for an ND-array
            of data voxels, with None's for non-given parameters.
        """
        set_parameters = {}
        parameter_cardinality = self.parameter_cardinality.copy()
        for parameter, value in parameters.items():
            if parameter in self.parameter_cardinality.keys():
                set_parameters[parameter] = value
                del parameter_cardinality[parameter]
            else:
                msg = '"{}" is not a valid model parameter.'.format(parameter)
                raise ValueError(msg)
        if len(parameter_cardinality) == 0:
            print("All model parameters set or have initial guess.")
        else:
            for parameter, card in parameter_cardinality.items():
                set_parameters[parameter] = np.tile(np.nan, card)
        return self.parameters_to_parameter_vector(**set_parameters)

    def _prepare_parameters(self):
        """Prepares the parameter ranges, scales, cadinality and parameter
        upon instantiating the MultiCompartmentModel"""
        self.model_names = []
        model_counts = {}

        for model in self.models:
            if model.__class__ not in model_counts:
                model_counts[model.__class__] = 1
            else:
                model_counts[model.__class__] += 1

            self.model_names.append(
                '{}_{:d}_'.format(
                    model.__class__.__name__,
                    model_counts[model.__class__]
                )
            )

        self.parameter_ranges = OrderedDict({
            model_name + k: v
            for model, model_name in zip(self.models, self.model_names)
            for k, v in model.parameter_ranges.items()
        })

        self.parameter_scales = OrderedDict({
            model_name + k: v
            for model, model_name in zip(self.models, self.model_names)
            for k, v in model.parameter_scales.items()
        })

        self.parameter_types = OrderedDict({
            model_name + k: v
            for model, model_name in zip(self.models, self.model_names)
            for k, v in model.parameter_types.items()
        })

        self._parameter_map = {
            model_name + k: (model, k)
            for model, model_name in zip(self.models, self.model_names)
            for k in model.parameter_ranges
        }

        self._inverted_parameter_map = {
            v: k for k, v in self._parameter_map.items()
        }

        self.parameter_cardinality = OrderedDict([
            (k, len(np.atleast_2d(self.parameter_ranges[k])))
            for k in self.parameter_ranges
        ])

    def _prepare_partial_volumes(self):
        "Prepares partial volumes upon instantiating the MultiCompartmentModel"
        if len(self.models) > 1:
            self.partial_volume_names = [
                'partial_volume_{:d}'.format(i)
                for i in range(len(self.models))
            ]

            for i, partial_volume_name in enumerate(
                    self.partial_volume_names):
                self.parameter_ranges[partial_volume_name] = (0.01, .99)
                self.parameter_scales[partial_volume_name] = 1.
                self._parameter_map[partial_volume_name] = (
                    None, partial_volume_name
                )
                self.parameter_types[partial_volume_name] = 'normal'
                self._inverted_parameter_map[(None, partial_volume_name)] = \
                    partial_volume_name
                self.parameter_cardinality[partial_volume_name] = 1
        else:
            self.partial_volume_names = []

    def _prepare_parameter_links(self):
        """Prepares parameter links if given as input to MultiCompartmentModel.
        It first checks if the parameter that will be linked exists. If so,
        then it removes it from the parameter ranges, scales and cardinality,
        so it will not be optimized (as it will be a function of other
        parameters)."""
        for i, parameter_function in enumerate(self.parameter_links):
            parameter_model, parameter_name, parameter_function, arguments = \
                parameter_function

            if (
                    (parameter_model, parameter_name)
                    not in self._inverted_parameter_map
            ):
                raise ValueError(
                    "Parameter function {} doesn't exist".format(i)
                )

            parameter_name = self._inverted_parameter_map[
                (parameter_model, parameter_name)
            ]

            del self.parameter_ranges[parameter_name]
            del self.parameter_cardinality[parameter_name]
            del self.parameter_scales[parameter_name]
            del self.parameter_types[parameter_name]
            del self.parameter_optimization_flags[parameter_name]

    def _prepare_model_properties(self):
        """Checks that spherical mean and regular models cannot be optimized
        together, and whether the model can estimate a Fiber Orientation
        Distribution (FOD)."""
        self.fod_available = False
        for model in self.models:
            try:
                model.fod
                self.fod_available = True
            except AttributeError:
                pass

    def _check_for_double_model_class_instances(self):
        "Checks all models have unique class instances."
        if len(self.models) != len(set(self.models)):
            msg = "Each model in the multi-compartment model must be "
            msg += "instantiated separately. For example, to make a model "
            msg += "with two sticks, the models must be given as "
            msg += "models = [stick1, stick2], not as "
            msg += "models = [stick1, stick1]."
            raise ValueError(msg)

    def add_linked_parameters_to_parameters(self, parameters):
        """When making the MultiCompartmentModel function call, adds the linked
        parameter to the optimized parameters by evaluating the parameter link
        function."""
        if len(self.parameter_links) == 0:
            return parameters
        parameters = parameters.copy()
        for parameter in self.parameter_links[::-1]:
            parameter_model, parameter_name, parameter_function, arguments = \
                parameter
            parameter_name = self._inverted_parameter_map[
                (parameter_model, parameter_name)
            ]

            if len(arguments) > 0:
                argument_values = []
                for argument in arguments:
                    argument_name = self._inverted_parameter_map[argument]
                    argument_values.append(parameters.get(
                        argument_name
                    ))

                parameters[parameter_name] = parameter_function(
                    *argument_values
                )
            else:
                parameters[parameter_name] = parameter_function()
        return parameters

    def _check_if_volume_fractions_are_fixed(self):
        "checks if volume fractions have been fixed."
        self.volume_fractions_fixed = True
        for name, flag in self.parameter_optimization_flags.items():
            if flag and name in self.partial_volume_names:
                self.volume_fractions_fixed = False

    def _prepare_parameters_to_optimize(self):
        "Sets up which parmameters to optimize."
        # T1 and eta are passive by default (like _T2): activated explicitly
        # via set_initial_guess_parameter().
        _passive_exact = {'T1', 'eta'}
        self.parameter_optimization_flags = OrderedDict({
            k: False if (k.endswith('_T2') or k in _passive_exact) else True
            for k, v in self.parameter_cardinality.items()
        })

    @property
    def bounds_for_optimization(self):
        "Returns the linear parameter bounds for the model optimization."
        bounds = []
        for parameter, card in self.parameter_cardinality.items():
            range_ = self.parameter_ranges[parameter]
            if card == 1:
                bounds.append(range_)
            else:
                for i in range(card):
                    bounds.append((range_[i][0], range_[i][1]))
        return bounds

    @property
    def opt_params_for_optimization(self):
        "Returns the linear bools whether to optimize a model parameter."
        params = []
        for parameter, card in self.parameter_cardinality.items():
            optimize_param = self.parameter_optimization_flags[parameter]
            if card == 1:
                params.append(optimize_param)
            else:
                for i in range(card):
                    params.append(optimize_param)
        return params

    @property
    def scales_for_optimization(self):
        "Returns the linear parameter scales for model optimization."
        return np.hstack([scale for parameter, scale in
                          self.parameter_scales.items()])

    def _check_for_tortuosity_constraint(self):
        for link in self.parameter_links:
            if isinstance(link[2], T1_tortuosity):
                msg = "Cannot use MIX optimization when the Tortuosity "
                msg += "constraint is set in the MultiCompartmentModel. To "
                msg += "use MIX while imposing Tortuosity, set the constraint "
                msg += "in the DistributedModel step."
                raise ValueError(msg)

    def set_initial_guess_parameter(self, parameter_name, value):
        """
        Allows the user to fix an optimization parameter to a static value.
        The fixed parameter will be removed from the optimized parameters and
        added as a linked parameter.

        Parameters
        ----------
        parameter_name: string
            name of the to-be-fixed parameters, see self.parameter_names.
        value: float or list of corresponding parameter_cardinality.
            the value to fix the parameter at in SI units.
        """
        if parameter_name in self.parameter_ranges.keys():
            if parameter_name.endswith('_T2') or parameter_name in ('T1', 'eta'):
                self.parameter_optimization_flags[parameter_name] = True
            card = self.parameter_cardinality[parameter_name]
            if card == 1:
                if isinstance(value, int) or isinstance(value, float):
                    self.x0_parameters[parameter_name] = value
                elif isinstance(value, np.ndarray):
                    self._add_initial_guess_parameter_array(
                        parameter_name, value)
            elif card >= 2:
                value = np.array(value, dtype=float)
                if value.shape[-1] != card:
                    msg = '{} can only be fixed to an array or list with ' \
                          'last dimension {}.'
                    raise ValueError(msg.format(parameter_name, type(value)))
                if value.ndim == 1:
                    self.x0_parameters[parameter_name] = value
                if value.ndim > 1:
                    self._add_initial_guess_parameter_array(
                        parameter_name, value)
        else:
            msg = '{} does not exist or has already been fixed.'.format(
                parameter_name)
            raise ValueError(msg)

    def _add_initial_guess_parameter_array(
            self, parameter_name, parameter_array):
        temp_dict = self.x0_parameters.copy()
        temp_dict[parameter_name] = parameter_array
        try:
            self.parameter_initial_guess_to_parameter_vector(
                **temp_dict)
            self.x0_parameters = temp_dict
        except ValueError:
            msg = '{} does not have the same shape'.format(parameter_name)
            msg += 'as the previously fixed parameters.'
            raise ValueError(msg)

    def set_fixed_parameter(self, parameter_name, value):
        """
        Allows the user to fix an optimization parameter to a static value.
        The fixed parameter will be removed from the optimized parameters and
        added as a linked parameter.

        Parameters
        ----------
        parameter_name: string
            name of the to-be-fixed parameters, see self.parameter_names.
        value: float or list of corresponding parameter_cardinality.
            the value to fix the parameter at in SI units.
        """
        if parameter_name in self.parameter_ranges.keys():
            card = self.parameter_cardinality[parameter_name]
            if card == 1:
                if isinstance(value, int) or isinstance(value, float):
                    self._add_fixed_parameter_value(parameter_name,
                                                    float(value))
                elif isinstance(value, np.ndarray):
                    self._add_fixed_parameter_array(parameter_name, value)
                else:
                    msg = 'fixed value for {} must be number or np.array, ' \
                          'currently {}'
                    raise ValueError(msg.format(parameter_name, type(value)))
            elif card >= 2:
                value = np.array(value, dtype=float)
                if value.shape[-1] != card:
                    msg = '{} can only be fixed to an array or list with ' \
                          'last dimension {}.'
                    raise ValueError(msg.format(parameter_name, card))
                if value.ndim == 1:
                    self._add_fixed_parameter_value(parameter_name, value)
                if value.ndim > 1:
                    self._add_fixed_parameter_array(parameter_name, value)
        else:
            msg = '{} does not exist or has already been fixed.'.format(
                parameter_name)
            raise ValueError(msg)

    def _add_fixed_parameter_value(self, parameter_name, value):
        model, name = self._parameter_map[parameter_name]
        parameter_link = (model, name, ReturnFixedValue(value), [])
        self.parameter_links.append(parameter_link)
        del self.parameter_ranges[parameter_name]
        del self.parameter_cardinality[parameter_name]
        del self.parameter_scales[parameter_name]
        del self.parameter_types[parameter_name]
        del self.parameter_optimization_flags[parameter_name]

    def _add_fixed_parameter_array(self, parameter_name, parameter_array):
        temp_dict = self.x0_parameters.copy()
        temp_dict[parameter_name] = parameter_array
        try:
            self.parameter_initial_guess_to_parameter_vector(
                **temp_dict)
            self.x0_parameters = temp_dict
            self.parameter_optimization_flags[parameter_name] = False
        except ValueError:
            msg = '{} does not have the same shape'.format(parameter_name)
            msg += 'as the previously fixed parameters.'
            raise ValueError(msg)

    def set_tortuous_parameter(self, lambda_perp_parameter_name,
                               lambda_par_parameter_name,
                               volume_fraction_intra_parameter_name,
                               volume_fraction_extra_parameter_name,
                               S0_correction=False):
        """
        Allows the user to set a tortuosity constraint on the perpendicular
        diffusivity of the extra-axonal compartment, which depends on the
        intra-axonal volume fraction and parallel diffusivity.

        The perpendicular diffusivity parameter will be removed from the
        optimized parameters and added as a linked parameter.

        To employ the multi-tissue correction of tortuosity it is sufficient to
        pass the S0_intra and S0_extra parameters.

        Parameters
        ----------
        lambda_perp_parameter_name: string
            name of the perpendicular diffusivity parameter, see
            self.parameter_names.
        lambda_par_parameter_name: string
            name of the parallel diffusivity parameter, see
            self.parameter_names.
        volume_fraction_intra_parameter_name: string
            name of the intra-axonal volume fraction parameter, see
            self.parameter_names.
        volume_fraction_extra_parameter_name: string
            name of the extra-axonal volume fraction parameter, see
            self.parameter_names.
        S0_correction: bool
            If True, it uses the S0 of the intra-axonal and extra-axonal
            compartments to define the tortuosity constraint. Default: False.
        """
        params = [lambda_perp_parameter_name, lambda_par_parameter_name,
                  volume_fraction_intra_parameter_name,
                  volume_fraction_extra_parameter_name]
        for param in params:
            try:
                self.parameter_cardinality[param]
            except KeyError:
                msg = ("{} does not exist or has already been fixed.").format(
                    param)
                raise ValueError(msg)

        model, name = self._parameter_map[lambda_perp_parameter_name]
        if S0_correction and self.S0_tissue_responses is not None:
            s0intra_tag = volume_fraction_intra_parameter_name.split('_')[-1]
            s0extra_tag = volume_fraction_extra_parameter_name.split('_')[-1]
            S0_intra = self.S0_tissue_responses[int(s0intra_tag)]
            S0_extra = self.S0_tissue_responses[int(s0extra_tag)]
            print('Employing S0 correction of tortuosity constraint with:')
            print('S0_intra: {}'.format(S0_intra))
            print('S0_extra: {}'.format(S0_extra))
        else:
            S0_intra = 1.
            S0_extra = 1.
        tortuosity = T1_tortuosity(S0_intra, S0_extra)

        self.parameter_links.append([model, name, tortuosity, [
            self._parameter_map[lambda_par_parameter_name],
            self._parameter_map[volume_fraction_intra_parameter_name],
            self._parameter_map[volume_fraction_extra_parameter_name]]])
        del self.parameter_ranges[lambda_perp_parameter_name]
        del self.parameter_cardinality[lambda_perp_parameter_name]
        del self.parameter_scales[lambda_perp_parameter_name]
        del self.parameter_types[lambda_perp_parameter_name]
        del self.parameter_optimization_flags[lambda_perp_parameter_name]

    def set_tortuosity_constraint(self, *args, **kwargs):
        """Deprecated alias of :meth:`set_tortuous_parameter` (the original dmipy
        public name). Forwards the call unchanged and emits a DeprecationWarning so
        1.x code keeps working."""
        import warnings
        warnings.warn(
            "set_tortuosity_constraint() is deprecated; use set_tortuous_parameter() "
            "(identical signature).", DeprecationWarning, stacklevel=2)
        return self.set_tortuous_parameter(*args, **kwargs)

    def set_equal_parameter(self, parameter_name_in, parameter_name_out):
        """
        Allows the user to set two parameters equal to each other. This is used
        for example in the NODDI model to set the parallel diffusivities of the
        Stick and Zeppelin compartment to the same value.

        The second input parameter will be removed from the optimized
        parameters and added as a linked parameter.

        Parameters
        ----------
        parameter_name_in: string
            the first parameter name, see self.parameter_names.
        parameter_name_out: string,
            the second parameter name, see self.parameter_names. This is the
            parameter that will be removed form the optimzed parameters.
        """
        params = [parameter_name_in, parameter_name_out]
        for param in params:
            try:
                self.parameter_cardinality[param]
            except KeyError:
                msg = ("{} does not exist or has already been fixed.").format(
                    param)
                raise ValueError(msg)
        model, name = self._parameter_map[parameter_name_out]
        self.parameter_links.append([model, name, parameter_equality, [
            self._parameter_map[parameter_name_in]]])
        del self.parameter_ranges[parameter_name_out]
        del self.parameter_cardinality[parameter_name_out]
        del self.parameter_scales[parameter_name_out]
        del self.parameter_types[parameter_name_out]
        del self.parameter_optimization_flags[parameter_name_out]

    def set_fractional_parameter(self,
                                 parameter1_smaller_equal_than, parameter2):
        r"""
        Allows to impose a constraint to make one parameter smaller or equal to
        another parameter. This is done by replacing parameter1 with a
        new parameter that is defined as a fraction between 0 and 1 of
        parameter2. The new parameter will be the same as the old parameter
        name with "_fraction" appended to it.

        Parameters
        ----------
        parameter1_smaller_equal_than: string
            parameter name to be made a fraction of parameter2
        parameter2: string
            the parameter that is larger or equal than parameter1
        """
        params = [parameter1_smaller_equal_than, parameter2]
        for param in params:
            try:
                self.parameter_cardinality[param]
            except KeyError:
                msg = ("{} does not exist or has already been fixed.").format(
                    param)
                raise ValueError(msg)
        # append new parameter to parameters
        new_parameter_name = parameter1_smaller_equal_than + '_fraction'

        self._add_optimization_parameter(
            new_parameter_name, [0., 1.], 1., 1, 'normal', True)
        model, name = self._parameter_map[parameter1_smaller_equal_than]
        self.parameter_links.append([model, name, fractional_parameter, [
            self._parameter_map[new_parameter_name],
            self._parameter_map[parameter2]]])

        # remove old parameter1
        del self.parameter_ranges[parameter1_smaller_equal_than]
        del self.parameter_cardinality[parameter1_smaller_equal_than]
        del self.parameter_scales[parameter1_smaller_equal_than]
        del self.parameter_types[parameter1_smaller_equal_than]
        del self.parameter_optimization_flags[parameter1_smaller_equal_than]

    def _add_optimization_parameter(
            self,
            parameter_name,
            parameter_range,
            parameter_scale,
            parameter_card,
            parameter_type,
            parameter_flag):
        """
        Creates new ordered dictionaries for model properties with the
        optimization parameter on top.
        """
        old_parameter_ranges = self.parameter_ranges
        old_parameter_scales = self.parameter_scales
        old_parameter_cardinality = self.parameter_cardinality
        old_parameter_types = self.parameter_types
        old_optimization_flags = self.parameter_optimization_flags

        self.parameter_ranges = OrderedDict({parameter_name: parameter_range})
        self.parameter_scales = OrderedDict({parameter_name: parameter_scale})
        self.parameter_cardinality = OrderedDict(
            {parameter_name: parameter_card})
        self.parameter_types = OrderedDict({parameter_name: parameter_type})
        self.parameter_optimization_flags = OrderedDict(
            {parameter_name: parameter_flag})

        for name, _ in old_parameter_ranges.items():
            self.parameter_ranges.update({name: old_parameter_ranges[name]})
            self.parameter_scales.update({name: old_parameter_scales[name]})
            self.parameter_cardinality.update(
                {name: old_parameter_cardinality[name]})
            self.parameter_types.update({name: old_parameter_types[name]})
            self.parameter_optimization_flags.update(
                {name: old_optimization_flags[name]})

        self._parameter_map.update({parameter_name: (None, 'fraction')})
        self._inverted_parameter_map.update(
            {(None, 'fraction'): parameter_name})

    def _check_model_params_with_acquisition_params(self, acquisition_scheme):
        for model in self.models:
            for parameter in model._required_acquisition_parameters:
                if getattr(acquisition_scheme, parameter) is None:
                    msg = "{} is not compatible with ".format(
                        model.__class__.__name__)
                    msg += "given acquisition scheme because it needs "
                    msg += "{} as an acquisition parameter.".format(parameter)
                    raise ValueError(msg)

    def visualize_model_setup(
            self, view=True, cleanup=True, with_parameters=False,
            im_format='png'):
        """
        Visualizes MultiCompartmentModel setup using graphviz module. It uses
        the uuid module to create a unique identifier for each model in the
        MultiCompartmentModel to make sure each node is referenced in a unique
        way.

        If cleanup is set to False it will save the PDF of the graph in the
        current working directory.

        If with_parameters is set to true, it will include all the parameters
        of each model in the graph. Note the graph will ignore any parameter
        links that may have already been imposed (e.g. parameter equality or
        fixed parameters).

        Parameters
        ----------
        view: boolean,
            Whether or not to visualize the graph in a popup screen.
        cleanup: boolean,
            Whether or not to delete the PDF file of the model setup.
        with_parameters: boolean,
            Whether or not to also visualize the parameters of each model.
        """
        if not have_graphviz:
            raise ImportError('graphviz package not installed.')
        dot = Digraph('Model Setup', format=im_format)
        base_model = self.__class__.__name__
        base_uuid = str(uuid4())
        dot.node(base_uuid, base_model)
        self._add_recursive_graph_node(dot, base_uuid, self, with_parameters)
        dot.render('Model Setup', view=view, cleanup=cleanup)

    def _add_recursive_graph_node(
            self, graph_model, entry_uuid, entry_model, with_parameters):
        """
        Recursive function to visualize model setup. For every model in a
        MultiCompartmentModel or a distributed model it will check if it is
        a distribution, in which case the function will call itself with the
        sub-model as input and continue until it has found the bottom of the
        model setup.

        Parameters
        ----------
        graph_model: graphviz model instance,
            Instantiated model instance to keep growing with nodes.
        entry_uuid: string,
            Entry model unique identifier from which to keep growing the graph.
        entry_model: dmipy model instance,
            Entry dmipy model from which to keep growing the graph.
        """
        for sub_model in entry_model.models:
            model_name = sub_model.__class__.__name__
            model_uuid = str(uuid4())
            graph_model.node(model_uuid, model_name)
            graph_model.edge(model_uuid, entry_uuid)
            if (sub_model._model_type == 'SphericalDistributedModel' or
                    sub_model._model_type == 'SpatialDistributedModel' or
                    sub_model._model_type == 'BundleModel'):
                self._add_recursive_graph_node(
                    graph_model, model_uuid, sub_model, with_parameters)
            elif with_parameters:
                self._add_parameter_nodes(graph_model, model_uuid, sub_model)
        if hasattr(entry_model, 'distribution'):
            dist_name = entry_model.distribution.__class__.__name__
            dist_uuid = str(uuid4())
            graph_model.node(dist_uuid, dist_name)
            graph_model.edge(dist_uuid, entry_uuid)
            if with_parameters:
                self._add_parameter_nodes(
                    graph_model, dist_uuid, entry_model.distribution)

    def _add_parameter_nodes(self, graph_model, entry_uuid, entry_model):
        """
        Adds the parameters to the graph truee if with_parameters=True in the
        visualize_model_setup function.

        Parameters
        ----------
        graph_model: graphviz model instance,
            Instantiated model instance to keep growing with nodes.
        entry_uuid: string,
            Entry model unique identifier from which to keep growing the graph.
        entry_model: dmipy model instance,
            Entry dmipy model from which to keep growing the graph.
        """
        for parameter_name in entry_model.parameter_names:
            parameter_uuid = str(uuid4())
            graph_model.node(parameter_uuid, parameter_name)
            graph_model.edge(parameter_uuid, entry_uuid)

    def _check_tissue_model_acquisition_scheme(self, acquisition_scheme):
        """Tests if acquisition scheme between MC-model and tissue response
        model are the same.

        Parameters
        ----------
        acquisition_scheme : PGSEAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using Dmipy.
        """
        for model in self.models:
            if model._model_type == 'TissueResponseModel':
                mc_scheme_params = [
                    acquisition_scheme.shell_bvalues,
                    acquisition_scheme.shell_delta,
                    acquisition_scheme.shell_Delta,
                    acquisition_scheme.shell_gradient_strengths]
                tr_scheme_params = [
                    model.acquisition_scheme.shell_bvalues,
                    model.acquisition_scheme.shell_delta,
                    model.acquisition_scheme.shell_Delta,
                    model.acquisition_scheme.shell_gradient_strengths]
                try:
                    np.testing.assert_array_almost_equal(
                        mc_scheme_params, tr_scheme_params)
                except AssertionError:
                    msg = "Acquisition scheme of MC-model and tissue response "
                    msg += "model are not the same."
                    raise ValueError(msg)

    def _check_acquisition_scheme_has_b0s(self, acquisition_scheme):
        """
        Checks if acquisition scheme has any b0-measurements. This for the
        moment is a prerequisite for signal-attenuation-based model fitting.

        Parameters
        ----------
        acquisition_scheme : PGSEAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using Dmipy.
        """
        if not np.any(acquisition_scheme.b0_mask):
            raise ValueError('acquisition scheme must have b0-measurements '
                             'for signal-attenuation-based model fitting.')

    def _construct_convolution_kernel(self, acquisition_scheme=None, **kwargs):
        """
        Helper function that constructs the convolution kernel for the given
        multi-compartment model and the initial condition x0_vector.

        First the parameter vector is converted to a dictionary with the
        corresponding parameter names. Then, the linked parameters are added to
        the given ones. Finally, the rotational harmonics of the model is
        passed to the construct_model_based_A_matrix, which constructs the
        kernel for an arbitrary PGSE-acquisition scheme.

        For multiple models with fixed volume fractions, the A-matrices
        are combined to have a combined convolution kernel.

        For multiple models without fixed volume fractions, the convolution
        kernels for anisotropic and isotropic models are concatenated, with
        the isotropic kernels always having a spherical harmonics order of 0.

        Parameters
        ----------
        acquisition_scheme: PGSEAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using Dmipy.

        Returns
        -------
        kernel: array of size (N_coef, N_data),
            Observation matrix that maps the FOD spherical harmonics
            coefficients to the DWI signal values.
        """
        parameters_dict = self.add_linked_parameters_to_parameters(kwargs)

        if acquisition_scheme is None:
            acquisition_scheme = self.scheme

        if self.volume_fractions_fixed:
            if self.N_models > 1:
                partial_volumes = [
                    parameters_dict[p] for p in self.partial_volume_names
                ]
            else:
                partial_volumes = [1.]
            kernel = 0.
            for model, partial_volume, S0 in zip(self.models,
                                                 partial_volumes,
                                                 self.S0_responses):
                parameters = {}
                for parameter in model.parameter_ranges:
                    parameter_name = self._inverted_parameter_map[
                        (model, parameter)
                    ]
                    parameters[parameter] = parameters_dict.get(
                        parameter_name
                    )
                kernel += S0 * partial_volume * (
                    model.convolution_kernel_matrix(
                        acquisition_scheme, self.sh_order, **parameters))
        else:
            kernel = []
            for model, S0 in zip(self.models, self.S0_responses):
                parameters = {}
                for parameter in model.parameter_ranges:
                    parameter_name = self._inverted_parameter_map[
                        (model, parameter)
                    ]
                    parameters[parameter] = parameters_dict.get(
                        parameter_name
                    )
                if 'orientation' in model.parameter_types.values():
                    kernel.append(S0 * model.convolution_kernel_matrix(
                        acquisition_scheme, self.sh_order, **parameters))
                else:
                    kernel.append(S0 * model.convolution_kernel_matrix(
                        acquisition_scheme, 0, **parameters))

            kernel = np.hstack(kernel)
        return kernel

    def set_parameter_optimization_bounds(self, parameter_name, bounds):
        """
        Sets the parameter optimization bounds for a given parameter.

        Parameters
        ----------
        parameter_name: string,
            name of the parameter whose bounds should be changed.
        bounds: array or size(card, 2),
            upper and lower bound for each optimized value for the given
            parameter, where card is
            self.parameter_cardinality[parameter_name]).

        Raises
        ------
        ValueError: parameter name not in model parameters
        ValueError: input bounds are not of correct shape [card, 2]
        ValueError: input higher bound is lower than lower bound
        """
        if parameter_name not in self.parameter_names:
            raise ValueError(
                '{} not in model parameters'.format(parameter_name))
        card = self.parameter_cardinality[parameter_name]
        bounds_array = np.atleast_2d(bounds)
        input_card, N_bounds = bounds_array.shape[:2]
        if bounds_array.ndim > 2 or input_card != card or N_bounds != 2:
            msg = '{} bounds must be of shape ({}, 2), currently {}.'
            raise ValueError(
                msg.format(parameter_name, card, bounds_array.shape))
        for lower, higher in bounds_array:
            if higher < lower:
                msg = 'given optimization bounds for {} are invalid: lower ' \
                      'bound {} is higher than upper bound {}.'
                raise ValueError(msg.format(parameter_name, lower, higher))
        parameter_scale = np.max(bounds)
        ranges = np.array(bounds) / parameter_scale
        self.parameter_ranges[parameter_name] = ranges
        self.parameter_scales[parameter_name] = parameter_scale




def homogenize_x0_to_data(data, x0):
    """
    Function that checks if data and initial guess x0 are of the same size.
    If x0 is 1D, it will be tiled to be the same size as data.
    """
    if x0 is not None:
        if x0.ndim == 1:
            # the same x0 will be used for every voxel in N-dimensional data.
            x0_as_data = np.tile(x0, np.r_[data.shape[:-1], 1])
        else:
            x0_as_data = x0.copy()
    if not np.all(
            x0_as_data.shape[:-1] == data.shape[:-1]
    ):
        # if x0 and data are both N-dimensional but have different shapes.
        msg = "data and x0 both N-dimensional but have different shapes. "
        msg += "Current shapes are {} and {}.".format(
            data.shape[:-1],
            x0_as_data.shape[:-1])
        raise ValueError(msg)
    return x0_as_data


class ReturnFixedValue:
    "Parameter fixing class for parameter links."

    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value
