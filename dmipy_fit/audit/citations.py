"""Citation graph walker for composed dmipy models.

Traverses a model tree (MultiCompartmentModel, distributed models,
signal models) collecting all ``_citations`` and ``_validity_constraints``
into a deduplicated result dictionary.
"""

from collections import OrderedDict


def _collect_from_model(model, context_prefix=''):
    """Recursively collect citations and constraints from a single model."""
    citations = []
    constraints = []
    default_params = {}

    cls_name = type(model).__name__
    context = '{}{}'.format(context_prefix, cls_name)

    # Direct _citations on this class
    if hasattr(model, '_citations'):
        cit = model._citations
        for c in cit.get('definition', []):
            c = dict(c)
            c['_source_model'] = context
            citations.append(c)
        for pname, pinfo in cit.get('default_parameters', {}).items():
            key = '{}_{}'.format(context, pname)
            default_params[key] = dict(pinfo, _source_model=context)

    # Direct _validity_constraints
    if hasattr(model, '_validity_constraints'):
        for vc in model._validity_constraints:
            vc = dict(vc)
            vc['_source_model'] = context
            constraints.append(vc)

    # Recurse into sub-models for distributed / capped-cylinder composites
    if hasattr(model, 'models'):
        for sub in model.models:
            sub_c, sub_v, sub_d = _collect_from_model(
                sub, context_prefix=context + '.')
            citations.extend(sub_c)
            constraints.extend(sub_v)
            default_params.update(sub_d)

    # Internal cylinder/plane sub-models (capped cylinder)
    for attr in ('_cylinder_model', '_plane_model'):
        if hasattr(model, attr):
            sub = getattr(model, attr)
            sub_c, sub_v, sub_d = _collect_from_model(
                sub, context_prefix=context + '.')
            citations.extend(sub_c)
            constraints.extend(sub_v)
            default_params.update(sub_d)

    # Distribution wrapper
    if hasattr(model, 'distribution'):
        sub_c, sub_v, sub_d = _collect_from_model(
            model.distribution, context_prefix=context + '.')
        citations.extend(sub_c)
        constraints.extend(sub_v)
        default_params.update(sub_d)

    # Tortuosity citations/constraints (set by set_tortuous_parameter)
    if hasattr(model, '_tortuosity_citations'):
        for c in model._tortuosity_citations:
            c = dict(c)
            c['_source_model'] = context
            citations.append(c)
    if hasattr(model, '_tortuosity_constraints'):
        for vc in model._tortuosity_constraints:
            vc = dict(vc)
            vc['_source_model'] = context
            constraints.append(vc)

    # MT-CSD citations/constraints (set when S0_tissue_responses provided)
    if hasattr(model, '_mt_csd_citations'):
        for c in model._mt_csd_citations:
            c = dict(c)
            c['_source_model'] = context
            citations.append(c)
    if hasattr(model, '_mt_csd_constraints'):
        for vc in model._mt_csd_constraints:
            vc = dict(vc)
            vc['_source_model'] = context
            constraints.append(vc)

    return citations, constraints, default_params


def _deduplicate_citations(citations):
    """Deduplicate by DOI (or by key if DOI missing)."""
    seen = OrderedDict()
    for c in citations:
        doi = c.get('doi', '')
        key = doi if doi else c.get('key', id(c))
        if key not in seen:
            seen[key] = c
    return list(seen.values())


def _collect_fixed_parameters(model):
    """Collect information about fixed vs fitted parameters.

    Returns a dict mapping parameter_name -> {
        'status': 'fixed' | 'fitted' | 'passive',
        'value': fixed value (if fixed),
        'source_citation': citation dict (if value matches a biophysical constant),
    }
    """
    from .biophysical_constants import BIOPHYSICAL_CONSTANTS

    # Build reverse lookup: value -> constant entry
    value_lookup = {}
    for const_name, entry in BIOPHYSICAL_CONSTANTS.items():
        default = entry.get('default', {})
        val = default.get('value')
        if val is not None:
            value_lookup[(float(val), default.get('unit', ''))] = {
                'constant_name': const_name,
                'entry': entry,
            }

    param_info = OrderedDict()

    if not hasattr(model, 'parameter_optimization_flags'):
        return param_info

    # Collect parameters that are still in the optimization set
    for param_name, is_optimized in model.parameter_optimization_flags.items():
        info = {'name': param_name}

        if is_optimized:
            info['status'] = 'fitted'
        elif param_name.endswith('_T2') or param_name in ('T1', 'eta', 'S0_global'):
            x0_val = model.x0_parameters.get(param_name)
            if x0_val is not None:
                info['status'] = 'fixed'
                info['value'] = x0_val
            else:
                info['status'] = 'passive'
        else:
            info['status'] = 'fixed'
            x0_val = model.x0_parameters.get(param_name)
            if x0_val is not None:
                info['value'] = x0_val

        param_info[param_name] = info

    # Collect parameters that were fixed via set_fixed_parameter
    # (removed from parameter_ranges, stored as parameter_links)
    if hasattr(model, 'parameter_links'):
        for link in model.parameter_links:
            link_model, link_name, link_func, link_args = link
            # ReturnFixedValue stores the value as .value
            if hasattr(link_func, 'value'):
                # Reconstruct the parameter name
                if hasattr(model, '_inverted_parameter_map'):
                    param_name = model._inverted_parameter_map.get(
                        (link_model, link_name), link_name)
                else:
                    param_name = link_name
                info = {
                    'name': param_name,
                    'status': 'fixed',
                    'value': link_func.value,
                }
                param_info[param_name] = info

    # For all fixed params, try to match value to a biophysical constant
    for pname, info in param_info.items():
        if info.get('value') is not None and info.get('status') == 'fixed':
            val = info['value']
            if hasattr(val, '__float__'):
                val_f = float(val)
                for (const_val, const_unit), const_info in value_lookup.items():
                    if abs(val_f - const_val) / max(abs(const_val), 1e-30) < 0.01:
                        entry = const_info['entry']
                        info['source_constant'] = const_info['constant_name']
                        info['source_citation'] = entry.get('citation', {})
                        info['source_location'] = entry.get('default', {}).get('location', '')
                        break

    return param_info


def walk_citation_graph(model):
    """Walk a composed model tree, collect all citations and constraints.

    Parameters
    ----------
    model : dmipy model instance
        Any signal model, distributed model, or MultiCompartmentModel.
        If the model has been fit (via model.fit()), the result also includes
        optimizer citations and fixed/fitted parameter status.

    Returns
    -------
    dict with keys:
        'citations' : list of citation dicts (deduplicated by DOI)
        'constraints' : list of constraint dicts with source model context
        'default_parameters' : dict of parameter source citations
        'optimizers' : list of optimizer citations (populated if model was fit)
        'parameter_status' : OrderedDict of param_name -> {status, value, source}
        'solver' : str or None — name of the solver used for fitting
    """
    citations, constraints, default_params = _collect_from_model(model)
    citations = _deduplicate_citations(citations)

    # Check for optimizer citations (set by fitting)
    optimizer_citations = []
    if hasattr(model, '_optimizer_citations'):
        optimizer_citations = list(model._optimizer_citations)

    # Collect fixed/fitted parameter status
    param_status = _collect_fixed_parameters(model)

    # If any T2 parameter is actively fitted, cite TEdDI (Veraart et al. 2018)
    t2_active = any(
        k.endswith('_T2') and v.get('status') == 'fitted'
        for k, v in param_status.items()
    )
    if t2_active:
        citations.append({
            'key': 'veraart2018',
            'authors': 'Veraart J, Novikov DS, Fieremans E',
            'title': 'TE dependent Diffusion Imaging (TEdDI) distinguishes '
                     'between compartmental T2 relaxation times',
            'journal': 'NeuroImage',
            'year': 2018,
            'doi': '10.1016/j.neuroimage.2017.09.030',
        })
        citations = _deduplicate_citations(citations)
        constraints.append({
            'id': 'per_compartment_t2',
            'name': 'Per-compartment T2 relaxation',
            'condition_human': 'Each compartment has its own T2 relaxation '
                               'time, fitted jointly with diffusion parameters '
                               'from multi-TE data. The signal model is '
                               'S = sum_i(f_i * exp(-TE/T2_i) * E_diff_i(b)). '
                               'Requires acquisition with multiple echo times.',
            'severity': 'info',
            'source_key': 'veraart2018',
            '_source_model': 'MultiCompartmentModel',
        })

    # Solver name
    solver = getattr(model, '_fit_solver', None)

    return {
        'citations': citations,
        'constraints': constraints,
        'default_parameters': default_params,
        'optimizers': optimizer_citations,
        'parameter_status': param_status,
        'solver': solver,
    }
