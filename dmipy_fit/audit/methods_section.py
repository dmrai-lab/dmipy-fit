"""Methods-section and BibTeX generation from citation graph results."""

from . import biophysical_constants as _bp_mod
from .biophysical_constants import BIOPHYSICAL_CONSTANTS

# Reverse lookup: source_key -> citation dict (for author/year formatting).
# Built from (1) main citations in BIOPHYSICAL_CONSTANTS, and (2) module-level
# _CITATION_* dicts that are shared across entries.
_SOURCE_KEY_CITATIONS = {}
for _entry in BIOPHYSICAL_CONSTANTS.values():
    cit = _entry.get('citation', {})
    if 'key' in cit:
        _SOURCE_KEY_CITATIONS[cit['key']] = cit
# Also pick up module-level shared citation dicts (e.g. _CITATION_BARAKOVIC2023)
for _name in dir(_bp_mod):
    if _name.startswith('_CITATION_'):
        _cit_obj = getattr(_bp_mod, _name)
        if isinstance(_cit_obj, dict) and 'key' in _cit_obj:
            _SOURCE_KEY_CITATIONS[_cit_obj['key']] = _cit_obj


def _format_author_year(citation):
    """Format 'Author et al. (YEAR)' from a citation dict."""
    authors = citation.get('authors', 'Unknown')
    year = citation.get('year', '')
    # Shorten to first author + et al. if comma present
    if ',' in authors:
        first = authors.split(',')[0].strip()
        return '{} et al. ({})'.format(first, year)
    return '{} ({})'.format(authors, year)


def _format_reference_line(i, citation):
    """Format a numbered reference line."""
    authors = citation.get('authors', 'Unknown')
    year = citation.get('year', '')
    title = citation.get('title', '')
    journal = citation.get('journal', '')
    doi = citation.get('doi', '')
    parts = []
    if authors:
        parts.append(authors)
    if year:
        parts.append('({})'.format(year))
    if title:
        parts.append(title + '.')
    if journal:
        parts.append(journal + '.')
    if doi:
        parts.append('DOI: {}'.format(doi))
    return '{}. {}'.format(i + 1, ' '.join(parts))


def generate_methods_section(graph_result, acquisition_scheme=None):
    """Generate a Markdown methods section from the citation graph.

    Parameters
    ----------
    graph_result : dict
        Output of walk_citation_graph().
    acquisition_scheme : optional
        If provided, used for constraint evaluation context.

    Returns
    -------
    str
        Markdown-formatted methods section.
    """
    citations = graph_result.get('citations', [])
    constraints = graph_result.get('constraints', [])
    optimizers = graph_result.get('optimizers', [])

    lines = []
    lines.append('## Diffusion Model')
    lines.append('')

    if citations:
        # Model description paragraph
        model_refs = ', '.join(
            _format_author_year(c) for c in citations)
        lines.append(
            'The diffusion signal was modeled using compartment models '
            'described in the following references: {}.'.format(model_refs))
        lines.append('')

    # Default parameters
    default_params = graph_result.get('default_parameters', {})
    if default_params:
        lines.append('### Parameter Values')
        lines.append('')
        for pname, pinfo in default_params.items():
            val = pinfo.get('value', '')
            unit = pinfo.get('unit', '')
            src_key = pinfo.get('source_key', '')

            # Look up the source citation for author/year formatting
            src_cit = _SOURCE_KEY_CITATIONS.get(src_key, {})
            if src_cit:
                src_ref = _format_author_year(src_cit)
            else:
                src_ref = src_key

            # Build the parameter line
            param_text = '{} = {} {}'.format(pname, val, unit).strip()
            param_text += ' [{}'.format(src_ref)

            # Check if there's a matching biophysical constant with alternatives
            field_str = ''
            if src_cit.get('year'):
                # Try to find field_T from the biophysical constant
                for bc_entry in BIOPHYSICAL_CONSTANTS.values():
                    if bc_entry.get('default', {}).get('source_key') == src_key:
                        field_t = bc_entry['default'].get('field_T')
                        if field_t is not None:
                            field_str = ', {}T'.format(field_t)
                        # Append alternatives
                        alts = bc_entry.get('alternatives', [])
                        if alts:
                            alt_parts = []
                            for alt in alts:
                                alt_val = alt.get('value', '')
                                alt_unit = alt.get('unit', unit)
                                alt_ft = alt.get('field_T')
                                alt_sk = alt.get('source_key', '')
                                alt_cit = _SOURCE_KEY_CITATIONS.get(alt_sk, {})
                                if alt_cit:
                                    alt_ref = _format_author_year(alt_cit)
                                else:
                                    alt_ref = alt_sk
                                alt_text = '{} {} '.format(alt_val, alt_unit).strip()
                                if alt_ft is not None:
                                    alt_text += ' at {}T'.format(alt_ft)
                                alt_text += ' [{}]'.format(alt_ref)
                                alt_parts.append(alt_text)
                            if alt_parts:
                                param_text += field_str + '] (see also: {})'.format(
                                    '; '.join(alt_parts))
                                break
                        else:
                            param_text += field_str + ']'
                            break
                else:
                    param_text += ']'
            else:
                param_text += ']'

            lines.append('- ' + param_text)
        lines.append('')

    # Parameter status (fixed vs fitted)
    param_status = graph_result.get('parameter_status', {})
    if param_status:
        fixed_params = {k: v for k, v in param_status.items()
                        if v.get('status') == 'fixed' and v.get('value') is not None}
        fitted_params_list = [k for k, v in param_status.items()
                              if v.get('status') == 'fitted']
        passive_params = [k for k, v in param_status.items()
                          if v.get('status') == 'passive']

        if fixed_params:
            lines.append('### Fixed Parameters')
            lines.append('')
            for pname, pinfo in fixed_params.items():
                val = pinfo['value']
                src_cit = pinfo.get('source_citation', {})
                src_const = pinfo.get('source_constant', '')
                if src_cit:
                    src_ref = _format_author_year(src_cit)
                    loc = pinfo.get('source_location', '')
                    loc_str = ' ({})'.format(loc) if loc else ''
                    lines.append('- {} = {} [{}{}]'.format(
                        pname, val, src_ref, loc_str))
                else:
                    lines.append('- {} = {} (user-specified)'.format(pname, val))
            lines.append('')

        if fitted_params_list:
            lines.append('### Fitted Parameters')
            lines.append('')
            lines.append(', '.join(fitted_params_list))
            lines.append('')

    # Optimizer section
    solver = graph_result.get('solver')
    if optimizers or solver:
        lines.append('### Optimization')
        lines.append('')
        if solver:
            lines.append('Solver: {}'.format(solver))
        if optimizers:
            opt_refs = ', '.join(_format_author_year(c) for c in optimizers)
            lines.append('Parameter estimation was performed using: {}.'.format(
                opt_refs))
        lines.append('')

    # Validity constraints
    if constraints:
        lines.append('### Validity Constraints')
        lines.append('')
        lines.append('| Constraint | Model | Severity | Detail |')
        lines.append('|---|---|---|---|')
        for vc in constraints:
            cname = vc.get('name', vc.get('id', ''))
            model_ctx = vc.get('_source_model', '')
            severity = vc.get('severity', 'info')
            detail = vc.get('condition_human', '')
            lines.append('| {} | {} | {} | {} |'.format(
                cname, model_ctx, severity.upper(), detail))
        lines.append('')

    # References
    all_citations = list(citations) + list(optimizers)
    if all_citations:
        lines.append('### References')
        lines.append('')
        for i, c in enumerate(all_citations):
            lines.append(_format_reference_line(i, c))
        lines.append('')

    return '\n'.join(lines)


def _citation_to_bibtex_key(citation):
    """Generate a BibTeX key from a citation dict."""
    return citation.get('key', 'unknown')


def generate_bibtex(graph_result):
    """Generate BibTeX entries for all citations in the graph.

    Parameters
    ----------
    graph_result : dict
        Output of walk_citation_graph().

    Returns
    -------
    str
        BibTeX-formatted string.
    """
    all_citations = (
        graph_result.get('citations', []) +
        graph_result.get('optimizers', []))

    entries = []
    seen_keys = set()
    for c in all_citations:
        bkey = _citation_to_bibtex_key(c)
        if bkey in seen_keys:
            continue
        seen_keys.add(bkey)

        entry_lines = ['@article{{{},'.format(bkey)]
        if c.get('authors'):
            entry_lines.append('  author = {{{}}},'.format(c['authors']))
        if c.get('title'):
            entry_lines.append('  title = {{{}}},'.format(c['title']))
        if c.get('journal'):
            entry_lines.append('  journal = {{{}}},'.format(c['journal']))
        if c.get('year'):
            entry_lines.append('  year = {{{}}},'.format(c['year']))
        if c.get('doi'):
            entry_lines.append('  doi = {{{}}},'.format(c['doi']))
        entry_lines.append('}')
        entries.append('\n'.join(entry_lines))

    return '\n\n'.join(entries) + '\n' if entries else ''
