"""Every published reference model constructs — the catalog can't rot.

Enumerates all factory functions in reference_models (the compendium behind the
dmipy.org Model catalog page) and asserts each returns a configured, parameterised
model. Construction only (no fit) — fast and deterministic.
"""
import inspect
import pytest

from dmipy_fit.custom_optimizers import reference_models as rm

FACTORIES = sorted(
    n for n in dir(rm)
    if not n.startswith('_') and callable(getattr(rm, n))
    and getattr(getattr(rm, n), '__module__', '') == rm.__name__
)


def test_catalog_nonempty():
    assert len(FACTORIES) >= 30, FACTORIES


@pytest.mark.parametrize("name", FACTORIES)
def test_reference_model_constructs(name):
    model = getattr(rm, name)()
    # a configured MultiCompartment(-spherical-mean) model exposes parameter_names
    assert hasattr(model, 'parameter_names')
    assert len(model.parameter_names) >= 1, (name, model.parameter_names)


import inspect
import re

# Models whose primary reference is a conference abstract with no DOI.
_NO_DOI_ALLOWLIST = {"mte_sandi"}

_DOI_RE = re.compile(r'(?:doi:|https?://doi\.org/)\s*(10\.\d{4,}/\S+)')


@pytest.mark.parametrize("name", FACTORIES)
def test_reference_has_year(name):
    """Every model docstring cites a source (a 19xx/20xx year) — no danging models."""
    doc = inspect.getdoc(getattr(rm, name)) or ""
    assert re.search(r'\b(19|20)\d{2}\b', doc), f"{name}: no citation year in docstring"


@pytest.mark.parametrize("name", sorted(set(FACTORIES) - _NO_DOI_ALLOWLIST))
def test_reference_has_wellformed_doi(name):
    """Every model (except known abstracts) carries a well-formed DOI in its docstring,
    so the dmipy.org catalog can render a resolvable link (not a possibly-wrong free-text
    reference)."""
    doc = inspect.getdoc(getattr(rm, name)) or ""
    m = _DOI_RE.search(doc)
    assert m, f"{name}: no DOI in docstring (add `doi:10.xxxx/...` or allowlist it)"
    doi = m.group(1).rstrip('.,);')
    assert re.fullmatch(r'10\.\d{4,}/\S+', doi), f"{name}: malformed DOI {doi!r}"
