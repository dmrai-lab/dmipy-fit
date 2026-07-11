"""Biophysical constants with full citation provenance.

SINGLE SOURCE: the catalogue now lives in dmipy-sim (the forward truth) at
:mod:`dmipy_sim.substrate.biophysical_constants`.  This module re-exports it
verbatim so the historical ``dmipy_fit.audit.biophysical_constants`` import path
keeps working for the citation/methods machinery and the signal models, while
there is exactly one definition of every physical constant.

The full module namespace (the public ``BIOPHYSICAL_CONSTANTS`` /
``canonical_white_matter`` / ``get_value`` / ``get_constant`` /
``get_default_value`` AND the module-level ``_CITATION_*`` dicts that
``methods_section`` introspects via ``dir()``) is copied in below.
"""
from dmipy_sim.substrate import biophysical_constants as _src

# Re-export every name from the sim catalogue (public + the _CITATION_* dicts and
# helpers the audit tooling introspects), so this module is a faithful alias.
_EXCLUDE = ('__name__', '__loader__', '__spec__', '__package__', '__file__',
            '__builtins__', '__cached__', '__doc__')
globals().update({_k: _v for _k, _v in vars(_src).items() if _k not in _EXCLUDE})
del _src, _EXCLUDE
