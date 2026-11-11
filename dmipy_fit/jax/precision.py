"""Fitting precision a model requires under ``solver='jax'``.

A model declares ``required_precision`` ('float32' or 'float64'); a composed
model requires the strictest of its compartments. ``fit_precision`` fits a
float64 model in float64 and refuses, naming the compartment, when the
runtime cannot provide it.
"""
from contextlib import contextmanager

import jax
import numpy as np

_VALID = ('float32', 'float64')


def _compartments(model):
    """Yield every compartment of ``model``, descending into wrappers."""
    for m in getattr(model, 'models', None) or []:
        yield m
        yield from _compartments(m)
    inner = getattr(model, 'model', None)
    if inner is not None:
        yield inner
        yield from _compartments(inner)


def float64_compartments(model):
    """Names of the compartments of ``model`` that require float64."""
    names = []
    for m in _compartments(model):
        prec = m.required_precision
        if prec not in _VALID:
            raise ValueError("{} declares required_precision={!r}; expected "
                             "one of {}".format(type(m).__name__, prec, _VALID))
        if prec == 'float64':
            names.append(type(m).__name__)
    return names


@contextmanager
def fit_precision(model):
    """Context in which a JAX fit of ``model`` runs at its required precision.

    Yields the numpy dtype to fit in. For a float64 model it enables
    ``jax_enable_x64`` for the duration and raises ``ValueError`` naming the
    compartment if float64 is still unavailable; for any other model it
    leaves the ambient JAX configuration untouched.
    """
    names = float64_compartments(model)
    if not names:
        yield np.float32 if not jax.config.jax_enable_x64 else np.float64
        return
    with jax.enable_x64(True):
        if jax.dtypes.canonicalize_dtype(np.float64) != np.float64:
            raise ValueError(
                "solver='jax' cannot fit {} in float32: it requires float64 "
                "and this JAX runtime does not provide it".format(
                    ', '.join(sorted(set(names)))))
        yield np.float64
