"""Fit-method decorator: ``solver='jax'`` runs at the model's required precision."""
import functools
import inspect


def at_required_precision(fit):
    """Run ``fit`` inside ``dmipy_fit.jax.precision.fit_precision(self)`` when ``solver='jax'``.

    JAX is imported only for that solver, so other solvers need no JAX install.
    """
    sig = inspect.signature(fit)

    @functools.wraps(fit)
    def wrapper(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        if bound.arguments['solver'] != 'jax':
            return fit(self, *args, **kwargs)
        from ..jax.precision import fit_precision
        with fit_precision(self):
            return fit(self, *args, **kwargs)
    return wrapper
