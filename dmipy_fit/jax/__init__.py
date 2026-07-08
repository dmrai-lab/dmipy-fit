"JAX-based forward modeling and fitting for dmipy."

import os as _os

# XLA's Triton-GEMM autotuner fails to find a valid config for the small-
# contracting-dimension matmuls in the CSD/OSQP solver and the dispersed-model
# forward on some GPUs ("Too small divisible part of the contracting
# dimension"), which aborts compilation of solver='jax'/'csd_jax'. Falling back
# to cuBLAS for GEMM compiles correctly and is numerically equivalent. Set
# before the JAX backend initialises; only appended if the user has not already
# configured Triton GEMM themselves.
_xla_flags = _os.environ.get('XLA_FLAGS', '')
if 'triton_gemm' not in _xla_flags:
    _os.environ['XLA_FLAGS'] = (
        _xla_flags + ' --xla_gpu_enable_triton_gemm=false').strip()

from .losses_jax import mse_loss, rician_nll, rician_nll_fittable, nc_chi_nll
from .dti_jax import build_dti_fitter, detect_mu_indices
from .convergence import (analyze_convergence,
                           SDMStratifiedSampler,
                           ThreeTissueSampler,
                           MaskSampler)
