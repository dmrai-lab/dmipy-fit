"""``dmipy_fit.torch``: the PyTorch backends, for hosts that run PyTorch only (Hugging Face's shared GPU pool): the
batched Tournier CSD (``csd_tournier_torch``, tested against its JAX twin), the batched multi-tissue CSD
(``csd_msmt_torch``, tested against cvxpy) and the tensor fit (``dti_torch``, tested against dipy). torch is imported
inside the modules, so the package imports without it."""
