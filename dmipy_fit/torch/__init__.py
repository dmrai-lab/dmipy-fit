"""``dmipy_fit.torch``: the PyTorch backends (dmipy-fit#37), for hosts that run PyTorch only (Hugging Face's shared
GPU pool). Each module is one kernel of a JAX solver in torch, tested against it; torch is imported inside the
modules, so the package imports without it."""
