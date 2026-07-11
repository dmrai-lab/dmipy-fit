"""Sanity checks for dmipy/core/constants.py.

These tests do not verify exact literature values to 10 decimal places —
they verify that the constants are in the right ballpark and have the
right physical units, catching obvious copy-paste errors or unit mistakes.
"""
import numpy as np
from dmipy_fit.core.constants import CONSTANTS


def test_gyromagnetic_ratio_units_and_magnitude():
    # Proton gyromagnetic ratio: ~267.522e6 rad/(s·T)
    # NIST value: 267.52218744e6
    gamma = CONSTANTS['water_gyromagnetic_ratio']
    assert 260e6 < gamma < 275e6, (
        f"gyromagnetic ratio {gamma:.3e} outside expected range")


def test_water_diffusion_constant_units_and_magnitude():
    # Free water diffusivity at body temperature: ~2.0–3.0 × 10⁻⁹ m²/s
    D = CONSTANTS['water_diffusion_constant']
    assert 1.5e-9 < D < 3.5e-9, (
        f"water diffusion constant {D:.3e} outside expected range")


def test_intra_axonal_diffusivity_less_than_free_water():
    # Intra-axonal water is more restricted than free water
    D_free = CONSTANTS['water_diffusion_constant']
    D_axon = CONSTANTS['water_in_axons_diffusion_constant']
    assert D_axon < D_free, (
        "intra-axonal diffusivity should be less than free water diffusivity")


def test_naa_diffusivity_much_less_than_water():
    # NAA diffuses much more slowly than water (~0.15 µm²/ms vs ~2.3 µm²/ms)
    D_water = CONSTANTS['water_diffusion_constant']
    D_naa = CONSTANTS['naa_in_axons']
    assert D_naa < D_water / 10, (
        "NAA diffusivity should be at least 10× slower than water")


def test_all_constants_positive():
    for name, value in CONSTANTS.items():
        assert value > 0, f"CONSTANTS['{name}'] = {value} is not positive"


def test_constants_are_floats():
    for name, value in CONSTANTS.items():
        assert isinstance(value, float), (
            f"CONSTANTS['{name}'] should be a float, got {type(value)}")
