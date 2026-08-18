"""Loading the compiled kernels, and what happens when one is missing.

Pulserver builds :mod:`pulserver._ext` -- one module holding every kernel --
for every interpreter it supports, so a kernel that will not load is a broken
installation. The
property these tests pin is that such an installation says so: a caller must
never be handed a slower Python path in place of a kernel, because the only
symptom would be unexplained runtime.
"""

from __future__ import annotations

import sys

import pytest

from pulserver._accelerators import require

BUNDLED = [
    "pulseg",
    "pulseqpp",
    "arbgrad",
    "sampling",
    "recon_cpu",
]


@pytest.mark.parametrize("name", BUNDLED)
def test_every_bundled_accelerator_loads_for_this_interpreter(name):
    assert require(name) is not None


def test_a_missing_accelerator_raises_instead_of_degrading():
    with pytest.raises(ImportError) as caught:
        require("absent_kernel")
    assert "absent_kernel" in str(caught.value)


def test_the_failure_names_the_interpreter_it_could_not_match():
    with pytest.raises(ImportError) as caught:
        require("absent_kernel")
    assert sys.implementation.cache_tag in str(caught.value)


def test_a_binary_missing_a_symbol_fails_at_the_load_not_the_call():
    with pytest.raises(ImportError) as caught:
        require("recon_cpu", attribute="a_symbol_that_does_not_exist")
    assert "a_symbol_that_does_not_exist" in str(caught.value)


def test_an_attribute_that_exists_comes_back_callable():
    assert callable(require("recon_cpu", attribute="wave_psf"))
