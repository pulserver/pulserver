"""Test Opts parsing helper routine."""

import pytest
import pypulseq as pp

from pulserver import get_opts


def test_get_opts_default():
    """Test that get_opts returns default options when input is None."""
    opts = get_opts()
    assert isinstance(opts, pp.Opts)
    assert vars(opts) == vars(pp.Opts.default)


def test_get_opts_valid_string_gehc_premier():
    """Test valid GEHC scanner string input for Premier."""
    opts = get_opts("GEHC.3T.Premier")
    expected_params = {
        "max_grad": 70.0,
        "max_slew": 200.0,
        "rf_dead_time": 100e-6,
        "rf_ringdown_time": 60e-6,
        "adc_dead_time": 40e-6,
        "rf_raster_time": 2e-6,
        "adc_raster_time": 4e-6,
        "grad_raster_time": 4e-6,
        "B0": 3.0,
    }
    assert isinstance(opts, pp.Opts)
    assert vars(opts) == vars(
        pp.Opts(**expected_params, grad_unit="mT/m", slew_unit="T/m/s")
    )


def test_get_opts_valid_string_gehc_magnus():
    """Test valid GEHC scanner string input for Magnus."""
    opts = get_opts("GEHC.7T.Magnus")
    expected_params = {
        "max_grad": 300.0,
        "max_slew": 750.0,
        "rf_dead_time": 100e-6,
        "rf_ringdown_time": 60e-6,
        "adc_dead_time": 40e-6,
        "rf_raster_time": 2e-6,
        "adc_raster_time": 4e-6,
        "grad_raster_time": 4e-6,
        "B0": 7.0,
    }
    assert isinstance(opts, pp.Opts)
    assert vars(opts) == vars(
        pp.Opts(**expected_params, grad_unit="mT/m", slew_unit="T/m/s")
    )


def test_get_opts_dict_input():
    """Test passing a dictionary input directly to get_opts."""
    input_dict = {"max_grad": 20.0, "max_slew": 60, "B0": 3.0}
    opts = get_opts(input_dict)
    assert isinstance(opts, pp.Opts)
    assert vars(opts) == vars(
        pp.Opts(**input_dict, grad_unit="mT/m", slew_unit="T/m/s")
    )


def test_get_opts_unsupported_vendor():
    """Test that an unsupported vendor raises a NotImplementedError."""
    with pytest.raises(
        NotImplementedError,
        match="Currently, we only support GEHC identifier. Please provide a dict",
    ):
        get_opts("Siemens.3T.Skyra")


def test_get_opts_invalid_string():
    """Test that an invalid string format raises an error."""
    with pytest.raises(ValueError):
        get_opts("Invalid.String.Format")
