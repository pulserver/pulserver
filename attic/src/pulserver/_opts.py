"""System configuration helper."""

__all__ = ["get_opts"]

import pypulseq as pp


def get_opts(input: str | dict | pp.opts.Opts | None = None) -> pp.Opts:
    """
    Initialize system hardware specifications structure.

    Parameters
    ----------
    input : str | dict | None, optional
        Either scanner identifier or a dictionary with the following keys:

            * max_grad: maximum gradient strength in ``[mT/m]``.
            * max_slew: maximum gradient slew rate in ``[T/m/s]``.
            * rf_dead_time: initial RF delay time in ``[s]``.
            * rf_ringdown_time: final RF wait time in ``[s]``.
            * adc_dead_time: initial ADC delay time in ``[s]``.
            * adc_raster_time: ADC raster time (i.e., signal dwell time) in ``[s]``.
            * rf_raster_time: RF raster time in ``[s]``.
            * grad_raster_time: gradient raster time in ``[s]``.
            * B0: field strength in ``[T]``

        If ``None``, use PyPulseq default Opts. The default is ``None``.


    Returns
    -------
    pp.Opts
        PyPulseq structure containing system specifications.

    Notes
    -----
    The expected format for scanner identifier (i.e., ``type(input) == str``)
    is ``{vendor}.{B0}T.{scanner_model}``. At the moment, valid values are:

    * vendor (case-insensitive): ``"gehc"``.
    * B0: ``(1.5, 3, 3.0, 7, 7.0)``
    * scanner_model (case-insensitive):

        - ``"MR750w"``
        - ``"MR750"``
        - ``"HDx-whole"``
        - ``"HDx-zoom"``
        - ``"UHP"``
        - ``"Premier"``
        - ``"Magnus"``

    Example
    -------
    >>> import pulserver

    >>> opts1 = pulserver.get_opts("GEHC.3T.Premier")
    >>> opts2 = pulserver.get_opts("gehc.3.0T.magnus")
    >>> opts3 = pulserver.get_opts("gehc.3.0T.magnus")
    >>> opts4 = pulserver.get_opts({"max_grad": 20.0, "max_slew": 60})

    """
    if input is None:
        return pp.Opts.default

    if isinstance(input, pp.opts.Opts):
        return input

    # split
    if isinstance(input, str):
        # get vendor, field strength and model ID
        parts = input.lower().split(".")
        vendor = parts[0]
        field_strength = float(".".join(parts[1:-1])[:-1])
        model = parts[-1]

        # define common HW specs for Siemens/GEHC
        _common_hw = {
            "rf_dead_time": 100e-6,
            "rf_ringdown_time": 60e-6,
            "adc_dead_time": 40e-6,
        }

        # get platform-specific HW specs
        if vendor == "gehc":
            input = (
                _gehc_hw | _common_hw | _scanner_model(model) | {"B0": field_strength}
            )
        elif vendor == "siemens":
            raise NotImplementedError(
                "Currently, we only support GEHC identifier. Please provide a dict"
            )

    return pp.Opts(**input, grad_unit="mT/m", slew_unit="T/m/s")


# %% local sub-routines
_siemens_hw = {
    "rf_raster_time": 1e-6,
    "adc_raster_time": 0.1e-6,
    "grad_raster_time": 10e-6,
}

_gehc_hw = {
    "rf_raster_time": 2e-6,
    "adc_raster_time": 4e-6,
    "grad_raster_time": 4e-6,
}


def _scanner_model(model):
    if model.lower() == "mr750w":
        return {"max_grad": 33.0, "max_slew": 120.0}
    if model.lower() == "mr750":
        return {"max_grad": 50.0, "max_slew": 120.0}
    if model.lower() == "hdx-whole":
        return {"max_grad": 23.0, "max_slew": 77.0}
    if model.lower() == "hdx-zoom":
        return {"max_grad": 40.0, "max_slew": 150.0}
    if model.lower() == "uhp":
        return {"max_grad": 100.0, "max_slew": 200.0}
    if model.lower() == "premier":
        return {"max_grad": 70.0, "max_slew": 200.0}
    if model.lower() == "signa":
        return {"max_grad": 70.0, "max_slew": 200.0}
    if model.lower() == "magnus":
        return {"max_grad": 300.0, "max_slew": 750.0}
