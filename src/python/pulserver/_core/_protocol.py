"""Read and write protocol values, and resolve the quantities derived from them.

Reached as ``pulserver.params``. Pulling a number out of a protocol means
indexing by key, unwrapping the parameter object and coercing its value —
repeated at every use site and easy to get subtly wrong. These accessors do it
once, by canonical key, with the type the key declares, and the ``*_optional``
forms fall back to a default when a key is absent.

The module also resolves the quantities a plugin would otherwise recompute
from several entries at a time: the phase field of view, the ACS block size,
and which physical gradient axes the readout and phase encode map onto after
``swap_phase_freq``.

Examples
--------
>>> from pulserver import TypeinFloatParam, UIParam, params
>>> protocol = {UIParam.TR: TypeinFloatParam(value=500.0, unit="ms")}
>>> params.param_float(protocol, UIParam.TR)
500.0
>>> params.param_float_optional(protocol, UIParam.TE, 8.0)
8.0

Use it inside a plugin's two protocol-reading methods::

    def make_sequence(self, system, protocol, output_path):
        tr_s = params.param_float(protocol, UIParam.TR) * 1e-3
        ro_axis, pe_axis = params.resolve_readout_phase_axes(protocol)

See Also
--------
pulserver.protocol_to_dict : produce the serialized form these read.
pulserver.set_protocol_value : write a value back, dropdowns included.
"""

from __future__ import annotations

import inspect

from pulserver import UIParam

from ._params import set_protocol_value as _set_protocol_value


def param_float(protocol: dict, key: UIParam) -> float:
    """Read a required protocol entry as a float.

    Parameters
    ----------
    protocol : dict
        Protocol dictionary keyed by ``str(UIParam)``.
    key : UIParam
        Parameter to read.

    Returns
    -------
    float
        The entry's value.

    Examples
    --------
    >>> from pulserver import UIParam
    >>> from pulserver import params
    >>> from types import SimpleNamespace
    >>> protocol = {str(UIParam.TE): SimpleNamespace(value=5.0)}
    >>> params.param_float(protocol, UIParam.TE)
    5.0
    """
    return float(protocol[str(key)].value)


def param_int(protocol: dict, key: UIParam) -> int:
    """Read a required protocol entry as an int.

    Parameters
    ----------
    protocol : dict
        Protocol dictionary keyed by ``str(UIParam)``.
    key : UIParam
        Parameter to read.

    Returns
    -------
    int
        The entry's value.

    Examples
    --------
    >>> from pulserver import UIParam
    >>> from pulserver import params
    >>> from types import SimpleNamespace
    >>> protocol = {str(UIParam.NX): SimpleNamespace(value=128)}
    >>> params.param_int(protocol, UIParam.NX)
    128
    """
    return int(protocol[str(key)].value)


def param_float_optional(protocol: dict, key: UIParam | str, default: float) -> float:
    """Read a protocol entry as a float, falling back to ``default``.

    Parameters
    ----------
    protocol : dict
        Protocol dictionary keyed by ``str(UIParam)``.
    key : UIParam or str
        Parameter to read.
    default : float
        Value returned when the entry is absent.

    Returns
    -------
    float
        The entry's value, or ``default`` when missing.

    Examples
    --------
    >>> from pulserver import UIParam
    >>> from pulserver import params
    >>> params.param_float_optional({}, UIParam.TE, 3.0)
    3.0
    """
    name = str(key)
    if name not in protocol:
        return default
    return float(protocol[name].value)


def param_int_optional(protocol: dict, key: UIParam | str, default: int) -> int:
    """Read a protocol entry as an int, falling back to ``default``.

    Parameters
    ----------
    protocol : dict
        Protocol dictionary keyed by ``str(UIParam)``.
    key : UIParam or str
        Parameter to read.
    default : int
        Value returned when the entry is absent.

    Returns
    -------
    int
        The entry's value, or ``default`` when missing.

    Examples
    --------
    >>> from pulserver import UIParam
    >>> from pulserver import params
    >>> params.param_int_optional({}, UIParam.NX, 64)
    64
    """
    name = str(key)
    if name not in protocol:
        return default
    return int(protocol[name].value)


def param_bool_optional(protocol: dict, key: UIParam | str, default: bool) -> bool:
    """Read a protocol entry as a bool, falling back to ``default``.

    Parameters
    ----------
    protocol : dict
        Protocol dictionary keyed by ``str(UIParam)``.
    key : UIParam or str
        Parameter to read.
    default : bool
        Value returned when the entry is absent.

    Returns
    -------
    bool
        The entry's value, or ``default`` when missing.

    Examples
    --------
    >>> from pulserver import UIParam
    >>> from pulserver import params
    >>> params.param_bool_optional({}, UIParam.SWAP_PHASE_FREQ, False)
    False
    """
    name = str(key)
    if name not in protocol:
        return default
    return bool(protocol[name].value)


def user_float(protocol: dict, slot: int, default: float) -> float:
    """Read the user-CV float in ``slot``, falling back to ``default``.

    Parameters
    ----------
    protocol : dict
        Protocol dictionary keyed by ``str(UIParam)``.
    slot : int
        User-value slot number (``UIParam.user_value(slot)``).
    default : float
        Value returned when the slot is absent.

    Returns
    -------
    float
        The slot's value, or ``default`` when missing.

    Examples
    --------
    >>> from pulserver import params
    >>> params.user_float({}, 0, 1.5)
    1.5
    """
    return param_float_optional(protocol, UIParam.user_value(slot), default)


def phase_fov_mm_from_protocol(protocol: dict) -> float:
    """Resolve the phase-encode FOV in millimetres.

    ``PHASE_FOV`` may be stored either as an absolute FOV in mm or as a
    fraction of the readout FOV (values ``<= 1.5`` are treated as a
    fraction, per the GE UI convention).

    Parameters
    ----------
    protocol : dict
        Protocol dictionary keyed by ``str(UIParam)``.

    Returns
    -------
    float
        Phase-encode FOV in mm.

    Examples
    --------
    >>> from pulserver import UIParam
    >>> from pulserver import params
    >>> from types import SimpleNamespace
    >>> protocol = {str(UIParam.FOV): SimpleNamespace(value=240.0)}
    >>> params.phase_fov_mm_from_protocol(protocol)
    240.0
    """
    fov_mm = param_float(protocol, UIParam.FOV)
    phase_fov = param_float_optional(protocol, UIParam.PHASE_FOV, fov_mm)
    if phase_fov <= 1.5:
        return max(0.0, phase_fov * fov_mm)
    return phase_fov


def acs_lines_from_protocol(protocol: dict, ny_pe: int, slot: int) -> int:
    """Read the autocalibration (ACS) line count from a user-CV slot.

    Parameters
    ----------
    protocol : dict
        Protocol dictionary keyed by ``str(UIParam)``.
    ny_pe : int
        Number of phase-encode lines; the ACS count is clamped to this.
    slot : int
        User-value slot carrying the ACS count.

    Returns
    -------
    int
        ACS line count in ``[0, ny_pe]``.

    Examples
    --------
    >>> from pulserver import params
    >>> params.acs_lines_from_protocol({}, 128, 0)
    0
    """
    raw = user_float(protocol, slot, 0.0)
    acs = round(raw)
    if acs < 0:
        return 0
    return min(acs, ny_pe)


def resolve_readout_phase_axes(protocol: dict) -> tuple[str, str]:
    """Resolve the readout/phase gradient axes from the swap flag.

    Parameters
    ----------
    protocol : dict
        Protocol dictionary keyed by ``str(UIParam)``.

    Returns
    -------
    ro_axis : str
        Readout gradient channel (``"x"`` or ``"y"``).
    pe_axis : str
        Phase-encode gradient channel (``"y"`` or ``"x"``).

    Examples
    --------
    >>> from pulserver import params
    >>> params.resolve_readout_phase_axes({})
    ('x', 'y')
    """
    ro_axis = "x"
    pe_axis = "y"
    if param_bool_optional(protocol, UIParam.SWAP_PHASE_FREQ, False):
        ro_axis, pe_axis = "y", "x"
    return ro_axis, pe_axis


def set_protocol_value(protocol: dict, key: UIParam | str, value) -> None:
    """Set a protocol entry's value (and dropdown index when applicable).

    Parameters
    ----------
    protocol : dict
        Protocol dictionary (dict-of-dicts form) keyed by ``str(UIParam)``.
    key : UIParam or str
        Parameter to set.
    value : object
        New value; for ``stringlist`` entries it must be one of the options.

    Returns
    -------
    None
        ``protocol`` is modified in place.

    Examples
    --------
    >>> from pulserver import UIParam
    >>> from pulserver import params
    >>> protocol = {str(UIParam.TE): {"value": 5.0, "type": "float"}}
    >>> params.set_protocol_value(protocol, UIParam.TE, 7.0)
    >>> protocol[str(UIParam.TE)]["value"]
    7.0
    """
    _set_protocol_value(protocol, key, value)


# ---------------------------------------------------------------------------
# Protocol to main() arguments
# ---------------------------------------------------------------------------

#: How to read each argument name a sequence's ``main`` might declare, in the
#: unit ``main`` states it in -- metres and seconds, against the protocol's
#: millimetres and milliseconds.
#:
#: One entry per quantity the scanner prescribes, which is what makes this
#: shared: a plugin's own controls live in its user slots and are passed to
#: :func:`main_kwargs` as overrides instead.
_PROTOCOL_ARGUMENTS = {
    "fov": lambda p: (
        param_float(p, UIParam.FOV) * 1e-3,
        phase_fov_mm_from_protocol(p) * 1e-3,
    ),
    "fov_offset": lambda p: tuple(
        param_float_optional(p, key, 0.0) * 1e-3
        for key in (UIParam.FOV_OFFSET_X, UIParam.FOV_OFFSET_Y, UIParam.FOV_OFFSET_Z)
    ),
    "n_x": lambda p: param_int(p, UIParam.NX),
    "n_y": lambda p: param_int(p, UIParam.NY),
    "n_slices": lambda p: param_int(p, UIParam.NSLICES),
    "n_slabs": lambda p: param_int(p, UIParam.NUM_SLABS),
    "n_echoes": lambda p: param_int(p, UIParam.NUM_ECHOES),
    "n_frames": lambda p: param_int(p, UIParam.NUM_FRAMES),
    "n_shots": lambda p: param_int(p, UIParam.NUM_SHOTS),
    "etl": lambda p: param_int(p, UIParam.ETL),
    "n_averages": lambda p: max(1, round(param_float_optional(p, UIParam.NEX, 1.0))),
    "slice_thickness": lambda p: param_float(p, UIParam.SLICE_THICKNESS) * 1e-3,
    "slice_gap": lambda p: max(
        0.0,
        (
            param_float(p, UIParam.SLICE_SPACING)
            - param_float(p, UIParam.SLICE_THICKNESS)
        )
        * 1e-3,
    ),
    "flip_angle_deg": lambda p: param_float(p, UIParam.FLIP),
    "te": lambda p: _time_or_preset(param_float(p, UIParam.TE)),
    "te2": lambda p: _time_or_preset(param_float(p, UIParam.TE2)),
    "tr": lambda p: _time_or_preset(param_float(p, UIParam.TR)),
    "recovery_time": lambda p: _time_or_preset(param_float(p, UIParam.TRECOVERY)),
    "readout_bandwidth_hz": lambda p: param_float_optional(p, UIParam.BANDWIDTH, 250e3),
    "acceleration": lambda p: max(1, round(param_float_optional(p, UIParam.RY, 1.0))),
    "acceleration_z": lambda p: max(1, round(param_float_optional(p, UIParam.RZ, 1.0))),
    "swap_phase_freq": lambda p: param_bool_optional(p, UIParam.SWAP_PHASE_FREQ, False),
}


def _typed(protocol: dict) -> dict:
    """The protocol as parameter objects, whichever form it arrived in.

    The bridge hands over the serialized form and a plugin building its own
    protocol holds the typed one; the accessors here take the typed one.
    """
    from ._params import dict_to_param

    return {
        key: dict_to_param(value) if isinstance(value, dict) else value
        for key, value in protocol.items()
    }


def _time_or_preset(milliseconds: float) -> float | None:
    """Seconds, or ``None`` for the presets a UI shows as words.

    :class:`pulserver.TEPreset` and :class:`pulserver.TRPreset` are negative
    numbers standing for "whatever the sequence can reach", which is what every
    readout module means by ``None``.
    """
    return None if milliseconds <= 0.0 else milliseconds * 1e-3


def main_kwargs(main, system, protocol: dict, **overrides) -> dict:
    """Translate a scanner protocol into keyword arguments for ``main``.

    Reads what ``main`` declares rather than what the protocol carries: every
    keyword-only parameter whose name names a prescribed quantity is filled in
    from the protocol, in the unit ``main`` states it in, and every other
    parameter is left to its own default. A plugin's own controls -- the ones
    living in its user slots, which only it can interpret -- are passed as
    ``**overrides``.

    Parameters
    ----------
    main : callable
        The sequence's entry point, read for its signature.
    system : pypulseq.Opts
        System limits, passed on as ``system`` if ``main`` takes one.
    protocol : dict
        The protocol as the bridge hands it over, or as
        :func:`pulserver.protocol_to_dict` produces it.
    **overrides
        Values that win over anything read from the protocol. Names ``main``
        does not declare raise, rather than being passed to a call that would
        reject them further down.

    Returns
    -------
    dict
        Keyword arguments to call ``main`` with.

    Raises
    ------
    TypeError
        If an override names a parameter ``main`` does not take.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> from pulserver import TypeinFloatParam, UIParam, main_kwargs, protocol_to_dict
    >>> def main(*, system=None, te=8e-3, n_acs=24): ...
    >>> protocol = protocol_to_dict({UIParam.TE: TypeinFloatParam(value=5.0, unit="ms")})
    >>> kwargs = main_kwargs(main, pp.Opts(), protocol, n_acs=8)
    >>> kwargs["te"], kwargs["n_acs"]
    (0.005, 8)

    See Also
    --------
    pulserver.run_cli : the offline entry point that feeds this a protocol.
    """
    declared = set(inspect.signature(main).parameters)
    unknown = set(overrides) - declared
    if unknown:
        raise TypeError(
            f"main_kwargs(): {sorted(unknown)} names no parameter of "
            f"{getattr(main, '__name__', main)}; it takes {sorted(declared)}"
        )

    values = _typed(protocol)
    kwargs: dict = {}
    if "system" in declared:
        kwargs["system"] = system
    for name, read in _PROTOCOL_ARGUMENTS.items():
        if name in declared and name not in overrides:
            try:
                kwargs[name] = read(values)
            except KeyError:
                # The protocol does not carry it, so `main`'s own default is
                # the best answer available.
                continue
    kwargs.update(overrides)
    return kwargs
