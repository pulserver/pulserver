"""Protocol-dictionary getters/setters and axis resolution for plugins."""

from __future__ import annotations

from pulserver import UIParam


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
    >>> from pulserver.design import params
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
    >>> from pulserver.design import params
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
    >>> from pulserver.design import params
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
    >>> from pulserver.design import params
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
    >>> from pulserver.design import params
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
    >>> from pulserver.design import params
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
    >>> from pulserver.design import params
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
    >>> from pulserver.design import params
    >>> params.acs_lines_from_protocol({}, 128, 0)
    0
    """
    raw = user_float(protocol, slot, 0.0)
    acs = int(round(raw))
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
    >>> from pulserver.design import params
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
    >>> from pulserver.design import params
    >>> protocol = {str(UIParam.TE): {"value": 5.0, "type": "float"}}
    >>> params.set_protocol_value(protocol, UIParam.TE, 7.0)
    >>> protocol[str(UIParam.TE)]["value"]
    7.0
    """
    entry = protocol[str(key)]
    entry["value"] = value
    if entry.get("type") == "stringlist":
        entry["index"] = entry["options"].index(value)
