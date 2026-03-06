"""Typed protocol keys and values for pulserver sequence plugins."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import StrEnum


class Validate(StrEnum):
    """Per-parameter validation strategy (maps to nimpulseqgui PropertyValidate)."""

    SEARCH = "search"  # binary-search for valid min/max
    CLIP = "clip"  # clamp to [min, max]
    NONE = "none"  # no auto-validation


class UIParam(StrEnum):
    """Standard MR protocol parameter keys.

    StrEnum members *are* strings — zero conversion cost at bridge boundary.
    Plugin authors may subclass or use raw strings for one-off params.
    """

    # Timing
    TE = "TE"
    TR = "TR"
    TI = "TI"
    # Spatial
    FOV = "FOV"
    SLICE_THICKNESS = "SliceThickness"
    NSLICES = "NSlices"
    MATRIX = "Matrix"
    NECHOES = "NEchoes"
    # Contrast
    FLIP_ANGLE = "FlipAngle"
    BANDWIDTH = "Bandwidth"
    # Flags
    FAT_SAT = "FatSat"
    SPOILER = "Spoiler"
    RF_SPOILING = "RFSpoiling"
    # Description row
    TA = "TA"

    @staticmethod
    def user(n: int) -> str:
        """GE user CV slot: UIParam.user(0) -> 'User0'."""
        return f"User{n}"


# ---------------------------------------------------------------------------
# Protocol value dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FloatParam:
    """Floating-point protocol parameter with bounds and step."""

    value: float
    min: float
    max: float
    incr: float
    unit: str = ""
    validate: Validate = Validate.SEARCH
    type: str = "float"


@dataclass
class IntParam:
    """Integer protocol parameter with bounds and step."""

    value: int
    min: int
    max: int
    incr: int
    unit: str = ""
    validate: Validate = Validate.SEARCH
    type: str = "int"


@dataclass
class BoolParam:
    """Boolean toggle parameter."""

    value: bool
    type: str = "bool"


@dataclass
class StringListParam:
    """Dropdown parameter with a list of options."""

    options: list[str]
    index: int
    type: str = "stringlist"


@dataclass
class Description:
    """Read-only description / section header row."""

    text: str
    type: str = "description"


ProtocolValue = FloatParam | IntParam | BoolParam | StringListParam | Description
Protocol = dict[UIParam | str, ProtocolValue]


# ---------------------------------------------------------------------------
# Bridge serialization helpers (used at Nim <-> Python boundary)
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[str, type] = {
    "float": FloatParam,
    "int": IntParam,
    "bool": BoolParam,
    "stringlist": StringListParam,
    "description": Description,
}


def param_to_dict(p: ProtocolValue) -> dict:
    """Convert a protocol value dataclass to a plain dict."""
    return asdict(p)


def dict_to_param(d: dict) -> ProtocolValue:
    """Reconstruct a protocol value dataclass from a plain dict."""
    d = dict(d)  # shallow copy to avoid mutating caller's dict
    tag = d.pop("type")
    cls = _TYPE_MAP[tag]
    return cls(**d)


def protocol_to_dict(protocol: Protocol) -> dict[str, dict]:
    """Serialize an entire protocol to nested plain dicts."""
    return {str(k): param_to_dict(v) for k, v in protocol.items()}


def dict_to_protocol(d: dict[str, dict]) -> Protocol:
    """Deserialize nested plain dicts back to a typed Protocol."""
    return {k: dict_to_param(v) for k, v in d.items()}
