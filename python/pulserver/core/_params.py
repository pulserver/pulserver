"""Typed protocol keys and values for pulserver sequence plugins.

Example
-------
This example builds a minimal protocol using canonical keys and the typed
convenience subclasses for GE UI behavior:

- `TypeinFloatParam` / `TypeinIntParam`: GE shows a type-in field
- `DropdownFloatParam` / `DropdownIntParam`: GE shows type-in plus dropdown
  entries computed from explicit `options` or inferred from `min/max/incr`

    from pulserver.core import (
        DropdownFloatParam,
        DropdownIntParam,
        SequenceType,
        TypeinFloatParam,
        UIParam,
        make_enum_param,
    )

    protocol = {
        UIParam.TE: DropdownFloatParam(
            value=12.0,
            min=5.0,
            max=80.0,
            incr=1.0,
            unit="ms",
            options=[8.0, 12.0, 16.0],  # num_entries == 4
        ),
        UIParam.TR: TypeinFloatParam(
            value=500.0,
            unit="ms",
            # min/max/incr are optional for type-in convenience declarations
        ),
        UIParam.NX: DropdownIntParam(
            value=128,
            min=64,
            max=512,
            incr=1,
            options=[64, 128, 256],     # num_entries == 4
        ),
        UIParam.SEQUENCE_TYPE: make_enum_param(UIParam.SEQUENCE_TYPE, SequenceType.SPIN_ECHO),
    }
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from enum import StrEnum


class Validate(StrEnum):
    """Validation strategy applied to a protocol parameter.

    Notes
    -----
    These values are serialized into the bridge protocol and interpreted by the
    Nim host.

    Examples
    --------
    >>> from pulserver import TypeinFloatParam, UIParam, Validate
    >>> protocol = {UIParam.TR: TypeinFloatParam(value=500.0, validate=Validate.NONE)}
    >>> protocol[UIParam.TR].validate
    <Validate.NONE: 'none'>
    """

    SEARCH = "search"  # binary-search for valid min/max
    CLIP = "clip"  # clamp to [min, max]
    NONE = "none"  # no auto-validation


class InputMode(StrEnum):
    """GE interpreter input mode for a numeric parameter.

    Notes
    -----
    ``OFF`` maps to ``num_entries=0``, ``TYPEIN`` to ``1``, and ``DROPDOWN``
    to ``1 + len(options)``.
    """

    OFF = "off"
    TYPEIN = "typein"
    DROPDOWN = "dropdown"


class SequenceType(StrEnum):
    """Pulse sequence family shown in the UI."""

    SPIN_ECHO = "spin_echo"
    GRADIENT_ECHO = "gradient_echo"


class ImagingMode(StrEnum):
    """Acquisition dimensionality shown in the UI."""

    TWO_D = "2d"
    THREE_D = "3d"


class PreparationType(StrEnum):
    """Preparation pulse family shown in the UI."""

    INVERSION = "inversion"
    T2_PREP = "t2_prep"


class TriggerType(StrEnum):
    """Trigger source selection shown in the UI.

    Notes
    -----
    ``physio1`` and ``physio2`` match the pypulseq trigger channel names used
    when creating trigger events.
    """

    NONE = "none"
    RESPIRATORY = "physio1"
    ECG = "physio2"


class ParamKind(StrEnum):
    """Expected protocol value kind for a key."""

    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STRINGLIST = "stringlist"
    DESCRIPTION = "description"


MAX_DROPDOWN_OPTIONS = 5
MAX_INFERRED_DROPDOWN_OPTIONS = 4


class FloatKey(StrEnum):
    """Canonical keys whose values are expected to be float-like params."""

    SAT_X_LOC1 = "sat_x_loc1"
    SAT_X_LOC2 = "sat_x_loc2"
    SAT_Y_LOC1 = "sat_y_loc1"
    SAT_Y_LOC2 = "sat_y_loc2"
    SAT_Z_LOC1 = "sat_z_loc1"
    SAT_Z_LOC2 = "sat_z_loc2"
    SAT_X_THICKNESS = "sat_x_thickness"
    SAT_Y_THICKNESS = "sat_y_thickness"
    SAT_Z_THICKNESS = "sat_z_thickness"

    DELAY_TIME = "delay_time"
    TRIGGER_DELAY = "trigger_delay"
    TRIGGER_WINDOW = "trigger_window"
    PREP_TIME = "prep_time"

    DIFFUSION_BVALUES = "diffusion_bvalues"

    FLIP = "flip"
    TE = "TE"
    TE2 = "TE2"
    TR = "TR"
    TRECOVERY = "Trecovery"

    NEX = "nex"
    BANDWIDTH = "bandwidth"

    FOV = "fov"
    PHASE_FOV = "phase_fov"
    SLICE_THICKNESS = "slice_thickness"
    SLICE_SPACING = "slice_spacing"

    RY = "Ry"
    RZ = "Rz"
    COMPRESSED_SENSING = "compressed_sensing"
    MULTIBAND = "multiband"


class IntKey(StrEnum):
    """Canonical keys whose values are expected to be int-like params."""

    SAT_X = "sat_x"
    SAT_Y = "sat_y"
    SAT_Z = "sat_z"
    NUM_FRAMES = "num_frames"
    DIFFUSION_DIRECTIONS = "diffusion_directions"
    NUM_SHOTS = "num_shots"
    ETL = "etl"
    NUM_ECHOES = "num_echoes"
    NX = "nx"
    NY = "ny"
    NSLICES = "nslices"
    NUM_SLABS = "num_slabs"
    OVERLAP_LOCATIONS = "overlap_locations"


class BoolKey(StrEnum):
    """Canonical keys whose values are expected to be boolean params."""

    SWAP_PHASE_FREQ = "swap_phase_freq"
    ENABLE_SATURATION_UI = "enable_saturation_ui"
    RECORD_PHYSIO = "record_physio"


class EnumKey(StrEnum):
    """Canonical keys whose values are expected to be enum/string-list params."""

    SEQUENCE_TYPE = "sequence_type"
    IMAGING_MODE = "imaging_mode"
    PREPARATION_TYPE = "preparation_type"
    TRIGGER_TYPE = "trigger_type"


class UIParam:
    """Convenience namespace exposing all canonical protocol keys.

    Use `UIParam.TE`, `UIParam.NX`, `UIParam.SEQUENCE_TYPE`, etc. while the
    implementation keeps separate typed enums underneath.

    Examples
    --------
    >>> from pulserver import UIParam, TypeinFloatParam
    >>> protocol = {UIParam.TR: TypeinFloatParam(value=500.0, unit="ms")}
    >>> str(UIParam.TR)
    'TR'
    """

    # Float-like keys
    SAT_X_LOC1 = FloatKey.SAT_X_LOC1
    SAT_X_LOC2 = FloatKey.SAT_X_LOC2
    SAT_Y_LOC1 = FloatKey.SAT_Y_LOC1
    SAT_Y_LOC2 = FloatKey.SAT_Y_LOC2
    SAT_Z_LOC1 = FloatKey.SAT_Z_LOC1
    SAT_Z_LOC2 = FloatKey.SAT_Z_LOC2
    SAT_X_THICKNESS = FloatKey.SAT_X_THICKNESS
    SAT_Y_THICKNESS = FloatKey.SAT_Y_THICKNESS
    SAT_Z_THICKNESS = FloatKey.SAT_Z_THICKNESS
    DELAY_TIME = FloatKey.DELAY_TIME
    TRIGGER_DELAY = FloatKey.TRIGGER_DELAY
    TRIGGER_WINDOW = FloatKey.TRIGGER_WINDOW
    PREP_TIME = FloatKey.PREP_TIME
    DIFFUSION_BVALUES = FloatKey.DIFFUSION_BVALUES
    FLIP = FloatKey.FLIP
    TE = FloatKey.TE
    TE2 = FloatKey.TE2
    TR = FloatKey.TR
    TRECOVERY = FloatKey.TRECOVERY
    BANDWIDTH = FloatKey.BANDWIDTH
    NEX = FloatKey.NEX
    FOV = FloatKey.FOV
    PHASE_FOV = FloatKey.PHASE_FOV
    SLICE_THICKNESS = FloatKey.SLICE_THICKNESS
    SLICE_SPACING = FloatKey.SLICE_SPACING
    RY = FloatKey.RY
    RZ = FloatKey.RZ
    COMPRESSED_SENSING = FloatKey.COMPRESSED_SENSING
    MULTIBAND = FloatKey.MULTIBAND

    # Int-like keys
    SAT_X = IntKey.SAT_X
    SAT_Y = IntKey.SAT_Y
    SAT_Z = IntKey.SAT_Z
    NUM_FRAMES = IntKey.NUM_FRAMES
    DIFFUSION_DIRECTIONS = IntKey.DIFFUSION_DIRECTIONS
    NUM_SHOTS = IntKey.NUM_SHOTS
    ETL = IntKey.ETL
    NUM_ECHOES = IntKey.NUM_ECHOES
    NX = IntKey.NX
    NY = IntKey.NY
    NSLICES = IntKey.NSLICES
    NUM_SLABS = IntKey.NUM_SLABS
    OVERLAP_LOCATIONS = IntKey.OVERLAP_LOCATIONS

    # Bool-like keys
    SWAP_PHASE_FREQ = BoolKey.SWAP_PHASE_FREQ
    ENABLE_SATURATION_UI = BoolKey.ENABLE_SATURATION_UI
    RECORD_PHYSIO = BoolKey.RECORD_PHYSIO

    # Enum-backed keys
    SEQUENCE_TYPE = EnumKey.SEQUENCE_TYPE
    IMAGING_MODE = EnumKey.IMAGING_MODE
    PREPARATION_TYPE = EnumKey.PREPARATION_TYPE
    TRIGGER_TYPE = EnumKey.TRIGGER_TYPE

    @staticmethod
    def user_value(n: int) -> str:
        """SeqParams user value key: user0_value..user16_value.

        Indices are 0-based.  ``userN_value`` maps to ``opuser(N+3)``
        (i.e. ``opuser3``..``opuser19``).

        The following opuser slots are reserved by the runtime and are
        **not** reachable from Python:

        * ``opuser0`` — file-mode seqfile index
        * ``opuser1`` — physio recording channel bitmask (set via
          ``UIParam.RECORD_PHYSIO``)
        * ``opuser2`` — diffusion tensor file index (0 = none)
        """
        _validate_user_index(n)
        return f"user{n}_value"

    @staticmethod
    def user_name(n: int) -> str:
        """SeqParams/UIControls user-name key: user0_name..user16_name."""
        _validate_user_index(n)
        return f"user{n}_name"

    @staticmethod
    def user_enabled(n: int) -> str:
        """UIControls user-enable key: user1_enabled..user19_enabled."""
        _validate_user_index(n)
        return f"user{n}_enabled"

    @staticmethod
    def user_min(n: int) -> str:
        """UIControls user minimum key: user1_min..user19_min."""
        _validate_user_index(n)
        return f"user{n}_min"

    @staticmethod
    def user_max(n: int) -> str:
        """UIControls user maximum key: user1_max..user19_max."""
        _validate_user_index(n)
        return f"user{n}_max"


def _validate_user_index(n: int) -> None:
    if not 0 <= n <= 16:
        raise ValueError(
            "User slot index must be in [0, 16] (opuser0-2 are reserved; user0..user16 map to opuser3..opuser19)."
        )


# ---------------------------------------------------------------------------
# Protocol value dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FloatParam:
    """Floating-point protocol parameter with bounds and GE UI metadata.

    Parameters
    ----------
    value : float
        Current parameter value.
    min, max : float or None, optional
        Bounds used for serialization and optional dropdown inference.
        Defaults to ``0.0`` and ``inf`` when omitted.
    incr : float or None, optional
        Increment used for serialization and dropdown inference.
    unit : str, optional
        Human-readable unit string.
    validate : Validate, optional
        Validation strategy. Defaults to ``Validate.NONE``.
    mode : InputMode, optional
        GE interpreter input mode.
    options : list of float, optional
        Explicit dropdown options used when ``mode='dropdown'``.

    Examples
    --------
    >>> from pulserver import DropdownFloatParam
    >>> te = DropdownFloatParam(value=12.0, min=5.0, max=20.0, incr=5.0)
    >>> te.options
    [5.0, 10.0, 15.0, 20.0]
    >>> te.num_entries
    5
    """

    value: float
    min: float | None = None
    max: float | None = None
    incr: float | None = None
    unit: str = ""
    validate: Validate = Validate.NONE
    mode: InputMode = InputMode.TYPEIN
    options: list[float] = field(default_factory=list)
    type: str = "float"

    def __post_init__(self) -> None:
        self.validate = Validate(self.validate)
        self.mode = InputMode(self.mode)
        self._fill_numeric_defaults()
        self._maybe_infer_dropdown_options()
        if self.mode == InputMode.DROPDOWN and not (1 <= len(self.options) <= MAX_DROPDOWN_OPTIONS):
            raise ValueError("Dropdown mode requires 1..5 options for GE interpreter compatibility.")

    def _fill_numeric_defaults(self) -> None:
        # Keep params valid for the bridge wire schema even when callers provide
        # value-only convenience declarations.
        if self.min is None:
            self.min = 0.0
        if self.max is None:
            self.max = float("inf")
        if self.incr is None:
            self.incr = 1.0

    def _maybe_infer_dropdown_options(self) -> None:
        if self.mode != InputMode.DROPDOWN or self.options:
            return
        if self.incr is None or self.incr <= 0:
            raise ValueError("Dropdown inference requires incr > 0.")
        assert self.min is not None and self.max is not None
        if self.max < self.min:
            raise ValueError("Dropdown inference requires max >= min.")
        values: list[float] = []
        current = float(self.min)
        while current <= float(self.max) and len(values) < MAX_INFERRED_DROPDOWN_OPTIONS:
            values.append(round(current, 12))
            current += float(self.incr)
        self.options = values

    @property
    def num_entries(self) -> int:
        """Return the GE ``num_entries`` representation.

        Returns
        -------
        int
            ``0`` for off, ``1`` for type-in, and ``2..6`` for type-in plus
            ``1..5`` dropdown options.
        """
        if self.mode == InputMode.OFF:
            return 0
        if self.mode == InputMode.TYPEIN:
            return 1
        return 1 + len(self.options)


@dataclass
class TypeinFloatParam(FloatParam):
    """Float parameter displayed as a GE type-in control.

    Examples
    --------
    >>> from pulserver import TypeinFloatParam
    >>> TypeinFloatParam(value=500.0, unit="ms").num_entries
    1
    """

    mode: InputMode = InputMode.TYPEIN
    options: list[float] = field(default_factory=list)


@dataclass
class DropdownFloatParam(FloatParam):
    """Float parameter displayed as a GE type-in plus dropdown control."""

    mode: InputMode = InputMode.DROPDOWN


@dataclass
class OffFloatParam(FloatParam):
    """Float parameter hidden from the GE UI while remaining serializable."""

    mode: InputMode = InputMode.OFF
    options: list[float] = field(default_factory=list)


@dataclass
class IntParam:
    """Integer protocol parameter with bounds and GE UI metadata.

    Examples
    --------
    >>> from pulserver import DropdownIntParam
    >>> nx = DropdownIntParam(value=128, min=64, max=256, incr=64)
    >>> nx.options
    [64, 128, 192, 256]
    """

    value: int
    min: int | None = None
    max: int | None = None
    incr: int | None = None
    unit: str = ""
    validate: Validate = Validate.NONE
    mode: InputMode = InputMode.TYPEIN
    options: list[int] = field(default_factory=list)
    type: str = "int"

    def __post_init__(self) -> None:
        self.validate = Validate(self.validate)
        self.mode = InputMode(self.mode)
        self._fill_numeric_defaults()
        self._maybe_infer_dropdown_options()
        if self.mode == InputMode.DROPDOWN and not (1 <= len(self.options) <= MAX_DROPDOWN_OPTIONS):
            raise ValueError("Dropdown mode requires 1..5 options for GE interpreter compatibility.")

    def _fill_numeric_defaults(self) -> None:
        if self.min is None:
            self.min = 0
        if self.max is None:
            self.max = sys.maxsize
        if self.incr is None:
            self.incr = 1

    def _maybe_infer_dropdown_options(self) -> None:
        if self.mode != InputMode.DROPDOWN or self.options:
            return
        if self.incr is None or self.incr <= 0:
            raise ValueError("Dropdown inference requires incr > 0.")
        assert self.min is not None and self.max is not None
        if self.max < self.min:
            raise ValueError("Dropdown inference requires max >= min.")
        values: list[int] = []
        current = int(self.min)
        while current <= int(self.max) and len(values) < MAX_INFERRED_DROPDOWN_OPTIONS:
            values.append(current)
            current += int(self.incr)
        self.options = values

    @property
    def num_entries(self) -> int:
        """Return the GE ``num_entries`` representation."""
        if self.mode == InputMode.OFF:
            return 0
        if self.mode == InputMode.TYPEIN:
            return 1
        return 1 + len(self.options)


@dataclass
class TypeinIntParam(IntParam):
    """Int parameter displayed as a GE type-in control."""

    mode: InputMode = InputMode.TYPEIN
    options: list[int] = field(default_factory=list)


@dataclass
class DropdownIntParam(IntParam):
    """Int parameter displayed as a GE type-in plus dropdown control."""

    mode: InputMode = InputMode.DROPDOWN


@dataclass
class OffIntParam(IntParam):
    """Int parameter hidden from the GE UI while remaining serializable."""

    mode: InputMode = InputMode.OFF
    options: list[int] = field(default_factory=list)


@dataclass
class BoolParam:
    """Boolean toggle parameter.

    Examples
    --------
    >>> from pulserver import BoolParam
    >>> BoolParam(value=True).value
    True
    """

    value: bool
    validate: Validate = Validate.NONE
    type: str = "bool"

    def __post_init__(self) -> None:
        self.validate = Validate(self.validate)


@dataclass
class StringListParam:
    """String-list parameter with an explicit selected value.

    Parameters
    ----------
    options : list of str
        Allowed option values.
    value : str, optional
        Explicit selected value.
    index : int, optional
        Explicit selected index. If only ``index`` is provided, ``value`` is
        derived from ``options``.

    Examples
    --------
    >>> from pulserver import StringListParam
    >>> trigger = StringListParam(options=["none", "physio1", "physio2"], value="physio2")
    >>> trigger.index
    2
    """

    options: list[str]
    value: str | None = None
    index: int | None = None
    validate: Validate = Validate.NONE
    type: str = "stringlist"

    def __post_init__(self) -> None:
        self.validate = Validate(self.validate)
        if not self.options:
            raise ValueError("StringListParam requires at least one option.")
        if self.value is None and self.index is None:
            self.index = 0
            self.value = self.options[0]
            return
        if self.value is not None:
            self.value = str(self.value)
            if self.value not in self.options:
                raise ValueError(f"Value '{self.value}' is not present in options.")
            resolved_index = self.options.index(self.value)
            if self.index is None:
                self.index = resolved_index
            elif self.index != resolved_index:
                raise ValueError("StringListParam value and index refer to different options.")
            return
        assert self.index is not None
        if not 0 <= self.index < len(self.options):
            raise ValueError("StringListParam index is out of range.")
        self.value = self.options[self.index]


@dataclass
class Description:
    """Read-only description or section header row."""

    text: str
    type: str = "description"


ProtocolValue = FloatParam | IntParam | BoolParam | StringListParam | Description
ProtocolKey = FloatKey | IntKey | BoolKey | EnumKey | str
Protocol = dict[ProtocolKey, ProtocolValue]


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

_PARAM_KIND_TO_TYPES: dict[ParamKind, tuple[type, ...]] = {
    ParamKind.FLOAT: (FloatParam,),
    ParamKind.INT: (IntParam,),
    ParamKind.BOOL: (BoolParam,),
    ParamKind.STRINGLIST: (StringListParam,),
    ParamKind.DESCRIPTION: (Description,),
}

_PARAM_KINDS: dict[str, ParamKind] = {
    IntKey.SAT_X.value: ParamKind.INT,
    IntKey.SAT_Y.value: ParamKind.INT,
    IntKey.SAT_Z.value: ParamKind.INT,
    UIParam.SAT_X_LOC1.value: ParamKind.FLOAT,
    UIParam.SAT_X_LOC2.value: ParamKind.FLOAT,
    UIParam.SAT_Y_LOC1.value: ParamKind.FLOAT,
    UIParam.SAT_Y_LOC2.value: ParamKind.FLOAT,
    UIParam.SAT_Z_LOC1.value: ParamKind.FLOAT,
    UIParam.SAT_Z_LOC2.value: ParamKind.FLOAT,
    UIParam.SAT_X_THICKNESS.value: ParamKind.FLOAT,
    UIParam.SAT_Y_THICKNESS.value: ParamKind.FLOAT,
    UIParam.SAT_Z_THICKNESS.value: ParamKind.FLOAT,
    IntKey.NUM_FRAMES.value: ParamKind.INT,
    UIParam.DELAY_TIME.value: ParamKind.FLOAT,
    UIParam.TRIGGER_DELAY.value: ParamKind.FLOAT,
    UIParam.TRIGGER_WINDOW.value: ParamKind.FLOAT,
    UIParam.PREP_TIME.value: ParamKind.FLOAT,
    UIParam.DIFFUSION_BVALUES.value: ParamKind.FLOAT,
    IntKey.DIFFUSION_DIRECTIONS.value: ParamKind.INT,
    UIParam.FLIP.value: ParamKind.FLOAT,
    UIParam.TE.value: ParamKind.FLOAT,
    UIParam.TE2.value: ParamKind.FLOAT,
    UIParam.TR.value: ParamKind.FLOAT,
    UIParam.TRECOVERY.value: ParamKind.FLOAT,
    UIParam.BANDWIDTH.value: ParamKind.FLOAT,
    UIParam.NEX.value: ParamKind.FLOAT,
    IntKey.NUM_SHOTS.value: ParamKind.INT,
    IntKey.ETL.value: ParamKind.INT,
    IntKey.NUM_ECHOES.value: ParamKind.INT,
    UIParam.FOV.value: ParamKind.FLOAT,
    UIParam.PHASE_FOV.value: ParamKind.FLOAT,
    UIParam.SLICE_THICKNESS.value: ParamKind.FLOAT,
    UIParam.SLICE_SPACING.value: ParamKind.FLOAT,
    IntKey.NX.value: ParamKind.INT,
    IntKey.NY.value: ParamKind.INT,
    IntKey.NSLICES.value: ParamKind.INT,
    IntKey.NUM_SLABS.value: ParamKind.INT,
    IntKey.OVERLAP_LOCATIONS.value: ParamKind.INT,
    UIParam.RY.value: ParamKind.FLOAT,
    UIParam.RZ.value: ParamKind.FLOAT,
    UIParam.COMPRESSED_SENSING.value: ParamKind.FLOAT,
    UIParam.MULTIBAND.value: ParamKind.FLOAT,
    BoolKey.SWAP_PHASE_FREQ.value: ParamKind.BOOL,
    EnumKey.SEQUENCE_TYPE.value: ParamKind.STRINGLIST,
    EnumKey.IMAGING_MODE.value: ParamKind.STRINGLIST,
    BoolKey.ENABLE_SATURATION_UI.value: ParamKind.BOOL,
    EnumKey.PREPARATION_TYPE.value: ParamKind.STRINGLIST,
    EnumKey.TRIGGER_TYPE.value: ParamKind.STRINGLIST,
    BoolKey.RECORD_PHYSIO.value: ParamKind.BOOL,
}

_ENUM_OPTIONS: dict[str, list[str]] = {
    UIParam.SEQUENCE_TYPE.value: [e.value for e in SequenceType],
    UIParam.IMAGING_MODE.value: [e.value for e in ImagingMode],
    UIParam.PREPARATION_TYPE.value: [e.value for e in PreparationType],
    UIParam.TRIGGER_TYPE.value: [e.value for e in TriggerType],
}


def expected_param_kind(key: UIParam | str) -> ParamKind | None:
    """Return the expected protocol value kind for a canonical key.

    Parameters
    ----------
    key : UIParam or str
        Canonical protocol key.

    Returns
    -------
    ParamKind or None
        Expected kind for known keys, otherwise ``None``.
    """
    key_str = str(key)
    if key_str in _PARAM_KINDS:
        return _PARAM_KINDS[key_str]
    if key_str.endswith("_value") and key_str.startswith("user"):
        return ParamKind.FLOAT
    if key_str.endswith("_enabled") and key_str.startswith("user"):
        return ParamKind.BOOL
    if (key_str.endswith("_min") or key_str.endswith("_max")) and key_str.startswith("user"):
        return ParamKind.FLOAT
    return None


def validate_protocol_entry(key: ProtocolKey, value: ProtocolValue) -> None:
    """Validate that a key/value pair matches the expected protocol value type.

    Raises
    ------
    TypeError
        If the provided value object is incompatible with the key.
    """
    expected = expected_param_kind(key)
    if expected is None:
        return
    expected_types = _PARAM_KIND_TO_TYPES[expected]
    if not isinstance(value, expected_types):
        expected_names = ", ".join(cls.__name__ for cls in expected_types)
        raise TypeError(
            f"Key '{key}' expects {expected.value} parameter type ({expected_names}), " f"got {type(value).__name__}."
        )


def validate_protocol(protocol: Protocol) -> None:
    """Validate all key/value pairs in a protocol mapping.

    Parameters
    ----------
    protocol : Protocol
        Protocol mapping to validate.
    """
    for key, value in protocol.items():
        validate_protocol_entry(key, value)


def enum_options(key: UIParam | str) -> list[str]:
    """Return allowed string options for enum-backed UI controls.

    Examples
    --------
    >>> from pulserver import UIParam, enum_options
    >>> enum_options(UIParam.IMAGING_MODE)
    ['2d', '3d']
    """
    return list(_ENUM_OPTIONS.get(str(key), []))


def make_enum_param(key: UIParam | str, value: StrEnum | str) -> StringListParam:
    """Create a ``StringListParam`` for an enum-backed control.

    Examples
    --------
    >>> from pulserver import TriggerType, UIParam, make_enum_param
    >>> param = make_enum_param(UIParam.TRIGGER_TYPE, TriggerType.ECG)
    >>> param.value
    'physio2'
    """
    options = enum_options(key)
    if not options:
        raise ValueError(f"Key '{key}' is not an enum-backed control.")
    value_str = str(value)
    if value_str not in options:
        raise ValueError(f"Value '{value_str}' is not valid for key '{key}'.")
    return StringListParam(options=options, value=value_str)


def param_to_dict(p: ProtocolValue) -> dict:
    """Convert a protocol value object to a plain dictionary."""
    return asdict(p)


def dict_to_param(d: dict) -> ProtocolValue:
    """Reconstruct a protocol value object from a plain dictionary."""
    d = dict(d)  # shallow copy to avoid mutating caller's dict
    tag = d.pop("type")
    if "validate" in d:
        d["validate"] = Validate(d["validate"])
    if tag in {"float", "int"} and "mode" in d:
        d["mode"] = InputMode(d["mode"])
    if tag in {"bool", "stringlist", "description"}:
        d.pop("unit", None)
    cls = _TYPE_MAP[tag]
    if tag == "float":
        mode = d.get("mode", InputMode.TYPEIN)
        if mode == InputMode.TYPEIN:
            cls = TypeinFloatParam
        elif mode == InputMode.DROPDOWN:
            cls = DropdownFloatParam
        elif mode == InputMode.OFF:
            cls = OffFloatParam
    elif tag == "int":
        mode = d.get("mode", InputMode.TYPEIN)
        if mode == InputMode.TYPEIN:
            cls = TypeinIntParam
        elif mode == InputMode.DROPDOWN:
            cls = DropdownIntParam
        elif mode == InputMode.OFF:
            cls = OffIntParam
    return cls(**d)


def protocol_to_dict(protocol: Protocol) -> dict[str, dict]:
    """Serialize an entire protocol to nested plain dictionaries.

    Notes
    -----
    The protocol is validated before serialization, so incompatible key/value
    pairs fail fast.
    """
    validate_protocol(protocol)
    return {str(k): param_to_dict(v) for k, v in protocol.items()}


def dict_to_protocol(d: dict[str, dict]) -> Protocol:
    """Deserialize nested plain dictionaries back to a typed protocol."""
    return {k: dict_to_param(v) for k, v in d.items()}
