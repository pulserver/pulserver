"""Typed protocol keys and values for pulserver sequence plugins.

Example
-------
This example builds a minimal protocol using canonical keys and the typed
convenience subclasses for GE UI behavior:

- `TypeinFloatParam` / `TypeinIntParam`: GE shows a type-in field
- `DropdownFloatParam` / `DropdownIntParam`: GE shows type-in plus dropdown
  entries computed from explicit `options` or inferred from `min/max/incr`

    from pulserver.design import (
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
from enum import Enum, IntEnum

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):
        """``enum.StrEnum`` for interpreters that do not carry one.

        Members are strings, compare equal to their value and format as it, so
        a value written into the bridge protocol reads the same whichever
        interpreter the scanner runs.
        """

        __str__ = str.__str__
        __format__ = str.__format__


class TEPreset(IntEnum):
    """Echo-time dropdown entries the UI shows as words rather than numbers.

    A TE control offers presets alongside the type-in field, and picking one
    sends its value through in place of a time. The values are GE's, from
    ``epic_ui_control.h``; a plugin puts them in a dropdown's ``options``.

    Every one of them is a request for a TE the sequence works out for itself,
    so :func:`pulserver.design.main_kwargs` passes them on as ``te=None`` -- which is
    what a readout module reads as "as short as possible".

    Examples
    --------
    >>> from pulserver.design import DropdownFloatParam, TEPreset
    >>> te = DropdownFloatParam(value=8.0, min=1.0, max=80.0, unit="ms",
    ...                         options=[TEPreset.MINIMUM, 5.0, 8.0, 15.0])
    >>> te.options[0]
    -2.0
    """

    MIN_FULL = -1
    MINIMUM = -2
    IN_PHASE = -3
    OUT_PHASE = -4
    MAXIMUM = -5


class TRPreset(IntEnum):
    """Repetition-time dropdown entries the UI shows as words.

    :class:`TEPreset` for TR, and separate from it because the numbers mean
    different things: ``-1`` is the shortest TR here and the shortest *full*
    TE there.

    Examples
    --------
    >>> from pulserver.design import DropdownFloatParam, TRPreset
    >>> tr = DropdownFloatParam(value=250.0, min=5.0, max=5000.0, unit="ms",
    ...                         options=[TRPreset.MINIMUM, 250.0, 500.0])
    >>> tr.options[0]
    -1.0
    """

    MINIMUM = -1


class Validate(StrEnum):
    """Validation strategy applied to a protocol parameter.

    Notes
    -----
    These values are serialized into the bridge protocol and interpreted by the
    Nim host.

    Examples
    --------
    >>> from pulserver.design import TypeinFloatParam, UIParam
    >>> from pulserver.design._params import Validate
    >>> protocol = {UIParam.TR: TypeinFloatParam(value=500.0, validate=Validate.NONE)}
    >>> protocol[UIParam.TR].validate
    <Validate.NONE: 'none'>
    """

    SEARCH = "search"  # binary-search for valid min/max
    CLIP = "clip"  # clamp to [min, max]
    NONE = "none"  # no auto-validation


class InputMode(StrEnum):
    """How a numeric parameter is presented in the GE interpreter UI.

    Set implicitly by the parameter class you choose — ``Typein*``,
    ``Dropdown*`` or ``Off*`` — rather than passed by hand.

    Notes
    -----
    ``OFF`` maps to ``num_entries=0``, ``TYPEIN`` to ``1``, and ``DROPDOWN``
    to ``1 + len(options)``.

    Examples
    --------
    >>> from pulserver.design import DropdownIntParam, TypeinIntParam
    >>> from pulserver.design._params import InputMode
    >>> TypeinIntParam(value=1).mode is InputMode.TYPEIN
    True
    >>> DropdownIntParam(value=128, options=[64, 128]).num_entries
    3
    """

    OFF = "off"
    TYPEIN = "typein"
    DROPDOWN = "dropdown"


class SequenceType(StrEnum):
    """Pulse sequence family shown in the UI.

    Examples
    --------
    >>> from pulserver.design import SequenceType, UIParam, make_enum_param
    >>> make_enum_param(UIParam.SEQUENCE_TYPE, SequenceType.GRADIENT_ECHO).value
    'gradient_echo'
    """

    SPIN_ECHO = "spin_echo"
    GRADIENT_ECHO = "gradient_echo"


class ImagingMode(StrEnum):
    """Acquisition dimensionality shown in the UI.

    Examples
    --------
    >>> from pulserver.design import ImagingMode
    >>> str(ImagingMode.THREE_D)
    '3d'
    """

    TWO_D = "2d"
    THREE_D = "3d"


class PreparationType(StrEnum):
    """Preparation pulse family shown in the UI.

    Selects which magnetization-preparation module a plugin plays before the
    imaging train — see :func:`pulserver.design.make_inversion_pulse` and
    :func:`pulserver.design.make_t2prep_pulse`.

    Examples
    --------
    >>> from pulserver.design import PreparationType
    >>> str(PreparationType.T2_PREP)
    't2_prep'
    """

    INVERSION = "inversion"
    T2_PREP = "t2_prep"


class TriggerType(StrEnum):
    """Trigger source selection shown in the UI.

    The member *values* are the pypulseq trigger channel names, so a selected
    option can be handed straight to
    :func:`pulserver.pypulseq.make_trigger`.

    Notes
    -----
    ``physio1`` and ``physio2`` match the pypulseq trigger channel names used
    when creating trigger events.

    Examples
    --------
    >>> from pulserver.design import TriggerType, UIParam, make_enum_param
    >>> make_enum_param(UIParam.TRIGGER_TYPE, TriggerType.ECG).value
    'physio2'

    Gate the sequence on the selected channel::

        if channel != TriggerType.NONE:
            seq.add_block(pp.make_trigger(channel, duration=1e-3))
    """

    NONE = "none"
    RESPIRATORY = "physio1"
    ECG = "physio2"


class ParamKind(StrEnum):
    """Value kind a canonical protocol key expects.

    Returned by :func:`expected_param_kind` and used by
    :func:`validate_protocol` to reject a control of the wrong type before it
    ever reaches the scanner.

    Examples
    --------
    >>> from pulserver.design import UIParam
    >>> from pulserver.design._params import ParamKind, expected_param_kind
    >>> expected_param_kind(UIParam.TR) is ParamKind.FLOAT
    True
    """

    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STRINGLIST = "stringlist"
    DESCRIPTION = "description"


MAX_DROPDOWN_OPTIONS = 5
MAX_INFERRED_DROPDOWN_OPTIONS = 4


class FloatKey(StrEnum):
    """Canonical keys whose values must be float parameters.

    One of four key groups (see also :class:`IntKey`, :class:`BoolKey`,
    :class:`EnumKey`) that together define which value type each canonical key
    accepts. Reach them through :class:`UIParam`; the groups exist so
    validation can classify a key without a lookup table.

    Examples
    --------
    >>> from pulserver.design import UIParam
    >>> from pulserver.design._params import FloatKey
    >>> UIParam.TR is FloatKey.TR
    True
    """

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

    # Off-isocentre translation, mm, along the LOGICAL readout/phase/slice
    # axes -- the frame the sequence is designed in, and the frame the
    # scanner already prescribes them in, so nothing is rotated on the way.
    FOV_OFFSET_X = "fov_offset_x"
    FOV_OFFSET_Y = "fov_offset_y"
    FOV_OFFSET_Z = "fov_offset_z"

    RY = "Ry"
    RZ = "Rz"
    COMPRESSED_SENSING = "compressed_sensing"
    MULTIBAND = "multiband"


class IntKey(StrEnum):
    """Canonical keys whose values must be integer parameters.

    Examples
    --------
    >>> from pulserver.design._params import IntKey, ParamKind, expected_param_kind
    >>> expected_param_kind(IntKey.NUM_ECHOES) is ParamKind.INT
    True
    """

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
    """Canonical keys whose values must be boolean parameters.

    Examples
    --------
    >>> from pulserver.design._params import BoolKey, ParamKind, expected_param_kind
    >>> expected_param_kind(BoolKey.RECORD_PHYSIO) is ParamKind.BOOL
    True
    """

    SWAP_PHASE_FREQ = "swap_phase_freq"
    ENABLE_SATURATION_UI = "enable_saturation_ui"
    RECORD_PHYSIO = "record_physio"
    ENABLE_SAR_BURST_MODE = "enable_sar_burst_mode"


class EnumKey(StrEnum):
    """Canonical keys whose values must be string-list (enum) parameters.

    Build their values with :func:`make_enum_param` rather than by hand, so
    the selected string and its dropdown index stay consistent.

    Examples
    --------
    >>> from pulserver.design._params import EnumKey, enum_options
    >>> enum_options(EnumKey.IMAGING_MODE)
    ['2d', '3d']
    """

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
    >>> from pulserver.design import UIParam, TypeinFloatParam
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
    FOV_OFFSET_X = FloatKey.FOV_OFFSET_X
    FOV_OFFSET_Y = FloatKey.FOV_OFFSET_Y
    FOV_OFFSET_Z = FloatKey.FOV_OFFSET_Z
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
    ENABLE_SAR_BURST_MODE = BoolKey.ENABLE_SAR_BURST_MODE

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
    >>> from pulserver.design import DropdownFloatParam
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
        # A TEPreset or TRPreset entry is an int subclass; the wire format is
        # plain numbers, so it becomes one here rather than at every reader.
        self.options = [float(option) for option in self.options]
        self._fill_numeric_defaults()
        self._maybe_infer_dropdown_options()
        if self.mode == InputMode.DROPDOWN and not (
            1 <= len(self.options) <= MAX_DROPDOWN_OPTIONS
        ):
            raise ValueError(
                "Dropdown mode requires 1..5 options for GE interpreter compatibility."
            )

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
        while (
            current <= float(self.max) and len(values) < MAX_INFERRED_DROPDOWN_OPTIONS
        ):
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
    >>> from pulserver.design import TypeinFloatParam
    >>> TypeinFloatParam(value=500.0, unit="ms").num_entries
    1
    """

    mode: InputMode = InputMode.TYPEIN
    options: list[float] = field(default_factory=list)


@dataclass
class DropdownFloatParam(FloatParam):
    """Float parameter offering preset values plus free type-in.

    Use this instead of :class:`TypeinFloatParam` when a handful of values
    cover almost every scan — the presets appear in the dropdown, while the
    field stays typeable for anything else.

    Examples
    --------
    >>> from pulserver.design import DropdownFloatParam
    >>> te = DropdownFloatParam(value=8.0, min=2.0, max=80.0, unit="ms",
    ...                         options=[4.0, 8.0, 16.0])
    >>> te.num_entries
    4
    """

    mode: InputMode = InputMode.DROPDOWN


@dataclass
class OffFloatParam(FloatParam):
    """Float parameter hidden from the GE UI while remaining serializable.

    Examples
    --------
    >>> from pulserver.design import OffFloatParam, protocol_to_dict
    >>> slice_gap = OffFloatParam(value=2.5)
    >>> slice_gap.mode.value
    'off'
    >>> protocol_to_dict({"slice_gap": slice_gap})["slice_gap"]["value"]
    2.5
    """

    mode: InputMode = InputMode.OFF
    options: list[float] = field(default_factory=list)


@dataclass
class IntParam:
    """Integer protocol parameter with bounds and GE UI metadata.

    Examples
    --------
    >>> from pulserver.design import DropdownIntParam
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
        if self.mode == InputMode.DROPDOWN and not (
            1 <= len(self.options) <= MAX_DROPDOWN_OPTIONS
        ):
            raise ValueError(
                "Dropdown mode requires 1..5 options for GE interpreter compatibility."
            )

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
    """Integer parameter displayed as a GE type-in control.

    The default integer control: a free numeric field bounded by ``min`` and
    ``max`` — matrix size, echo train length, number of averages.

    Examples
    --------
    >>> from pulserver.design import TypeinIntParam
    >>> etl = TypeinIntParam(value=16, min=1, max=64)
    >>> etl.num_entries
    1
    """

    mode: InputMode = InputMode.TYPEIN
    options: list[int] = field(default_factory=list)


@dataclass
class DropdownIntParam(IntParam):
    """Integer parameter offering preset values plus free type-in.

    Examples
    --------
    >>> from pulserver.design import DropdownIntParam
    >>> matrix = DropdownIntParam(value=128, min=32, max=512,
    ...                           options=[64, 128, 256])
    >>> matrix.num_entries
    4
    """

    mode: InputMode = InputMode.DROPDOWN


@dataclass
class OffIntParam(IntParam):
    """Int parameter hidden from the GE UI while remaining serializable.

    Examples
    --------
    >>> from pulserver.design import OffIntParam, protocol_to_dict
    >>> shots = OffIntParam(value=4)
    >>> shots.mode.value
    'off'
    >>> protocol_to_dict({"shots": shots})["shots"]["value"]
    4
    """

    mode: InputMode = InputMode.OFF
    options: list[int] = field(default_factory=list)


@dataclass
class BoolParam:
    """Boolean toggle parameter.

    Examples
    --------
    >>> from pulserver.design import BoolParam
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
    >>> from pulserver.design import StringListParam
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
                raise ValueError(
                    "StringListParam value and index refer to different options."
                )
            return
        assert self.index is not None
        if not 0 <= self.index < len(self.options):
            raise ValueError("StringListParam index is out of range.")
        self.value = self.options[self.index]


@dataclass
class Description:
    """Read-only text row: a section header or an explanatory note.

    Carries no value and is never read back by the sequence — it exists only
    to group and annotate the controls around it in the scanner UI.

    Examples
    --------
    >>> from pulserver.design import Description
    >>> header = Description(text="Diffusion preparation")
    >>> header.type
    'description'
    """

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
    UIParam.FOV_OFFSET_X.value: ParamKind.FLOAT,
    UIParam.FOV_OFFSET_Y.value: ParamKind.FLOAT,
    UIParam.FOV_OFFSET_Z.value: ParamKind.FLOAT,
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
    BoolKey.ENABLE_SAR_BURST_MODE.value: ParamKind.BOOL,
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

    Examples
    --------
    >>> from pulserver.design import UIParam
    >>> from pulserver.design._params import ParamKind, expected_param_kind
    >>> expected_param_kind(UIParam.TR)
    <ParamKind.FLOAT: 'float'>
    >>> expected_param_kind("not_a_key") is None
    True

    ``userN_*`` slots are classified by suffix rather than by name:

    >>> expected_param_kind("user3_value")
    <ParamKind.FLOAT: 'float'>
    """
    key_str = str(key)
    if key_str in _PARAM_KINDS:
        return _PARAM_KINDS[key_str]
    if key_str.endswith("_value") and key_str.startswith("user"):
        return ParamKind.FLOAT
    if key_str.endswith("_enabled") and key_str.startswith("user"):
        return ParamKind.BOOL
    if (key_str.endswith(("_min", "_max"))) and key_str.startswith("user"):
        return ParamKind.FLOAT
    if key_str.endswith("_name") and key_str.startswith("user"):
        return ParamKind.DESCRIPTION
    return None


def validate_protocol_entry(key: ProtocolKey, value: ProtocolValue) -> None:
    """Check that one protocol entry uses the value type its key requires.

    Unknown keys pass silently: a plugin may carry its own entries alongside
    the canonical ones, and only the canonical keys have a declared type.

    Parameters
    ----------
    key : ProtocolKey
        Canonical key or plugin-specific string.
    value : ProtocolValue
        Parameter object to check against it.

    Raises
    ------
    TypeError
        If the value object is incompatible with the key.

    Examples
    --------
    >>> from pulserver.design import TypeinFloatParam, TypeinIntParam, UIParam
    >>> from pulserver.design._params import validate_protocol_entry
    >>> validate_protocol_entry(UIParam.TR, TypeinFloatParam(value=500.0, unit="ms"))
    >>> validate_protocol_entry(UIParam.TR, TypeinIntParam(value=500))
    Traceback (most recent call last):
        ...
    TypeError: Key 'TR' expects float parameter type (FloatParam), got TypeinIntParam.

    See Also
    --------
    validate_protocol : the whole-mapping form.
    """
    expected = expected_param_kind(key)
    if expected is None:
        return
    expected_types = _PARAM_KIND_TO_TYPES[expected]
    if not isinstance(value, expected_types):
        expected_names = ", ".join(cls.__name__ for cls in expected_types)
        raise TypeError(
            f"Key '{key}' expects {expected.value} parameter type ({expected_names}), "
            f"got {type(value).__name__}."
        )


def validate_protocol(protocol: Protocol) -> None:
    """Check every entry of a protocol against its key's declared type.

    Called automatically by :func:`protocol_to_dict`, so a mistyped control
    fails at serialisation rather than on the scanner.

    Parameters
    ----------
    protocol : Protocol
        Protocol mapping to validate.

    Raises
    ------
    TypeError
        On the first entry whose value type does not match its key.

    Examples
    --------
    >>> from pulserver.design import TypeinFloatParam, UIParam
    >>> from pulserver.design._params import validate_protocol
    >>> validate_protocol({UIParam.TR: TypeinFloatParam(value=500.0, unit="ms")})

    See Also
    --------
    validate_protocol_entry : the single-entry form.
    """
    for key, value in protocol.items():
        validate_protocol_entry(key, value)


def enum_options(key: UIParam | str) -> list[str]:
    """Return allowed string options for enum-backed UI controls.

    Examples
    --------
    >>> from pulserver.design import UIParam
    >>> from pulserver.design._params import enum_options
    >>> enum_options(UIParam.IMAGING_MODE)
    ['2d', '3d']
    """
    return list(_ENUM_OPTIONS.get(str(key), []))


def make_enum_param(key: UIParam | str, value: StrEnum | str) -> StringListParam:
    """Create a ``StringListParam`` for an enum-backed control.

    Examples
    --------
    >>> from pulserver.design import TriggerType, UIParam, make_enum_param
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
    """Convert one protocol parameter object to a plain dictionary.

    The wire format at the Nim/Python bridge boundary; the ``type`` field it
    carries is what lets :func:`dict_to_param` rebuild the right class.

    Parameters
    ----------
    p : ProtocolValue
        Parameter object.

    Returns
    -------
    dict
        Plain, JSON-serialisable representation.

    Examples
    --------
    >>> from pulserver.design import TypeinFloatParam
    >>> from pulserver.design._params import param_to_dict
    >>> entry = param_to_dict(TypeinFloatParam(value=500.0, unit="ms"))
    >>> entry["type"], entry["value"], entry["unit"]
    ('float', 500.0, 'ms')

    See Also
    --------
    dict_to_param : the inverse.
    """
    return asdict(p)


def dict_to_param(d: dict) -> ProtocolValue:
    """Rebuild a typed protocol parameter from its dictionary form.

    The concrete class is recovered from ``type`` *and* ``mode`` together, so
    a float entry comes back as :class:`TypeinFloatParam`,
    :class:`DropdownFloatParam` or ``OffFloatParam`` — the same object the
    plugin originally declared.

    Parameters
    ----------
    d : dict
        Dictionary produced by :func:`param_to_dict`.

    Returns
    -------
    ProtocolValue
        Reconstructed parameter object.

    Examples
    --------
    >>> from pulserver.design import TypeinFloatParam
    >>> from pulserver.design._params import dict_to_param, param_to_dict
    >>> original = TypeinFloatParam(value=500.0, unit="ms")
    >>> dict_to_param(param_to_dict(original)) == original
    True

    See Also
    --------
    param_to_dict : the inverse.
    """
    d = dict(d)  # shallow copy to avoid mutating caller's dict
    tag = d.pop("type")
    if "validate" in d:
        d["validate"] = Validate(d["validate"])
    if tag in {"float", "int"} and "mode" in d:
        d["mode"] = InputMode(d["mode"])
    if tag in {"bool", "stringlist", "description"}:
        d.pop("unit", None)
    if tag == "description":
        d.pop("validate", None)
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
    """Serialize a whole protocol to nested plain dictionaries.

    Keys are stringified and values converted with :func:`param_to_dict`. The
    protocol is validated first, so an incompatible key/value pair fails here
    rather than downstream.

    Parameters
    ----------
    protocol : Protocol
        Typed protocol mapping.

    Returns
    -------
    dict
        Mapping of key string to plain parameter dictionary.

    Raises
    ------
    TypeError
        If any entry fails :func:`validate_protocol`.

    Examples
    --------
    >>> from pulserver.design import TypeinFloatParam, UIParam, protocol_to_dict
    >>> serialized = protocol_to_dict({UIParam.TR: TypeinFloatParam(value=500.0, unit="ms")})
    >>> sorted(serialized)
    ['TR']
    >>> serialized["TR"]["value"]
    500.0

    See Also
    --------
    dict_to_protocol : the inverse.
    """
    validate_protocol(protocol)
    return {str(k): param_to_dict(v) for k, v in protocol.items()}


def dict_to_protocol(d: dict[str, dict]) -> Protocol:
    """Deserialize nested plain dictionaries back into a typed protocol.

    Parameters
    ----------
    d : dict
        Mapping of key string to plain parameter dictionary.

    Returns
    -------
    Protocol
        Mapping of the same keys to reconstructed parameter objects.

    Examples
    --------
    >>> from pulserver.design import TypeinFloatParam, UIParam
    >>> from pulserver.design import dict_to_protocol, protocol_to_dict
    >>> original = {UIParam.TR: TypeinFloatParam(value=500.0, unit="ms")}
    >>> dict_to_protocol(protocol_to_dict(original))["TR"].value
    500.0

    See Also
    --------
    protocol_to_dict : the inverse.
    """
    return {k: dict_to_param(v) for k, v in d.items()}


def set_protocol_value(protocol: dict, key: UIParam | str, value) -> None:
    """Set a value in a *serialized* protocol, keeping dropdowns consistent.

    Operates on the plain-dictionary form (post
    :func:`protocol_to_dict`), in place. For a string-list entry it also
    updates ``index`` to match the new selection — writing ``value`` alone
    would leave the UI pointing at the previous option.

    Parameters
    ----------
    protocol : dict
        Serialized protocol, modified in place.
    key : UIParam or str
        Canonical key to set.
    value : Any
        New value; for a string-list entry it must be one of its ``options``.

    Raises
    ------
    KeyError
        If ``key`` is not present in the protocol.
    ValueError
        If a string-list value is not among that entry's options.

    Examples
    --------
    >>> from pulserver.design import TriggerType, UIParam
    >>> from pulserver.design import make_enum_param, protocol_to_dict
    >>> from pulserver.design._params import set_protocol_value
    >>> serialized = protocol_to_dict(
    ...     {UIParam.TRIGGER_TYPE: make_enum_param(UIParam.TRIGGER_TYPE, TriggerType.NONE)}
    ... )
    >>> set_protocol_value(serialized, UIParam.TRIGGER_TYPE, "physio2")
    >>> serialized["trigger_type"]["value"], serialized["trigger_type"]["index"]
    ('physio2', 2)
    """
    entry = protocol[str(key)]
    entry["value"] = value
    if entry.get("type") == "stringlist":
        entry["index"] = entry["options"].index(value)
