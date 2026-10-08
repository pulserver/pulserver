"""Protocol keys the interpreter knows, and the options of its string-list keys.

The keys are the wire names of the interpreter's parameter table,
``g_param_table`` in ``src/c/io/pulseg_protocol.c``, grouped by the value type
the table declares for them. A name outside the table is dropped by the
interpreter's parser.
"""

from __future__ import annotations

import sys
from enum import Enum

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):
        """``enum.StrEnum`` for Python 3.10: members format as their value."""

        __str__ = str.__str__
        __format__ = str.__format__


# user0_value .. user45_value and user0_name .. user45_name.
NUM_USER_ENTRIES = 46


class FloatKey(StrEnum):
    """Keys the interpreter declares as float.

    Time keys (``TE``, ``TR``, ...) are nonetheless bound to
    :class:`~pulserver.design.TimeParam` entries, which carry integer
    microseconds. ``TA`` is read-only information.
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
    TA = "TA"

    NEX = "nex"
    BANDWIDTH = "bandwidth"

    FOV = "fov"
    PHASE_FOV = "phase_fov"
    SLICE_THICKNESS = "slice_thickness"
    SLICE_SPACING = "slice_spacing"

    # Translation in mm along the logical readout, phase and slice axes.
    FOV_OFFSET_X = "fov_offset_x"
    FOV_OFFSET_Y = "fov_offset_y"
    FOV_OFFSET_Z = "fov_offset_z"
    FOV_ROTATION_11 = "fov_rotation_11"
    FOV_ROTATION_12 = "fov_rotation_12"
    FOV_ROTATION_13 = "fov_rotation_13"
    FOV_ROTATION_21 = "fov_rotation_21"
    FOV_ROTATION_22 = "fov_rotation_22"
    FOV_ROTATION_23 = "fov_rotation_23"
    FOV_ROTATION_31 = "fov_rotation_31"
    FOV_ROTATION_32 = "fov_rotation_32"
    FOV_ROTATION_33 = "fov_rotation_33"

    RY = "Ry"
    RZ = "Rz"
    COMPRESSED_SENSING = "compressed_sensing"
    MULTIBAND = "multiband"


class IntKey(StrEnum):
    """Keys the interpreter declares as integer."""

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
    """Keys the interpreter declares as boolean."""

    FAT_SAT = "FatSat"
    SPOILER = "Spoiler"
    RF_SPOILING = "RFSpoiling"
    SWAP_PHASE_FREQ = "swap_phase_freq"
    ENABLE_SATURATION_UI = "enable_saturation_ui"
    RECORD_PHYSIO = "record_physio"


class EnumKey(StrEnum):
    """Keys the interpreter declares as string lists.

    Their options are :class:`SequenceType`, :class:`ImagingMode`,
    :class:`PreparationType` and :class:`TriggerType`, in that order.
    """

    SEQUENCE_TYPE = "sequence_type"
    IMAGING_MODE = "imaging_mode"
    PREPARATION_TYPE = "preparation_type"
    TRIGGER_TYPE = "trigger_type"


class ConfigKey(StrEnum):
    """Keys the sequence declares to the interpreter rather than shows.

    The interpreter reads a configuration key once while setting the scan up;
    it has no widget and is never sent back. Its entry is a
    :class:`~pulserver.design.ConfigParam`.
    """

    ENABLE_SAR_BURST_MODE = "enable_sar_burst_mode"


class UserKey(StrEnum):
    """Keys of the float user entries, ``user0_value`` to ``user45_value``.

    ``UserKey.USER3`` is ``UIParam.user_value(3)``. The entry is a
    :class:`~pulserver.design.FloatParam`; the description naming it is bound
    to the matching :class:`UserNameKey` member.
    """

    USER0 = "user0_value"
    USER1 = "user1_value"
    USER2 = "user2_value"
    USER3 = "user3_value"
    USER4 = "user4_value"
    USER5 = "user5_value"
    USER6 = "user6_value"
    USER7 = "user7_value"
    USER8 = "user8_value"
    USER9 = "user9_value"
    USER10 = "user10_value"
    USER11 = "user11_value"
    USER12 = "user12_value"
    USER13 = "user13_value"
    USER14 = "user14_value"
    USER15 = "user15_value"
    USER16 = "user16_value"
    USER17 = "user17_value"
    USER18 = "user18_value"
    USER19 = "user19_value"
    USER20 = "user20_value"
    USER21 = "user21_value"
    USER22 = "user22_value"
    USER23 = "user23_value"
    USER24 = "user24_value"
    USER25 = "user25_value"
    USER26 = "user26_value"
    USER27 = "user27_value"
    USER28 = "user28_value"
    USER29 = "user29_value"
    USER30 = "user30_value"
    USER31 = "user31_value"
    USER32 = "user32_value"
    USER33 = "user33_value"
    USER34 = "user34_value"
    USER35 = "user35_value"
    USER36 = "user36_value"
    USER37 = "user37_value"
    USER38 = "user38_value"
    USER39 = "user39_value"
    USER40 = "user40_value"
    USER41 = "user41_value"
    USER42 = "user42_value"
    USER43 = "user43_value"
    USER44 = "user44_value"
    USER45 = "user45_value"


class UserNameKey(StrEnum):
    """Keys of the descriptions naming the user entries, ``user0_name`` to ``user45_name``.

    ``UserNameKey.USER3`` is ``UIParam.user_name(3)``. The entry is a
    :class:`~pulserver.design.Description`.
    """

    USER0 = "user0_name"
    USER1 = "user1_name"
    USER2 = "user2_name"
    USER3 = "user3_name"
    USER4 = "user4_name"
    USER5 = "user5_name"
    USER6 = "user6_name"
    USER7 = "user7_name"
    USER8 = "user8_name"
    USER9 = "user9_name"
    USER10 = "user10_name"
    USER11 = "user11_name"
    USER12 = "user12_name"
    USER13 = "user13_name"
    USER14 = "user14_name"
    USER15 = "user15_name"
    USER16 = "user16_name"
    USER17 = "user17_name"
    USER18 = "user18_name"
    USER19 = "user19_name"
    USER20 = "user20_name"
    USER21 = "user21_name"
    USER22 = "user22_name"
    USER23 = "user23_name"
    USER24 = "user24_name"
    USER25 = "user25_name"
    USER26 = "user26_name"
    USER27 = "user27_name"
    USER28 = "user28_name"
    USER29 = "user29_name"
    USER30 = "user30_name"
    USER31 = "user31_name"
    USER32 = "user32_name"
    USER33 = "user33_name"
    USER34 = "user34_name"
    USER35 = "user35_name"
    USER36 = "user36_name"
    USER37 = "user37_name"
    USER38 = "user38_name"
    USER39 = "user39_name"
    USER40 = "user40_name"
    USER41 = "user41_name"
    USER42 = "user42_name"
    USER43 = "user43_name"
    USER44 = "user44_name"
    USER45 = "user45_name"


#: A key of the protocol: the wire name of any entry the interpreter knows,
#: as the member of the enum that groups it.
ProtocolKey = FloatKey | IntKey | BoolKey | EnumKey | ConfigKey | UserKey | UserNameKey


def _check_user_index(n: int) -> None:
    if not 0 <= n < NUM_USER_ENTRIES:
        raise ValueError(f"user entries are numbered 0 to {NUM_USER_ENTRIES - 1}")


class UIParam:
    """The keys of the scanner UI's controls, from every typed group.

    ``UIParam.TE`` is ``FloatKey.TE``; configuration keys are reached through
    :class:`ConfigKey`.

    Examples
    --------
    >>> from pulserver.protocol import FloatKey, UIParam
    >>> UIParam.TE is FloatKey.TE
    True
    >>> f"{UIParam.PHASE_FOV}"
    'phase_fov'
    >>> UIParam.user_value(3)
    <UserKey.USER3: 'user3_value'>
    """

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
    TA = FloatKey.TA
    NEX = FloatKey.NEX
    BANDWIDTH = FloatKey.BANDWIDTH
    FOV = FloatKey.FOV
    PHASE_FOV = FloatKey.PHASE_FOV
    SLICE_THICKNESS = FloatKey.SLICE_THICKNESS
    SLICE_SPACING = FloatKey.SLICE_SPACING
    FOV_OFFSET_X = FloatKey.FOV_OFFSET_X
    FOV_OFFSET_Y = FloatKey.FOV_OFFSET_Y
    FOV_OFFSET_Z = FloatKey.FOV_OFFSET_Z
    FOV_ROTATION_11 = FloatKey.FOV_ROTATION_11
    FOV_ROTATION_12 = FloatKey.FOV_ROTATION_12
    FOV_ROTATION_13 = FloatKey.FOV_ROTATION_13
    FOV_ROTATION_21 = FloatKey.FOV_ROTATION_21
    FOV_ROTATION_22 = FloatKey.FOV_ROTATION_22
    FOV_ROTATION_23 = FloatKey.FOV_ROTATION_23
    FOV_ROTATION_31 = FloatKey.FOV_ROTATION_31
    FOV_ROTATION_32 = FloatKey.FOV_ROTATION_32
    FOV_ROTATION_33 = FloatKey.FOV_ROTATION_33
    RY = FloatKey.RY
    RZ = FloatKey.RZ
    COMPRESSED_SENSING = FloatKey.COMPRESSED_SENSING
    MULTIBAND = FloatKey.MULTIBAND

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

    FAT_SAT = BoolKey.FAT_SAT
    SPOILER = BoolKey.SPOILER
    RF_SPOILING = BoolKey.RF_SPOILING
    SWAP_PHASE_FREQ = BoolKey.SWAP_PHASE_FREQ
    ENABLE_SATURATION_UI = BoolKey.ENABLE_SATURATION_UI
    RECORD_PHYSIO = BoolKey.RECORD_PHYSIO

    SEQUENCE_TYPE = EnumKey.SEQUENCE_TYPE
    IMAGING_MODE = EnumKey.IMAGING_MODE
    PREPARATION_TYPE = EnumKey.PREPARATION_TYPE
    TRIGGER_TYPE = EnumKey.TRIGGER_TYPE

    @staticmethod
    def user_value(n: int) -> UserKey:
        """Return the key of float user entry ``n``, 0 to 45.

        Raises
        ------
        ValueError
            If ``n`` is out of range.
        """
        _check_user_index(n)
        return UserKey[f"USER{n}"]

    @staticmethod
    def user_name(n: int) -> UserNameKey:
        """Return the key of the description naming user entry ``n``, 0 to 45.

        Raises
        ------
        ValueError
            If ``n`` is out of range.
        """
        _check_user_index(n)
        return UserNameKey[f"USER{n}"]


class SequenceType(StrEnum):
    """Options of ``sequence_type``."""

    SPIN_ECHO = "spin_echo"
    GRADIENT_ECHO = "gradient_echo"


class ImagingMode(StrEnum):
    """Options of ``imaging_mode``."""

    TWO_D = "2d"
    THREE_D = "3d"


class PreparationType(StrEnum):
    """Options of ``preparation_type``."""

    INVERSION = "inversion"
    T2_PREP = "t2_prep"


class TriggerType(StrEnum):
    """Options of ``trigger_type``.

    The values are the Pulseq trigger channel names, which
    ``pypulseqpp.make_trigger`` takes.
    """

    NONE = "none"
    RESPIRATORY = "physio1"
    ECG = "physio2"


#: Every wire name of the interpreter's parameter table, with its key.
WIRE_NAMES: dict[str, ProtocolKey] = {
    key.value: key
    for keys in (FloatKey, IntKey, BoolKey, EnumKey, ConfigKey, UserKey, UserNameKey)
    for key in keys
}
