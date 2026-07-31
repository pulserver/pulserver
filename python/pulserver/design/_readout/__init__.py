"""Reusable Cartesian and non-Cartesian readout trains.

Set a train's dynamic state with ``set_state()``, then append its sequential
blocks by iterating them into ``seq.add_block``::

    train.set_state(lin_idx=lin_idx)
    for block in train:
        seq.add_block(*block)

``len(train)`` reports the current number of blocks, ``train[index]`` returns
one block tuple, and ``train.get()`` builds a standalone sequence.
"""

from __future__ import annotations

from ._base import Readout
from ._procedural import (
    add_echo_train_blocks,
    add_epi_train_blocks,
    build_base_waveform,
    build_epi_readout,
    build_half_echo_readout,
    build_hard_pulse,
    build_pe_blip,
    build_readout_adc,
    build_readout_gradients,
    compute_readout_and_echo_train,
    order_mode_name,
    pre_post_180_delays,
    rosette,
    spiral,
    trajectory_name,
    unbalanced_line,
    wave_gradients,
)
from ._procedural import (
    line as _legacy_line,
)
from ._surgery import (
    EXCITATION_FLIP_DEG,
    RF_REFOCUS_TIME_S,
    build_kz_crusher_bridge,
    build_readout_surgery_chain,
    build_slice_surgery_chain,
    build_surgery_readout,
    fse_timing,
    scale_refocusing_pulse,
    surgery_ramp_time_s,
    surgery_timing,
)
from .bssfp import Bssfp2D, Bssfp3D
from .epi import Epi2D, Epi2DFlyback, Epi3D, Epi3DFlyback
from .fse import (
    CPMG_PHASE_OFFSET_RAD,
    DEFAULT_REFOCUS_FLIP_DEG,
    Fse2D,
    Fse3D,
    MultiEchoSE,
    build_refocus_flip_schedule,
    build_refocusing_pulse,
    build_z_crusher,
)
from .line import DEFAULT_SPOIL_FACTOR, Line2D, Line3D
from .noncartesian import (
    Arbitrary,
    NonCartesian2D,
    NonCartesian3D,
    NonCartesianGradient,
    NonCartesianProjection,
    NonCartesianStack3D,
    Projection,
    Radial,
    Rosette,
    Spiral,
    StackOfTrajectories,
    radial_trajectory,
    rosette_trajectory,
    spiral_trajectory,
)
from .zte import DEFAULT_RADIAL_OVERSAMP, DEFAULT_ZTE_BANDWIDTH_HZ, Zte

line = _legacy_line

__all__ = [
    # Compatibility functions from the former readout.py module.
    "compute_readout_and_echo_train",
    "line",
    "add_echo_train_blocks",
    "build_epi_readout",
    "build_pe_blip",
    "pre_post_180_delays",
    "add_epi_train_blocks",
    "trajectory_name",
    "order_mode_name",
    "build_base_waveform",
    "build_readout_gradients",
    "build_readout_adc",
    "build_hard_pulse",
    "build_half_echo_readout",
    "unbalanced_line",
    "spiral",
    "rosette",
    "wave_gradients",
    "Readout",
    "MultiEchoSE",
    "Fse2D",
    "Fse3D",
    "build_refocusing_pulse",
    "scale_refocusing_pulse",
    "build_refocus_flip_schedule",
    "build_z_crusher",
    "fse_timing",
    "surgery_ramp_time_s",
    "build_slice_surgery_chain",
    "build_kz_crusher_bridge",
    "build_surgery_readout",
    "build_readout_surgery_chain",
    "surgery_timing",
    "CPMG_PHASE_OFFSET_RAD",
    "DEFAULT_REFOCUS_FLIP_DEG",
    "RF_REFOCUS_TIME_S",
    "EXCITATION_FLIP_DEG",
    "Epi2D",
    "Epi3D",
    "Epi2DFlyback",
    "Epi3DFlyback",
    "Bssfp2D",
    "Bssfp3D",
    "Line2D",
    "Line3D",
    "DEFAULT_SPOIL_FACTOR",
    "NonCartesianGradient",
    "Arbitrary",
    "Radial",
    "Spiral",
    "Rosette",
    "NonCartesian2D",
    "StackOfTrajectories",
    "Projection",
    "NonCartesian3D",
    "NonCartesianStack3D",
    "NonCartesianProjection",
    "radial_trajectory",
    "spiral_trajectory",
    "rosette_trajectory",
    "Zte",
    "DEFAULT_ZTE_BANDWIDTH_HZ",
    "DEFAULT_RADIAL_OVERSAMP",
]
