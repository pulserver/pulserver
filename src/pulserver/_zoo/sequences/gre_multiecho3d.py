"""pypulseqpp's 3D multi-echo gradient echo bound to the scanner UI, its partitions counted by the number of slices."""

import functools

from pypulseqpp.sequences.sequence.gre_multiecho3D_sequence import gre_multiecho3d

from pulserver._zoo._evaluation import cartesian_3d, evaluation
from pulserver._zoo._slab import slab
from pulserver._zoo._user import shortest_at_zero, user_entries
from pulserver.design import (
    FloatParam,
    IntParam,
    SequencePlugin,
    StatedParam,
    TimeParam,
)
from pulserver.protocol import ImagingMode, TEPreset, TRPreset, UIParam

#: Wave-CAIPI: sinusoidal gradients on the phase- and partition-encoding axes
#: during each readout, the calibration region acquired first without them.
WAVE = False
#: Peak wave-encoding gradient amplitude requested, in T/m.
WAVE_AMPLITUDE = 6e-3
#: Wave periods across the sampling window.
WAVE_CYCLES = 8

#: The excitation, one of ``pypulseqpp.sequences.EXCITATIONS``: ``"slab"``
#: selects the slab, ``"spsp"`` excites water alone in it and
#: ``"nonselective"`` plays a hard pulse.
EXCITATION = "slab"

#: Sample only the phase encodes inside the ky-kz ellipse.
ELLIPTICAL = True

#: Monopolar echo train, each echo rewound; bipolar, with even echoes read
#: backwards, otherwise.
FLYBACK = True


class GreMultiecho3D(SequencePlugin):
    app = shortest_at_zero(slab(gre_multiecho3d), "echo_spacing")
    if WAVE:
        app = functools.partial(
            app, wave_amplitude=WAVE_AMPLITUDE, wave_cycles=WAVE_CYCLES
        )
    app = functools.partial(
        app, excitation=EXCITATION, elliptical_sampling=ELLIPTICAL, flyback=FLYBACK
    )
    protocol = {
        UIParam.IMAGING_MODE: StatedParam(ImagingMode.THREE_D),
        UIParam.FLIP: FloatParam(
            "flip_angle_deg", unit="deg", range_min=1.0, range_max=90.0
        ),
        UIParam.TE: TimeParam(
            "te", range_min=1000, range_max=80000, presets={TEPreset.MINIMUM: None}
        ),
        UIParam.TR: TimeParam(
            "tr", range_min=5000, range_max=5_000_000, presets={TRPreset.MINIMUM: None}
        ),
        UIParam.NUM_ECHOES: IntParam("n_echoes", range_min=1, range_max=16),
        UIParam.BANDWIDTH: FloatParam(
            "readout_bandwidth_hz", unit="Hz", range_min=1e3, range_max=1e6
        ),
        UIParam.FOV: FloatParam(
            "fov_x", unit="mm", scale=1e-3, range_min=50.0, range_max=500.0
        ),
        UIParam.PHASE_FOV: FloatParam(
            "fov_y", unit="mm", scale=1e-3, range_min=50.0, range_max=500.0
        ),
        UIParam.NX: IntParam("n_x", range_min=32, range_max=512, range_incr=2),
        UIParam.NY: IntParam("n_y", range_min=32, range_max=512, range_incr=2),
        UIParam.NSLICES: IntParam("n_z", range_min=4, range_max=256),
        UIParam.SLICE_THICKNESS: FloatParam(
            "slice_thickness",
            unit="mm",
            scale=1e-3,
            range_min=0.1,
            range_max=20.0,
            range_incr=0.1,
        ),
        UIParam.RY: IntParam("ry", range_min=1, range_max=4),
        UIParam.RZ: IntParam("rz", range_min=1, range_max=4),
    }

    protocol |= user_entries(app, protocol)

    def evaluate(self, system, protocol):
        return evaluation(self, system, protocol, cartesian_3d)
