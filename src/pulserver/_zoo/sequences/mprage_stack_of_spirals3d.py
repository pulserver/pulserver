"""pypulseqpp's 3D MPRAGE on a stack of spirals bound to the scanner UI, its partitions counted by the number of slices."""

import functools

from pypulseqpp.sequences.sequence.mprage_stack_of_spirals3D_sequence import (
    mprage_stack_of_spirals3d,
)

from pulserver._zoo._inversion import inversion_train, shots_of_spirals
from pulserver._zoo._slab import slab
from pulserver._zoo._user import OWN, shortest_at_zero, user_entries
from pulserver.design import (
    Description,
    FloatParam,
    IntParam,
    SequencePlugin,
    StatedParam,
    TimeParam,
)
from pulserver.protocol import ImagingMode, TEPreset, TRPreset, UIParam

#: Variable-density spirals: adds how much sparser the periphery is sampled
#: than the centre as a user entry.
VARIABLE_DENSITY = False


class MprageStackOfSpirals3D(SequencePlugin):
    # Every partition plays the same in-plane readouts, so that the
    # reconstruction can Fourier transform along the partitions first.
    app = shortest_at_zero(
        slab(
            functools.partial(
                mprage_stack_of_spirals3d,
                partition_angle_shift="none",
                density="variable" if VARIABLE_DENSITY else "constant",
            )
        ),
        "esp",
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
            "tr",
            range_min=100_000,
            range_max=20_000_000,
            presets={TRPreset.MINIMUM: None},
        ),
        UIParam.PREP_TIME: TimeParam("ti", range_min=50_000, range_max=5_000_000),
        UIParam.BANDWIDTH: FloatParam(
            "readout_bandwidth_hz", unit="Hz", range_min=1e3, range_max=1e6
        ),
        UIParam.FOV: FloatParam(
            "fov", unit="mm", scale=1e-3, range_min=50.0, range_max=500.0
        ),
        UIParam.NX: IntParam("n", range_min=32, range_max=512, range_incr=2),
        UIParam.NUM_SHOTS: IntParam("n_shots", range_min=1, range_max=128),
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
    if VARIABLE_DENSITY:
        protocol |= {
            UIParam.user_name(OWN): Description("Periphery undersampling"),
            UIParam.user_value(OWN): FloatParam(
                "periphery_undersampling", range_min=1.0, range_max=8.0, range_incr=0.1
            ),
        }

    protocol |= user_entries(app, protocol)

    def evaluate(self, system, protocol):
        return inversion_train(self, system, protocol, shots_of_spirals)
