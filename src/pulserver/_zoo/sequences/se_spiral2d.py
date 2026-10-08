"""pypulseqpp's 2D spiral spin echo bound to the scanner UI."""

import functools

from pypulseqpp.sequences.sequence.se_spiral2D_sequence import se_spiral2d

from pulserver._zoo._evaluation import evaluation, spiral_2d
from pulserver._zoo._user import OWN, user_entries
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


class SeSpiral2D(SequencePlugin):
    app = functools.partial(
        se_spiral2d, density="variable" if VARIABLE_DENSITY else "constant"
    )
    protocol = {
        UIParam.IMAGING_MODE: StatedParam(ImagingMode.TWO_D),
        UIParam.TE: TimeParam(
            "te",
            range_min=2000,
            range_max=200000,
            presets={TEPreset.MINIMUM: None},
            default=TEPreset.MINIMUM,
        ),
        UIParam.TR: TimeParam(
            "tr",
            range_min=10000,
            range_max=10_000_000,
            presets={TRPreset.MINIMUM: None},
        ),
        UIParam.BANDWIDTH: FloatParam(
            "readout_bandwidth_hz", unit="Hz", range_min=1e3, range_max=1e6
        ),
        UIParam.FOV: FloatParam(
            "fov", unit="mm", scale=1e-3, range_min=50.0, range_max=500.0
        ),
        UIParam.NX: IntParam("n", range_min=32, range_max=512, range_incr=2),
        UIParam.NUM_SHOTS: IntParam("n_shots", range_min=1, range_max=128),
        UIParam.NSLICES: IntParam("n_slices", range_min=1, range_max=64),
        UIParam.SLICE_SPACING: FloatParam(
            "slice_spacing",
            unit="mm",
            scale=1e-3,
            range_min=0.0,
            range_max=100.0,
            range_incr=0.1,
        ),
        UIParam.RY: IntParam("ry", range_min=1, range_max=4),
        UIParam.SLICE_THICKNESS: FloatParam(
            "slice_thickness", unit="mm", scale=1e-3, range_min=1.0, range_max=20.0
        ),
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
        return evaluation(self, system, protocol, spiral_2d)
