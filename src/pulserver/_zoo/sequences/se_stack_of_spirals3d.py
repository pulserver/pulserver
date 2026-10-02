"""pypulseqpp's stack-of-spirals spin echo bound to the scanner UI, its partitions counted by the number of slices."""

from pypulseqpp.sequences.sequence.se_stack_of_spirals3D_sequence import (
    se_stack_of_spirals3d,
)

from pulserver._zoo._evaluation import evaluation
from pulserver.design import FloatParam, IntParam, SequencePlugin, TimeParam
from pulserver.protocol import TEPreset, TRPreset, UIParam


class SeStackOfSpirals3D(SequencePlugin):
    app = se_stack_of_spirals3d
    protocol = {
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
        UIParam.NSLICES: IntParam("n_z", range_min=4, range_max=256),
    }

    def evaluate(self, system, protocol):
        return evaluation(self, system, protocol)
