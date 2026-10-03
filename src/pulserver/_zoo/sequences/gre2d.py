"""pypulseqpp's 2D gradient echo bound to the scanner UI."""

from pypulseqpp.sequences.sequence.gre2D_sequence import gre2d

from pulserver._zoo._evaluation import cartesian_2d, evaluation
from pulserver.design import FloatParam, IntParam, SequencePlugin, TimeParam
from pulserver.protocol import TEPreset, TRPreset, UIParam


class Gre2D(SequencePlugin):
    app = gre2d
    protocol = {
        UIParam.FLIP: FloatParam(
            "flip_angle_deg", unit="deg", range_min=1.0, range_max=90.0
        ),
        UIParam.TE: TimeParam(
            "te",
            range_min=1000,
            range_max=80000,
            range_incr=10,
            options=(5000, 8000),
            presets={TEPreset.MINIMUM: None},
        ),
        UIParam.TR: TimeParam(
            "tr",
            range_min=1000,
            range_max=5_000_000,
            presets={TRPreset.MINIMUM: None},
        ),
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
        UIParam.RY: IntParam("ry", range_min=1, range_max=4),
    }

    def evaluate(self, system, protocol):
        return evaluation(self, system, protocol, cartesian_2d)
