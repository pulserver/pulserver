"""pypulseqpp's 2D balanced SSFP bound to the scanner UI."""

from pypulseqpp.sequences.sequence.bssfp2D_sequence import Bssfp2DApp

from pulserver.design import FloatParam, IntParam, SequencePlugin, TimeParam
from pulserver.protocol import TRPreset, UIParam


class Bssfp2D(SequencePlugin):
    app = Bssfp2DApp
    protocol = {
        UIParam.FLIP: FloatParam(
            "flip_angle_deg", unit="deg", range_min=1.0, range_max=90.0
        ),
        UIParam.TR: TimeParam(
            "tr", range_min=2000, range_max=20000, presets={TRPreset.MINIMUM: None}
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
        UIParam.NSLICES: IntParam("n_slices", range_min=1, range_max=64),
        UIParam.SLICE_THICKNESS: FloatParam(
            "slice_thickness",
            unit="mm",
            scale=1e-3,
            range_min=1.0,
            range_max=20.0,
            default=8.0,
        ),
        UIParam.RY: IntParam("ry", range_min=1, range_max=4),
    }
