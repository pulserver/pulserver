"""pypulseqpp's 2D gradient-echo EPI bound to the scanner UI."""

import functools

from pypulseqpp.sequences.sequence.epi2D_sequence import Epi2DApp

from pulserver.design import FloatParam, IntParam, SequencePlugin, TimeParam
from pulserver.protocol import TEPreset, TRPreset, UIParam


class _Epi2DApp(Epi2DApp):
    # Twofold readout oversampling keeps the ramp-sampled flat top within the
    # spacing of the readout field of view, so it can be resampled onto a grid.
    init_sequence = functools.partialmethod(
        Epi2DApp.init_sequence, readout_oversampling=2.0
    )


class Epi2D(SequencePlugin):
    app = _Epi2DApp
    protocol = {
        UIParam.FLIP: FloatParam(
            "flip_angle_deg", unit="deg", range_min=1.0, range_max=90.0
        ),
        UIParam.TE: TimeParam(
            "te",
            range_min=1000,
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
            "readout_bandwidth_hz", unit="Hz", range_min=1e3, range_max=2e6
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
            "slice_thickness", unit="mm", scale=1e-3, range_min=1.0, range_max=20.0
        ),
        UIParam.NUM_FRAMES: IntParam("n_frames", range_min=1, range_max=1000),
        UIParam.NUM_SHOTS: IntParam("n_shots", range_min=1, range_max=16),
        UIParam.RY: IntParam("ry", range_min=1, range_max=4),
    }
