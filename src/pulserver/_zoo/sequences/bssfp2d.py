"""pypulseqpp's 2D balanced SSFP bound to the scanner UI."""

from pypulseqpp.sequences.sequence.bssfp2D_sequence import MAX_SLICE_DURATION, bssfp2d

from pulserver._zoo._evaluation import achieved, arguments, cartesian_2d, rf_layout
from pulserver.design import (
    Evaluation,
    FloatParam,
    IntParam,
    SequencePlugin,
    TimeParam,
)
from pulserver.protocol import TRPreset, UIParam


class Bssfp2D(SequencePlugin):
    app = bssfp2d
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

    def evaluate(self, system, protocol):
        # Each slice plays its own train: the half-angle pulse, then one TR per
        # repetition. The design of one line plays one repetition of one slice.
        a = arguments(self, protocol)
        one, repetitions = cartesian_2d(a)
        line = self.app(system, **(protocol.arguments | one | {"n_slices": 1}))
        tr = line.definitions["TR"][0]
        once = line.duration()[0]
        train = once + (repetitions - 1) * tr
        if train > MAX_SLICE_DURATION:
            raise ValueError(
                f"a slice's train lasts {train:.1f} s, longer than the "
                f"{MAX_SLICE_DURATION:.0f} s one repetition of the scan may; "
                "raise ry, or lower n_y"
            )
        return Evaluation(
            protocol.replace(achieved(self, line)),
            a["n_slices"] * train,
            rf_layout=rf_layout(line, scaled=True, start=once - tr),
        )
