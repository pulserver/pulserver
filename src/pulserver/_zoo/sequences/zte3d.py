"""pypulseqpp's 3D zero echo time bound to the scanner UI."""

import math

import numpy as np
import pypulseqpp as pp
from pypulseqpp import sequences
from pypulseqpp.sequences.sequence.zte3D_sequence import (
    HARD_PULSE_DURATION,
    MAX_GRAD,
    MAX_SLEW,
    zte3d,
)

from pulserver._zoo._evaluation import achieved, arguments, rf_layout
from pulserver.design import (
    Evaluation,
    FloatParam,
    IntParam,
    SequencePlugin,
    StatedParam,
    TimeParam,
)
from pulserver.protocol import ImagingMode, TRPreset, UIParam


class Zte3D(SequencePlugin):
    app = zte3d
    protocol = {
        UIParam.IMAGING_MODE: StatedParam(ImagingMode.THREE_D),
        UIParam.FLIP: FloatParam(
            "flip_angle_deg", unit="deg", range_min=0.5, range_max=30.0
        ),
        UIParam.TR: TimeParam(
            "tr", range_min=100, range_max=20_000, presets={TRPreset.MINIMUM: None}
        ),
        UIParam.BANDWIDTH: FloatParam(
            "readout_bandwidth_hz", unit="Hz", range_min=1e4, range_max=1e6
        ),
        UIParam.FOV: FloatParam(
            "fov", unit="mm", scale=1e-3, range_min=50.0, range_max=500.0
        ),
        UIParam.NX: IntParam("n", range_min=32, range_max=512, range_incr=2),
        UIParam.RY: IntParam("r", range_min=1, range_max=8),
    }

    def evaluate(self, system, protocol):
        # A TR is one view: a pulse on the held gradient, then the readout,
        # which turns the gradient onto the next view. Every turn is allotted
        # the time of the widest, so the two views that bound the widest turn
        # are designed, with the TR of the scan, and a shell lasts that design
        # and one TR for each view after its second.
        a = arguments(self, protocol)
        n_shots = a["n_shots"]
        if n_shots is None:
            n_shots = max(1, math.ceil(math.pi * (a["n"] - 1)))
        n_views = max(3, -(-math.ceil(math.pi * a["n"] ** 2) // n_shots))
        directions, _ = pp.calc_projection_shell(n_views, n_shots, scheme=a["scheme"])
        widest = int(np.argmax(np.linalg.norm(np.diff(directions, axis=0), axis=1)))
        system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
        pulse = sequences.NonSelectiveExcitation(
            system, a["flip_angle_deg"], duration_s=HARD_PULSE_DURATION
        )
        turn = sequences.ZteReadout(
            system,
            pulse.rf,
            fov=a["fov"],
            matrix=a["n"],
            directions=directions[widest : widest + 2],
            oversampling=a["readout_oversampling"],
            readout_bandwidth_hz=a["readout_bandwidth_hz"],
            tr=a["tr"],
        )
        main = turn.seq
        main.set_definition(key="TR", value=turn.tr)
        shell = main.duration()[0] + (n_views - 2) * turn.tr
        shells = a["n_dummy"] + len(range(0, n_shots, a["r"]))
        return Evaluation(
            protocol.replace(achieved(self, main)),
            shells * shell,
            rf_layout=rf_layout(main, scaled=True),
        )
