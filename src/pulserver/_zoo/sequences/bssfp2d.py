"""pypulseqpp's 2D balanced SSFP bound to the scanner UI, with ECG-gated cine."""

import math

import pypulseqpp as pp
from pypulseqpp.sequences.sequence.bssfp2D_sequence import MAX_SLICE_DURATION, bssfp2d

from pulserver._zoo._evaluation import achieved, arguments, cartesian_2d, rf_layout
from pulserver._zoo._user import user_entries
from pulserver.design import (
    ChoiceParam,
    Evaluation,
    FloatParam,
    IntParam,
    SequencePlugin,
    StatedParam,
    TimeParam,
)
from pulserver.protocol import ImagingMode, TriggerType, TRPreset, UIParam

#: ECG gating is retrospective: each segment cycled over a heartbeat, binned
#: into cardiac phases by the reconstruction. Prospective otherwise: each
#: heartbeat triggered and acquired once per cardiac phase.
RETROSPECTIVE = False


class Bssfp2D(SequencePlugin):
    app = bssfp2d
    protocol = {
        UIParam.IMAGING_MODE: StatedParam(ImagingMode.TWO_D),
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
        UIParam.SLICE_SPACING: FloatParam(
            "slice_spacing",
            unit="mm",
            scale=1e-3,
            range_min=0.0,
            range_max=100.0,
            range_incr=0.1,
        ),
        UIParam.SLICE_THICKNESS: FloatParam(
            "slice_thickness",
            unit="mm",
            scale=1e-3,
            range_min=1.0,
            range_max=20.0,
            default=8.0,
        ),
        UIParam.RY: IntParam("ry", range_min=1, range_max=4),
        UIParam.TRIGGER_TYPE: ChoiceParam(
            "trigger_type", TriggerType, default=TriggerType.NONE
        ),
        UIParam.NUM_FRAMES: IntParam("n_phases", range_min=1, range_max=100),
        UIParam.TRIGGER_DELAY: FloatParam(
            "trigger_delay", unit="ms", scale=1e-3, range_min=0.0, range_max=2000.0
        ),
        UIParam.HEART_RATE: IntParam(
            "heart_rate_bpm", unit="bpm", range_min=30, range_max=200, default=60
        ),
    }

    protocol |= user_entries(app, protocol)

    def generate(self, system, protocol):
        return self.app(system, **_gated(self, system, protocol))

    def evaluate(self, system, protocol):
        # Each slice plays its own train: the half-angle pulse, then one TR per
        # repetition. The design of one line plays one repetition of one slice.
        a = arguments(self, protocol)
        one, lines = cartesian_2d(a)
        repetitions = a["n_dummy"] + lines
        line = self.app(system, **(_ungated(protocol) | one | {"n_slices": 1}))
        tr = line.definitions["TR"][0]
        once = line.duration()[0]
        train = once + (repetitions - 1) * tr
        gating = _gating(protocol)
        if gating != "none":
            g = _segmented(a, tr, protocol)
            segments = _segments(a, g["views_per_segment"])
            rr = 60.0 / a["heart_rate_bpm"]
            if gating == "prospective":
                train = len(segments) * rr
            else:
                cycles = sum(max(1, round(rr / (n * tr))) * n for n in segments)
                train = once + (a["n_dummy"] + cycles - 1) * tr
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


def _gating(protocol) -> str:
    """Return the ``gating`` of the app the trigger choice selects.

    Raises
    ------
    ValueError
        If respiratory triggering is chosen, which the cine does not offer.
    """
    trigger = protocol.arguments["trigger_type"]
    if trigger == TriggerType.RESPIRATORY:
        raise ValueError("the cine is ECG-gated; respiratory triggering is not offered")
    if trigger == TriggerType.ECG:
        return "retrospective" if RETROSPECTIVE else "prospective"
    return "none"


def _ungated(protocol) -> dict:
    """Return the app's arguments of an ungated design of ``protocol``."""
    return {k: v for k, v in protocol.arguments.items() if k != "trigger_type"}


def _segmented(a, tr: float, protocol) -> dict:
    """Return the gating arguments: as many lines per segment as one cardiac phase of a heartbeat holds.

    Raises
    ------
    ValueError
        If one line per cardiac phase does not fit the heartbeat.
    """
    rr = 60.0 / a["heart_rate_bpm"]
    free = rr - a["trigger_delay"] - (0.5 + a["n_dummy"]) * tr
    views = math.floor(free / (a["n_phases"] * tr) + 1e-9)
    if views < 1:
        raise ValueError(
            f"{a['n_phases']} cardiac phases of one line do not fit the "
            f"{rr * 1e3:.0f} ms heartbeat"
        )
    return {
        "gating": _gating(protocol),
        "views_per_segment": views,
        "n_phases": a["n_phases"],
        "trigger_delay": a["trigger_delay"],
    }


def _segments(a, views: int) -> list[int]:
    """Return the lines of each segment of a slice."""
    calibrating, imaging = pp.make_cartesian_axis_sampling(
        a["n_y"], a["ry"], a["n_acs_y"], partial_fourier=a["partial_fourier_y"]
    )
    lines = len(calibrating) + len(imaging)
    return [min(views, lines - i) for i in range(0, lines, views)]


def _gated(plugin, system, protocol) -> dict:
    """Return the app's arguments of ``protocol``, gated as its trigger choice selects."""
    arguments_ = _ungated(protocol)
    if _gating(protocol) == "none":
        return arguments_
    a = arguments(plugin, protocol)
    one, _ = cartesian_2d(a)
    line = plugin.app(system, **(arguments_ | one | {"n_slices": 1}))
    return arguments_ | _segmented(a, line.definitions["TR"][0], protocol)
