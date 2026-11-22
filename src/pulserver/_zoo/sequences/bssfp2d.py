"""pypulseqpp's 2D balanced SSFP bound to the scanner UI: free-running, retrospective and prospective cine."""

import math

import pypulseqpp as pp
from pypulseqpp import sequences
from pypulseqpp.sequences.sequence.bssfp2D_sequence import (
    ORDERINGS,
    PROSPECTIVE_DUMMIES,
    RETROSPECTIVE_WINDOW,
    STEADY_STATE_TOLERANCE,
    bssfp2d,
    prospective_starts,
)

from pulserver._zoo._evaluation import (
    achieved,
    arguments,
    cartesian_2d,
    edge,
    rf_layout,
)
from pulserver._zoo._user import OPTIONS, user_entries
from pulserver.design import (
    ChoiceParam,
    Evaluation,
    FloatParam,
    IntParam,
    SequencePlugin,
    StatedParam,
    TimeParam,
    option,
    recon_parameters,
)
from pulserver.protocol import ImagingMode, TriggerType, TRPreset, UIParam

#: ECG gating is retrospective: each segment cycled over ``RETROSPECTIVE_WINDOW``
#: heartbeats, binned into cardiac phases by the reconstruction from the
#: recorded ECG.
#: Prospective otherwise: each slice triggered once, on its first excitation,
#: then every nominal heartbeat acquired once per cardiac phase, the slice one
#: train with no rest in it; with a trigger window, every heartbeat waits for
#: its own trigger.
RETROSPECTIVE = option("retrospective", False, **OPTIONS["retrospective"])

#: Phase-encode order of an ungated cine: every frame in k-space order, or
#: every line of the grid in golden-ratio order, for real-time frames or a
#: binning by the recorded physiological signal in the reconstruction.
ORDERING = option("ordering", "sequential", choices=ORDERINGS, **OPTIONS["ordering"])

#: Breathing rate a respiratory-triggered cine is timed to (breaths per
#: minute): each breath triggers one segment of every cardiac phase.
RESPIRATORY_RATE_BPM = 15.0

#: Shortest part of a breath left free for the next respiratory trigger (s).
RESPIRATORY_WINDOW = 1.5


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
        UIParam.TRIGGER_WINDOW: FloatParam(
            "trigger_window", unit="ms", scale=1e-3, range_min=0.0, range_max=1000.0
        ),
        UIParam.HEART_RATE: IntParam(
            "heart_rate_bpm", unit="bpm", range_min=30, range_max=200, default=60
        ),
    }

    protocol |= user_entries(app, protocol)

    def generate(self, system, protocol):
        designed = self.app(system, **_gated(self, system, protocol))
        a = arguments(self, protocol)
        stated = {
            "Gating": _gating(protocol),
            "Ordering": _ordering(protocol),
            "CardiacPhases": int(a["n_phases"]),
        }
        for part in designed if isinstance(designed, list) else [designed]:
            recon_parameters(part, **stated)
        return designed

    def evaluate(self, system, protocol):
        # Each slice plays its own train, one TR per repetition, the first
        # ramping the flip angle up, ungated its lines once per frame. The design of the
        # centre and edge lines of one frame plays those repetitions of one
        # slice; ``once`` is the train of one line.
        a = arguments(self, protocol)
        one, lines = cartesian_2d(a)
        if _ordering(protocol) == "golden":
            lines = len(_golden_lines(a))
        line = self.app(system, **(_ungated(protocol) | one | _ONE_TRAIN))
        tr = line.definitions["TR"][0]
        once = line.duration()[0] - (len(_designed_lines(a)) - 1) * tr
        gating = _gating(protocol)
        n_dummy = _dummies(a, tr, gating)
        frames = a["n_phases"] if gating == "none" else 1
        train = once + (n_dummy + frames * lines - 1) * tr
        # A frame of an ungated cine plays every line once; a gated one's
        # cardiac phase plays the lines of a segment.
        frame_time = lines * tr if a["n_phases"] > 1 else None
        if gating != "none":
            g = _segmented(a, tr, protocol)
            frame_time = g["views_per_segment"] * tr
            segments = _segments(a, g["views_per_segment"])
            rr = _period(a, protocol)
            if gating == "prospective" and g["trigger_window"] > 0.0:
                # Every heartbeat opens on its own trigger with the dummies:
                # a heartbeat a segment.
                lead = n_dummy + round(a["trigger_delay"] / tr)
                last = lead + a["n_phases"] * segments[-1]
                train = (len(segments) - 1) * rr + once + (last - 1) * tr
            elif gating == "prospective":
                # One trigger, on the first excitation, opens the slice; the trigger
                # delay and the heartbeats after the first are repetitions
                # that do not acquire.
                starts = prospective_starts(
                    segments, a["n_phases"], n_dummy, rr, tr, a["trigger_delay"]
                )
                played = starts[-1] + a["n_phases"] * segments[-1]
                train = once + (played - 1) * tr
            else:
                per_segment = math.ceil(
                    RETROSPECTIVE_WINDOW * rr / (g["views_per_segment"] * tr)
                )
                cycles = per_segment * sum(segments)
                train = once + (n_dummy + cycles - 1) * tr
        return Evaluation(
            protocol.replace(achieved(self, line)),
            a["n_slices"] * train,
            rf_layout=rf_layout(line, scaled=True, start=once - tr),
            sequence=line,
            frame_time=frame_time,
        )


#: The app's arguments for one train of one slice: ungated and in k-space
#: order, the frames counted rather than designed.
_ONE_TRAIN = {"n_slices": 1, "n_phases": 1, "ordering": "sequential"}


def _designed_lines(a) -> list[int]:
    """Return the lines the evaluation designs: the centre and edge lines of :func:`~pulserver._zoo._evaluation.extremes`."""
    calibrating, imaging = pp.make_cartesian_axis_sampling(
        a["n_y"], edge(a["n_y"]), 0, partial_fourier=1.0
    )
    return [*calibrating, *imaging]


def _gating(protocol) -> str:
    """Return the ``gating`` of the app the trigger choice selects: a respiratory trigger is prospective."""
    trigger = protocol.arguments["trigger_type"]
    if trigger == TriggerType.RESPIRATORY:
        return "prospective"
    if trigger == TriggerType.ECG:
        return "retrospective" if RETROSPECTIVE else "prospective"
    return "none"


def _ordering(protocol) -> str:
    """Return the ``ordering`` of the app: :data:`ORDERING` when ungated, else ``'sequential'``."""
    return ORDERING if _gating(protocol) == "none" else "sequential"


def _period(a, protocol) -> float:
    """Return the nominal interval between triggers (s): the heartbeat, or the breath under respiratory triggering."""
    if protocol.arguments["trigger_type"] == TriggerType.RESPIRATORY:
        return 60.0 / RESPIRATORY_RATE_BPM
    return 60.0 / a["heart_rate_bpm"]


def _window(a, protocol) -> float:
    """Return the trigger window (s): the one asked for, at least :data:`RESPIRATORY_WINDOW` under respiratory triggering."""
    if protocol.arguments["trigger_type"] == TriggerType.RESPIRATORY:
        return max(a["trigger_window"], RESPIRATORY_WINDOW)
    return a["trigger_window"]


def _golden_lines(a) -> list[int]:
    """Return the lines a golden free-running frame visits: every line of the grid partial Fourier keeps."""
    calibrating, imaging = pp.make_cartesian_axis_sampling(
        a["n_y"], 1, 0, partial_fourier=a["partial_fourier_y"]
    )
    return [*calibrating, *imaging]


def _dummies(a, tr: float, gating: str) -> int:
    """Return the dummy repetitions the app plays ahead of a slice's first readout."""
    if a["n_dummy"] is not None:
        return int(a["n_dummy"])
    if gating == "prospective":
        return PROSPECTIVE_DUMMIES
    n = sequences.steady_state_dummies(
        tr,
        a["flip_angle_deg"],
        t2_s=sequences.STEADY_STATE_T2_S,
        tolerance=STEADY_STATE_TOLERANCE,
    )
    if gating == "retrospective":
        n = min(n, max(1, round(60.0 / a["heart_rate_bpm"] / tr)))
    return n


def _ungated(protocol) -> dict:
    """Return the app's arguments of an ungated design of ``protocol``, in its view order."""
    arguments_ = {
        k: v
        for k, v in protocol.arguments.items()
        if k not in ("trigger_type", "trigger_window")
    }
    return arguments_ | {"ordering": _ordering(protocol)}


def _segmented(a, tr: float, protocol) -> dict:
    """Return the gating arguments: as many lines per segment as one cardiac phase of a heartbeat holds.

    Raises
    ------
    ValueError
        If one line per cardiac phase does not fit the heartbeat.
    """
    rr = _period(a, protocol)
    gating = _gating(protocol)
    window = _window(a, protocol) if gating == "prospective" else 0.0
    leading = _dummies(a, tr, gating) * tr if gating == "prospective" else 0.0
    free = rr - window - a["trigger_delay"] - leading
    views = math.floor(free / (a["n_phases"] * tr) + 1e-9)
    if views < 1:
        raise ValueError(
            f"{a['n_phases']} cardiac phases of one line do not fit the "
            f"{rr * 1e3:.0f} ms between triggers"
        )
    gated = {
        "gating": gating,
        "views_per_segment": views,
        "n_phases": a["n_phases"],
        "trigger_delay": a["trigger_delay"],
        "trigger_window": window,
        "trigger_channel": str(protocol.arguments["trigger_type"]),
    }
    if gating != "none":
        gated["heart_rate_bpm"] = 60.0 / rr
    return gated


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
    line = plugin.app(system, **(arguments_ | one | _ONE_TRAIN))
    return arguments_ | _segmented(a, line.definitions["TR"][0], protocol)
