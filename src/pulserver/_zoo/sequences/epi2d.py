"""pypulseqpp's 2D gradient-echo EPI bound to the scanner UI."""

import functools
from typing import Any

import pypulseqpp as pp
from pypulseqpp import sequences
from pypulseqpp.sequences.sequence.epi2D_sequence import epi2d

from pulserver._zoo._evaluation import achieved, arguments, packets, rf_layout
from pulserver._zoo._user import user_entries
from pulserver.design import (
    Evaluation,
    FloatParam,
    IntParam,
    SequencePlugin,
    StatedParam,
    TimeParam,
)
from pulserver.protocol import (
    ImagingMode,
    TEPreset,
    TRPreset,
    UIParam,
)

#: Simultaneous multislice: adds the multiband factor to the protocol.
MULTIBAND = False


#: Saturate fat before every shot: with the fat shifted many pixels along the
#: phase encode, an EPI without it shows the scalp over the brain.
FAT_SATURATION = True


class Epi2D(SequencePlugin):
    # Twofold readout oversampling keeps the ramp-sampled flat top within the
    # spacing of the readout field of view, so it can be resampled onto a grid.
    app = functools.partial(
        epi2d, readout_oversampling=2.0, fat_saturation=FAT_SATURATION
    )
    protocol = {
        UIParam.IMAGING_MODE: StatedParam(ImagingMode.TWO_D),
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
        UIParam.SLICE_SPACING: FloatParam(
            "slice_spacing",
            unit="mm",
            scale=1e-3,
            range_min=0.0,
            range_max=100.0,
            range_incr=0.1,
        ),
        UIParam.SLICE_THICKNESS: FloatParam(
            "slice_thickness", unit="mm", scale=1e-3, range_min=1.0, range_max=20.0
        ),
        UIParam.NUM_FRAMES: IntParam("n_frames", range_min=1, range_max=1000),
        UIParam.NUM_SHOTS: IntParam("n_shots", range_min=1, range_max=16),
        UIParam.RY: IntParam("ry", range_min=1, range_max=4),
    }
    if MULTIBAND:
        protocol[UIParam.MULTIBAND] = IntParam("multiband", range_min=1, range_max=8)

    protocol |= user_entries(app, protocol)

    def evaluate(self, system, protocol):
        # A cycle plays one shot of every slice group of a packet, and a TR
        # the shots of a volume. The design of one group at the shortest TR
        # plays one cycle of that group per shot, after its calibration.
        a = arguments(self, protocol)
        multiband = a["multiband"]
        if a["n_slices"] % multiband:
            raise ValueError(
                f"the slice count {a['n_slices']} is not a multiple of the "
                f"multiband factor {multiband}"
            )
        groups = a["n_slices"] // multiband
        # The bands of a group lie `groups` slices apart, which the one group
        # designed alone keeps by spacing its slices that far apart.
        step = groups * (a["slice_thickness"] + a["slice_spacing"])
        one = {
            "n_slices": multiband,
            "slice_spacing": step - a["slice_thickness"],
            "tr": None,
            "n_frames": 1,
            "n_dummy": 0,
        }
        *calibration, _, volume = self.app(system, **(protocol.arguments | one))
        n_shots, n_frames = a["n_shots"], a["n_frames"]
        shot = volume.definitions["TR"][0] / n_shots
        sizes, cycles = _cycles(
            a | {"n_slices": groups}, shot, system.block_duration_raster
        )
        # The reference volume and the time series each play their dummy
        # cycles first, packet by packet.
        dummies = a["n_dummy"] * (n_shots if n_frames > 1 else 1)
        played = 2 * dummies + (1 + n_frames) * n_shots
        tr = n_shots * max(cycles)
        return Evaluation(
            protocol.replace(achieved(self, volume) | {UIParam.TR: tr}),
            played * sum(cycles) + groups * sequences.duration(calibration),
            rf_layout=rf_layout(volume, scaled=True, copies=max(sizes), period=tr),
        )


def _cycles(
    a: dict[str, Any], shot: float, raster: float
) -> tuple[list[int], list[float]]:
    """Return the slices of each packet, and the duration of one cycle of each.

    A cycle plays one shot of every slice of a packet, each ``shot`` long, and
    a TR ``n_shots`` cycles.

    Raises
    ------
    ValueError
        If the TR cannot hold the shots of one slice, or the slices of a
        volume when there is more than one frame.
    """
    n_slices, n_shots, tr = a["n_slices"], a["n_shots"], a["tr"]
    cycle = None if tr is None else tr / n_shots
    sizes, cycles = packets(n_slices, cycle, shot, raster)
    if a["n_frames"] > 1 and len(sizes) > 1:
        raise ValueError(
            f"the requested TR of {tr * 1e3:.3f} ms cannot hold the {n_slices} "
            f"excitations of a volume, {shot * 1e3:.3f} ms each per shot"
        )
    if cycle is not None and pp.round_to_raster(cycle - shot, raster) < 0:
        raise ValueError(
            f"the requested TR of {tr * 1e3:.3f} ms is shorter than the "
            f"{n_shots * shot * 1e3:.3f} ms the shots of one slice take"
        )
    return sizes, cycles
