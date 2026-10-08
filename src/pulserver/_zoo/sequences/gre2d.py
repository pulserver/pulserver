"""pypulseqpp's 2D gradient echo bound to the scanner UI."""

from pypulseqpp.sequences.sequence.gre2D_sequence import gre2d

from pulserver._zoo._evaluation import cartesian_2d, evaluation
from pulserver._zoo._saturation import band_entries, explicit_bands
from pulserver._zoo._user import user_entries
from pulserver.design import (
    FloatParam,
    IntParam,
    SequencePlugin,
    StatedParam,
    TimeParam,
)
from pulserver.protocol import ImagingMode, TEPreset, TRPreset, UIParam

#: Explicit saturation bands the sequence plays, prescribed on the console's
#: graphic Rx and always on; 0 plays none.
BANDS = 2


class Gre2D(SequencePlugin):
    app = explicit_bands(gre2d, BANDS) if BANDS else gre2d
    protocol = {
        UIParam.IMAGING_MODE: StatedParam(ImagingMode.TWO_D),
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
        UIParam.RY: IntParam("ry", range_min=1, range_max=4),
    }

    if BANDS:
        protocol |= band_entries(BANDS)

    protocol |= user_entries(app, protocol)

    def evaluate(self, system, protocol):
        return evaluation(self, system, protocol, cartesian_2d)
