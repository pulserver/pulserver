"""pypulseqpp's stack-of-stars spin echo bound to the scanner UI, reconstructed by bartorch's NUFFT, its partitions counted by the number of slices."""

from pypulseqpp.sequences.sequence.se_stack_of_stars3D_sequence import (
    SeStackOfStars3DApp,
)

from pulserver.design import FloatParam, IntParam, ScannerSequence, TimeParam
from pulserver.protocol import TEPreset, TRPreset, UIParam


class SeStackOfStars3D(ScannerSequence):
    app = SeStackOfStars3DApp
    recon = "nufft"
    ui = {
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
        UIParam.NSLICES: IntParam("n_z", range_min=4, range_max=256),
    }
