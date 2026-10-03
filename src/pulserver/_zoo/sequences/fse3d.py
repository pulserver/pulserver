"""pypulseqpp's 3D fast spin echo bound to the scanner UI, its partitions counted by the number of slices."""

import pypulseqpp as pp
from pypulseqpp.sequences.sequence.fse3D_sequence import fse3d, shot_parameters

from pulserver._zoo._evaluation import achieved, arguments, rf_layout
from pulserver.design import (
    Evaluation,
    FloatParam,
    IntParam,
    SequencePlugin,
    TimeParam,
)
from pulserver.protocol import TEPreset, TRPreset, UIParam


class Fse3D(SequencePlugin):
    app = fse3d
    protocol = {
        UIParam.TE: TimeParam(
            "te", range_min=2000, range_max=500000, presets={TEPreset.MINIMUM: None}
        ),
        UIParam.TR: TimeParam(
            "tr",
            range_min=10000,
            range_max=10_000_000,
            presets={TRPreset.MINIMUM: None},
        ),
        UIParam.ETL: IntParam("etl", range_min=1, range_max=256),
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
        UIParam.NSLICES: IntParam("n_z", range_min=4, range_max=256),
        UIParam.RY: IntParam("ry", range_min=1, range_max=4),
        UIParam.RZ: IntParam("rz", range_min=1, range_max=4),
    }

    def evaluate(self, system, protocol):
        # A TR is one echo train, every train as long as the first. The design
        # of the centre view plays one train of the full length.
        a = arguments(self, protocol)
        one = {"n_dummy": 0, "ry": a["n_y"], "rz": a["n_z"], "n_acs_y": 0, "n_acs_z": 0}
        train = self.app(system, **(protocol.arguments | one))
        calibrating, imaging = pp.make_cartesian_plane_sampling(
            (a["n_y"], a["n_z"]),
            (a["ry"], a["rz"]),
            (a["n_acs_y"], a["n_acs_z"]),
            caipi_shift=a["caipi_shift"],
            partial_fourier=(a["partial_fourier_y"], a["partial_fourier_z"]),
            elliptical=True,
            elliptical_acs=a["elliptical_acs"],
        )
        # The repetition times do not enter the number of trains.
        lengths, _, _ = shot_parameters(
            len(calibrating) + len(imaging), a["etl"], a["etl"], 0.0, 0.0
        )
        return Evaluation(
            protocol.replace(achieved(self, train)),
            (a["n_dummy"] + len(lengths)) * train.duration()[0],
            rf_layout=rf_layout(train, scaled=False),
        )
