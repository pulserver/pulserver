"""pypulseqpp's 3D balanced SSFP bound to the scanner UI, its partitions counted by the number of slices."""

import pypulseqpp as pp
from pypulseqpp.sequences.sequence.bssfp3D_sequence import bssfp3d

from pulserver._zoo._evaluation import achieved, arguments, rf_layout
from pulserver._zoo._slab import slab
from pulserver.design import (
    Evaluation,
    FloatParam,
    IntParam,
    SequencePlugin,
    StatedParam,
    TimeParam,
)
from pulserver.protocol import ImagingMode, TRPreset, UIParam


class Bssfp3D(SequencePlugin):
    app = slab(bssfp3d)
    protocol = {
        UIParam.IMAGING_MODE: StatedParam(ImagingMode.THREE_D),
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
        UIParam.NSLICES: IntParam("n_z", range_min=4, range_max=256),
        UIParam.SLICE_THICKNESS: FloatParam(
            "slice_thickness",
            unit="mm",
            scale=1e-3,
            range_min=0.1,
            range_max=20.0,
            range_incr=0.1,
        ),
        UIParam.RY: IntParam("ry", range_min=1, range_max=4),
        UIParam.RZ: IntParam("rz", range_min=1, range_max=4),
    }

    def evaluate(self, system, protocol):
        # Each phase cycle is a catalyst, which plays the half flip, and one
        # repetition per view of the ky-kz ellipse. The design of the centre
        # view plays one catalyst and one repetition.
        a = arguments(self, protocol)
        one = {
            "ry": a["n_y"],
            "rz": a["n_z"],
            "n_acs_y": 0,
            "n_acs_z": 0,
            "n_phase_cycles": 1,
        }
        catalyst, view = self.app(system, **(protocol.arguments | one))
        calibrating, imaging = pp.make_cartesian_plane_sampling(
            (a["n_y"], a["n_z"]),
            (a["ry"], a["rz"]),
            (a["n_acs_y"], a["n_acs_z"]),
            caipi_shift=a["caipi_shift"],
            partial_fourier=(a["partial_fourier_y"], a["partial_fourier_z"]),
            elliptical=True,
            elliptical_acs=a["elliptical_acs"],
        )
        cycle = (
            catalyst.duration()[0]
            + (len(calibrating) + len(imaging)) * view.duration()[0]
        )
        return Evaluation(
            protocol.replace(achieved(self, view)),
            a["n_phase_cycles"] * cycle,
            rf_layout=rf_layout(view, scaled=True),
        )
