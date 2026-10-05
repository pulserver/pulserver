"""pypulseqpp's 3D MPRAGE bound to the scanner UI, its partitions counted by the number of slices."""

import itertools

import pypulseqpp as pp
from pypulseqpp.sequences.sequence.mprage3D_sequence import mprage3d

from pulserver._zoo._evaluation import achieved, arguments, rf_layout
from pulserver._zoo._pmc import navigated
from pulserver.design import (
    Evaluation,
    FloatParam,
    IntParam,
    SequencePlugin,
    TimeParam,
)
from pulserver.protocol import TEPreset, TRPreset, UIParam

#: Three-plane navigators after each train, whose pose the ``pmc``
#: reconstruction states to the scan.
NAVIGATOR = False


class Mprage3D(SequencePlugin):
    app = navigated(mprage3d) if NAVIGATOR else mprage3d
    protocol = {
        UIParam.FLIP: FloatParam(
            "flip_angle_deg", unit="deg", range_min=1.0, range_max=90.0
        ),
        UIParam.TE: TimeParam(
            "te", range_min=1000, range_max=80000, presets={TEPreset.MINIMUM: None}
        ),
        UIParam.TR: TimeParam(
            "tr",
            range_min=100_000,
            range_max=20_000_000,
            presets={TRPreset.MINIMUM: None},
        ),
        UIParam.PREP_TIME: TimeParam("ti", range_min=50_000, range_max=5_000_000),
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
        # A TR is one inversion shot: the inversion, then one excitation per
        # line of the fullest partition, which is the centre one. The scan
        # plays a shot per partition, after the dummy shots. The design of
        # the centre partition alone plays one shot of the full length.
        a = arguments(self, protocol)
        calibrating, imaging = pp.make_cartesian_plane_sampling(
            (a["n_y"], a["n_z"]),
            (a["ry"], a["rz"]),
            (a["n_acs_y"], a["n_acs_z"]),
            caipi_shift=a["caipi_shift"],
            partial_fourier=(a["partial_fourier_y"], a["partial_fourier_z"]),
            elliptical=True,
            elliptical_acs=a["elliptical_acs"],
        )
        partitions = {z for _, z in itertools.chain(calibrating, imaging)}
        one = {"n_dummy": 0, "rz": a["n_z"], "n_acs_z": 1}
        shot = self.app(system, **(protocol.arguments | one))
        return Evaluation(
            protocol.replace(achieved(self, shot)),
            (a["n_dummy"] + len(partitions)) * shot.duration()[0],
            rf_layout=rf_layout(shot, scaled=True),
        )
