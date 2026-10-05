"""pypulseqpp's 3D fast spin echo bound to the scanner UI, its partitions counted by the number of slices."""

import functools

import pypulseqpp as pp
from pypulseqpp.sequences.sequence.fse3D_sequence import fse3d, shot_parameters

from pulserver._zoo._evaluation import achieved, arguments, rf_layout
from pulserver._zoo._pmc import navigated
from pulserver.design import (
    Description,
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

#: Refocusing trains designed with torchsim's extended phase graphs around
#: the refocusing angle at the TE echo, one train shared by every shot.
OPTIMIZED = False

#: Optimized trains individually parameterized for the centre and the
#: periphery of k-space: adds the TR and the echo train length at the
#: periphery as user entries, TR and ETL being the centre's; each shot's train
#: is a cubic step between the two designs.
DUAL_REGION = False

if OPTIMIZED or DUAL_REGION:
    import torchsim  # noqa: F401  the trains are designed with it


class Fse3D(SequencePlugin):
    app = navigated(fse3d) if NAVIGATOR else fse3d
    if OPTIMIZED or DUAL_REGION:
        app = functools.partial(app, flip_modulation="optimized")
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
    if OPTIMIZED or DUAL_REGION:
        protocol[UIParam.FLIP] = FloatParam(
            "refocusing_angle_deg",
            unit="deg",
            range_min=60.0,
            range_max=180.0,
            default=120.0,
        )
    if DUAL_REGION:
        protocol |= {
            UIParam.user_name(0): Description("TR at the periphery"),
            UIParam.user_value(0): FloatParam(
                "tr_periphery",
                unit="ms",
                scale=1e-3,
                range_min=10.0,
                range_max=10_000.0,
                default=1800.0,
            ),
            UIParam.user_name(1): Description("ETL at the periphery"),
            UIParam.user_value(1): IntParam(
                "etl_periphery", range_min=1, range_max=256, default=45
            ),
        }

    def evaluate(self, system, protocol):
        # A TR is one echo train. The design of the centre view plays one
        # train of the full length at the centre's TR, at a constant
        # refocusing angle standing in for an optimized train.
        a = arguments(self, protocol)
        one = {
            "n_dummy": 0,
            "ry": a["n_y"],
            "rz": a["n_z"],
            "n_acs_y": 0,
            "n_acs_z": 0,
            "flip_modulation": "constant",
        }
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
        # Each train lasts its own TR, a cubic step from the centre's to the
        # periphery's; dummies play the first.
        tr = train.duration()[0]
        tr_periphery = a["tr_periphery"]
        etl_periphery = a["etl_periphery"]
        _, times, _ = shot_parameters(
            len(calibrating) + len(imaging),
            a["etl"],
            a["etl"] if etl_periphery is None else etl_periphery,
            tr,
            tr if tr_periphery is None else tr_periphery,
        )
        raster = system.block_duration_raster
        times = [pp.round_to_raster(time, raster) for time in times]
        return Evaluation(
            protocol.replace(achieved(self, train)),
            a["n_dummy"] * times[0] + sum(times),
            rf_layout=rf_layout(train, scaled=False),
        )
