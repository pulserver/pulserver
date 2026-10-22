"""pypulseqpp's 3D MPRAGE bound to the scanner UI, its partitions counted by the number of slices."""

import functools

import pypulseqpp as pp
from pypulseqpp.sequences.sequence.mprage3D_sequence import ORDERINGS, mprage3d

from pulserver._zoo._evaluation import achieved, arguments, rf_layout, waved
from pulserver._zoo._pmc import navigated
from pulserver._zoo._slab import slab
from pulserver._zoo._user import (
    LINE_SPACING,
    shortest_at_zero,
    user_entries,
    wave_entries,
)
from pulserver.design import (
    ChoiceParam,
    Evaluation,
    FloatParam,
    IntParam,
    SequencePlugin,
    StatedParam,
    TimeParam,
    option,
)
from pulserver.protocol import Excitation, ImagingMode, TEPreset, TRPreset, UIParam

#: Three-plane navigators after each train, whose pose the ``pmc``
#: reconstruction states to the scan.
NAVIGATOR = option("navigator", False)

#: Wave-CAIPI: sinusoidal gradients on the phase- and partition-encoding axes
#: during each readout, the calibration region acquired first without them;
#: adds their amplitude and cycles as user entries.
WAVE = option("wave", False)

#: Order of the views in the trains: ``"radial"``, angular wedges of the
#: CAIPIRINHA lattice read centre-out, or ``"shuffling"``, Poisson-disc views
#: in random order for a time-resolved reconstruction.
ORDERING = option("ordering", "radial", choices=ORDERINGS)


class Mprage3D(SequencePlugin):
    app = (
        navigated(shortest_at_zero(slab(mprage3d), "esp"))
        if NAVIGATOR
        else shortest_at_zero(slab(mprage3d), "esp")
    )
    app = functools.partial(app, ordering=ORDERING)
    protocol = {
        UIParam.IMAGING_MODE: StatedParam(ImagingMode.THREE_D),
        UIParam.EXCITATION: ChoiceParam(
            "excitation", Excitation, default=Excitation.SLAB
        ),
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

    if WAVE:
        protocol |= wave_entries()

    protocol |= user_entries(app, protocol, LINE_SPACING)

    def evaluate(self, system, protocol):
        # A TR is one inversion shot: the inversion, then a train of one
        # excitation per view. Every train holds as many views as a line has
        # at the undersampling, and the views fill whole trains, so the scan
        # plays a shot per train after the dummy shots and, under the wave,
        # the wave-free trains of the calibration region. The design of one
        # partition's lattice without calibration fills exactly one train.
        a = arguments(self, protocol)
        length = -(-a["n_y"] // a["ry"])
        calibrating, imaging = pp.count_cartesian_plane_sampling(
            (a["n_y"], a["n_z"]),
            (a["ry"], a["rz"]),
            (a["n_acs_y"], a["n_acs_z"]),
            caipi_shift=a["caipi_shift"],
            partial_fourier=(a["partial_fourier_y"], a["partial_fourier_z"]),
            elliptical=True,
            elliptical_acs=a["elliptical_acs"],
            sampling="poisson" if a["ordering"] == "shuffling" else "lattice",
            train_length=length,
        )
        trains = -(-(calibrating + imaging) // length)
        references = -(-calibrating // length) if waved(a) else 0
        one = {
            "n_dummy": 0,
            "ordering": "radial",
            "rz": a["n_z"],
            "n_acs_y": 0,
            "n_acs_z": 0,
            "caipi_shift": 0,
        }
        shot = self.app(system, **(protocol.arguments | one))
        return Evaluation(
            protocol.replace(achieved(self, shot)),
            (a["n_dummy"] + references + trains) * shot.duration()[0],
            rf_layout=rf_layout(shot, scaled=True),
        )
