"""pypulseqpp's 3D MPRAGE bound to the scanner UI, its partitions counted by the number of slices."""

import functools
import itertools

import pypulseqpp as pp
from pypulseqpp.sequences.sequence.mprage3D_sequence import mprage3d

from pulserver._zoo._evaluation import achieved, arguments, rf_layout, waved
from pulserver._zoo._pmc import navigated
from pulserver._zoo._slab import slab
from pulserver._zoo._user import shortest_at_zero, user_entries
from pulserver.design import (
    Evaluation,
    FloatParam,
    IntParam,
    SequencePlugin,
    StatedParam,
    TimeParam,
)
from pulserver.protocol import ImagingMode, TEPreset, TRPreset, UIParam

#: Three-plane navigators after each train, whose pose the ``pmc``
#: reconstruction states to the scan.
NAVIGATOR = False

#: Wave-CAIPI: sinusoidal gradients on the phase- and partition-encoding axes
#: during each readout, the calibration region acquired first without them.
WAVE = False
#: Peak wave-encoding gradient amplitude requested, in T/m.
WAVE_AMPLITUDE = 6e-3
#: Wave periods across the sampling window.
WAVE_CYCLES = 8

#: The excitation, one of ``pypulseqpp.sequences.EXCITATIONS``: ``"slab"``
#: selects the slab, ``"spsp"`` excites water alone in it and
#: ``"nonselective"`` plays a hard pulse.
EXCITATION = "slab"


class Mprage3D(SequencePlugin):
    app = (
        navigated(shortest_at_zero(slab(mprage3d), "esp"))
        if NAVIGATOR
        else shortest_at_zero(slab(mprage3d), "esp")
    )
    if WAVE:
        app = functools.partial(
            app, wave_amplitude=WAVE_AMPLITUDE, wave_cycles=WAVE_CYCLES
        )
    app = functools.partial(app, excitation=EXCITATION)
    protocol = {
        UIParam.IMAGING_MODE: StatedParam(ImagingMode.THREE_D),
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

    protocol |= user_entries(app, protocol)

    def evaluate(self, system, protocol):
        # A TR is one inversion shot: the inversion, then one excitation per
        # line of the fullest partition, which is the centre one. The scan
        # plays a shot per partition, after the dummy shots and, under the
        # wave, a wave-free shot per partition of the calibration region. The
        # design of the centre partition alone plays one shot of the full
        # length, after its wave-free shot under the wave.
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
        references = len({z for _, z in calibrating}) if waved(a) else 0
        one = {"n_dummy": 0, "rz": a["n_z"], "n_acs_z": 1}
        shot = self.app(system, **(protocol.arguments | one))
        designed = 2 if waved(a) and a["n_acs_y"] > 0 else 1
        return Evaluation(
            protocol.replace(achieved(self, shot)),
            (a["n_dummy"] + references + len(partitions))
            * shot.duration()[0]
            / designed,
            rf_layout=rf_layout(shot, scaled=True),
        )
