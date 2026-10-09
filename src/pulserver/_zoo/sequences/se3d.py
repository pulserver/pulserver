"""pypulseqpp's 3D spin echo bound to the scanner UI, its partitions counted by the number of slices."""

import functools

from pypulseqpp.sequences.sequence.se3D_sequence import se3d

from pulserver._zoo._evaluation import cartesian_3d, evaluation
from pulserver._zoo._slab import slab
from pulserver._zoo._user import user_entries
from pulserver.design import (
    FloatParam,
    IntParam,
    SequencePlugin,
    StatedParam,
    TimeParam,
)
from pulserver.protocol import ImagingMode, TEPreset, TRPreset, UIParam

#: The excitation, one of ``pypulseqpp.sequences.EXCITATIONS``: ``"slab"``
#: selects the slab, ``"spsp"`` excites water alone in it and
#: ``"nonselective"`` plays a hard pulse.
EXCITATION = "slab"

#: Sample only the phase encodes inside the ky-kz ellipse.
ELLIPTICAL = True


class Se3D(SequencePlugin):
    app = functools.partial(
        slab(se3d), excitation=EXCITATION, elliptical_sampling=ELLIPTICAL
    )
    protocol = {
        UIParam.IMAGING_MODE: StatedParam(ImagingMode.THREE_D),
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
        return evaluation(self, system, protocol, cartesian_3d)
