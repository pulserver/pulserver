"""pypulseqpp's 3D multi-echo gradient echo bound to the scanner UI, its partitions counted by the number of slices."""

import functools
from typing import Any

import pypulseqpp as pp
from pypulseqpp.sequences.sequence.gre_multiecho3D_sequence import (
    SAMPLINGS,
    gre_multiecho3d,
)

from pulserver._zoo._evaluation import cartesian_3d, evaluation
from pulserver._zoo._slab import slab
from pulserver._zoo._user import (
    OPTIONS,
    OWN,
    shortest_at_zero,
    user_entries,
    wave_entries,
)
from pulserver.design import (
    ChoiceParam,
    FloatParam,
    IntParam,
    SequencePlugin,
    StatedParam,
    TimeParam,
    option,
    recon_parameters,
)
from pulserver.protocol import Excitation, ImagingMode, TEPreset, TRPreset, UIParam

#: Wave-CAIPI: sinusoidal gradients on the phase- and partition-encoding axes
#: during each readout, the calibration region acquired first without them;
#: adds their amplitude and cycles as user entries.
WAVE = option("wave", False, **OPTIONS["wave"])

#: Sample only the phase encodes inside the ky-kz ellipse.
ELLIPTICAL = option("elliptical", True, **OPTIONS["elliptical"])

#: Monopolar echo train, each echo rewound; bipolar, with even echoes read
#: backwards, otherwise.
FLYBACK = option("flyback", True, **OPTIONS["flyback"])

#: Views outside the calibration region: ``"lattice"``, the CAIPIRINHA
#: lattice, or ``"poisson"``, a variable-density Poisson-disc draw at the
#: undersampling ``ry * rz``, for the compressed-sensing ``pics`` solve. This
#: plugin's own, at its first own user entry: the shared options fill user36
#: to user44, and user45 reaches no console entry on a GE scanner.
SAMPLING = option(
    "sampling", "lattice", choices=SAMPLINGS, user=OWN, label="View sampling"
)

#: The object the ``qsm`` reconstruction maps, each with T2* and SWI:
#: ``"phantom"`` (a MEDI+0 QSM of the whole object), ``"brain"`` (QSM in the
#: brain and DECOMPOSE) or ``"body"``. The sequence states its index among
#: these in its ``QsmAnatomy`` definition, which reaches the reconstruction in
#: the MRD header.
QSM_ANATOMIES = ("phantom", "brain", "body")
QSM_ANATOMY = option(
    "qsm_anatomy",
    "phantom",
    choices=QSM_ANATOMIES,
    user=OWN + 1,
    label="QSM anatomy",
)


def _stating_anatomy(app: Any) -> Any:
    """Return ``app`` with its main sequence stating ``QsmAnatomy`` as a reconstruction parameter."""

    @functools.wraps(app)
    def design(*args: Any, **kwargs: Any) -> Any:
        designed = app(*args, **kwargs)
        main = designed if isinstance(designed, pp.Sequence) else designed[-1]
        recon_parameters(main, QsmAnatomy=QSM_ANATOMIES.index(QSM_ANATOMY))
        return designed

    return design


class GreMultiecho3D(SequencePlugin):
    app = shortest_at_zero(slab(gre_multiecho3d), "echo_spacing")
    app = functools.partial(
        app, elliptical_sampling=ELLIPTICAL, flyback=FLYBACK, sampling=SAMPLING
    )
    app = _stating_anatomy(app)
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
            "tr", range_min=5000, range_max=5_000_000, presets={TRPreset.MINIMUM: None}
        ),
        UIParam.NUM_ECHOES: IntParam("n_echoes", range_min=1, range_max=16),
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

    protocol |= user_entries(app, protocol)

    def evaluate(self, system, protocol):
        return evaluation(self, system, protocol, cartesian_3d)
