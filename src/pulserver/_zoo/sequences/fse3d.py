"""pypulseqpp's 3D fast spin echo bound to the scanner UI, its partitions counted by the number of slices."""

import functools
import importlib.util
import inspect

import numpy as np
import pypulseqpp as pp
from pypulseqpp.sequences.sequence.fse3D_sequence import (
    ORDERINGS,
    fse3d,
    shot_parameters,
)

from pulserver._zoo._evaluation import (
    achieved,
    arguments,
    dummies,
    rf_layout,
    waved,
)
from pulserver._zoo._pmc import navigated
from pulserver._zoo._slab import slab
from pulserver._zoo._user import (
    OPTIONS,
    OWN,
    shortest_at_zero,
    user_entries,
    wave_entries,
)
from pulserver.design import (
    BoolParam,
    ChoiceParam,
    Description,
    Evaluation,
    FloatParam,
    IntParam,
    RfLayout,
    SequencePlugin,
    StatedParam,
    TimeParam,
    option,
)
from pulserver.protocol import Excitation, ImagingMode, TEPreset, TRPreset, UIParam

#: Three-plane navigators after each train, whose pose the ``pmc``
#: reconstruction states to the scan.
NAVIGATOR = option("navigator", False, **OPTIONS["navigator"])

#: The refocusing trains: ``"constant"``, at the refocusing angle;
#: ``"optimized"``, designed with blochsim's extended phase graphs around the
#: refocusing angle at the TE echo, one train shared by every shot; or
#: ``"dual_region"``, optimized trains individually parameterized for the
#: centre and the periphery of k-space, each shot's train a cubic step between
#: the two designs, which adds the TR and the echo train length at the
#: periphery as user entries, TR and ETL being the centre's.
REFOCUSING = option(
    "refocusing",
    "constant",
    choices=("constant", "optimized", "dual_region", "busse", "busse_dual_region"),
    **OPTIONS["refocusing"],
)
OPTIMIZED = REFOCUSING != "constant"
DUAL_REGION = REFOCUSING in ("dual_region", "busse_dual_region")
BUSSE = REFOCUSING in ("busse", "busse_dual_region")

#: Busse modes use prescribed-PSS inverse EPG; the dual-region option fits
#: both endpoints and interpolates their angle controls cubically.

#: Wave-CAIPI: sinusoidal gradients on the phase- and partition-encoding axes
#: during each readout, the calibration region acquired first without them;
#: adds their amplitude and cycles as user entries.
WAVE = option("wave", False, **OPTIONS["wave"])

# Found, not imported: only generating designs a train, and blochsim imports
# torch, which a listing or an evaluation would wait for.
if OPTIMIZED and importlib.util.find_spec("blochsim") is None:
    raise ModuleNotFoundError(
        "the optimized refocusing trains are designed with blochsim, which is "
        "not installed",
        name="blochsim",
    )

#: Order of the views across shots, ``"radial"`` or ``"shuffling"``.
ORDERING = option("ordering", "radial", choices=ORDERINGS, **OPTIONS["ordering"])


def on_host(app):
    """Return ``app`` designing its refocusing trains with blochsim on the CPU.

    blochsim runs a problem above its work threshold on a CUDA device when
    one is present. The design runs on the scanner host, whose card may be
    one blochsim carries no kernels for ("no kernel image is available for
    execution on the device"), and a train the size of a clinical protocol
    crosses the threshold; on the CPU it is designed in seconds. A constant
    train, which an evaluation designs, imports no blochsim.
    """

    @functools.wraps(app)
    def designed(*args, **kwargs):
        modulation = kwargs.get("flip_modulation", "constant")
        if modulation == "constant" or (
            modulation == "busse" and not kwargs.get("busse_optimize", False)
        ):
            return app(*args, **kwargs)
        from blochsim._execution import execution

        with execution("cpu"):
            return app(*args, **kwargs)

    return designed


def busse_echo_time(app):
    """Translate physical TE to an acquired echo; an explicit echo index wins.

    The default TE uses the Busse midpoint. Both timing discovery and UI
    evaluation construct a prescribed train without importing the optimizer.
    """
    signature = inspect.signature(app)
    stated = signature.replace(
        parameters=[
            p.replace(default=None) if p.name == "te" else p
            for p in signature.parameters.values()
        ]
    )

    @functools.wraps(app)
    def designed(*args, **kwargs):
        if kwargs.get("flip_modulation") == "busse":
            requested = kwargs.get("te")
            kwargs["te"] = None
            if requested is not None and kwargs.get("busse_centre_echo") is None:
                probe = app(*args, **{**kwargs, "busse_optimize": False, "n_dummy": 0})
                acquired = int(probe.definitions["EchoTrainLength"][0])
                startup = int(probe.definitions["StartupEchoes"][0])
                spacing = float(probe.definitions["EchoSpacing"][0])
                first = (
                    float(probe.definitions["TE"][0])
                    - (startup + (acquired - 1) // 2) * spacing
                )
                kwargs["busse_centre_echo"] = (
                    round((requested - first) / spacing) - startup
                )
        return app(*args, **kwargs)

    designed.__signature__ = stated
    return designed


class Fse3D(SequencePlugin):
    app = (
        navigated(shortest_at_zero(slab(fse3d), "esp"))
        if NAVIGATOR
        else shortest_at_zero(slab(fse3d), "esp")
    )
    if BUSSE:
        app = functools.partial(
            on_host(busse_echo_time(app)),
            flip_modulation="busse",
            busse_optimize=True,
        )
    elif OPTIMIZED:
        app = functools.partial(on_host(app), flip_modulation="optimized")
    app = functools.partial(app, ordering=ORDERING)
    if ORDERING == "busse_radial":
        app = functools.partial(app, te=None)
        if BUSSE:
            app = functools.partial(app, busse_centre_echo=0)
    protocol = {
        UIParam.IMAGING_MODE: StatedParam(ImagingMode.THREE_D),
        UIParam.EXCITATION: ChoiceParam(
            "excitation", Excitation, default=Excitation.SLAB
        ),
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
    if OPTIMIZED:
        protocol[UIParam.FLIP] = FloatParam(
            "busse_maximum_angle_deg" if BUSSE else "refocusing_angle_deg",
            unit="deg",
            range_min=60.0,
            range_max=179.0 if BUSSE else 180.0,
            default=120.0,
        )
    if DUAL_REGION:
        protocol |= {
            UIParam.user_name(OWN): Description("TR at the periphery"),
            UIParam.user_value(OWN): FloatParam(
                "tr_periphery",
                unit="ms",
                scale=1e-3,
                range_min=10.0,
                range_max=10_000.0,
                default=1800.0,
            ),
            UIParam.user_name(OWN + 1): Description("ETL at the periphery"),
            UIParam.user_value(OWN + 1): IntParam(
                "etl_periphery", range_min=1, range_max=256, default=45
            ),
        }

    if BUSSE:
        protocol |= {
            UIParam.user_name(OWN + 2): Description("Busse minimum flip"),
            UIParam.user_value(OWN + 2): FloatParam(
                "busse_minimum_angle_deg", unit="deg", range_min=1.0, range_max=90.0
            ),
            UIParam.user_name(OWN + 3): Description("Busse centre flip"),
            UIParam.user_value(OWN + 3): FloatParam(
                "busse_centre_angle_deg", unit="deg", range_min=2.0, range_max=178.0
            ),
            UIParam.user_name(OWN + 4): Description("Busse startup echoes"),
            UIParam.user_value(OWN + 4): IntParam(
                "busse_startup_echoes", range_min=0, range_max=32
            ),
            UIParam.user_name(OWN + 5): Description(
                "Busse centre acquired echo, -1 midpoint"
            ),
            UIParam.user_value(OWN + 5): IntParam(
                "busse_centre_echo", range_min=-1, range_max=255, automatic=-1
            ),
            UIParam.user_name(OWN + 6): Description("Overlap RF dead time"),
            UIParam.user_value(OWN + 6): BoolParam("rf_overlap", default=False),
            UIParam.user_name(OWN + 7): Description("Read-axis crusher cycles"),
            UIParam.user_value(OWN + 7): FloatParam(
                "crusher_cycles", range_min=0.0, range_max=8.0, default=0.5
            ),
        }

    if WAVE:
        protocol |= wave_entries()

    protocol |= user_entries(app, protocol)
    if BUSSE:
        # Busse's maximum angle is bound to FLIP; the direct-angle parameter
        # is absent from its prescription.
        protocol.pop(UIParam.user_name(8), None)
        protocol.pop(UIParam.user_value(8), None)

    def evaluate(self, system, protocol):
        # A TR is one echo train. The design of the centre view plays one
        # train of the full length at a constant refocusing angle, which is
        # the peak of an optimized train and so bounds its RF. The RF of a
        # train does not depend on the view order.
        a = arguments(self, protocol)
        one = {
            "n_dummy": 0,
            "ry": a["n_y"],
            "rz": a["n_z"],
            "n_acs_y": 0,
            "n_acs_z": 0,
            "flip_modulation": "busse" if BUSSE else "constant",
            "busse_optimize": False,
            "ordering": "radial",
        }
        train = self.app(system, **(protocol.arguments | one))
        tr = train.duration()[0]
        tr_periphery = tr if a["tr_periphery"] is None else a["tr_periphery"]
        etl_periphery = a["etl"] if a["etl_periphery"] is None else a["etl_periphery"]
        # Identical trains are all full: the views fill whole trains.
        individual = tr_periphery != tr or etl_periphery != a["etl"]
        length = 1 if individual else a["etl"]
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
        views = calibrating + imaging
        # Each train lasts its own TR, a cubic step from the centre's to the
        # periphery's; dummies and, under the wave, the wave-free trains of
        # the calibration region, each as long as the first, play the first.
        lengths, times, _ = shot_parameters(
            views, a["etl"], etl_periphery, tr, tr_periphery
        )
        raster = system.block_duration_raster
        times = [pp.round_to_raster(time, raster) for time in times]
        references = -(-calibrating // lengths[0]) if waved(a) else 0
        return Evaluation(
            protocol.replace(achieved(self, train)),
            (dummies(a, times[0]) + references) * times[0] + sum(times),
            rf_layout=_longest_train(train, min(times)),
        )


def _longest_train(train: pp.Sequence, period: float) -> RfLayout:
    """Return the RF layout of the longest train of the scan at its shortest TR.

    A train holds as many refocusing pulses as the longest train, those beyond
    its own length at zero amplitude; the layout plays them all at the nominal
    refocusing angle.
    """
    layout = rf_layout(train, scaled=False, period=period)
    instances = layout.instances
    refocusing = np.array(
        [instances.definitions[n].use == "refocusing" for n in instances.definition]
    )
    amplitude = instances.amplitude.copy()
    amplitude[refocusing] = amplitude[refocusing].max()
    return RfLayout(
        pp.RfInstances(instances.definitions, instances.definition, amplitude),
        layout.control,
        period,
    )
