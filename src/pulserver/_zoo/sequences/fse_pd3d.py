"""Prescribed-signal PD fast spin echo with optional CPU EPG control fitting."""

import functools
import inspect

import pypulseqpp as pp
from pypulseqpp.sequences.sequence.fse3D_sequence import (
    fse3d,
    shot_parameters,
)

from pulserver._zoo._evaluation import (
    achieved,
    arguments,
    dummies,
    waved,
)
from pulserver._zoo._fse import (
    bounding_train,
    echo_train_layout,
    effective_te,
    sar_burst,
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
    ChoiceParam,
    ConfigParam,
    Description,
    Evaluation,
    FloatParam,
    IntParam,
    SequencePlugin,
    StatedParam,
    TimeParam,
    option,
)
from pulserver.protocol import (
    ConfigKey,
    Excitation,
    ImagingMode,
    TEPreset,
    TRPreset,
    UIParam,
)

#: Three-plane navigators after each train, whose pose the ``pmc``
#: reconstruction states to the scan.
NAVIGATOR = option("navigator", False, **OPTIONS["navigator"])

#: Independently prescribed centre/periphery TR, ETL and final angle.
DUAL_REGION = option(
    "dual_region", False, user=20, label="PD centre/periphery transition"
)
#: Optional multi-tissue/B1 forward EPG refinement of the inverse-EPG design.
FIT_CONTROLS = option("fit_controls", False, user=21, label="Fit PD controls with EPG")
WAVE = option("wave", False, **OPTIONS["wave"])

#: Shared trains use Busse radial/PCLO; changing endpoint parameters uses
#: Buonincontri joint shot/echo assignment. ``radial`` selects from the protocol.
ORDERINGS = ("radial", "shuffling", "busse_radial", "pclo")
ORDERING = option("ordering", "radial", choices=ORDERINGS, **OPTIONS["ordering"])


def on_host(app):
    """Use CPU forward EPG when requested and expose only PD angle controls."""

    @functools.wraps(app)
    def designed(*args, **kwargs):
        if not kwargs.get("pd_optimize", False):
            return app(*args, **kwargs)
        from blochsim._execution import execution

        with execution("cpu"):
            return app(*args, **kwargs)

    signature = inspect.signature(app)
    designed.__signature__ = signature.replace(
        parameters=[
            p
            for name, p in signature.parameters.items()
            if name != "refocusing_angle_deg"
        ]
    )
    return designed


class FsePd3D(SequencePlugin):
    app = sar_burst(
        navigated(shortest_at_zero(slab(fse3d), "esp"))
        if NAVIGATOR
        else shortest_at_zero(slab(fse3d), "esp")
    )
    app = functools.partial(
        on_host(app),
        te=None,
        flip_modulation="pd",
        pd_optimize=FIT_CONTROLS,
    )
    app = functools.partial(app, ordering=ORDERING)
    if ORDERING == "busse_radial":
        app = functools.partial(app, te=None)
        app = functools.partial(app, busse_centre_echo=0)
    protocol = {
        UIParam.IMAGING_MODE: StatedParam(ImagingMode.THREE_D),
        # Costed under the SAR burst limits, as a product 3D FSE asks to be.
        ConfigKey.ENABLE_SAR_BURST_MODE: ConfigParam(1),
        UIParam.EXCITATION: ChoiceParam(
            "excitation", Excitation, default=Excitation.SLAB
        ),
        UIParam.TE: TimeParam(
            "pd_effective_te",
            range_min=2000,
            range_max=500000,
            presets={TEPreset.MINIMUM: None},
        ),
        UIParam.TR: TimeParam(
            "tr",
            range_min=10000,
            range_max=10_000_000,
            presets={TRPreset.MINIMUM: None},
        ),
        UIParam.ETL: IntParam("etl", range_min=2, range_max=256),
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
    protocol[UIParam.FLIP] = FloatParam(
        "pd_maximum_angle_deg",
        unit="deg",
        range_min=60.0,
        range_max=179.0,
        default=140.0,
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
                "etl_periphery", range_min=2, range_max=256, default=45
            ),
        }

    protocol |= {
        UIParam.user_name(14): Description("First refocusing flip"),
        UIParam.user_value(14): FloatParam(
            "pd_first_angle_deg", unit="deg", range_min=1.0, range_max=179.0
        ),
        UIParam.user_name(15): Description("Final refocusing flip"),
        UIParam.user_value(15): FloatParam(
            "pd_final_angle_deg", unit="deg", range_min=1.0, range_max=179.0
        ),
        UIParam.user_name(16): Description("Startup echoes"),
        UIParam.user_value(16): IntParam(
            "busse_startup_echoes", range_min=0, range_max=32
        ),
        UIParam.user_name(17): Description("Centre acquired echo, -1 from TE"),
        UIParam.user_value(17): IntParam(
            "busse_centre_echo", range_min=-1, range_max=255, automatic=-1
        ),
        UIParam.user_name(19): Description("Read-axis crusher cycles"),
        UIParam.user_value(19): FloatParam(
            "crusher_cycles", range_min=0.0, range_max=8.0, default=0.5
        ),
        UIParam.user_name(22): Description("Signal-tail onset fraction"),
        UIParam.user_value(22): FloatParam(
            "pd_plateau_fraction", range_min=0.1, range_max=0.95, range_incr=0.01
        ),
    }
    if DUAL_REGION:
        protocol |= {
            UIParam.user_name(23): Description("Peripheral final flip, 0 inherits"),
            UIParam.user_value(23): IntParam(
                "pd_final_angle_periphery_deg",
                unit="deg",
                range_min=0,
                range_max=179,
                automatic=0,
            ),
        }

    if WAVE:
        protocol |= wave_entries()

    protocol |= user_entries(app, protocol)
    protocol.pop(UIParam.user_name(8), None)
    protocol.pop(UIParam.user_value(8), None)

    def evaluate(self, system, protocol):
        # A TR is one echo train. The design of the centre view plays one
        # train of the full length, startup echoes included, at a constant
        # refocusing angle: the maximum of the PD train, which no pulse of it
        # exceeds, so the train bounds its RF and lasts as long. The angles
        # are solved for only to state the effective TE, and to refuse a
        # prescription no train realizes, once per train under a warm
        # server. The RF of a train does not depend on the view order.
        a = arguments(self, protocol)
        one = {
            "n_dummy": 0,
            "ry": a["n_y"],
            "rz": a["n_z"],
            "n_acs_y": 0,
            "n_acs_z": 0,
            "ordering": "radial",
        }
        solved = protocol.arguments | one | {"flip_modulation": "pd"}
        # The effective TE depends on no TR: one solve serves every TR asked for.
        solved |= {"pd_optimize": False, "busse_optimize": False, "tr": None}
        one |= bounding_train(a, a["pd_maximum_angle_deg"])
        train = self.app(system, **(protocol.arguments | one))
        tr = train.duration()[0]
        tr_periphery = tr if a["tr_periphery"] is None else a["tr_periphery"]
        etl_periphery = a["etl"] if a["etl_periphery"] is None else a["etl_periphery"]
        # Identical trains are all full: the views fill whole trains.
        individual = (
            tr_periphery != tr
            or etl_periphery != a["etl"]
            or a["pd_final_angle_periphery_deg"] not in (None, a["pd_final_angle_deg"])
        )
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
        leading = dummies(a, times[0]) + references
        # The effective TE a train reaches follows from its angles.
        values = achieved(self, train)
        values[UIParam.TE] = effective_te(self.app, system, solved)
        return Evaluation(
            protocol.replace(values),
            leading * times[0] + sum(times),
            sequence=train,
            rf_layout=echo_train_layout(train, lengths, times),
        )
