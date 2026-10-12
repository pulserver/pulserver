"""pypulseqpp's 3D fast spin echo bound to the scanner UI, its partitions counted by the number of slices."""

import functools

import numpy as np
import pypulseqpp as pp
from pypulseqpp.sequences.sequence.fse3D_sequence import (
    ORDERINGS,
    fse3d,
    shot_parameters,
)

from pulserver._zoo._evaluation import achieved, arguments, rf_layout, waved
from pulserver._zoo._pmc import navigated
from pulserver._zoo._slab import slab
from pulserver._zoo._user import OWN, shortest_at_zero, user_entries, wave_entries
from pulserver.design import (
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
NAVIGATOR = option("navigator", False)

#: The refocusing trains: ``"constant"``, at the refocusing angle;
#: ``"optimized"``, designed with blochsim's extended phase graphs around the
#: refocusing angle at the TE echo, one train shared by every shot; or
#: ``"dual_region"``, optimized trains individually parameterized for the
#: centre and the periphery of k-space, each shot's train a cubic step between
#: the two designs, which adds the TR and the echo train length at the
#: periphery as user entries, TR and ETL being the centre's.
REFOCUSING = option(
    "refocusing", "constant", choices=("constant", "optimized", "dual_region")
)
OPTIMIZED = REFOCUSING != "constant"
DUAL_REGION = REFOCUSING == "dual_region"

#: Wave-CAIPI: sinusoidal gradients on the phase- and partition-encoding axes
#: during each readout, the calibration region acquired first without them;
#: adds their amplitude and cycles as user entries.
WAVE = option("wave", False)

if OPTIMIZED:
    import blochsim  # noqa: F401  the trains are designed with it

#: Order of the views across shots, ``"radial"`` or ``"shuffling"``.
ORDERING = option("ordering", "radial", choices=ORDERINGS)


class Fse3D(SequencePlugin):
    app = (
        navigated(shortest_at_zero(slab(fse3d), "esp"))
        if NAVIGATOR
        else shortest_at_zero(slab(fse3d), "esp")
    )
    if OPTIMIZED:
        app = functools.partial(app, flip_modulation="optimized")
    app = functools.partial(app, ordering=ORDERING)
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
            "refocusing_angle_deg",
            unit="deg",
            range_min=60.0,
            range_max=180.0,
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

    if WAVE:
        protocol |= wave_entries()

    protocol |= user_entries(app, protocol)

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
            "flip_modulation": "constant",
            "ordering": "radial",
        }
        train = self.app(system, **(protocol.arguments | one))
        calibrating, imaging = pp.count_cartesian_plane_sampling(
            (a["n_y"], a["n_z"]),
            (a["ry"], a["rz"]),
            (a["n_acs_y"], a["n_acs_z"]),
            caipi_shift=a["caipi_shift"],
            partial_fourier=(a["partial_fourier_y"], a["partial_fourier_z"]),
            elliptical=True,
            elliptical_acs=a["elliptical_acs"],
        )
        views = calibrating + imaging
        if a["ordering"] == "shuffling" and a["ry"] * a["rz"] > 1:
            # A Poisson-disc draw is counted by its expected density.
            views = round(
                sum(
                    p.sum()
                    for p in pp.calc_cartesian_plane_sampling_probability(
                        (a["n_y"], a["n_z"]),
                        (a["ry"], a["rz"]),
                        (a["n_acs_y"], a["n_acs_z"]),
                        partial_fourier=(
                            a["partial_fourier_y"],
                            a["partial_fourier_z"],
                        ),
                        elliptical=True,
                        elliptical_acs=a["elliptical_acs"],
                        sampling="poisson",
                    )
                )
            )
        # Each train lasts its own TR, a cubic step from the centre's to the
        # periphery's; dummies and, under the wave, the wave-free trains of
        # the calibration region, each as long as the first, play the first.
        tr = train.duration()[0]
        tr_periphery = a["tr_periphery"]
        etl_periphery = a["etl_periphery"]
        lengths, times, _ = shot_parameters(
            views,
            a["etl"],
            a["etl"] if etl_periphery is None else etl_periphery,
            tr,
            tr if tr_periphery is None else tr_periphery,
        )
        raster = system.block_duration_raster
        times = [pp.round_to_raster(time, raster) for time in times]
        references = -(-calibrating // lengths[0]) if waved(a) else 0
        return Evaluation(
            protocol.replace(achieved(self, train)),
            (a["n_dummy"] + references) * times[0] + sum(times),
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
