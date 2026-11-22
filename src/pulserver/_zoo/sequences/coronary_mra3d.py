"""Whole-heart coronary MRA: pypulseqpp's 3D balanced SSFP in ECG-triggered segments with a T2 preparation, fat saturation and an image navigator."""

from pypulseqpp.sequences.sequence.bssfp3D_sequence import bssfp3d

from pulserver._zoo._segmented import segmented_evaluation, specialised
from pulserver._zoo._slab import slab
from pulserver._zoo._user import OPTIONS, OWN, user_entries
from pulserver.design import (
    BoolParam,
    ChoiceParam,
    Description,
    FloatParam,
    IntParam,
    SequencePlugin,
    StatedParam,
    TimeParam,
    option,
    recon_parameters,
)
from pulserver.protocol import Excitation, ImagingMode, TriggerType, TRPreset, UIParam

#: An image navigator read during each segment's catalyst ramp: the
#: reconstruction corrects each segment's respiratory displacement from it.
NAVIGATOR = option("navigator", True, **OPTIONS["navigator"])


class CoronaryMra3D(SequencePlugin):
    app = specialised(
        slab(bssfp3d),
        fixed={
            "gating": "prospective",
            "trigger_channel": "physio2",
            "n_phase_cycles": 1,
        },
        defaults={
            "fov_x": 0.30,
            "fov_y": 0.30,
            "n_x": 240,
            "n_y": 240,
            "n_z": 60,
            "slice_thickness": 1.5e-3,
            "readout_bandwidth_hz": 250e3,
            "ry": 2,
            "excitation": "slab",
            "flip_angle_deg": 70.0,
            "t2_prep_te": 0.05,
            "fat_saturation": True,
            "views_per_segment": 30,
            "trigger_delay": 0.6,
        },
        period=("heart_rate_bpm", 60, lambda bpm: 60.0 / bpm),
    )
    protocol = {
        UIParam.IMAGING_MODE: StatedParam(ImagingMode.THREE_D),
        UIParam.TRIGGER_TYPE: StatedParam(TriggerType.ECG),
        UIParam.EXCITATION: ChoiceParam(
            "excitation", Excitation, default=Excitation.SLAB
        ),
        UIParam.FLIP: FloatParam(
            "flip_angle_deg", unit="deg", range_min=1.0, range_max=110.0
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
        UIParam.FAT_SAT: BoolParam("fat_saturation", default=True),
        UIParam.TRIGGER_DELAY: FloatParam(
            "trigger_delay", unit="ms", scale=1e-3, range_min=0.0, range_max=2000.0
        ),
        UIParam.HEART_RATE: IntParam(
            "heart_rate_bpm", unit="bpm", range_min=30, range_max=200, default=60
        ),
        UIParam.PREP_TIME: TimeParam("t2_prep_te", range_min=0, range_max=100_000),
        UIParam.user_name(OWN): Description("Views per heartbeat"),
        UIParam.user_value(OWN): IntParam(
            "views_per_segment", range_min=4, range_max=256
        ),
    }

    protocol |= user_entries(app, protocol)

    def generate(self, system, protocol):
        [seq] = self.app(system, **protocol.arguments, navigator=NAVIGATOR)
        recon_parameters(seq, Gating="prospective", Navigator=int(NAVIGATOR))
        return seq

    def evaluate(self, system, protocol):
        return segmented_evaluation(self, system, protocol, navigator=NAVIGATOR)
