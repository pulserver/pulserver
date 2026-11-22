"""Non-contrast inflow MRA: pypulseqpp's 3D balanced SSFP in respiratory-triggered segments, each after a slab-selective inversion and fat saturation."""

import functools
import inspect
import math
from collections.abc import Callable
from typing import Any

import numpy as np
from pypulseqpp.sequences.sequence.bssfp3D_sequence import bssfp3d

from pulserver._zoo._segmented import segmented_evaluation, specialised
from pulserver._zoo._slab import slab
from pulserver._zoo._user import OWN, user_entries
from pulserver.design import (
    BoolParam,
    ChoiceParam,
    Description,
    FloatParam,
    IntParam,
    SequencePlugin,
    StatedParam,
    TimeParam,
    recon_parameters,
)
from pulserver.protocol import Excitation, ImagingMode, TriggerType, TRPreset, UIParam

#: Breathing rate the respiratory trigger is timed to (breaths per minute).
RESPIRATORY_RATE_BPM = 15.0
#: Longitudinal relaxation time of blood (s) at the fields (T) it is tabulated
#: at, interpolated linearly in between and held beyond.
BLOOD_T1_S = {1.5: 1.55, 3.0: 1.932, 7.0: 2.587}


def venous_null_inversion_time(field_t: float) -> float:
    """Return the inversion time (s) that nulls blood inverted from equilibrium at ``field_t`` tesla, ``ln 2`` times its T1."""
    fields = sorted(BLOOD_T1_S)
    t1 = float(np.interp(field_t, fields, [BLOOD_T1_S[f] for f in fields]))
    return math.log(2.0) * t1


def _venous_null(app: Callable[..., Any]) -> Callable[..., Any]:
    """Return ``app`` with ``null_venous_blood``: when true, the inversion time is the one that nulls venous blood at the scanner's field."""
    signature = inspect.signature(app)
    flag = inspect.Parameter(
        "null_venous_blood", inspect.Parameter.KEYWORD_ONLY, default=True
    )

    @functools.wraps(app)
    def bound(
        system: Any, *args: Any, null_venous_blood: bool = True, **kwargs: Any
    ) -> Any:
        if null_venous_blood:
            kwargs["inversion_time"] = venous_null_inversion_time(system.B0)
        return app(system, *args, **kwargs)

    bound.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=[*signature.parameters.values(), flag]
    )
    return bound


class InflowMra3D(SequencePlugin):
    app = _venous_null(
        specialised(
            slab(bssfp3d),
            fixed={
                "gating": "prospective",
                "trigger_channel": "physio1",
                "n_phase_cycles": 1,
                "navigator": False,
                "t2_prep_te": 0.0,
            },
            defaults={
                "fov_x": 0.36,
                "fov_y": 0.36,
                "n_x": 240,
                "n_y": 240,
                "n_z": 64,
                "slice_thickness": 1.5e-3,
                "readout_bandwidth_hz": 250e3,
                "ry": 2,
                "excitation": "slab",
                "flip_angle_deg": 90.0,
                "fat_saturation": True,
                "views_per_segment": 80,
                "inversion_time": 1.2,
                "inversion_thickness": 0.15,
                "inversion_axis": "x",
                "inversion_offset": -0.05,
            },
            period=(
                "breaths_per_minute",
                RESPIRATORY_RATE_BPM,
                lambda rate: 60.0 / rate,
            ),
        )
    )
    protocol = {
        UIParam.IMAGING_MODE: StatedParam(ImagingMode.THREE_D),
        UIParam.TRIGGER_TYPE: StatedParam(TriggerType.RESPIRATORY),
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
        UIParam.PREP_TIME: TimeParam(
            "inversion_time", range_min=100_000, range_max=3_000_000
        ),
        UIParam.user_name(OWN): Description("Views per breath"),
        UIParam.user_value(OWN): IntParam(
            "views_per_segment", range_min=4, range_max=512
        ),
        UIParam.user_name(OWN + 1): Description("Inversion slab, mm"),
        UIParam.user_value(OWN + 1): FloatParam(
            "inversion_thickness",
            unit="mm",
            scale=1e-3,
            range_min=0.0,
            range_max=500.0,
        ),
        UIParam.user_name(OWN + 2): Description(
            "Inversion slab offset along readout, mm"
        ),
        UIParam.user_value(OWN + 2): FloatParam(
            "inversion_offset",
            unit="mm",
            scale=1e-3,
            range_min=-250.0,
            range_max=250.0,
        ),
        UIParam.user_name(OWN + 3): Description("TI nulls venous blood at B0"),
        UIParam.user_value(OWN + 3): BoolParam("null_venous_blood", default=True),
    }

    protocol |= user_entries(app, protocol)

    def generate(self, system, protocol):
        [seq] = self.app(system, **protocol.arguments)
        recon_parameters(seq, Gating="prospective", Navigator=0)
        return seq

    def evaluate(self, system, protocol):
        return segmented_evaluation(self, system, protocol)
