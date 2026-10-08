"""pypulseqpp's 3D gradient-echo EPI bound to the scanner UI, its partitions counted by the number of slices."""

import functools

import numpy as np
import pypulseqpp as pp
from pypulseqpp import sequences
from pypulseqpp.sequences.sequence.epi3D_sequence import (
    FAT_SHIFT_PPM,
    HARD_PULSE_DURATION,
    MAX_GRAD,
    MAX_SLEW,
    NAVIGATOR_LINES,
    PULSE_DURATION,
    SPOILING_CYCLES,
    TIME_BW_PRODUCT,
    caipi_shift,
    epi3d,
    shell_bases,
    train_lines,
)

from pulserver._zoo._evaluation import arguments, dummies, rf_layout
from pulserver._zoo._slab import slab
from pulserver._zoo._user import user_entries
from pulserver.design import (
    Evaluation,
    FloatParam,
    IntParam,
    SequencePlugin,
    StatedParam,
    TimeParam,
)
from pulserver.protocol import ImagingMode, TEPreset, TRPreset, UIParam

#: The slab's excitation, one of ``sequences.EXCITATIONS``. ``"spsp"`` excites
#: water alone, which keeps fat, shifted many pixels along the phase encode,
#: out of the image without a fat saturation pulse.
EXCITATION = "spsp"


class Epi3D(SequencePlugin):
    # Twofold readout oversampling keeps the ramp-sampled flat top within the
    # spacing of the readout field of view, so it can be resampled onto a grid.
    app = slab(
        functools.partial(epi3d, readout_oversampling=2.0, excitation=EXCITATION)
    )
    protocol = {
        UIParam.IMAGING_MODE: StatedParam(ImagingMode.THREE_D),
        UIParam.FLIP: FloatParam(
            "flip_angle_deg", unit="deg", range_min=1.0, range_max=90.0
        ),
        UIParam.TE: TimeParam(
            "te",
            range_min=1000,
            range_max=200000,
            presets={TEPreset.MINIMUM: None},
            default=TEPreset.MINIMUM,
        ),
        UIParam.TR: TimeParam(
            "tr",
            range_min=10000,
            range_max=60_000_000,
            presets={TRPreset.MINIMUM: None},
        ),
        UIParam.BANDWIDTH: FloatParam(
            "readout_bandwidth_hz", unit="Hz", range_min=1e3, range_max=2e6
        ),
        UIParam.FOV: FloatParam(
            "fov_x", unit="mm", scale=1e-3, range_min=50.0, range_max=500.0
        ),
        UIParam.PHASE_FOV: FloatParam(
            "fov_y", unit="mm", scale=1e-3, range_min=50.0, range_max=500.0
        ),
        UIParam.NX: IntParam("n_x", range_min=32, range_max=512, range_incr=2),
        UIParam.NY: IntParam("n_y", range_min=32, range_max=512, range_incr=2),
        UIParam.NSLICES: IntParam("n_z", range_min=4, range_max=128),
        UIParam.SLICE_THICKNESS: FloatParam(
            "slice_thickness",
            unit="mm",
            scale=1e-3,
            range_min=0.1,
            range_max=20.0,
            range_incr=0.1,
        ),
        UIParam.NUM_FRAMES: IntParam("n_frames", range_min=1, range_max=1000),
        UIParam.NUM_SHOTS: IntParam("n_shots", range_min=1, range_max=16),
        UIParam.RY: IntParam("ry", range_min=1, range_max=4),
        UIParam.RZ: IntParam("rz", range_min=1, range_max=4),
    }

    protocol |= user_entries(app, protocol)

    def evaluate(self, system, protocol):
        # A TR is a volume: one shot of every shell, each an excitation, the
        # navigator lines and a train. The shot is built from the modules the
        # function builds it from, at its prescription, and the volumes the
        # scan plays are counted from the function's own rules.
        a = arguments(self, protocol)
        a["fov_z"] = a["n_z"] * a.pop("slice_thickness")
        system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
        raster = system.block_duration_raster
        fov = (a["fov_x"], a["fov_y"], a["fov_z"])
        matrix = (a["n_x"], a["n_y"], a["n_z"])
        n_y, n_z, ry, rz, n_shots = (
            a["n_y"],
            a["n_z"],
            a["ry"],
            a["rz"],
            a["n_shots"],
        )
        exc = sequences.make_excitation(
            system,
            a["excitation"],
            a["flip_angle_deg"],
            a["fov_z"],
            duration_s=PULSE_DURATION,
            time_bw_product=TIME_BW_PRODUCT,
            hard_duration_s=HARD_PULSE_DURATION,
            fat_shift_ppm=FAT_SHIFT_PPM,
        )
        gz = [] if getattr(exc, "gz", None) is None else [exc.gz]
        start, etl = train_lines(n_y, ry, n_shots, a["partial_fourier_y"])
        shells = shell_bases(n_z, rz, a["partial_fourier_z"])
        # The first shot of the first shell, whose train every shot's has the
        # duration of.
        lines = start + np.arange(etl) * n_shots * ry
        lattice = n_z // 2 + caipi_shift(ry, rz) * ((lines - n_y // 2) // ry)
        epi = sequences.EpiReadout3D(
            system,
            exc.rf,
            *gz,
            order=np.column_stack(
                (np.arange(etl) * n_shots * ry, (lattice - shells[0]) % rz)
            ),
            te=a["te"],
            te_line=(n_y // 2 - start) // ry / n_shots,
            navigator_lines=NAVIGATOR_LINES,
            echo_shifts=n_shots,
            fov=fov,
            matrix=matrix,
            oversampling=a["readout_oversampling"],
            readout_bandwidth_hz=a["readout_bandwidth_hz"],
            spoiling_cycles=SPOILING_CYCLES,
            labels=("LIN", "PAR"),
        )
        # A shot is the train and a closing delay of at least one raster.
        volume = n_shots * len(shells)
        shortest = epi.duration + raster
        pad = raster
        if a["tr"] is not None:
            per_shot = a["tr"] / volume
            if per_shot < shortest - 1e-9:
                raise ValueError(
                    f"the requested TR of {a['tr'] * 1e3:.3f} ms is shorter than "
                    f"the {volume * shortest * 1e3:.3f} ms the {volume} shots of "
                    "a volume take"
                )
            pad += pp.round_to_raster(per_shot - shortest, raster)
        shot = shortest - raster + pad
        tr = volume * shot
        # The time series and the reference volume, with the phase encode
        # reversed, each play their dummies first: whole volumes in a series,
        # shots otherwise.
        n_dummy = a["n_dummy"]
        if n_dummy is None:
            # As the function plays them: every shot excites the slab.
            excitations = dummies(a, shot)
            n_dummy = -(-excitations // volume) if a["n_frames"] > 1 else excitations
        dummy_shots = n_dummy * (volume if a["n_frames"] > 1 else 1)
        duration = (2 * dummy_shots + (1 + a["n_frames"]) * volume) * shot
        # An undersampled scan is preceded by a gradient echo at every view
        # of the central rectangle of the phase and partition encodes.
        if ry > 1 or rz > 1 or a["partial_fourier_y"] < 1 or a["partial_fourier_z"] < 1:
            views = _centre(n_y, a["n_acs_y"]) * _centre(n_z, a["n_acs_z"])
            if views:
                gre = sequences.LineReadout3D(
                    system,
                    exc.rf,
                    *gz,
                    fov=fov,
                    matrix=matrix,
                    oversampling=a["readout_oversampling"],
                    readout_bandwidth_hz=a["readout_bandwidth_hz"],
                    spoiling_cycles=SPOILING_CYCLES,
                    labels=("LIN", "PAR"),
                )
                duration += views * gre.duration
        values = {
            UIParam.TE: epi.echo_time,
            UIParam.TR: tr,
            UIParam.BANDWIDTH: 1.0 / epi.adc.dwell,
        }
        excitation = pp.Sequence(system)
        excitation.add_block(exc.rf, *gz)
        excitation.set_definition(key="TR", value=tr)
        return Evaluation(
            protocol.replace(values),
            duration,
            rf_layout=rf_layout(excitation, scaled=True, copies=volume),
        )


def _centre(n: int, extent: int) -> int:
    """Return the views a calibration of ``extent`` lines about the centre of ``n`` reads."""
    low = n // 2 - extent // 2
    return len(range(max(low, 0), min(low + extent, n)))
