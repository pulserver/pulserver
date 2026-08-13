"""RF-spoiled 2D Cartesian gradient echo, multi-slice.

One frequency-encoded line per repetition, from a slice-selective SLR
excitation. Phase encoding may be uniformly undersampled with a fully sampled
autocalibration block, and the readout may be a partial echo; both are what
:mod:`pulserver.reczoo.gre_2d` reads back.

Transverse magnetisation is destroyed by quadratic RF spoiling together with
the residual dephasing the readout leaves on its own axis, so the readout's
``spoiling_cycles`` is the whole gradient spoiling: adding a second spoiler
outside the module would lengthen the repetition the module has already
budgeted.

Two entry points, one implementation:

``main``
    Explicit keyword controls, returns the :class:`pulserver.pypulseq.Sequence`.
``PLUGIN``
    The same sequence behind the scanner protocol contract, for the bridge.
    Running this module as a script writes a ``.seq`` from the same controls.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

import pulserver.design as design
import pulserver.pypulseq as pp
from pulserver import (
    Description,
    DropdownFloatParam,
    DropdownIntParam,
    Sequence,
    TypeinFloatParam,
    UIParam,
    Validate,
    dict_to_protocol,
    params,
    protocol_to_dict,
    run_cli,
)


@dataclass
class _Plan:
    """One designed repetition and the encoding plan that repeats it.

    Whatever both the scan loop and the feasibility check need, designed once:
    the excitation and readout modules, the phase-encode lines the sampling
    asks for, and the total scan time they add up to.
    """

    excitation: Any
    readout: Any
    fov: tuple[float, float]
    sampled_lines: list[int]
    duration: float


def _plan(
    system: pp.Opts,
    *,
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 128,
    n_y: int = 128,
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    flip_angle_deg: float = 12.0,
    te: float | None = 8e-3,
    tr: float | None = 250e-3,
    readout_bandwidth_hz: float = 250e3,
    partial_echo: float = 1.0,
    acceleration: int = 1,
    n_acs: int = 24,
    spoiling_cycles: float = 4.0,
) -> _Plan:
    """Design one repetition, which is where every parameter is validated.

    Building the excitation and readout *is* the feasibility check: a TE or TR
    shorter than one repetition can achieve makes :class:`design.LineReadout2D`
    raise ``ValueError``, as does any out-of-range matrix, fov or fraction. So
    the same call that the scan loop is built from tells :meth:`Gre2D.validate_protocol`
    whether the protocol is feasible and, when it is, how long it takes -- there
    is no second timing path to drift out of step with the sequence.
    """
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

    excitation = design.SpatialSelectiveExcitation(
        system, flip_angle_deg, slice_thickness
    )
    readout = design.LineReadout2D(
        system,
        excitation.rf,
        excitation.gz,
        excitation.gz_reph,
        fov=(fov_x, fov_y),
        matrix=(n_x, n_y),
        te=te,
        tr=None if tr is None else tr / n_slices,
        partial_echo=partial_echo,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=spoiling_cycles,
        labels=("LIN", "SLC", "REF", "IMA"),
    )

    # One repetition per acquired line per slice; the readout has already
    # padded itself to the per-slice TR, so the whole scan is simply their sum.
    sampled_lines = pp.calc_sampled_lines(n_y, acceleration, n_acs)
    duration = len(sampled_lines) * n_slices * readout.duration

    return _Plan(
        excitation=excitation,
        readout=readout,
        fov=(fov_x, fov_y),
        sampled_lines=sampled_lines,
        duration=duration,
    )


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "gre_2d.seq",
    *,
    system: pp.Opts | None = None,
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 128,
    n_y: int = 128,
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_gap: float = 0.0,
    slice_order: str = "interleaved",
    flip_angle_deg: float = 12.0,
    te: float | None = 8e-3,
    tr: float | None = 250e-3,
    readout_bandwidth_hz: float = 250e3,
    partial_echo: float = 1.0,
    acceleration: int = 1,
    n_acs: int = 24,
    rf_spoiling_increment_deg: float = 117.0,
    spoiling_cycles: float = 4.0,
) -> pp.Sequence:
    """Create an RF-spoiled 2D Cartesian gradient-echo sequence.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'gre_2d.seq'.
    system : pypulseq.Opts, optional
        System limits. Default is `pp.Opts()`.
    fov : float or tuple of float, optional
        In-plane field of view in meters. If a single value, it is used for
        both x and y. If a tuple, it is (fov_x, fov_y). Default is 220e-3.
    n_x : int, optional
        Number of readout samples. Default is 128.
    n_y : int, optional
        Number of phase encoding steps. Default is 128.
    n_slices : int, optional
        Number of slices. Default is 1.
    slice_thickness : float, optional
        Slice thickness in meters. Default is 5e-3.
    slice_gap : float, optional
        Gap between adjacent slices in meters. Default is 0.0.
    slice_order : str, optional
        Order the slices are excited in, as accepted by
        `pp.calc_traversal_order`. Default is 'interleaved'.
    flip_angle_deg : float, optional
        Excitation flip angle in degrees. Default is 12.0.
    te : float or None, optional
        Echo time in seconds. None is as short as possible. Default is 8e-3.
    tr : float or None, optional
        Repetition time in seconds, between successive excitations of the same
        slice. None is as short as possible. Default is 250e-3.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth in Hz. The achieved value is reported by
        the readout module and is generally lower. Default is 250e3.
    partial_echo : float, optional
        Fraction of the full echo acquired, in (0.5, 1]. Truncates the samples
        before the echo, which shortens the minimum TE. Default is 1.0.
    acceleration : int, optional
        Uniform phase-encode undersampling factor. Default is 1.
    n_acs : int, optional
        Number of fully sampled autocalibration lines at the center of
        k-space. Default is 24.
    rf_spoiling_increment_deg : float, optional
        Quadratic RF spoiling phase increment in degrees. Default is 117.0.
    spoiling_cycles : float, optional
        Cycles of dephasing left on the readout axis at the end of each
        repetition, counted across one voxel. Default is 4.0.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The GRE sequence object.
    """
    system = pp.Opts() if system is None else system

    # Design one repetition -- and, in doing so, validate TE, TR and the rest.
    plan = _plan(
        system,
        fov=fov,
        n_x=n_x,
        n_y=n_y,
        n_slices=n_slices,
        slice_thickness=slice_thickness,
        flip_angle_deg=flip_angle_deg,
        te=te,
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        partial_echo=partial_echo,
        acceleration=acceleration,
        n_acs=n_acs,
        spoiling_cycles=spoiling_cycles,
    )
    excitation, readout = plan.excitation, plan.readout
    fov_x, fov_y = plan.fov
    sampled_lines = plan.sampled_lines

    lin_label, slc_label, ref_label, ima_label = readout.adc_labels
    wait_te = getattr(readout, "wait_te", None)
    wait_tr = getattr(readout, "wait_tr", None)

    # The rest of the encoding plan: in which order the slices are excited, at
    # what offset, and the RF-spoiling phase every repetition is played with.
    acs_start = max(0, n_y // 2 - n_acs // 2)
    acs_stop = min(n_y, acs_start + n_acs)
    slice_indices = pp.calc_traversal_order(n_slices, slice_order)
    slice_positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
        slice_thickness + slice_gap
    )
    rf_phases = pp.make_rf_spoiling_schedule(
        len(sampled_lines) * n_slices,
        increment=np.deg2rad(rf_spoiling_increment_deg),
    )

    seq = pp.Sequence(system)
    for i_phase, line in enumerate(sampled_lines):
        ky = (line - n_y / 2) / (n_y / 2)
        is_calibration = acs_start <= line < acs_stop
        on_grid = line % acceleration == 0
        # A calibration line that also lies on the acceleration grid is both
        # reference and imaging data; one that does not is reference only.
        # Label state persists, so both flags are written every repetition.
        lin_label.value = line
        ref_label.value = int(is_calibration and not on_grid)
        ima_label.value = int(is_calibration and on_grid)

        for i_slice in slice_indices:
            rf_phase = rf_phases[i_phase * n_slices + i_slice]
            readout.rf.freq_offset = excitation.gz.amplitude * slice_positions[i_slice]
            readout.rf.phase_offset = (
                rf_phase - 2 * np.pi * readout.rf.freq_offset * readout.rf.center
            )
            readout.adc.phase_offset = rf_phase
            slc_label.value = int(i_slice)

            seq.add_block(readout.rf, readout.gz)
            if wait_te is not None:
                seq.add_block(wait_te, readout.gz_reph)
                seq.add_block(readout.gx_pre, pp.scale_grad(readout.gy_pre, ky))
            else:
                seq.add_block(
                    readout.gx_pre,
                    pp.scale_grad(readout.gy_pre, ky),
                    readout.gz_reph,
                )
            seq.add_block(readout.gx, readout.adc, *readout.adc_labels)
            seq.add_block(readout.gx_spoil, pp.scale_grad(readout.gy_rew, ky))
            if wait_tr is not None:
                seq.add_block(wait_tr)

    is_ok, error_report = seq.check_timing()
    if not is_ok:
        print(f"Timing check failed: {len(error_report)} errors")

    if test_report:
        print(seq.test_report())

    if plot:
        seq.plot()

    slab_thickness = n_slices * (slice_thickness + slice_gap) - slice_gap
    seq.set_definition(key="FOV", value=[fov_x, fov_y, slab_thickness])
    seq.set_definition(key="Name", value="gre_2d")
    seq.set_definition(key="TE", value=readout.echo_time)
    seq.set_definition(key="TR", value=n_slices * readout.duration)

    if write_seq:
        seq.write(seq_filename)

    return seq


class Gre2D(Sequence):
    """The 2D gradient echo behind the scanner protocol contract."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        """Return the protocol the scanner UI is built from."""
        del opts
        return protocol_to_dict(
            {
                UIParam.TE: DropdownFloatParam(
                    value=8.0,
                    min=1.0,
                    max=80.0,
                    incr=0.1,
                    unit="ms",
                    options=[3.0, 5.0, 8.0, 15.0, 30.0],
                    validate=Validate.NONE,
                ),
                UIParam.TR: DropdownFloatParam(
                    value=250.0,
                    min=5.0,
                    max=5000.0,
                    incr=0.1,
                    unit="ms",
                    options=[150.0, 250.0, 500.0, 1000.0, 2000.0],
                    validate=Validate.NONE,
                ),
                UIParam.FLIP: DropdownFloatParam(
                    value=12.0,
                    min=1.0,
                    max=90.0,
                    incr=1.0,
                    unit="deg",
                    options=[5.0, 12.0, 30.0, 60.0, 90.0],
                    validate=Validate.NONE,
                ),
                UIParam.FOV: DropdownFloatParam(
                    value=220.0,
                    min=80.0,
                    max=500.0,
                    incr=1.0,
                    unit="mm",
                    options=[180.0, 220.0, 280.0, 340.0, 500.0],
                    validate=Validate.NONE,
                ),
                UIParam.PHASE_FOV: DropdownFloatParam(
                    value=220.0,
                    min=80.0,
                    max=500.0,
                    incr=1.0,
                    unit="mm",
                    options=[180.0, 220.0, 280.0, 340.0, 500.0],
                    validate=Validate.NONE,
                ),
                UIParam.SLICE_THICKNESS: DropdownFloatParam(
                    value=5.0,
                    min=1.0,
                    max=20.0,
                    incr=0.5,
                    unit="mm",
                    options=[1.0, 3.0, 5.0, 8.0, 10.0],
                    validate=Validate.NONE,
                ),
                UIParam.SLICE_SPACING: DropdownFloatParam(
                    value=5.0,
                    min=1.0,
                    max=20.0,
                    incr=0.5,
                    unit="mm",
                    options=[1.0, 3.0, 5.0, 8.0, 10.0],
                    validate=Validate.NONE,
                ),
                UIParam.NX: DropdownIntParam(
                    value=128,
                    min=16,
                    max=512,
                    incr=1,
                    options=[64, 128, 192, 256, 384],
                    validate=Validate.NONE,
                ),
                UIParam.NY: DropdownIntParam(
                    value=128,
                    min=16,
                    max=512,
                    incr=1,
                    options=[64, 128, 192, 256, 384],
                    validate=Validate.NONE,
                ),
                UIParam.NSLICES: DropdownIntParam(
                    value=1,
                    min=1,
                    max=128,
                    incr=1,
                    options=[1, 5, 10, 20, 40],
                    validate=Validate.NONE,
                ),
                UIParam.BANDWIDTH: TypeinFloatParam(
                    value=250e3,
                    min=5e3,
                    max=500e3,
                    incr=100.0,
                    unit="Hz",
                    validate=Validate.NONE,
                ),
                UIParam.RY: TypeinFloatParam(
                    value=1.0,
                    min=1.0,
                    max=8.0,
                    incr=1.0,
                    unit="",
                    validate=Validate.NONE,
                ),
                UIParam.user_name(0): Description(text="ACS lines"),
                UIParam.user_value(0): TypeinFloatParam(
                    value=24.0,
                    min=0.0,
                    max=512.0,
                    incr=1.0,
                    unit="lines",
                    validate=Validate.NONE,
                ),
                UIParam.user_name(1): Description(text="Partial echo"),
                UIParam.user_value(1): TypeinFloatParam(
                    value=1.0,
                    min=0.55,
                    max=1.0,
                    incr=0.05,
                    unit="",
                    validate=Validate.NONE,
                ),
            }
        )

    def validate_protocol(self, opts: pp.Opts, protocol: dict[str, dict]) -> dict:
        """Report whether the protocol is feasible, and how long it will take.

        Designing one repetition through :func:`_plan` is the whole check: the
        same construction the sequence is built from raises ``ValueError`` on an
        infeasible TE or TR, and otherwise reports the scan duration directly.
        """
        kwargs = _main_kwargs(opts, protocol)
        try:
            plan = _plan(
                opts,
                fov=kwargs["fov"],
                n_x=kwargs["n_x"],
                n_y=kwargs["n_y"],
                n_slices=kwargs["n_slices"],
                slice_thickness=kwargs["slice_thickness"],
                flip_angle_deg=kwargs["flip_angle_deg"],
                te=kwargs["te"],
                tr=kwargs["tr"],
                readout_bandwidth_hz=kwargs["readout_bandwidth_hz"],
                partial_echo=kwargs["partial_echo"],
                acceleration=kwargs["acceleration"],
                n_acs=kwargs["n_acs"],
            )
        except ValueError as error:
            return {"valid": False, "duration": None, "info": str(error)}

        return {
            "valid": True,
            "duration": plan.duration,
            "info": f"TA = {plan.duration:.1f} s at {plan.readout.bandwidth_hz * 1e-3:.1f} kHz",
        }

    def make_sequence(
        self, opts: pp.Opts, protocol: dict[str, dict], output_path: str
    ) -> None:
        """Build the sequence and write it to ``output_path``."""
        seq = main(**_main_kwargs(opts, protocol))
        seq.write(output_path, remove_duplicates=False, check_timing=False)


def _main_kwargs(opts: pp.Opts, protocol: dict[str, dict]) -> dict:
    """Translate a scanner protocol into :func:`main` keyword arguments."""
    prot = dict_to_protocol(protocol)
    n_y = params.param_int(prot, UIParam.NY)
    return {
        "system": opts,
        "fov": (
            params.param_float(prot, UIParam.FOV) * 1e-3,
            params.phase_fov_mm_from_protocol(prot) * 1e-3,
        ),
        "n_x": params.param_int(prot, UIParam.NX),
        "n_y": n_y,
        "n_slices": params.param_int(prot, UIParam.NSLICES),
        "slice_thickness": params.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3,
        "slice_gap": max(
            0.0,
            (
                params.param_float(prot, UIParam.SLICE_SPACING)
                - params.param_float(prot, UIParam.SLICE_THICKNESS)
            )
            * 1e-3,
        ),
        "flip_angle_deg": params.param_float(prot, UIParam.FLIP),
        "te": params.param_float(prot, UIParam.TE) * 1e-3,
        "tr": params.param_float(prot, UIParam.TR) * 1e-3,
        "readout_bandwidth_hz": params.param_float_optional(
            prot, UIParam.BANDWIDTH, 250e3
        ),
        "partial_echo": params.user_float(prot, 1, 1.0),
        "acceleration": max(
            1, round(params.param_float_optional(prot, UIParam.RY, 1.0))
        ),
        "n_acs": params.acs_lines_from_protocol(prot, n_y, 0),
    }


PLUGIN = Gre2D()


def get_default_protocol(opts):
    """Bridge entry point: the plugin's default protocol."""
    return PLUGIN.get_default_protocol(opts)


def validate_protocol(opts, protocol):
    """Bridge entry point: protocol feasibility and scan duration."""
    return PLUGIN.validate_protocol(opts, protocol)


def make_sequence(opts, protocol, output_path):
    """Bridge entry point: write the ``.seq`` file."""
    return PLUGIN.make_sequence(opts, protocol, output_path)


_ARG_MAP = [
    ("--te-ms", UIParam.TE, float, "Echo time [ms]"),
    ("--tr-ms", UIParam.TR, float, "Repetition time [ms]"),
    ("--flip-deg", UIParam.FLIP, float, "Flip angle [deg]"),
    ("--fov-mm", UIParam.FOV, float, "Readout FOV [mm]"),
    ("--phase-fov-mm", UIParam.PHASE_FOV, float, "Phase-encode FOV [mm]"),
    ("--slice-thickness-mm", UIParam.SLICE_THICKNESS, float, "Slice thickness [mm]"),
    ("--slice-spacing-mm", UIParam.SLICE_SPACING, float, "Slice spacing [mm]"),
    ("--nx", UIParam.NX, int, "Readout matrix size"),
    ("--ny", UIParam.NY, int, "Phase-encode matrix size"),
    ("--nslices", UIParam.NSLICES, int, "Number of slices"),
    ("--bandwidth-hz", UIParam.BANDWIDTH, float, "Requested receiver bandwidth [Hz]"),
    ("--ry", UIParam.RY, float, "Phase-encode undersampling factor"),
    ("--acs-lines", UIParam.user_value(0), float, "Number of ACS lines"),
    (
        "--partial-echo",
        UIParam.user_value(1),
        float,
        "Acquired echo fraction in (0.5, 1]",
    ),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description="Generate a 2D Cartesian gradient-echo .seq offline.",
            default_output="gre_2d.seq",
        )
    )
