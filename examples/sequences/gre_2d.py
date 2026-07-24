"""Standalone Cartesian 2D GRE sequence plugin for pulserver.

This module is a self-contained example of a ``pulserver`` sequence plugin.
It implements the three mandatory module-level entry points required by the
bridge dispatcher:

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

The sequence is a true minimum-TE / minimum-TR Cartesian gradient-echo with:

- Sinc-pulse slice-selective excitation (2 ms, TBW = 4, apodization = 0.5).
- Bandwidth-driven readout timing quantised to both the gradient and ADC
  rasters.
- RF spoiling with a 117 ° quadratic phase increment.
- Bridged x-spoiler (``make_extended_trapezoid_area``) for smooth crusher
  waveforms matching the GE EPIC fixture.
- Optional phase-oversampling / partial Fourier with ACS lines.
- Phase/frequency axis swap.

Usage
-----
Register the plugin in the interpreter tree by symlinking (or copying) this
file to the ``sequences/src/`` directory and creating a numbered alias::

    cp gre_2d.py pulserver-interpreter/tree/pulserver/sequences/src/gre_2d.py
    ln -sf src/gre_2d.py  pulserver-interpreter/tree/pulserver/sequences/sequence5.py

Or pass the file directly to the bridge host for testing::

    python -m pulserver.bridge --plugin examples/sequences/gre_2d.py --action default_protocol
"""

from __future__ import annotations

import sys

import numpy as np
import pulserver.design as design
import pulserver.io as pio
import pulserver.pypulseq as pp
from pulserver import (
    BoolParam,
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

DEFAULT_BANDWIDTH_HZ_PX = 125_000.0
RF_SPOILING_INCREMENT_RAD = np.deg2rad(117.0)


class GrePulseqSequence(Sequence):
    """Generate a true Cartesian 2D GRE sequence using pypulseq."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        del opts
        protocol = {
            UIParam.TE: DropdownFloatParam(
                value=8.0,
                min=2.0,
                max=80.0,
                incr=0.1,
                unit="ms",
                options=[3.0, 5.0, 8.0, 15.0, 30.0],
                validate=Validate.NONE,
            ),
            UIParam.TR: DropdownFloatParam(
                value=250.0,
                min=110.0,
                max=2000.0,
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
                value=64,
                min=16,
                max=512,
                incr=1,
                options=[64, 128, 192, 256, 384],
                validate=Validate.NONE,
            ),
            UIParam.NY: DropdownIntParam(
                value=64,
                min=8,
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
                value=DEFAULT_BANDWIDTH_HZ_PX,
                min=5_000.0,
                max=500_000.0,
                incr=100.0,
                unit="Hz/px",
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
            UIParam.SWAP_PHASE_FREQ: BoolParam(value=False, validate=Validate.NONE),
            UIParam.user_name(0): Description(text="ACS lines"),
            UIParam.user_value(0): TypeinFloatParam(
                value=24.0,
                min=0.0,
                max=512.0,
                incr=1.0,
                unit="lines",
                validate=Validate.NONE,
            ),
        }
        return protocol_to_dict(protocol)

    def validate_protocol(self, opts: pp.Opts, protocol: dict[str, dict]) -> dict:
        prot = dict_to_protocol(protocol)

        te_s = params.param_float(prot, UIParam.TE) * 1e-3
        tr_s = params.param_float(prot, UIParam.TR) * 1e-3
        flip_deg = params.param_float(prot, UIParam.FLIP)
        fov_ro_m = params.param_float(prot, UIParam.FOV) * 1e-3
        fov_pe_m = params.phase_fov_mm_from_protocol(prot) * 1e-3
        slice_thickness_m = params.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
        nx_ro = params.param_int(prot, UIParam.NX)
        ny_pe = params.param_int(prot, UIParam.NY)
        nslices = params.param_int(prot, UIParam.NSLICES)
        bandwidth_hz_px = params.param_float_optional(prot, UIParam.BANDWIDTH, DEFAULT_BANDWIDTH_HZ_PX)
        ry = max(1, int(round(params.param_float_optional(prot, UIParam.RY, 1.0))))
        acs_lines = params.acs_lines_from_protocol(prot, ny_pe, 0)
        ro_axis, pe_axis = params.resolve_readout_phase_axes(prot)

        if te_s <= 0.0 or tr_s <= 0.0:
            return {"valid": False, "duration": None, "info": "TE and TR must be > 0"}
        if fov_ro_m <= 0.0 or fov_pe_m <= 0.0 or slice_thickness_m <= 0.0:
            return {"valid": False, "duration": None, "info": "FOV and slice thickness must be > 0"}
        if not (0.0 < flip_deg <= 180.0):
            return {"valid": False, "duration": None, "info": "Flip angle must be in (0, 180] deg"}
        if nx_ro < 1 or ny_pe < 1 or nslices < 1:
            return {"valid": False, "duration": None, "info": "NX, NY, and NSLICES must be >= 1"}
        if bandwidth_hz_px <= 0.0:
            return {"valid": False, "duration": None, "info": "Bandwidth must be > 0"}

        # Check TE feasibility and get the per-slice block duration (nslices=1 so
        # that TR validity is not conflated with TE validity at this stage).
        timing = _compute_timing(
            opts=opts,
            flip_deg=flip_deg,
            fov_ro_m=fov_ro_m,
            fov_pe_m=fov_pe_m,
            slice_thickness_m=slice_thickness_m,
            nx_ro=nx_ro,
            ny_pe=ny_pe,
            nslices=1,
            bandwidth_hz_px=bandwidth_hz_px,
            ro_axis=ro_axis,
            pe_axis=pe_axis,
            te_s=te_s,
            tr_s=tr_s,
        )
        if timing is None:
            return {
                "valid": False,
                "duration": None,
                "info": "TE or TR too short for gradients and readout timing",
            }

        # Check that the requested number of slices fits within TR.
        min_block_s = timing["min_block_s"]
        nslices_max = int(tr_s / min_block_s)
        if nslices > nslices_max:
            return {
                "valid": False,
                "duration": None,
                "info": (
                    f"TR {tr_s * 1e3:.1f} ms too short for {nslices} slices "
                    f"(Treadout = {min_block_s * 1e3:.1f} ms, max {nslices_max} slice(s))"
                ),
            }

        sampling = design.make_cartesian_sampling((nx_ro, ny_pe), acceleration=ry, calibration=acs_lines)
        duration_s = tr_s * float(len(sampling))
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)

        te_s = params.param_float(prot, UIParam.TE) * 1e-3
        tr_s = params.param_float(prot, UIParam.TR) * 1e-3
        flip_deg = params.param_float(prot, UIParam.FLIP)
        fov_ro_m = params.param_float(prot, UIParam.FOV) * 1e-3
        fov_pe_m = params.phase_fov_mm_from_protocol(prot) * 1e-3
        slice_thickness_m = params.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
        slice_spacing_m = params.param_float(prot, UIParam.SLICE_SPACING) * 1e-3
        nx_ro = params.param_int(prot, UIParam.NX)
        ny_pe = params.param_int(prot, UIParam.NY)
        nslices = params.param_int(prot, UIParam.NSLICES)
        bandwidth_hz_px = params.param_float_optional(prot, UIParam.BANDWIDTH, DEFAULT_BANDWIDTH_HZ_PX)
        ry = max(1, int(round(params.param_float_optional(prot, UIParam.RY, 1.0))))
        acs_lines = params.acs_lines_from_protocol(prot, ny_pe, 0)
        ro_axis, pe_axis = params.resolve_readout_phase_axes(prot)

        timing = _compute_timing(
            opts=opts,
            flip_deg=flip_deg,
            fov_ro_m=fov_ro_m,
            fov_pe_m=fov_pe_m,
            slice_thickness_m=slice_thickness_m,
            nx_ro=nx_ro,
            ny_pe=ny_pe,
            nslices=nslices,
            bandwidth_hz_px=bandwidth_hz_px,
            ro_axis=ro_axis,
            pe_axis=pe_axis,
            te_s=te_s,
            tr_s=tr_s,
            strict=False,
        )

        pulse = timing["pulse"]
        line = timing["readout"]
        gz_spoil = timing["crusher"]
        te_delay_s = timing["te_delay_s"]
        tr_delay_s = timing["tr_delay_s"]

        te_delay = pp.make_delay(te_delay_s) if te_delay_s > 0.0 else None
        tr_delay = pp.make_delay(tr_delay_s) if tr_delay_s > 0.0 else None

        seq = pp.Sequence(opts)

        sampling = design.make_cartesian_sampling((nx_ro, ny_pe), acceleration=ry, calibration=acs_lines)
        slice_step_m = slice_spacing_m if nslices > 1 else 0.0
        slices = design.make_slice_loop(nslices, slice_step_m or slice_thickness_m, order="sequential")
        slice_offsets_hz = slices.to_frequencies(pulse.gradients[0].amplitude) if slice_step_m else None
        rf_phases = design.make_rf_spoiling_schedule(
            len(sampling) * len(slices),
            increment=RF_SPOILING_INCREMENT_RAD,
        )
        phase_index = 0

        # PE-outer / SLC-inner loop order: fftrecon reshapes as [cha, RO, PE, SLC]
        # assuming slice is the fast (innermost) dimension.  LABELSET SLC and LIN
        # are required so pulserverlib writes correct slc/lin entries into the
        # trajectory cache; without them every readout gets slc=0 and multi-slice
        # reconstruction silently collapses to a single image.
        for shot in sampling:
            ky = int(shot[0, 0])
            label_lin = pp.make_label(type="SET", label="LIN", value=ky)

            for sl, band in enumerate(slices.shots):
                offset_hz = float(slice_offsets_hz[band[0]]) if slice_offsets_hz is not None else 0.0

                rf_phase = float(rf_phases[phase_index])
                pulse.set_state(freq_offset_hz=offset_hz, phase_offset_rad=rf_phase)
                pulse.set_labels(*slices.labels(sl), label_lin)
                for block in pulse:
                    seq.add_block(*block)
                if te_delay is not None:
                    seq.add_block(te_delay)
                line.set_state(lin_idx=ky, phase_offset_rad=rf_phase)
                for block in line:
                    seq.add_block(*block)
                seq.add_block(gz_spoil)
                phase_index += 1

            # TR delay is appended once after all slices so that the time between
            # successive excitations of the same slice equals the user-set TR.
            if tr_delay is not None:
                seq.add_block(tr_delay)

        seq.set_definition("Name", "gre_2d")
        seq.set_definition(
            "FOV",
            [fov_ro_m, fov_pe_m, slice_step_m * nslices if nslices > 1 else slice_thickness_m],
        )
        seq.set_definition("TE", te_s)
        seq.set_definition("TR", tr_s)
        seq.set_definition("Flip", flip_deg)
        seq.set_definition("ReadoutAxis", ro_axis)
        seq.set_definition("PhaseAxis", pe_axis)
        seq.set_definition("BandwidthHzPerPx", bandwidth_hz_px)
        seq.set_definition("Ry", ry)
        seq.set_definition("AcsLines", acs_lines)
        seq.set_definition("RfSpoilingIncDeg", 117.0)
        seq.set_definition("Nx", nx_ro)
        seq.set_definition("Ny", ny_pe)
        seq.set_definition("NySampled", len(sampling))
        seq.set_definition("NumSlices", nslices)
        pio.write(
            seq,
            output=output_path,
            remove_duplicates=False,
            check_timing=False,
        )


def _compute_timing(
    opts: pp.Opts,
    flip_deg: float,
    fov_ro_m: float,
    fov_pe_m: float,
    slice_thickness_m: float,
    nx_ro: int,
    ny_pe: int,
    nslices: int,
    bandwidth_hz_px: float,
    ro_axis: str,
    pe_axis: str,
    te_s: float,
    tr_s: float,
    strict: bool = True,
):
    if (ro_axis, pe_axis) != ("x", "y"):
        # Public readouts deliberately use the fixed x/y/z convention.
        ro_axis, pe_axis = "x", "y"
    line = design.make_line_readout(
        opts,
        (fov_ro_m, fov_pe_m),
        (nx_ro, ny_pe),
        bandwidth_hz_px=bandwidth_hz_px,
        spoil_position="post",
        spoil_cycles=1.0,
    )
    pulse = design.make_slice_selective_pulse(
        np.deg2rad(flip_deg),
        slice_thickness_m,
        system=opts,
    )
    crusher = design.make_crusher(
        opts,
        "z",
        dephasing_cycles=4.0,
        voxel_size=slice_thickness_m,
    )

    d_pulse = sum(pp.calc_duration(*block) for block in pulse)
    rf_center_s = pp.calc_rf_center(pulse.rf)[0] + pulse.rf.delay
    min_te_s = (d_pulse - rf_center_s) + line.t_first_echo_s
    block_raster_s = opts.block_duration_raster
    te_delay_s = round((te_s - min_te_s) / block_raster_s) * block_raster_s
    if te_delay_s < -1e-9 and strict:
        return None
    if te_delay_s < 0.0:
        te_delay_s = 0.0

    min_block_s = d_pulse + te_delay_s + line.duration + pp.calc_duration(crusher)
    tr_delay_s = round((tr_s - nslices * min_block_s) / block_raster_s) * block_raster_s
    if tr_delay_s < -1e-9 and strict:
        return None
    if tr_delay_s < 0.0:
        tr_delay_s = 0.0

    return {
        "pulse": pulse,
        "readout": line,
        "crusher": crusher,
        "te_delay_s": te_delay_s,
        "tr_delay_s": tr_delay_s,
        "min_block_s": min_block_s,
    }


PLUGIN = GrePulseqSequence()


def get_default_protocol(opts):
    return PLUGIN.get_default_protocol(opts)


def validate_protocol(opts, protocol):
    return PLUGIN.validate_protocol(opts, protocol)


def make_sequence(opts, protocol, output_path):
    return PLUGIN.make_sequence(opts, protocol, output_path)


def makeSeq(opts, protocol, output_path):
    """Offline alias for compatibility with older helper naming."""
    return PLUGIN.make_sequence(opts, protocol, output_path)


_ARG_MAP = [
    ('--te-ms', UIParam.TE, float, 'Echo time [ms]'),
    ('--tr-ms', UIParam.TR, float, 'Repetition time [ms]'),
    ('--flip-deg', UIParam.FLIP, float, 'Flip angle [deg]'),
    ('--fov-mm', UIParam.FOV, float, 'Readout FOV [mm]'),
    ('--phase-fov-mm', UIParam.PHASE_FOV, float, 'Phase-encode FOV [mm]'),
    ('--slice-thickness-mm', UIParam.SLICE_THICKNESS, float, 'Slice thickness [mm]'),
    ('--slice-spacing-mm', UIParam.SLICE_SPACING, float, 'Slice spacing [mm]'),
    ('--nx', UIParam.NX, int, 'Readout matrix size'),
    ('--ny', UIParam.NY, int, 'Phase matrix size'),
    ('--nslices', UIParam.NSLICES, int, 'Number of slices'),
    ('--bandwidth-hz-px', UIParam.BANDWIDTH, float, 'Readout bandwidth [Hz/px]'),
    ('--ry', UIParam.RY, float, 'Phase undersampling factor'),
    ('--acs-lines', UIParam.user_value(0), float, 'Number of ACS lines (user0)'),
    ('--swap-phase-freq', UIParam.SWAP_PHASE_FREQ, ("const", True), 'Swap readout/phase axes'),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description='Generate a Cartesian GRE .seq offline using the same implementation as the nimpulseqgui plugin path.',
            default_output='gre_2d.seq',
        )
    )
