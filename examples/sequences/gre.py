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

    cp gre.py pulserver-interpreter/tree/pulserver/sequences/src/gre.py
    ln -sf src/gre.py  pulserver-interpreter/tree/pulserver/sequences/sequence5.py

Or pass the file directly to the bridge host for testing::

    python -m pulserver.bridge --plugin examples/gre.py --action default_protocol
"""

from __future__ import annotations

import sys

import numpy as np
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
from pulserver.pypulseq import _gradients as encoding
from pulserver.pypulseq import _readout as readout
from pulserver.pypulseq import _sampling as sampling
from pulserver.pypulseq import _system as system
from pulserver.pypulseq._rf import _excitation_helpers as excitation


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
                value=system.DEFAULT_BANDWIDTH_HZ_PX,
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
        bandwidth_hz_px = params.param_float_optional(prot, UIParam.BANDWIDTH, system.DEFAULT_BANDWIDTH_HZ_PX)
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

        sampled_pe = sampling.calc_sampled_lines(ny_pe, ry, acs_lines)
        duration_s = tr_s * float(len(sampled_pe))
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
        bandwidth_hz_px = params.param_float_optional(prot, UIParam.BANDWIDTH, system.DEFAULT_BANDWIDTH_HZ_PX)
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

        rf = timing["rf"]
        gz = timing["gz"]
        gz_reph = timing["gz_reph"]
        gx = timing["gx"]
        adc = timing["adc"]
        gx_pre = timing["gx_pre"]
        gx_spoil = timing["gx_spoil"]
        gz_spoil = timing["gz_spoil"]
        gy_template = timing["gy_template"]
        te_delay_s = timing["te_delay_s"]
        tr_delay_s = timing["tr_delay_s"]

        te_delay = pp.make_delay(te_delay_s) if te_delay_s > 0.0 else None
        tr_delay = pp.make_delay(tr_delay_s) if tr_delay_s > 0.0 else None

        seq = pp.Sequence(opts)

        delta_k_pe = 1.0 / fov_pe_m
        phase_areas = (np.arange(ny_pe) - 0.5 * ny_pe) * delta_k_pe
        max_pe_area = float(np.max(np.abs(phase_areas)))
        sampled_pe = sampling.calc_sampled_lines(ny_pe, ry, acs_lines)
        slice_step_m = slice_spacing_m if nslices > 1 else 0.0
        rf_phase_deg = 0.0
        rf_phase_inc_deg = 0.0

        # PE-outer / SLC-inner loop order: fftrecon reshapes as [cha, RO, PE, SLC]
        # assuming slice is the fast (innermost) dimension.  LABELSET SLC and LIN
        # are required so pulserverlib writes correct slc/lin entries into the
        # trajectory cache; without them every readout gets slc=0 and multi-slice
        # reconstruction silently collapses to a single image.
        for ky in sampled_pe:
            y_scale = phase_areas[ky] / max_pe_area if max_pe_area > 0.0 else 0.0
            gy_pre = pp.scale_grad(gy_template, y_scale)
            gy_reph = pp.scale_grad(gy_template, -y_scale)
            label_lin = pp.make_label(type="SET", label="LIN", value=ky)

            for sl in range(nslices):
                slice_offset_m = (sl - 0.5 * (nslices - 1)) * slice_step_m

                rf_curr = system.copy_event(rf)
                rf_curr.freq_offset = gz.amplitude * slice_offset_m
                rf_curr.phase_offset = np.deg2rad(rf_phase_deg)
                adc_curr = system.copy_event(adc)
                adc_curr.phase_offset = rf_curr.phase_offset

                label_slc = pp.make_label(type="SET", label="SLC", value=sl)

                seq.add_block(rf_curr, gz, label_slc, label_lin)
                seq.add_block(gx_pre, gy_pre, gz_reph)
                if te_delay is not None:
                    seq.add_block(te_delay)
                seq.add_block(gx, adc_curr)
                seq.add_block(gx_spoil, gy_reph, gz_spoil)

                # Standard RF spoiling phase progression per TR.
                rf_phase_deg = (rf_phase_deg + rf_phase_inc_deg) % 360.0
                rf_phase_inc_deg = (rf_phase_inc_deg + excitation.RF_SPOILING_INC_DEG) % 360.0

            # TR delay is appended once after all slices so that the time between
            # successive excitations of the same slice equals the user-set TR.
            if tr_delay is not None:
                seq.add_block(tr_delay)

        seq.set_definition("Name", "gre")
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
        seq.set_definition("RfSpoilingIncDeg", excitation.RF_SPOILING_INC_DEG)
        seq.set_definition("Nx", nx_ro)
        seq.set_definition("Ny", ny_pe)
        seq.set_definition("NySampled", len(sampled_pe))
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
    system.apply_system_derates(opts)

    rf, gz, gz_reph = excitation.slice_selective(opts, flip_deg, slice_thickness_m)

    ro_events = readout.unbalanced_line(
        opts,
        fov_ro_m,
        nx_ro,
        bandwidth_hz_px=bandwidth_hz_px,
        slice_thickness_m=slice_thickness_m,
        axis=ro_axis,
    )
    gx = ro_events["gx"]
    adc = ro_events["adc"]
    gx_pre = ro_events["gx_pre"]
    gx_spoil = ro_events["gx_spoil"]
    gz_spoil = pp.make_trapezoid(channel="z", area=encoding.SPOIL_FACTOR_Z / slice_thickness_m, system=opts)
    max_pe_area = 0.5 * ny_pe * (1.0 / fov_pe_m)
    gy_template = pp.make_trapezoid(
        channel=pe_axis,
        area=max_pe_area,
        system=opts,
    )

    d_rf = pp.calc_duration(rf, gz)
    d_pre = pp.calc_duration(gx_pre, gy_template, gz_reph)
    d_ro = pp.calc_duration(gx, adc)
    d_spoil = pp.calc_duration(gx_spoil, gy_template, gz_spoil)

    rf_center_s = pp.calc_rf_center(rf)[0]
    adc_center_s = ro_events["adc_center_s"]
    min_te_s = (d_rf - rf_center_s) + d_pre + adc_center_s
    te_delay_s = te_s - min_te_s
    if te_delay_s < -1e-9 and strict:
        return None
    if te_delay_s < 0.0:
        te_delay_s = 0.0

    min_block_s = d_rf + d_pre + te_delay_s + d_ro + d_spoil
    tr_delay_s = tr_s - nslices * min_block_s
    if tr_delay_s < -1e-9 and strict:
        return None
    if tr_delay_s < 0.0:
        tr_delay_s = 0.0

    return {
        "rf": rf,
        "gz": gz,
        "gz_reph": gz_reph,
        "gx": gx,
        "adc": adc,
        "gx_pre": gx_pre,
        "gx_spoil": gx_spoil,
        "gz_spoil": gz_spoil,
        "gy_template": gy_template,
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
            default_output='gre.seq',
        )
    )
