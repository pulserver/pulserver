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

import argparse
import json
import sys

import numpy as np
import pypulseq as pp

import pulserver.io as pio
import pulserver.pulseq as ps

from pulserver import (
    PulseqSequence,
    BoolParam,
    TypeinFloatParam,
    TypeinIntParam,
    UIParam,
    Validate,
    dict_to_protocol,
    protocol_to_dict,
)


RF_TIME_S = 2.00e-3
RF_APODIZATION = 0.5
RF_TIME_BW_PRODUCT = 4.0
DEFAULT_BANDWIDTH_HZ_PX = 125_000.0
MAX_RASTER_SEARCH_STEPS = 50_000
RF_SPOILING_INC_DEG = 117.0
MAX_GRAD_DERATE = 0.9
MAX_SLEW_DERATE = 0.9
SPOIL_FACTOR_X = 4.0
SPOIL_FACTOR_Z = 2.0


class GrePulseqSequence(PulseqSequence):
    """Generate a true Cartesian 2D GRE sequence using pypulseq."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        del opts
        protocol = {
            UIParam.TE: TypeinFloatParam(
                value=8.0,
                min=2.0,
                max=80.0,
                incr=0.1,
                unit="ms",
                validate=Validate.NONE,
            ),
            UIParam.TR: TypeinFloatParam(
                value=20.0,
                min=5.0,
                max=2000.0,
                incr=0.1,
                unit="ms",
                validate=Validate.NONE,
            ),
            UIParam.FLIP: TypeinFloatParam(
                value=12.0,
                min=1.0,
                max=90.0,
                incr=1.0,
                unit="deg",
                validate=Validate.NONE,
            ),
            UIParam.FOV: TypeinFloatParam(
                value=220.0,
                min=80.0,
                max=500.0,
                incr=1.0,
                unit="mm",
                validate=Validate.NONE,
            ),
            UIParam.PHASE_FOV: TypeinFloatParam(
                value=220.0,
                min=80.0,
                max=500.0,
                incr=1.0,
                unit="mm",
                validate=Validate.NONE,
            ),
            UIParam.SLICE_THICKNESS: TypeinFloatParam(
                value=5.0,
                min=1.0,
                max=20.0,
                incr=0.5,
                unit="mm",
                validate=Validate.NONE,
            ),
            UIParam.SLICE_SPACING: TypeinFloatParam(
                value=5.0,
                min=1.0,
                max=20.0,
                incr=0.5,
                unit="mm",
                validate=Validate.NONE,
            ),
            UIParam.NX: TypeinIntParam(
                value=64,
                min=16,
                max=512,
                incr=1,
                validate=Validate.NONE,
            ),
            UIParam.NY: TypeinIntParam(
                value=64,
                min=8,
                max=512,
                incr=1,
                validate=Validate.NONE,
            ),
            UIParam.NSLICES: TypeinIntParam(
                value=1,
                min=1,
                max=128,
                incr=1,
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

        te_s = _param_float(prot, UIParam.TE) * 1e-3
        tr_s = _param_float(prot, UIParam.TR) * 1e-3
        flip_deg = _param_float(prot, UIParam.FLIP)
        fov_ro_m = _param_float(prot, UIParam.FOV) * 1e-3
        fov_pe_m = _phase_fov_mm_from_protocol(prot) * 1e-3
        slice_thickness_m = _param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
        nx_ro = _param_int(prot, UIParam.NX)
        ny_pe = _param_int(prot, UIParam.NY)
        nslices = _param_int(prot, UIParam.NSLICES)
        bandwidth_hz_px = _param_float_optional(prot, UIParam.BANDWIDTH, DEFAULT_BANDWIDTH_HZ_PX)
        ry = max(1, int(round(_param_float_optional(prot, UIParam.RY, 1.0))))
        acs_lines = _acs_lines_from_protocol(prot, ny_pe)
        ro_axis, pe_axis = _resolve_readout_phase_axes(prot)

        if te_s <= 0.0 or tr_s <= 0.0:
            return {"valid": False, "duration": None, "info": "TE and TR must be > 0"}
        if te_s >= tr_s:
            return {"valid": False, "duration": None, "info": "TE must be < TR"}
        if fov_ro_m <= 0.0 or fov_pe_m <= 0.0 or slice_thickness_m <= 0.0:
            return {"valid": False, "duration": None, "info": "FOV and slice thickness must be > 0"}
        if not (0.0 < flip_deg <= 180.0):
            return {"valid": False, "duration": None, "info": "Flip angle must be in (0, 180] deg"}
        if nx_ro < 1 or ny_pe < 1 or nslices < 1:
            return {"valid": False, "duration": None, "info": "NX, NY, and NSLICES must be >= 1"}
        if bandwidth_hz_px <= 0.0:
            return {"valid": False, "duration": None, "info": "Bandwidth must be > 0"}

        timing = _compute_timing(
            opts=opts,
            flip_deg=flip_deg,
            fov_ro_m=fov_ro_m,
            fov_pe_m=fov_pe_m,
            slice_thickness_m=slice_thickness_m,
            nx_ro=nx_ro,
            ny_pe=ny_pe,
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
                "info": "TE/TR too short for gradients and readout timing",
            }

        sampled_pe = _sampled_phase_lines(ny_pe, ry, acs_lines)
        duration_s = tr_s * float(len(sampled_pe)) * float(nslices)
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)

        te_s = _param_float(prot, UIParam.TE) * 1e-3
        tr_s = _param_float(prot, UIParam.TR) * 1e-3
        flip_deg = _param_float(prot, UIParam.FLIP)
        fov_ro_m = _param_float(prot, UIParam.FOV) * 1e-3
        fov_pe_m = _phase_fov_mm_from_protocol(prot) * 1e-3
        slice_thickness_m = _param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
        slice_spacing_m = _param_float(prot, UIParam.SLICE_SPACING) * 1e-3
        nx_ro = _param_int(prot, UIParam.NX)
        ny_pe = _param_int(prot, UIParam.NY)
        nslices = _param_int(prot, UIParam.NSLICES)
        bandwidth_hz_px = _param_float_optional(prot, UIParam.BANDWIDTH, DEFAULT_BANDWIDTH_HZ_PX)
        ry = max(1, int(round(_param_float_optional(prot, UIParam.RY, 1.0))))
        acs_lines = _acs_lines_from_protocol(prot, ny_pe)
        ro_axis, pe_axis = _resolve_readout_phase_axes(prot)

        timing = _compute_timing(
            opts=opts,
            flip_deg=flip_deg,
            fov_ro_m=fov_ro_m,
            fov_pe_m=fov_pe_m,
            slice_thickness_m=slice_thickness_m,
            nx_ro=nx_ro,
            ny_pe=ny_pe,
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

        seq = ps.Sequence(opts)

        delta_k_pe = 1.0 / fov_pe_m
        phase_areas = (np.arange(ny_pe) - 0.5 * ny_pe) * delta_k_pe
        max_pe_area = float(np.max(np.abs(phase_areas)))
        sampled_pe = _sampled_phase_lines(ny_pe, ry, acs_lines)
        slice_step_m = slice_spacing_m if nslices > 1 else 0.0
        rf_phase_deg = 0.0
        rf_phase_inc_deg = 0.0

        for sl in range(nslices):
            slice_offset_m = (sl - 0.5 * (nslices - 1)) * slice_step_m

            for ky in sampled_pe:
                rf_curr = _copy_event(rf)
                rf_curr.freq_offset = gz.amplitude * slice_offset_m
                rf_curr.phase_offset = np.deg2rad(rf_phase_deg)
                adc_curr = _copy_event(adc)
                adc_curr.phase_offset = rf_curr.phase_offset

                y_scale = phase_areas[ky] / max_pe_area if max_pe_area > 0.0 else 0.0
                gy_pre = pp.scale_grad(gy_template, y_scale)
                gy_reph = pp.scale_grad(gy_template, -y_scale)

                seq.add_block(rf_curr, gz)
                seq.add_block(gx_pre, gy_pre, gz_reph)
                if te_delay is not None:
                    seq.add_block(te_delay)
                seq.add_block(gx, adc_curr)
                seq.add_block(gx_spoil, gy_reph, gz_spoil)
                if tr_delay is not None:
                    seq.add_block(tr_delay)

                # Standard RF spoiling phase progression per TR.
                rf_phase_deg = (rf_phase_deg + rf_phase_inc_deg) % 360.0
                rf_phase_inc_deg = (rf_phase_inc_deg + RF_SPOILING_INC_DEG) % 360.0

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
        seq.set_definition("RfSpoilingIncDeg", RF_SPOILING_INC_DEG)
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


def _param_float(protocol: dict, key: UIParam) -> float:
    return float(protocol[str(key)].value)


def _param_int(protocol: dict, key: UIParam) -> int:
    return int(protocol[str(key)].value)


def _param_float_optional(protocol: dict, key: UIParam | str, default: float) -> float:
    name = str(key)
    if name not in protocol:
        return default
    return float(protocol[name].value)


def _param_bool_optional(protocol: dict, key: UIParam | str, default: bool) -> bool:
    name = str(key)
    if name not in protocol:
        return default
    return bool(protocol[name].value)


def _phase_fov_mm_from_protocol(protocol: dict) -> float:
    fov_mm = _param_float(protocol, UIParam.FOV)
    phase_fov = _param_float_optional(protocol, UIParam.PHASE_FOV, fov_mm)
    if phase_fov <= 1.5:
        # Allow fraction-style PhaseFOV if explicitly provided by upstream UI.
        return max(0.0, phase_fov * fov_mm)
    return phase_fov


def _acs_lines_from_protocol(protocol: dict, ny_pe: int) -> int:
    raw = _param_float_optional(protocol, UIParam.user_value(0), 0.0)
    acs = int(round(raw))
    if acs < 0:
        return 0
    return min(acs, ny_pe)


def _sampled_phase_lines(ny_pe: int, ry: int, acs_lines: int) -> list[int]:
    sampled = {i for i in range(ny_pe) if (i % ry) == 0}
    if acs_lines > 0:
        center = ny_pe // 2
        start = max(0, center - acs_lines // 2)
        stop = min(ny_pe, start + acs_lines)
        for i in range(start, stop):
            sampled.add(i)
    return sorted(sampled)


def _resolve_readout_phase_axes(protocol: dict) -> tuple[str, str]:
    ro_axis = "x"
    pe_axis = "y"

    if _param_bool_optional(protocol, UIParam.SWAP_PHASE_FREQ, False):
        ro_axis, pe_axis = "y", "x"

    return ro_axis, pe_axis


def _copy_event(event):
    event_copy = type(event)()
    event_copy.__dict__.update(event.__dict__)
    return event_copy


def _compute_timing(
    opts: pp.Opts,
    flip_deg: float,
    fov_ro_m: float,
    fov_pe_m: float,
    slice_thickness_m: float,
    nx_ro: int,
    ny_pe: int,
    bandwidth_hz_px: float,
    ro_axis: str,
    pe_axis: str,
    te_s: float,
    tr_s: float,
    strict: bool = True,
):
    _apply_system_derates(opts)

    rf, gz, gz_reph = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(flip_deg),
        duration=RF_TIME_S,
        slice_thickness=slice_thickness_m,
        apodization=RF_APODIZATION,
        time_bw_product=RF_TIME_BW_PRODUCT,
        system=opts,
        return_gz=True,
    )

    delta_k_ro = 1.0 / fov_ro_m
    readout_area = nx_ro * delta_k_ro
    dwell_s, flat_time_s = _quantize_readout_timing(
        nx_ro=nx_ro,
        target_bw_hz_px=bandwidth_hz_px,
        grad_raster_s=opts.grad_raster_time,
        adc_raster_s=opts.adc_raster_time,
        min_flat_time_s=abs(readout_area) / (0.95 * opts.max_grad),
    )

    gx_full = pp.make_trapezoid(
        channel=ro_axis,
        flat_area=readout_area,
        flat_time=flat_time_s,
        system=opts,
    )
    gx_parts = pp.split_gradient_at(gx_full, gx_full.rise_time + gx_full.flat_time, system=opts)
    gx = gx_parts[0]
    adc = pp.make_adc(
        num_samples=nx_ro,
        dwell=dwell_s,
        delay=gx_full.rise_time,
        system=opts,
    )

    gx_pre = pp.make_trapezoid(channel=ro_axis, area=-0.5 * gx_full.area, system=opts)
    gx_spoil_area = SPOIL_FACTOR_X / slice_thickness_m
    gx_spoil, _, _ = pp.make_extended_trapezoid_area(
        area=gx_spoil_area,
        channel=ro_axis,
        grad_start=gx_full.amplitude,
        grad_end=0.0,
        system=opts,
    )
    gz_spoil = pp.make_trapezoid(channel="z", area=SPOIL_FACTOR_Z / slice_thickness_m, system=opts)
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
    adc_center_s = adc.delay + 0.5 * gx_full.flat_time
    min_te_s = (d_rf - rf_center_s) + d_pre + adc_center_s
    te_delay_s = te_s - min_te_s
    if te_delay_s < -1e-9 and strict:
        return None
    if te_delay_s < 0.0:
        te_delay_s = 0.0

    min_tr_s = d_rf + d_pre + te_delay_s + d_ro + d_spoil
    tr_delay_s = tr_s - min_tr_s
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
    }


def _apply_system_derates(opts: pp.Opts) -> None:
    if not hasattr(opts, "_pulserver_base_max_grad"):
        opts._pulserver_base_max_grad = float(opts.max_grad)
    if not hasattr(opts, "_pulserver_base_max_slew"):
        opts._pulserver_base_max_slew = float(opts.max_slew)

    opts.max_grad = opts._pulserver_base_max_grad * float(MAX_GRAD_DERATE)
    opts.max_slew = opts._pulserver_base_max_slew * float(MAX_SLEW_DERATE)


def _quantize_readout_timing(
    nx_ro: int,
    target_bw_hz_px: float,
    grad_raster_s: float,
    adc_raster_s: float,
    min_flat_time_s: float,
) -> tuple[float, float]:
    target_dwell = 1.0 / target_bw_hz_px
    m_target = max(1, int(round(target_dwell / adc_raster_s)))

    m_min = max(m_target, int(np.ceil(min_flat_time_s / (nx_ro * adc_raster_s))))

    best_m = None
    best_cost = None

    for step in range(MAX_RASTER_SEARCH_STEPS):
        m = m_min + step
        flat = nx_ro * m * adc_raster_s
        ratio = flat / grad_raster_s
        if abs(ratio - round(ratio)) > 1e-9:
            continue
        cost = abs((m * adc_raster_s) - target_dwell)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_m = m
            if cost == 0.0:
                break

    if best_m is None:
        best_m = m_min

    dwell = best_m * adc_raster_s
    flat_time = nx_ro * dwell
    return dwell, flat_time


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


def _set_protocol_value(protocol: dict, key: UIParam, value) -> None:
    protocol[str(key)]["value"] = value





def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Cartesian GRE .seq offline using the same implementation "
            "as the nimpulseqgui plugin path."
        )
    )
    parser.add_argument("-o", "--output", default="gre.seq", help="Output .seq file path")

    parser.add_argument("--te-ms", type=float, help="Echo time [ms]")
    parser.add_argument("--tr-ms", type=float, help="Repetition time [ms]")
    parser.add_argument("--flip-deg", type=float, help="Flip angle [deg]")
    parser.add_argument("--fov-mm", type=float, help="Readout FOV [mm]")
    parser.add_argument("--phase-fov-mm", type=float, help="Phase-encode FOV [mm]")
    parser.add_argument("--slice-thickness-mm", type=float, help="Slice thickness [mm]")
    parser.add_argument("--slice-spacing-mm", type=float, help="Slice spacing [mm]")
    parser.add_argument("--nx", type=int, help="Readout matrix size")
    parser.add_argument("--ny", type=int, help="Phase matrix size")
    parser.add_argument("--nslices", type=int, help="Number of slices")
    parser.add_argument("--bandwidth-hz-px", type=float, help="Readout bandwidth [Hz/px]")
    parser.add_argument("--ry", type=float, help="Phase undersampling factor")
    parser.add_argument("--acs-lines", type=float, help="Number of ACS lines (user0)")
    parser.add_argument(
        "--swap-phase-freq",
        action="store_true",
        help="Swap readout/phase axes",
    )

    parser.add_argument(
        "--max-grad-mtm",
        type=float,
        help="System max gradient amplitude [mT/m] for offline generation",
    )
    parser.add_argument(
        "--max-slew-tm-s",
        type=float,
        help="System max slew [T/m/s] for offline generation",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run protocol validation only (do not write sequence)",
    )

    return parser


def _cli(argv: list[str]) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    opts_kwargs = {}
    if args.max_grad_mtm is not None:
        opts_kwargs["max_grad"] = args.max_grad_mtm
        opts_kwargs["grad_unit"] = "mT/m"
    if args.max_slew_tm_s is not None:
        opts_kwargs["max_slew"] = args.max_slew_tm_s
        opts_kwargs["slew_unit"] = "T/m/s"
    opts = pp.Opts(**opts_kwargs)

    protocol = PLUGIN.get_default_protocol(opts)

    if args.te_ms is not None:
        _set_protocol_value(protocol, UIParam.TE, args.te_ms)
    if args.tr_ms is not None:
        _set_protocol_value(protocol, UIParam.TR, args.tr_ms)
    if args.flip_deg is not None:
        _set_protocol_value(protocol, UIParam.FLIP, args.flip_deg)
    if args.fov_mm is not None:
        _set_protocol_value(protocol, UIParam.FOV, args.fov_mm)
    if args.phase_fov_mm is not None:
        _set_protocol_value(protocol, UIParam.PHASE_FOV, args.phase_fov_mm)
    if args.slice_thickness_mm is not None:
        _set_protocol_value(protocol, UIParam.SLICE_THICKNESS, args.slice_thickness_mm)
    if args.slice_spacing_mm is not None:
        _set_protocol_value(protocol, UIParam.SLICE_SPACING, args.slice_spacing_mm)
    if args.nx is not None:
        _set_protocol_value(protocol, UIParam.NX, args.nx)
    if args.ny is not None:
        _set_protocol_value(protocol, UIParam.NY, args.ny)
    if args.nslices is not None:
        _set_protocol_value(protocol, UIParam.NSLICES, args.nslices)
    if args.bandwidth_hz_px is not None:
        _set_protocol_value(protocol, UIParam.BANDWIDTH, args.bandwidth_hz_px)
    if args.ry is not None:
        _set_protocol_value(protocol, UIParam.RY, args.ry)
    if args.acs_lines is not None:
        _set_protocol_value(protocol, UIParam.user_value(0), args.acs_lines)
    if args.swap_phase_freq:
        _set_protocol_value(protocol, UIParam.SWAP_PHASE_FREQ, True)

    result = PLUGIN.validate_protocol(opts, protocol)
    if not result.get("valid", False):
        info = result.get("info", "Protocol invalid")
        print(f"ERROR: {info}", file=sys.stderr)
        return 2

    print(result.get("info", "Protocol valid"))
    if args.validate_only:
        return 0

    PLUGIN.make_sequence(opts, protocol, args.output)
    print(f"Wrote sequence: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
