"""2D Gradient Echo sequence plugin."""

from __future__ import annotations

import math

import numpy as np
import pypulseq as pp

from ..core._base import PulseqSequence
from ..core._params import (
    BoolParam,
    Description,
    FloatParam,
    IntParam,
    Protocol,
    UIParam,
    Validate,
)


class GRE2D(PulseqSequence):
    """2D Cartesian Gradient Echo built with pypulseq."""

    def get_default_protocol(self, opts: pp.Opts) -> Protocol:
        return {
            "Description": Description("2D GRE (pypulseq)"),
            "D_Resolution": Description("Resolution"),
            UIParam.FOV: IntParam(256, 64, 512, 1, unit="mm"),
            UIParam.SLICE_THICKNESS: IntParam(5, 1, 20, 1, unit="mm"),
            UIParam.MATRIX: IntParam(256, 32, 512, 32),
            "Ny": IntParam(256, 32, 512, 1, validate=Validate.NONE),
            "D_Timing": Description("Timing"),
            UIParam.TE: FloatParam(5.0, 1.0, 100.0, 0.1, unit="ms"),
            UIParam.TR: FloatParam(10.0, 1.0, 1000.0, 0.1, unit="ms"),
            UIParam.RF_SPOILING: BoolParam(True),
            UIParam.FLIP_ANGLE: FloatParam(15.0, 1.0, 90.0, 1.0, unit="deg"),
            UIParam.TA: Description(""),
        }

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _make_system(opts: pp.Opts) -> pp.Opts:
        return opts

    @staticmethod
    def _read_params(protocol: Protocol) -> dict:
        fov_mm = protocol[UIParam.FOV].value
        return {
            "fov": fov_mm * 1e-3,
            "slice_thickness": protocol[UIParam.SLICE_THICKNESS].value * 1e-3,
            "Nx": protocol[UIParam.MATRIX].value,
            "Ny": protocol["Ny"].value,
            "TE": protocol[UIParam.TE].value * 1e-3,
            "TR": protocol[UIParam.TR].value * 1e-3,
            "flip": protocol[UIParam.FLIP_ANGLE].value,
            "rf_spoiling_inc": 117.0 if protocol[UIParam.RF_SPOILING].value else 0.0,
        }

    # -----------------------------------------------------------------
    # Contract
    # -----------------------------------------------------------------

    def validate_protocol(self, opts: pp.Opts, protocol: Protocol) -> dict:
        try:
            system = self._make_system(opts)
            p = self._read_params(protocol)
            seq = pp.Sequence(system=system)

            _, gz, _ = pp.make_sinc_pulse(
                flip_angle=p["flip"] * math.pi / 180.0,
                duration=3e-3,
                slice_thickness=p["slice_thickness"],
                apodization=0.5,
                time_bw_product=4.0,
                system=system,
                return_gz=True,
                use="excitation",
            )

            delta_k = 1.0 / p["fov"]
            ro_duration = 3.2e-3
            gx = pp.make_trapezoid(
                channel="x",
                flat_area=p["Nx"] * delta_k,
                flat_time=ro_duration,
                system=system,
            )
            gx_pre = pp.make_trapezoid(
                channel="x", area=-gx.area / 2.0, duration=1e-3, system=system
            )
            gz_reph = pp.make_trapezoid(
                channel="z", area=-gz.area / 2.0, duration=1e-3, system=system
            )

            phase_area = (p["Ny"] / 2.0) * delta_k
            gy_pre = pp.make_trapezoid(
                channel="y", area=phase_area, system=system
            )

            if pp.calc_duration(gy_pre) > pp.calc_duration(gx_pre):
                return {
                    "valid": False,
                    "duration": None,
                    "info": "Phase-encode gradient too long for readout prewinder",
                }

            gx_spoil = pp.make_trapezoid(
                channel="x", area=2.0 * p["Nx"] * delta_k, system=system
            )
            gz_spoil = pp.make_trapezoid(
                channel="z", area=4.0 / p["slice_thickness"], system=system
            )

            delay_TE = math.ceil(
                (
                    p["TE"]
                    - pp.calc_duration(gx_pre)
                    - gz.fall_time
                    - gz.flat_time / 2.0
                    - pp.calc_duration(gx) / 2.0
                )
                / seq.grad_raster_time
            ) * seq.grad_raster_time

            delay_TR = math.ceil(
                (
                    p["TR"]
                    - pp.calc_duration(gz)
                    - pp.calc_duration(gx_pre)
                    - pp.calc_duration(gx)
                    - delay_TE
                )
                / seq.grad_raster_time
            ) * seq.grad_raster_time

            if delay_TE < 0:
                min_TE = (
                    pp.calc_duration(gx_pre)
                    + gz.fall_time
                    + gz.flat_time / 2.0
                    + pp.calc_duration(gx) / 2.0
                )
                return {
                    "valid": False,
                    "duration": None,
                    "info": f"TE too short (min {min_TE * 1e3:.2f} ms)",
                }

            if delay_TR < pp.calc_duration(gx_spoil, gz_spoil):
                min_TR = (
                    pp.calc_duration(gz)
                    + pp.calc_duration(gx_pre)
                    + pp.calc_duration(gx)
                    + pp.calc_duration(gx_spoil, gz_spoil)
                )
                return {
                    "valid": False,
                    "duration": None,
                    "info": f"TR too short (min {min_TR * 1e3:.2f} ms)",
                }

            duration = p["TR"] * p["Ny"]
            return {
                "valid": True,
                "duration": duration,
                "info": f"TA = {duration:.1f} s",
            }

        except Exception as exc:
            return {"valid": False, "duration": None, "info": str(exc)}

    def make_sequence(
        self, opts: pp.Opts, protocol: Protocol, output_path: str
    ) -> None:
        system = self._make_system(opts)
        p = self._read_params(protocol)
        seq = pp.Sequence(system=system)

        rf, gz, _ = pp.make_sinc_pulse(
            flip_angle=p["flip"] * math.pi / 180.0,
            duration=3e-3,
            slice_thickness=p["slice_thickness"],
            apodization=0.5,
            time_bw_product=4.0,
            system=system,
            return_gz=True,
            delay=system.rf_dead_time,
            use="excitation",
        )

        delta_k = 1.0 / p["fov"]
        ro_duration = 3.2e-3
        gx = pp.make_trapezoid(
            channel="x",
            flat_area=p["Nx"] * delta_k,
            flat_time=ro_duration,
            system=system,
        )
        adc = pp.make_adc(
            num_samples=p["Nx"],
            duration=gx.flat_time,
            delay=gx.rise_time,
            system=system,
        )
        gx_pre = pp.make_trapezoid(
            channel="x", area=-gx.area / 2.0, duration=1e-3, system=system
        )
        gz_reph = pp.make_trapezoid(
            channel="z", area=-gz.area / 2.0, duration=1e-3, system=system
        )

        phase_areas = np.arange(-p["Ny"] / 2, p["Ny"] / 2) * delta_k

        gx_spoil = pp.make_trapezoid(
            channel="x", area=2.0 * p["Nx"] * delta_k, system=system
        )
        gz_spoil = pp.make_trapezoid(
            channel="z", area=4.0 / p["slice_thickness"], system=system
        )

        delay_TE = math.ceil(
            (
                p["TE"]
                - pp.calc_duration(gx_pre)
                - gz.fall_time
                - gz.flat_time / 2.0
                - pp.calc_duration(gx) / 2.0
            )
            / seq.grad_raster_time
        ) * seq.grad_raster_time

        delay_TR = math.ceil(
            (
                p["TR"]
                - pp.calc_duration(gz)
                - pp.calc_duration(gx_pre)
                - pp.calc_duration(gx)
                - delay_TE
            )
            / seq.grad_raster_time
        ) * seq.grad_raster_time

        rf_phase = 0.0
        rf_inc = 0.0

        for i in range(p["Ny"]):
            rf.phase_offset = rf_phase / 180.0 * math.pi
            adc.phase_offset = rf_phase / 180.0 * math.pi
            rf_inc = (rf_inc + p["rf_spoiling_inc"]) % 360.0
            rf_phase = (rf_phase + rf_inc) % 360.0

            seq.add_block(rf, gz)
            gy_pre = pp.make_trapezoid(
                channel="y",
                area=phase_areas[i],
                duration=pp.calc_duration(gx_pre),
                system=system,
            )
            seq.add_block(gx_pre, gy_pre, gz_reph)
            seq.add_block(pp.make_delay(delay_TE))
            seq.add_block(gx, adc)

            gy_reph = pp.make_trapezoid(
                channel="y",
                area=-phase_areas[i],
                duration=pp.calc_duration(gx_spoil),
                system=system,
            )
            seq.add_block(pp.make_delay(delay_TR), gx_spoil, gy_reph, gz_spoil)

        ok, error_report = seq.check_timing()
        if not ok:
            raise RuntimeError(
                "Timing check failed:\n" + "\n".join(str(e) for e in error_report)
            )

        seq.set_definition("Name", "gre_2d")
        seq.set_definition("FOV", [p["fov"], p["fov"], p["slice_thickness"]])

        seq.write(output_path)


# Auto-expose for bridge fallback + direct import + testing
_instance = GRE2D()
get_default_protocol = _instance.get_default_protocol
validate_protocol = _instance.validate_protocol
make_sequence = _instance.make_sequence
