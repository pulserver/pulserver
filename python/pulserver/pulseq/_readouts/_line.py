"""
"""

__all__ = ["LineReadout2D"]


from types import SimpleNamespace

import numpy as np
import pypulseq as pp

from ...tools import ISMRMRDBuilder

from ._base import calc_kspace_band_jump


class LineReadout2D:
    def __init__(
        self,
        fov: tuple[float],
        npix: tuple[int],
        oversamp: float = 1.0,
        receive_bandwidth: float = 250.0,
        lbl: str = 'lin',
        num_echoes: int = 1,
        flyback: bool = False,
        net_area: float = 0.0,
        system: pp.Opts | None = None,
    ):
        fov_read, fov_phase = fov
        npix_read, npix_phase = npix
        if system is None:
            self.system = pp.Opts.default
        else:
            self.system = system
            
        # Compute number of samples
        num_samples = int(np.ceil(oversamp * npix_read))
        
        # Compute readout duration based on number of samples and receive bandwidth
        dwell_time = 1.0 / (2.0 * receive_bandwidth * 1e3)
        
        # Round to nearest multiple of ADC raster time
        dwell_time = self.system.adc_raster_time * np.ceil(dwell_time / self.system.adc_raster_time)
        dwell_time = dwell_time.item()
        t_read = dwell_time * num_samples

        # Compute frequency encoding area
        kr_area = npix_read / fov_read
        
        # Compute phase encoding area and scalings
        kp_area, self.kp_scaling = calc_kspace_band_jump(fov_phase, npix_phase)

        # Create phase encoding gradient
        self.registered = False
        self.phase_encoding = pp.make_extended_trapezoid_area(
            channel='y',
            area=kp_area,
            grad_start=0.0,
            grad_end=0.0,
        )

    def __call__(self, seq=None, idx=None):
        seq = self.add_prewind(seq, idx)
        seq = self.add_readout(seq)
        seq = self.add_rewinder(seq, idx)
        return seq

    def _add_phase_enc(
        self,
        seq: pp.Sequence | None = None,
    ):
        if seq is not None and self.registered is False:
            result = seq.register_grad_event(self.phase_encoding)
            self.phase_encoding.id = result if isinstance(result, int) else result[0]
            self.registered = True
        if seq is None:
            dummy_sys = pp.Opts
            dummy_sys.max_grad = np.inf
            dummy_sys.max_slew = np.inf
            seq = pp.Sequence(system=dummy_sys)
        return seq

    def add_prewind(
        self,
        seq: pp.Sequence | None = None,
        idx: int | None = None,
        *events: list[SimpleNamespace],
    ):
        seq = self._add_phase_enc(seq)
        if idx is None:
            seq.add_block(self.phase_encoding, *events)
        else:
            phase_encoding = pp.scale_grad(
                self.phase_encoding,
                self.kp_scaling.pre[idx],
                self.system,
            )
            seq.add_block(phase_encoding, *events)
        return seq

    def add_rewinder(
        self,
        seq: pp.Sequence | None = None,
        idx: int | None = None,
        *events: list[SimpleNamespace],
    ):
        seq = self._add_phase_enc(seq)
        if idx is None:
            phase_encoding = pp.scale_grad(
                self.phase_encoding,
                -1.0,
            )
        else:
            phase_encoding = pp.scale_grad(
                self.phase_encoding,
                self.kp_scaling.rew[idx],
                self.system,
            )
        seq.add_block(phase_encoding, *events)
        return seq

    def add_readout(): ...
