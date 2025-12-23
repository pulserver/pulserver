import numpy as np
import pypulseq as pp

from ._base import calc_kspace_readout_params

# class GradientLobe:
#     ...

# class CompositeLobe(GradientLobe):
#     ...


class ReadoutLobe:
    def __init__(
        self,
        fov: float,
        npix: int,
        receive_bandwidth: float = 250.0,
        oversamp: float = 1.0,
        system: pp.Opts | None = None,
        rampsamp: bool = False,
        dead_time: float = 0.0,
        pf_factor: float = 1.0,
    ):
        # actual number of samples
        npix = np.ceil(pf_factor * npix).astype(int).item()
        readout_area, readout_time, num_samples = calc_kspace_readout_params(
            fov,
            npix,
            receive_bandwidth,
            oversamp,
            system.adc_raster_time,
            system.grad_raster_time,
        )

        # Generate ADC
        pp.make_adc(num_samples)


class GeneralizedPhasor:
    ...


# bSSFP readout [GeneralizedPhasor(area=read/2; start=0.0, end=0.0), Readout(area=area, ...), -1 * GeneralizedPhasor(...)]
