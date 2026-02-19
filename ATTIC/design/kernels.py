"""Spoiled Gradien Echo Kernels"""

__all__ = ['CartesianGre2D']


import copy
import logging
from typing import Literal

import numpy as np

from .. import pulseq as pp
from . import excitation, phasor, readout

ScanType = Literal['SCAN', 'DUMMY']


class CartesianGre2D:
    """
    Cartesian 2D Spoiled Gradient Echo Readout.

    Parameters
    ----------
    system : pp.Opts
        Pulseq system limits.
    fov_m : tuple[float]
        Field of view ``(Dx, Dy)`` in ``[m]``.
    slice_thickness_m : float
        Slice thickness in ``[m]``.
    matrix : tuple[int]
        Image size ``(nx, ny)``.
    num_slices : int
        Number of slices.
    flip_angle_rad : float
        Flip Angle in ``[deg]``.
    echo_time_s : float, optional
        Target Echo Time in ``[s]``. If not provided, use minimum TE.
    repetition_time_s : float, optional
        Target Repetition Time in ``[s]``. If not provided, use minimum TR.
    receive_bandwidth_Hz : float, optional
        Receive bandwidth in ``[Hz]``.
        The default is ``250e3 Hz``.
    parallel_imaging_accel : int, optional
        Parallel Imaging acceleration factor along ``y`` axis.
        The default is ``1`` (no acceleration).
    acr_y_size : int, optional
        Parallel Imaging Autocalibration Region size along ``y`` axis.
        The default is ``0``.
    slice_gap_m : float, optional
        Slice gap in ``[m]``. The default is ``0.0``.
    partial_fourier_accel_x : float, optional
        Partial Fourier Factor along ``x`` axis for asymmeric echo.
        The default is ``1.0``.
    partial_fourier_accel_y : float, optional
        Partial Fourier Factor along ``y`` axis.
        The default is ``1.0``.
    readout_oversamp : float, optional
        Readout oversampling factor.
        The default is ``1.0``.
    rf_spoiling_inc_rad : float, optional
        Phase increment seed for rf spoiling in ``[rad]``. The default corresponds to
        ``117.0 [deg]``.
    gx_spoiling_angle_rad : float, optional
        Dephasing along ``x`` across `slice_thickness_m`` in units of ``[rad]``.
        The default is ``32 * np.pi``.
    gz_spoiling_angle_rad : float, optional
        Dephasing along ``z`` across `slice_thickness_m`` in units of ``[rad]``.
        The default is ``32 * np.pi``.
    time_bw_product : float, optional
        Pulse time / bandwidth product. The default is ``4.0``.
    rf_duration_s : float, optional
        Pulse duration in ``[s]``. Default is ``2.0e-3 s``.
    rf_filter_type : str, optional
        Type of filter to use: ``"ms"`` (sinc),
        ``"pm``, (Parks-McClellan equal-ripple),
        ``"min"`` (minphase using factored pm),
        ``"max"`` (maxphase using factored pm), ``"ls"`` (least squares).
        Default is ``"ls"``.
    rf_passband_ripple_lvl : float, optional
        Passband ripple level in :math:'M_0^{-1}'.
        Default is ``0.01``.
    rf_stopband_ripple_lvl : float, optional
        Stopband ripple level in :math:'M_0^{-1}'.
        Default is ``0.01``.
    rf_cancel_alpha_phs : bool, optional
        For ``'ex'`` pulses, absorb the alpha phase
        profile from beta's profile, so they cancel for a flatter
        total phase. Default is ``False``.
    dummy_scans : int, optional
        Dummy scans to achieve Steady State condition. The default is ``32``.

    """

    def __init__(
        self,
        system: pp.Opts,
        fov_m: tuple[float],
        slice_thickness_m: float,
        matrix: tuple[int],
        num_slices: int,
        flip_angle_rad: float,
        echo_time_s: float | None = None,
        repetition_time_s: float | None = None,
        receive_bandwidth_Hz: float = 250e3,
        parallel_imaging_accel: int = 1,
        acr_y_size: int | None = None,
        slice_gap_m: float = 0.0,
        partial_fourier_accel_x: float = 1.0,
        partial_fourier_accel_y: float = 1.0,
        readout_oversamp: float = 1.0,
        rf_phase_inc_rad: float | None = None,
        gx_spoiling_angle: float = 32 * np.pi,
        gz_spoiling_angle: float = 32 * np.pi,
        time_bw_product: float = 4.0,
        rf_duration_s: float = 2.0e-3,
        rf_filter_type: str = 'ls',
        rf_passband_ripple_lvl: float = 0.01,
        rf_stopband_ripple_lvl: float = 0.01,
        rf_cancel_alpha_phs: bool = False,
        dummy_scans: int = 32,
    ):
        # Default
        if rf_phase_inc_rad is None:
            rf_phase_inc_rad = np.deg2rad(117.0)

        # Unpack parameters
        fov_x, fov_y = fov_m
        nx, ny = matrix

        # Compute fov_z
        slice_spacing_m = slice_thickness_m + slice_gap_m
        fov_z = slice_spacing_m * num_slices

        # Create excitation module
        pulse = excitation.SpatiallySelectiveExcitation(
            system,
            flip_angle_rad,
            slice_thickness_m,
            duration_s=rf_duration_s,
            time_bw_product=time_bw_product,
            filter_type=rf_filter_type,
            passband_ripple_lvl=rf_passband_ripple_lvl,
            stopband_ripple_lvl=rf_stopband_ripple_lvl,
            cancel_alpha_phs=rf_cancel_alpha_phs,
        )
        rf_exc = pulse.rf
        gz_slc_sel = pulse.gz
        gz_slc_reph = pulse.gzr

        # Initialize phasor
        gy_phasor = phasor.make_phasor('y', system, fov_y, ny)
        gy_prewind = copy.deepcopy(gy_phasor)
        gy_rewind = copy.deepcopy(gy_phasor)
        gz_spoil = phasor.make_crusher(
            'z',
            system,
            gz_spoiling_angle,
            slice_thickness_m,
        )

        # Initialize readout
        read, metadata = readout.spoiled_line_readout(
            system,
            fov_x,
            nx,
            spoiling_angle_rad=gx_spoiling_angle,
            spoiling_thickness_m=slice_thickness_m,
            receive_bandwidth_Hz=receive_bandwidth_Hz,
            oversamp=readout_oversamp,
            partial_fourier_factor=partial_fourier_accel_x,
        )
        gx_prewind = read.gx_prewind
        gx_read = read.gx_read
        gx_spoil = read.gx_spoil

        # Timing
        # ======
        # 1) TE
        rf_duration = pulse.duration
        rf_center_time, _ = pp.calc_rf_center(rf_exc)
        prewind_duration = pp.calc_duration(gz_slc_reph, gy_prewind, gx_prewind)
        min_te = (
            (rf_duration - rf_center_time)
            + prewind_duration
            + metadata.encoding_center_time
        )
        if echo_time_s is None:
            logging.info(f'TE not selected - using minimum TE: {min_te * 1e3} ms')
            echo_time_s = min_te
            te_delay = 0.0
        elif echo_time_s < min_te:
            logging.warning(
                f'Target TE (={echo_time_s * 1e3} ms smaller than minimum {min_te * 1e3} ms - using minimum'
            )
            echo_time_s = min_te
            te_delay = 0.0
        else:
            te_delay = echo_time_s - min_te
        if te_delay:
            te_wait = pp.make_delay(te_delay)
        else:
            te_wait = None

        # 2) TR
        # min_slice_tr = (
        #     echo_time_s +
        #     (pp.calc_duration(gx_read) - metadata.encoding_center_time) +
        #     pp.calc_duration(gz_spoil, gy_prewind, gx_spoil) +
        #     rf_center_time
        # )
        # min_tr = num_slices * min_slice_tr

        # if repetition_time_s is None:
        #     logging.info(f'TR not selected - using minimum TR: {min_tr * 1e3} ms')
        #     repetition_time_s = min_tr
        #     tr_delay = 0.0
        # elif echo_time_s < min_tr:
        #     logging.warning(f'Target TR (={repetition_time_s * 1e3} ms smaller than minimum {min_tr * 1e3} ms - using minimum')
        #     repetition_time_s = min_tr
        #     tr_delay = 0.0
        # else:
        #     tr_delay = (repetition_time_s - min_tr) / num_slices
        min_tr = (
            echo_time_s
            + (pp.calc_duration(gx_read) - metadata.encoding_center_time)
            + pp.calc_duration(gz_spoil, gy_prewind, gx_spoil)
            + rf_center_time
        )
        if repetition_time_s is None:
            logging.info(f'TR not selected - using minimum TR: {min_tr * 1e3} ms')
            repetition_time_s = min_tr
            tr_delay = 0.0
        elif repetition_time_s < min_tr:
            logging.warning(
                f'Target TR (={repetition_time_s * 1e3} ms smaller than minimum {min_tr * 1e3} ms - using minimum'
            )
            repetition_time_s = min_tr
            tr_delay = 0.0
        else:
            tr_delay = repetition_time_s - min_tr
        if tr_delay:
            tr_wait = pp.make_delay(tr_delay)
        else:
            tr_wait = None

        # Perform alignment
        # =================
        # 1) Prewind
        if te_delay:
            gz_slc_reph, te_wait, gy_prewind, gx_prewind = pp.align(
                left=gz_slc_reph,
                right=[te_wait, gy_prewind, gx_prewind],
            )
        else:
            gz_slc_reph, gy_prewind, gx_prewind = pp.align(
                left=gz_slc_reph,
                right=[gy_prewind, gx_prewind],
            )

        # 2) Spoil
        gz_spoil, gy_rewind, gx_spoil = pp.align(
            left=[gx_spoil, gy_rewind, gz_spoil],
        )

        # Scan Table
        # ==========
        # 0) Compute phase encoding partial fourier
        partial_fourer_y_sampling = np.zeros(ny, dtype=int)
        acquired_lines = np.ceil(partial_fourier_accel_y * ny).astype(int).item()
        partial_fourer_y_sampling[-acquired_lines:] = 1

        # 1) Phase encoding
        sampling_mask = pp.make_regular_sampling(
            shape=ny,
            accel=parallel_imaging_accel,
            calib=acr_y_size,
        )
        sampling_mask = partial_fourer_y_sampling * sampling_mask
        pey_scale = (np.arange(ny) - ny // 2) / ny
        pey_scale = pey_scale[sampling_mask]
        y_label = np.arange(ny)[sampling_mask]

        # 2) RF / ADC frequency
        slice_ordering = pp.make_interleaved_ordering_1d(num_slices, 2)
        max_freq_offset = gz_slc_sel.amplitude * slice_spacing_m
        freq_offset = (np.arange(num_slices) - num_slices // 2) * max_freq_offset
        freq_offset = freq_offset[slice_ordering]
        slice_label = np.arange(num_slices)[slice_ordering]

        # 3) RF phase offset
        num_scans = dummy_scans + (pey_scale.size * freq_offset.size)
        phase_offset = excitation.rf_spoil_table(num_scans, rf_phase_inc_rad)

        # Assign
        # ======
        # 1) Dynamic parameters
        self.dummy_scans = dummy_scans
        self.phase_offset = phase_offset
        self.freq_offset = freq_offset
        self.phase_offset = phase_offset
        self.pey_scale = pey_scale

        # 2) Excitation
        self.rf_exc = rf_exc
        self.gz_slc_sel = gz_slc_sel

        # 3) Prewind
        self.gx_prewind = gx_prewind
        self.gy_prewind = gy_prewind
        self.gz_slc_reph = gz_slc_reph
        self.te_wait = te_wait

        # 4) Readout
        self.gx_read = gx_read
        self.adc = read.adc

        # 5) Spoil
        self.gx_spoil = gx_spoil
        self.gy_rewind = gy_rewind
        self.gz_spoil = gz_spoil

        # 6) Pause
        self.tr_wait = tr_wait

        # 7) Metadata
        self.system = system
        self.fov_m = [fov_x, fov_y, fov_z]
        self.matrix = [nx, ny, num_slices]
        self.resolution = [fov_x / nx, fov_y / ny, slice_thickness_m]
        self.encoding_size = [metadata.encoding_size, ny, num_slices]
        self.encoding_center = [metadata.encoding_center, ny // 2, num_slices // 2]
        self.echo_time_s = echo_time_s
        self.repetition_time_s = repetition_time_s
        self.duration = repetition_time_s * num_scans
        self.y_label = y_label
        self.slice_label = slice_label
        self.count = 0

    def append(
        self,
        seq: pp.Sequence | None = None,
        y_index: int | None = None,
        z_index: int | None = None,
        label_TR_start: bool = False,
        mode: ScanType = 'SCAN',
    ) -> pp.Sequence:
        """
        Append block to input sequence.

        Parameters
        ----------
        seq : pp.Sequence, optional
            Input Pulseq Sequence. If not provided, create a new sequence
            and append the block(s).
        y_index : int, optional
            Phase encoding line to be acquired.
        z_index : int, optional
            Slice to be acquired.
        label_TR_start : bool, optional
            If ``True``, label the excitation block using ``TRID`` label.
        mode : ScanType, optional
            Scan mode flag. If ``'SCAN'``, this is an imaging scan.
            If ``'DUMMY'``, turn off ADC. The default is ``'SCAN'``.

        Returns
        -------
        seq : pp.Sequence
            Modified Pulseq Sequence with a full SPGR TR.

        """
        if seq is None:
            standalone = True
            seq = pp.Sequence(system=self.system)
        else:
            standalone = False

        # Get current rf/adc phase
        phase_offset = self.phase_offset[self.count]

        # Get phase encoding scaling and rf/adc frequency offsets
        if standalone:
            pey_scale = 1.0
            freq_offset = 0.0
        else:
            pey_scale = self.pey_scale[y_index]
            freq_offset = self.freq_offset[z_index]

        # Apply phase/freq offsets
        self.rf_exc.freq_offset = freq_offset
        self.adc.freq_offset = freq_offset
        self.rf_exc.phase_offset = phase_offset
        self.adc.phase_offset = phase_offset

        # Apply scaling
        gy_prewind = pp.scale_grad(self.gy_prewind, scale=pey_scale)
        gy_rewind = pp.scale_grad(self.gy_rewind, scale=-pey_scale)

        # Add blocks
        # ==========
        # 1) Slice selective excitation
        if label_TR_start:
            trid_label = pp.make_label(type='SET', label='TRID', value=1)
            seq.add_block(self.rf_exc, self.gz_slc_sel, trid_label)
        else:
            seq.add_block(self.rf_exc, self.gz_slc_sel)

        # 2) Slice rephasing, phase encoding and readout prewinder
        if self.te_wait is not None:
            seq.add_block(self.gx_prewind, gy_prewind, self.gz_slc_reph, self.te_wait)
        else:
            seq.add_block(self.gx_prewind, gy_prewind, self.gz_slc_reph)

        # 3) Line readout
        if mode == 'SCAN' and standalone:
            seq.add_block(self.gx_read, self.adc)
        elif mode == 'SCAN':
            y_label = pp.make_label(
                type='SET', label='LIN', value=self.y_label[y_index]
            )
            z_label = pp.make_label(
                type='SET', label='SLC', value=self.z_label[z_index]
            )
            seq.add_block(self.gx_read, self.adc, y_label, z_label)
        elif mode == 'DUMMY':
            seq.add_block(self.gx_read)

        # 4) Phase encoding rewinder, readout and slice spoil
        seq.add_block(self.gx_spoil, gy_rewind, self.gz_spoil)

        # 5) Pause until next TR
        if self.tr_wait is not None:
            seq.add_block(self.tr_wait)

        return seq

    def __call__(
        self,
        seq: pp.Sequence | None = None,
        y_index: int | None = None,
        z_index: int | None = None,
        label_TR_start: bool = False,
        mode: ScanType = 'SCAN',
    ) -> pp.Sequence:
        return self.append(seq, y_index, z_index, label_TR_start, mode)
