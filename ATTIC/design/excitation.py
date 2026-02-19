"""Pulseq Excitation design helpers."""

__all__ = [
    'FrequencySelectiveExcitation',
    'NonselectiveExcitation',
    'SmsExcitation',
    'SpatiallySelectiveExcitation',
    'SpspExcitation',
    'phase_cycling_table',
    'rf_spoil_table',
]


import numpy as np
from numpy.typing import NDArray

from .. import pulseq as pp


class RFBlockMixin:
    """
    Base RF Block pulse.

    Attributes
    ----------
    rf_id : int
        ID of underlying Pulseq RF pulse event.
    freq_offset : float
        RF pulse event frequency offset in ``[Hz]``.
    freq_ppm : float
        RF pulse event frequency offset in ``[ppm]``.
    phase_offset : float
        RF pulse event phase offset in ``[rad]``.
    phase_ppm : float
        RF pulse event phase offset in ``[ppm]``.
    duration : float
        RF segment duraiton in ``[s]``.
    """

    @property
    def rf_id(self):
        return self.rf.id

    @rf_id.setter
    def rf_id(self, value: int):
        self.rf.id = value

    @property
    def freq_offset(self):
        return self.rf.freq_offset

    @freq_offset.setter
    def freq_offset(self, value_Hz: float):
        self.rf.freq_offset = value_Hz

    @property
    def freq_ppm(self):
        return self.rf.freq_ppm

    @freq_ppm.setter
    def freq_ppm(self, value_ppm: float):
        self.rf.freq_ppm = value_ppm

    @property
    def phase_offset(self):
        return self.rf.phase_offset

    @phase_offset.setter
    def phase_offset(self, value_rad: float):
        self.rf.phase_offset = value_rad

    @property
    def phase_ppm(self):
        return self.rf.phase_ppm

    @phase_ppm.setter
    def phase_ppm(self, value_ppm: float):
        self.rf.phase_ppm = value_ppm

    @property
    def duration(self):
        return self._duration

    def append(self, seq: pp.Sequence | None = None) -> pp.Sequence:
        """
        Append block to input sequence.

        Parameters
        ----------
        seq : pp.Sequence | None, optional
            Input Pulseq Sequence. If not provided, create a new sequence
            and append the block(s).

        Returns
        -------
        seq : pp.Sequence
            Modified Pulseq Sequence with RF block as last added block.

        """
        if seq is None:
            seq = pp.Sequence(system=self.system)

        # Add excitation
        seq.add_block(self.rf)

        return seq

    def __call__(self, seq: pp.Sequence | None = None) -> pp.Sequence:
        return self.append(seq)


class SoftRFBlockMixin(RFBlockMixin):
    """
    Base soft RF Block pulse.

    Attributes
    ----------
    rf_id : int
        ID of underlying Pulseq RF pulse event.
    gz_id : int
        ID of underlying Pulseq selection gradient event.
    gz_area: float
        Area of underlying Pulseq selection gradient event (read-only).
    gz_first: float
        Initial amplitude of underlying Pulseq selection gradient event (read-only).
    gz_last: float
        Final amplitude of underlying Pulseq selection gradient event (read-only).
    gzr_id : int
        ID of underlying Pulseq rephasing gradient event.
    gzr_area: float
        Area of underlying Pulseq rephasing gradient event (read-only).
    gzr_first: float
        Initial amplitude of underlying Pulseq rephasing gradient event (read-only).
    gzr_last: float
        Final amplitude of underlying Pulseq rephasing gradient event (read-only).
    freq_offset : float
        RF pulse event frequency offset in ``[Hz]``.
    freq_ppm : float
        RF pulse event frequency offset in ``[ppm]``.
    phase_offset : float
        RF pulse event phase offset in ``[rad]``.
    phase_ppm : float
        RF pulse event phase offset in ``[ppm]``.
    duration : float
        RF segment duraiton in ``[s]``.
    """

    @property
    def gz_id(self):
        return self.gz.id

    @gz_id.setter
    def gz_id(self, value: int):
        self.gz.id = value

    @property
    def gz_area(self):
        return self.gz.area

    @property
    def gz_first(self):
        if self.gz.type == 'trap':
            return 0.0
        return self.gz.first

    @property
    def gz_last(self):
        if self.gz.type == 'trap':
            return 0.0
        return self.gz.last

    @property
    def gzr_id(self):
        if self.gzr is not None:
            return self.gzr.id

    @gzr_id.setter
    def gzr_id(self, value: int):
        if self.gzr is not None:
            self.gzr.id = value

    @property
    def gzr_area(self):
        if self.gzr is not None:
            return self.gzr.area

    @property
    def gzr_first(self):
        if self.gzr is not None:
            if self.gzr.type == 'trap':
                return 0.0
            return self.gzr.first

    @property
    def gzr_last(self):
        if self.gzr is not None:
            if self.gzr.type == 'trap':
                return 0.0
            return self.gzr.last

    def append(self, seq: pp.Sequence | None = None) -> pp.Sequence:
        if seq is None:
            seq = pp.Sequence(system=self.system)

        # Add excitation
        seq.add_block(self.rf, self.gz)

        # Add slice rephasing
        if self.gzr is not None:
            seq.add_block(self.gzr)

        return seq


class NonselectiveExcitation(RFBlockMixin):
    """
    Nonselective RF excitation.

    Parameters
    ----------
    system : pp.Opts
        Pulseq system limits.
    flip_angle_rad : float
        Flip angle in ``[rad]``.
    duration_s : float, optional
        Pulse duration in ``[s]``. Default is ``0.5e-3 s``

    Attributes
    ----------
    system : pp.Opts
        Pulseq system limits.
    rf : SimpleNamespace
        Pulseq RF event.
    rf_id : int
        ID of underlying Pulseq RF pulse event.
    gz_id : int
        ID of underlying Pulseq selection gradient event.
    gzr_id : int
        ID of underlying Pulseq rephasing gradient event.
    freq_offset : float
        RF pulse event frequency offset in ``[Hz]``.
    freq_ppm : float
        RF pulse event frequency offset in ``[ppm]``.
    phase_offset : float
        RF pulse event phase offset in ``[rad]``.
    phase_ppm : float
        RF pulse event phase offset in ``[ppm]``.
    duration : float
        RF segment duraiton in ``[s]``.
    """

    def __init__(
        self,
        system: pp.Opts,
        flip_angle_rad: float,
        duration_s: float = 0.5e-3,
    ):
        self.system = system
        self.rf = pp.make_block_pulse(
            system=system,
            flip_angle=flip_angle_rad,
            duration=duration_s,
            use='excitation',
        )
        self._duration = pp.calc_duration(self.rf)


class FrequencySelectiveExcitation(RFBlockMixin):
    r"""
    Shinnar-LeRoux frequency selective RF excitation.

    Parameters
    ----------
    system : pp.Opts
        Pulseq system limits.
    flip_angle_rad : float
        Flip angle in ``[rad]``.
    bandwidth_Hz : float
        Pulse spectral bandwidth in ``[Hz]``.
    duration_s : float
        Pulse duration in ``[s]``.
    filter_type : str, optional
        Type of filter to use: ``"ms"`` (sinc),
        ``"pm``, (Parks-McClellan equal-ripple),
        ``"min"`` (minphase using factored pm),
        ``"max"`` (maxphase using factored pm), ``"ls"`` (least squares).
        Default is ``"ls"``.
    passband_ripple_lvl : float, optional
        Passband ripple level in :math:'M_0^{-1}'.
        Default is ``0.01``.
    stopband_ripple_lvl : float, optional
        Stopband ripple level in :math:'M_0^{-1}'.
        Default is ``0.01``.
    cancel_alpha_phs : bool, optional
        For ``'ex'`` pulses, absorb the alpha phase
        profile from beta's profile, so they cancel for a flatter
        total phase. Default is ``False``.

    Attributes
    ----------
    system : pp.Opts
        Pulseq system limits.
    rf : SimpleNamespace
        Pulseq RF event.
    rf_id : int
        ID of underlying Pulseq RF pulse event.
    gz_id : int
        ID of underlying Pulseq selection gradient event.
    gzr_id : int
        ID of underlying Pulseq rephasing gradient event.
    freq_offset : float
        RF pulse event frequency offset in ``[Hz]``.
    freq_ppm : float
        RF pulse event frequency offset in ``[ppm]``.
    phase_offset : float
        RF pulse event phase offset in ``[rad]``.
    phase_ppm : float
        RF pulse event phase offset in ``[ppm]``.
    duration : float
        RF segment duraiton in ``[s]``.
    """

    def __init__(
        self,
        system: pp.Opts,
        flip_angle_rad: float,
        bandwidth_Hz: float,
        duration_s: float = 2.0e-3,
        filter_type: str = 'ls',
        passband_ripple_lvl: float = 0.01,
        stopband_ripple_lvl: float = 0.01,
        cancel_alpha_phs: bool = False,
    ):
        self.system = system
        self.rf = pp.make_slr_pulse(
            system=system,
            flip_angle=flip_angle_rad,
            duration=duration_s,
            bandwidth=bandwidth_Hz,
            use='excitation',
            filter_type=filter_type,
            passband_ripple_lvl=passband_ripple_lvl,
            stopband_ripple_lvl=stopband_ripple_lvl,
            cancel_alpha_phs=cancel_alpha_phs,
        )
        self._duration = pp.calc_duration(self.rf)


class SpatiallySelectiveExcitation(SoftRFBlockMixin):
    r"""
    Shinnar-LeRoux spatially (slice or slab) selective RF excitation.

    Parameters
    ----------
    system : pp.Opts
        Pulseq system limits.
    flip_angle_rad : float
        Flip angle in ``[rad]``.
    slice_thickness_m : float
        Slice thickness in ``[m]``.
    time_bw_product : float, optional
        Pulse time / bandwidth product. The default is ``4.0``.
    duration_s : float, optional
        Pulse duration in ``[s]``. Default is ``2.0e-3 s``.
    filter_type : str, optional
        Type of filter to use: ``"ms"`` (sinc),
        ``"pm``, (Parks-McClellan equal-ripple),
        ``"min"`` (minphase using factored pm),
        ``"max"`` (maxphase using factored pm), ``"ls"`` (least squares).
        Default is ``"ls"``.
    passband_ripple_lvl : float, optional
        Passband ripple level in :math:'M_0^{-1}'.
        Default is ``0.01``.
    stopband_ripple_lvl : float, optional
        Stopband ripple level in :math:'M_0^{-1}'.
        Default is ``0.01``.
    cancel_alpha_phs : bool, optional
        For ``'ex'`` pulses, absorb the alpha phase
        profile from beta's profile, so they cancel for a flatter
        total phase. Default is ``False``.
    truncate_block: bool, optional
        If ``True``, truncate the excitation immediately after rf_deadtime.
        It can be used to build time-optimized sequences like Fast Spin Echo,
        in order to efficiently merge e.g., excitation and spoil. The default is
        ``False``

    Attributes
    ----------
    system : pp.Opts
        Pulseq system limits.
    rf : SimpleNamespace
        Pulseq RF event.
    gz : SimpleNamespace
        Pulseq slice selection Grad event.
    gzr : SimpleNamespace
        Pulseq slice rephasing Grad event.
    rf_id : int
        ID of underlying Pulseq RF pulse event.
    gz_id : int
        ID of underlying Pulseq selection gradient event.
    gzr_id : int
        ID of underlying Pulseq rephasing gradient event.
    freq_offset : float
        RF pulse event frequency offset in ``[Hz]``.
    freq_ppm : float
        RF pulse event frequency offset in ``[ppm]``.
    phase_offset : float
        RF pulse event phase offset in ``[rad]``.
    phase_ppm : float
        RF pulse event phase offset in ``[ppm]``.
    duration : float
        RF segment duraiton in ``[s]``.
    """

    def __init__(
        self,
        system: pp.Opts,
        flip_angle_rad: float,
        slice_thickness_m: float,
        time_bw_product: float = 4.0,
        duration_s: float = 2.0e-3,
        filter_type: str = 'ls',
        passband_ripple_lvl: float = 0.01,
        stopband_ripple_lvl: float = 0.01,
        cancel_alpha_phs: bool = False,
        truncate_block: bool = False,
    ):
        self.system = system
        rf, gz, gzr = pp.make_slr_pulse(
            system=system,
            flip_angle=flip_angle_rad,
            duration=duration_s,
            return_gz=True,
            slice_thickness=slice_thickness_m,
            time_bw_product=time_bw_product,
            use='excitation',
            filter_type=filter_type,
            passband_ripple_lvl=passband_ripple_lvl,
            stopband_ripple_lvl=stopband_ripple_lvl,
            cancel_alpha_phs=cancel_alpha_phs,
            absorb_rf_deadtime=truncate_block,
        )
        if truncate_block:
            gz, _ = pp.split_waveform_at(
                gz,
                time_point=gz.delay + gz.rise_time + gz.flat_time,
                system=system,
            )
            gzr = None
        self.rf = rf
        self.gz = gz
        self.gzr = gzr
        self._duration = pp.calc_duration(rf, gz)
        if gzr is not None:
            self._duration += pp.calc_duration(gzr)


class SmsExcitation(SoftRFBlockMixin):
    r"""
    Shinnar-LeRoux multislice RF excitation.

    Parameters
    ----------
    system : pp.Opts
        Pulseq system limits.
    flip_angle_rad : float
        Flip angle in ``[rad]``.
    num_slices : int
        Number of simultaneously excited slices.
    slice_thickness_m : float
        Slice thickness in ``[m]``.
    slice_separation_m : float, optional
        Slice separation in ``[m]``. Default to contiguous slices
        (``slice_separation_m =  num_slices * slice_thickness_m``).
    time_bw_product : float, optional
        Pulse time / bandwidth product. The default is ``4.0``.
    duration_s : float, optional
        Pulse duration in ``[s]``. Default is ``4.0e-3 s``.
    filter_type : str, optional
        Type of filter to use: ``"ms"`` (sinc),
        ``"pm``, (Parks-McClellan equal-ripple),
        ``"min"`` (minphase using factored pm),
        ``"max"`` (maxphase using factored pm), ``"ls"`` (least squares).
        Default is ``"ls"``.
    passband_ripple_lvl : float, optional
        Passband ripple level in :math:'M_0^{-1}'.
        Default is ``0.01``.
    stopband_ripple_lvl : float, optional
        Stopband ripple level in :math:'M_0^{-1}'.
        Default is ``0.01``.
    cancel_alpha_phs : bool, optional
        For ``'ex'`` pulses, absorb the alpha phase
        profile from beta's profile, so they cancel for a flatter
        total phase. Default is ``False``.
    truncate_block: bool, optional
        If ``True``, truncate the excitation immediately after rf_deadtime.
        It can be used to build time-optimized sequences like Fast Spin Echo,
        in order to efficiently merge e.g., excitation and spoil. The default is
        ``False``

    Attributes
    ----------
    system : pp.Opts
        Pulseq system limits.
    rf : SimpleNamespace
        Pulseq RF event.
    gz : SimpleNamespace
        Pulseq slice selection Grad event.
    gzr : SimpleNamespace
        Pulseq slice rephasing Grad event.
    rf_id : int
        ID of underlying Pulseq RF pulse event.
    gz_id : int
        ID of underlying Pulseq selection gradient event.
    gzr_id : int
        ID of underlying Pulseq rephasing gradient event.
    freq_offset : float
        RF pulse event frequency offset in ``[Hz]``.
    freq_ppm : float
        RF pulse event frequency offset in ``[ppm]``.
    phase_offset : float
        RF pulse event phase offset in ``[rad]``.
    phase_ppm : float
        RF pulse event phase offset in ``[ppm]``.
    duration : float
        RF segment duraiton in ``[s]``.
    """

    def __init__(
        self,
        system: pp.Opts,
        flip_angle_rad: float,
        num_slices: int,
        slice_thickness_m: float,
        slice_separation_m: float | None = None,
        time_bw_product: float = 4.0,
        duration_s: float = 4.0e-3,
        filter_type: str = 'ls',
        passband_ripple_lvl: float = 0.01,
        stopband_ripple_lvl: float = 0.01,
        cancel_alpha_phs: bool = False,
        reference_phase: str = 'None',
        truncate_block: bool = False,
    ):
        self.system = system
        rf, gz, gzr = pp.make_sms_pulse(
            system=system,
            flip_angle=flip_angle_rad,
            n_slices=num_slices,
            slice_thickness=slice_thickness_m,
            slice_separation=slice_separation_m,
            time_bw_product=time_bw_product,
            duration=duration_s,
            use='excitation',
            filter_type=filter_type,
            passband_ripple_lvl=passband_ripple_lvl,
            stopband_ripple_lvl=stopband_ripple_lvl,
            cancel_alpha_phs=cancel_alpha_phs,
            reference_phase=reference_phase,
            absorb_rf_deadtime=truncate_block,
        )
        if truncate_block:
            gz, _ = pp.split_waveform_at(
                gz,
                time_point=gz.delay + gz.rise_time + gz.flat_time,
                system=system,
            )
            gzr = None
        self.rf = rf
        self.gz = gz
        self.gzr = gzr
        self._duration = pp.calc_duration(rf, gz)
        if gzr is not None:
            self._duration += pp.calc_duration(gzr)


class SpspExcitation(SoftRFBlockMixin):
    r"""
    Shinnar-LeRoux spectral and spatial selective RF excitation.

    Parameters
    ----------
    system : pp.Opts
        Pulseq system limits.
    flip_angle_rad : float
        Flip angle in ``[rad]``.
    slice_thickness_m : float
        Slice thickness in ``[m]``.
    freq_bandwidth_Hz : float
        Pulse spectral bandwidth in ``[Hz]``.
    spat_time_bw_product : float, optional
        Pulse time/spatial_bandwidth product. The default is ``4.0``.
    duration_s : float, optional
        Pulse duration in ``[s]``. Default is ``10.0e-3 s``.
    spat_filter_type : str, optional
        Type of filter to use for spatial profile: ``"ms"`` (sinc),
        ``"pm``, (Parks-McClellan equal-ripple),
        ``"min"`` (minphase using factored pm),
        ``"max"`` (maxphase using factored pm), ``"ls"`` (least squares).
        Default is ``"ls"``.
    freq_filter_type : str, optional
        Type of filter to use for spectral profile: ``"ms"`` (sinc),
        ``"pm``, (Parks-McClellan equal-ripple),
        ``"min"`` (minphase using factored pm),
        ``"max"`` (maxphase using factored pm), ``"ls"`` (least squares).
        Default is ``"pm"``.
    spat_passband_ripple_lvl : float, optional
        Spatial passband ripple level in :math:'M_0^{-1}'.
        Default is ``0.01``.
    spat_stopband_ripple_lvl : float, optional
        Spatial stopband ripple level in :math:'M_0^{-1}'.
        Default is ``0.01``.
    freq_passband_ripple_lvl : float, optional
        Spectral passband ripple level in :math:'M_0^{-1}'.
        Default is ``0.01``.
    freq_stopband_ripple_lvl : float, optional
        Spectral stopband ripple level in :math:'M_0^{-1}'.
        Default is ``0.01``.
    num_lobes : int, optional
        Number of sub-lobes within the given duration. Default is ``14``.
    flyback : bool, optional
        If ``True``, use flyback EPI slice selection grad instead of bipolar.
        Default is ``False``.

    Attributes
    ----------
    system : pp.Opts
        Pulseq system limits.
    rf : SimpleNamespace
        Pulseq RF event.
    gz : SimpleNamespace
        Pulseq slice selection Grad event.
    gzr : SimpleNamespace
        Pulseq slice rephasing Grad event.
    rf_id : int
        ID of underlying Pulseq RF pulse event.
    gz_id : int
        ID of underlying Pulseq selection gradient event.
    gzr_id : int
        ID of underlying Pulseq rephasing gradient event.
    freq_offset : float
        RF pulse event frequency offset in ``[Hz]``.
    freq_ppm : float
        RF pulse event frequency offset in ``[ppm]``.
    phase_offset : float
        RF pulse event phase offset in ``[rad]``.
    phase_ppm : float
        RF pulse event phase offset in ``[ppm]``.
    duration : float
        RF segment duraiton in ``[s]``.
    """

    def __init__(
        self,
        system: pp.Opts,
        flip_angle_rad: float,
        slice_thickness_m: float,
        freq_bandwidth_Hz: float,
        spat_time_bw_product: float = 4.0,
        duration_s: float = 10.0e-3,
        spat_filter_type: str = 'ls',
        freq_filter_type: str = 'pm',
        spat_passband_ripple_lvl: float = 0.01,
        spat_stopband_ripple_lvl: float = 0.01,
        freq_passband_ripple_lvl: float = 0.01,
        freq_stopband_ripple_lvl: float = 0.01,
        num_lobes: int = 14,
        flyback: bool = False,
    ):
        self.system = system
        rf, gz, gzr = pp.make_spsp_pulse(
            system=system,
            flip_angle=flip_angle_rad,
            slice_thickness=slice_thickness_m,
            freq_bandwidth=freq_bandwidth_Hz,
            duration=duration_s,
            spat_time_bw_product=spat_time_bw_product,
            use='excitation',
            spat_filter_type=spat_filter_type,
            spat_passband_ripple_lvl=spat_passband_ripple_lvl,
            spat_stopband_ripple_lvl=spat_stopband_ripple_lvl,
            freq_filter_type=freq_filter_type,
            freq_passband_ripple_lvl=freq_passband_ripple_lvl,
            freq_stopband_ripple_lvl=freq_stopband_ripple_lvl,
            n_lobes=num_lobes,
            flyback=flyback,
        )
        self.rf = rf
        self.gz = gz
        self.gzr = gzr
        self._duration = pp.calc_duration(rf, gz) + pp.calc_duration(gzr)


def rf_spoil_table(num_pulses: int, phase_increment: float) -> NDArray[float]:
    """
    Calculate phase increment pattern for RF spoiling.

    Parameters
    ----------
    num_pulses : int
        Number of RF pulses.
    phase_increment : float
        Phase increment in radians (float) for quadratic spoiling.

    Returns
    -------
    numpy.ndarray
        Phase increment pattern in radians.

    Examples
    --------
    Quadratic phase spoiling with 90-degree increments for 4 RF pulses:

    >>> from pulserver.design import rf_spoil_table
    >>> rf_spoil_table(4, np.pi / 2)
    array([ 0.        ,  1.57079633,  4.71238898,  9.42477796])

    Ported from:
    Shaihan Malik, July 2017

    """
    pulse_indices = np.arange(num_pulses)
    phase_pattern = pulse_indices * (pulse_indices + 1) / 2 * phase_increment
    phase = np.deg2rad(phase_pattern)
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def phase_cycling_table(num_pulses: int, phase_increment: float) -> NDArray[float]:
    """
    Calculate phase increment pattern for RF cycling.

    Parameters
    ----------
    num_pulses : int
        Number of RF pulses.
    phase_increment : float
        Phase increment in radians (float) for phase cycling.

    Returns
    -------
    numpy.ndarray
        Phase increment pattern in radians.

    Examples
    --------
    phase cycling for 4 RF pulses:

    >>> from pulserver.design import phase_cycling_table
    >>> phase_cycling_table(4, np.pi)
    array([0.        , 3.14159265, 0.        , 3.14159265])

    Ported from:
    Shaihan Malik, July 2017

    """
    phase = np.arange(num_pulses) * phase_increment
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


# %% Internal Helpers
# def _map_to_0_pi(angles: NDArray[float]) -> NDArray[float]:
#     angles = np.mod(angles, 360.0)
#     angles = np.where(angles > 180.0, 360.0 - angles, angles)
#     return angles
