"""2D Spoiled Gradient Echo sequence."""

__all__ = ["design_2D_spgr"]

from collections.abc import Iterable

import warnings

import numpy as np
import pypulseq as pp

from .._core import Sequence
from .._opts import get_opts

from .. import blocks
from .. import plan


def design_2D_spgr(
    fov: Iterable[float],
    slice_thickness: float,
    matrix_size: Iterable[int],
    n_slices: int,
    flip_angle: float,
    TE: float = 0.0,
    TR: float = 0.0,
    Ry: int = 1,
    Cy: int | None = None,
    opts_dict: str | dict | None = None,
    slice_gap: float = 0.0,
    dummy_scans=10,
    calib_scans=10,
    platform: str = "pulseq",
):
    """
    Generate a 2D Spoiled Gradient Recalled Echo (SPGR) pulse sequence.

    This function designs a 2D SPGR sequence based on the provided field of view (FOV), matrix size,
    number of slices, slice thickness and spacing, flip angle, recovery time,
    and hardware constraints such as maximum gradient amplitude and slew rate.
    The output can be formatted in different sequence file formats if specified.

    Parameters
    ----------
    fov : Iterable[float]
        Field of view along each spatial dimension (fov_x, fov_y) in mm.
        If scalar, assume squared fov.
    slice_thickness : float)
        Slice thickness in mm.
    matrix_size  : Iterable[int]
        Number of voxels along each spatial dimension (nx, ny) (matrix size).
        If scalar, assume squared matrix size.
    n_slices : int
        Number of slices.
    flip_angle : float
        Flip angle in degrees.
    TE: float, optional
        Target Echo Time in ``[s]``. It is automatically extended to minimum TE.
        The default is ``0.0``
    TR: float, optional
        Target Repetition Time in ``[s]``. It is automatically extended to minimum TR.
        The default is ``0.0``
    Ry: int, optional
        Parallel Imaging undersampling factor. The default is ``1`` (no undersampling).
    Cy: int | None, optional
        Parallel Imaging autocalibration lines. The default is ``None`` (external calibration).
    opts_dict : str | dict | None, optional
        Either scanner identifier or a dictionary with the following keys:

            * max_grad: maximum gradient strength in ``[mT/m]``.
            * max_slew: maximum gradient slew rate in ``[T/m/s]``.
            * rf_dead_time: initial RF delay time in ``[s]``.
            * rf_ringdown_time: final RF wait time in ``[s]``.
            * adc_dead_time: initial ADC delay time in ``[s]``.
            * adc_raster_time: ADC raster time (i.e., signal dwell time) in ``[s]``.
            * rf_raster_time: RF raster time in ``[s]``.
            * grad_raster_time: gradient raster time in ``[s]``.
            * B0: field strength in ``[T]``

        If ``None``, use PyPulseq default Opts. The default is ``None``.
    slice_gap : float, optional
        Additional slice gap in mm. The default is 0.0 (contiguous slices).
    dummy_scans : int, optional
        Number of dummy scans (without ADC) to reach steady state.
        The default is ``10``.
    calib_scans : int, optional
        Number of scans (with ADC, without phase encoding) to calibrate transmit gain.
        The default is ``10``.
    platform : str, optional
        The target platform for the sequence. Acceptable values are 'pulseq' (alias for 'siemens')
        and 'toppe' (alias for 'gehc').

    Returns
    -------
    seq : object or dict
        The generated SPGR sequence. If `seqformat` is a string, the sequence is returned in the specified format.
        If `seqformat` is False, the sequence is returned as an internal representation.

    Notes
    -----
    - This function is designed to work within the constraints of MRI scanners, taking into account the physical limits
      on gradient amplitude and slew rates.
    - The flip angle (`flip_angle`) controls the excitation of spins and directly impacts the signal-to-noise ratio (SNR) and contrast.

    Examples
    --------
    Generate a 2D SPGR sequence for a single 5 mm thick slice and 240x240 mm FOV, 256x256 matrix size,
    15-degree flip angle and hardware limits 40 mT/m, 150 T/m/s, 4e-6 s raster time as:

    >>> from pulserver.sequences import design_2D_spgr
    >>> opts_dict = {"max_grad": 40, "max_slew": 150, "grad_raster_time": 4e-6, "rf_raster_time": 4e-6}

    Actual design:

    >>> design_2D_spgr(240.0, 5.0, 256, 1, 15.0, opts_dict=opts_dict=opts_dict)

    Generate the same sequence and export it in GEHC format:

    >>> design_2D_spgr(240.0, 5.0, 256, 1, 15.0, opts_dict=opts_dict, platform='gehc')

    """
    # Sequence Parameters
    # -------------------
    rf_spoiling_inc = 117.0  # RF spoiling increment

    # initialize prescription
    slice_spacing = slice_thickness + slice_gap

    if np.isscalar(fov):
        FOVx, FOVy = fov, fov
    else:
        FOVx, FOVy = fov

    if np.isscalar(matrix_size):
        Nx, Ny = matrix_size, matrix_size
    else:
        Nx, Ny = matrix_size[0], matrix_size[1]

    # initialize system limits
    if opts_dict is None:
        warnings.warn("opts_dict not provided - using PyPulseq defaults")

    system_limits = get_opts(opts_dict)

    # initialize sequence object
    seq = Sequence(system=system_limits, platform=platform)

    # Define Blocks
    # -------------
    # create excitation and slice rephasing blocks
    exc_block, slice_reph_block = blocks.make_slr_pulse(
        system_limits, flip_angle, slice_thickness
    )

    # create phase encoding gradient, readout pre-/re-winder and readout/adc blocks:
    phase_enc = blocks.make_phase_encoding("y", system_limits, FOVy, Ny)
    readout_block, readout_prewind_block = blocks.make_line_readout(
        system_limits, FOVx, Nx
    )

    # create combined phase encoding + readout prewinder block
    phase_enc_block = {"gy": phase_enc, **readout_prewind_block}

    # create spoiling block
    spoil_block = {
        "gz": blocks.make_spoiler_gradient(
            "z", system_limits, ncycles=4, voxel_size=slice_thickness
        )
    }

    # Calculate timing
    # ----------------
    delay_TE, act_TE = blocks.calc_delay(
        system_limits,
        TE,
        0.5 * exc_block["gz"].flat_time,
        exc_block["gz"].fall_time,
        slice_reph_block["gz"],
        pp.calc_duration(*phase_enc_block.values()),
        0.5 * pp.calc_duration(readout_block["gx"]),
    )

    delay_TR, act_TR = blocks.calc_delay(
        system_limits,
        TR,
        act_TE,
        0.5 * pp.calc_duration(readout_block["gx"]),
        pp.calc_duration(*phase_enc_block.values()),
        spoil_block["gz"],
    )

    # register parent blocks
    seq.register_block(name="excitation", **exc_block)
    seq.register_block(name="slice_rephasing", **slice_reph_block, delay=delay_TE)
    seq.register_block(name="readout", **readout_block)
    seq.register_block(name="dummy_readout", gx=readout_block["gx"])
    seq.register_block(name="phase_encoding", **phase_enc_block)
    seq.register_block(name="spoiling", **spoil_block, delay=delay_TR)

    # Prepare header
    # --------------
    seq.initialize_header(2)
    seq.set_definition("shape", Nx, Ny, n_slices)
    seq.set_definition("fov", FOVx, FOVy, slice_spacing)
    seq.set_definition("limits", n_views=Ny, n_slices=n_slices)
    seq.set_definition("flip", flip_angle)
    seq.set_definition("TE", act_TE)
    seq.set_definition("TR", act_TR)
    seq.set_definition("dwell", system_limits.adc_raster_time)
    seq.set_definition("spoiling_inc", 117.0)
    seq.set_definition("ndummies", dummy_scans)

    # Define sequence plan
    # --------------------
    # scan duration
    imaging_scans = Ny * n_slices

    # generate rf phase schedule
    rf_phases = plan.RfPhaseCycle(
        num_pulses=dummy_scans + imaging_scans + calib_scans,
        phase_increment=rf_spoiling_inc,
    )

    # create Gy and RF frequency offset schedule to achieve the requested FOV, in-plane resolution and number of slices
    encoding_plan, _ = plan.cartesian2D(
        g_slice_select=exc_block["gz"],
        slice_thickness=slice_thickness,
        slice_gap=slice_gap,
        Ry=int(Ry),
        calib=Cy,
        n_slices=n_slices,
        ny=Ny,
        dummy_shots=calib_scans + dummy_scans,
    )

    # Set up scan loop
    # ----------------
    # Steady state preparation
    seq.section(name="ss_prep")
    for n in range(dummy_scans):
        # get dynamic sequence parameters
        rf_phase = rf_phases()
        encoding, _ = encoding_plan()

        # update sequence loop
        seq.add_block("excitation", rf_phase=rf_phase, rf_freq=encoding.rf_freq)
        seq.add_block("slice_rephasing")
        seq.add_block("phase_encoding", gy_amp=encoding.gy_amp)
        seq.add_block("dummy_readout")
        seq.add_block("phase_encoding", gy_amp=-encoding.gy_amp)
        seq.add_block("spoiling")

    # Actual sequence
    seq.section(name="scan_loop")
    for n in range(imaging_scans + calib_scans):
        # get dynamic sequence parameters
        rf_phase = rf_phases()
        encoding, label = encoding_plan()

        # update sequence loop
        seq.add_block("excitation", rf_phase=rf_phase, rf_freq=encoding.rf_freq)
        seq.add_block("slice_rephasing")
        seq.add_block("phase_encoding", gy_amp=encoding.gy_amp)
        seq.add_block("readout", adc_phase=rf_phase)
        seq.add_block("phase_encoding", gy_amp=-encoding.gy_amp)
        seq.add_block("spoiling")

        # update data labeling
        seq.set_label(iy=label.iy, islice=label.islice)

    # build the sequence
    return seq.build(ngain=calib_scans)
