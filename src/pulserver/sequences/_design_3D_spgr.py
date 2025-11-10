"""3D Spoiled Gradient Echo sequence."""

__all__ = ["design_3D_spgr"]

import numpy as np
import pypulseq as pp

def design_3D_spgr(
    fov: tuple[float],
    npix: tuple[int],
    alpha: float,
    max_grad: float,
    max_slew: float,
    raster_time: float,
):
    """
    Generate a 3D Spoiled Gradient Recalled Echo (SPGR) pulse sequence.

    This function designs a 3D SPGR sequence based on the provided field of view (FOV), matrix size,
    flip angle, and hardware constraints such as maximum gradient amplitude and slew rate. The output
    can be formatted in different sequence file formats if specified.

    Parameters
    ----------
    fov : tuple[float]
        Field of view along each spatial dimension [fov_plane, fov_z] in mm.
        If scalar, assume cubic fov.
    npix : tuple[int]
        Number of voxels along each spatial dimension [plane_mtx, nz] (matrix size).
        If scalar, assume cubic matrix size.
    alpha : float
        Flip angle in degrees.
    max_grad : float
        Maximum gradient amplitude in mT/m.
    max_slew : float
        Maximum gradient slew rate in T/m/s.
    raster_time : float
        Waveform raster time in seconds (the time between successive gradient samples).
    seqformat : str or bool, optional
        Output sequence format. If a string is provided, it specifies the desired output format (e.g., 'pulseq', 'bytes').
        If False, the sequence is returned as an internal object. Default is False.

    Returns
    -------
    seq : object or dict
        The generated SPGR sequence. If `seqformat` is a string, the sequence is returned in the specified format.
        If `seqformat` is False, the sequence is returned as an internal representation.

    Notes
    -----
    - This function is designed to work within the constraints of MRI scanners, taking into account the physical limits
      on gradient amplitude and slew rates.
    - The flip angle (`alpha`) controls the excitation of spins and directly impacts the signal-to-noise ratio (SNR) and contrast.

    Examples
    --------
    Generate a 3D SPGR sequence for a 256x256x128 matrix with a 240x240x120 mm FOV, a 15-degree flip angle, and hardware limits:

    >>> from pulseforge import SPGR3D
    >>> SPGR3D([240, 120], [256, 128], 15, 0.04, 150, 4e-6)

    Generate the same sequence and export it in bytes format:

    >>> SPGR3D([240, 120], [256, 128], 15, 0.04, 150, 4e-6, seqformat='bytes')

    """
    # RF specs
    rf_spoiling_inc = 117.0  # RF spoiling increment

    # Initialize system limits
    system = pp.Opts(
        max_grad=max_grad,
        grad_unit="mT/m",
        max_slew=max_slew,
        slew_unit="T/m/s",
        grad_raster_time=raster_time,
        rf_raster_time=raster_time,
    )

    # Initialize sequence
    seq = pp.Sequence(system=system)

    # Initialize prescription
    # =======================
    # 1) FOV
    if np.isscalar(fov):
        fov, slab_thickness = fov * 1e-3, fov * 1e-3  # isotropic
    else:
        fov, slab_thickness = (
            fov[0] * 1e-3,
            fov[1] * 1e-3,
        )  # in-plane FOV, slab thickness
        
    # 2) Matrix size
    if np.isscalar(npix):
        Nx, Ny, Nz = npix, npix, npix  # in-plane resolution, slice thickness
    else:
        Nx, Ny, Nz = npix[0], npix[0], npix[1]  # in-plane resolution, slice thickness

    # Initialize events
    # =================
    # 1) RF pulse
    rf, gss, _ = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(alpha),
        duration=3e-3,
        slice_thickness=slab_thickness,
        apodization=0.42,
        time_bw_product=4,
        system=system,
        return_gz=True,
    )
    gss_reph = pp.make_trapezoid(
        channel="z", area=-gss.area / 2, duration=1e-3, system=system
    )
    EXC = (
        pp.make_label('TRID', 'SET', 1),
        pp.make_label('COREID', 'SET', 1),
        pp.make_label('BLOCKID', 'SET', 1),
    )
    SLAB_REPH = pp.make_label('BLOCKID', 'SET', 2)
    
    # 2) Readout
    delta_kx, delta_ky, delta_kz = 1 / fov, 1 / fov, 1 / slab_thickness
    gx_read = pp.make_trapezoid(
        channel="x", flat_area=Nx * delta_kx, flat_time=3.2e-3, system=system
    )
    adc = pp.make_adc(
        num_samples=Nx, duration=gx_read.flat_time, delay=gx_read.rise_time, system=system
    )
    ECHO = pp.make_label('BLOCKID', 'SET', 4)

    # 3) Phase encoding
    # X axis
    gx_pre = pp.make_trapezoid(
        channel="x", area=-gx_read.area / 2, duration=1e-3, system=system
    )
    gx_rew = pp.scale_grad(grad=gx_pre, scale=-1.0)
    
    # Y axis
    gy_phase = pp.make_trapezoid(channel="y", area=delta_ky * Ny, system=system)
    
    # Z axis
    gz_phase = pp.make_trapezoid(channel="z", area=delta_kz * Nz, system=system)
   
    PHASE_ENC = pp.make_label('BLOCKID', 'SET', 3)

    # 4) Crusher gradient
    gz_spoil = pp.make_trapezoid(channel="z", area=32 / slab_thickness, system=system)
    CRUSHER = pp.make_label('BLOCKID', 'SET', 5)

    # Compute variable event settings
    # ===============================
    # 1) Phase encoding plan TODO: helper routine
    pey_steps = ((np.arange(Ny)) - (Ny / 2)) / Ny
    pez_steps = ((np.arange(Nz)) - (Nz / 2)) / Nz
    encoding_plan = np.meshgrid(pey_steps, pez_steps, indexing="xy")
    encoding_plan = [enc.ravel() for enc in encoding_plan]
    
    # 2) RF and ADC phases
    # generate rf phases    
    rf_phase = 0
    rf_inc = 0

    # Compute event table length
    # ==========================
    Nscans = Ny * Nz

    # Construct sequence
    # ==================
    for n in range(Nscans):
        
        # Update phase
        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        
        # Add excitation
        seq.add_block(*EXC, rf, gss)
        seq.add_block(SLAB_REPH, gss_reph)
        
        # Add spatial encoding
        gy_pre = pp.scale_grad(grad=gy_phase, scale=encoding_plan[0][n])
        gy_rew = pp.scale_grad(grad=gy_pre, scale=-1.0)
        gz_pre = pp.scale_grad(grad=gz_phase, scale=encoding_plan[1][n])
        gz_rew = pp.scale_grad(grad=gz_pre, scale=-1.0)
        
        seq.add_block(PHASE_ENC, gx_pre, gy_pre, gz_pre)
        seq.add_block(ECHO, gx_read, adc)
        seq.add_block(PHASE_ENC, gx_rew, gy_rew, gz_rew)
        
        # Add crusher
        seq.add_block(CRUSHER, gz_spoil)
        
        # Apply phase increment
        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

    return seq
