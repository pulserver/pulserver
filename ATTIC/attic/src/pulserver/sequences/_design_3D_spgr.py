"""3D Spoiled Gradient Echo sequence."""

__all__ = ["design_3D_spgr"]

import copy

import numpy as np
import pypulseq as pp
import ismrmrd

from .._mrd import ISMRMRDBuilder


def design_3D_spgr(
    fov: tuple[float],
    npix: tuple[int],
    alpha: float,
    dwell_time: float,
    max_grad: float,
    max_slew: float,
    raster_time: float,
    ndummy: int = 1,
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
        Waveform raster time in seconds (the time between successive rf/gradient samples).

    Returns
    -------
    seq : Sequence
        The generated SPGR sequence.

    Notes
    -----
    - This function is designed to work within the constraints of MRI scanners, taking into account the physical limits
      on gradient amplitude and slew rates.
    - The flip angle (`alpha`) controls the excitation of spins and directly impacts the signal-to-noise ratio (SNR) and contrast.

    Examples
    --------
    Generate a 3D SPGR sequence for a 256x256x128 matrix with a 240x240x120 mm FOV,
    and a 15-degree flip angle and maxGrad = 40 ``mT/m``, maxSlew = 150 ``T/m/s,
    rf_raster_time = grad_raster_time = 4 ``us``:

    >>> from pulserver.sequences import design_3D_spgr
    >>> design_3D_spgr([240, 120], [256, 128], 15.0, 40, 150, 4e-6)

    """
    # Initialize system limits
    # ========================
    system = pp.Opts(
        max_grad=max_grad,
        grad_unit="mT/m",
        max_slew=max_slew,
        slew_unit="T/m/s",
        grad_raster_time=raster_time,  # for simplicty, here we set each board to same raster time
        rf_raster_time=raster_time,
        block_duration_raster=raster_time,
        adc_raster_time=raster_time,
    )

    # Initialize sequence
    # ===================
    seq = pp.Sequence(system=system)
    prot = ISMRMRDBuilder()

    # Initialize prescription
    # =======================
    fov, slab_thickness = setup_fov(fov)
    Nx, Ny, Nz = setup_matrix(npix)
    NDummy = ndummy

    # Get RF Spoil increment
    # ======================
    rf_spoiling_inc = 117.0  # RF spoiling increment
    rf_phase = 0
    rf_inc = 0
    phase_offset = design_phaseinc_table(
        Ny + NDummy, Nz, rf_spoiling_inc, rf_phase, rf_inc
    )

    # Initialize events
    # =================
    EXC, rf, gss = design_excitation(system, alpha, slab_thickness, phase_offset)
    SLAB_REPH, gss_reph = design_slabreph(system, gss)
    ECHO, gx_read, adc = design_readout(system, fov, Nx, dwell_time, phase_offset)
    PHASE_ENC, gx_pre, gx_rew, gy_pre, gy_rew, gz_pre, gz_rew = design_phaseenc(
        system, gx_read, fov, slab_thickness, Ny, Nz
    )
    CRUSHER, gz_spoil = design_crusher(system, slab_thickness)

    # Initialize labels
    # =================
    zlabelset = pp.make_label("PAR", "SET", 0)
    zlabelinc = pp.make_label("PAR", "INC", 1)
    ylabelset = pp.make_label("LIN", "SET", 0)
    ylabelinc = pp.make_label("LIN", "INC", 1)
    ylabelinc0 = pp.make_label("LIN", "INC", 0)

    # Register events
    # ===============
    rf, gss = register_excitation(seq, rf, gss)
    gss_reph = register_slabreph(seq, gss_reph)
    gx_read, adc = register_readout(seq, gx_read, adc)
    gx_pre, gx_rew, gy_pre, gy_rew, gz_pre, gz_rew = register_phaseenc(
        seq, gx_pre, gx_rew, gy_pre, gy_rew, gz_pre, gz_rew
    )
    gz_spoil = register_crusher(seq, gz_spoil)

    # Register labels
    # ===============
    zlabelset.id = seq.register_label_event(zlabelset)
    zlabelinc.id = seq.register_label_event(zlabelinc)
    ylabelset.id = seq.register_label_event(ylabelset)
    ylabelinc.id = seq.register_label_event(ylabelinc)
    ylabelinc0.id = seq.register_label_event(ylabelinc0)

    # Set sidecar MRD header
    # ======================
    prot.set_name("3DSPGR")
    prot.set_H1resonanceFrequency_Hz(seq.system.gamma, seq.system.B0)
    prot.set_fov(1e3 * fov, 1e3 * fov, 1e3 * slab_thickness)
    prot.set_matrix(Nx, Ny, Nz)
    prot.set_limits("k0", Nx - 1)
    prot.set_limits("k1", Ny - 1)
    prot.set_limits("k2", Nz - 1)
    prot.set_flipAngle_deg(alpha)

    # Compute TE and TR
    # =================
    TE = prot.set_TE(
        0.5 * 1e3 * pp.calc_duration(gss)
        + 1e3 * pp.calc_duration(gss_reph)
        + 1e3 * pp.calc_duration(gx_pre, gy_pre[0], gz_pre[0])
        + 0.5 * 1e3 * pp.calc_duration(gx_read)
    )
    prot.set_TR(
        TE
        + 0.5 * 1e3 * pp.calc_duration(gx_read)
        + 1e3 * pp.calc_duration(gx_rew, gy_rew[0], gz_rew[0])
        + 1e3 * pp.calc_duration(gz_spoil)
        + 0.5 * 1e3 * pp.calc_duration(gss)
    )

    # Calculate trajectory for readout
    # ================================
    kspace_traj = prot.calc_trajectory((gx_pre,), (gx_read, adc[0]))

    # Construct sequence
    # ==================
    count = 0
    for y in range(-NDummy, Ny):
        for z in range(Nz):
            if y == 0:
                ylabel = ylabelset
            elif z == 0:
                ylabel = ylabelinc
            else:
                ylabel = ylabelinc0
            if z == 0:
                zlabel = zlabelset
            else:
                zlabel = zlabelinc
            if count == (Ny + NDummy) * Nz - 1:
                flag = ismrmrd.ACQ_LAST_IN_MEASUREMENT
            else:
                flag = None

            if y >= 0:
                prot.add_acquisition(kspace_traj, (ylabel, zlabel), flag)

            if count == 0:
                seq.add_block(*EXC, rf[count], gss)
            elif count == 1:
                seq.add_block(EXC[0], rf[count], gss)
            else:
                seq.add_block(rf[count], gss)

            if count == 0:
                seq.add_block(SLAB_REPH, gss_reph)

                # Add prewinder, readout and rewinder
                seq.add_block(PHASE_ENC, gx_pre, gy_pre[y], gz_pre[z])
                if y < 0:
                    seq.add_block(
                        ECHO, gx_read, adc[count], pp.make_label("OFF", "SET", 1)
                    )
                else:
                    seq.add_block(ECHO, gx_read, adc[count], ylabel, zlabel)
                seq.add_block(PHASE_ENC, gx_rew, gy_rew[y], gz_rew[z])

                # Add crusher
                seq.add_block(CRUSHER, gz_spoil)
            else:
                seq.add_block(gss_reph)

                # Add prewinder, readout and rewinder
                seq.add_block(gx_pre, gy_pre[y], gz_pre[z])
                if y < 0:
                    seq.add_block(gx_read, adc[count], pp.make_label("OFF", "SET", 1))
                else:
                    seq.add_block(gx_read, adc[count], ylabel, zlabel)
                seq.add_block(gx_rew, gy_rew[y], gz_rew[z])

                # Add crusher
                seq.add_block(gz_spoil)

            # Update phase increment
            count += 1

    seq.set_definition("NDummies", NDummy * Nz)

    return seq, prot


# %% Helpers
def setup_fov(fov):
    if np.isscalar(fov):
        return fov * 1e-3, fov * 1e-3  # isotropic
    return fov[0] * 1e-3, fov[1] * 1e-3  # in-plane FOV, slab thickness


def setup_matrix(npix):
    if np.isscalar(npix):
        return npix, npix, npix  # in-plane resolution, slice thickness
    return npix[0], npix[0], npix[1]  # in-plane resolution, slice thickness


def design_phaseinc_table(Ny, Nz, rf_spoiling_inc, rf_phase, rf_inc):
    phase_offset = []
    for y in range(Ny):
        for z in range(Nz):
            phase_offset.append(np.deg2rad(rf_phase))
            rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
    return phase_offset


def design_excitation(system, alpha, slab_thickness, phase_offset):
    rf, _rf, gss, _ = [], *pp.make_sinc_pulse(
        flip_angle=np.deg2rad(alpha),
        duration=3e-3,
        slice_thickness=slab_thickness,
        apodization=0.42,
        time_bw_product=4,
        system=system,
        return_gz=True,
        use="excitation",
    )

    # Enable rf spoil
    for count, _phase_offset in enumerate(phase_offset):
        rf.append(copy.deepcopy(_rf))
        rf[-1].phase_offset = _phase_offset

    # Labeling
    EXC = (
        pp.make_label("TRID", "SET", 1),
        pp.make_label("COREID", "SET", 1),
        pp.make_label("BLOCKID", "SET", 1),
    )
    return EXC, rf, gss


def register_excitation(seq, rf, gss):
    for count, _rf in enumerate(rf):
        if count == 0:
            _rf.id, _ = seq.register_rf_event(_rf)
            _rf_data = seq.rf_library.data[1]
        else:
            _rf.id = count + 1
            seq.rf_library.insert(
                count + 1, (*_rf_data[:-1], _rf.phase_offset), _rf.use[0]
            )
    gss.id = seq.register_grad_event(gss)
    return rf, gss


def design_slabreph(system, gss):
    gss_reph = pp.make_trapezoid(
        channel="z", area=-gss.area / 2, duration=1e-3, system=system
    )

    # Labeling
    SLAB_REPH = pp.make_label("BLOCKID", "SET", 2)
    return SLAB_REPH, gss_reph


def register_slabreph(seq, gss_reph):
    gss_reph.id = seq.register_grad_event(gss_reph)
    return gss_reph


def design_readout(system, fov, Nx, dwell_time, phase_offset):
    delta_kx = 1 / fov

    # Design blocks
    gx_read = pp.make_trapezoid(
        channel="x", flat_area=Nx * delta_kx, flat_time=Nx * dwell_time, system=system
    )
    adc, _adc = [], pp.make_adc(
        num_samples=Nx,
        duration=Nx * dwell_time,
        delay=gx_read.rise_time,
        system=system,
    )

    # Enable rf spoil
    for count, _phase_offset in enumerate(phase_offset):
        adc.append(copy.deepcopy(_adc))
        adc[-1].phase_offset = _phase_offset

    # Labeling
    ECHO = pp.make_label("BLOCKID", "SET", 4)
    return ECHO, gx_read, adc


def register_readout(seq, gx_read, adc):
    gx_read.id = seq.register_grad_event(gx_read)
    for count, _adc in enumerate(adc):
        if count == 0:
            _adc.id, _ = seq.register_adc_event(_adc)
            _adc_data = seq.adc_library.data[1]
        else:
            _adc.id = count + 1
            seq.adc_library.insert(
                count + 1, (*_adc_data[:-3], _adc.phase_offset, *_adc_data[-2:])
            )
    return gx_read, adc


def design_phaseenc(system, gx_read, fov, slab_thickness, Ny, Nz):
    delta_ky, delta_kz = 1 / fov, 1 / slab_thickness

    # Prewinders and rewinders
    # ========================
    # X axis
    gx_pre = pp.make_trapezoid(channel="x", area=-gx_read.area / 2, system=system)
    gx_rew = pp.scale_grad(grad=gx_pre, scale=-1.0)

    # Y axis
    gy_phase = pp.make_trapezoid(channel="y", area=delta_ky * Ny, system=system)

    # Z axis
    gz_phase = pp.make_trapezoid(channel="z", area=delta_kz * Nz, system=system)

    # Phase encoding plan
    # ===================
    pey_steps = ((np.arange(Ny)) - (Ny / 2)) / Ny
    pez_steps = ((np.arange(Nz)) - (Nz / 2)) / Nz

    # Y axis
    gy_pre = []
    gy_rew = []
    for y in range(Ny):
        gy_pre.append(pp.scale_grad(grad=gy_phase, scale=pey_steps[y]))
        gy_rew.append(pp.scale_grad(grad=gy_phase, scale=-pey_steps[y]))

    # Z axis
    gz_pre = []
    gz_rew = []
    for z in range(Nz):
        gz_pre.append(pp.scale_grad(grad=gz_phase, scale=pez_steps[z]))
        gz_rew.append(pp.scale_grad(grad=gz_phase, scale=-pez_steps[z]))

    # Labeling
    PHASE_ENC = pp.make_label("BLOCKID", "SET", 3)

    return PHASE_ENC, gx_pre, gx_rew, gy_pre, gy_rew, gz_pre, gz_rew


def register_phaseenc(seq, gx_pre, gx_rew, gy_pre, gy_rew, gz_pre, gz_rew):
    gx_pre.id = seq.register_grad_event(gx_pre)
    gx_rew.id = seq.register_grad_event(gx_rew)

    Ny, Nz = len(gy_pre), len(gz_pre)
    for y in range(Ny):
        gy_pre[y].id = seq.register_grad_event(gy_pre[y])
        gy_rew[y].id = seq.register_grad_event(gy_rew[y])
    for z in range(Nz):
        gz_pre[z].id = seq.register_grad_event(gz_pre[z])
        gz_rew[z].id = seq.register_grad_event(gz_rew[z])

    return gx_pre, gx_rew, gy_pre, gy_rew, gz_pre, gz_rew


def design_crusher(system, slab_thickness):
    gz_spoil = pp.make_trapezoid(channel="z", area=32 / slab_thickness, system=system)
    CRUSHER = pp.make_label("BLOCKID", "SET", 5)
    return CRUSHER, gz_spoil


def register_crusher(seq, gz_spoil):
    gz_spoil.id = seq.register_grad_event(gz_spoil)
    return gz_spoil
