"""3D MPRAGE on a golden-angle stack of spirals.

``main`` returns the :class:`pulserver.pypulseq.Sequence`; ``PLUGIN`` is the
same sequence behind the scanner protocol contract, and running this module
as a script writes a ``.seq`` from the same controls.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pulserver.design as design
import pulserver.pypulseq as pp
from pulserver.design import (
    Description,
    DropdownFloatParam,
    DropdownIntParam,
    OffFloatParam,
    SequencePlugin,
    TRPreset,
    TypeinFloatParam,
    UIParam,
    dict_to_protocol,
    main_kwargs,
    params,
    protocol_to_dict,
    run_cli,
    write_sequence,
)
from pulserver.app.sequence.gre_stack_of_stars3D_sequence import (
    StackShotKernel,
    stack_angles,
)
from scipy.spatial.transform import Rotation


#: SLR design of the selective pulses, held here rather than left at the design
#: module's default so a script can retune the excitation without touching the
#: loop. The selection amplitude follows as
#: ``time_bw_product / (duration * thickness)``, which is also what a slice
#: offset is converted against.
PULSE_DURATION = 3e-3
TIME_BW_PRODUCT = 4.0

#: Per-plugin ceilings on the gradient and slew limits, in mT/m and T/m/s. The
#: sequence is held below the smaller of these and what the scanner reports, so
#: lowering them here -- on the scanner console, even -- reruns the whole script
#: under gentler gradients (for PNS headroom, acoustic comfort, eddy currents)
#: without touching anything else. Defaults sit above typical hardware, so they
#: cap nothing until you lower them.
MAX_GRAD = 80.0
MAX_SLEW = 200.0

#: Swap the slab excitation for a spectral-spatial, slab- *and* water-selective
#: pulse. A long readout lets fat -- a few ppm off water -- shift and blur; the
#: non-Cartesian trajectories here spread that shift into a swirl rather than a
#: clean displacement, so exciting water only removes the source. A script-level
#: toggle rather than a UI control: it reshapes the excitation the whole
#: sequence is timed around, so it belongs to whoever runs the script.
SPSP_EXCITATION = False

#: Chemical shift of the fat methylene resonance from water, in ppm. Held in
#: ppm rather than hertz so it is field-strength independent: the water-only
#: excitation converts it against ``system.B0`` when the pulse is built, so the
#: same script targets fat at 1.5 T and 3 T alike.
FAT_SHIFT_PPM = -3.4


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "mprage_stack_of_spirals_3d.seq",
    *,
    system: pp.Opts | None = None,
    fov: float = 220e-3,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int = 128,
    n_z: int = 64,
    slab_thickness: float = 128e-3,
    n_arms: int = 16,
    design_interleaves: int | None = None,
    flip_angle_deg: float = 9.0,
    ti: float = 900e-3,
    tr_outer: float = 2000e-3,
    te: float | None = None,
    readout_bandwidth_hz: float = 250e3,
    n_dummy: int = 1,
    n_gain_calibration_readouts: int = 1,
    rf_spoiling_increment_deg: float = 117.0,
    spoiling_cycles: float = 4.0,
    etl: int | None = None,
    angular_increment_deg: float | None = None,
    partition_angle_offset_deg: float = 0.0,
    use_rotation_ext: bool = True,
    spsp: bool = False,
) -> pp.Sequence:
    """Create a golden-angle stack-of-spirals 3D MPRAGE.

    Partitions are the outer loop: one inversion per train, and a train plays
    ``etl`` arms of one partition with its first echo at the requested TI. Arms
    step by the golden angle, so any prefix of the scan covers the disc
    near-uniformly, and a given arm is played at the same angle in every partition.
    A partition angle offset turns it a little further with each one instead,
    staggering the sampling along kz the way a CAIPIRINHA shift does.

    By default the arm is held once and turned per shot by a ``ROTATIONS``
    extension. ``use_rotation_ext=False`` writes every shot out as its own
    gradient waveform instead, with the shape library growing with the shot count
    -- the untemplatable-waveform case, which is exactly the path an interpreter's
    waveform streaming has to survive.
    :mod:`pulserver.app.noncartesian_stack_recon` reconstructs the stack
    against the trajectory the acquisitions carry.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is
        'mprage_stack_of_spirals_3d.seq'.
    system : pypulseq.Opts, optional
        System limits. Default is `pp.Opts()`.
    fov : float, optional
        Isotropic in-plane field of view in meters. Default is 220e-3.
    fov_offset : tuple of float, optional
        Where the prescribed volume sits, in meters along the logical
        readout, phase and slab axes. Applied in server mode. Default is
        (0.0, 0.0, 0.0).
    n_x : int, optional
        In-plane matrix size. Default is 128.
    n_z : int, optional
        Number of partitions. Default is 64.
    slab_thickness : float, optional
        Excited slab thickness in meters, also the field of view along z.
        Default is 128e-3.
    n_arms : int, optional
        Arms stepping by ``angular_increment_deg`` around the disc. Default
        is 16.
    design_interleaves : int or None, optional
        The pitch the spiral is designed for. None matches ``n_arms``.
        Default is None.
    flip_angle_deg : float, optional
        Readout excitation flip angle in degrees. Default is 9.0.
    ti : float, optional
        Inversion time in seconds, inversion centre to the echo of the
        train's first view. Default is 900e-3.
    tr_outer : float, optional
        Inversion-to-inversion interval in seconds. Default is 2000e-3.
    te : float or None, optional
        Readout echo time in seconds. None is as short as possible. Default
        is None.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth in Hz. Default is 250e3.
    n_gain_calibration_readouts : int, optional
        Written as the ``NumGainCalibrationReadouts`` definition. Default
        is 1.
    rf_spoiling_increment_deg : float, optional
        Quadratic RF spoiling phase increment in degrees. Default is 117.0.
    spoiling_cycles : float, optional
        Cycles of dephasing left at the end of each inner repetition.
        Default is 4.0.
    etl : int or None, optional
        Arms per inversion train. It has to divide the arm count evenly, so
        that every train samples its arms at the same points of the recovery;
        a shorter train costs more inversions and buys a narrower spread of
        T1 weighting within one image. None is one train per partition,
        covering every arm. Default is None.
    angular_increment_deg : float or None, optional
        Angle between successive arms, in degrees. None is the golden angle,
        which leaves any prefix of the arms close to uniformly distributed.
        Default is None.
    partition_angle_offset_deg : float, optional
        Angle added per partition step, in degrees. Zero plays a given arm at
        the same angle in every partition -- one train's arms are the same
        set of angles wherever it sits in the slab; anything else turns them
        a little further with each partition, staggering the sampling along
        kz the way a CAIPIRINHA shift does. Default is 0.0.
    use_rotation_ext : bool, optional
        Hold one arm and turn it per shot with a ``ROTATIONS`` extension (the
        default), or write every shot out as its own waveform. The second
        costs a waveform per shot -- one per arm and partition both, under a
        partition offset -- and reads without composing a rotation. Default
        is True.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The MPRAGE sequence object.

    Examples
    --------
    >>> from pulserver.app import mprage_stack_of_spirals3D_sequence
    >>> seq = mprage_stack_of_spirals3D_sequence(n_x=32, n_z=8, ti=100e-3, tr_outer=300e-3)
    >>> seq.num_trs, seq.num_segments
    (9, 3)

    The waveform figures below are one design, prescribed to be *legible*
    rather than diagnostic: the shortest TI, TE and outer TR the inversion and
    readout admit, so nothing waits longer than the physics requires; heavy
    spoiling, so that lobe is unmistakable; and eight arms over eight
    partitions, so a whole shot fits on a page.

    .. plot::
       :include-source:
       :nofigs:
       :context:

       from pulserver.app import mprage_stack_of_spirals3D_sequence

       seq = mprage_stack_of_spirals3D_sequence(
           n_x=128, n_z=8, n_arms=8, ti=55e-3, tr_outer=175e-3, te=None,
           n_dummy=0, spoiling_cycles=6.0,
       )

    **The pulses.** The inversion that opens the shot, then the excitation
    each view is read from -- one drawn against what it leaves along ``z``,
    the other against what it tips into the transverse plane.

    .. plot::
       :include-source:
       :context: close-figs

       seq.plot_rf("inversion", title="inversion", plot_now=False)
       seq.plot_rf("excitation", title="excitation", plot_now=False)

    **One shot**: the inversion, the wait that places the TI, the segment of
    spiral readouts taken under the recovering magnetisation, and the
    recovery delay that closes the outer TR.

    .. plot::
       :include-source:
       :context: close-figs

       seq.plot(tr="worst_case", time_disp="ms", grad_disp="mT/m", plot_now=False)

    **The segments**, which are the interpreter's units of playout. Each is
    drawn as the instance carrying the most gradient energy -- the one the
    safety checks were run against -- over the span of the scan where it
    plays.

    .. plot::
       :include-source:
       :context: close-figs

       for index in range(seq.num_segments):
           seq.plot(
               segment_idx=index, time_disp="ms", grad_disp="mT/m", plot_now=False
           )

    **What the scan covers**: the in-plane spiral repeated at every
    partition, coloured by the order the views were acquired in. Under one
    inversion that order *is* the inversion weighting, so this is also the
    contrast map.

    .. plot::
       :include-source:
       :context: close-figs

       seq.plot_kspace(color_by="order", plot_now=False)

    **Mechanical resonance.** The repetition is periodic, so its gradients
    have energy only at multiples of ``1 / T_TR``. Those lines are what a
    forbidden band is judged against, and the verdict panel is the whole of
    the acoustic check: every line that falls inside a band, against the
    amplitude that band allows.

    .. plot::
       :include-source:
       :context: close-figs

       gamma = 42.576e3  # Hz/m per mT/m
       seq.calculate_gradient_spectrum(
           tr="worst_case",
           resonance_lines=True,
           bands=[(550.0, 700.0, 3.0 * gamma), (1150.0, 1300.0, 3.0 * gamma)],
       )

    **Peripheral nerve stimulation**, under the rheobase/chronaxie model the
    scanner's own gate applies, over the same repetition played back to back.
    This design asks for the shortest timing the hardware admits, so its spiral
    runs close to the slew limit throughout. A lower ``MAX_SLEW``, a longer
    echo time, or a narrower readout bandwidth each bring the response down.

    .. plot::
       :include-source:
       :context: close-figs

       seq.calculate_pns(
           {"chronaxie_us": 360.0, "rheobase": 20.0, "alpha": 0.333},
           tr="worst_case",
       )
    """
    system = pp.Opts() if system is None else system
    system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)

    kernel = MprageStackOfSpiralsKernel(
        system,
        fov=fov,
        n_x=n_x,
        n_z=n_z,
        slab_thickness=slab_thickness,
        n_arms=n_arms,
        design_interleaves=design_interleaves,
        flip_angle_deg=flip_angle_deg,
        ti=ti,
        tr_outer=tr_outer,
        te=te,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=spoiling_cycles,
        etl=etl,
        angular_increment_deg=angular_increment_deg,
        partition_angle_offset_deg=partition_angle_offset_deg,
        use_rotation_ext=use_rotation_ext,
        spsp=spsp,
    )
    readout = kernel.readout
    inversion = kernel.inversion
    etl = kernel.etl
    shot_index = kernel.shot_index

    rf_phases = pp.make_rf_spoiling_schedule(
        n_arms * n_z + n_dummy * etl,
        increment=np.deg2rad(rf_spoiling_increment_deg),
    )
    rotations = (
        [
            pp.make_rotation(Rotation.from_euler("z", float(angle)))
            for angle in kernel.shot_angles
        ]
        if use_rotation_ext
        else [None] * len(kernel.shot_angles)
    )

    seq = pp.Sequence(system)
    spoiling_phase = iter(rf_phases)
    lin_label, par_label, seg_label = readout.adc_labels

    def repetition(arm: int, partition: int, *, acquire: bool = True) -> None:
        """One inner repetition, at this arm's orientation and partition."""
        shot = shot_index(arm, partition)
        rf_phase = next(spoiling_phase)
        readout.rf.phase_offset = rf_phase
        readout.adc.phase_offset = rf_phase
        lin_label.value = int(arm)
        par_label.value = int(partition)
        StackShotKernel(
            seq,
            readout,
            readout.arm(shot),
            rotations[shot],
            (partition - n_z / 2) / (n_z / 2),
            acquire=acquire,
        )

    def inversion_train(arms, partition, *, acquire: bool, mark=None) -> None:
        """One outer TR: the inversion, the wait, the arms, the recovery."""
        seq.add_block(inversion.rf_prep, *([mark] if mark is not None else []))
        seq.add_block(inversion.gz_spoil)
        seq.add_block(kernel.wait_ti)
        for arm in arms:
            repetition(arm, partition, acquire=acquire)
        seq.add_block(kernel.wait_recovery)

    # Steady state first: whole inversion trains without acquiring, so the
    # magnetisation the first acquired train measures is the one every later
    # train measures. The outer TR is the unit because that is what the
    # longitudinal magnetisation recovers over.
    #
    # `ONCE` is what keeps them out of the averages: 1 plays on the first pass
    # only, and the first acquired train clears it back to 0 so the body
    # repeats.
    for i_dummy in range(n_dummy):
        inversion_train(
            range(etl),
            0,
            acquire=False,
            mark=pp.make_label("ONCE", "SET", 1) if i_dummy == 0 else None,
        )

    clear_once = pp.make_label("ONCE", "SET", 0) if n_dummy else None
    train = 0
    for partition in range(n_z):
        for start in range(0, n_arms, etl):
            seg_label.value = train
            train += 1
            inversion_train(
                range(start, start + etl), partition, acquire=True, mark=clear_once
            )
            clear_once = None

    pp.TransformFOV(
        translation=tuple(offset * 1e3 for offset in fov_offset),
        system=system,
        compat=False,
    ).apply_to_sequence(seq, in_place=True)

    if test_report:
        print(seq.test_report())

    if plot:
        seq.plot()

    seq.set_definition(key="FOV", value=[fov, fov, slab_thickness])
    seq.set_definition(key="Matrix", value=[n_x, n_x, n_z])
    seq.set_definition(key="Name", value="mprage_stack_of_spirals_3d")
    seq.set_definition(key="TE", value=kernel.echo_time)
    seq.set_definition(key="TR", value=tr_outer)
    seq.set_definition(key="TI", value=ti)
    seq.set_definition(key="InnerTR", value=kernel.inner_tr)
    seq.set_definition(key="Trajectory", value="stack_of_spirals")
    seq.set_definition(key="NumArms", value=n_arms)
    seq.set_definition(
        key="AngleScheme",
        value="golden" if angular_increment_deg is None else "uniform",
    )
    seq.set_definition(
        key="AngularIncrement", value=np.rad2deg(kernel.angular_increment)
    )
    seq.set_definition(key="PartitionAngleOffset", value=partition_angle_offset_deg)
    seq.set_definition(key="EchoTrainLength", value=etl)
    seq.set_definition(key="ExplicitArms", value=0 if use_rotation_ext else 1)
    seq.set_definition(key="kSpaceCenterPartition", value=n_z // 2)
    seq.set_definition(key="kSpaceCenterSample", value=readout.center_sample)
    seq.set_definition(key="SliceThickness", value=slab_thickness)
    seq.set_definition(
        key="NumGainCalibrationReadouts", value=n_gain_calibration_readouts
    )

    if write_seq:
        write_sequence(seq, seq_filename, offline=True)

    return seq


# ======================================================================
# Subroutines of main()
# ======================================================================


def SlabExcitationKernel(
    system: pp.Opts,
    flip_angle_deg: float,
    thickness_m: float,
    spsp: bool = False,
):
    """The slab excitation, spectral-spatial when ``spsp`` is set.

    Returns ``(excitation, rf, gz)``. The selection gradient carries its own
    rephaser folded onto the end -- as a slab excitation does -- so the stack's
    partition prewinder block, which already encodes z, never holds a second
    z gradient.
    """
    if spsp:
        fat_offset_hz = FAT_SHIFT_PPM * 1e-6 * system.gamma * system.B0
        excitation = design.SpspExcitation(
            system,
            flip_angle_deg,
            thickness_m=thickness_m,
            spectral_bandwidth_hz=abs(fat_offset_hz),
            freq_offset_hz=0.0,
            is_slab=True,
        )
        return excitation, excitation.rf, excitation.gz
    excitation = design.SpatialSelectiveExcitation(
        system,
        flip_angle_deg,
        thickness_m,
        duration_s=PULSE_DURATION,
        is_slab=True,
        time_bw_product=TIME_BW_PRODUCT,
    )
    return excitation, excitation.rf, excitation.gz


def MprageStackOfSpiralsKernel(
    system: pp.Opts,
    *,
    fov: float = 220e-3,
    n_x: int = 128,
    n_z: int = 64,
    slab_thickness: float = 128e-3,
    n_arms: int = 16,
    design_interleaves: int | None = None,
    flip_angle_deg: float = 9.0,
    ti: float = 900e-3,
    tr_outer: float = 2000e-3,
    te: float | None = None,
    readout_bandwidth_hz: float = 250e3,
    spoiling_cycles: float = 4.0,
    etl: int | None = None,
    angular_increment_deg: float | None = None,
    partition_angle_offset_deg: float = 0.0,
    use_rotation_ext: bool = True,
    spsp: bool = False,
) -> SimpleNamespace:
    """Design the arms and the segment timing around them.

    TI runs from the inversion pulse's centre to the echo of the train's first
    view. A TI or an outer TR too short for the segment raises ``ValueError``.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov, n_x, n_z, slab_thickness, n_arms, design_interleaves, \
flip_angle_deg, ti, tr_outer, te, readout_bandwidth_hz, spoiling_cycles, \
etl, angular_increment_deg, partition_angle_offset_deg, use_rotation_ext
        As for :func:`main`.

    Returns
    -------
    types.SimpleNamespace
        ``inversion``, ``excitation``, ``readout``, ``angles``,
        ``shot_angles``, ``shot_index``, ``angular_increment``, ``etl``,
        ``wait_ti``, ``wait_recovery``, ``echo_time``, ``inner_tr``,
        ``bandwidth_hz`` and ``duration``.
    """
    inversion = design.InversionPreparation(
        system, voxel_size_m=min(fov / n_x, slab_thickness / n_z)
    )
    excitation, exc_rf, exc_gz = SlabExcitationKernel(
        system, flip_angle_deg, slab_thickness, spsp
    )
    if etl is None:
        etl = n_arms
    etl = int(etl)
    if etl < 1 or n_arms % etl:
        raise ValueError(
            f"etl must divide the {n_arms} arms evenly, and {etl} does not; a "
            "short final train would sample its arms at a different point of "
            "the recovery than the others"
        )

    if angular_increment_deg is None:
        angles = np.asarray(pp.calc_golden_angles(n_arms, full_circle=True))
        angular_increment = float(
            np.diff(pp.calc_golden_angles(2, full_circle=True))[0]
        )
    else:
        angular_increment = np.deg2rad(float(angular_increment_deg))
        angles = np.mod(np.arange(n_arms) * angular_increment, 2 * np.pi)
    shot_angles, shot_index = stack_angles(angles, n_z, partition_angle_offset_deg)

    readout = design.SpiralStackReadout(
        system,
        exc_rf,
        exc_gz,
        None,
        fov=fov,
        matrix=n_x,
        fov_z=slab_thickness,
        matrix_z=n_z,
        design_interleaves=n_arms if design_interleaves is None else design_interleaves,
        explicit=not use_rotation_ext,
        angles=None if use_rotation_ext else shot_angles,
        te=te,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=spoiling_cycles,
        # LIN carries the shot -- a spiral plane has no phase-encode line --
        # PAR the partition, and SEG which inversion train it recovered from.
        labels=("LIN", "PAR", "SEG"),
    )
    # One repetition spans four blocks; every arm lays the same timing, so the
    # first arm's span is the inner TR.
    inner_tr = sum(pp.calc_duration(*block) for block in readout.arm(0))

    inversion_centre = inversion.rf_prep.delay + inversion.rf_prep.center
    inversion_tail = inversion.seq.duration()[0] - inversion_centre
    excitation_centre = readout.rf.delay + readout.rf.center
    ti_floor = inversion_tail + excitation_centre + readout.echo_time
    wait = ti - ti_floor
    if wait < 0:
        raise ValueError(
            f"TI {ti * 1e3:.0f} ms is shorter than the inversion tail and the "
            f"first view's echo time admit; the minimum is "
            f"{ti_floor * 1e3:.0f} ms"
        )
    wait_ti = pp.make_delay(
        max(
            system.block_duration_raster,
            pp.round_to_raster(wait, system.block_duration_raster),
        )
    )

    segment_body = inversion.seq.duration()[0] + wait_ti.delay + etl * inner_tr
    recovery = tr_outer - segment_body
    if recovery < 0:
        raise ValueError(
            f"TR {tr_outer * 1e3:.0f} ms is shorter than one segment takes "
            f"({segment_body * 1e3:.0f} ms)"
        )
    wait_recovery = pp.make_delay(
        max(
            system.block_duration_raster,
            pp.round_to_raster(recovery, system.block_duration_raster),
        )
    )

    duration = n_z * (n_arms // etl) * (segment_body + wait_recovery.delay)

    return SimpleNamespace(
        inversion=inversion,
        excitation=excitation,
        readout=readout,
        angles=angles,
        shot_angles=shot_angles,
        shot_index=shot_index,
        angular_increment=angular_increment,
        etl=etl,
        wait_ti=wait_ti,
        wait_recovery=wait_recovery,
        echo_time=readout.echo_time,
        inner_tr=inner_tr,
        bandwidth_hz=readout.bandwidth_hz,
        duration=duration,
    )


# ======================================================================
# The scanner protocol contract
# ======================================================================


class MprageStackOfSpirals3D(SequencePlugin):
    """The stack-of-spirals MPRAGE behind the scanner protocol contract."""

    def get_default_protocol(self, system: pp.Opts) -> dict[str, dict]:
        """Return the protocol the scanner UI is built from."""
        return protocol_to_dict(
            {
                UIParam.PREP_TIME: TypeinFloatParam(
                    value=900.0, min=50.0, max=3000.0, incr=10.0, unit="ms"
                ),
                UIParam.TR: DropdownFloatParam(
                    value=2000.0,
                    min=300.0,
                    max=8000.0,
                    incr=10.0,
                    unit="ms",
                    options=[TRPreset.MINIMUM, 1500.0, 2000.0, 2500.0, 3000.0],
                ),
                UIParam.FLIP: DropdownFloatParam(
                    value=9.0,
                    min=2.0,
                    max=20.0,
                    incr=1.0,
                    unit="deg",
                    options=[7.0, 9.0, 12.0, 15.0, 18.0],
                ),
                UIParam.FOV: DropdownFloatParam(
                    value=220.0,
                    min=80.0,
                    max=500.0,
                    incr=1.0,
                    unit="mm",
                    options=[180.0, 220.0, 280.0, 340.0, 500.0],
                ),
                UIParam.SLICE_THICKNESS: DropdownFloatParam(
                    value=2.0,
                    min=0.5,
                    max=10.0,
                    incr=0.1,
                    unit="mm",
                    options=[1.0, 1.5, 2.0, 3.0, 5.0],
                ),
                UIParam.NX: DropdownIntParam(
                    value=128, min=16, max=512, incr=1, options=[64, 128, 192, 256, 384]
                ),
                UIParam.NSLICES: DropdownIntParam(
                    value=64, min=8, max=256, incr=1, options=[32, 64, 96, 128, 192]
                ),
                UIParam.BANDWIDTH: TypeinFloatParam(
                    value=250e3, min=5e3, max=500e3, incr=100.0, unit="Hz"
                ),
                UIParam.FOV_OFFSET_X: OffFloatParam(
                    value=0.0, min=-500.0, max=500.0, unit="mm"
                ),
                UIParam.FOV_OFFSET_Y: OffFloatParam(
                    value=0.0, min=-500.0, max=500.0, unit="mm"
                ),
                UIParam.FOV_OFFSET_Z: OffFloatParam(
                    value=0.0, min=-500.0, max=500.0, unit="mm"
                ),
                UIParam.user_name(0): Description(text="Arms (one per inversion)"),
                UIParam.user_value(0): TypeinFloatParam(
                    value=16.0, min=1.0, max=512.0, incr=1.0, unit=""
                ),
            }
        )

    def validate_protocol(self, system: pp.Opts, protocol: dict[str, dict]) -> dict:
        """Report whether the protocol is feasible, and how long it will take."""
        system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
        kwargs = protocol_kwargs(system, protocol)
        try:
            kernel = MprageStackOfSpiralsKernel(
                system,
                **{
                    name: value
                    for name, value in kwargs.items()
                    if name in KERNEL_ARGUMENTS
                },
            )
        except ValueError as error:
            return {"valid": False, "duration": None, "info": str(error)}

        return {
            "valid": True,
            "duration": kernel.duration,
            "info": (
                f"TA = {kernel.duration:.1f} s over {len(kernel.angles)} "
                f"arms, inner TR = {kernel.inner_tr * 1e3:.2f} ms"
            ),
        }

    def make_sequence(
        self,
        system: pp.Opts,
        protocol: dict[str, dict],
        output_path: str,
        *,
        offline: bool = False,
    ) -> None:
        """Build the sequence and write it to ``output_path``."""
        seq = main(**protocol_kwargs(system, protocol))
        write_sequence(seq, output_path, offline=offline)


KERNEL_ARGUMENTS = frozenset(
    (
        "fov",
        "spsp",
        "n_x",
        "n_z",
        "slab_thickness",
        "n_arms",
        "design_interleaves",
        "flip_angle_deg",
        "ti",
        "tr_outer",
        "te",
        "readout_bandwidth_hz",
        "spoiling_cycles",
    )
)


def protocol_kwargs(system: pp.Opts, protocol: dict[str, dict]) -> dict:
    """The prescribed quantities, plus this sequence's own user slots."""
    prot = dict_to_protocol(protocol)
    n_z = params.param_int(prot, UIParam.NSLICES)
    partition_thickness = params.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
    return main_kwargs(
        main,
        system,
        protocol,
        spsp=SPSP_EXCITATION,
        fov=params.param_float(prot, UIParam.FOV) * 1e-3,
        n_z=n_z,
        slab_thickness=n_z * partition_thickness,
        ti=params.param_float(prot, UIParam.PREP_TIME) * 1e-3,
        tr_outer=params.param_float(prot, UIParam.TR) * 1e-3,
        n_arms=max(1, round(params.user_float(prot, 0, 16.0))),
    )


PLUGIN = MprageStackOfSpirals3D()


def get_default_protocol(system):
    """Bridge entry point: the plugin's default protocol."""
    return PLUGIN.get_default_protocol(system)


def validate_protocol(system, protocol):
    """Bridge entry point: protocol feasibility and scan duration."""
    return PLUGIN.validate_protocol(system, protocol)


def make_sequence(system, protocol, output_path):
    """Bridge entry point: write the ``.seq`` file."""
    return PLUGIN.make_sequence(system, protocol, output_path)


ARG_MAP = [
    ("--ti-ms", UIParam.PREP_TIME, float, "Inversion time [ms]"),
    ("--tr-ms", UIParam.TR, float, "Inversion-to-inversion interval [ms]"),
    ("--flip-deg", UIParam.FLIP, float, "Readout flip angle [deg]"),
    ("--fov-mm", UIParam.FOV, float, "In-plane FOV [mm]"),
    (
        "--partition-thickness-mm",
        UIParam.SLICE_THICKNESS,
        float,
        "Partition thickness [mm]; the slab is this times the partition count",
    ),
    ("--nx", UIParam.NX, int, "In-plane matrix size"),
    ("--nz", UIParam.NSLICES, int, "Partition count"),
    ("--bandwidth-hz", UIParam.BANDWIDTH, float, "Requested receiver bandwidth [Hz]"),
    ("--offset-x-mm", UIParam.FOV_OFFSET_X, float, "Volume offset along readout [mm]"),
    (
        "--offset-y-mm",
        UIParam.FOV_OFFSET_Y,
        float,
        "Volume offset along phase encode [mm]",
    ),
    ("--offset-z-mm", UIParam.FOV_OFFSET_Z, float, "Volume offset along slab [mm]"),
    ("--arms", UIParam.user_value(0), float, "Golden-angle arms, one per inversion"),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=ARG_MAP,
            description="Generate a stack-of-spirals MPRAGE .seq offline.",
            default_output="mprage_stack_of_spirals_3d.seq",
        )
    )
