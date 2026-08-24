"""Balanced SSFP 3D Cartesian, slab-selective.

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

#: SLR design of the selective pulses, held here rather than left at the design
#: module's default so a script can retune the excitation without touching the
#: loop. The selection amplitude follows as
#: ``time_bw_product / (duration * thickness)``, which is also what a slice
#: offset is converted against.
PULSE_DURATION = 0.6e-3
TIME_BW_PRODUCT = 4.0

#: Per-plugin ceilings on the gradient and slew limits, in mT/m and T/m/s. The
#: sequence is held below the smaller of these and what the scanner reports, so
#: lowering them here -- on the scanner console, even -- reruns the whole script
#: under gentler gradients (for PNS headroom, acoustic comfort, eddy currents)
#: without touching anything else. Defaults sit above typical hardware, so they
#: cap nothing until you lower them.
MAX_GRAD = 80.0
MAX_SLEW = 200.0


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "bssfp_3d.seq",
    *,
    system: pp.Opts | None = None,
    fov: float | tuple[float, float] = 220e-3,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int = 128,
    n_y: int = 128,
    n_z: int = 64,
    slab_thickness: float = 128e-3,
    flip_angle_deg: float = 45.0,
    tr: float | None = None,
    readout_bandwidth_hz: float = 125e3,
    partial_fourier: float = 1.0,
    partial_fourier_z: float = 1.0,
    acceleration: int = 1,
    acceleration_z: int = 1,
    caipi_shift: int = 0,
    elliptical: bool = True,
    n_acs: int = 24,
    n_acs_z: int = 16,
    n_dummy: int = 10,
    n_gain_calibration_readouts: int = 1,
) -> pp.Sequence:
    """Create a balanced SSFP 3D Cartesian sequence.

    One steady-state train over the whole ``(ky, kz)`` traversal --
    :class:`design.BssfpReadout3D`, whose partition encode shares the slab
    rephasers' windows and is added onto them, so a partition costs an amplitude
    rather than a waveform. Entry through the half-flip catalyst, alternating
    phase, TE at exactly TR/2, the autocalibration rectangle leading the
    traversal as every 3D Cartesian module orders it. Regular undersampling lays a
    CAIPIRINHA lattice with a selectable kz shift per ky block.
    :mod:`pulserver.app.cartesian3D_recon` reads the result back.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'bssfp_3d.seq'.
    system : pypulseq.Opts, optional
        System limits. Default is `pp.Opts()`.
    fov : float or tuple of float, optional
        In-plane field of view in meters, (fov_x, fov_y) if a tuple. Default
        is 220e-3.
    fov_offset : tuple of float, optional
        Where the prescribed volume sits, in meters along the logical
        readout, phase and slab axes. Default is (0.0, 0.0, 0.0).
    n_x : int, optional
        Number of readout samples. Default is 128.
    n_y : int, optional
        Number of phase encoding steps. Default is 128.
    n_z : int, optional
        Number of partition encoding steps. Default is 64.
    slab_thickness : float, optional
        Excited slab thickness in meters, also the field of view along z.
        Default is 128e-3.
    flip_angle_deg : float, optional
        Excitation flip angle in degrees. Default is 45.0.
    tr : float or None, optional
        Repetition time in seconds. The echo time is always TR/2. None is as
        short as possible. Default is None.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth in Hz. A balanced train wants a window
        long enough beside the pulse for TE to sit at TR/2, hence the lower
        default. Default is 125e3.
    partial_fourier : float, optional
        Fraction of the phase-encode extent acquired along y, in (0.5, 1].
        Default is 1.0.
    partial_fourier_z : float, optional
        Fraction of the partition-encode extent acquired along z, in (0.5, 1].
        Truncates the partitions before the centre. Default is 1.0.
    acceleration : int, optional
        Uniform phase-encode undersampling factor along y. Default is 1.
    acceleration_z : int, optional
        Uniform partition-encode undersampling factor along z. Default is 1.
    caipi_shift : int, optional
        CAIPIRINHA shift along kz per sampled-ky block,
        0 <= caipi_shift < acceleration_z. 0 is a regular lattice. Default
        is 0.
    elliptical : bool, optional
        Restrict the phase-encode support to the inscribed ky-kz ellipse,
        dropping the corners a round object never fills. Default is True.
    n_acs : int, optional
        Autocalibration extent along y, in lines. Default is 24.
    n_acs_z : int, optional
        Autocalibration extent along z, in partitions. Default is 16.
    n_dummy : int, optional
        Repetitions played without acquiring after the catalyst pulse.
        Default is 10.
    n_gain_calibration_readouts : int, optional
        Written as the ``NumGainCalibrationReadouts`` definition. Default
        is 1.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The bSSFP sequence object.

    Examples
    --------
    >>> from pulserver.app import bssfp3D_sequence
    >>> seq = bssfp3D_sequence(n_x=64, n_y=16, n_z=4, readout_bandwidth_hz=100e3, n_dummy=0)
    >>> seq.num_trs, seq.num_segments
    (1, 2)

    The waveform figures below are one design, prescribed to be *legible*
    rather than diagnostic: the shortest TR the readout admits; a long readout,
    so its flat top dominates the repetition; and a small phase-encode and
    partition grid, so the whole traversal fits on a page.

    .. plot::
       :include-source:
       :nofigs:
       :context:

       from pulserver.app import bssfp3D_sequence

       seq = bssfp3D_sequence(
           n_x=256, n_y=8, n_z=2, tr=None, n_acs=0, n_acs_z=0, n_dummy=0,
       )

    **The excitation**, and the magnetisation it leaves behind: the pulse's
    own ``B1`` envelope beside the profile it writes across the selected
    axis.

    .. plot::
       :include-source:
       :context: close-figs

       seq.plot_rf(plot_now=False)

    **One canonical TR.** The traversal comes out as a single canonical TR
    here, so the figure is zoomed to its first 15 ms -- three repetitions,
    each returning every gradient axis to zero before the next excitation,
    which is what balanced means.

    .. plot::
       :include-source:
       :context: close-figs

       seq.plot(
           tr="worst_case",
           time_range=(0.0, 0.015),
           time_disp="ms",
           grad_disp="mT/m",
           plot_now=False,
       )

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

    **What the scan covers**, as a phase-encode against partition grid.

    .. plot::
       :include-source:
       :context: close-figs

       seq.plot_kspace(plane="yz", color_by="order", plot_now=False)

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
    This design asks for the shortest timing the hardware admits, so its
    prewinders and rewinders ramp as fast as they are allowed to. A lower
    ``MAX_SLEW``, a longer echo time, or a narrower readout bandwidth each
    bring the response down.

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

    kernel = Bssfp3DKernel(
        system,
        fov=fov,
        n_x=n_x,
        n_y=n_y,
        n_z=n_z,
        slab_thickness=slab_thickness,
        flip_angle_deg=flip_angle_deg,
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        partial_fourier=partial_fourier,
        partial_fourier_z=partial_fourier_z,
        acceleration=acceleration,
        acceleration_z=acceleration_z,
        caipi_shift=caipi_shift,
        elliptical=elliptical,
        n_acs=n_acs,
        n_acs_z=n_acs_z,
        n_dummy=n_dummy,
    )
    bssfp = kernel.readout
    fov_x, fov_y = kernel.fov
    pairs = kernel.pairs
    last_calibration_pair = kernel.n_calibration - 1

    seq = pp.Sequence(system)
    lin_label, par_label, ima_label, seg_label = bssfp.adc_labels
    nominal_amplitude = bssfp.rf.amplitude

    def rephaser(base, rewind, kz: float):
        """A slab rephaser with the partition encode folded onto it."""
        partition = bssfp.gz_partition_rew if rewind else bssfp.gz_partition
        return pp.add_gradients([base, pp.scale_grad(partition, kz)], system=system)

    # The catalyst: half the flip, half a TR early, and *opposite* in phase
    # to the first excitation -- which alternation makes phase pi.
    bssfp.rf.amplitude = 0.5 * nominal_amplitude
    bssfp.rf.phase_offset = 0.0
    seq.add_block(bssfp.rf, bssfp.gz, *bssfp.prep_labels)
    seq.add_block(bssfp.wait_prep, *bssfp.train_labels)
    bssfp.rf.amplitude = nominal_amplitude

    previous = None
    shots = [None] * n_dummy + list(pairs)
    for shot, pair in enumerate(shots):
        alternation = np.pi * ((shot + 1) % 2)
        bssfp.rf.phase_offset = alternation
        bssfp.adc.phase_offset = alternation

        if shot:
            seq.add_block(
                bssfp.gx_rew,
                pp.scale_grad(bssfp.gy_rew, previous[0]),
                rephaser(bssfp.gz_rew, True, previous[1]),
            )
        else:
            seq.add_block(bssfp.wait_rewind, bssfp.gz_rew)
        seq.add_block(bssfp.rf, bssfp.gz)

        if pair is None:
            ky, kz = 0.0, 0.0
            seq.add_block(
                bssfp.gx,
                pp.scale_grad(bssfp.gy_pre, ky),
                rephaser(bssfp.gz_pre, False, kz),
            )
        else:
            line, partition = pair
            ky = (line - n_y / 2) / (n_y / 2)
            kz = (partition - n_z / 2) / (n_z / 2)
            index = shot - n_dummy
            lin_label.value = line
            par_label.value = partition
            ima_label.value = int(index <= last_calibration_pair)
            seg_label.value = int(index > last_calibration_pair)
            seq.add_block(
                bssfp.gx,
                bssfp.adc,
                pp.scale_grad(bssfp.gy_pre, ky),
                rephaser(bssfp.gz_pre, False, kz),
                *bssfp.adc_labels,
            )
        previous = (ky, kz)

    seq.add_block(
        bssfp.gx_rew,
        pp.scale_grad(bssfp.gy_rew, previous[0]),
        rephaser(bssfp.gz_rew, True, previous[1]),
        *bssfp.end_labels,
    )

    pp.TransformFOV(
        translation=tuple(offset * 1e3 for offset in fov_offset), system=system
    ).apply_to_sequence(seq, in_place=True)

    if test_report:
        print(seq.test_report())

    if plot:
        seq.plot()

    seq.set_definition(key="FOV", value=[fov_x, fov_y, slab_thickness])
    seq.set_definition(key="Matrix", value=[n_x, n_y, n_z])
    seq.set_definition(key="Name", value="bssfp_3d")
    seq.set_definition(key="TE", value=bssfp.te)
    seq.set_definition(key="TR", value=bssfp.tr)
    seq.set_definition(
        key="NumGainCalibrationReadouts", value=n_gain_calibration_readouts
    )

    seq.set_definition(key="kSpaceCenterLine", value=n_y // 2)
    seq.set_definition(key="kSpaceCenterPartition", value=n_z // 2)
    seq.set_definition(key="kSpaceCenterSample", value=bssfp.center_sample)
    seq.set_definition(key="SliceThickness", value=kernel.excitation.slice_thickness)

    if write_seq:
        write_sequence(seq, seq_filename, offline=True)

    return seq


# ======================================================================
# Subroutines of main()
# ======================================================================


def Bssfp3DKernel(
    system: pp.Opts,
    *,
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 128,
    n_y: int = 128,
    n_z: int = 64,
    slab_thickness: float = 128e-3,
    flip_angle_deg: float = 45.0,
    tr: float | None = None,
    readout_bandwidth_hz: float = 125e3,
    partial_fourier: float = 1.0,
    partial_fourier_z: float = 1.0,
    acceleration: int = 1,
    acceleration_z: int = 1,
    caipi_shift: int = 0,
    elliptical: bool = True,
    n_acs: int = 24,
    n_acs_z: int = 16,
    n_dummy: int = 10,
) -> SimpleNamespace:
    """Design the repetition, and the plan that repeats it.

    Building :class:`design.BssfpReadout3D` *is* the feasibility check; the
    ``(ky, kz)`` traversal -- autocalibration rectangle first -- is the one
    every 3D Cartesian module of the zoo builds.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov, n_x, n_y, n_z, slab_thickness, flip_angle_deg, tr, \
readout_bandwidth_hz, partial_fourier, partial_fourier_z, acceleration, \
acceleration_z, caipi_shift, elliptical, n_acs, n_acs_z, n_dummy
        As for :func:`main`.

    Returns
    -------
    types.SimpleNamespace
        ``excitation``, ``readout``, ``fov``, ``pairs``, ``n_calibration``,
        ``echo_time``, ``repetition_time``, ``bandwidth_hz`` and
        ``duration``.
    """
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

    excitation = design.SpatialSelectiveExcitation(
        system,
        flip_angle_deg,
        slab_thickness,
        duration_s=PULSE_DURATION,
        rephase=False,
        time_bw_product=TIME_BW_PRODUCT,
    )
    readout = design.BssfpReadout3D(
        system,
        excitation.rf,
        excitation.gz,
        fov=(fov_x, fov_y, slab_thickness),
        matrix=(n_x, n_y, n_z),
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        labels=("LIN", "PAR", "IMA", "SEG"),
    )

    pairs, n_calibration = pp.calc_sampled_pairs(
        (n_y, n_z),
        (acceleration, acceleration_z),
        (n_acs, n_acs_z),
        partial_fourier=(partial_fourier, partial_fourier_z),
        caipi_shift=caipi_shift,
        elliptical=elliptical,
        order="calibration_first",
    )

    duration = (n_dummy + len(pairs) + 1.5) * readout.tr

    return SimpleNamespace(
        excitation=excitation,
        readout=readout,
        fov=(fov_x, fov_y),
        pairs=pairs,
        n_calibration=n_calibration,
        echo_time=readout.te,
        repetition_time=readout.tr,
        bandwidth_hz=readout.bandwidth_hz,
        duration=duration,
    )


# ======================================================================
# The scanner protocol contract
# ======================================================================


class Bssfp3D(SequencePlugin):
    """The 3D balanced SSFP behind the scanner protocol contract."""

    def get_default_protocol(self, system: pp.Opts) -> dict[str, dict]:
        """Return the protocol the scanner UI is built from."""
        return protocol_to_dict(
            {
                UIParam.TR: DropdownFloatParam(
                    value=-1.0,
                    min=-1.0,
                    max=20.0,
                    incr=0.05,
                    unit="ms",
                    options=[TRPreset.MINIMUM, 3.0, 4.0, 5.0, 8.0],
                ),
                UIParam.FLIP: DropdownFloatParam(
                    value=45.0,
                    min=10.0,
                    max=90.0,
                    incr=1.0,
                    unit="deg",
                    options=[30.0, 45.0, 60.0, 70.0, 90.0],
                ),
                UIParam.FOV: DropdownFloatParam(
                    value=220.0,
                    min=80.0,
                    max=500.0,
                    incr=1.0,
                    unit="mm",
                    options=[180.0, 220.0, 280.0, 340.0, 500.0],
                ),
                UIParam.PHASE_FOV: DropdownFloatParam(
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
                    value=128,
                    min=16,
                    max=512,
                    incr=1,
                    options=[64, 128, 192, 256, 384],
                ),
                UIParam.NY: DropdownIntParam(
                    value=128,
                    min=16,
                    max=512,
                    incr=1,
                    options=[64, 128, 192, 256, 384],
                ),
                UIParam.NSLICES: DropdownIntParam(
                    value=64,
                    min=8,
                    max=256,
                    incr=1,
                    options=[32, 64, 96, 128, 192],
                ),
                UIParam.BANDWIDTH: TypeinFloatParam(
                    value=125e3,
                    min=5e3,
                    max=500e3,
                    incr=100.0,
                    unit="Hz",
                ),
                UIParam.RY: TypeinFloatParam(
                    value=1.0,
                    min=1.0,
                    max=8.0,
                    incr=1.0,
                    unit="",
                ),
                UIParam.RZ: TypeinFloatParam(
                    value=1.0,
                    min=1.0,
                    max=8.0,
                    incr=1.0,
                    unit="",
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
                UIParam.user_name(0): Description(text="ACS lines (y)"),
                UIParam.user_value(0): TypeinFloatParam(
                    value=24.0,
                    min=0.0,
                    max=512.0,
                    incr=1.0,
                    unit="lines",
                ),
                UIParam.user_name(1): Description(text="CAIPI shift (kz per ky)"),
                UIParam.user_value(1): TypeinFloatParam(
                    value=0.0,
                    min=0.0,
                    max=8.0,
                    incr=1.0,
                    unit="",
                ),
                UIParam.user_name(2): Description(text="Dummy repetitions"),
                UIParam.user_value(2): TypeinFloatParam(
                    value=10.0,
                    min=0.0,
                    max=200.0,
                    incr=1.0,
                    unit="TR",
                ),
                UIParam.user_name(3): Description(text="Partial Fourier (y)"),
                UIParam.user_value(3): TypeinFloatParam(
                    value=1.0,
                    min=0.75,
                    max=1.0,
                    incr=0.05,
                    unit="",
                ),
                UIParam.user_name(4): Description(text="Partial Fourier (z)"),
                UIParam.user_value(4): TypeinFloatParam(
                    value=1.0,
                    min=0.75,
                    max=1.0,
                    incr=0.05,
                    unit="",
                ),
                UIParam.user_name(5): Description(text="ACS partitions (z)"),
                UIParam.user_value(5): TypeinFloatParam(
                    value=16.0,
                    min=0.0,
                    max=256.0,
                    incr=1.0,
                    unit="lines",
                ),
                UIParam.user_name(6): Description(text="Elliptical sampling"),
                UIParam.user_value(6): TypeinFloatParam(
                    value=1.0, min=0.0, max=1.0, incr=1.0, unit=""
                ),
            }
        )

    def validate_protocol(self, system: pp.Opts, protocol: dict[str, dict]) -> dict:
        """Report whether the protocol is feasible, and how long it will take."""
        system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
        kwargs = protocol_kwargs(system, protocol)
        try:
            kernel = Bssfp3DKernel(
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
                f"TA = {kernel.duration:.1f} s, "
                f"TR = {kernel.repetition_time * 1e3:.2f} ms, "
                f"TE = TR/2 = {kernel.echo_time * 1e3:.2f} ms"
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
        "n_x",
        "n_y",
        "n_z",
        "slab_thickness",
        "flip_angle_deg",
        "tr",
        "readout_bandwidth_hz",
        "partial_fourier",
        "partial_fourier_z",
        "acceleration",
        "acceleration_z",
        "caipi_shift",
        "elliptical",
        "n_acs",
        "n_acs_z",
        "n_dummy",
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
        n_z=n_z,
        slab_thickness=n_z * partition_thickness,
        partial_fourier=params.user_float(prot, 3, 1.0),
        partial_fourier_z=params.user_float(prot, 4, 1.0),
        n_acs=params.acs_lines_from_protocol(
            prot, params.param_int(prot, UIParam.NY), 0
        ),
        n_dummy=max(0, round(params.user_float(prot, 2, 10.0))),
        n_acs_z=max(0, round(params.user_float(prot, 5, 16.0))),
        caipi_shift=max(0, round(params.user_float(prot, 1, 0.0))),
        elliptical=bool(round(params.user_float(prot, 6, 1.0))),
    )


PLUGIN = Bssfp3D()


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
    ("--tr-ms", UIParam.TR, float, "Repetition time [ms], or a negative TRPreset"),
    ("--flip-deg", UIParam.FLIP, float, "Flip angle [deg]"),
    ("--fov-mm", UIParam.FOV, float, "Readout FOV [mm]"),
    ("--phase-fov-mm", UIParam.PHASE_FOV, float, "Phase-encode FOV [mm]"),
    (
        "--partition-thickness-mm",
        UIParam.SLICE_THICKNESS,
        float,
        "Partition thickness [mm]; the slab is this times the partition count",
    ),
    ("--nx", UIParam.NX, int, "Readout matrix size"),
    ("--ny", UIParam.NY, int, "Phase-encode matrix size"),
    ("--nz", UIParam.NSLICES, int, "Partition count"),
    ("--bandwidth-hz", UIParam.BANDWIDTH, float, "Requested receiver bandwidth [Hz]"),
    ("--ry", UIParam.RY, float, "Phase-encode undersampling factor along y"),
    ("--rz", UIParam.RZ, float, "Partition-encode undersampling factor along z"),
    ("--offset-x-mm", UIParam.FOV_OFFSET_X, float, "Volume offset along readout [mm]"),
    (
        "--offset-y-mm",
        UIParam.FOV_OFFSET_Y,
        float,
        "Volume offset along phase encode [mm]",
    ),
    ("--offset-z-mm", UIParam.FOV_OFFSET_Z, float, "Volume offset along slab [mm]"),
    ("--acs-lines", UIParam.user_value(0), float, "Number of ACS lines along y"),
    ("--caipi-shift", UIParam.user_value(1), float, "CAIPIRINHA kz shift per ky block"),
    (
        "--dummies",
        UIParam.user_value(2),
        float,
        "Unacquired repetitions after the catalyst",
    ),
    (
        "--partial-fourier",
        UIParam.user_value(3),
        float,
        "Acquired phase-encode fraction along y in (0.5, 1]",
    ),
    (
        "--partial-fourier-z",
        UIParam.user_value(4),
        float,
        "Acquired partition-encode fraction along z in (0.5, 1]",
    ),
    (
        "--acs-partitions",
        UIParam.user_value(5),
        float,
        "Number of ACS partitions along z",
    ),
    (
        "--no-elliptical",
        UIParam.user_value(6),
        lambda value: 0.0 if float(value) else 1.0,
        "Pass 1 to sample the full ky-kz rectangle instead of the ellipse",
    ),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=ARG_MAP,
            description="Generate a 3D balanced SSFP .seq offline.",
            default_output="bssfp_3d.seq",
        )
    )
