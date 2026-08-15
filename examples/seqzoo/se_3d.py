"""3D Cartesian spin echo, slab-selective.

The refocusing k-flip on a volume: a slab-selective SLR excitation, one SLR
180 with bridged crushers a half-TE later, and one frequency-encoded line
phase-encoded along y and z at the echo -- :class:`design.LineReadout3D`
opened by the refocusing pulse. The autocalibration rectangle leads the
``(ky, kz)`` traversal exactly as :mod:`pulserver.seqzoo.gre_3d` orders it,
regular undersampling laying a CAIPIRINHA lattice with a selectable kz shift
per ky block, and :mod:`pulserver.reczoo.se_3d` reads it back.

TE spans excitation centre to echo, with the 180 at its midpoint: the readout
solves the second half (``te = TE/2`` from the refocusing pulse), and a delay
after the excitation solves the first.

``main`` returns the :class:`pulserver.pypulseq.Sequence`; ``PLUGIN`` is the
same sequence behind the scanner protocol contract, and running this module
as a script writes a ``.seq`` from the same controls.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pulserver.design as design
import pulserver.pypulseq as pp
from pulserver import (
    Description,
    DropdownFloatParam,
    DropdownIntParam,
    OffFloatParam,
    SequencePlugin,
    TEPreset,
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

#: SLR design shared by the excitation and the refocusing pulse.
_PULSE_DURATION = 3e-3
_TIME_BW_PRODUCT = 4.0

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
    seq_filename: str = "se_3d.seq",
    *,
    system: pp.Opts | None = None,
    fov: float | tuple[float, float] = 220e-3,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int = 128,
    n_y: int = 128,
    n_z: int = 64,
    slab_thickness: float = 128e-3,
    te: float | None = 15e-3,
    tr: float | None = 100e-3,
    readout_bandwidth_hz: float = 250e3,
    partial_echo: float = 1.0,
    partial_fourier: float = 1.0,
    partial_fourier_z: float = 1.0,
    acceleration: int = 1,
    acceleration_z: int = 1,
    caipi_shift: int = 0,
    n_acs: int = 24,
    n_acs_z: int = 16,
    n_averages: int = 1,
    n_dummy: int = 0,
    n_gain_calibration_readouts: int = 1,
    crusher_cycles: float = 4.0,
    spoiling_cycles: float = 4.0,
) -> pp.Sequence:
    """Create a 3D Cartesian spin-echo sequence.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'se_3d.seq'.
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
    te : float or None, optional
        Echo time in seconds, excitation centre to echo, with the refocusing
        pulse at its midpoint. None is as short as possible. Default is
        15e-3.
    tr : float or None, optional
        Repetition time in seconds, one per excitation. None is as short as
        possible. Default is 100e-3.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth in Hz. Default is 250e3.
    partial_echo : float, optional
        Fraction of the full echo acquired, in (0.5, 1]. Default is 1.0.
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
    n_acs : int, optional
        Autocalibration extent along y, in lines. Default is 24.
    n_acs_z : int, optional
        Autocalibration extent along z, in partitions. Default is 16.
    n_averages : int, optional
        How many times the scan is acquired, written into the block table.
        Default is 1.
    n_dummy : int, optional
        Repetitions played without acquiring before the first pair. Default
        is 0.
    n_gain_calibration_readouts : int, optional
        Written as the ``NumGainCalibrationReadouts`` definition. Default
        is 1.
    crusher_cycles : float, optional
        Cycles of dephasing each crusher beside the refocusing pulse winds
        across one voxel. Default is 4.0.
    spoiling_cycles : float, optional
        Cycles of dephasing left on the readout axis at the end of each
        repetition. Default is 4.0.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The spin-echo sequence object.
    """
    system = pp.Opts() if system is None else system
    system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)

    kernel = SE3DKernel(
        system,
        fov=fov,
        n_x=n_x,
        n_y=n_y,
        n_z=n_z,
        slab_thickness=slab_thickness,
        te=te,
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        partial_echo=partial_echo,
        partial_fourier=partial_fourier,
        partial_fourier_z=partial_fourier_z,
        acceleration=acceleration,
        acceleration_z=acceleration_z,
        caipi_shift=caipi_shift,
        n_acs=n_acs,
        n_acs_z=n_acs_z,
        n_averages=n_averages,
        n_dummy=n_dummy,
        crusher_cycles=crusher_cycles,
        spoiling_cycles=spoiling_cycles,
    )
    excitation = kernel.excitation
    readout = kernel.readout
    timing = kernel.timing
    fov_x, fov_y = kernel.fov
    pairs = kernel.pairs
    last_calibration_pair = kernel.n_calibration - 1

    seq = pp.Sequence(system)
    ima_label, seg_label = readout.adc_labels
    wait_te = getattr(readout, "wait_te", None)

    def repetition(ky: float, kz: float, acquire: bool, mark=None) -> None:
        """Play one TR, acquiring or not."""
        seq.add_block(
            excitation.rf, excitation.gz, *([mark] if mark is not None else [])
        )
        if timing.wait_half_te is not None:
            seq.add_block(timing.wait_half_te)
        seq.add_block(readout.rf, readout.gz)
        if wait_te is not None:
            seq.add_block(wait_te)
        seq.add_block(
            readout.gx_pre,
            pp.scale_grad(readout.gy_pre, ky),
            pp.scale_grad(readout.gz_pre, kz),
        )
        if acquire:
            seq.add_block(readout.gx, readout.adc, ima_label, seg_label)
        else:
            seq.add_block(readout.gx)
        seq.add_block(
            readout.gx_spoil,
            pp.scale_grad(readout.gy_rew, ky),
            pp.scale_grad(readout.gz_rew, kz),
        )
        if timing.wait_tr is not None:
            seq.add_block(timing.wait_tr)

    for i_dummy in range(n_dummy):
        repetition(
            0.0,
            0.0,
            acquire=False,
            mark=pp.make_label("ONCE", "SET", 1) if i_dummy == 0 else None,
        )

    clear_once = pp.make_label("ONCE", "SET", 0) if n_dummy else None
    for index, (line, partition) in enumerate(pairs):
        ky = (line - n_y / 2) / (n_y / 2)
        kz = (partition - n_z / 2) / (n_z / 2)
        ima_label.value = int(index <= last_calibration_pair)
        seg_label.value = int(index > last_calibration_pair)

        repetition(ky, kz, acquire=True, mark=clear_once)
        clear_once = None

    pp.TransformFOV(
        translation=tuple(offset * 1e3 for offset in fov_offset), system=system
    ).apply_to_sequence(seq, in_place=True)

    if test_report:
        print(seq.test_report())

    if plot:
        seq.plot()

    seq.set_definition(key="FOV", value=[fov_x, fov_y, slab_thickness])
    seq.set_definition(key="Matrix", value=[n_x, n_y, n_z])
    seq.set_definition(key="Name", value="se_3d")
    seq.set_definition(key="TE", value=kernel.echo_time)
    seq.set_definition(key="TR", value=kernel.repetition_time)
    seq.set_definition(
        key="NumGainCalibrationReadouts", value=n_gain_calibration_readouts
    )

    seq.auto_label()
    seq.expand_repeats(n_averages)

    if write_seq:
        write_sequence(seq, seq_filename, offline=True)

    return seq


def SE3DKernel(
    system: pp.Opts,
    *,
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 128,
    n_y: int = 128,
    n_z: int = 64,
    slab_thickness: float = 128e-3,
    te: float | None = 15e-3,
    tr: float | None = 100e-3,
    readout_bandwidth_hz: float = 250e3,
    partial_echo: float = 1.0,
    partial_fourier: float = 1.0,
    partial_fourier_z: float = 1.0,
    acceleration: int = 1,
    acceleration_z: int = 1,
    caipi_shift: int = 0,
    n_acs: int = 24,
    n_acs_z: int = 16,
    n_averages: int = 1,
    n_dummy: int = 0,
    crusher_cycles: float = 4.0,
    spoiling_cycles: float = 4.0,
) -> SimpleNamespace:
    """Design the repetition, and the plan that repeats it.

    The echo time is solved in two halves around the refocusing pulse
    exactly as :func:`pulserver.seqzoo.se_2d.SE2DKernel` solves it; the
    ``(ky, kz)`` traversal -- autocalibration rectangle first -- is the one
    :func:`pulserver.seqzoo.gre_3d.GRE3DKernel` builds. A slab excitation
    carries its rephaser inside its selection gradient, so the excitation is
    one block and k is already zero when the 180 flips it.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov, n_x, n_y, n_z, slab_thickness, te, tr, readout_bandwidth_hz, \
partial_echo, partial_fourier, partial_fourier_z, acceleration, acceleration_z, \
caipi_shift, n_acs, n_acs_z, n_averages, n_dummy, crusher_cycles, spoiling_cycles
        As for :func:`main`.

    Returns
    -------
    types.SimpleNamespace
        ``excitation``, ``refocusing``, ``readout``, ``timing``
        (``wait_half_te``/``wait_tr``/``length``), ``fov``, ``pairs``,
        ``n_calibration``, ``n_averages``, ``echo_time``,
        ``repetition_time``, ``bandwidth_hz`` and ``duration``.
    """
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

    excitation = design.SpatialSelectiveExcitation(
        system,
        90.0,
        slab_thickness,
        duration_s=_PULSE_DURATION,
        time_bw_product=_TIME_BW_PRODUCT,
        is_slab=True,
    )
    refocusing = design.SpatialSelectiveRefocusing(
        system,
        slab_thickness,
        duration_s=_PULSE_DURATION,
        time_bw_product=_TIME_BW_PRODUCT,
        spoiling_cycles=crusher_cycles,
    )

    exc_center = excitation.rf.delay + excitation.rf.center
    half_te_floor = (
        excitation.seq.duration()[0] - exc_center
    ) + refocusing.center

    def build(half_te: float | None):
        readout = design.LineReadout3D(
            system,
            refocusing.rf_ref,
            refocusing.gz,
            fov=(fov_x, fov_y, slab_thickness),
            matrix=(n_x, n_y, n_z),
            te=half_te,
            partial_echo=partial_echo,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=spoiling_cycles,
            labels=("IMA", "SEG"),
        )
        achieved_half = readout.echo_time
        if half_te is not None and half_te_floor > achieved_half + 1e-9:
            raise ValueError(
                f"TE {2 * half_te * 1e3:.2f} ms is shorter than the "
                f"excitation half admits; the minimum is "
                f"{2 * half_te_floor * 1e3:.2f} ms"
            )
        wait = max(0.0, achieved_half - half_te_floor)
        wait_half_te = (
            pp.make_delay(pp.round_to_raster(wait, system.block_duration_raster))
            if wait > 0
            else None
        )
        length = (
            excitation.seq.duration()[0]
            + (wait_half_te.delay if wait_half_te is not None else 0.0)
            + readout.duration
        )
        wait_tr = None
        if tr is not None:
            if tr < length - 1e-9:
                raise ValueError(
                    f"TR {tr * 1e3:.1f} ms is shorter than one repetition "
                    f"takes ({length * 1e3:.1f} ms)"
                )
            pad = pp.round_to_raster(tr - length, system.block_duration_raster)
            if pad > 0:
                wait_tr = pp.make_delay(pad)
                length = length + pad
        return readout, SimpleNamespace(
            wait_half_te=wait_half_te, wait_tr=wait_tr, length=length
        )

    shortest, _ = build(None)
    half_te = max(shortest.echo_time, half_te_floor) if te is None else te / 2
    readout, timing = build(half_te)

    pairs, n_calibration = pp.calc_sampled_pairs(
        (n_y, n_z),
        (acceleration, acceleration_z),
        (n_acs, n_acs_z),
        partial_fourier=(partial_fourier, partial_fourier_z),
        caipi_shift=caipi_shift,
        order="calibration_first",
    )

    duration = (n_dummy + n_averages * len(pairs)) * timing.length

    return SimpleNamespace(
        excitation=excitation,
        refocusing=refocusing,
        readout=readout,
        timing=timing,
        fov=(fov_x, fov_y),
        pairs=pairs,
        n_calibration=n_calibration,
        n_averages=n_averages,
        echo_time=2 * half_te,
        repetition_time=timing.length,
        bandwidth_hz=readout.bandwidth_hz,
        duration=duration,
    )


class Se3D(SequencePlugin):
    """The 3D spin echo behind the scanner protocol contract."""

    def get_default_protocol(self, system: pp.Opts) -> dict[str, dict]:
        """Return the protocol the scanner UI is built from."""
        return protocol_to_dict(
            {
                UIParam.TE: DropdownFloatParam(
                    value=15.0,
                    min=5.0,
                    max=200.0,
                    incr=0.1,
                    unit="ms",
                    options=[TEPreset.MINIMUM, 15.0, 30.0, 60.0, 90.0],
                ),
                UIParam.TR: DropdownFloatParam(
                    value=100.0,
                    min=20.0,
                    max=5000.0,
                    incr=1.0,
                    unit="ms",
                    options=[TRPreset.MINIMUM, 100.0, 250.0, 500.0, 1000.0],
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
                    value=250e3,
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
                UIParam.NEX: DropdownFloatParam(
                    value=1.0,
                    min=1.0,
                    max=32.0,
                    incr=1.0,
                    unit="",
                    options=[1.0, 2.0, 4.0, 8.0, 16.0],
                ),
                UIParam.FOV_OFFSET_X: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.FOV_OFFSET_Y: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.FOV_OFFSET_Z: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.user_name(0): Description(text="ACS lines (y)"),
                UIParam.user_value(0): TypeinFloatParam(
                    value=24.0,
                    min=0.0,
                    max=512.0,
                    incr=1.0,
                    unit="lines",
                ),
                UIParam.user_name(1): Description(text="Partial echo"),
                UIParam.user_value(1): TypeinFloatParam(
                    value=1.0,
                    min=0.75,
                    max=1.0,
                    incr=0.05,
                    unit="",
                ),
                UIParam.user_name(2): Description(text="Dummy scans"),
                UIParam.user_value(2): TypeinFloatParam(
                    value=0.0,
                    min=0.0,
                    max=512.0,
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
                UIParam.user_name(6): Description(text="CAIPI shift (kz per ky)"),
                UIParam.user_value(6): TypeinFloatParam(
                    value=0.0,
                    min=0.0,
                    max=8.0,
                    incr=1.0,
                    unit="",
                ),
            }
        )

    def validate_protocol(self, system: pp.Opts, protocol: dict[str, dict]) -> dict:
        """Report whether the protocol is feasible, and how long it will take."""
        system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
        kwargs = _main_kwargs(system, protocol)
        try:
            kernel = SE3DKernel(
                system,
                **{name: value for name, value in kwargs.items() if name in _KERNEL_ARGUMENTS},
            )
        except ValueError as error:
            return {"valid": False, "duration": None, "info": str(error)}

        return {
            "valid": True,
            "duration": kernel.duration,
            "info": (
                f"TA = {kernel.duration:.1f} s at "
                f"{kernel.bandwidth_hz * 1e-3:.1f} kHz, "
                f"TR = {kernel.repetition_time * 1e3:.2f} ms"
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
        seq = main(**_main_kwargs(system, protocol))
        write_sequence(seq, output_path, offline=offline)


_KERNEL_ARGUMENTS = frozenset(
    (
        "fov",
        "n_x",
        "n_y",
        "n_z",
        "slab_thickness",
        "te",
        "tr",
        "readout_bandwidth_hz",
        "partial_echo",
        "partial_fourier",
        "partial_fourier_z",
        "acceleration",
        "acceleration_z",
        "caipi_shift",
        "n_acs",
        "n_acs_z",
        "n_averages",
        "n_dummy",
        "crusher_cycles",
        "spoiling_cycles",
    )
)


def _main_kwargs(system: pp.Opts, protocol: dict[str, dict]) -> dict:
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
        partial_echo=params.user_float(prot, 1, 1.0),
        partial_fourier=params.user_float(prot, 3, 1.0),
        partial_fourier_z=params.user_float(prot, 4, 1.0),
        n_acs=params.acs_lines_from_protocol(prot, params.param_int(prot, UIParam.NY), 0),
        n_dummy=max(0, round(params.user_float(prot, 2, 0.0))),
        n_acs_z=max(0, round(params.user_float(prot, 5, 16.0))),
        caipi_shift=max(0, round(params.user_float(prot, 6, 0.0))),
    )


PLUGIN = Se3D()


def get_default_protocol(system):
    """Bridge entry point: the plugin's default protocol."""
    return PLUGIN.get_default_protocol(system)


def validate_protocol(system, protocol):
    """Bridge entry point: protocol feasibility and scan duration."""
    return PLUGIN.validate_protocol(system, protocol)


def make_sequence(system, protocol, output_path):
    """Bridge entry point: write the ``.seq`` file."""
    return PLUGIN.make_sequence(system, protocol, output_path)


_ARG_MAP = [
    ("--te-ms", UIParam.TE, float, "Echo time [ms], or a negative TEPreset"),
    ("--tr-ms", UIParam.TR, float, "Repetition time [ms], or a negative TRPreset"),
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
    ("--nex", UIParam.NEX, float, "Number of signal averages"),
    ("--offset-x-mm", UIParam.FOV_OFFSET_X, float, "Volume offset along readout [mm]"),
    ("--offset-y-mm", UIParam.FOV_OFFSET_Y, float, "Volume offset along phase encode [mm]"),
    ("--offset-z-mm", UIParam.FOV_OFFSET_Z, float, "Volume offset along slab [mm]"),
    ("--acs-lines", UIParam.user_value(0), float, "Number of ACS lines along y"),
    ("--partial-echo", UIParam.user_value(1), float, "Acquired echo fraction in (0.5, 1]"),
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
    ("--acs-partitions", UIParam.user_value(5), float, "Number of ACS partitions along z"),
    ("--caipi-shift", UIParam.user_value(6), float, "CAIPIRINHA kz shift per ky block"),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description="Generate a 3D Cartesian spin-echo .seq offline.",
            default_output="se_3d.seq",
        )
    )
