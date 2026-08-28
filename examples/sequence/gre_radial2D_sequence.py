"""RF-spoiled 2D radial gradient echo, multi-slice.

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
from scipy.spatial.transform import Rotation

#: The spoke-angle schemes on offer.
ANGLE_SCHEMES = ("golden", "uniform")

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


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "gre_radial_2d.seq",
    *,
    system: pp.Opts | None = None,
    fov: float = 220e-3,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int = 128,
    n_spokes: int | None = None,
    angle_scheme: str = "golden",
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_gap: float = 0.0,
    slice_order: str = "interleaved",
    flip_angle_deg: float = 12.0,
    te: float | None = None,
    tr: float | None = 20e-3,
    readout_bandwidth_hz: float = 250e3,
    n_dummy: int = 16,
    n_gain_calibration_readouts: int | None = None,
    rf_spoiling_increment_deg: float = 117.0,
    spoiling_cycles: float = 4.0,
    use_rotation_ext: bool = True,
) -> pp.Sequence:
    """Create an RF-spoiled 2D radial gradient-echo sequence.

    One full spoke per repetition -- :class:`design.RadialReadout2D`, whose
    prephaser, traversal and rewinder are one continuous waveform. By default the
    spoke is turned per shot by a ``ROTATIONS`` extension rather than by
    re-registering gradients: one waveform however many spokes the scan plays,
    which is the mechanism this slot exists to stress. ``use_rotation_ext=False``
    writes every spoke out as its own waveform instead, for a reader that will not
    compose a rotation. The FOV offset goes through ``TransformFOV`` in server
    mode, where a rotated readout defers its ADC shift to the consumer of the base
    trajectory. :mod:`pulserver.app.noncartesian2D_recon` reconstructs by NUFFT against
    the trajectory the file itself carries.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'gre_radial_2d.seq'.
    system : pypulseq.Opts, optional
        System limits. Default is `pp.Opts()`.
    fov : float, optional
        Isotropic in-plane field of view in meters. Default is 220e-3.
    fov_offset : tuple of float, optional
        Where the prescribed volume sits, in meters along the logical
        readout, phase and slice axes. Applied in server mode: the rotated
        readouts defer their ADC shift to the consumer. Default is
        (0.0, 0.0, 0.0).
    n_x : int, optional
        In-plane matrix size; a spoke reads this many samples edge to edge.
        Default is 128.
    n_spokes : int or None, optional
        How many spokes to play. None is the radial Nyquist count,
        ``ceil(pi/2 * n_x)``. Default is None.
    angle_scheme : str, optional
        ``golden`` or ``uniform``. Default is 'golden'.
    n_slices : int, optional
        Number of slices. Default is 1.
    slice_thickness : float, optional
        Slice thickness in meters. Default is 5e-3.
    slice_gap : float, optional
        Gap between adjacent slices in meters. Default is 0.0.
    slice_order : str, optional
        Order the slices of one pass are excited in. Default is
        'interleaved'.
    flip_angle_deg : float, optional
        Excitation flip angle in degrees. Default is 12.0.
    te : float or None, optional
        Echo time in seconds, to the spoke's centre crossing. None is as
        short as possible. Default is None.
    tr : float or None, optional
        Repetition time in seconds, between successive excitations of the
        same slice. Default is 20e-3.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth in Hz. Default is 250e3.
    n_dummy : int, optional
        Repetitions played without acquiring, before the first spoke of
        each pass. Default is 16.
    n_gain_calibration_readouts : int or None, optional
        Written as the ``NumGainCalibrationReadouts`` definition. None is
        one per slice. Default is None.
    rf_spoiling_increment_deg : float, optional
        Quadratic RF spoiling phase increment in degrees. Default is 117.0.
    spoiling_cycles : float, optional
        Cycles of dephasing left on the slice axis at the end of each
        repetition. Default is 4.0.
    use_rotation_ext : bool, optional
        Hold one spoke and turn it per shot with a ``ROTATIONS`` extension
        (the default), or write every spoke out as its own waveform. The
        second costs a waveform per spoke and reads without composing a
        rotation. Default is True.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The radial GRE sequence object.

    Examples
    --------
    >>> from pulserver.app import gre_radial2D_sequence
    >>> seq = gre_radial2D_sequence(n_x=32, n_spokes=13, n_slices=1, n_dummy=0)
    >>> seq.num_trs, seq.num_segments
    (13, 2)

    The waveform figures below are one design, prescribed to be *legible*
    rather than diagnostic: the shortest TE and TR the readout admits, so
    nothing waits; heavy spoiling, so that lobe is unmistakable; and sixteen
    spokes, so the ordering is legible.

    .. plot::
       :include-source:
       :nofigs:
       :context:

       from pulserver.app import gre_radial2D_sequence

       seq = gre_radial2D_sequence(
           n_x=256, n_spokes=16, n_slices=1, te=None, tr=None, n_dummy=0,
           spoiling_cycles=6.0,
       )

    **The excitation**, and the magnetisation it leaves behind: the pulse's
    own ``B1`` envelope beside the profile it writes across the selected
    axis.

    .. plot::
       :include-source:
       :context: close-figs

       seq.plot_rf(plot_now=False)

    **One repetition**, which is one spoke: the excitation, the prewinder
    that reaches the far edge of k-space, the readout straight through the
    centre, and the spoiler. The waveform is stored once and turned by a
    rotation extension, so every spoke plays these same shapes.

    .. plot::
       :include-source:
       :context: close-figs

       seq.plot(tr="worst_case", time_disp="ms", grad_disp="mT/m", plot_now=False)

    **What the scan covers.** Sixteen golden-angle spokes, each landing in
    the widest gap the ones before it left, which is what the ordering colour
    shows.

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
    This design asks for the shortest timing the hardware admits, so its
    prewinder and spoiler ramp as fast as they are allowed to. A lower
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

    kernel = RadialKernel(
        system,
        fov=fov,
        n_x=n_x,
        n_spokes=n_spokes,
        angle_scheme=angle_scheme,
        n_slices=n_slices,
        slice_thickness=slice_thickness,
        slice_order=slice_order,
        flip_angle_deg=flip_angle_deg,
        te=te,
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        n_dummy=n_dummy,
        spoiling_cycles=spoiling_cycles,
        use_rotation_ext=use_rotation_ext,
    )
    excitation = kernel.excitation
    angles = kernel.angles
    slice_positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
        slice_thickness + slice_gap
    )
    rf_phases = pp.make_rf_spoiling_schedule(
        (len(angles) + n_dummy) * n_slices,
        increment=np.deg2rad(rf_spoiling_increment_deg),
    )
    rotations = (
        [pp.make_rotation(Rotation.from_euler("z", float(angle))) for angle in angles]
        if use_rotation_ext
        else [None] * len(angles)
    )

    seq = pp.Sequence(system)
    spoiling_phase = iter(rf_phases)

    def repetition(readout, slices, i_spoke: int, acquire: bool, mark=None) -> None:
        """Play one spoke of every slice of a pass, acquiring or not."""
        lin_label, slc_label = readout.adc_labels
        for i_slice in slices:
            rf_phase = next(spoiling_phase)
            readout.rf.freq_offset = excitation.gz.amplitude * slice_positions[i_slice]
            readout.rf.phase_offset = (
                rf_phase - 2 * np.pi * readout.rf.freq_offset * readout.rf.center
            )
            readout.adc.phase_offset = rf_phase
            lin_label.value = int(i_spoke)
            slc_label.value = int(i_slice)
            ShotKernel(
                seq,
                readout.arm(i_spoke),
                rotations[i_spoke],
                acquire=acquire,
                first_extra=() if mark is None else (mark,),
            )
            mark = None

    for slices in kernel.passes:
        readout = kernel.readouts[len(slices)]
        for i_dummy in range(n_dummy):
            repetition(
                readout,
                slices,
                0,
                acquire=False,
                mark=pp.make_label("ONCE", "SET", 1) if i_dummy == 0 else None,
            )
        clear_once = pp.make_label("ONCE", "SET", 0) if n_dummy else None
        for i_spoke in range(len(angles)):
            repetition(readout, slices, i_spoke, acquire=True, mark=clear_once)
            clear_once = None

    # Server mode: the rotated readouts defer their ADC shift to the consumer
    # of the base trajectory, which is the mechanism this family demonstrates.
    pp.TransformFOV(
        translation=tuple(offset * 1e3 for offset in fov_offset),
        system=system,
        compat=False,
    ).apply_to_sequence(seq, in_place=True)

    if test_report:
        print(seq.test_report())

    if plot:
        seq.plot()

    slab_thickness = n_slices * (slice_thickness + slice_gap) - slice_gap
    seq.set_definition(key="FOV", value=[fov, fov, slab_thickness])
    seq.set_definition(key="Matrix", value=[n_x, n_x, n_slices])
    seq.set_definition(key="Name", value="gre_radial_2d")
    seq.set_definition(key="TE", value=kernel.echo_time)
    seq.set_definition(key="TR", value=kernel.repetition_time)
    seq.set_definition(key="Trajectory", value="radial")
    seq.set_definition(key="NumSpokes", value=len(angles))
    seq.set_definition(key="AngleScheme", value=angle_scheme)
    seq.set_definition(
        key="kSpaceCenterSample",
        value=kernel.readouts[len(kernel.passes[0])].center_sample,
    )
    seq.set_definition(key="SliceThickness", value=excitation.slice_thickness)
    seq.set_definition(
        key="NumGainCalibrationReadouts",
        value=n_slices
        if n_gain_calibration_readouts is None
        else n_gain_calibration_readouts,
    )

    if write_seq:
        write_sequence(seq, seq_filename, offline=True)

    return seq


# ======================================================================
# Subroutines of main()
# ======================================================================

# ======================================================================
# Subroutines of main()
# ======================================================================


def ShotKernel(seq, blocks, rotation, *, acquire: bool, first_extra=()) -> None:
    """Replay one spoke's blocks with the shot's orientation injected.

    Every block that drives an in-plane gradient gains the rotation, when there
    is one to give -- an explicit spoke is already turned. A dummy drops the
    ADC and the counters that go with it. The blocks are replayed as the module
    laid them out, delays included, so the TE budget the module solved survives
    the loop untouched.
    """
    extra_first = list(first_extra)
    for block in blocks:
        events = [
            event
            for event in block
            if acquire or getattr(event, "type", "") not in ("adc", "labelset")
        ]
        additions = list(extra_first)
        extra_first = []
        if rotation is not None and any(
            getattr(event, "channel", "") in ("x", "y") for event in events
        ):
            additions.append(rotation)
        seq.add_block(*events, *additions)


def spoke_angles(n_spokes: int, scheme: str) -> np.ndarray:
    """The in-plane angle of every spoke, in radians.

    ``golden`` increments by the golden angle, so any prefix of the scan is
    close to uniformly distributed -- what an interrupted or dynamic scan
    wants. ``uniform`` spaces the spokes evenly over half a turn, which a
    full spoke covers.

    Parameters
    ----------
    n_spokes : int
        How many spokes the scan plays.
    scheme : str
        One of :data:`ANGLE_SCHEMES`.

    Returns
    -------
    numpy.ndarray
        One angle per spoke.
    """
    if scheme not in ANGLE_SCHEMES:
        raise ValueError(f"scheme must be one of {ANGLE_SCHEMES}, got {scheme!r}")
    if scheme == "golden":
        return np.asarray(pp.calc_golden_angles(n_spokes))
    return np.asarray(pp.calc_uniform_angles(n_spokes, span=np.pi))


def RadialKernel(
    system: pp.Opts,
    *,
    fov: float = 220e-3,
    n_x: int = 128,
    n_spokes: int | None = None,
    angle_scheme: str = "golden",
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_order: str = "interleaved",
    flip_angle_deg: float = 12.0,
    te: float | None = None,
    tr: float | None = 20e-3,
    readout_bandwidth_hz: float = 250e3,
    n_dummy: int = 16,
    spoiling_cycles: float = 4.0,
    use_rotation_ext: bool = True,
) -> SimpleNamespace:
    """Design the spoke, and the plan that turns it.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov, n_x, n_spokes, angle_scheme, n_slices, slice_thickness, \
slice_order, flip_angle_deg, te, tr, readout_bandwidth_hz, n_dummy, \
spoiling_cycles, use_rotation_ext
        As for :func:`main`.

    Returns
    -------
    types.SimpleNamespace
        ``excitation``, ``readouts`` (keyed by pass size), ``passes``,
        ``angles``, ``echo_time``, ``repetition_time``, ``bandwidth_hz``
        and ``duration``.
    """
    excitation = design.SpatialSelectiveExcitation(
        system,
        flip_angle_deg,
        slice_thickness,
        duration_s=PULSE_DURATION,
        time_bw_product=TIME_BW_PRODUCT,
    )

    count = int(np.pi / 2 * n_x) if n_spokes is None else int(n_spokes)
    angles = spoke_angles(count, angle_scheme)

    def readout(module_tr: float | None):
        return design.RadialReadout2D(
            system,
            excitation.rf,
            excitation.gz,
            excitation.gz_reph,
            fov=fov,
            matrix=n_x,
            te=te,
            tr=module_tr,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=spoiling_cycles,
            # LIN carries the shot: a radial scan has no phase-encode line, so
            # the counter a reconstruction grids by is the spoke index.
            labels=("LIN", "SLC"),
            explicit=not use_rotation_ext,
            angles=None if use_rotation_ext else angles,
        )

    shortest = readout(None)
    per_pass = n_slices if tr is None else max(1, int(tr / shortest.duration))
    n_passes = -(-n_slices // per_pass)
    passes = [
        [int(i) for i in range(start, n_slices, n_passes)] for start in range(n_passes)
    ]
    passes = [
        [group[int(i)] for i in pp.calc_traversal_order(len(group), slice_order)]
        for group in passes
        if group
    ]
    readouts = {
        size: shortest if tr is None else readout(tr / size)
        for size in {len(group) for group in passes}
    }

    pass_time = sum(len(group) * readouts[len(group)].duration for group in passes)
    duration = (n_dummy + len(angles)) * pass_time

    return SimpleNamespace(
        excitation=excitation,
        readouts=readouts,
        passes=passes,
        angles=angles,
        echo_time=shortest.echo_time,
        repetition_time=max(
            len(group) * readouts[len(group)].duration for group in passes
        ),
        bandwidth_hz=shortest.bandwidth_hz,
        duration=duration,
    )


# ======================================================================
# The scanner protocol contract
# ======================================================================

# ======================================================================
# The scanner protocol contract
# ======================================================================


class GreRadial2D(SequencePlugin):
    """The 2D radial gradient echo behind the scanner protocol contract."""

    def get_default_protocol(self, system: pp.Opts) -> dict[str, dict]:
        """Return the protocol the scanner UI is built from."""
        return protocol_to_dict(
            {
                UIParam.TE: DropdownFloatParam(
                    value=-1.0,
                    min=-1.0,
                    max=40.0,
                    incr=0.1,
                    unit="ms",
                    options=[TEPreset.MINIMUM, 2.0, 4.0, 8.0, 15.0],
                ),
                UIParam.TR: DropdownFloatParam(
                    value=20.0,
                    min=2.0,
                    max=5000.0,
                    incr=0.1,
                    unit="ms",
                    options=[TRPreset.MINIMUM, 10.0, 20.0, 50.0, 250.0],
                ),
                UIParam.FLIP: DropdownFloatParam(
                    value=12.0,
                    min=1.0,
                    max=90.0,
                    incr=1.0,
                    unit="deg",
                    options=[5.0, 12.0, 30.0, 60.0, 90.0],
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
                    value=5.0,
                    min=1.0,
                    max=20.0,
                    incr=0.5,
                    unit="mm",
                    options=[1.0, 3.0, 5.0, 8.0, 10.0],
                ),
                UIParam.SLICE_SPACING: DropdownFloatParam(
                    value=5.0,
                    min=1.0,
                    max=20.0,
                    incr=0.5,
                    unit="mm",
                    options=[1.0, 3.0, 5.0, 8.0, 10.0],
                ),
                UIParam.NX: DropdownIntParam(
                    value=128, min=16, max=512, incr=1, options=[64, 128, 192, 256, 384]
                ),
                UIParam.NSLICES: DropdownIntParam(
                    value=1, min=1, max=128, incr=1, options=[1, 5, 10, 20, 40]
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
                UIParam.user_name(0): Description(text="Spokes (0 = Nyquist)"),
                UIParam.user_value(0): TypeinFloatParam(
                    value=0.0, min=0.0, max=4096.0, incr=1.0, unit=""
                ),
                UIParam.user_name(1): Description(text="Angles 0=golden 1=uniform"),
                UIParam.user_value(1): TypeinFloatParam(
                    value=0.0, min=0.0, max=1.0, incr=1.0, unit=""
                ),
                UIParam.user_name(2): Description(text="Dummy scans"),
                UIParam.user_value(2): TypeinFloatParam(
                    value=16.0, min=0.0, max=128.0, incr=1.0, unit="TR"
                ),
            }
        )

    def validate_protocol(self, system: pp.Opts, protocol: dict[str, dict]) -> dict:
        """Report whether the protocol is feasible, and how long it will take."""
        system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
        kwargs = protocol_kwargs(system, protocol)
        try:
            kernel = RadialKernel(
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
                f"TA = {kernel.duration:.1f} s over {len(kernel.angles)} spokes, "
                f"{len(kernel.passes)} pass(es)"
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
        "n_spokes",
        "angle_scheme",
        "n_slices",
        "slice_thickness",
        "slice_order",
        "flip_angle_deg",
        "te",
        "tr",
        "readout_bandwidth_hz",
        "n_dummy",
        "spoiling_cycles",
    )
)


def protocol_kwargs(system: pp.Opts, protocol: dict[str, dict]) -> dict:
    """The prescribed quantities, plus this sequence's own user slots."""
    prot = dict_to_protocol(protocol)
    requested = round(params.user_float(prot, 0, 0.0))
    return main_kwargs(
        main,
        system,
        protocol,
        fov=params.param_float(prot, UIParam.FOV) * 1e-3,
        n_spokes=None if requested <= 0 else requested,
        angle_scheme=ANGLE_SCHEMES[
            int(np.clip(round(params.user_float(prot, 1, 0.0)), 0, 1))
        ],
        n_dummy=max(0, round(params.user_float(prot, 2, 16.0))),
    )


PLUGIN = GreRadial2D()


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
    ("--te-ms", UIParam.TE, float, "Echo time [ms], or a negative TEPreset"),
    ("--tr-ms", UIParam.TR, float, "Repetition time [ms], or a negative TRPreset"),
    ("--flip-deg", UIParam.FLIP, float, "Flip angle [deg]"),
    ("--fov-mm", UIParam.FOV, float, "In-plane FOV [mm]"),
    ("--slice-thickness-mm", UIParam.SLICE_THICKNESS, float, "Slice thickness [mm]"),
    ("--slice-spacing-mm", UIParam.SLICE_SPACING, float, "Slice spacing [mm]"),
    ("--nx", UIParam.NX, int, "In-plane matrix size"),
    ("--nslices", UIParam.NSLICES, int, "Number of slices"),
    ("--bandwidth-hz", UIParam.BANDWIDTH, float, "Requested receiver bandwidth [Hz]"),
    ("--offset-x-mm", UIParam.FOV_OFFSET_X, float, "Volume offset along readout [mm]"),
    (
        "--offset-y-mm",
        UIParam.FOV_OFFSET_Y,
        float,
        "Volume offset along phase encode [mm]",
    ),
    ("--offset-z-mm", UIParam.FOV_OFFSET_Z, float, "Volume offset along slice [mm]"),
    ("--spokes", UIParam.user_value(0), float, "Spokes to play (0 = Nyquist count)"),
    ("--angles", UIParam.user_value(1), float, "Angle scheme: 0 golden, 1 uniform"),
    ("--dummies", UIParam.user_value(2), float, "Unacquired repetitions per pass"),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=ARG_MAP,
            description="Generate a 2D radial gradient-echo .seq offline.",
            default_output="gre_radial_2d.seq",
        )
    )
