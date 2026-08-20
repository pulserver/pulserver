"""3D Cartesian fast spin echo, slab-selective, with selectable view ordering.

One long CPMG train per excitation over a ``(ky, kz)`` grid --
:class:`design.FseReadout3D` -- with the two decisions a long train forces
made explicit, after Busse et al. 2008 (MRM 60:640), followed loosely:

**View ordering** maps the train's signal modulation into k-space, and is
delegated to the :mod:`pulserver.pypulseq` echo-train ordering builtins. The
*coherent* orderings sort views so successive echoes stay close in k-space
and the centre lands at the echo whose time is the requested effective TE:
``linear`` (raster bands rolled onto the target echo), ``centric`` (global
centre-out radius bands, rolled onto the target echo), ``radial`` (centre-out
by radius, effective TE at the first echo), and ``radial_adaptive`` (by
radius, but the bands assigned outward from the target echo, so the centre is
late without a seam). The *incoherent* ordering, ``shuffling``, scatters views
across echoes for a subspace reconstruction. Every acquisition carries its echo
index as ``ECO``, which is what any of them reconstruct from.

**Sampling** is undersampled through the same builtins: the regular orderings
lay a CAIPIRINHA lattice (``caipi_shift`` staggers kz per ky block) around a
fully sampled autocalibration rectangle, while ``shuffling`` draws an
incoherent variable-density Poisson-disc set when accelerated.

**Refocusing flip modulation** trades signal for blur over long trains. The
optional variable train is a TRAPS-style piecewise ramp in flip space --
``alpha_max`` at the first echo, down to ``alpha_min``, back up to
``alpha_center`` at the effective-TE echo, and on to ``alpha_max`` by the
train's end -- the shape Busse's prescribed-envelope method produces,
without the EPG inversion. The played angles are written into the
``RefocusingFlipAngles`` definition.

``main`` returns the :class:`pulserver.pypulseq.Sequence`; ``PLUGIN`` is the
same sequence behind the scanner protocol contract, and running this module
as a script writes a ``.seq`` from the same controls.

Examples
--------
>>> from pulserver.app import fse3D_sequence
>>> seq = fse3D_sequence(n_x=64, n_y=16, n_z=4, etl=4, te=20e-3, tr=None)
>>> seq.num_trs, seq.num_segments
(16, 3)

A non-selective refocusing train over a slab, which is what makes single-slab 3D FSE efficient:

.. plot::
   :include-source:

   from pulserver.app import fse3D_sequence

   seq = fse3D_sequence(n_x=64, n_y=16, n_z=4, etl=4, te=20e-3, tr=None)
   seq.plot(tr="worst_case", time_disp="ms", grad_disp="mT/m", stacked=True,
            plot_now=False)

Which view each echo encodes is what the effective echo time is, and the
ordering is where that is decided. All five are dealt from the same
``(ky, kz)`` grid; only the rank they sort on differs:

.. plot::

   import numpy as np
   from pulserver.app.sequence.fse3D_sequence import ORDERINGS, order_views
   from _figures import order_figure

   grid = (32, 32)
   views = [(y, z) for y in range(grid[0]) for z in range(grid[1])]
   order_figure(
       [
           (name, order_views(views, 32, 0, name, grid))
           for name in ("linear", "centric", "radial", "shuffling")
       ],
       views,
       title="the same views, four orderings",
   )
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
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

#: The selectable view orderings; see the module docstring. Each dispatches to
#: the matching :mod:`pulserver.pypulseq` echo-train ordering builtin.
ORDERINGS = ("linear", "centric", "radial", "radial_adaptive", "shuffling")


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "fse_3d.seq",
    *,
    system: pp.Opts | None = None,
    fov: float | tuple[float, float] = 220e-3,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int = 128,
    n_y: int = 128,
    n_z: int = 64,
    slab_thickness: float = 128e-3,
    etl: int = 32,
    te: float | None = 100e-3,
    tr: float | None = 1000e-3,
    ordering: str = "radial_adaptive",
    variable_flip: bool = True,
    alpha_min_deg: float = 60.0,
    alpha_center_deg: float = 100.0,
    alpha_max_deg: float = 160.0,
    readout_bandwidth_hz: float = 250e3,
    acceleration: int = 1,
    acceleration_z: int = 1,
    caipi_shift: int = 0,
    elliptical: bool = True,
    n_acs: int = 24,
    n_acs_z: int = 16,
    n_dummy: int = 0,
    shuffle_seed: int = 0,
    n_gain_calibration_readouts: int = 1,
    crusher_cycles: float = 4.0,
    readout_crusher_cycles: float = 0.0,
) -> pp.Sequence:
    """Create a 3D Cartesian fast spin-echo sequence.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'fse_3d.seq'.
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
    etl : int, optional
        Echo train length: views per excitation. Default is 32.
    te : float or None, optional
        Effective echo time in seconds, rounded onto the echo grid. Ignored
        by the ``radial`` ordering (centre-out puts it at the first echo)
        and meaningless under ``shuffling``. None is the first echo.
        Default is 100e-3.
    tr : float or None, optional
        Repetition time in seconds, one per train. Default is 1000e-3.
    ordering : str, optional
        One of ``linear``, ``radial``, ``radial_adaptive``, ``shuffling``.
        Default is 'radial_adaptive'.
    variable_flip : bool, optional
        Play the TRAPS-style refocusing train of
        :func:`traps_flip_schedule` rather than constant 180s. Default is
        True.
    alpha_min_deg, alpha_center_deg, alpha_max_deg : float, optional
        The variable train's control points. Defaults are 60, 100 and 160.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth in Hz. Default is 250e3.
    acceleration : int, optional
        Uniform phase-encode undersampling factor along y. Default is 1.
    acceleration_z : int, optional
        Uniform partition-encode undersampling factor along z. Default is 1.
    caipi_shift : int, optional
        CAIPIRINHA shift along kz per sampled-ky block, for the regular
        (non-shuffling) orderings. ``0`` is a plain lattice. Ignored by
        ``shuffling``, which draws a Poisson-disc set instead. Default is 0.
    elliptical : bool, optional
        Restrict the phase-encode support to the inscribed ky-kz ellipse,
        dropping the corners a round object never fills. Default is True.
    n_acs : int, optional
        Autocalibration extent along y, in lines. Default is 24.
    n_acs_z : int, optional
        Autocalibration extent along z, in partitions. Default is 16.
    n_dummy : int, optional
        Trains played without acquiring before the first. Default is 0.
    shuffle_seed : int, optional
        Seed of the shuffling permutation. Default is 0.
    n_gain_calibration_readouts : int, optional
        Written as the ``NumGainCalibrationReadouts`` definition. Default
        is 1.
    crusher_cycles : float, optional
        Cycles of dephasing each crusher beside every refocusing pulse
        winds. Default is 4.0.
    readout_crusher_cycles : float, optional
        Read-axis crushing each side of every acquisition. Default is 0.0.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The FSE sequence object.
    """
    system = pp.Opts() if system is None else system
    system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)

    kernel = FSE3DKernel(
        system,
        fov=fov,
        n_x=n_x,
        n_y=n_y,
        n_z=n_z,
        slab_thickness=slab_thickness,
        etl=etl,
        te=te,
        tr=tr,
        ordering=ordering,
        variable_flip=variable_flip,
        alpha_min_deg=alpha_min_deg,
        alpha_center_deg=alpha_center_deg,
        alpha_max_deg=alpha_max_deg,
        readout_bandwidth_hz=readout_bandwidth_hz,
        acceleration=acceleration,
        acceleration_z=acceleration_z,
        caipi_shift=caipi_shift,
        elliptical=elliptical,
        n_acs=n_acs,
        n_acs_z=n_acs_z,
        n_dummy=n_dummy,
        shuffle_seed=shuffle_seed,
        crusher_cycles=crusher_cycles,
        readout_crusher_cycles=readout_crusher_cycles,
    )
    fse = kernel.readout
    timing = kernel.timing
    fov_x, fov_y = kernel.fov
    flips = kernel.flip_schedule_deg

    acs_y = range(max(0, n_y // 2 - n_acs // 2), min(n_y, n_y // 2 + -(-n_acs // 2)))
    acs_z = range(
        max(0, n_z // 2 - n_acs_z // 2), min(n_z, n_z // 2 + -(-n_acs_z // 2))
    )

    seq = pp.Sequence(system)
    lin_label, par_label, ima_label, seg_label, eco_label = fse.adc_labels
    seg_label.value = 0
    nominal = fse.rf_ref.amplitude

    def train(views, acquire: bool, mark=None) -> None:
        """Play one train, acquiring or not."""
        seq.add_block(fse.rf, fse.gz, *([mark] if mark is not None else []))
        seq.add_block(fse.gx_pre)
        for echo in range(etl):
            view = views[echo]
            fse.rf_ref.amplitude = nominal * np.deg2rad(flips[echo]) / np.pi
            seq.add_block(fse.rf_ref, fse.gz_ref)
            if echo == 0 and fse.esp_first > fse.esp:
                seq.add_block(fse.wait_esp1)
            if view is None:
                ky, kz = 0.0, 0.0
            else:
                line, partition = view
                ky = (line - n_y / 2) / (n_y / 2)
                kz = (partition - n_z / 2) / (n_z / 2)
            seq.add_block(
                fse.gx_bridge_pre,
                pp.scale_grad(fse.gy_pre, ky),
                pp.scale_grad(fse.gz_pre, kz),
            )
            if acquire and view is not None:
                line, partition = view
                lin_label.value = line
                par_label.value = partition
                ima_label.value = int(line in acs_y and partition in acs_z)
                eco_label.value = echo
                seq.add_block(fse.gx, fse.adc, *fse.adc_labels)
            else:
                seq.add_block(fse.gx)
            seq.add_block(
                fse.gx_bridge_post,
                pp.scale_grad(fse.gy_rew, ky),
                pp.scale_grad(fse.gz_rew, kz),
            )
        if timing.wait_tr is not None:
            seq.add_block(timing.wait_tr)
        fse.rf_ref.amplitude = nominal

    for i_dummy in range(n_dummy):
        train(
            [None] * etl,
            acquire=False,
            mark=pp.make_label("ONCE", "SET", 1) if i_dummy == 0 else None,
        )

    clear_once = pp.make_label("ONCE", "SET", 0) if n_dummy else None
    for views in kernel.trains:
        train(views, acquire=True, mark=clear_once)
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
    seq.set_definition(key="Name", value="fse_3d")
    seq.set_definition(key="TE", value=kernel.echo_time)
    seq.set_definition(key="TR", value=kernel.repetition_time)
    seq.set_definition(key="EchoSpacing", value=kernel.echo_spacing)
    seq.set_definition(key="EchoTrainLength", value=etl)
    seq.set_definition(key="ViewOrdering", value=ordering)
    seq.set_definition(key="RefocusingFlipAngles", value=list(flips))
    seq.set_definition(
        key="NumGainCalibrationReadouts", value=n_gain_calibration_readouts
    )

    seq.set_definition(key="kSpaceCenterLine", value=n_y // 2)
    seq.set_definition(key="kSpaceCenterPartition", value=n_z // 2)
    seq.set_definition(key="kSpaceCenterSample", value=kernel.readout.center_sample)
    seq.set_definition(key="SliceThickness", value=kernel.excitation.slice_thickness)

    if write_seq:
        write_sequence(seq, seq_filename, offline=True)

    return seq


# ======================================================================
# Subroutines of main()
# ======================================================================


def traps_flip_schedule(
    etl: int,
    n_center: int,
    *,
    alpha_min_deg: float = 60.0,
    alpha_center_deg: float = 100.0,
    alpha_max_deg: float = 160.0,
    n_down: int = 6,
) -> np.ndarray:
    """A TRAPS-style refocusing flip train, in degrees.

    Piecewise-linear in flip space between the four control points Busse's
    method exposes: ``alpha_max`` at the first echo, ``alpha_min`` from echo
    ``n_down``, ``alpha_center`` at the effective-TE echo, and ``alpha_max``
    again at the train's end. Loose by design: the reference derives the
    ramps from a prescribed signal envelope through EPG, and this keeps its
    control points and its shape without the inversion.

    Parameters
    ----------
    etl : int
        Echo train length.
    n_center : int
        Zero-based echo index of the k-space centre.
    alpha_min_deg : float, optional
        Floor of the ramp-down, reached at echo ``n_down``. Default is 60.
    alpha_center_deg : float, optional
        Flip at the centre-of-k-space echo. Default is 100.
    alpha_max_deg : float, optional
        Flip at the first and last echoes. Default is 160.
    n_down : int, optional
        Echo (zero-based) where the initial ramp-down bottoms out. Default
        is 6.

    Returns
    -------
    numpy.ndarray
        One flip per echo, degrees.
    """
    if etl < 1:
        raise ValueError("etl must be >= 1")
    control_echoes = [0]
    control_flips = [alpha_max_deg]
    bottom = min(max(1, n_down), max(1, etl - 1))
    if bottom > 0 and bottom < etl:
        control_echoes.append(bottom)
        control_flips.append(alpha_min_deg)
    if n_center > bottom:
        control_echoes.append(n_center)
        control_flips.append(alpha_center_deg)
    if etl - 1 > max(control_echoes):
        control_echoes.append(etl - 1)
        control_flips.append(alpha_max_deg)
    return np.interp(np.arange(etl), control_echoes, control_flips)


def order_views(
    views: list[tuple[int, int]],
    etl: int,
    n_center: int,
    ordering: str,
    grid: tuple[int, int],
    *,
    seed: int = 0,
) -> list[list[tuple[int, int] | None]]:
    """Deal ``(line, partition)`` views into trains, one view per echo.

    Parameters
    ----------
    views : list of tuple
        The views the sampling plan asks for.
    etl : int
        Echo train length.
    n_center : int
        Zero-based echo index the k-space centre should be acquired at.
    ordering : str
        One of :data:`ORDERINGS`.
    grid : tuple of int
        ``(n_y, n_z)``, locating the centre of k-space.
    seed : int, optional
        Seed of the shuffling permutation, so a scan is reproducible.

    Returns
    -------
    list of list
        One list per train, indexed by echo; ``None`` pads echoes with
        nothing left to encode.
    """
    if ordering not in ORDERINGS:
        raise ValueError(f"ordering must be one of {ORDERINGS}, got {ordering!r}")
    n_y, n_z = grid
    # The ordering builtins rank on the coordinates they are given, so pass the
    # views centred on the k-space middle and normalised by the matrix -- what
    # makes the radius isotropic in fractional k-space -- and index the shots of
    # positions they return back into the original ``(line, partition)`` views.
    coords = [
        ((line - n_y / 2) / n_y, (partition - n_z / 2) / n_z)
        for line, partition in views
    ]
    if ordering == "shuffling":
        shots = pp.make_shuffling_order(coords, etl, seed=seed, pad=True)
    elif ordering == "linear":
        shots = pp.make_linear_order(
            coords, etl, center=(0.0, 0.0), center_echo=n_center, pad=True
        )
    elif ordering == "centric":
        shots = pp.make_centric_order(
            coords, etl, center=(0.0, 0.0), center_echo=n_center, pad=True
        )
    elif ordering == "radial":
        shots = pp.make_radial_order(coords, etl, center=(0.0, 0.0), pad=True)
    else:  # radial_adaptive
        shots = pp.make_radial_adaptive_order(
            coords, etl, center=(0.0, 0.0), center_echo=n_center, pad=True
        )
    return [
        [views[index] if index is not None else None for index in shot]
        for shot in shots
    ]


def FSE3DKernel(
    system: pp.Opts,
    *,
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 128,
    n_y: int = 128,
    n_z: int = 64,
    slab_thickness: float = 128e-3,
    etl: int = 32,
    te: float | None = 100e-3,
    tr: float | None = 1000e-3,
    ordering: str = "radial_adaptive",
    variable_flip: bool = True,
    alpha_min_deg: float = 60.0,
    alpha_center_deg: float = 100.0,
    alpha_max_deg: float = 160.0,
    readout_bandwidth_hz: float = 250e3,
    acceleration: int = 1,
    acceleration_z: int = 1,
    caipi_shift: int = 0,
    elliptical: bool = True,
    n_acs: int = 24,
    n_acs_z: int = 16,
    n_dummy: int = 0,
    shuffle_seed: int = 0,
    crusher_cycles: float = 4.0,
    readout_crusher_cycles: float = 0.0,
) -> SimpleNamespace:
    """Design the train, its flip schedule, and the view order that fills it.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov, n_x, n_y, n_z, slab_thickness, etl, te, tr, ordering, \
variable_flip, alpha_min_deg, alpha_center_deg, alpha_max_deg, \
readout_bandwidth_hz, acceleration, acceleration_z, caipi_shift, elliptical, n_acs, n_acs_z, \
n_dummy, shuffle_seed, crusher_cycles, readout_crusher_cycles
        As for :func:`main`.

    Returns
    -------
    types.SimpleNamespace
        ``excitation``, ``refocusing``, ``readout``, ``timing``, ``fov``,
        ``trains`` (one view list per train, indexed by echo),
        ``n_center``, ``flip_schedule_deg``, ``echo_spacing``,
        ``echo_time``, ``repetition_time``, ``bandwidth_hz`` and
        ``duration``.
    """
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

    excitation = design.SpatialSelectiveExcitation(
        system,
        90.0,
        slab_thickness,
        duration_s=PULSE_DURATION,
        time_bw_product=TIME_BW_PRODUCT,
        is_slab=True,
    )
    refocusing = design.SpatialSelectiveRefocusing(
        system,
        slab_thickness,
        duration_s=PULSE_DURATION,
        time_bw_product=TIME_BW_PRODUCT,
        spoiling_cycles=crusher_cycles,
    )

    fse = design.FseReadout3D(
        system,
        excitation.rf,
        excitation.gz,
        rf_ref=refocusing.rf_ref,
        gz_ref=refocusing.gz,
        fov=(fov_x, fov_y, slab_thickness),
        matrix=(n_x, n_y, n_z),
        etl=etl,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=readout_crusher_cycles,
        labels=("LIN", "PAR", "IMA", "SEG", "ECO"),
    )
    esp = fse.esp

    if ordering == "radial" or te is None:
        n_center = 0
    else:
        n_center = int(np.clip(round(te / esp) - 1, 0, etl - 1))
    echo_time = float(fse.echo_times[n_center])

    length = fse.seq.duration()[0]
    wait_tr = None
    if tr is not None:
        if tr < length - 1e-9:
            raise ValueError(
                f"TR {tr * 1e3:.1f} ms is shorter than one train takes "
                f"({length * 1e3:.1f} ms)"
            )
        pad = pp.round_to_raster(tr - length, system.block_duration_raster)
        if pad > 0:
            wait_tr = pp.make_delay(pad)
            length += pad
    timing = SimpleNamespace(wait_tr=wait_tr, length=length)

    flips = (
        traps_flip_schedule(
            etl,
            n_center,
            alpha_min_deg=alpha_min_deg,
            alpha_center_deg=alpha_center_deg,
            alpha_max_deg=alpha_max_deg,
        )
        if variable_flip
        else np.full(etl, 180.0)
    )

    # Regular orderings sample the CAIPIRINHA lattice with its fully sampled
    # rectangle; shuffling draws an incoherent variable-density Poisson set for
    # a subspace reconstruction (Tamir et al., "T2 Shuffling"). At R = 1 both
    # are the full grid, and Poisson needs acceleration to have something to
    # thin, so shuffling falls back to the full grid there.
    if ordering == "shuffling":
        total_acceleration = acceleration * acceleration_z
        if total_acceleration > 1:
            mask = pp.make_poisson_disc_mask(
                (n_y, n_z),
                float(total_acceleration),
                calib=(n_acs, n_acs_z),
                seed=shuffle_seed,
            )
            views = [
                (int(line), int(partition)) for line, partition in np.argwhere(mask)
            ]
        else:
            views = [
                (line, partition) for partition in range(n_z) for line in range(n_y)
            ]
    else:
        views, _ = pp.calc_sampled_pairs(
            (n_y, n_z),
            (acceleration, acceleration_z),
            (n_acs, n_acs_z),
            caipi_shift=caipi_shift,
            elliptical=elliptical,
            order="ascending",
        )
    trains = order_views(views, etl, n_center, ordering, (n_y, n_z), seed=shuffle_seed)

    duration = (n_dummy + len(trains)) * timing.length

    return SimpleNamespace(
        excitation=excitation,
        refocusing=refocusing,
        readout=fse,
        timing=timing,
        fov=(fov_x, fov_y),
        trains=trains,
        n_center=n_center,
        flip_schedule_deg=flips,
        echo_spacing=esp,
        echo_time=echo_time,
        repetition_time=timing.length,
        bandwidth_hz=fse.bandwidth_hz,
        duration=duration,
    )


# ======================================================================
# The scanner protocol contract
# ======================================================================


class Fse3D(SequencePlugin):
    """The 3D fast spin echo behind the scanner protocol contract."""

    def get_default_protocol(self, system: pp.Opts) -> dict[str, dict]:
        """Return the protocol the scanner UI is built from."""
        return protocol_to_dict(
            {
                UIParam.TE: DropdownFloatParam(
                    value=100.0,
                    min=8.0,
                    max=600.0,
                    incr=1.0,
                    unit="ms",
                    options=[TEPreset.MINIMUM, 60.0, 100.0, 150.0, 250.0],
                ),
                UIParam.TR: DropdownFloatParam(
                    value=1000.0,
                    min=100.0,
                    max=5000.0,
                    incr=10.0,
                    unit="ms",
                    options=[TRPreset.MINIMUM, 600.0, 1000.0, 1500.0, 2500.0],
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
                    value=24.0, min=0.0, max=512.0, incr=1.0, unit="lines"
                ),
                UIParam.user_name(1): Description(text="Echo train length"),
                UIParam.user_value(1): TypeinFloatParam(
                    value=32.0, min=1.0, max=256.0, incr=1.0, unit=""
                ),
                UIParam.user_name(2): Description(
                    text="Order 0=lin 1=centric 2=rad 3=adaptive 4=shuffle"
                ),
                UIParam.user_value(2): TypeinFloatParam(
                    value=3.0, min=0.0, max=4.0, incr=1.0, unit=""
                ),
                UIParam.user_name(3): Description(text="Variable flip train"),
                UIParam.user_value(3): TypeinFloatParam(
                    value=1.0, min=0.0, max=1.0, incr=1.0, unit=""
                ),
                UIParam.user_name(4): Description(text="CAIPI shift (kz per ky)"),
                UIParam.user_value(4): TypeinFloatParam(
                    value=0.0, min=0.0, max=8.0, incr=1.0, unit=""
                ),
                UIParam.user_name(5): Description(text="ACS partitions (z)"),
                UIParam.user_value(5): TypeinFloatParam(
                    value=16.0, min=0.0, max=256.0, incr=1.0, unit="lines"
                ),
                UIParam.user_name(6): Description(text="Min/centre flips [deg]"),
                UIParam.user_value(6): TypeinFloatParam(
                    value=60.0, min=10.0, max=180.0, incr=1.0, unit="deg"
                ),
                UIParam.user_name(7): Description(text="Centre flip [deg]"),
                UIParam.user_value(7): TypeinFloatParam(
                    value=100.0, min=10.0, max=180.0, incr=1.0, unit="deg"
                ),
                UIParam.user_name(8): Description(text="Elliptical sampling"),
                UIParam.user_value(8): TypeinFloatParam(
                    value=1.0, min=0.0, max=1.0, incr=1.0, unit=""
                ),
            }
        )

    def validate_protocol(self, system: pp.Opts, protocol: dict[str, dict]) -> dict:
        """Report whether the protocol is feasible, and how long it will take."""
        system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
        kwargs = protocol_kwargs(system, protocol)
        try:
            kernel = FSE3DKernel(
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
                f"TA = {kernel.duration:.1f} s over {len(kernel.trains)} trains, "
                f"TEeff = {kernel.echo_time * 1e3:.1f} ms at echo "
                f"{kernel.n_center + 1}, ESP = {kernel.echo_spacing * 1e3:.2f} ms"
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
        "etl",
        "te",
        "tr",
        "ordering",
        "variable_flip",
        "alpha_min_deg",
        "alpha_center_deg",
        "alpha_max_deg",
        "readout_bandwidth_hz",
        "acceleration",
        "acceleration_z",
        "caipi_shift",
        "elliptical",
        "n_acs",
        "n_acs_z",
        "n_dummy",
        "shuffle_seed",
        "crusher_cycles",
        "readout_crusher_cycles",
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
        etl=max(1, round(params.user_float(prot, 1, 32.0))),
        ordering=ORDERINGS[
            int(np.clip(round(params.user_float(prot, 2, 3.0)), 0, len(ORDERINGS) - 1))
        ],
        variable_flip=bool(round(params.user_float(prot, 3, 1.0))),
        caipi_shift=max(0, round(params.user_float(prot, 4, 0.0))),
        n_acs=params.acs_lines_from_protocol(
            prot, params.param_int(prot, UIParam.NY), 0
        ),
        n_acs_z=max(0, round(params.user_float(prot, 5, 16.0))),
        alpha_min_deg=params.user_float(prot, 6, 60.0),
        alpha_center_deg=params.user_float(prot, 7, 100.0),
        elliptical=bool(round(params.user_float(prot, 8, 1.0))),
    )


PLUGIN = Fse3D()


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
    ("--te-ms", UIParam.TE, float, "Effective echo time [ms], or a negative TEPreset"),
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
    ("--offset-x-mm", UIParam.FOV_OFFSET_X, float, "Volume offset along readout [mm]"),
    (
        "--offset-y-mm",
        UIParam.FOV_OFFSET_Y,
        float,
        "Volume offset along phase encode [mm]",
    ),
    ("--offset-z-mm", UIParam.FOV_OFFSET_Z, float, "Volume offset along slab [mm]"),
    ("--acs-lines", UIParam.user_value(0), float, "Number of ACS lines along y"),
    ("--etl", UIParam.user_value(1), float, "Echo train length"),
    (
        "--ordering",
        UIParam.user_value(2),
        float,
        "View ordering: 0 linear, 1 centric, 2 radial, 3 radial-adaptive, 4 shuffling",
    ),
    (
        "--constant-flip",
        UIParam.user_value(3),
        lambda value: 0.0 if float(value) else 1.0,
        "Pass 1 for constant 180s (0, the default value, keeps the variable train)",
    ),
    ("--caipi-shift", UIParam.user_value(4), float, "CAIPIRINHA kz shift per ky block"),
    (
        "--acs-partitions",
        UIParam.user_value(5),
        float,
        "Number of ACS partitions along z",
    ),
    (
        "--alpha-min-deg",
        UIParam.user_value(6),
        float,
        "Variable train's floor flip [deg]",
    ),
    (
        "--alpha-centre-deg",
        UIParam.user_value(7),
        float,
        "Variable train's centre flip [deg]",
    ),
    (
        "--no-elliptical",
        UIParam.user_value(8),
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
            description="Generate a 3D Cartesian fast spin-echo .seq offline.",
            default_output="fse_3d.seq",
        )
    )
