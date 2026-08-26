"""3D zero-echo-time imaging: a continuous-gradient koosh-ball of spokes.

``main`` returns the :class:`pulserver.pypulseq.Sequence`; ``PLUGIN`` is the
same sequence behind the scanner protocol contract, and running this module
as a script writes a ``.seq`` from the same controls.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pulserver.design as design
import pulserver.pypulseq as pp
from pulserver.design import (
    Description,
    DropdownFloatParam,
    DropdownIntParam,
    OffFloatParam,
    SequencePlugin,
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
    seq_filename: str = "zte_3d.seq",
    *,
    system: pp.Opts | None = None,
    fov: float = 220e-3,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int = 128,
    n_views: int | None = None,
    n_shots: int | None = None,
    flip_angle_deg: float = 3.0,
    pulse_duration: float = 10e-6,
    readout_bandwidth_hz: float = 250e3,
    n_dummy: int = 2,
    n_gain_calibration_readouts: int = 1,
) -> pp.Sequence:
    """Create a 3D zero-echo-time sequence.

    The auto-SSP-placement stress case: the readout gradient is at full
    amplitude when the hard pulse fires and never returns to zero inside a
    shell -- :class:`design.ZteReadout`, which writes the whole shell out as one
    continuous waveform, view by view, ramping up once at the north pole and
    down once at the south. Each shot replays it turned about ``z``, so the
    sphere costs one shell's worth of waveform memory. The interpreter has almost
    no dead time to hide its SSP packets in, which is exactly what this slot
    exercises.
    The centre of k-space is not acquired -- ``n_missing`` samples fall in the
    dead-time gap and are declared in the definitions for the reconstruction to
    handle. :mod:`pulserver.app.noncartesian3D_recon` reconstructs the sphere by 3D NUFFT
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
        Output filename for the .seq file. Default is 'zte_3d.seq'.
    system : pypulseq.Opts, optional
        System limits. Default is `pp.Opts()`.
    fov : float, optional
        Isotropic field of view in meters. Default is 220e-3.
    fov_offset : tuple of float, optional
        Where the prescribed volume sits, in meters. Applied in server mode:
        every readout is rotated, so all of them defer their ADC shift to
        the consumer. Default is (0.0, 0.0, 0.0).
    n_x : int, optional
        Isotropic matrix size. Default is 128.
    n_views : int or None, optional
        Views per shell. None is the Nyquist-matched count. Default is None.
    n_shots : int or None, optional
        Shells the sphere is dealt into. None is the module's Nyquist
        default. Default is None.
    flip_angle_deg : float, optional
        Hard-pulse flip angle in degrees; small, because the pulse plays on
        a gradient at full amplitude. Default is 3.0.
    pulse_duration : float, optional
        Hard-pulse duration in seconds. Default is 10e-6.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth in Hz. Default is 250e3.
    n_dummy : int, optional
        Whole shells played without acquiring before the first acquired one,
        so the first acquired spoke sees the magnetisation every later spoke
        sees. A dummy shell is an acquiring shell with the ADC left off --
        same pulses, same trajectory, same duration. Default is 2.
    n_gain_calibration_readouts : int, optional
        Written as the ``NumGainCalibrationReadouts`` definition. Default
        is 1.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The ZTE sequence object.

    Examples
    --------
    >>> from pulserver.app import zte3D_sequence
    >>> seq = zte3D_sequence(n_x=32, n_views=24, n_shots=2, n_dummy=0)
    >>> seq.num_trs, seq.num_segments
    (2, 1)

    The waveform figures below are one design, prescribed to be *legible*
    rather than diagnostic: a long readout, so the flat top dominates; and
    sixteen views over four shots, so one shell segment fits on a page.

    .. plot::
       :include-source:
       :nofigs:
       :context:

       from pulserver.app import zte3D_sequence

       seq = zte3D_sequence(n_x=256, n_views=16, n_shots=4, n_dummy=0)

    **The excitation**, and the magnetisation it leaves behind. The pulse is
    hard and only a few microseconds long, and the readout gradient is already
    on while it plays -- so the profile is drawn against position, and it is
    flat across a span far wider than anything the scan images.

    .. plot::
       :include-source:
       :context: close-figs

       seq.plot_rf(plot_now=False)

    **One shot**: the gradient is already on when each pulse plays, so a
    view starts at the centre of k-space and runs outward, and only the
    small step between successive directions is ramped.

    .. plot::
       :include-source:
       :context: close-figs

       seq.plot(tr="worst_case", time_disp="ms", grad_disp="mT/m", plot_now=False)

    **What the scan covers**: radial spokes from the centre outward,
    coloured by the order the shots were played in.

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
    readout steps ramp as fast as they are allowed to. A lower ``MAX_SLEW``, a
    longer echo time, or a narrower readout bandwidth each bring the response
    down.

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

    kernel = ZteKernel(
        system,
        fov=fov,
        n_x=n_x,
        n_views=n_views,
        n_shots=n_shots,
        flip_angle_deg=flip_angle_deg,
        pulse_duration=pulse_duration,
        readout_bandwidth_hz=readout_bandwidth_hz,
    )
    zte = kernel.readout

    seq = pp.Sequence(system)
    lin_label, seg_label = zte.adc_labels

    # Steady state is reached once, at the head of the first shell: the pulse
    # train never stops, so later shells inherit it.
    #
    # `ONCE` is what keeps the dummies out of the averages: 1 plays on the
    # first pass only, and the first acquired view clears it back to 0 so the
    # body repeats.
    clear_once = None
    for shot_index, shot in enumerate(zte.shot_rotations):
        seg_label.value = int(shot_index)
        # One rotation for the whole shell. The shell itself is written out
        # spoke by spoke, so this only says where the shot samples.
        turn = pp.make_rotation(Rotation.from_matrix(shot))
        if shot_index == 0:
            # A dummy is a whole shell, played exactly as an acquiring one but
            # without the ADC. Structure is what the interpreter reads a
            # repetition from, and whether a block acquires is not part of it,
            # so a preparation shell partitions identically to a real one.
            for i_dummy in range(n_dummy):
                seq.add_block(*zte.g_ramp, turn)
                for view in range(len(zte.directions)):
                    mark = (
                        pp.make_label("ONCE", "SET", 1)
                        if i_dummy == 0 and view == 0
                        else None
                    )
                    seq.add_block(
                        zte.rf, *zte.g_hold[view], turn, *([mark] if mark else ())
                    )
                    seq.add_block(*zte.g_read[view], turn)
            clear_once = pp.make_label("ONCE", "SET", 0) if n_dummy else None
        seq.add_block(*zte.g_ramp, turn)
        for view in range(len(zte.directions)):
            lin_label.value = int(view)
            seq.add_block(
                zte.rf, *zte.g_hold[view], turn, *([clear_once] if clear_once else ())
            )
            clear_once = None
            seq.add_block(zte.adc, *zte.g_read[view], turn, lin_label, seg_label)

    pp.TransformFOV(
        translation=tuple(offset * 1e3 for offset in fov_offset),
        system=system,
        server_mode=True,
    ).apply_to_sequence(seq, in_place=True)

    if test_report:
        print(seq.test_report())

    if plot:
        seq.plot()

    seq.set_definition(key="FOV", value=[fov, fov, fov])
    seq.set_definition(key="Matrix", value=[n_x, n_x, n_x])
    seq.set_definition(key="Name", value="zte_3d")
    seq.set_definition(key="TE", value=0.0)
    seq.set_definition(key="TR", value=zte.tr)
    seq.set_definition(key="Trajectory", value="zte")
    seq.set_definition(key="NumShots", value=len(zte.shot_rotations))
    seq.set_definition(key="ViewsPerShot", value=len(zte.directions))
    seq.set_definition(key="MissingSamples", value=zte.n_missing)
    seq.set_definition(
        key="NumGainCalibrationReadouts", value=n_gain_calibration_readouts
    )

    if write_seq:
        write_sequence(seq, seq_filename, offline=True)

    return seq


# ======================================================================
# Subroutines of main()
# ======================================================================


def ZteKernel(
    system: pp.Opts,
    *,
    fov: float = 220e-3,
    n_x: int = 128,
    n_views: int | None = None,
    n_shots: int | None = None,
    flip_angle_deg: float = 3.0,
    pulse_duration: float = 10e-6,
    readout_bandwidth_hz: float = 250e3,
) -> SimpleNamespace:
    """Design the shell, and the plan that spins it.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov, n_x, n_views, n_shots, flip_angle_deg, pulse_duration, \
readout_bandwidth_hz
        As for :func:`main`.

    Returns
    -------
    types.SimpleNamespace
        ``excitation``, ``readout``, ``repetition_time``, ``bandwidth_hz``
        and ``duration``.
    """
    excitation = design.NonSelectiveExcitation(
        system, flip_angle_deg, duration_s=pulse_duration
    )
    kwargs = {}
    if n_views is not None:
        kwargs["n_views"] = int(n_views)
    if n_shots is not None:
        kwargs["n_shots"] = int(n_shots)
    readout = design.ZteReadout(
        system,
        excitation.rf,
        fov=fov,
        matrix=n_x,
        readout_bandwidth_hz=readout_bandwidth_hz,
        # LIN carries the shot: a projection is what a reconstruction grids
        # by, and SEG says which interleave of the shell it came from.
        labels=("LIN", "SEG"),
        **kwargs,
    )
    duration = len(readout.shot_rotations) * readout.duration

    return SimpleNamespace(
        excitation=excitation,
        readout=readout,
        repetition_time=readout.tr,
        bandwidth_hz=getattr(readout, "bandwidth_hz", readout_bandwidth_hz),
        duration=duration,
    )


# ======================================================================
# The scanner protocol contract
# ======================================================================


class Zte3D(SequencePlugin):
    """The 3D ZTE behind the scanner protocol contract."""

    def get_default_protocol(self, system: pp.Opts) -> dict[str, dict]:
        """Return the protocol the scanner UI is built from."""
        return protocol_to_dict(
            {
                UIParam.FLIP: DropdownFloatParam(
                    value=3.0,
                    min=0.5,
                    max=10.0,
                    incr=0.5,
                    unit="deg",
                    options=[1.0, 2.0, 3.0, 4.0, 5.0],
                ),
                UIParam.FOV: DropdownFloatParam(
                    value=220.0,
                    min=80.0,
                    max=500.0,
                    incr=1.0,
                    unit="mm",
                    options=[180.0, 220.0, 280.0, 340.0, 500.0],
                ),
                UIParam.NX: DropdownIntParam(
                    value=128, min=16, max=384, incr=1, options=[64, 96, 128, 192, 256]
                ),
                UIParam.BANDWIDTH: TypeinFloatParam(
                    value=250e3, min=25e3, max=1000e3, incr=1000.0, unit="Hz"
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
                UIParam.user_name(0): Description(text="Views per shell (0 = Nyquist)"),
                UIParam.user_value(0): TypeinFloatParam(
                    value=0.0, min=0.0, max=65536.0, incr=1.0, unit=""
                ),
                UIParam.user_name(1): Description(text="Shells (0 = Nyquist)"),
                UIParam.user_value(1): TypeinFloatParam(
                    value=0.0, min=0.0, max=4096.0, incr=1.0, unit=""
                ),
            }
        )

    def validate_protocol(self, system: pp.Opts, protocol: dict[str, dict]) -> dict:
        """Report whether the protocol is feasible, and how long it will take."""
        system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
        kwargs = protocol_kwargs(system, protocol)
        try:
            kernel = ZteKernel(
                system,
                **{
                    name: value
                    for name, value in kwargs.items()
                    if name in KERNEL_ARGUMENTS
                },
            )
        except ValueError as error:
            return {"valid": False, "duration": None, "info": str(error)}

        zte = kernel.readout
        return {
            "valid": True,
            "duration": kernel.duration,
            "info": (
                f"TA = {kernel.duration:.1f} s over "
                f"{len(zte.shot_rotations)} shells of "
                f"{len(zte.directions)} views, "
                f"{zte.n_missing} centre samples missing"
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
        "n_views",
        "n_shots",
        "flip_angle_deg",
        "pulse_duration",
        "readout_bandwidth_hz",
    )
)


def protocol_kwargs(system: pp.Opts, protocol: dict[str, dict]) -> dict:
    """The prescribed quantities, plus this sequence's own user slots."""
    prot = dict_to_protocol(protocol)
    views = round(params.user_float(prot, 0, 0.0))
    shots = round(params.user_float(prot, 1, 0.0))
    return main_kwargs(
        main,
        system,
        protocol,
        fov=params.param_float(prot, UIParam.FOV) * 1e-3,
        n_views=None if views <= 0 else views,
        n_shots=None if shots <= 0 else shots,
    )


PLUGIN = Zte3D()


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
    ("--flip-deg", UIParam.FLIP, float, "Hard-pulse flip angle [deg]"),
    ("--fov-mm", UIParam.FOV, float, "Isotropic FOV [mm]"),
    ("--nx", UIParam.NX, int, "Isotropic matrix size"),
    ("--bandwidth-hz", UIParam.BANDWIDTH, float, "Requested receiver bandwidth [Hz]"),
    ("--offset-x-mm", UIParam.FOV_OFFSET_X, float, "Volume offset along x [mm]"),
    ("--offset-y-mm", UIParam.FOV_OFFSET_Y, float, "Volume offset along y [mm]"),
    ("--offset-z-mm", UIParam.FOV_OFFSET_Z, float, "Volume offset along z [mm]"),
    ("--views", UIParam.user_value(0), float, "Views per shell (0 = Nyquist)"),
    ("--shells", UIParam.user_value(1), float, "Shells (0 = Nyquist)"),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=ARG_MAP,
            description="Generate a 3D zero-echo-time .seq offline.",
            default_output="zte_3d.seq",
        )
    )
