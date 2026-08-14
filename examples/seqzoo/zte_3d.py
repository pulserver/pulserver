"""3D zero-echo-time imaging: a continuous-gradient koosh-ball of spokes.

The auto-SSP-placement stress case: the readout gradient is at full
amplitude when the hard pulse fires and never returns to zero inside a
shell -- :class:`design.ZteReadout`, whose view is two blocks, pulse-and-
hold then acquire-and-turn, with consecutive views a constant angle apart so
one designed transition serves the lot. The interpreter has almost no dead
time to hide its SSP packets in, which is exactly what this slot exercises.
The centre of k-space is not acquired -- ``n_missing`` samples fall in the
dead-time gap and are declared in the definitions for the reconstruction to
handle. :mod:`pulserver.reczoo.zte_3d` reconstructs the sphere by 3D NUFFT
against the trajectory the acquisitions carry.

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
from pulserver import (
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
    n_gain_calibration_readouts: int = 1,
) -> pp.Sequence:
    """Create a 3D zero-echo-time sequence.

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
    n_gain_calibration_readouts : int, optional
        Written as the ``NumGainCalibrationReadouts`` definition. Default
        is 1.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The ZTE sequence object.
    """
    system = pp.Opts() if system is None else system

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
    seg_label = pp.make_label("SEG", "SET", 0)

    for shot_index, shot in enumerate(zte.shot_rotations):
        seg_label.value = int(shot_index)
        turns = [
            pp.make_rotation(Rotation.from_matrix(shot @ turn))
            for turn in zte.view_rotations
        ]
        seq.add_block(*zte.g_ramp, turns[0])
        for view, turn in enumerate(turns):
            last = view == len(turns) - 1
            seq.add_block(zte.rf, *zte.g_hold, turn)
            seq.add_block(
                zte.adc, *(zte.g_end if last else zte.g_read), turn, seg_label
            )

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
    seq.set_definition(key="ViewsPerShot", value=len(zte.view_rotations))
    seq.set_definition(key="MissingSamples", value=zte.n_missing)
    seq.set_definition(
        key="NumGainCalibrationReadouts", value=n_gain_calibration_readouts
    )

    if write_seq:
        write_sequence(seq, seq_filename, offline=True)

    return seq


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
        **kwargs,
    )
    n_total = len(readout.shot_rotations) * len(readout.view_rotations)
    duration = n_total * readout.tr

    return SimpleNamespace(
        excitation=excitation,
        readout=readout,
        repetition_time=readout.tr,
        bandwidth_hz=getattr(readout, "bandwidth_hz", readout_bandwidth_hz),
        duration=duration,
    )


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
                UIParam.FOV_OFFSET_X: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.FOV_OFFSET_Y: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.FOV_OFFSET_Z: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
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
        kwargs = _main_kwargs(system, protocol)
        try:
            kernel = ZteKernel(
                system,
                **{name: value for name, value in kwargs.items() if name in _KERNEL_ARGUMENTS},
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
                f"{len(zte.view_rotations)} views, "
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
        seq = main(**_main_kwargs(system, protocol))
        write_sequence(seq, output_path, offline=offline)


_KERNEL_ARGUMENTS = frozenset(
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


def _main_kwargs(system: pp.Opts, protocol: dict[str, dict]) -> dict:
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


_ARG_MAP = [
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
            arg_map=_ARG_MAP,
            description="Generate a 3D zero-echo-time .seq offline.",
            default_output="zte_3d.seq",
        )
    )
