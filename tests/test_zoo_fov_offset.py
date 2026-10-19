"""Every shipped sequence, in every parameter set, prescribed off the isocentre along each axis.

A spin at ``r + d`` under the prescription to ``d`` sees what a spin at ``r``
sees under the design when every pulse and every receiver sample is played at
the phase ``2 pi m(t) . d``, ``m(t)`` the gradient moment from the start of the
chain: the frame rotating at that phase cancels the precession ``G . d`` the
offset adds. That phase, up to one constant, is checked here at every RF and
ADC sample the converted cache plays, the proxy's correction of the readout
included, against ``m(t)`` integrated from the designed gradients. It moves
the slice or slab (the RF frequency under the selection gradient) and every
phase-encoded and non-Cartesian readout by exactly ``d``.
"""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest
from _host import LIMITS, SMALL, SMALLER

from pulserver import _plugins, ir, virtual
from pulserver.design import load_plugin, read_options
from pulserver.mrd._sequence import read_chain
from pulserver.proxy import SequenceTable

SYSTEM = pp.Opts(**LIMITS)
PARAMETERS = _plugins.SEQUENCES.parent / "parameters"
#: The parameter sets whose refocusing trains blochsim designs.
WITH_BLOCHSIM = {("fse3d", "optimized"), ("fse3d", "dual_region")}
#: Prescriptions that excite more than one slice: in a slice group, and two
#: slices at once.
SLICES = {"gre2d": {"nslices": 3}, "epi2d": {"nslices": 4, "multiband": 2}}
VARIANTS = (
    [(path.stem, None) for path in sorted(_plugins.SEQUENCES.glob("*.py"))]
    + [(name, None, changes) for name, changes in SLICES.items()]
    + [
        pytest.param(
            (toml.parent.name, toml.stem),
            marks=pytest.mark.skipif(
                (toml.parent.name, toml.stem) in WITH_BLOCHSIM
                and importlib.util.find_spec("blochsim") is None,
                reason="needs blochsim",
            ),
        )
        for toml in sorted(PARAMETERS.glob("*/*.toml"))
    ]
)
#: An offset along the logical readout, phase and slice axes, in metres.
OFFSETS = {
    "read": (0.013, 0.0, 0.0),
    "phase": (0.0, -0.011, 0.0),
    "slice": (0.0, 0.0, 0.007),
}
# In turns. The cache holds offsets and waveforms in single precision, and the
# phase of a sample is reached from offsets of up to a few hundred turns.
TOLERANCE = 5e-4


def _wire(plugin, changes):
    wire = {
        key: entry.value for key, entry in plugin.listing().items() if entry.editable
    }
    for key, value in changes.items():
        if value is None:
            wire.pop(key, None)
        else:
            wire[key] = value
    return wire


def _moment(chain):
    """The gradient moment ``m(t)`` of a chain, in 1/m, along x, y and z, from its start."""
    pieces, start = [[], [], []], 0.0
    for sequence in chain:
        waves = sequence.waveforms_and_times(compat=False).waveforms
        for axis, corners in enumerate((waves.gx, waves.gy, waves.gz)):
            if corners.size:
                pieces[axis].append(np.vstack([corners[0] + start, corners[1]]))
        start += float(np.sum(sequence._native.block_durations()))
    corners = [
        np.concatenate(axis, axis=1) if axis else np.array([[0.0, start], [0.0, 0.0]])
        for axis in pieces
    ]

    def moment(t):
        t = np.asarray(t, dtype=float)
        swept = np.zeros((3, t.size))
        for axis, (when, amplitude) in enumerate(corners):
            step = np.diff(when)
            area = np.concatenate(
                [[0.0], np.cumsum(0.5 * (amplitude[1:] + amplitude[:-1]) * step)]
            )
            i = np.clip(np.searchsorted(when, t, side="right") - 1, 0, when.size - 2)
            tau = np.clip(t - when[i], 0.0, step[i])
            slope = np.divide(
                amplitude[i + 1] - amplitude[i],
                step[i],
                out=np.zeros(t.size),
                where=step[i] > 0,
            )
            swept[axis] = np.where(
                t < when[0], 0.0, area[i] + amplitude[i] * tau + 0.5 * slope * tau**2
            )
        return swept

    return moment


def _positioned(chain):
    """Whether each block of a chain, by start time, follows the prescription: not ``NOPOS``."""
    starts, follows, start = [], [], 0.0
    for sequence in chain:
        durations = np.asarray(sequence._native.block_durations())
        starts.append(start + np.concatenate([[0.0], np.cumsum(durations)[:-1]]))
        flags = sequence.evaluate_labels(evolution="blocks").get("NOPOS")
        exempt = np.zeros(durations.size, dtype=bool)
        if flags is not None:
            flags = np.atleast_1d(flags)[: durations.size]
            exempt[: flags.size] = flags != 0
        follows.append(~exempt)
        start += float(durations.sum())
    starts, follows = np.concatenate(starts), np.concatenate(follows)
    return lambda t: follows[
        np.clip(np.searchsorted(starts, t, side="right") - 1, 0, None)
    ]


def _played(first, offset, directory):
    """The pulses and receiver phases the cache converted at ``offset`` plays, with the proxy's correction added."""
    ir.convert(first, SYSTEM, fov_offset=offset)
    target = directory / "played.seq"
    receiver = virtual.export(first, target, SYSTEM)
    table = SequenceTable.read(first, fov_offset_m=offset)
    for index, phase in enumerate(receiver):
        correction = table.readout_phase_modulation(index)
        if correction is not None:
            receiver[index] = phase + correction
    played = pp.Sequence(SYSTEM)
    played.read(str(target))
    with warnings.catch_warnings():
        # A gradient carried across a block boundary ends its block off zero.
        warnings.filterwarnings("ignore", "waveforms_and_times", UserWarning)
        expanded = played.waveforms_and_times(append_RF=True, compat=False)
    receiver = np.concatenate(receiver) if receiver else np.zeros(0)
    return expanded.waveforms.rf, expanded.adc.t, receiver


def _designed(name, parameters, directory, changes=None):
    """A small design of a shipped sequence in a parameter set, with the wire values ``changes``: its first file, moment, positioned blocks and what it plays at the isocentre."""
    options = (
        None
        if parameters is None
        else read_options(PARAMETERS / name / f"{parameters}.toml")
    )
    plugin = load_plugin(_plugins.SEQUENCES / f"{name}.py", options)
    validation, paths = plugin.design(
        SYSTEM,
        _wire(plugin, {**SMALL, **SMALLER.get(name, {}), **(changes or {})}),
        directory,
    )
    assert paths, validation
    first = Path(paths[0])
    chain = [sequence for _, sequence in read_chain(first)]
    return (
        first,
        _moment(chain),
        _positioned(chain),
        _played(first, (0.0,) * 3, directory),
    )


def _id(variant):
    name, parameters, *changes = variant
    if changes:
        return f"{name}-" + "-".join(f"{k}{v}" for k, v in changes[0].items())
    return name if parameters is None else f"{name}+{parameters}"


@pytest.fixture(scope="module", params=VARIANTS, ids=_id)
def design(request, tmp_path_factory):
    name, parameters, *changes = request.param
    return _designed(name, parameters, tmp_path_factory.mktemp(name), *changes)


def _misplacement(design, offset, directory, judged_at=None):
    """The largest departure, in turns, of an RF or ADC sample of the cache converted at ``offset`` from ``m(t) . judged_at`` plus one constant.

    ``judged_at`` is ``offset`` unless given.
    """
    first, moment, positioned, (pulse, sampled, receiver) = design
    moved_pulse, moved_sampled, moved_receiver = _played(first, offset, directory)
    np.testing.assert_array_equal(moved_sampled, sampled)
    shift = np.asarray(offset if judged_at is None else judged_at)
    departures = [(moved_receiver - receiver) / (2 * np.pi) - shift @ moment(sampled)]
    if pulse is not None and pulse.size:
        when = np.real(pulse[0])
        on = (np.abs(pulse[1]) > 1e-6 * np.abs(pulse[1]).max()) & positioned(when)
        turned = np.angle(moved_pulse[1][on] * np.conj(pulse[1][on])) / (2 * np.pi)
        departures.append(turned - shift @ moment(when[on]))
    departure = np.concatenate(departures)
    constant = np.angle(np.mean(np.exp(2j * np.pi * departure))) / (2 * np.pi)
    return float(np.abs((departure - constant + 0.5) % 1.0 - 0.5).max())


@pytest.mark.parametrize("axis", OFFSETS)
def test_every_shipped_sequence_plays_every_pulse_and_sample_where_its_offset_moves_it(
    design, axis, tmp_path
):
    assert _misplacement(design, OFFSETS[axis], tmp_path) < TOLERANCE


@pytest.mark.parametrize("axis", OFFSETS)
def test_a_scan_converted_at_another_offset_is_found_misplaced(axis, tmp_path):
    design = _designed("gre2d", None, tmp_path)
    assert _misplacement(design, (0.0,) * 3, tmp_path, OFFSETS[axis]) > 0.1
