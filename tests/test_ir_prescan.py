"""The receive-gain prescan a conversion plays: whole repetitions of its sequence."""

import importlib.util

import numpy as np
import pypulseqpp as pp
import pytest
from test_zoo_evaluation import CHANGED, OPTIMIZED, SCANS, SHIPPED, SYSTEM, TOGGLED

from pulserver import _plugins
from pulserver.design import Protocol, load_plugin
from pulserver.ir._source import conversion_payload


def _double_echoes(repetitions, stated=None, exempt=0):
    """A sequence of ``repetitions`` excitations, each followed by two readouts
    and ``exempt`` more that NOPRESCAN keeps out of the prescan."""
    adc = pp.make_adc(num_samples=64, duration=3.2e-3, system=SYSTEM)
    skip = pp.make_label(type="SET", label="NOPRESCAN", value=1)
    keep = pp.make_label(type="SET", label="NOPRESCAN", value=0)
    seq = pp.Sequence(SYSTEM)
    for _ in range(repetitions):
        seq.add_block(pp.make_block_pulse(0.1, duration=1e-3, system=SYSTEM))
        seq.add_block(adc, keep)
        seq.add_block(adc)
        for _ in range(exempt):
            seq.add_block(adc, skip)
    if stated is not None:
        seq.set_definition(key="NumGainCalibrationReadouts", value=stated)
    return seq


def _prescan(seq):
    return conversion_payload(seq, SYSTEM)["reserved"]["num_gain_cal_readouts"]


def test_a_stated_count_that_ends_with_a_repetition_is_played():
    assert _prescan(_double_echoes(4, stated=4)) == 4


def test_a_sequence_that_states_no_count_plays_one_repetition():
    assert _prescan(_double_echoes(4)) == 2


def test_a_count_that_ends_inside_a_repetition_is_refused_naming_the_repetition():
    with pytest.raises(ValueError, match=r"ends inside a repetition.* holds 2"):
        _prescan(_double_echoes(4, stated=3))


def test_a_count_takes_only_the_readouts_the_prescan_acquires():
    assert _prescan(_double_echoes(4, stated=4, exempt=1)) == 4
    assert _prescan(_double_echoes(4, exempt=1)) == 2


def test_a_count_that_includes_exempt_readouts_is_refused():
    with pytest.raises(ValueError, match="NumGainCalibrationReadouts=3 ends inside"):
        _prescan(_double_echoes(4, stated=3, exempt=1))


def test_a_count_of_no_readout_is_refused():
    with pytest.raises(ValueError, match="NumGainCalibrationReadouts=0 ends inside"):
        _prescan(_double_echoes(4, stated=0))


def _variants():
    """Every shipped and toggled sequence at a small protocol, and the scans that change its repetitions."""
    scans = [
        (key, CHANGED[key.partition("+")[0]], "small") for key in [*SHIPPED, *TOGGLED]
    ]
    scans += [(s.values[0], s.values[1], s.id) for s in SCANS if s.values[1]]
    for key, changes, label in scans:
        name, options = TOGGLED.get(key, (key, {}))
        marks = []
        if key in OPTIMIZED and importlib.util.find_spec("blochsim") is None:
            marks = [pytest.mark.skip(reason="blochsim is not installed")]
        yield pytest.param(name, options, changes, id=f"{key}-{label}", marks=marks)


@pytest.mark.parametrize(("name", "options", "changes"), list(_variants()))
def test_every_zoo_file_with_a_readout_states_a_prescan_of_whole_repetitions(
    tmp_path, name, options, changes
):
    plugin = load_plugin(_plugins.SEQUENCES / f"{name}.py", options)
    wire = {k: e.value for k, e in plugin.listing().items() if e.editable}
    protocol = Protocol.from_wire(plugin.protocol, {**wire, **changes}, SYSTEM)
    designed = plugin.generate(SYSTEM, protocol)
    chain = [designed] if isinstance(designed, pp.Sequence) else designed
    for index, seq in enumerate(chain):
        seq.write(str(tmp_path / f"{index}.seq"))
        written = pp.io.read(str(tmp_path / f"{index}.seq"))
        acquires = any(row[4] for row in written._native.block_events())
        stated = written.definitions.get("NumGainCalibrationReadouts")

        assert (stated is not None) == acquires
        if acquires:
            assert _prescan(written) == int(np.atleast_1d(stated)[0])
