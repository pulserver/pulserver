"""The flags each block plays and the labels each readout records: those in force there."""

import pypulseqpp as pp
import pytest

from pulserver import ir

SYSTEM = pp.Opts(B0=3.0)
# LIN, SLC and ECO: the three label columns a GE interpreter records per readout.
GE_LABELS = (8, 0, 6)


@pytest.fixture
def labelled(tmp_path):
    """Four repetitions of an excitation and two readouts, under changing labels."""
    sequence = pp.Sequence(pp.Opts())
    rf = pp.make_sinc_pulse(flip_angle=0.1, duration=1e-3)
    adc = pp.make_adc(num_samples=64, duration=1e-3, delay=1e-4)
    for rep in range(4):
        sequence.add_block(
            rf,
            pp.make_label(label="TRID", type="SET", value=1 + rep // 2),
            pp.make_label(label="NOROT", type="SET", value=rep % 2),
        )
        sequence.add_block(
            adc,
            pp.make_label(label="LIN", type="SET", value=rep),
            pp.make_label(label="ECO", type="SET", value=0),
        )
        sequence.add_block(
            adc,
            pp.make_label(label="ECO", type="INC", value=1),
            pp.make_label(label="NOPOS", type="SET", value=int(rep == 2)),
        )
        sequence.add_block(
            pp.make_delay(1e-3), pp.make_label(label="SLC", type="INC", value=1)
        )
    path = tmp_path / "labelled.seq"
    sequence.write(path)
    return path


def test_the_scanner_plays_each_block_under_the_flags_in_force_at_it(labelled):
    ir.convert(labelled, SYSTEM)
    played = ir.play(labelled)
    assert played["norot"].tolist() == [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4
    assert played["nopos"].tolist() == [0] * 10 + [1] * 4 + [0] * 2
    assert played["trid"].tolist() == [1] * 8 + [2] * 8


def test_each_readout_records_the_labels_in_force_at_it(labelled):
    """A slice counted up after a repetition's readouts is recorded from the next."""
    (unit,) = ir.summary(labelled, SYSTEM, label_column_map=GE_LABELS)["subsequences"]
    assert unit["readout_labels"] == [
        [line, line, echo] for line in range(4) for echo in (0, 1)
    ]
