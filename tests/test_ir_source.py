"""The libraries read through pypulseqpp against the file they were read from.

The text fixtures only. ``se_propeller_2d.bin`` is binary, and the oracle here
is a text file read section by section.
"""

from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest
from _seqtext import rf_use, sections, shapes, table
from pypulseqpp import _ext as core

from pulserver.ir._source import conversion_payload, sequence_libraries

FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
# The file writes its cells at six significant digits, so every comparison is
# to single precision against the largest value in hand.
SINGLE = 1e-6
# The use letter a file records, as the code the libraries carry.
USE_CODES = {"e": 1, "r": 2, "i": 3, "s": 4, "p": 5, "o": 6, "u": 0}
_TRAPEZOID = 0


def fixtures():
    return sorted(p.name for p in FIXTURES.glob("*.seq"))


def read(name):
    sequence = pp.Sequence()
    sequence.read(FIXTURES / name)
    return sequence_libraries(sequence)


def written(name):
    return sections(FIXTURES / name)


def decompress(count, samples):
    return np.asarray(core.decompress_shape(np.asarray(samples, float), count))


def close(ours, theirs):
    """Whether two arrays agree to single precision, scaled by what they hold."""
    ours, theirs = np.asarray(ours, float), np.asarray(theirs, float)
    scale = max(float(np.max(np.abs(theirs))) if theirs.size else 0.0, 1.0)
    return np.allclose(ours, theirs, rtol=SINGLE, atol=SINGLE * scale)


def shape_map(libraries, file_sections):
    """Map each minted shape id onto the file's shape holding the same samples."""
    theirs = {
        identifier: decompress(count, samples)
        for identifier, (count, samples) in shapes(file_sections["SHAPES"]).items()
    }
    mapping = {0: 0}
    for index, shape in enumerate(libraries.shapes):
        samples = decompress(shape.num_uncompressed_samples, shape.samples)
        matches = [
            identifier
            for identifier, candidate in theirs.items()
            if candidate.size == samples.size and close(samples, candidate)
        ]
        assert matches, f"shape {index + 1} is in no shape the file holds"
        mapping[index + 1] = matches[0]
    return mapping


def grad_rows(file_sections):
    """The file's two gradient tables as one, keyed by the id they share."""
    rows = {
        identifier: [_TRAPEZOID, *values, 0.0]
        for identifier, values in table(file_sections.get("TRAP", [])).items()
    }
    rows.update(
        {
            identifier: [1.0, *values]
            for identifier, values in table(file_sections.get("GRADIENTS", [])).items()
        }
    )
    return rows


def played(libraries, columns):
    """Ids of ``columns`` of the block table that some block plays."""
    return sorted(
        {int(value) for column in columns for value in libraries.blocks[:, column]}
        - {0}
    )


def shape_columns(library, row):
    """Columns of a row holding a shape id; a trapezoid carries times in its."""
    if library == "rf":
        return (1, 2, 3)
    if library == "adc":
        return (7,)
    return (4, 5) if row[0] != _TRAPEZOID else ()


def compare(name, library, columns, reference):
    libraries = read(name)
    file_sections = written(name)
    mapping = shape_map(libraries, file_sections)
    theirs = reference(file_sections)
    for identifier in played(libraries, columns):
        mine = np.array(getattr(libraries, library)[identifier - 1], dtype=np.float64)
        for index in shape_columns(library, mine):
            mine[index] = mapping[int(mine[index])]
        assert close(mine, theirs[identifier]), (
            f"{library} {identifier} of {name}: {mine} != {theirs[identifier]}"
        )


@pytest.mark.parametrize("name", fixtures())
def test_the_block_table_is_the_one_the_file_lists(name):
    libraries = read(name)
    theirs = table(written(name)["BLOCKS"])
    assert len(libraries.blocks) == len(theirs)
    for index, row in enumerate(libraries.blocks):
        np.testing.assert_array_equal(
            row.astype(np.int64), np.array(theirs[index + 1], dtype=np.int64)
        )


@pytest.mark.parametrize("name", fixtures())
def test_every_played_rf_event_is_the_one_the_file_holds(name):
    compare(name, "rf", (1,), lambda s: table(s["RF"]))


@pytest.mark.parametrize("name", fixtures())
def test_every_played_gradient_is_the_one_the_file_holds(name):
    compare(name, "grad", (2, 3, 4), grad_rows)


@pytest.mark.parametrize("name", fixtures())
def test_every_played_readout_is_the_one_the_file_holds(name):
    compare(name, "adc", (5,), lambda s: table(s["ADC"]))


@pytest.mark.parametrize("name", fixtures())
def test_the_rf_use_tags_are_the_ones_the_file_records(name):
    libraries = read(name)
    uses = rf_use(written(name)["RF"])
    for identifier in played(libraries, (1,)):
        assert int(libraries.rf_use[identifier - 1]) == USE_CODES[uses[identifier]], (
            f"rf {identifier} of {name}"
        )


@pytest.mark.parametrize("name", fixtures())
def test_every_shape_a_played_event_names_is_a_shape_the_file_holds(name):
    libraries = read(name)
    assert libraries.shapes
    shape_map(libraries, written(name))


def test_a_pulse_the_file_does_not_label_reads_as_an_unknown_use(tmp_path):
    sequence = pp.Sequence(pp.Opts())
    sequence.add_block(pp.make_sinc_pulse(flip_angle=0.1, duration=1e-3))
    path = tmp_path / "unlabelled.seq"
    sequence.write(path)
    loaded = pp.Sequence()
    loaded.read(path)
    assert int(sequence_libraries(loaded).rf_use[0]) == 0


@pytest.mark.parametrize("name", fixtures())
def test_each_played_gradient_carries_the_statistics_pypulseqpp_measures_for_it(name):
    """The payload renumbers the gradients it plays; their statistics follow them."""
    sequence = pp.Sequence()
    sequence.read(FIXTURES / name)
    measured = sequence.gradient_statistics()
    by_id = np.stack(
        (measured.peak_slew, measured.energy, measured.slew_energy), axis=1
    )
    played = sorted(
        {int(v) for row in sequence.block_events.values() for v in row[2:5]} - {0}
    )

    payload = conversion_payload(sequence, pp.Opts(B0=3.0))

    assert payload["grad_statistics"].shape == (len(played), 3)
    np.testing.assert_array_equal(
        payload["grad_statistics"], by_id[[i - 1 for i in played]]
    )
