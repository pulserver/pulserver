"""The specification tables read through pypulseqpp against the file's own rows."""

from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest
from _seqtext import extension_chains, sections

from pulserver.ir._source import LABEL_IDS, specification_libraries

FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
#: Soft-delay hint, as a Pulseq file numbers it; anything else is -1.
HINT_IDS = {"TE": 1, "TR": 2, "TI": 3, "ESP": 4, "RECTIME": 5}
# The file writes its cells at six significant digits.
SINGLE = 1e-6


def fixtures():
    return sorted(p.name for p in FIXTURES.glob("*.seq"))


@pytest.fixture
def extended(tmp_path):
    """A sequence playing every extension a block can carry."""
    sequence = pp.Sequence(pp.Opts())
    quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    shim = pp.make_rf_shim(np.array([1.0, 0.5 * np.exp(1j * np.pi / 3)]))
    sequence.add_block(pp.make_delay(2e-3), pp.make_trigger("physio1", duration=1e-3))
    sequence.add_block(pp.make_delay(1e-3), pp.make_rotation(quarter_turn))
    sequence.add_block(pp.make_delay(1e-3), pp.make_rotation(np.eye(3)))
    sequence.add_block(pp.make_delay(1e-3), pp.make_trigger("physio1", duration=2e-3))
    sequence.add_block(
        pp.make_delay(1e-3), pp.make_label(label="TRID", type="SET", value=5)
    )
    sequence.add_block(
        pp.make_delay(1e-3),
        pp.make_label(label="LIN", type="INC", value=1),
        pp.make_label(label="NAV", type="SET", value=1),
    )
    sequence.add_block(
        pp.make_delay(1e-3), pp.make_sinc_pulse(flip_angle=0.1, duration=1e-3), shim
    )
    sequence.add_block(
        pp.make_delay(2e-3), pp.make_soft_delay("TE", offset=1e-4, factor=2.0)
    )
    sequence.add_block(pp.make_delay(2e-3), pp.make_digital_output_pulse("osc0", 1e-3))
    # Three labels on one block, so a row is attributed by its place in the
    # chain rather than by there being only one of its kind.
    sequence.add_block(
        pp.make_delay(1e-3),
        pp.make_label(label="LIN", type="SET", value=3),
        pp.make_label(label="SLC", type="SET", value=2),
        pp.make_label(label="ECO", type="INC", value=1),
    )
    path = tmp_path / "extended.seq"
    sequence.write(path)
    return path


def specification_rows(path):
    """Each specification table of the file, in the layout the libraries use.

    A label row names its label and a soft delay names its hint; the libraries
    hold the number the scanner converter files them under, so the name is
    translated here and everything else is the file's own cells.
    """
    _, declared = extension_chains(sections(path).get("EXTENSIONS", []))
    rows = {
        name: {}
        for name in (
            "rotations",
            "triggers",
            "soft_delays",
            "labelset",
            "labelinc",
            "rf_shims",
        )
    }
    for kind, table in declared.items():
        for identifier, cells in table.items():
            if kind == "LABELSET":
                rows["labelset"][identifier] = [cells[0], LABEL_IDS[cells[1]]]
            elif kind == "LABELINC":
                rows["labelinc"][identifier] = [cells[0], LABEL_IDS[cells[1]]]
            elif kind == "DELAYS":
                rows["soft_delays"][identifier] = [
                    *cells[:3],
                    HINT_IDS.get(cells[3], -1),
                ]
            elif kind == "ROTATIONS":
                rows["rotations"][identifier] = list(cells)
            elif kind == "TRIGGERS":
                rows["triggers"][identifier] = list(cells)
            elif kind == "RF_SHIMS":
                # The file counts its channels first; the library holds the
                # magnitude and phase pairs alone.
                rows["rf_shims"][identifier] = list(cells[1:])
    return rows


def compare_specifications(path):
    sequence = pp.Sequence()
    sequence.read(path)
    ours = specification_libraries(sequence)
    theirs = specification_rows(path)
    for name, reference in theirs.items():
        mine = getattr(ours, name)
        for identifier in ours.referenced[name]:
            np.testing.assert_allclose(
                np.asarray(mine[identifier - 1], dtype=float),
                np.asarray(reference[identifier], dtype=float),
                rtol=SINGLE,
                atol=SINGLE,
                err_msg=f"{name} {identifier} of {Path(path).name}",
            )


@pytest.mark.parametrize("name", fixtures())
def test_every_label_row_a_fixture_holds_is_the_one_the_file_lists(name):
    compare_specifications(FIXTURES / name)


def test_every_specification_row_is_the_one_the_file_lists(extended):
    compare_specifications(extended)


def test_the_extended_sequence_fills_every_specification_table(extended):
    sequence = pp.Sequence()
    sequence.read(extended)
    libraries = specification_libraries(sequence)
    assert len(libraries.rotations) == 2
    assert len(libraries.triggers) == 3
    assert len(libraries.rf_shims) == 1
    assert len(libraries.soft_delays) == 1
    assert len(libraries.labelset) == 4
    assert len(libraries.labelinc) == 2
