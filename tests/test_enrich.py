from pathlib import Path

import ismrmrd
import ismrmrd.xsd
import numpy as np
import pypulseqpp as pp
import pytest
from _synthetic import add_readout

from pulserver.mrd import AcquisitionFlag, EncodingSpace
from pulserver.vre._enrich import (
    SequenceTable,
    enrich_acquisition,
    enrich_header,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sequences"

HEADER = """<?xml version="1.0"?>
<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
  <experimentalConditions><H1resonanceFrequency_Hz>63500000</H1resonanceFrequency_Hz></experimentalConditions>
  <encoding>
    <encodedSpace><matrixSize><x>1</x><y>1</y><z>1</z></matrixSize><fieldOfView_mm><x>1</x><y>1</y><z>1</z></fieldOfView_mm></encodedSpace>
    <reconSpace><matrixSize><x>1</x><y>1</y><z>1</z></matrixSize><fieldOfView_mm><x>1</x><y>1</y><z>1</z></fieldOfView_mm></reconSpace>
    <encodingLimits/>
    <trajectory>cartesian</trajectory>
  </encoding>
</ismrmrdHeader>
"""


def header():
    return ismrmrd.xsd.CreateFromDocument(HEADER)


def written(seq, tmp_path):
    path = tmp_path / "synthetic.seq"
    seq.write(path)
    return SequenceTable.read(path)


def fixture(name):
    return SequenceTable.read(FIXTURES / name)


def reference(name):
    seq = pp.Sequence()
    seq.read(FIXTURES / name)
    return seq


def acquisitions(table, data=None):
    return [
        ismrmrd.Acquisition.from_array(
            np.ones((2, int(table.num_samples[index])), np.complex64)
            if data is None
            else data(index)
        )
        for index in range(len(table))
    ]


def readout_k(k, table, index):
    start = int(table.num_samples[:index].sum())
    return k[:, start : start + int(table.num_samples[index])]


def has(table, flag):
    return (table.flags & np.uint64(flag.value)) != 0


@pytest.mark.parametrize(
    "name", ["gre_2d_3sl.seq", "epi_2d_main.seq", "mprage_stack_of_spirals_3d.seq"]
)
def test_counters_are_the_labels_each_readout_sees(name):
    table = fixture(name)
    labels = reference(name).evaluate_labels(evolution="adc")
    for label in ("LIN", "PAR", "SLC", "SEG", "REP"):
        expected = np.broadcast_to(labels.get(label, 0), (len(table),))
        np.testing.assert_array_equal(table.counters[label], expected)


def test_a_slice_closes_once_per_echo(tmp_path):
    seq = pp.Sequence(pp.Opts())
    for slc in range(2):
        for lin in range(3):
            for eco in range(2):
                add_readout(
                    seq,
                    pp.make_label("SLC", "SET", slc),
                    pp.make_label("LIN", "SET", lin),
                    pp.make_label("ECO", "SET", eco),
                )
    table = written(seq, tmp_path)
    closes = has(table, AcquisitionFlag.LAST_IN_SLICE)
    slc, eco, lin = (table.counters[name] for name in ("SLC", "ECO", "LIN"))
    assert closes.sum() == 4
    assert set(zip(slc[closes], eco[closes], strict=True)) == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    assert (lin[closes] == 2).all()


def test_only_the_last_readout_of_a_chain_ends_the_measurement():
    table = fixture("dedup_gre_pair.seq")
    ends = np.flatnonzero(has(table, AcquisitionFlag.LAST_IN_MEASUREMENT))
    assert ends.tolist() == [len(table) - 1]
    assert [space.subsequence for space in table.spaces] == [0, 1]


def test_a_chain_readout_carries_the_k_of_its_own_file():
    table = fixture("dedup_gre_pair.seq")
    files = [reference(name) for name in ("dedup_gre_pair.seq", "dedup_gre_pair_b.seq")]
    first_of_second = int(np.flatnonzero(table.encoding_space == 1)[0])
    for index in range(len(table)):
        file = int(index >= first_of_second)
        local = index - file * first_of_second
        k = files[file].calculate_kspace()[0]
        start = int(table.num_samples[file * first_of_second : index].sum())
        np.testing.assert_allclose(
            table.readout_k(index),
            k[:, start : start + int(table.num_samples[index])],
            atol=1e-6 * np.abs(k).max(),
            err_msg=f"row {index}, readout {local} of file {file}",
        )


def test_navigator_readouts_form_their_own_encoding_space(tmp_path):
    seq = pp.Sequence(pp.Opts())
    seq.set_definition("Matrix", [32, 3, 1])
    seq.set_definition("NavMatrix", [32, 1, 1])
    for lin in range(3):
        add_readout(
            seq, pp.make_label("LIN", "SET", lin), pp.make_label("NAV", "SET", 0)
        )
        add_readout(seq, pp.make_label("NAV", "SET", 1))
    table = written(seq, tmp_path)
    assert table.encoding_space.tolist() == [0, 1] * 3
    assert has(table, AcquisitionFlag.IS_NAVIGATION_DATA).tolist() == [False, True] * 3
    enriched = header()
    enrich_header(enriched, table)
    assert [encoding.encodedSpace.matrixSize.y for encoding in enriched.encoding] == [
        3,
        1,
    ]


@pytest.mark.parametrize("name", ["gre_2d_3sl.seq", "epi_2d_main.seq"])
def test_a_line_of_k_carries_one_axis_under_a_cartesian_header(name):
    table = fixture(name)
    k_adc = reference(name).calculate_kspace()[0]
    for index, acquisition in enumerate(acquisitions(table)):
        enrich_acquisition(acquisition, table, index)
        assert acquisition.trajectory_dimensions == 1
        np.testing.assert_allclose(
            acquisition.traj[:, 0],
            readout_k(k_adc, table, index)[0],
            rtol=1e-5,
            atol=1e-3,
        )
        assert acquisition.center_sample == table.center_sample[index]
    enriched = header()
    enrich_header(enriched, table)
    assert enriched.encoding[0].trajectory == ismrmrd.xsd.trajectoryType.CARTESIAN


@pytest.mark.parametrize("name", ["zte_3d.seq", "mprage_stack_of_spirals_3d.seq"])
def test_a_non_cartesian_readout_carries_its_absolute_k(name):
    table = fixture(name)
    k_adc = reference(name).calculate_kspace()[0]
    assert all(space.trajectory for space in table.spaces)
    for index, acquisition in enumerate(acquisitions(table)):
        enrich_acquisition(acquisition, table, index)
        k = readout_k(k_adc, table, index)
        dimensions = acquisition.trajectory_dimensions
        assert dimensions >= 2
        np.testing.assert_allclose(
            acquisition.traj, k[:dimensions].T, rtol=1e-5, atol=1e-3
        )
    enriched = header()
    enrich_header(enriched, table)
    assert enriched.encoding[0].trajectory == ismrmrd.xsd.trajectoryType.OTHER


def test_a_rotated_flat_readout_makes_its_space_non_cartesian(tmp_path):
    seq = pp.Sequence(pp.Opts())
    for angle in (0.0, np.pi / 2):
        add_readout(seq, rotation=pp.make_rotation(angle))
    assert written(seq, tmp_path).spaces[0].trajectory


def test_a_readout_whose_k_does_not_move_keeps_the_received_centre_sample(tmp_path):
    seq = pp.Sequence(pp.Opts())
    add_readout(seq, moving=False)
    table = written(seq, tmp_path)
    acquisition = acquisitions(table)[0]
    acquisition.center_sample = 7
    enrich_acquisition(acquisition, table, 0)
    assert acquisition.center_sample == 7
    assert acquisition.trajectory_dimensions == 0


def test_the_header_describes_each_encoding_space():
    table = fixture("gre_2d_3sl.seq")
    enriched = header()
    enrich_header(enriched, table)
    encoding = enriched.encoding[0]
    size = encoding.encodedSpace.matrixSize
    assert (size.x, size.y, size.z) == (64, 8, 3)
    fov = encoding.reconSpace.fieldOfView_mm
    assert (fov.x, fov.y, fov.z) == pytest.approx((220.0, 220.0, 15.0))
    assert encoding.encodingLimits.slice.maximum == 2
    assert encoding.encodingLimits.kspace_encoding_step_1.maximum == 7
    assert pytest.approx([30.0]) == enriched.sequenceParameters.TR
    assert pytest.approx([5.0]) == enriched.sequenceParameters.TE
    reparsed = ismrmrd.xsd.CreateFromDocument(ismrmrd.xsd.ToXML(enriched))
    space = EncodingSpace.from_header(reparsed)
    assert space.loops == ("slice",)
    assert space.phase_encodes == 8


def test_an_acquisition_of_the_wrong_length_is_refused():
    table = fixture("gre_2d_3sl.seq")
    samples = np.ones((2, int(table.num_samples[0]) + 1), np.complex64)
    with pytest.raises(ValueError, match="samples"):
        enrich_acquisition(ismrmrd.Acquisition.from_array(samples), table, 0)
