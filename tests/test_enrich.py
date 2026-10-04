from pathlib import Path

import ismrmrd
import ismrmrd.xsd
import numpy as np
import pypulseqpp as pp
import pytest
from _synthetic import DELTA_K, SAMPLES, add_readout
from pypulseqpp import sequences

from pulserver.mrd import AcquisitionFlag, EncodingSpace
from pulserver.proxy._enrich import (
    SequenceTable,
    enrich_acquisition,
    enrich_header,
    header_fov_offset_m,
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
    return pp.io.read(FIXTURES / name)


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


def add_partial_readout(seq, fraction, *labels):
    """A readout along x whose k starts at ``-fraction`` of its extent, so that its echo is not at its middle."""
    system = pp.Opts()
    gx = pp.make_trapezoid(
        "x", flat_area=SAMPLES * DELTA_K, flat_time=3.2e-3, system=system
    )
    adc = pp.make_adc(
        num_samples=SAMPLES, duration=3.2e-3, delay=gx.rise_time, system=system
    )
    seq.add_block(
        pp.make_trapezoid("x", area=-fraction * gx.area, duration=1e-3, system=system)
    )
    seq.add_block(gx, adc, *labels)


def enriched_encoding(table):
    """The first encoding of an empty header enriched from ``table``."""
    enriched = header()
    enrich_header(enriched, table)
    return enriched.encoding[0]


def designed(tmp_path, result):
    path = tmp_path / "designed.seq"
    sequences.write(path, result)
    return SequenceTable.read(path)


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


def test_a_readout_carries_every_axis_its_encoding_space_varies_along(tmp_path):
    seq = pp.Sequence(pp.Opts())
    seq.add_block(pp.make_trapezoid("y", area=3 * DELTA_K, duration=1e-3))
    add_readout(seq)
    add_readout(seq, rotation=pp.make_rotation(np.pi / 2))
    path = tmp_path / "blade.seq"
    seq.write(path)
    table = SequenceTable.read(path)
    reference = pp.Sequence()
    reference.read(path)
    k_adc = reference.calculate_kspace()[0]
    assert table.trajectory_dimensions.tolist() == [2, 2]
    first = acquisitions(table)[0]
    enrich_acquisition(first, table, 0)
    np.testing.assert_allclose(first.traj[:, 1], 3 * DELTA_K, rtol=1e-5)
    np.testing.assert_allclose(
        first.traj, readout_k(k_adc, table, 0)[:2].T, rtol=1e-5, atol=1e-3
    )


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


def test_the_header_lists_every_flip_angle_the_sequence_plays():
    seq = reference("gre_2d_3sl.seq")
    measured = sorted({round(a, 3) for a in seq.test_report_dict()["flip_angles_deg"]})
    enriched = header()
    enrich_header(enriched, fixture("gre_2d_3sl.seq"))
    assert pytest.approx(measured) == enriched.sequenceParameters.flipAngle_deg


def test_the_header_centres_k_space_where_the_design_puts_its_centre(tmp_path):
    from pypulseqpp.sequences.sequence.gre3D_sequence import gre3d

    system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=170, slew_unit="T/m/s")
    enriched = header()
    enrich_header(enriched, designed(tmp_path, gre3d(system, n_x=32, n_y=16, n_z=8)))
    limits = enriched.encoding[0].encodingLimits
    line, partition = limits.kspace_encoding_step_1, limits.kspace_encoding_step_2
    assert (line.maximum, line.center) == (15, 8)
    assert (partition.maximum, partition.center) == (7, 4)


def test_the_limits_start_at_the_first_counter_a_scan_plays(tmp_path):
    seq = pp.Sequence(pp.Opts())
    for lin in range(2, 6):
        add_readout(seq, pp.make_label("LIN", "SET", lin))

    limits = enriched_encoding(written(seq, tmp_path)).encodingLimits

    line = limits.kspace_encoding_step_1
    assert (line.minimum, line.maximum) == (2, 5)
    assert (limits.slice.minimum, limits.slice.maximum) == (0, 0)


def test_the_limits_centre_is_the_centre_line_the_sequence_defines(tmp_path):
    seq = pp.Sequence(pp.Opts())
    seq.set_definition("kSpaceCenterLine", 5)
    for lin in range(2, 8):
        add_readout(seq, pp.make_label("LIN", "SET", lin))

    limits = enriched_encoding(written(seq, tmp_path)).encodingLimits

    assert limits.kspace_encoding_step_1.center == 5


@pytest.mark.parametrize(
    ("lines", "centre"),
    [(range(8), 4), (range(2, 6), 4), (range(5), 2), (range(3, 6), 4)],
)
def test_the_limits_centre_is_the_middle_of_the_lines_when_the_sequence_defines_none(
    tmp_path, lines, centre
):
    seq = pp.Sequence(pp.Opts())
    for lin in lines:
        add_readout(seq, pp.make_label("LIN", "SET", lin))

    limits = enriched_encoding(written(seq, tmp_path)).encodingLimits

    assert limits.kspace_encoding_step_1.center == centre


@pytest.mark.parametrize(
    ("options", "samples"),
    [
        ({}, 64),
        ({"partial_fourier_x": 0.75}, 64),
        ({"readout_oversampling": 1.0}, 32),
    ],
)
def test_the_encoded_readout_is_the_full_echo_with_its_oversampling(
    tmp_path, options, samples
):
    from pypulseqpp.sequences.sequence.gre2D_sequence import gre2d

    system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=170, slew_unit="T/m/s")
    seq = gre2d(system, fov_x=0.22, fov_y=0.22, n_x=32, n_y=16, **options)

    encoding = enriched_encoding(designed(tmp_path, seq))

    encoded, recon = encoding.encodedSpace, encoding.reconSpace
    assert (encoded.matrixSize.x, encoded.matrixSize.y) == (samples, 16)
    assert (recon.matrixSize.x, recon.matrixSize.y) == (32, 16)
    assert encoded.fieldOfView_mm.x == pytest.approx(220.0 * samples / 32)
    assert recon.fieldOfView_mm.x == pytest.approx(220.0)
    assert encoded.fieldOfView_mm.y == recon.fieldOfView_mm.y == pytest.approx(220.0)


@pytest.mark.parametrize("fraction", [0.25, 0.75])
def test_the_full_echo_of_a_partial_echo_has_as_many_samples_on_each_side_as_its_longer_side(
    tmp_path, fraction
):
    seq = pp.Sequence(pp.Opts())
    seq.set_definition("Matrix", [SAMPLES, 1, 1])
    seq.set_definition("FOV", [0.2, 0.2, 0.005])
    add_partial_readout(seq, fraction)
    table = written(seq, tmp_path)
    samples, echo = int(table.num_samples[0]), int(table.center_sample[0])
    full = 2 * max(echo, samples - echo)

    encoding = enriched_encoding(table)

    assert echo != samples // 2
    assert full > samples
    assert table.spaces[0].readout_samples == full
    assert encoding.encodedSpace.matrixSize.x == full
    assert encoding.encodedSpace.fieldOfView_mm.x == pytest.approx(
        200.0 * full / SAMPLES
    )
    assert encoding.reconSpace.matrixSize.x == SAMPLES


def test_a_reversed_readout_is_counted_from_its_end():
    """The echo of a reversed readout is at the mirror image of the forward readouts' echo: sample 15 of 32 against 16."""
    table = fixture("epi_2d_main.seq")

    assert has(table, AcquisitionFlag.IS_REVERSE).any()
    assert set(table.center_sample.tolist()) == {15, 16}
    assert table.spaces[0].readout_samples == 32


def test_a_non_cartesian_space_keeps_its_matrix_and_field_of_view_along_the_readout():
    table = fixture("mprage_stack_of_spirals_3d.seq")

    encoding = enriched_encoding(table)

    assert table.spaces[0].readout_samples is None
    for space in (encoding.encodedSpace, encoding.reconSpace):
        assert space.matrixSize.x == 32
        assert space.fieldOfView_mm.x == pytest.approx(220.0)


def test_an_acquisition_of_the_wrong_length_is_refused():
    table = fixture("gre_2d_3sl.seq")
    samples = np.ones((2, int(table.num_samples[0]) + 1), np.complex64)
    with pytest.raises(ValueError, match="samples"):
        enrich_acquisition(ismrmrd.Acquisition.from_array(samples), table, 0)


def curving(tmp_path, shift_m):
    """A readout under a gradient that does not hold one value, moved by shift_m."""
    system = pp.Opts(B0=3.0)
    seq = pp.Sequence(system)
    samples = 64
    waveform = 2.0e4 * np.sin(2 * np.pi * np.arange(samples) / samples)
    seq.add_block(
        pp.make_arbitrary_grad("x", waveform=waveform, system=system),
        pp.make_adc(num_samples=samples, dwell=4e-6),
    )
    if any(shift_m):
        pp.TransformFOV(translation=shift_m, through_rotation=True).apply_to_sequence(
            seq, in_place=True
        )
    return written(seq, tmp_path)


def test_a_curving_readout_carries_the_phase_its_receiver_cannot_apply(tmp_path):
    """A shifted field of view curves the phase of a readout whose gradient varies.

    Under a gradient that holds one value the shift is a phase and a frequency
    offset, which the receiver applies. Under one that does not, what is left
    is a curve over the readout, and it arrives unapplied.
    """
    table = curving(tmp_path, (0.04, 0.0, 0.0))
    modulation = table.readout_phase_modulation(0)
    assert modulation is not None
    assert modulation.size == int(table.num_samples[0])
    assert np.ptp(modulation) > 1e-3, "a curving readout's phase should not be flat"

    samples = np.ones((2, int(table.num_samples[0])), np.complex64)
    acquisition = ismrmrd.Acquisition.from_array(samples)
    enrich_acquisition(acquisition, table, 0)
    assert np.allclose(
        np.asarray(acquisition.data), samples * np.exp(1j * modulation), atol=1e-6
    )


def test_an_unmoved_readout_arrives_as_it_was_received(tmp_path):
    """Nothing is applied where the field of view was not moved."""
    table = curving(tmp_path, (0.0, 0.0, 0.0))
    assert table.readout_phase_modulation(0) is None

    samples = (np.arange(64, dtype=np.complex64) + 1j).reshape(1, 64)
    acquisition = ismrmrd.Acquisition.from_array(samples)
    enrich_acquisition(acquisition, table, 0)
    assert np.array_equal(np.asarray(acquisition.data), samples)


def test_a_steady_readout_needs_no_curve(tmp_path):
    """A gradient holding one value across the readout leaves nothing over."""
    system = pp.Opts(B0=3.0)
    seq = pp.Sequence(system)
    readout = pp.make_trapezoid("x", flat_area=2000, flat_time=2.56e-3, system=system)
    seq.add_block(
        readout,
        pp.make_adc(num_samples=64, duration=2.56e-3, delay=readout.rise_time),
    )
    pp.TransformFOV(
        translation=(0.04, 0.0, 0.0), through_rotation=True
    ).apply_to_sequence(seq, in_place=True)
    assert written(seq, tmp_path).readout_phase_modulation(0) is None


def test_the_header_states_the_shift_in_millimetres_along_read_phase_and_slice():
    stated = header()
    stated.userParameters = ismrmrd.xsd.userParametersType(
        userParameterString=[
            ismrmrd.xsd.userParameterStringType(name="fov_offset_mm", value="20 -12 5")
        ]
    )
    assert header_fov_offset_m(stated) == pytest.approx((0.02, -0.012, 0.005))
    assert header_fov_offset_m(header()) is None


def test_a_stated_position_away_from_the_converted_shift_adds_its_whole_phase():
    path = FIXTURES / "mprage_stack_of_spirals_3d.seq"
    shift = np.array([0.02, -0.012, 0.005])
    moved = np.array([0.003, 0.001, -0.002])
    converted = SequenceTable.read(path, fov_offset_m=tuple(shift))
    unshifted = SequenceTable.read(path)
    moving = [
        i
        for i in range(len(converted))
        if converted.readout_phase_modulation(i) is not None
    ]
    assert moving
    for index in moving[:20]:
        k = converted.readout_k(index)
        np.testing.assert_allclose(
            converted.readout_phase_modulation(index, tuple(shift)),
            converted.readout_phase_modulation(index),
        )
        np.testing.assert_allclose(
            converted.readout_phase_modulation(index, tuple(shift + moved)),
            converted.readout_phase_modulation(index) + 2 * np.pi * (moved @ k),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            unshifted.readout_phase_modulation(index, tuple(shift)),
            2 * np.pi * (shift @ k),
            atol=1e-9,
        )
