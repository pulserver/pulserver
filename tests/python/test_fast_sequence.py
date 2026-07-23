from __future__ import annotations

from io import BytesIO
from tempfile import TemporaryDirectory

import numpy as np
import pulserver.io as pio
import pulserver.pypulseq as ps
import pypulseq as pp
import pytest


def _make_simple_seq(seq):
    g = pp.make_trapezoid(channel="x", amplitude=0.8 * seq.system.max_grad, flat_time=1e-3, system=seq.system)
    adc = pp.make_adc(num_samples=16, duration=1e-3, delay=0.0, system=seq.system)
    seq.add_block(g, adc)
    seq.add_block(pp.make_delay(1e-3))


def test_fast_sequence_disables_positional_set_block():
    seq = ps.Sequence()
    with pytest.raises(NotImplementedError):
        seq.set_block(1, pp.make_delay(1e-3))


def test_fast_sequence_add_block_without_dedup():
    seq = ps.Sequence()
    trap1 = pp.make_trapezoid(channel="x", amplitude=0.5 * seq.system.max_grad, flat_time=1e-3, system=seq.system)
    trap2 = pp.make_trapezoid(channel="x", amplitude=0.5 * seq.system.max_grad, flat_time=1e-3, system=seq.system)
    seq.add_block(trap1)
    seq.add_block(trap2)

    # Same event inserted twice without dedup during build.
    assert len(seq.grad_library.data) == 2


def test_write_supports_binary_stream_output():
    seq = ps.Sequence()
    _make_simple_seq(seq)

    stream = BytesIO()
    signature = pio.write(seq, output=stream, check_timing=False)

    assert signature is None
    payload = stream.getvalue()
    assert len(payload) > 0
    assert b"[BLOCKS]" in payload


def test_write_supports_path_output():
    seq = ps.Sequence()
    _make_simple_seq(seq)

    with TemporaryDirectory() as tmp:
        out_path = f"{tmp}/fast"
        signature = pio.write(seq, output=out_path, check_timing=False)
        assert signature is None
        with open(f"{out_path}.seq", "rb") as f:
            payload = f.read()

    assert len(payload) > 0
    assert b"# Pulseq sequence file" in payload


def test_get_block_decodes_through_the_plain_view():
    seq = ps.Sequence()
    _make_simple_seq(seq)

    block = seq.get_block(1)
    assert block.gx is not None
    assert block.adc.num_samples == 16


def test_seq_write_not_implemented_for_fast_builder():
    seq = ps.Sequence()
    _make_simple_seq(seq)

    with pytest.raises(NotImplementedError):
        seq.write("ignored.seq")


def test_custom_label_registered_on_add_block():
    seq = ps.Sequence()
    lbl = ps.make_label(label="MYLAB", type="SET", value=7)
    seq.add_block(lbl)

    assert "MYLAB" in seq.custom_labels
    assert seq.custom_labels["MYLAB"] > 0


def test_builtin_label_not_in_custom_labels():
    seq = ps.Sequence()
    lbl = ps.make_label(label="SLC", type="SET", value=1)
    seq.add_block(lbl)

    assert "SLC" not in seq.custom_labels


def test_custom_label_registered_on_add_block_fast_builder():
    seq = ps.Sequence()
    lbl = ps.make_label(label="CUSTLABEL", type="SET", value=42)
    seq.add_block(lbl)

    assert "CUSTLABEL" in seq.custom_labels


def test_rf_shim_extension_registered_in_library():
    seq = ps.Sequence()
    shim_event = ps.make_rf_shim([1 + 0j, 0.5 + 0.25j])
    seq.add_block(shim_event)

    assert len(seq.rf_shim_library.data) == 1


def test_rotation_extension_registered_in_library():
    class DummyQuat:
        def __init__(self, q):
            self._q = np.asarray(q, dtype=float)

        def as_quat(self, canonical=True, scalar_first=True):
            assert canonical
            if scalar_first:
                return self._q
            return np.asarray([self._q[1], self._q[2], self._q[3], self._q[0]], dtype=float)

    seq = ps.Sequence()
    rot_event = ps.make_rotation(DummyQuat([1.0, 0.0, 0.0, 0.0]))
    seq.add_block(rot_event)

    assert len(seq.rotation_library.data) == 1


def test_trap_event_registered_in_trap_and_grad_libraries():
    seq = ps.Sequence()
    trap = pp.make_trapezoid(channel="y", amplitude=0.5 * seq.system.max_grad, flat_time=1e-3, system=seq.system)
    seq.add_block(trap)

    grad_id = int(seq.block_events[1][3])  # y channel slot
    trap_id = int(seq.grad_library.data[grad_id][0])

    assert seq.grad_library.type[grad_id] == "t"
    assert trap_id in seq.trap_library.data


def test_remove_duplicates_dedups_extension_libraries_and_chain():
    class DummyQuat:
        def __init__(self, q):
            self._q = np.asarray(q, dtype=float)

        def as_quat(self, canonical=True, scalar_first=True):
            assert canonical
            if scalar_first:
                return self._q
            return np.asarray([self._q[1], self._q[2], self._q[3], self._q[0]], dtype=float)

    seq = ps.Sequence()
    label = ps.make_label("MYLAB", "SET", 3)
    rf_shim = ps.make_rf_shim([1 + 0j, 0.5 + 0.25j])
    rot = ps.make_rotation(DummyQuat([1.0, 0.0, 0.0, 0.0]))
    seq.add_block(label, rf_shim, rot)
    seq.add_block(ps.make_label("MYLAB", "SET", 3), ps.make_rf_shim([1 + 0j, 0.5 + 0.25j]), rot)

    seq2 = seq.remove_duplicates()

    assert len(seq2.label_set_library.data) == 1
    assert len(seq2.rf_shim_library.data) == 1
    assert len(seq2.rotation_library.data) == 1
    assert seq2.block_events[1][6] == seq2.block_events[2][6]


def test_remove_duplicates_in_place_updates_sequence():
    seq = ps.Sequence()
    seq.add_block(ps.make_label("MYLAB", "SET", 1))
    seq.add_block(ps.make_label("MYLAB", "SET", 1))

    seq.remove_duplicates(in_place=True)

    assert len(seq.label_set_library.data) == 1


def test_remove_duplicates_preserves_rf_use_type():
    seq = ps.Sequence()
    seq.shape_library.insert(1, np.asarray([2.0, 0.0, 1.0]))
    seq.shape_library.insert(2, np.asarray([2.0, 0.0, 1.0]))
    seq.shape_library.insert(3, np.asarray([2.0, 0.0, 1.0]))
    rf_data = (1.0, 1, 2, 3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    seq.rf_library.insert(1, rf_data, "e")
    seq.rf_library.insert(2, rf_data, "r")

    dedup_exact = seq.remove_duplicates()

    assert len(dedup_exact.rf_library.data) == 2


def test_rf_use_stored_as_numeric_code():
    seq = ps.Sequence()
    rf, _, _ = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(10.0),
        duration=1e-3,
        slice_thickness=8e-3,
        apodization=0.5,
        time_bw_product=4,
        system=seq.system,
        return_gz=True,
        use="excitation",
    )
    seq.add_block(rf)
    rf_id = int(seq.block_events[1][1])
    assert isinstance(seq.rf_library.type[rf_id], int | np.integer)


def test_write_with_rf_event_serializes_rf_use_as_char():
    seq = ps.Sequence()
    rf, _, _ = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(10.0),
        duration=1e-3,
        slice_thickness=8e-3,
        apodization=0.5,
        time_bw_product=4,
        system=seq.system,
        return_gz=True,
        use="excitation",
    )
    seq.add_block(rf)

    stream = BytesIO()
    pio.write(seq, output=stream, check_timing=False, remove_duplicates=False)
    content = stream.getvalue().decode("utf-8")

    assert "[RF]" in content
    assert " e\n" in content


def test_write_version_revision_bumped_for_custom_extensions_and_labels():
    class DummyQuat:
        def __init__(self, q):
            self._q = np.asarray(q, dtype=float)

        def as_quat(self, canonical=True, scalar_first=True):
            assert canonical
            if scalar_first:
                return self._q
            return np.asarray([self._q[1], self._q[2], self._q[3], self._q[0]], dtype=float)

    seq_ext = ps.Sequence()
    seq_ext.add_block(ps.make_rotation(DummyQuat([1.0, 0.0, 0.0, 0.0])))
    stream_ext = BytesIO()
    pio.write(seq_ext, output=stream_ext, check_timing=False)
    lines_ext = stream_ext.getvalue().decode("utf-8").splitlines()
    rev_line_ext = next(line for line in lines_ext if line.startswith("revision "))
    rev_ext = int(rev_line_ext.split()[1])
    assert rev_ext >= 1

    seq_lbl = ps.Sequence()
    seq_lbl.add_block(ps.make_label("MY_CUSTOM_LABEL", "SET", 1))
    stream_lbl = BytesIO()
    pio.write(seq_lbl, output=stream_lbl, check_timing=False)
    lines_lbl = stream_lbl.getvalue().decode("utf-8").splitlines()
    rev_line_lbl = next(line for line in lines_lbl if line.startswith("revision "))
    rev_lbl = int(rev_line_lbl.split()[1])
    assert rev_lbl >= 2


def test_module_label_registered_and_round_trips():
    """MODULE (DESIGN_safety_module_labels.md / PLAN_safety_module_labels.md)
    is a novel pulseq LABELSET name, not in pypulseq's/pulserver's builtin
    set -- verifies the generic custom-label authoring path (already proven
    by test_custom_label_registered_on_add_block for an arbitrary name)
    also works end-to-end for the actual production label name, including
    the on-disk LABELSET/EXTENSIONS round-trip pulseg_parse.c reads back
    (see external/pulserver/csrc/src/core/pulseg_error.c label_table[]).
    """
    seq = ps.Sequence()
    lbl = ps.make_label(label="MODULE", type="SET", value=1)
    seq.add_block(pp.make_delay(1e-3), lbl)

    assert "MODULE" in seq.custom_labels
    assert seq.custom_labels["MODULE"] > 0

    stream = BytesIO()
    pio.write(seq, output=stream, check_timing=False)
    text = stream.getvalue().decode("utf-8")
    lines = text.splitlines()

    rev_line = next(line for line in lines if line.startswith("revision "))
    assert int(rev_line.split()[1]) >= 2

    assert "extension LABELSET" in text
    labelset_idx = next(i for i, line in enumerate(lines) if line.startswith("extension LABELSET"))
    # Data row immediately follows: "<block_id> <value> MODULE"
    data_row = lines[labelset_idx + 1].split()
    assert data_row[1] == "1"
    assert data_row[2] == "MODULE"


def test_soft_delay_is_ignored_in_fast_path():
    seq = ps.Sequence()
    seq.add_block(pp.make_soft_delay(hint="IGNORED_HINT", default_duration=1e-3))

    assert len(seq.block_events) == 1
    assert int(seq.block_events[1][6]) == 0
