"""The C interpreter's resolved IR against the spec-first oracle, in process.

No stored truth: for every corpus fixture the file is loaded twice -- once
through the C interpreter (via the ``pulseg`` projections) and once
through the Python oracle, which computes the PulSeg SegmentInstance /
BaseBlock content directly from the parsed events. The two readings must
agree wherever the spec fixes the answer.

Lanes:

* **instance table** -- per-block durations, RF amplitude/phase/frequency,
  gradient amplitudes (signed for trapezoids, unsigned for sampled shapes,
  whose sign lives in the shape), ADC presence/offsets, rotations, sticky
  TRID/NOROT/NOPOS, digital-out; plus the explicit per-ADC counter columns.
* **segment definitions** -- the partition is validated structurally (see
  ``test_pulseg_oracle``), then each segment's per-block event content is
  held against the file blocks its definition resolves to.
* **TR waveforms** -- the interpreter's composed per-TR gradient waveforms
  against a direct piecewise-linear composition of the file events over the
  same window.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.pypulseq as pp
from fixture_corpus import CORPUS, FIXTURES_DIR
from pulseg_oracle import adc_label_table, instance_table, tile_partition
from pulseg_oracle.content import block_boundaries, tr_window_waveforms
from pulserver._ext import pulseg as wrapper
from pulserver.pypulseq import _safety

GAMMA = 42577478.0
B0 = 3.0

#: Single-file corpus slots plus the NextSequence chains, as file lists.
CASES: dict[str, list[str]] = {stem: [f"{stem}.seq"] for stem in CORPUS}
CASES["epi_2d"] = ["epi_2d.seq", "epi_2d_main.seq"]
CASES["epi_3d"] = ["epi_3d.seq", "epi_3d_main.seq"]


#: The GE console's label-table column injection: [LIN, SLC, ECO].
LABEL_COLUMNS = ("LIN", "SLC", "ECO")
LABEL_COLUMN_MAP = [8, 0, 6]


def collection_for(files: list[str]):
    buffers = [(FIXTURES_DIR / name).read_bytes() for name in files]
    return wrapper._PulseqCollection(
        buffers,
        GAMMA,
        B0,
        GAMMA * 1.0,
        GAMMA * 10000.0,
        1e-6,
        10e-6,
        0.1e-6,
        10e-6,
        True,
        label_column_map=LABEL_COLUMN_MAP,
    )


def read(name: str) -> pp.Sequence:
    seq = pp.Sequence()
    seq.read(str(FIXTURES_DIR / name))
    return seq


@pytest.fixture(scope="module", params=sorted(CASES), ids=sorted(CASES))
def case(request):
    files = CASES[request.param]
    return collection_for(files), [read(name) for name in files]


def test_the_instance_table_is_the_file_content(case):
    coll, seqs = case
    table = wrapper._get_scan_table(coll)
    offset = 0
    for subseq, seq in enumerate(seqs):
        n = seq.num_blocks
        rows = slice(offset, offset + n)
        assert (table["subseq_idx"][rows] == subseq).all()
        oracle = instance_table(seq, GAMMA, B0)

        np.testing.assert_array_equal(table["duration_us"][rows], oracle["duration_us"])
        np.testing.assert_allclose(
            table["rf_amp_hz"][rows], oracle["rf_amp_hz"], rtol=1e-5, atol=1e-4
        )
        np.testing.assert_allclose(
            table["rf_phase_rad"][rows], oracle["rf_phase_rad"], atol=1e-5
        )
        np.testing.assert_allclose(
            table["rf_freq_hz"][rows], oracle["rf_freq_hz"], rtol=1e-6, atol=1e-3
        )
        for axis in ("gx", "gy", "gz"):
            c_amp = table[f"{axis}_amp_hz_per_m"][rows]
            o_amp = oracle[f"{axis}_amp_hz_per_m"]
            trap = oracle[f"{axis}_is_trap"].astype(bool)
            np.testing.assert_allclose(
                c_amp[trap], o_amp[trap], rtol=1e-5, atol=1e-2, err_msg=axis
            )
            np.testing.assert_allclose(
                np.abs(c_amp[~trap]), o_amp[~trap], rtol=1e-5, atol=1e-2, err_msg=axis
            )
        np.testing.assert_array_equal(table["adc_flag"][rows], oracle["adc_flag"])
        np.testing.assert_allclose(
            table["adc_phase_rad"][rows], oracle["adc_phase_rad"], atol=1e-5
        )
        np.testing.assert_allclose(
            table["adc_freq_hz"][rows], oracle["adc_freq_hz"], rtol=1e-6, atol=1e-3
        )
        np.testing.assert_allclose(table["rotmat"][rows], oracle["rotmat"], atol=2e-6)
        np.testing.assert_array_equal(table["trid"][rows], oracle["trid"])
        np.testing.assert_array_equal(table["norot_flag"][rows], oracle["norot_flag"])
        np.testing.assert_array_equal(table["nopos_flag"][rows], oracle["nopos_flag"])
        np.testing.assert_array_equal(
            table["digitalout_flag"][rows], oracle["digitalout_flag"]
        )
        offset += n
    assert offset == table["duration_us"].shape[0]


def test_the_counter_columns_are_the_written_labels(case):
    coll, seqs = case
    for subseq, seq in enumerate(seqs):
        c_labels = wrapper._get_adc_labels(coll, subseq)
        o_labels = adc_label_table(seq, columns=LABEL_COLUMNS)
        assert c_labels.shape == o_labels.shape
        np.testing.assert_array_equal(c_labels, o_labels)


def _normalized(wave: np.ndarray) -> np.ndarray:
    wave = np.asarray(wave, dtype=float)
    peak = wave[np.argmax(np.abs(wave))]
    return wave / (peak if peak != 0.0 else 1.0)


def test_the_segment_definitions_hydrate_to_the_file_events(case):
    coll, seqs = case
    all_segments = wrapper._get_segment_blocks(coll)
    for subseq, seq in enumerate(seqs):
        layouts = wrapper._get_segments(coll, subseq)
        instances = tile_partition(seq, layouts)
        for layout in layouts:
            segment = all_segments[layout["global_index"]]
            indices = layout["block_indices"]
            assert segment["num_blocks"] == len(indices)
            for b, block_idx in enumerate(indices):
                blk = segment["blocks"][b]
                ref = seq.get_block(int(block_idx) + 1)
                if not blk["is_variable_delay"] and not segment["pure_delay"]:
                    # A pure delay's duration is per-instance (setperiod);
                    # the definition holds a placeholder, the scan table the
                    # played value.
                    assert blk["duration_us"] == round(float(ref.block_duration) * 1e6)
                assert blk["has_rf"] == int(ref.rf is not None)
                assert blk["has_adc"] >= 0  # presence is instance metadata
                if ref.rf is not None:
                    assert blk["rf_delay_us"] == round(float(ref.rf.delay) * 1e6)
                    signal = np.abs(np.asarray(ref.rf.signal))
                    peak = signal.max() or 1.0
                    mag = np.asarray(blk["rf_magnitude"])
                    assert mag.shape[1] == signal.size
                    np.testing.assert_allclose(mag[0], signal / peak, atol=2e-3)
                for axis_idx, axis in enumerate(("gx", "gy", "gz")):
                    grad = getattr(ref, axis)
                    g = blk["grads"][axis_idx]
                    if grad is None:
                        assert g is None
                        continue
                    assert g is not None
                    assert g["delay_us"] == round(float(grad.delay) * 1e6)
                    if grad.type == "trap":
                        assert g["is_trapezoid"] == 1
                        file_t = (
                            np.cumsum(
                                [0.0, grad.rise_time, grad.flat_time, grad.fall_time]
                            )
                            * 1e6
                        )
                        file_v = np.array([0.0, grad.amplitude, grad.amplitude, 0.0])
                    else:
                        file_t = np.asarray(grad.tt, dtype=float) * 1e6
                        file_v = np.asarray(grad.waveform, dtype=float)
                    # The definition's amplitude is captured from ONE
                    # instance (the max-energy one); the block this layout
                    # resolves to may be a different instance of the same
                    # position, so equality holds against the instance SET.
                    instance_amps = []
                    for seg_id, inst_start, _count in instances:
                        if seg_id != layout["global_index"]:
                            continue
                        other = getattr(seq.get_block(inst_start + 1 + b), axis)
                        if other is None:
                            continue
                        if other.type == "trap":
                            instance_amps.append(abs(float(other.amplitude)))
                        else:
                            instance_amps.append(
                                float(np.abs(np.asarray(other.waveform)).max())
                            )
                    assert instance_amps, f"{axis}: no instance carries this event"
                    deviation = np.abs(
                        np.asarray(instance_amps) - g["max_amplitude_hz_per_m"]
                    )
                    assert deviation.min() <= max(
                        1e-5 * g["max_amplitude_hz_per_m"], 1e-2
                    ), (
                        f"{axis}: definition amplitude"
                        f" {g['max_amplitude_hz_per_m']} matches no instance"
                        f" (instance amplitudes {sorted(set(instance_amps))})"
                    )
                    if (
                        g["num_shots"] == 1
                        and "amplitude" in g
                        and "time_us" in g
                        and np.abs(file_v).max() > 0.0
                    ):
                        # The stored shape is peak-normalized; compare the
                        # unsigned envelope on the interpreter's time grid
                        # (signed amplitudes are covered by the instance
                        # table, full composition by the TR-waveform lane).
                        c_t = np.asarray(g["time_us"], dtype=float)
                        c_a = np.asarray(g["amplitude"], dtype=float)[0]
                        o = np.interp(c_t, file_t, file_v)
                        peak = np.abs(file_v).max() or 1.0
                        np.testing.assert_allclose(
                            np.abs(c_a) / max(np.abs(c_a).max(), 1e-12),
                            np.abs(o) / peak,
                            atol=5e-3,
                        )


def test_the_tr_waveforms_are_the_composed_file_gradients(case):
    coll, seqs = case
    actual = _safety.AMPLITUDE_MODES["actual"]
    for subseq, seq in enumerate(seqs):
        tr = wrapper._find_tr(coll, subseq)
        tr_size = tr["tr_size"]
        num = tr["num_tr_instances"]
        norot = instance_table(seq, GAMMA, B0)["norot_flag"]
        for tr_idx in {0, num - 1}:
            got = wrapper._get_tr_waveforms(coll, subseq, actual, tr_idx, False)
            start = (tr_idx % tr["num_trs"]) * tr_size
            times = np.asarray(got["gx"]["time_us"], dtype=float)
            raster = times[1] - times[0] if times.size > 1 else 10.0
            # A vertex that falls between raster points may be attributed
            # to either neighboring sample; accept a sample if the file
            # composition matches at t or one raster step to either side.
            candidates = [
                tr_window_waveforms(seq, start, tr_size, times + dt, norot)
                for dt in (0.0, -raster, raster)
            ]
            # A sample exactly on a block boundary is an attribution
            # convention (which side's amplitude and rotation it takes);
            # every interior sample must match.
            on_boundary = np.isin(times, block_boundaries(seq, start, tr_size))
            for a, axis in enumerate(("gx", "gy", "gz")):
                c_wave = np.asarray(got[axis]["amplitude"], dtype=float)
                mismatch = ~on_boundary
                for o_waves in candidates:
                    tol = np.maximum(1e-3 * np.abs(o_waves[a]), 1.0)
                    mismatch &= np.abs(c_wave - o_waves[a]) > tol
                assert not mismatch.any(), (
                    f"{axis} tr {tr_idx}: {int(mismatch.sum())} of {times.size} samples"
                    f" deviate, first at t={times[mismatch][0]} us"
                )


def test_the_meta_scalars_are_file_derived(case):
    coll, seqs = case
    for subseq, seq in enumerate(seqs):
        info = wrapper._find_tr(coll, subseq)
        tr_size = info["tr_size"]
        assert tr_size >= 1
        assert seq.num_blocks % tr_size == 0
        assert info["num_trs"] * tr_size == seq.num_blocks
        window_us = sum(
            round(float(seq.get_block(i + 1).block_duration) * 1e6)
            for i in range(tr_size)
        )
        assert info["tr_duration_us"] == pytest.approx(window_us, rel=1e-6)
