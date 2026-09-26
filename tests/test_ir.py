import copy
import re
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest
from pypulseqpp import sequences

from pulserver import ir
from pulserver.ir import cache_path, chain, convert, summary
from pulserver.mrd import read_chain

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "sequences"
C_SOURCES = ROOT / "src" / "c"
CONTINUATIONS = {
    match.decode()
    for path in FIXTURES.glob("*.seq")
    for match in re.findall(rb"^NextSequence\s+(\S+)", path.read_bytes(), re.MULTILINE)
}
SEQUENCES = sorted(
    path.name for path in FIXTURES.iterdir() if path.name not in CONTINUATIONS
)
# The limits the library's C cache tests convert these fixtures under.
SYSTEM = pp.Opts(
    max_grad=40.0,
    grad_unit="mT/m",
    max_slew=170.0,
    slew_unit="T/m/s",
    B0=3.0,
    rf_raster_time=1e-6,
    grad_raster_time=1e-5,
    adc_raster_time=1e-7,
    block_duration_raster=1e-5,
)
VENDOR = 5
LABELS = (8, 7, 6)
# The PULSEG_RF_USE_* code of each RF use pypulseqpp tags.
RF_USES = {
    "excitation": 1,
    "refocusing": 2,
    "inversion": 3,
    "saturation": 4,
    "preparation": 5,
    "other": 6,
}


def _copy(name, directory):
    """Copy the fixture folder, so a chain's continuation files come along."""
    shutil.copytree(FIXTURES, directory, dirs_exist_ok=True)
    return directory / name


@pytest.mark.parametrize("name", SEQUENCES)
def test_a_converted_cache_reads_back_as_the_sequence_it_came_from(name, tmp_path):
    seq = _copy(name, tmp_path)
    assert convert(seq, SYSTEM) == cache_path(seq)
    assert summary(seq, SYSTEM, cache_ext=".pseg") == summary(seq, SYSTEM)


def _designed(seq_path):
    """Each block of a chain as its files state it, in the units the scanner plays."""
    rows = []
    for subsequence, (_, sequence) in enumerate(read_chain(seq_path)):
        for index in range(1, len(sequence) + 1):
            block = sequence.get_block(index)
            rf, adc = block.rf, block.adc
            rf_offsets = _absolute(rf)
            adc_offsets = _absolute(adc)
            gradients = [getattr(block, f"g{axis}") for axis in "xyz"]
            rows.append(
                {
                    "subsequence": subsequence,
                    "duration_us": round(sequence.block_durations[index] * 1e6),
                    "rf_amp_hz": 0.0 if rf is None else np.abs(rf.signal).max(),
                    "rf_freq_hz": rf_offsets[0],
                    "rf_phase_rad": rf_offsets[1],
                    "rf_use": 0 if rf is None else RF_USES[rf.use],
                    "rf_delay_us": 0 if rf is None else round(rf.delay * 1e6),
                    "gradient_hz_per_m": [
                        0.0 if g is None else g.amplitude for g in gradients
                    ],
                    "rotation": _matrix(block.rotation),
                    "adc": adc is not None,
                    "adc_freq_hz": adc_offsets[0],
                    "adc_phase_rad": adc_offsets[1],
                    "adc_delay_us": 0 if adc is None else round(adc.delay * 1e6),
                    "adc_dwell_ns": 0 if adc is None else round(adc.dwell * 1e9),
                    "adc_samples": 0 if adc is None else int(adc.num_samples),
                }
            )
    return {key: np.array([row[key] for row in rows]) for key in rows[0]}


def _absolute(event):
    """An event's frequency and phase offsets at the scanner's field; zero without one."""
    if event is None:
        return 0.0, 0.0
    return pp.calc_absolute_offsets(event, system=SYSTEM)


def _matrix(rotation):
    if rotation is None:
        return np.eye(3)
    w, x, y, z = np.asarray(rotation.quaternion) / np.linalg.norm(rotation.quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _played_as_designed(seq):
    played, designed = ir.play(seq), _designed(seq)
    assert played["subsequence"].tolist() == designed["subsequence"].tolist()
    assert played["duration_us"].tolist() == designed["duration_us"].tolist()
    assert played["adc"].astype(bool).tolist() == designed["adc"].tolist()
    for key in ("rf_use", "rf_delay_us", "adc_delay_us", "adc_dwell_ns", "adc_samples"):
        assert played[key].tolist() == designed[key].tolist(), key
    for key in ("rf_amp_hz", "rf_freq_hz", "adc_freq_hz"):
        np.testing.assert_allclose(
            played[key], designed[key], rtol=1e-6, atol=1e-6, err_msg=key
        )
    for key in ("rf_phase_rad", "adc_phase_rad"):
        wrapped = np.angle(np.exp(1j * (played[key] - designed[key])))
        np.testing.assert_allclose(wrapped, 0.0, atol=1e-6, err_msg=key)
    # A block turned by its rotation plays a wave; every other plays its events.
    driven = np.abs(designed["gradient_hz_per_m"]).max(axis=1) > 0.0
    turned = np.abs(designed["rotation"] - np.eye(3)).max(axis=(1, 2)) > 1e-6
    assert (played["wave"] >= 0)[driven & turned].all()
    unturned = played["wave"] < 0
    np.testing.assert_allclose(
        played["gradient_hz_per_m"][unturned],
        designed["gradient_hz_per_m"][unturned],
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("name", SEQUENCES)
def test_the_scanner_plays_every_block_as_its_file_designs_it(name, tmp_path):
    seq = _copy(name, tmp_path)
    convert(seq, SYSTEM)
    _played_as_designed(seq)


def _played_gradient(played, block, axis):
    """One axis of a played block: point times from its start, in s, and values."""
    start, stop = played["gradient_span"][block, axis]
    times = 1e-6 * played["gradient_time_us"][start:stop].astype(float)
    return times, played["gradient_waveform_hz_per_m"][start:stop].astype(float)


def test_a_rotated_block_plays_its_gradients_turned_by_its_rotation(tmp_path):
    seq = _copy("zte_3d.seq", tmp_path)
    convert(seq, SYSTEM)
    played = ir.play(seq, waveforms=True)
    _, sequence = read_chain(seq)[0]
    rotated = np.flatnonzero(played["wave"] >= 0)
    assert rotated.size
    for block in rotated:
        turned = sequence.get_gradients(block_range=(block + 1, block + 1))
        scale = np.abs(played["gradient_hz_per_m"][block]).max()
        for axis, spline in enumerate(turned):
            times, values = _played_gradient(played, block, axis)
            expected = np.zeros_like(times) if spline is None else spline(times)
            np.testing.assert_allclose(values, expected, atol=1e-4 * scale)


def _turned_on_the_raster(path, factor):
    """A lobe on the gradient raster beside a trapezoid, rotated; then both times ``factor``."""
    lobe = 2e5 * np.sin(np.pi * (np.arange(40) + 0.5) / 40)
    arbitrary = pp.make_arbitrary_grad("x", lobe, first=0.0, last=0.0, system=SYSTEM)
    trapezoid = pp.make_trapezoid(
        "y", amplitude=1e5, rise_time=1e-4, flat_time=2e-4, system=SYSTEM
    )
    rotation = pp.make_rotation(np.pi / 6)
    sequence = pp.Sequence(SYSTEM)
    sequence.add_block(arbitrary, trapezoid, rotation)
    sequence.add_block(
        pp.scale_grad(arbitrary, factor), pp.scale_grad(trapezoid, factor), rotation
    )
    sequence.write(path)
    return lobe, trapezoid, rotation


def test_a_rotated_wave_on_the_raster_keeps_the_area_of_the_gradients_it_turns(
    tmp_path,
):
    seq = tmp_path / "turned.seq"
    lobe, trapezoid, _ = _turned_on_the_raster(seq, 1.0)
    convert(seq, SYSTEM)
    played = ir.play(seq, waveforms=True)
    _, sequence = read_chain(seq)[0]
    angle = np.pi / 6
    turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    designed = turn @ [lobe.sum() * SYSTEM.grad_raster_time, trapezoid.area]
    turned = sequence.get_gradients(block_range=(1, 1))
    for axis in range(2):
        times, values = _played_gradient(played, 0, axis)
        np.testing.assert_allclose(
            np.trapezoid(values, times), designed[axis], rtol=1e-5
        )
        # The raster centres, between the two held edges, lie on the design.
        np.testing.assert_allclose(
            values[1:-1], turned[axis](times[1:-1]), rtol=1e-5, atol=1.0
        )


@pytest.mark.parametrize("factor", [0.5, -0.5])
def test_rotated_blocks_in_one_proportion_share_a_wave_at_their_own_amplitude(
    factor, tmp_path
):
    seq = tmp_path / "turned.seq"
    _turned_on_the_raster(seq, factor)
    convert(seq, SYSTEM)
    played = ir.play(seq)
    assert played["wave"].tolist() == [0, 0]
    np.testing.assert_allclose(
        played["gradient_hz_per_m"][1],
        factor * played["gradient_hz_per_m"][0],
        rtol=1e-5,
    )


def test_a_binary_file_segments_into_the_scan_its_text_does(tmp_path):
    seq = _copy("gre_2d_3sl.seq", tmp_path)
    binary = tmp_path / "gre_2d_3sl.bin"
    sequence = pp.Sequence()
    sequence.read(seq)
    sequence.write_binary(binary)
    assert summary(binary, SYSTEM) == summary(seq, SYSTEM)


def test_the_chain_lists_every_file_in_play_order(tmp_path):
    seq = _copy("dedup_gre_pair.seq", tmp_path)
    assert [p.name for p in chain(seq)] == [
        "dedup_gre_pair.seq",
        "dedup_gre_pair_b.seq",
    ]


def test_the_cache_is_named_by_the_extension_it_was_given(tmp_path):
    seq = _copy("gre_2d_3sl.seq", tmp_path)
    assert convert(seq, SYSTEM, cache_ext=".cache") == tmp_path / "gre_2d_3sl.cache"
    assert not (tmp_path / "gre_2d_3sl.pseg").exists()


def test_the_cache_header_carries_the_vendor_and_file_size_it_was_given(tmp_path):
    seq = _copy("gre_2d_3sl.seq", tmp_path)
    header = convert(seq, SYSTEM, vendor=VENDOR).read_bytes()[:24]
    _marker, _major, _minor, _revision, vendor, size = struct.unpack("<6i", header)
    assert (vendor, size) == (VENDOR, seq.stat().st_size)


def test_a_reader_built_for_another_vendor_refuses_the_cache(tmp_path):
    seq = _copy("gre_2d_3sl.seq", tmp_path)
    convert(seq, SYSTEM, vendor=VENDOR)
    with pytest.raises(ValueError, match="cannot load"):
        summary(seq, SYSTEM, cache_ext=".pseg")


def test_an_edited_file_is_refused_when_verification_is_asked(tmp_path):
    seq = _copy("gre_2d_3sl.seq", tmp_path)
    # An edit the sequence survives, so what refuses it is the signature.
    seq.write_text(seq.read_text().replace("TE 0.005", "TE 0.006", 1))
    with pytest.raises(ValueError):
        convert(seq, SYSTEM)
    assert convert(seq, SYSTEM, verify_signature=False).is_file()


def _rf_train(path, flips):
    """A pulse and a gradient per repetition, with ``flips[i]`` in repetition i."""
    sequence = pp.Sequence(pp.Opts())
    gradient = pp.make_trapezoid("x", flat_area=1000, flat_time=1e-3)
    for flip in flips:
        sequence.add_block(
            pp.make_sinc_pulse(flip_angle=flip, duration=1e-3, use="excitation")
        )
        sequence.add_block(gradient)
    sequence.write(path)
    return path


def test_a_subsequence_says_whether_its_rf_amplitude_varies_across_repetitions(
    tmp_path,
):
    varying = _rf_train(tmp_path / "varying.seq", [0.1, 0.2, 0.3, 0.4])
    constant = _rf_train(tmp_path / "constant.seq", [0.1] * 4)
    assert summary(varying, SYSTEM)["subsequences"][0]["rf_amplitude_variable"] == 1
    assert summary(constant, SYSTEM)["subsequences"][0]["rf_amplitude_variable"] == 0


def test_the_cache_carries_the_variable_rf_amplitude_flag(tmp_path):
    # What a reader does with the RF definitions depends on it: a variable
    # subsequence reports a positional-max envelope rather than one instance.
    seq = _rf_train(tmp_path / "varying.seq", [0.1, 0.2, 0.3, 0.4])
    convert(seq, SYSTEM)
    assert summary(seq, SYSTEM, cache_ext=".pseg") == summary(seq, SYSTEM)


class _ChainApp(sequences.SequenceApp):
    MAX_GRAD = 40.0
    MAX_SLEW = 150.0

    def init_sequence(self, n_repetitions: int = 3) -> None:
        self.n_repetitions = n_repetitions
        self.adc = pp.make_adc(num_samples=64, duration=3.2e-3, system=self.system)

    def prescans(self):
        return {"dummy": self._dummy}

    def _dummy(self) -> None:
        self.seq.add_block(pp.make_delay(5e-3))

    def loop(self) -> None:
        for _ in range(self.n_repetitions):
            self.kernel()

    def kernel(self) -> None:
        self.seq.add_block(self.adc)
        self.seq.add_block(pp.make_delay(5e-3))


def test_a_prescan_chain_converts_as_subsequences(tmp_path):
    first = Path(_ChainApp(SYSTEM).write(tmp_path / "sequence.seq", offline=True)[0])
    convert(first, SYSTEM)
    assert summary(first, SYSTEM)["num_subsequences"] == 2


def test_a_chain_plays_its_prescan_file_then_its_scan(tmp_path):
    first = Path(_ChainApp(SYSTEM).write(tmp_path / "sequence.seq", offline=True)[0])
    convert(first, SYSTEM)
    _played_as_designed(first)


def test_each_file_of_a_chain_carries_its_sar_ratios_into_the_cache(tmp_path):
    first = Path(_ChainApp(SYSTEM).write(tmp_path / "sequence.seq", offline=True)[0])
    ratios = [ir.SarRatio(0.25, 0.125), ir.SarRatio(0.75, 0.5)]
    convert(first, SYSTEM, sar_ratios=ratios)
    loaded = summary(first, SYSTEM, cache_ext=".pseg")["subsequences"]
    assert [
        ir.SarRatio(x["vop_sar_ratio"], x["vop_global_sar_ratio"]) for x in loaded
    ] == ratios


def test_a_chain_takes_one_sar_ratio_per_file(tmp_path):
    first = Path(_ChainApp(SYSTEM).write(tmp_path / "sequence.seq", offline=True)[0])
    with pytest.raises(ValueError, match="one SAR ratio per file"):
        convert(first, SYSTEM, sar_ratios=[ir.SarRatio(1.0, 1.0)])


def _reader_lines(s):
    lines = [
        f"num_subsequences {s['num_subsequences']}",
        f"num_segments {s['num_segments']}",
        f"max_adc_samples {s['max_adc_samples']}",
        f"total_readouts {s['total_readouts']}",
    ]
    for i, x in enumerate(s["subsequences"]):
        lines.append(
            f"subsequence {i} num_trs {x['num_trs']} tr_size {x['tr_size']} "
            f"num_unique_adcs {x['num_unique_adcs']} num_unique_rf {x['num_unique_rf']} "
            f"vop_sar_ratio {x['vop_sar_ratio']:g} "
            f"vop_global_sar_ratio {x['vop_global_sar_ratio']:g}"
        )
        lines += [
            f"group {i} {n} trid {g['trid']} num_instances {g['num_instances']} "
            f"one_instance_duration_us {g['one_instance_duration_us']}"
            for n, g in enumerate(x["tr_groups"])
        ]
        lines += [
            f"wave {i} {n} points {w['points']} peak "
            + " ".join(f"{v:.4f}" for v in w["peak"])
            for n, w in enumerate(x["waves"])
        ]
    lines += [
        f"segment {i} duration_us {x['duration_us']} num_blocks {x['num_blocks']} "
        f"start_block {x['start_block']} is_nav {x['is_nav']}"
        for i, x in enumerate(s["segments"])
    ]
    return lines


def _repetition_lines(seq, count):
    lines = []
    for i in range(count):
        r = ir.repetition_gradients(seq, i)
        lines.append(
            f"repetition {i} rc 1 first {r['first_position']} "
            f"points {r['time_us'].size} duration_us {r['duration_us']:.1f}"
        )
    return lines


@pytest.fixture(name="scanner_reader", scope="module")
def scanner_reader_fixture(tmp_path_factory):
    """The cache reader compiled as a scanner builds it: 32-bit, for one vendor."""
    directory = tmp_path_factory.mktemp("reader")
    probe = directory / "probe.c"
    probe.write_text("int main(void) { return 0; }\n")
    toolchain = subprocess.run(
        ["gcc", "-m32", str(probe), "-o", str(directory / "probe")],
        capture_output=True,
        check=False,
    )
    if toolchain.returncode != 0:
        pytest.skip("no 32-bit C toolchain")
    folders = ("pulseq", "core", "io", "structure", "cache", "playout")
    sources = [
        str(p) for folder in folders for p in sorted((C_SOURCES / folder).glob("*.c"))
    ]
    includes = [
        f"-I{C_SOURCES / sub}"
        for sub in ("", "include", "include/pulseg", "include/pulseq", "pulseq")
    ]
    output = directory / "read_cache_summary"
    subprocess.run(
        [
            "gcc",
            "-m32",
            "-std=c89",
            f"-DPULSEG_VENDOR={VENDOR}",
            *includes,
            str(ROOT / "tests" / "native" / "read_cache_summary.c"),
            *sources,
            "-lm",
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


@pytest.mark.parametrize(
    "name",
    [
        "gre_2d_3sl.seq",
        "epi_2d_main.seq",
        "mprage_stack_of_spirals_3d.seq",
        "zte_3d.seq",
    ],
)
def test_a_vendor_cache_written_here_loads_in_the_scanner_reader(
    name, tmp_path, scanner_reader
):
    seq = _copy(name, tmp_path)
    cache = convert(
        seq, SYSTEM, vendor=VENDOR, label_column_map=LABELS, cache_ext=".cache"
    )
    printed = subprocess.run(
        [str(scanner_reader), str(cache), str(seq.stat().st_size)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    expected = summary(seq, SYSTEM, label_column_map=LABELS)
    # This build loads vendor-neutral caches alone.
    convert(seq, SYSTEM)
    assert printed.splitlines() == _reader_lines(expected) + _repetition_lines(
        seq, expected["num_subsequences"]
    )


def _plan_lines(plan):
    lines = [
        f"wave_plan rc 1 mode {('none', 'resident', 'streamed').index(plan['mode'])} "
        f"samples {' '.join(str(n) for n in plan['samples'])}"
    ]

    def region(what, i, j, r):
        offsets = " ".join(str(o) for o in r["offset"])
        return (
            f"{what} {i} {j} offset {offsets} samples {r['samples']} "
            f"start_us {r['start_us']:.3f}"
        )

    for i, waves in enumerate(plan["waves"]):
        lines += [region("wave", i, j, r) for j, r in enumerate(waves)]
    for i, positions in enumerate(plan["slots"]):
        halves = [half for pair in positions for half in pair]
        lines += [region("slot", i, j, r) for j, r in enumerate(halves)]
    return lines


@pytest.mark.parametrize("max_samples", [10**6, 3000])
def test_the_scanner_reader_lays_out_the_rotated_waves_as_the_host_does(
    max_samples, tmp_path, scanner_reader
):
    seq = _copy("zte_3d.seq", tmp_path)
    cache = convert(
        seq, SYSTEM, vendor=VENDOR, label_column_map=LABELS, cache_ext=".cache"
    )
    printed = subprocess.run(
        [
            str(scanner_reader),
            str(cache),
            str(seq.stat().st_size),
            str(max_samples),
            "4",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    first = printed.index(next(x for x in printed if x.startswith("wave_plan")))
    last = printed.index(next(x for x in printed if x.startswith("prepare")))
    # This build loads vendor-neutral caches alone.
    convert(seq, SYSTEM)
    plan = ir.plan_waves(seq, ir.WaveBudget(max_samples, 4.0))
    assert plan["mode"] == ("resident" if max_samples == 10**6 else "streamed")
    assert printed[first:last] == _plan_lines(plan)


def _playout_lines(record):
    """What the scanner reader prints of both stages, from the host's record."""
    blocks = record["blocks"]
    loads = record["loads"]
    streamed = record["mode"] == "streamed"
    lines = [
        f"prepare rc 1 positions {record['positions']['segment'].size} "
        f"loads {0 if streamed else loads}"
    ]
    for n in range(blocks["segment"].size):
        if blocks["position"][n] == 0:
            lines.append(
                f"instance {blocks['subsequence'][n]} {blocks['segment'][n]} "
                f"{blocks['instance'][n]} {blocks['half'][n]} {blocks['first_position'][n]}"
            )
        if blocks["wave"][n] >= 0:
            offsets = " ".join(str(o) for o in blocks["wave_offset"][n])
            lines.append(
                f"block {n} wave {blocks['wave'][n]} offset {offsets} "
                f"samples {blocks['wave_samples'][n]}"
            )
    lines.append(
        f"scan rc 1 instances {record['instances']} blocks {blocks['segment'].size} "
        f"loads {loads if streamed else 0}"
    )
    return lines


@pytest.mark.parametrize("max_samples", [10**6, 3000])
def test_the_scanner_reader_plays_the_waves_as_the_host_records_them(
    max_samples, tmp_path, scanner_reader
):
    seq = _copy("zte_3d.seq", tmp_path)
    cache = convert(
        seq, SYSTEM, vendor=VENDOR, label_column_map=LABELS, cache_ext=".cache"
    )
    printed = subprocess.run(
        [
            str(scanner_reader),
            str(cache),
            str(seq.stat().st_size),
            str(max_samples),
            "4",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    first = printed.index(next(x for x in printed if x.startswith("prepare")))
    # This build loads vendor-neutral caches alone.
    convert(seq, SYSTEM)
    record = ir.playout(seq, ir.WaveBudget(max_samples, 4.0))
    assert record["mode"] == ("resident" if max_samples == 10**6 else "streamed")
    assert printed[first:] == _playout_lines(record)


def test_the_scanner_reader_reads_the_sar_ratios_a_cache_carries(
    tmp_path, scanner_reader
):
    seq = _copy("gre_2d_3sl.seq", tmp_path)
    cache = convert(
        seq,
        SYSTEM,
        vendor=VENDOR,
        label_column_map=LABELS,
        cache_ext=".cache",
        sar_ratios=[ir.SarRatio(0.75, 0.5)],
    )
    printed = subprocess.run(
        [str(scanner_reader), str(cache), str(seq.stat().st_size)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (line,) = (x for x in printed.splitlines() if x.startswith("subsequence 0 "))
    assert line.endswith("vop_sar_ratio 0.75 vop_global_sar_ratio 0.5")


def _readouts(sequence):
    for index in range(1, len(sequence) + 1):
        block = sequence.get_block(index)
        if block.adc is not None:
            yield block


def test_a_prescribed_offset_moves_every_readout_of_a_file_by_one_frequency():
    """A file names one ADC row from many blocks; each readout is moved once."""
    sequence = pp.Sequence()
    sequence.read(FIXTURES / "gre_2d_3sl.seq")

    ir.prescribe(sequence, (0.01, 0.0, 0.0))

    for block in _readouts(sequence):
        assert block.adc.freq_offset == pytest.approx(0.01 * block.gx.amplitude)


def test_a_prescribed_slice_offset_moves_every_excitation_by_one_frequency():
    designed, moved = pp.Sequence(), pp.Sequence()
    designed.read(FIXTURES / "gre_2d_3sl.seq")
    moved.read(FIXTURES / "gre_2d_3sl.seq")

    ir.prescribe(moved, (0.0, 0.0, 0.004))

    for index in range(1, len(moved) + 1):
        pulse = moved.get_block(index).rf
        if pulse is not None:
            added = pulse.freq_offset - designed.get_block(index).rf.freq_offset
            assert added == pytest.approx(0.004 * moved.get_block(index).gz.amplitude)


def test_a_zero_offset_writes_the_cache_as_designed_and_another_does_not(tmp_path):
    source = _copy("gre_2d_3sl.seq", tmp_path)
    caches = []
    for offset in (None, (0.0, 0.0, 0.0), (0.0, 0.02, 0.0)):
        caches.append(ir.convert(source, SYSTEM, fov_offset=offset).read_bytes())
    assert caches[0] == caches[1] != caches[2]


def test_an_offset_is_three_values():
    with pytest.raises(ValueError, match="three values"):
        ir.prescribe(pp.Sequence(), (0.01, 0.0))


def _lobe_waveform():
    """A sine lobe on the gradient raster, in Hz/m.

    Its samples sit half a raster inside the event, so the last one is live
    whatever value a library row stores for the lobe's end.
    """
    count = 50
    return 0.5 * SYSTEM.max_grad * np.sin(np.pi * (np.arange(count) + 0.5) / count)


def _lobe(path, last):
    """A pulse, then the lobe on x, starting at zero and ending at ``last`` Hz/m."""
    sequence = pp.Sequence(SYSTEM)
    sequence.add_block(
        pp.make_sinc_pulse(
            flip_angle=0.1, duration=1e-3, use="excitation", system=SYSTEM
        )
    )
    sequence.add_block(
        pp.make_arbitrary_grad(
            "x", _lobe_waveform(), first=0.0, last=last, system=SYSTEM
        )
    )
    sequence.write(path)
    return path


def test_a_gradient_its_file_ends_at_zero_may_end_the_repetition(tmp_path):
    assert _lobe_waveform()[-1] > 100.0
    assert ir.convert(_lobe(tmp_path / "lobe.seq", last=0.0), SYSTEM).is_file()


def test_a_gradient_its_file_ends_live_may_not_end_the_repetition(tmp_path):
    seq = _lobe(tmp_path / "live.seq", last=float(_lobe_waveform()[-1]))
    with pytest.raises(ValueError, match="does not end with zero gradient"):
        ir.convert(seq, SYSTEM)


def test_a_file_within_the_scanner_limits_has_no_problem():
    """The fixture's 20 us gradient raster times its waveforms, not the scanner's 10 us."""
    assert ir.check(FIXTURES / "gre_2d_3sl.seq", SYSTEM) == []


def test_a_gradient_beyond_the_scanner_limits_is_a_problem():
    weak = pp.Opts(max_grad=5.0, grad_unit="mT/m", max_slew=20.0, slew_unit="T/m/s")
    assert ir.check(FIXTURES / "gre_2d_3sl.seq", weak) == [
        "gradient amplitude of 39.7 mT/m on z in block 2 exceeds 5.0 mT/m",
        "slew rate of 165.3 T/m/s on z in block 2 exceeds 20.0 T/m/s",
    ]


def test_a_dead_time_the_file_does_not_leave_is_a_timing_problem():
    slow = copy.copy(SYSTEM)
    slow.adc_dead_time = 1e-3
    (problem,) = ir.check(FIXTURES / "gre_2d_3sl.seq", slow)
    assert problem.startswith("timing: Block ")
    assert "more errors" in problem


def test_each_problem_of_a_chain_names_its_file():
    weak = pp.Opts(max_grad=5.0, grad_unit="mT/m", max_slew=170.0, slew_unit="T/m/s")
    problems = ir.check(FIXTURES / "dedup_gre_pair.seq", weak)
    assert [problem.split(":")[0] for problem in problems] == [
        "dedup_gre_pair.seq",
        "dedup_gre_pair_b.seq",
    ]
