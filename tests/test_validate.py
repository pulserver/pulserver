"""Checking that a sequence plays as it was written."""

import shutil
from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest

from pulserver.validate import read_waveform_xml, validate
from pulserver.validate._command import main as validate_command

SYSTEM = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
SEQUENCES = (
    "gre_2d_3sl.seq",
    "epi_2d_main.seq",
    "mprage_stack_of_spirals_3d.seq",
    "zte_3d.seq",
)


def _copied(name, tmp_path):
    path = tmp_path / name
    shutil.copy(FIXTURES / name, path)
    return path


def _recording(path, channels, end_time_us=1000.0):
    """Write a waveform recording holding ``channels`` of (time_us, G/cm)."""
    ids = {"gx": 0, "gy": 1, "gz": 2, "ssp": 3, "rho": 4, "theta": 5, "omega": 6}
    parts = [f'<PulseSequence name="t" endTime="{end_time_us}">']
    for name, (times, amplitudes) in channels.items():
        samples = "\n".join(
            f"{int(t)}\t{a:.6f}" for t, a in zip(times, amplitudes, strict=True)
        )
        parts.append(
            f'<sequencer id="{ids[name]}" title="Seq({ids[name]}) | {name.upper()}">'
            f'<data waveform="sequencerData">{samples}</data></sequencer>'
        )
    parts.append("</PulseSequence>")
    path.write_text("\n".join(parts))
    return path


# %% the recording


def test_a_recording_reads_back_its_gradients_in_physical_units(tmp_path):
    """A recording stores gauss per centimetre; a comparison works in mT/m."""
    path = _recording(tmp_path / "r.xml", {"gx": ([0, 100], [0.0, 1.5])})
    played = read_waveform_xml(path)
    assert played.end_time_us == 1000.0
    times, amplitudes = played.gradient_mt_per_m("gx")
    assert list(times) == [0, 100]
    assert amplitudes == pytest.approx([0.0, 15.0])


def test_a_channel_the_recording_does_not_hold_played_nothing(tmp_path):
    path = _recording(tmp_path / "r.xml", {"gx": ([0], [0.0])})
    times, amplitudes = read_waveform_xml(path).get("gy")
    assert times.size == 0
    assert amplitudes.size == 0


def test_a_recording_that_declares_a_document_type_is_refused(tmp_path):
    """Entity expansion needs one, and a recording has no use for one."""
    path = tmp_path / "bomb.xml"
    path.write_text('<!DOCTYPE x [<!ENTITY a "aa">]><PulseSequence endTime="1"/>')
    with pytest.raises(ValueError, match="document type"):
        read_waveform_xml(path)


def test_a_file_that_is_not_a_recording_is_refused(tmp_path):
    path = tmp_path / "other.xml"
    path.write_text("<Protocol/>")
    with pytest.raises(ValueError, match="not the <PulseSequence>"):
        read_waveform_xml(path)


# %% against the cache


@pytest.mark.parametrize("name", SEQUENCES)
def test_a_sequence_agrees_with_the_waveforms_its_cache_holds(name, tmp_path):
    """The conversion keeps the gradients: corners, rotations and arbitrary waves."""
    comparison = validate(_copied(name, tmp_path), system=SYSTEM)
    assert comparison.reference == "ir"
    assert comparison.agrees, str(comparison)


def test_checking_a_sequence_leaves_nothing_beside_it(tmp_path):
    """A check is not a reason to put a cache in the directory it read from."""
    seq = _copied("gre_2d_3sl.seq", tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    validate(seq, system=SYSTEM)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_a_cache_already_there_is_the_one_checked_against(tmp_path):
    """Converting again would check the sequence against a conversion of itself."""
    from pulserver import ir

    seq = _copied("gre_2d_3sl.seq", tmp_path)
    ir.convert(seq, SYSTEM, cache_ext=".pseg")
    stamp = ir.cache_path(seq, ".pseg").stat().st_mtime_ns
    validate(seq, system=SYSTEM)
    assert ir.cache_path(seq, ".pseg").stat().st_mtime_ns == stamp


def test_the_cache_check_says_what_it_checked_against(tmp_path):
    """A conversion and a playout wrong in the same way agree, so say which ran."""
    comparison = validate(_copied("gre_2d_3sl.seq", tmp_path), system=SYSTEM)
    assert "ir" in str(comparison)


# %% against a recording


def test_a_sequence_agrees_with_a_recording_of_itself(tmp_path):
    """The recording is built from what the sequence asks for, so it must agree."""
    seq = _copied("gre_2d_3sl.seq", tmp_path)
    asked = _gradients(seq)
    channels = {
        axis: (times, amplitudes / 10.0) for axis, (times, amplitudes) in asked.items()
    }
    played = _recording(tmp_path / "r.xml", channels)
    comparison = validate(seq, vendor="ge", played=played)
    assert comparison.reference == "played"
    assert comparison.agrees, str(comparison)


def test_a_recording_that_plays_a_gradient_the_sequence_never_asked_for_is_caught(
    tmp_path,
):
    """The whole point: a machine playing something else must not read as agreement."""
    seq = _copied("gre_2d_3sl.seq", tmp_path)
    asked = _gradients(seq)
    channels = {
        axis: (times, amplitudes / 10.0) for axis, (times, amplitudes) in asked.items()
    }
    times, amplitudes = channels["gx"]
    wrong = amplitudes.copy()
    wrong[len(wrong) // 2] += 1.0  # 10 mT/m where the sequence asks for none
    channels["gx"] = (times, wrong)
    played = _recording(tmp_path / "r.xml", channels)
    comparison = validate(seq, vendor="ge", played=played)
    assert not comparison.agrees
    assert not next(c for c in comparison.channels if c.channel == "gx").agrees


def test_a_recording_offset_in_time_agrees_once_the_offset_is_stated(tmp_path):
    """A machine records each timeline as it drove it; the offset is the machine's."""
    seq = _copied("gre_2d_3sl.seq", tmp_path)
    asked = _gradients(seq)
    channels = {
        axis: (times + 404.0, amplitudes / 10.0)
        for axis, (times, amplitudes) in asked.items()
    }
    played = _recording(tmp_path / "r.xml", channels)
    assert not validate(seq, vendor="ge", played=played).agrees
    assert validate(seq, vendor="ge", played=played, shift_us=404.0).agrees


# %% what it refuses


def test_a_machine_this_reads_nothing_of_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a machine"):
        validate(_copied("gre_2d_3sl.seq", tmp_path), vendor="acme")


def test_a_vendor_named_with_no_recording_and_no_tooling_says_so(tmp_path, monkeypatch):
    """Rather than quietly checking against the cache, which answers another question."""
    monkeypatch.setitem(__import__("sys").modules, "pulserver_gehc", None)
    with pytest.raises(FileNotFoundError, match="not installed here"):
        validate(_copied("gre_2d_3sl.seq", tmp_path), vendor="ge")


# %% the command


def test_the_command_reports_agreement(tmp_path, capsys):
    assert validate_command([str(_copied("gre_2d_3sl.seq", tmp_path))]) == 0
    assert "agrees with the ir waveforms" in capsys.readouterr().out


def test_the_command_fails_where_the_machine_played_something_else(tmp_path, capsys):
    seq = _copied("gre_2d_3sl.seq", tmp_path)
    asked = _gradients(seq)
    channels = {
        axis: (times, amplitudes / 10.0) for axis, (times, amplitudes) in asked.items()
    }
    times, amplitudes = channels["gx"]
    wrong = amplitudes.copy()
    wrong[len(wrong) // 2] += 1.0
    channels["gx"] = (times, wrong)
    played = _recording(tmp_path / "r.xml", channels)
    status = validate_command([str(seq), "--vendor", "ge", "--played", str(played)])
    assert status == 1
    assert "DIFFERS" in capsys.readouterr().out


#: The gradient raster a recording holds its samples on, in microseconds.
RASTER_US = 4.0


def _gradients(seq_path):
    """What a machine playing the sequence faithfully would record.

    Sampled on the gradient raster rather than at the sequence's own corners,
    because that is what a plotter writes: a value per raster point.
    """
    from pulserver.validate._compare import gradients_of_sequence

    asked = gradients_of_sequence(pp.io.read(seq_path))
    last = max(
        (float(np.asarray(t)[-1]) for t, _ in asked.values() if np.asarray(t).size),
        default=0.0,
    )
    times = np.arange(0.0, last + RASTER_US, RASTER_US)
    return {
        axis: (
            times,
            np.interp(times, np.asarray(t), np.asarray(a), left=0.0, right=0.0),
        )
        for axis, (t, a) in asked.items()
        if np.asarray(t).size
    }
