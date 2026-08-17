"""Write→read fidelity of the text and binary sequence files.

The baseline guarantee of the file layer: a sequence written to disk and
read back describes the same experiment, event by event, with no reference
file required — the sequence built in memory IS the ground truth. Every
event family and extension the writers handle appears in the corpus below,
and the zoo's stress slots cover the full-sequence paths (echo trains,
rotations, explicit arms, collections).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import pulserver.pypulseq as pp

# ---------------------------------------------------------------------------
# Event-coverage corpus: one small sequence per feature the file carries
# ---------------------------------------------------------------------------


def _system():
    return pp.Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
    )


def _readout(system):
    """A flat-top trapezoid with the ADC riding it."""
    gx = pp.make_trapezoid(
        channel="x", flat_area=800.0, flat_time=640e-6, system=system
    )
    adc = pp.make_adc(
        num_samples=64, duration=640e-6, delay=gx.rise_time, system=system
    )
    return gx, adc


def build_trapezoids(system):
    seq = pp.Sequence(system=system)
    seq.add_block(
        pp.make_trapezoid(channel="x", area=200.0, system=system),
        pp.make_trapezoid(channel="y", area=-150.0, system=system),
        pp.make_trapezoid(channel="z", area=90.0, delay=40e-6, system=system),
    )
    seq.add_block(pp.make_trapezoid(channel="x", amplitude=1e6, duration=1e-3, system=system))
    return seq


def build_extended_trapezoid(system):
    seq = pp.Sequence(system=system)
    grad = pp.make_extended_trapezoid(
        channel="y",
        times=np.array([0.0, 200e-6, 700e-6, 1e-3]),
        amplitudes=np.array([0.0, 8e5, 8e5, 2e5]),
        system=system,
    )
    seq.add_block(grad)
    seq.add_block(
        pp.make_extended_trapezoid(
            channel="y",
            times=np.array([0.0, 300e-6]),
            amplitudes=np.array([2e5, 0.0]),
            system=system,
        )
    )
    return seq


def build_arbitrary_gradient(system):
    seq = pp.Sequence(system=system)
    raster = system.grad_raster_time
    t = np.arange(0, 1e-3, raster) + raster / 2
    wave = 8e5 * np.sin(2 * np.pi * t / 1e-3) * np.hanning(t.size)
    seq.add_block(pp.make_arbitrary_grad(channel="z", waveform=wave, system=system))
    return seq


def build_shaped_rf(system):
    seq = pp.Sequence(system=system)
    seq.add_block(
        pp.make_sinc_pulse(
            np.pi / 2,
            duration=1e-3,
            freq_offset=440.0,
            phase_offset=np.pi / 5,
            system=system,
            use="excitation",
            return_gz=False,
        )
    )
    seq.add_block(
        pp.make_gauss_pulse(
            np.pi, duration=2e-3, system=system, use="refocusing", return_gz=False
        )
    )
    seq.add_block(
        pp.make_block_pulse(np.pi / 4, duration=500e-6, system=system, use="excitation")
    )
    return seq


def build_arbitrary_rf(system):
    seq = pp.Sequence(system=system)
    n = 250
    envelope = np.hanning(n) * np.exp(1j * np.linspace(0, np.pi / 3, n))
    seq.add_block(
        pp.make_arbitrary_rf(
            signal=envelope,
            flip_angle=np.pi / 2,
            dwell=4e-6,
            system=system,
            use="excitation",
        )
    )
    return seq


def build_adiabatic_rf(system):
    seq = pp.Sequence(system=system)
    seq.add_block(
        pp.make_adiabatic_pulse(
            pulse_type="hypsec", duration=8e-3, system=system, use="inversion"
        )
    )
    return seq


def build_adc_offsets(system):
    seq = pp.Sequence(system=system)
    gx, _ = _readout(system)
    adc = pp.make_adc(
        num_samples=64,
        duration=640e-6,
        delay=gx.rise_time,
        freq_offset=1300.0,
        phase_offset=np.pi / 7,
        system=system,
    )
    seq.add_block(gx, adc)
    return seq


def build_adc_phase_modulation(system):
    seq = pp.Sequence(system=system)
    gx, _ = _readout(system)
    adc = pp.make_adc(
        num_samples=64,
        duration=640e-6,
        delay=gx.rise_time,
        phase_modulation=np.linspace(-np.pi / 2, np.pi / 2, 64),
        system=system,
    )
    seq.add_block(gx, adc)
    return seq


def build_pure_delay(system):
    seq = pp.Sequence(system=system)
    seq.add_block(pp.make_delay(3.7e-3))
    return seq


def build_labels(system):
    seq = pp.Sequence(system=system)
    gx, adc = _readout(system)
    seq.add_block(
        gx,
        adc,
        pp.make_label("LIN", "SET", 5),
        pp.make_label("SLC", "SET", 2),
        pp.make_label("NAV", "SET", 1),
    )
    seq.add_block(
        gx,
        adc,
        pp.make_label("LIN", "INC", 1),
        pp.make_label("NAV", "SET", 0),
        pp.make_label("REV", "SET", 1),
    )
    return seq


def build_soft_delay(system):
    seq = pp.Sequence(system=system)
    seq.add_block(
        pp.make_delay(2e-3),
        pp.make_soft_delay(numID=0, hint="TE", offset=1e-3, factor=2.0),
    )
    return seq


def build_rotations(system):
    seq = pp.Sequence(system=system)
    gx, adc = _readout(system)
    for angle in (0.0, 111.25, 222.5):
        seq.add_block(
            gx, adc, pp.make_rotation(Rotation.from_euler("z", angle, degrees=True))
        )
    seq.add_block(
        gx,
        adc,
        pp.make_rotation(Rotation.from_euler("zyx", [30.0, 40.0, 50.0], degrees=True)),
    )
    return seq


def build_rf_shim(system):
    seq = pp.Sequence(system=system)
    shim = np.array(
        [0.9, 0.0, 0.7, np.pi / 3, 0.5, -np.pi / 4, 0.3, np.pi], dtype=float
    )
    rf = pp.make_block_pulse(
        np.pi / 6, duration=1e-3, system=system, use="excitation"
    )
    seq.add_block(rf, pp.make_rf_shim(shim))
    return seq


def build_triggers(system):
    seq = pp.Sequence(system=system)
    seq.add_block(
        pp.make_delay(1e-3),
        pp.make_trigger(channel="physio1", delay=100e-6, duration=200e-6, system=system),
    )
    seq.add_block(
        pp.make_delay(1e-3),
        pp.make_digital_output_pulse(
            channel="osc0", delay=100e-6, duration=200e-6, system=system
        ),
    )
    return seq


BUILDERS = {
    "trapezoids": build_trapezoids,
    "extended_trapezoid": build_extended_trapezoid,
    "arbitrary_gradient": build_arbitrary_gradient,
    "shaped_rf": build_shaped_rf,
    "arbitrary_rf": build_arbitrary_rf,
    "adiabatic_rf": build_adiabatic_rf,
    "adc_offsets": build_adc_offsets,
    "adc_phase_modulation": build_adc_phase_modulation,
    "pure_delay": build_pure_delay,
    "labels": build_labels,
    "soft_delay": build_soft_delay,
    "rotations": build_rotations,
    "rf_shim": build_rf_shim,
    "triggers": build_triggers,
}


# ---------------------------------------------------------------------------
# Event-level equivalence
# ---------------------------------------------------------------------------

#: Amplitudes survive the file at the writer's print precision.
REL = 1e-5


def _close(a, b, *, rel=REL, abs_tol=1e-9):
    np.testing.assert_allclose(
        np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=rel, atol=abs_tol
    )


def _assert_rf_match(built, read):
    assert read.use == built.use
    _close(read.delay, built.delay)
    _close(read.freq_offset, built.freq_offset, abs_tol=1e-3)
    _close(read.phase_offset, built.phase_offset, abs_tol=1e-6)
    _close(read.shape_dur, built.shape_dur)
    scale = float(np.abs(np.asarray(built.signal)).max()) or 1.0
    np.testing.assert_allclose(
        np.asarray(read.signal), np.asarray(built.signal), atol=REL * scale, rtol=0
    )
    _close(read.t, built.t)


def _assert_gradient_match(built, read):
    assert read.type == built.type
    assert read.channel == built.channel
    _close(read.delay, built.delay)
    if built.type == "trap":
        _close(read.amplitude, built.amplitude, abs_tol=1e-2)
        _close(read.rise_time, built.rise_time)
        _close(read.flat_time, built.flat_time)
        _close(read.fall_time, built.fall_time)
    else:
        scale = float(np.abs(np.asarray(built.waveform)).max()) or 1.0
        np.testing.assert_allclose(
            np.asarray(read.waveform),
            np.asarray(built.waveform),
            atol=REL * scale,
            rtol=0,
        )
        _close(read.tt, built.tt)
        _close(read.first, built.first, abs_tol=REL * scale)
        _close(read.last, built.last, abs_tol=REL * scale)


def _assert_adc_match(built, read):
    assert read.num_samples == built.num_samples
    _close(read.dwell, built.dwell, rel=1e-9, abs_tol=1e-12)
    _close(read.delay, built.delay)
    _close(read.freq_offset, built.freq_offset, abs_tol=1e-3)
    _close(read.phase_offset, built.phase_offset, abs_tol=1e-6)
    built_pm = getattr(built, "phase_modulation", None)
    read_pm = getattr(read, "phase_modulation", None)
    if built_pm is None or np.asarray(built_pm).size == 0:
        assert read_pm is None or np.asarray(read_pm).size == 0
    else:
        np.testing.assert_allclose(
            np.asarray(read_pm), np.asarray(built_pm), atol=1e-5, rtol=0
        )


def _assert_rotation_match(built, read):
    if built is None:
        assert read is None
        return
    q1 = np.asarray(built if not hasattr(built, "rot_quaternion") else built.rot_quaternion, dtype=float)
    q2 = np.asarray(read if not hasattr(read, "rot_quaternion") else read.rot_quaternion, dtype=float)
    # q and -q are the same rotation; the file may store either sign.
    assert abs(float(np.dot(q1, q2))) == pytest.approx(1.0, abs=1e-6)


def assert_same_events(built, read):
    """Block-by-block event equivalence of two sequences."""
    assert read.num_blocks == built.num_blocks
    for index in range(1, built.num_blocks + 1):
        ours, theirs = built.get_block(index), read.get_block(index)
        _close(theirs.block_duration, ours.block_duration)

        assert (theirs.rf is None) == (ours.rf is None), index
        if ours.rf is not None:
            _assert_rf_match(ours.rf, theirs.rf)

        for channel in ("gx", "gy", "gz"):
            mine = getattr(ours, channel)
            other = getattr(theirs, channel)
            assert (other is None) == (mine is None), (index, channel)
            if mine is not None:
                _assert_gradient_match(mine, other)

        assert (theirs.adc is None) == (ours.adc is None), index
        if ours.adc is not None:
            _assert_adc_match(ours.adc, theirs.adc)

        _assert_rotation_match(ours.rotation, theirs.rotation)

        assert sorted(theirs.labels) == sorted(ours.labels), index

        assert (theirs.soft_delay is None) == (ours.soft_delay is None), index
        if ours.soft_delay is not None:
            assert theirs.soft_delay.hint == ours.soft_delay.hint
            assert theirs.soft_delay.numID == ours.soft_delay.numID
            _close(theirs.soft_delay.offset, ours.soft_delay.offset)
            _close(theirs.soft_delay.factor, ours.soft_delay.factor)

        mine_trigs = ours.triggers or []
        other_trigs = theirs.triggers or []
        assert len(other_trigs) == len(mine_trigs), index
        for a, b in zip(mine_trigs, other_trigs):
            assert b.channel == a.channel
            _close(b.delay, a.delay)
            _close(b.duration, a.duration)

        mine_shim = ours.rf_shim
        other_shim = theirs.rf_shim
        assert (other_shim is None) == (mine_shim is None), index
        if mine_shim is not None:
            np.testing.assert_allclose(
                np.asarray(other_shim, dtype=complex),
                np.asarray(mine_shim, dtype=complex),
                atol=1e-5,
                rtol=0,
            )

    for key, value in built.definitions.items():
        got = read.get_definition(key)
        if isinstance(value, str):
            assert got == value, key
        else:
            _close(got, value, abs_tol=1e-9)


def _roundtrip(seq, path, mode):
    if mode == "text":
        seq.write(path)
    else:
        seq.write_binary(path)
    back = pp.Sequence(system=seq.system)
    back.read(path)
    return back


# ---------------------------------------------------------------------------
# The corpus, through both file formats
# ---------------------------------------------------------------------------


@pytest.fixture
def system():
    return _system()


@pytest.mark.parametrize("mode", ["text", "binary"])
@pytest.mark.parametrize("name", BUILDERS)
def test_every_event_family_survives_the_file(name, mode, system, tmp_path):
    built = BUILDERS[name](system)
    built.set_definition("FOV", [0.24, 0.24, 0.005])
    built.set_definition("Name", name)

    back = _roundtrip(built, tmp_path / f"{name}.seq", mode)
    assert_same_events(built, back)


@pytest.mark.parametrize("name", BUILDERS)
def test_text_and_binary_describe_the_same_sequence(name, system, tmp_path):
    built = BUILDERS[name](system)
    from_text = _roundtrip(built, tmp_path / "as_text.seq", "text")
    from_binary = _roundtrip(built, tmp_path / "as_binary.seq", "binary")
    assert_same_events(from_text, from_binary)


# ---------------------------------------------------------------------------
# Zoo stress slots: whole sequences through the same guarantee
# ---------------------------------------------------------------------------


def _zoo_gre_2d():
    from pulserver.app import gre2D_sequence

    return gre2D_sequence.main(n_x=32, n_y=32, te=5e-3, tr=30e-3)


def _zoo_fse_3d():
    from pulserver.app import fse3D_sequence

    return fse3D_sequence.main(
        n_x=32,
        n_y=8,
        n_z=4,
        slab_thickness=64e-3,
        n_acs=4,
        n_acs_z=2,
        etl=8,
        te=60e-3,
        tr=250e-3,
        readout_bandwidth_hz=125e3,
        readout_crusher_cycles=1.0,
        elliptical=False,
    )


def _zoo_zte_3d():
    from pulserver.app import zte3D_sequence

    return zte3D_sequence.main(n_x=32, n_views=16, n_shots=4, readout_bandwidth_hz=125e3)


def _zoo_explicit_arms():
    from pulserver.app import mprage_stack_of_spirals3D_sequence

    return mprage_stack_of_spirals3D_sequence.main(
        n_x=32,
        n_z=4,
        slab_thickness=64e-3,
        n_arms=3,
        ti=200e-3,
        tr_outer=500e-3,
        readout_bandwidth_hz=125e3,
    )


ZOO = {
    "gre_2d": _zoo_gre_2d,
    "fse_3d": _zoo_fse_3d,
    "zte_3d": _zoo_zte_3d,
    "mprage_explicit_arms": _zoo_explicit_arms,
}


@pytest.mark.parametrize("mode", ["text", "binary"])
@pytest.mark.parametrize("name", ZOO)
def test_a_zoo_slot_survives_the_file(name, mode, tmp_path):
    built = ZOO[name]()
    back = _roundtrip(built, tmp_path / f"{name}.seq", mode)
    assert_same_events(built, back)


def test_the_epi_collection_chain_is_a_parser_writer_fixed_point(tmp_path):
    """Each file of the NextSequence chain re-emits byte-identically once
    parsed: reading loses nothing the writer knows how to say."""
    from pulserver.app import epi2D_sequence

    lead = tmp_path / "scan.seq"
    epi2D_sequence.main(n_x=32, n_y=16, write_seq=True, seq_filename=str(lead))

    chain = []
    cursor = lead
    while cursor is not None:
        chain.append(cursor)
        seq = pp.Sequence()
        seq.read(cursor)
        nxt = seq.get_definition("NextSequence")
        cursor = cursor.parent / nxt if nxt else None
    assert len(chain) >= 2

    for path in chain:
        seq = pp.Sequence()
        seq.read(path)
        again = tmp_path / f"again_{path.name}"
        seq.write(again)
        assert again.read_bytes() == path.read_bytes()
