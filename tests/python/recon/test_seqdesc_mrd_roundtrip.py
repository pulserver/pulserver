"""``to_mrd`` is the inverse of ``from_mrd``, word for word.

The authority for the wire format is ``src/cpp/recon/sequence_cache.cpp``
-- what the scanner actually emits. These tests pin the Python encoder to the
Python decoder, which is what catches a drift on either side; they cannot catch
the two drifting together away from the C++, so the field order in
``_encode_*`` has to stay readable against that file.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver.recon.simulation import (
    RfDefinition,
    RfShape,
    SequenceDescription,
    SequenceDescriptionCollection,
    SequenceParameters,
    ShimDefinition,
    decode_sequence_description,
)

ismrmrd = pytest.importorskip("ismrmrd")

WAVEFORM_IDS = [999, 1000, 1002, 1005]


@pytest.fixture
def system():
    return pp.Opts(
        max_grad=30,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )


def _collection(system, *, use="excitation", flip=30.0):
    """A real description, straight off a designed sequence."""
    seq = pp.Sequence(system=system)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf, gz, gzr = pp.make_sinc_pulse(
            flip_angle=np.deg2rad(flip),
            duration=2e-3,
            slice_thickness=3e-3,
            apodization=0.5,
            time_bw_product=4,
            system=system,
            return_gz=True,
            use=use,
        )
        gx = pp.make_trapezoid(
            channel="x", flat_area=128 / 0.256, flat_time=2.56e-3, system=system
        )
        adc = pp.make_adc(
            num_samples=128, duration=gx.flat_time, delay=gx.rise_time, system=system
        )
        pre = pp.make_trapezoid(
            channel="x", area=-gx.area / 2, duration=1e-3, system=system
        )
        for _ in range(6):
            seq.add_block(rf, gz)
            seq.add_block(pre, gzr)
            seq.add_block(gx, adc)

    parameters = seq.sequence_parameters()
    return seq, SequenceDescriptionCollection(
        parameters=SequenceParameters(
            num_subsequences=parameters["num_subseqs"],
            min_te_us=parameters["min_te_us"],
            min_tr_us=parameters["min_tr_us"],
            max_tr_us=parameters["max_tr_us"],
            max_flip_angle_deg=parameters["max_flip_angle_deg"],
            total_scan_time_us=parameters["total_scan_time_us"],
        ),
        subsequences={0: seq.sequence_descriptor},
    )


def test_the_round_trip_is_exact(system):
    """Equality of the whole object, not of the fields somebody remembered."""
    _seq, original = _collection(system)
    assert SequenceDescriptionCollection.from_mrd(original.to_mrd()) == original


def test_it_emits_the_waveform_ids_the_scanner_emits(system):
    _seq, collection = _collection(system)
    assert [int(w.waveform_id) for w in collection.to_mrd()] == WAVEFORM_IDS


def test_the_payload_is_single_channel_uint32(system):
    """What ``_payload_words`` requires, and what the scanner sends."""
    _seq, collection = _collection(system)
    for waveform in collection.to_mrd():
        assert np.asarray(waveform.data).dtype == np.uint32
        assert int(waveform.channels) == 1


def test_a_subsequence_alone_omits_the_scan_header(system):
    """It describes one TR, not the scan -- so it cannot carry waveform 999."""
    _seq, collection = _collection(system)
    waveforms = collection.subsequence(0).to_mrd()

    assert [int(w.waveform_id) for w in waveforms] == [1000, 1002, 1005]
    with pytest.raises(ValueError, match="header"):
        decode_sequence_description(waveforms)


def test_a_subsequence_round_trips_through_from_mrd(system):
    _seq, collection = _collection(system)
    description = collection.subsequence(0)
    stream = [*collection.to_mrd()]
    assert SequenceDescription.from_mrd(stream, 0) == description


@pytest.mark.parametrize(
    "use",
    ["excitation", "refocusing", "inversion", "saturation", "preparation", "other"],
)
def test_every_rf_use_survives_the_wire(system, use):
    """PREPARATION and OTHER are the two that used to raise on decode."""
    _seq, original = _collection(system, use=use)
    decoded = SequenceDescriptionCollection.from_mrd(original.to_mrd())
    assert decoded == original


def test_the_flip_angle_survives_the_wire(system):
    """The envelope and its time base are what the round trip is really for."""
    _seq, original = _collection(system, flip=37.5)
    decoded = SequenceDescriptionCollection.from_mrd(original.to_mrd()).subsequence(0)
    source = original.subsequence(0)

    event = next(e for e in decoded.events if e.type.name == "RF")
    before, _ = source.rf_definitions[event.rf_definition_id].flip_angle(
        event.rf_amplitude_hz, rf_raster_time_s=system.rf_raster_time
    )
    after, _ = decoded.rf_definitions[event.rf_definition_id].flip_angle(
        event.rf_amplitude_hz, rf_raster_time_s=system.rf_raster_time
    )
    assert np.rad2deg(after) == pytest.approx(37.5, rel=1e-4)
    assert after == pytest.approx(before, rel=1e-12)


def test_a_perturbed_payload_does_not_round_trip(system):
    """Mutation check: the equality above has to be able to fail."""
    _seq, original = _collection(system)
    waveforms = original.to_mrd()

    events = next(w for w in waveforms if int(w.waveform_id) == 1000)
    words = np.array(events.data, dtype=np.uint32, copy=True).reshape(-1)
    # Move the RF amplitude by one ulp of its float32 representation.
    words[3 + 2 + 2] += 1
    replaced = ismrmrd.Waveform.from_array(words.reshape(1, -1), waveform_id=1000)

    perturbed = [replaced if int(w.waveform_id) == 1000 else w for w in waveforms]
    assert SequenceDescriptionCollection.from_mrd(perturbed) != original


def test_shims_are_encoded_even_though_the_scanner_sends_none():
    """The scanner writes num_shims=0 unconditionally, but a description built
    here may hold them, and the round trip must not quietly drop them.

    The values are float32-exact on purpose. Every word on this wire is a
    float32 bit pattern, so a description assembled in Python out of float64
    literals comes back rounded -- ``pi/2`` returns as 1.5707963705062866.
    That is the transport, not a defect, and one a scanner never hits because
    its numbers were float32 to begin with.
    """
    description = SequenceDescription(
        subsequence_index=0,
        tr_duration_us=10_000.0,
        events=(),
        rf_definitions={},
        shim_definitions={
            0: ShimDefinition(0, (1.0, 0.5), (0.0, 1.5)),
            1: ShimDefinition(1, (0.25,), (1.0,)),
        },
    )
    collection = SequenceDescriptionCollection(
        parameters=SequenceParameters(1, 0.0, 10_000.0, 10_000.0, 0.0, 10_000.0),
        subsequences={0: description},
    )
    assert SequenceDescriptionCollection.from_mrd(collection.to_mrd()) == collection


def test_a_compressed_shape_crosses_still_compressed():
    """A scanner-sourced description holds shapes RLE-packed; encoding must
    move the packed words, not a decompressed copy of them."""
    packed = np.array([1.0, 1.0, 2.0, -1.0], dtype=np.float32)
    description = SequenceDescription(
        subsequence_index=0,
        tr_duration_us=5_000.0,
        events=(),
        rf_definitions={
            0: RfDefinition(
                id=0,
                bandwidth_hz=1234.5,
                num_bands=2,
                band_frequency_offsets_hz=(0.0, 440.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                band_bandwidth_hz=99.0,
                total_b1sq_power=7.5,
                magnitude=RfShape(5, packed),
                phase=RfShape(5, packed),
                time=None,
            )
        },
        shim_definitions={},
    )
    collection = SequenceDescriptionCollection(
        parameters=SequenceParameters(1, 0.0, 5_000.0, 5_000.0, 0.0, 5_000.0),
        subsequences={0: description},
    )

    decoded = SequenceDescriptionCollection.from_mrd(collection.to_mrd()).subsequence(0)
    shape = decoded.rf_definitions[0].magnitude
    assert shape.num_uncompressed == 5
    np.testing.assert_array_equal(shape.samples, packed)
    np.testing.assert_allclose(shape.decompress(), [1, 2, 3, 4, 3])
