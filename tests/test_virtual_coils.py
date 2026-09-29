"""The virtual scanner's coils: their channels, their scaling at the isocentre, BART's models and field maps."""

import math
import sys

import numpy as np
import pypulseqpp as pp
import pytest
from _virtual import (
    FIELD_CHANNELS,
    FIELD_ISOCENTRE,
    FIELD_RESOLUTION,
    phantom,
    synthetic_sensitivities,
    write_fields,
)

from pulserver import ir, virtual
from pulserver.virtual import _coils

ISOCENTRE = np.zeros((1, 3))
SYSTEM = pp.Opts(B0=3.0)


@pytest.fixture
def modelled(monkeypatch):
    """Synthetic sensitivities in place of BART's models."""
    monkeypatch.setattr(_coils, "_sampled", synthetic_sensitivities)
    _coils._receive.cache_clear()
    _coils._transmit.cache_clear()
    yield
    _coils._receive.cache_clear()
    _coils._transmit.cache_clear()


def test_the_scanner_pairs_the_body_coil_with_48_receive_channels_and_8_ptx_channels_with_32():
    channels = {
        name: (coil.transmit_channels, coil.receive_channels)
        for name, coil in virtual.COILS.items()
    }

    assert channels == {
        "body": (1, 1),
        "body/head48": (1, 48),
        "head8/head32": (8, 32),
    }


def test_the_body_coil_is_one_channel_of_unit_sensitivity_each_way():
    body = virtual.COILS["body"]

    assert body.transmit(ISOCENTRE) is None
    assert body.receive(ISOCENTRE) is None
    assert body.default_shim is None


def test_receive_sensitivities_have_a_root_sum_of_squares_of_one_at_the_isocentre(
    modelled,
):
    received = virtual.COILS["head8/head32"].receive(ISOCENTRE)

    assert received.shape == (1, 32)
    assert np.sqrt(np.sum(np.abs(received) ** 2)) == pytest.approx(1.0, rel=1e-6)


def test_transmit_sensitivities_are_the_conjugates_of_the_receive_ones_of_their_model(
    modelled,
):
    points = np.array([[0.01, -0.03, 0.02], [0.05, 0.04, -0.06]])

    transmit = virtual.COILS["head8/head32"].transmit(points)
    receive = _coils._interpolated(_coils._receive(("HEAD_2D_8CH", 8)), points)

    ratio = transmit / np.conj(receive)
    np.testing.assert_allclose(ratio, ratio[0, 0], rtol=1e-5)
    assert ratio[0, 0].real > 0.0
    assert abs(ratio[0, 0].imag) < 1e-6 * ratio[0, 0].real


def test_sensitivities_are_interpolated_linearly_and_held_beyond_the_grid(modelled):
    step = _coils.MODEL_FOV / _coils._SAMPLES
    maps = _coils._receive(("HEAD_3D_64CH", 32)).values
    centre = _coils._SAMPLES // 2
    head32 = virtual.COILS["head8/head32"]

    halfway = head32.receive(np.array([[0.5 * step, 0.0, 0.0]]))[0]
    beyond = head32.receive(np.array([[0.0, 0.0, _coils.MODEL_FOV]]))[0]

    np.testing.assert_allclose(
        halfway,
        0.5 * (maps[:, centre, centre, centre] + maps[:, centre, centre, centre + 1]),
        rtol=1e-5,
    )
    np.testing.assert_allclose(beyond, maps[:, -1, centre, centre], rtol=1e-6)


def test_sensitivities_are_complex128_in_a_file_mapped_into_memory(modelled):
    head8 = virtual.COILS["head8/head32"]
    points = np.array([[0.01, -0.03, 0.02], [0.05, 0.04, -0.06]])

    for sensitivities in (head8.transmit(points), head8.receive(points)):
        assert isinstance(sensitivities, np.memmap)
        assert sensitivities.dtype == np.complex128
        assert sensitivities.flags.c_contiguous
    assert head8.transmit(np.zeros((0, 3))).shape == (0, 8)
    assert head8.receive(np.zeros((0, 3))).shape == (0, 32)


def test_a_two_dimensional_model_is_constant_along_z(modelled):
    head8 = virtual.COILS["head8/head32"]

    low, high = head8.transmit(np.array([[0.02, 0.01, -0.1], [0.02, 0.01, 0.1]]))

    np.testing.assert_array_equal(low, high)


def test_a_pulse_without_an_rf_shim_turns_the_isocentre_by_its_flip_angle_through_the_default_shim(
    modelled, tmp_path
):
    head8 = virtual.COILS["head8/head32"]
    seq = pp.Sequence(SYSTEM)
    seq.add_block(pp.make_block_pulse(math.pi / 2, duration=1e-3, system=SYSTEM))
    seq.add_block(pp.make_adc(4, duration=1e-3, system=SYSTEM))
    path = tmp_path / "excite.seq"
    seq.write(str(path))
    ir.convert(path, SYSTEM)
    spins = pp.Isochromats(ISOCENTRE, transmit=head8.transmit(ISOCENTRE))

    (readout,) = virtual.simulate(path, spins, default_shim=head8.default_shim)

    np.testing.assert_allclose(np.abs(readout), 1.0, rtol=1e-4)


def test_a_phantom_scanned_with_a_coil_takes_its_sensitivities_where_it_lies(
    modelled, monkeypatch
):
    head8 = virtual.COILS["head8/head32"]
    made = {}

    def isochromats(positions, **fields):
        made.update(fields, positions=positions)

    monkeypatch.setattr(pp, "Isochromats", isochromats)
    phantom(position=(0.01, 0.02, 0.0), coils=1).isochromats(2e-3, coil=head8)

    np.testing.assert_array_equal(made["transmit"], head8.transmit(made["positions"]))
    np.testing.assert_array_equal(made["receive"], head8.receive(made["positions"]))


def test_a_phantom_received_by_coils_of_its_own_is_not_scanned_with_a_coil():
    with pytest.raises(ValueError, match="coils of its own"):
        phantom(coils=2).isochromats(2e-3, coil=virtual.COILS["body/head48"])


def test_without_bartorch_a_head_coil_names_the_extra_that_samples_its_model(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "bartorch", None)
    monkeypatch.setitem(sys.modules, "bartorch.tools", None)
    _coils._receive.cache_clear()

    with pytest.raises(ImportError, match=r"pulserver\[coils\]"):
        virtual.COILS["body/head48"].receive(ISOCENTRE)


@pytest.mark.parametrize("name", ["body/head48", "head8/head32"])
def test_bartorch_samples_as_many_channels_of_barts_model_as_each_coil_takes(name):
    pytest.importorskip("bartorch")
    _coils._receive.cache_clear()
    _coils._transmit.cache_clear()
    coil = virtual.COILS[name]

    transmitted, received = coil.transmit(ISOCENTRE), coil.receive(ISOCENTRE)

    assert received.shape == (1, coil.receive_channels)
    assert np.sqrt(np.sum(np.abs(received) ** 2)) == pytest.approx(1.0, rel=1e-5)
    assert transmitted is None or transmitted.shape == (1, coil.transmit_channels)


@pytest.fixture
def fields(tmp_path):
    """A directory of every coil's field maps, and of VOPs for those that transmit."""
    return write_fields(tmp_path)


def _maps(fields, name):
    with np.load(fields / f"{name}.npz") as archive:
        return archive["plus"], archive["minus"], archive["mask"]


def _centres(voxels):
    """Physical positions of voxel centres ``(n, 3)`` of the maps :func:`write_fields` writes."""
    return FIELD_RESOLUTION * (np.asarray(voxels) - np.asarray(FIELD_ISOCENTRE))


def test_mapped_coils_are_the_scanners_coils_with_the_channels_of_their_files(fields):
    mapped = _coils.coils(fields)

    channels = {
        name: (coil.transmit_channels, coil.receive_channels)
        for name, coil in mapped.items()
    }

    assert channels == {
        "body": (FIELD_CHANNELS["body"], FIELD_CHANNELS["body"]),
        "body/head48": (FIELD_CHANNELS["body"], FIELD_CHANNELS["head48"]),
        "head8/head32": (FIELD_CHANNELS["head8"], FIELD_CHANNELS["head32"]),
    }


def test_a_mapped_coil_transmits_with_the_conjugate_of_minus_and_receives_with_that_of_plus(
    fields,
):
    head8 = _coils.coils(fields)["head8/head32"]
    _, minus, mask = _maps(fields, "head8")
    plus, _, _ = _maps(fields, "head32")
    voxels = np.argwhere(mask)[::7]
    points = _centres(voxels)

    for found, component in (
        (head8.transmit(points), minus),
        (head8.receive(points), plus),
    ):
        expected = np.conj(component[:, voxels[:, 0], voxels[:, 1], voxels[:, 2]]).T
        ratio = found / expected
        np.testing.assert_allclose(ratio, ratio[0, 0], rtol=1e-5)
        assert ratio[0, 0].real > 0.0
        assert abs(ratio[0, 0].imag) < 1e-6 * ratio[0, 0].real


def test_a_voxel_outside_the_body_takes_the_field_of_the_nearest_voxel_inside(fields):
    head8 = _coils.coils(fields)["head8/head32"]

    outside, nearest = head8.transmit(_centres([[0, 0, 3], [1, 0, 3]]))

    np.testing.assert_allclose(outside, nearest, rtol=1e-6)


def test_a_pulse_without_an_rf_shim_turns_the_isocentre_by_its_flip_angle_in_mapped_coils(
    fields, tmp_path
):
    head8 = _coils.coils(fields)["head8/head32"]
    seq = pp.Sequence(SYSTEM)
    seq.add_block(pp.make_block_pulse(math.pi / 2, duration=1e-3, system=SYSTEM))
    seq.add_block(pp.make_adc(4, duration=1e-3, system=SYSTEM))
    path = tmp_path / "excite.seq"
    seq.write(str(path))
    ir.convert(path, SYSTEM)
    spins = pp.Isochromats(ISOCENTRE, transmit=head8.transmit(ISOCENTRE))

    (readout,) = virtual.simulate(path, spins, default_shim=head8.default_shim)

    np.testing.assert_allclose(np.abs(readout), 1.0, rtol=1e-4)


@pytest.mark.parametrize(
    ("name", "transmit"), [("body", "body"), ("head8/head32", "head8")]
)
def test_a_mapped_coils_vop_drive_plays_a_pulse_at_its_amplitude_at_the_isocentre(
    fields, name, transmit
):
    coil = _coils.coils(fields)[name]
    _, minus, _ = _maps(fields, transmit)

    limits = coil.limits()

    assert limits["vop_file"] == str(fields / f"{transmit}_vops.npz")
    magnitudes, phases = np.reshape(
        [float(value) for value in limits["vop_default_shim"].split()], (-1, 2)
    ).T
    drive = float(limits["vop_drive_per_hz"]) * magnitudes * np.exp(1j * phases)
    np.testing.assert_allclose(drive / abs(drive), coil.default_shim, rtol=1e-6)
    # A channel's clockwise field, in T per unit drive, is half its minus.
    field = np.sum(drive * minus[(slice(None), *FIELD_ISOCENTRE)]) / 2.0
    assert abs(field) == pytest.approx(1.0 / SYSTEM.gamma, rel=1e-5)


def test_a_coil_of_barts_models_has_no_vops():
    assert virtual.COILS["head8/head32"].limits() == {}


def test_maps_solved_at_another_field_are_refused(fields):
    with pytest.raises(ValueError, match="solved at"):
        _coils.coils(fields, field_t=1.5)
