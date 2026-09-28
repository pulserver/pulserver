"""The virtual scanner's coils: their channels, their scaling at the isocentre, and BART's models."""

import math
import sys

import numpy as np
import pypulseqpp as pp
import pytest
from _virtual import phantom, synthetic_sensitivities

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


def test_the_scanner_has_a_body_coil_a_ptx_head_coil_and_two_receive_head_arrays():
    channels = {
        name: (coil.transmit_channels, coil.receive_channels)
        for name, coil in virtual.COILS.items()
    }

    assert channels == {
        "body": (1, 1),
        "head8": (8, 8),
        "head32": (1, 32),
        "head48": (1, 48),
    }


def test_the_body_coil_is_one_channel_of_unit_sensitivity_each_way():
    body = virtual.COILS["body"]

    assert body.transmit(ISOCENTRE) is None
    assert body.receive(ISOCENTRE) is None
    assert body.default_shim is None


def test_receive_sensitivities_have_a_root_sum_of_squares_of_one_at_the_isocentre(
    modelled,
):
    received = virtual.COILS["head32"].receive(ISOCENTRE)

    assert received.shape == (1, 32)
    assert np.sqrt(np.sum(np.abs(received) ** 2)) == pytest.approx(1.0, rel=1e-6)


def test_transmit_sensitivities_are_the_conjugates_of_the_receive_ones(modelled):
    head8 = virtual.COILS["head8"]
    points = np.array([[0.01, -0.03, 0.02], [0.05, 0.04, -0.06]])

    transmit, receive = head8.transmit(points), head8.receive(points)

    ratio = transmit / np.conj(receive)
    np.testing.assert_allclose(ratio, ratio[0, 0], rtol=1e-5)
    assert np.isreal(ratio[0, 0]) and ratio[0, 0].real > 0.0


def test_sensitivities_are_interpolated_linearly_and_held_beyond_the_grid(modelled):
    step = _coils.MODEL_FOV / _coils._SAMPLES
    maps = _coils._receive("HEAD_3D_64CH", 32)
    centre = _coils._SAMPLES // 2
    head32 = virtual.COILS["head32"]

    halfway = head32.receive(np.array([[0.5 * step, 0.0, 0.0]]))[0]
    beyond = head32.receive(np.array([[0.0, 0.0, _coils.MODEL_FOV]]))[0]

    np.testing.assert_allclose(
        halfway,
        0.5 * (maps[:, centre, centre, centre] + maps[:, centre, centre, centre + 1]),
        rtol=1e-5,
    )
    np.testing.assert_allclose(beyond, maps[:, -1, centre, centre], rtol=1e-6)


def test_a_two_dimensional_model_is_constant_along_z(modelled):
    head8 = virtual.COILS["head8"]

    low, high = head8.receive(np.array([[0.02, 0.01, -0.1], [0.02, 0.01, 0.1]]))

    np.testing.assert_array_equal(low, high)


def test_a_pulse_without_an_rf_shim_turns_the_isocentre_by_its_flip_angle_through_the_default_shim(
    modelled, tmp_path
):
    head8 = virtual.COILS["head8"]
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
    head8 = virtual.COILS["head8"]
    made = {}

    def isochromats(positions, **fields):
        made.update(fields, positions=positions)

    monkeypatch.setattr(pp, "Isochromats", isochromats)
    phantom(position=(0.01, 0.02, 0.0), coils=1).isochromats(2e-3, coil=head8)

    np.testing.assert_array_equal(made["transmit"], head8.transmit(made["positions"]))
    np.testing.assert_array_equal(made["receive"], head8.receive(made["positions"]))


def test_a_phantom_received_by_coils_of_its_own_is_not_scanned_with_a_coil():
    with pytest.raises(ValueError, match="coils of its own"):
        phantom(coils=2).isochromats(2e-3, coil=virtual.COILS["head32"])


def test_without_bartorch_a_head_coil_names_the_extra_that_samples_its_model(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "bartorch", None)
    monkeypatch.setitem(sys.modules, "bartorch.tools", None)
    _coils._receive.cache_clear()

    with pytest.raises(ImportError, match=r"pulserver\[coils\]"):
        virtual.COILS["head32"].receive(ISOCENTRE)


@pytest.mark.parametrize("name", ["head8", "head32", "head48"])
def test_bartorch_samples_as_many_channels_of_barts_model_as_each_coil_takes(name):
    pytest.importorskip("bartorch")
    _coils._receive.cache_clear()
    coil = virtual.COILS[name]

    received = coil.receive(ISOCENTRE)

    assert received.shape == (1, coil.receive_channels)
    assert np.sqrt(np.sum(np.abs(received) ** 2)) == pytest.approx(1.0, rel=1e-5)
