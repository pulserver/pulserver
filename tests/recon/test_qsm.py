"""The qsm reconstruction: each echo as pics makes it, then the maps of the stated anatomy once the last echo is in."""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import calibration_header, coil_phantom
from test_pics import (
    COILS,
    MATRIX,
    Images,
    contrasts_in_header,
    reconstruct,
    stream,
    vials,
)

from pulserver.recon.handlers import qsm
from pulserver.recon.handlers.qsm import QsmRecon

pytest.importorskip("bartorch")


class Maps(QsmRecon):
    """Records the echoes and echo times each call to ``maps`` is given, and makes no map."""

    def __init__(self):
        super().__init__()
        self.results = []
        self.calls = []

    def spawn(self):
        clone = super().spawn()
        clone.results, clone.calls = self.results, self.calls
        return clone

    def recon(self, context, branch, data):
        result = super().recon(context, branch, data)
        self.results.extend(result if isinstance(result, list) else [result])
        return result

    def maps(self, context, data, echoes, te):
        self.calls.append((echoes, te))
        return []


def multi_echo(te_ms):
    echoes = [vials(), coil_phantom(COILS, MATRIX)[2]][: len(te_ms)]
    acquisitions = []
    for echo, kspace in enumerate(echoes):
        for acquisition in stream(kspace, range(MATRIX)):
            acquisition.idx.contrast = echo
            acquisitions.append(acquisition)
    header = contrasts_in_header(len(echoes))
    header.sequenceParameters = SimpleNamespace(TE=list(te_ms))
    return echoes, acquisitions, header


def test_each_echo_is_the_pics_image_and_the_maps_follow_the_last_echo_once(device):
    _, acquisitions, header = multi_echo([3.0, 8.0])
    plugin = Maps()

    reconstruct(acquisitions, header, device=device, plugin=plugin)
    pics = Images()
    reconstruct(acquisitions, header, device=device, plugin=pics)

    images = [
        np.asarray(result.data) for result in plugin.results if result is not None
    ]
    expected = [
        np.asarray(result.data) for result in pics.results if result is not None
    ]
    assert len(images) == len(expected) == 2
    for image, reference in zip(images, expected, strict=True):
        np.testing.assert_allclose(
            image, reference, rtol=1e-5, atol=1e-6 * reference.max()
        )
    ((echoes, te),) = plugin.calls
    assert echoes.shape == (2, MATRIX, MATRIX)
    assert np.iscomplexobj(echoes)
    np.testing.assert_allclose(
        np.abs(echoes), np.stack(expected), rtol=1e-5, atol=1e-6 * expected[0].max()
    )
    assert te == pytest.approx([3e-3, 8e-3])


def test_a_single_echo_scan_makes_no_maps(device):
    _, acquisitions, header = multi_echo([3.0])
    plugin = Maps()

    reconstruct(acquisitions, header, device=device, plugin=plugin)

    assert plugin.calls == []


@pytest.mark.parametrize(
    ("value", "anatomy"),
    [(None, "phantom"), ("0", "phantom"), ("1", "brain"), ("2", "body")],
)
def test_the_anatomy_is_the_qsm_anatomy_recon_parameter(value, anatomy):
    strings = [] if value is None else [SimpleNamespace(name="QsmAnatomy", value=value)]
    header = SimpleNamespace(
        userParameters=SimpleNamespace(
            userParameterLong=[], userParameterDouble=[], userParameterString=strings
        )
    )
    assert qsm._anatomy(header) == anatomy


def test_an_anatomy_out_of_range_is_refused():
    header = SimpleNamespace(
        userParameters=SimpleNamespace(
            userParameterLong=[],
            userParameterDouble=[],
            userParameterString=[SimpleNamespace(name="QsmAnatomy", value="3")],
        )
    )
    with pytest.raises(ValueError, match="QsmAnatomy"):
        qsm._anatomy(header)


def test_voxel_size_and_b0_come_from_the_header():
    header = calibration_header(4, 32, partitions=8)
    header.experimentalConditions = SimpleNamespace(H1resonanceFrequency_Hz=298.0e6)
    assert qsm._voxel_size(header) == pytest.approx((220 / 32, 220 / 32, 5.0))
    assert qsm._b0(header) == pytest.approx(298.0e6 / qsm.GAMMA_HZ_T)


def _vial_echoes(n=40, te=(4e-3, 8e-3, 12e-3), b0=7.0):
    """``(echoes, z, y, x)`` complex echoes of a water cylinder along z with a paramagnetic and a diamagnetic sphere."""
    import torch
    from fastqsm.lab.kernels import dipole_kernel, forward_field

    g = np.arange(n) - (n - 1) / 2
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    water = (x**2 + y**2 < (0.42 * n) ** 2) & (np.abs(z) < 0.42 * n)
    chi = np.zeros((n, n, n), np.float32)
    plus = (x + 0.18 * n) ** 2 + y**2 + z**2 < (0.1 * n) ** 2
    minus = (x - 0.18 * n) ** 2 + y**2 + z**2 < (0.1 * n) ** 2
    chi[plus], chi[minus] = 0.4, -0.2
    big = torch.zeros(2 * n, 2 * n, 2 * n)
    s = slice(n // 2, n // 2 + n)
    big[s, s, s] = torch.from_numpy(chi)
    field = forward_field(big, dipole_kernel(big.shape, (2.0, 2.0, 2.0)))[
        s, s, s
    ].numpy()
    hz = field * 1e-6 * qsm.GAMMA_HZ_T * b0
    # The vials are darker than the water, as their walls and contents make them.
    density = water * (100.0 - 20.0 * (plus | minus))
    signal = density * np.exp(-np.array(te)[:, None, None, None] / 30e-3)
    xyz = signal * np.exp(2j * np.pi * hz[None] * np.array(te)[:, None, None, None])
    return np.transpose(xyz, (0, 3, 2, 1)).astype(np.complex64), plus, minus, list(te)


def test_the_phantom_maps_are_rss_t2star_and_a_medi_qsm_with_positive_paramagnetic_vials():
    pytest.importorskip("fastqsm")
    pytest.importorskip("scipy")
    echoes, plus, minus, te = _vial_echoes()
    n = echoes.shape[-1]
    header = SimpleNamespace(
        experimentalConditions=SimpleNamespace(
            H1resonanceFrequency_Hz=7.0 * qsm.GAMMA_HZ_T
        ),
        encoding=[
            SimpleNamespace(
                reconSpace=SimpleNamespace(
                    fieldOfView_mm=SimpleNamespace(x=2.0 * n, y=2.0 * n, z=2.0 * n),
                    matrixSize=SimpleNamespace(x=n, y=n, z=n),
                )
            )
        ],
        userParameters=SimpleNamespace(
            userParameterLong=[],
            userParameterDouble=[],
            userParameterString=[SimpleNamespace(name="QsmAnatomy", value="0")],
        ),
    )
    reference = SimpleNamespace(
        read_dir=(0.0, 0.0, 0.0), phase_dir=(0.0, 0.0, 0.0), slice_dir=(0.0, 0.0, 1.0)
    )
    context = SimpleNamespace(header=header, device="cpu")
    data = SimpleNamespace(data=SimpleNamespace(reference=reference))

    results = QsmRecon().maps(context, data, echoes, te)

    series = {result.series_index: result for result in results}
    assert set(series) == {qsm.SERIES["rss"], qsm.SERIES["t2star"], qsm.SERIES["qsm"]}
    t2 = np.transpose(series[qsm.SERIES["t2star"]].data, (2, 1, 0))
    assert np.median(t2[plus]) == pytest.approx(30.0, rel=0.05)
    chi = np.transpose(series[qsm.SERIES["qsm"]].data, (2, 1, 0))
    assert np.median(chi[plus]) == pytest.approx(0.4, rel=0.2)
    assert np.median(chi[minus]) == pytest.approx(-0.2, rel=0.2)


def test_synthstrip_weights_are_the_named_file_else_the_shipped_one(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FASTQSM_SYNTHSTRIP_WEIGHTS", "/named.pt")
    assert qsm.synthstrip_weights() == "/named.pt"
    monkeypatch.delenv("FASTQSM_SYNTHSTRIP_WEIGHTS")
    monkeypatch.setattr(qsm.sys, "prefix", str(tmp_path))
    assert qsm.synthstrip_weights() is None
    shipped = tmp_path / qsm.SYNTHSTRIP_WEIGHTS
    shipped.parent.mkdir(parents=True)
    shipped.write_bytes(b"")
    assert qsm.synthstrip_weights() == str(shipped)
