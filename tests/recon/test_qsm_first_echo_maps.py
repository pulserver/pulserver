"""The QSM wrapper gives every echo after the first the first echo's coil sensitivities."""

import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pulserver.recon.handlers.qsm")

WRAPPER = Path(__file__).resolve().parents[2] / "src/pulserver/_zoo/recon/qsm.py"


@pytest.fixture
def wrapper(monkeypatch):
    from pulserver.recon.handlers import pics
    from pulserver.recon.handlers.qsm import QsmRecon

    monkeypatch.setattr(pics, "coil_maps", pics.coil_maps)
    monkeypatch.setattr(pics, "nlinv_estimate", pics.nlinv_estimate)
    monkeypatch.setattr(QsmRecon, "wave_image", QsmRecon.wave_image)
    spec = importlib.util.spec_from_file_location("zoo_qsm_wrapper", WRAPPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unit(echo):
    reference = types.SimpleNamespace(idx=types.SimpleNamespace(contrast=echo))
    return types.SimpleNamespace(
        data=types.SimpleNamespace(reference=reference),
        counters={"slice": 0, "set": 0},
        ref=object(),
    )


def test_cartesian_echoes_after_the_first_reuse_its_maps(wrapper, monkeypatch):
    estimated = []

    def estimate(context, data, *, estimate, required=True):
        estimated.append(int(data.data.reference.idx.contrast))
        maps = np.full((6, 2, 3, 3), len(estimated), np.complex64)
        context.coil_maps[0] = types.SimpleNamespace(maps=maps)
        return torch.tensor(maps)

    monkeypatch.setattr(wrapper, "_estimated_maps", estimate)
    context = types.SimpleNamespace(coil_maps={})
    maps = [wrapper._first_echo_maps(context, unit(e), estimate=None) for e in range(7)]
    assert estimated == [0]
    for m in maps[1:]:
        assert torch.equal(m, maps[0])


def test_wave_echoes_after_the_first_reuse_its_maps(wrapper, monkeypatch):
    calls = []

    def nlinv(kspace, traj=None, shape=None):
        calls.append(1)
        return torch.full(
            (6, *kspace.shape[1:]), float(len(calls)), dtype=torch.complex64
        )

    monkeypatch.setattr(wrapper, "_nlinv_estimate", nlinv)
    context = types.SimpleNamespace()
    maps = []
    for echo in range(7):
        wrapper._unit.current = (context, echo)
        maps.append(wrapper._first_echo_nlinv(torch.zeros(32, 4, 5, 6)))
    wrapper._unit.current = None
    assert len(calls) == 1
    assert all(torch.equal(m, maps[0]) for m in maps)
    # Outside a wave unit, every call estimates.
    wrapper._first_echo_nlinv(torch.zeros(32, 4, 5, 6))
    assert len(calls) == 2
