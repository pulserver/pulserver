"""An operator answers in the array module it was asked in.

mri-nufft decorators read the array module off the argument they were handed
and hand the result back in it, and the operators it composes with rely on
that: :class:`~mrinufft.operators.MRIFourierCorrected` reads its own module off
the field model and then multiplies the base operator's output by it. An
operator that answers in Torch whatever it was asked in meets that
multiplication as a type error, so the contract is held here rather than left
to the composition that discovers it.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mrinufft._array_compat import get_array_module

from pulserver.recon import NonCartesian2D, OffResonance


def radial(spokes: int = 24, samples: int = 48) -> np.ndarray:
    angles = np.linspace(0, np.pi, spokes, endpoint=False)
    radius = np.linspace(-0.5, 0.5, samples, endpoint=False)
    return (
        np.stack(
            [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)], -1
        )
        .reshape(-1, 2)
        .astype(np.float32)
    )


def readout_time(spokes: int = 24, samples: int = 48) -> np.ndarray:
    return np.tile(np.linspace(0, 4e-3, samples, dtype=np.float32), spokes)


def field_map(size: int = 32) -> np.ndarray:
    hz = np.zeros((size, size), dtype=np.float32)
    hz[size // 4 : 3 * size // 4] = 80.0
    return hz


@pytest.mark.parametrize("backend", ["finufft", "cufinufft"])
def test_the_operator_answers_in_the_array_module_it_was_asked_in(backend):
    if backend == "cufinufft" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    physics = NonCartesian2D(radial(), (32, 32), backend=backend)
    native = physics.native_operator
    image = np.zeros((1, 32, 32), dtype=np.complex64)
    image[0, 16, 16] = 1.0

    asked = [("numpy", image), ("torch", torch.from_numpy(image))]
    if backend == "cufinufft":
        asked.append(("torch", torch.from_numpy(image).cuda()))
        cupy = pytest.importorskip("cupy")
        asked.append(("cupy", cupy.asarray(image)))

    for expected, argument in asked:
        assert get_array_module(native.op(argument)).__name__ == expected


@pytest.mark.parametrize("backend", ["finufft", "cufinufft"])
def test_off_resonance_corrects_a_real_non_cartesian_physics(backend):
    """The composition the array-module contract exists for.

    The correction is a sum over interpolators of the base operator's output
    weighted by the temporal factors, so base and factors have to meet in one
    array module for the multiplication to be defined at all.
    """
    if backend == "cufinufft" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    device = "cpu" if backend == "finufft" else "cuda"
    physics = NonCartesian2D(radial(), (32, 32), backend=backend)
    corrected = OffResonance(physics, field_map(), readout_time())

    image = torch.zeros(1, 32, 32, dtype=torch.complex64, device=device)
    image[0, 16, 16] = 1.0
    measured = corrected.A(image)
    restored = corrected.A_adjoint(measured)

    assert measured.device.type == device
    assert restored.device.type == device
    assert torch.isfinite(restored).all()
    assert float(restored.abs().max()) > 0.0


def test_the_correction_is_the_same_wherever_it_runs():
    """A field model on the device is the same model as one on the host."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    answers = {}
    for backend, device in (("finufft", "cpu"), ("cufinufft", "cuda")):
        physics = NonCartesian2D(radial(), (32, 32), backend=backend)
        corrected = OffResonance(physics, field_map(), readout_time())
        image = torch.zeros(1, 32, 32, dtype=torch.complex64, device=device)
        image[0, 16, 16] = 1.0
        answers[device] = corrected.A_adjoint(corrected.A(image)).cpu()

    difference = (answers["cpu"] - answers["cuda"]).abs().max()
    assert float(difference) < 1e-5
