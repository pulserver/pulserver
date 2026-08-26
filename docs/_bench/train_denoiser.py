#!/usr/bin/env python3
"""Train the denoiser the plug-and-play figures reconstruct with.

Documentation-only tooling, run by hand. It writes a model bundle under
``docs/_models`` that :func:`pulserver.recon.load_model` resolves by name --
the same deployment path a scanner uses, exercised end to end.

The training set is the two-slice fastMRI demo subset DeepInverse
distributes, knee and brain, four slices in all. That is a real dataset and a
real training run, and it is small: the model this produces is a worked
example of the path, not a reconstruction anyone should scan with.

Two details the data forces. The slices are root-sum-of-square
reconstructions, so their imaginary part is identically zero; training on
them directly would teach the network that the imaginary channel is always
zero, which is useless for MRI. Every patch is therefore given a smooth
random phase. And a plug-and-play prior is called at a schedule of noise
levels, so the network takes the level as an input channel and is trained
across the range a reconstruction sweeps -- a blind denoiser would make the
regularization parameter inert.

Usage::

    python docs/_bench/train_denoiser.py [--steps N] [--out DIR]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

#: Patch side, training batch, and the noise range the prior is called over.
PATCH = 64
BATCH = 16
SIGMA_RANGE = (0.005, 0.205)

#: What the manifest records, so the bundle rebuilds itself at load time.
ARCHITECTURE = "deepinv.models.DnCNN"
#: A third input channel carries the noise level. DnCNN is residual, so the
#: output must be as wide as the input; the extra channel is discarded.
ARCHITECTURE_KWARGS = {
    "in_channels": 3,
    "out_channels": 3,
    "depth": 8,
    "nf": 32,
    "pretrained": None,
}


def training_slices() -> torch.Tensor:
    """Return the magnitude slices of the fastMRI demo subset."""
    from deepinv.datasets import SimpleFastMRISliceDataset
    from deepinv.utils import get_cache_home

    slices = []
    for anatomy in ("knee", "brain"):
        dataset = SimpleFastMRISliceDataset(
            get_cache_home(), anatomy=anatomy, download=True
        )
        slices += [dataset[index] for index in range(len(dataset))]
    return torch.stack(slices)[:, 0]


def _smooth_phase(count: int, size: int) -> torch.Tensor:
    coarse = torch.randn(count, 1, 8, 8)
    return (
        torch.nn.functional.interpolate(
            coarse, size=(size, size), mode="bicubic", align_corners=False
        )[:, 0]
        * 2.0
    )


def _patches(slices: torch.Tensor, count: int) -> torch.Tensor:
    span = slices.shape[-1] - PATCH
    index = torch.randint(0, slices.shape[0], (count,))
    rows = torch.randint(0, span, (count,))
    columns = torch.randint(0, span, (count,))
    cropped = torch.stack(
        [
            slices[which, row : row + PATCH, column : column + PATCH]
            for which, row, column in zip(index, rows, columns, strict=True)
        ]
    )
    complex_patch = cropped * torch.exp(1j * _smooth_phase(count, PATCH))
    return torch.stack([complex_patch.real, complex_patch.imag], 1)


def train(steps: int, seed: int = 0) -> torch.nn.Module:
    """Train the inner network and return it."""
    import deepinv

    from pulserver.recon import NoiseConditioned

    torch.manual_seed(seed)
    slices = training_slices()
    inner = deepinv.models.DnCNN(**ARCHITECTURE_KWARGS)
    model = NoiseConditioned(inner)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    low, high = SIGMA_RANGE
    started = time.perf_counter()
    for step in range(steps):
        clean = _patches(slices, BATCH)
        sigma = torch.rand(BATCH) * (high - low) + low
        noisy = clean + sigma.reshape(-1, 1, 1, 1) * torch.randn_like(clean)
        loss = torch.nn.functional.mse_loss(model(noisy, sigma), clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 400 == 0:
            print(f"  step {step:5d}  loss {float(loss.detach()):.5f}")
    print(f"{steps} steps in {time.perf_counter() - started:.0f}s")
    return inner


def report(inner: torch.nn.Module) -> None:
    """Print what the network does, and check the noise level reaches it."""
    from pulserver.recon import NoiseConditioned

    model = NoiseConditioned(inner).eval()
    slices = training_slices()
    with torch.no_grad():
        clean = _patches(slices, 32)
        for sigma in (0.03, 0.08, 0.15):
            noisy = clean + sigma * torch.randn_like(clean)
            levels = torch.full((clean.shape[0],), sigma)

            def peak(value):
                error = torch.nn.functional.mse_loss(value, clean)
                return float(10 * torch.log10(1.0 / error))

            print(
                f"  sigma {sigma:.2f}: {peak(noisy):5.2f} dB "
                f"-> {peak(model(noisy, levels)):5.2f} dB"
            )
        sample = _patches(slices, 4)
        quiet, loud = model(sample, low := 0.01), model(sample, 0.2)
        del low
        if torch.allclose(quiet, loud):
            raise SystemExit(
                "the noise level does not reach the network: a blind denoiser "
                "leaves the regularization parameter inert"
            )
        print("  the noise level reaches the network")


def main() -> None:
    """Train and write the bundle."""
    from pulserver.recon import save_bundle

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "_models",
    )
    parser.add_argument("--version", default="1.0")
    arguments = parser.parse_args()

    inner = train(arguments.steps)
    report(inner)
    bundle = save_bundle(
        inner,
        arguments.out,
        name="fastmri-denoiser",
        version=arguments.version,
        architecture=ARCHITECTURE,
        kwargs=ARCHITECTURE_KWARGS,
        metadata={
            "trained_on": "fastMRI demo subset, knee and brain, 4 slices",
            "patch": PATCH,
            "steps": arguments.steps,
            "sigma_range": list(SIGMA_RANGE),
            "conditioning": "noise level as a third input channel",
        },
        promote=True,
    )
    print(f"wrote {bundle.manifest_path.parent}")


if __name__ == "__main__":
    main()
