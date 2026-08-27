#!/usr/bin/env python3
"""Train the denoiser the plug-and-play figures reconstruct with.

Documentation-only tooling, run by hand. It writes a model bundle under
``docs/_models`` that :func:`pulserver.recon.load_model` resolves by name --
the same deployment path a scanner uses, exercised end to end.

The training set is the fastMRI demo subset DeepInverse distributes, knee and
brain, less the one slice the learned figures reconstruct -- so what those
figures report is what the network does on data it has not seen. That is a
real dataset and a real training run, and it is small: three slices make the
model a worked example of the path, not a reconstruction anyone should scan
with.

The run is DeepInverse's own: :class:`deepinv.Trainer` over a
:class:`deepinv.physics.Denoising` forward operator whose noise level is drawn
per batch by a :class:`deepinv.physics.generator.SigmaGenerator`, supervised
by :class:`deepinv.loss.SupLoss` and reported through
:class:`deepinv.metric.PSNR`.

Two details the data forces. The slices are root-sum-of-square
reconstructions, so their imaginary part is identically zero; training on
them directly would teach the network that the imaginary channel is always
zero, which is useless for MRI. A smooth random phase is therefore applied as
a :class:`deepinv.transform.Transform`, composed with flips and quarter
turns. And a plug-and-play prior is called at a schedule of noise levels, so
the network takes the level as an input channel and is trained across the
range a reconstruction sweeps -- a blind denoiser would make the
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


def training_slices(hold_out_figure: bool = True) -> torch.Tensor:
    """Return the magnitude slices of the fastMRI demo subset.

    Parameters
    ----------
    hold_out_figure : bool, optional
        Drop the slice the learned figures reconstruct. The demo subset is
        four slices and one of them is that slice, so a model trained on all
        four is scored on its own training data and the figure's numbers mean
        nothing. Holding it out is what makes them generalization numbers.

    Returns
    -------
    torch.Tensor
        ``(slices, rows, columns)``, real, as the subset distributes them.
    """
    from deepinv.datasets import SimpleFastMRISliceDataset
    from deepinv.utils import get_cache_home

    slices = []
    for anatomy in ("knee", "brain"):
        dataset = SimpleFastMRISliceDataset(
            get_cache_home(), anatomy=anatomy, download=True
        )
        slices += [dataset[index] for index in range(len(dataset))]
    stacked = torch.stack(slices)[:, 0]
    if not hold_out_figure:
        return stacked
    return stacked[[
        index
        for index in range(stacked.shape[0])
        if not _is_figure_slice(stacked[index])
    ]]


def _is_figure_slice(candidate: torch.Tensor) -> bool:
    """True when ``candidate`` is the slice the learned figures reconstruct."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _figures import brain

    shown = brain(candidate.shape[-1], coils=1).image[0].abs()
    return bool(
        torch.allclose(
            candidate / candidate.max(), shown / shown.max(), atol=1e-4
        )
    )


def smooth_phase_transform():
    """Give a real slice the phase an MRI acquisition would have put on it.

    Returns
    -------
    deepinv.transform.Transform
        A transform drawing a smooth random phase field and applying it to a
        two-channel real/imaginary image.
    """
    from deepinv.transform import Transform

    class SmoothPhase(Transform):
        """Multiply by ``exp(i phi)`` for a smooth random ``phi``."""

        def _get_params(self, x: torch.Tensor) -> dict:
            coarse = torch.randn(
                x.shape[0], 1, 8, 8, device=x.device, dtype=x.dtype
            )
            phase = torch.nn.functional.interpolate(
                coarse, size=x.shape[-2:], mode="bicubic", align_corners=False
            )
            return {"phase": phase[:, 0] * 2.0}

        def _transform(self, x: torch.Tensor, phase=None, **kwargs) -> torch.Tensor:
            rotation = torch.exp(1j * phase)
            complex_image = torch.complex(x[:, 0], x[:, 1]) * rotation
            return torch.stack([complex_image.real, complex_image.imag], 1)

    return SmoothPhase()


class PatchDataset(torch.utils.data.Dataset):
    """Random complex patches of the training slices.

    Parameters
    ----------
    slices : torch.Tensor
        Magnitude slices, stacked on the leading axis.
    length : int
        Patches drawn per epoch.
    patch : int, optional
        Patch side.
    """

    def __init__(self, slices: torch.Tensor, length: int, patch: int = PATCH):
        self.slices = slices
        self.length = length
        self.patch = patch
        self.transform = smooth_phase_transform()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        span = self.slices.shape[-1] - self.patch
        which = int(torch.randint(0, self.slices.shape[0], ()))
        row = int(torch.randint(0, span, ()))
        column = int(torch.randint(0, span, ()))
        cropped = self.slices[
            which, row : row + self.patch, column : column + self.patch
        ]
        real = torch.stack([cropped, torch.zeros_like(cropped)])
        return self.transform(real[None])[0]


class DenoiserUnderTraining(torch.nn.Module):
    """Present a noise-conditioned denoiser as a DeepInverse reconstructor.

    :class:`deepinv.Trainer` calls a model as ``model(y, physics)``; a
    conditioned denoiser is called with the noise level. This reads the level
    off the physics that produced the batch.

    Parameters
    ----------
    inner : torch.nn.Module
        The network to train.
    """

    def __init__(self, inner: torch.nn.Module):
        from pulserver.recon import NoiseConditioned

        super().__init__()
        self.inner = inner
        self.conditioned = NoiseConditioned(inner)

    def forward(self, y: torch.Tensor, physics, **kwargs) -> torch.Tensor:
        sigma = getattr(getattr(physics, "noise_model", None), "sigma", None)
        return self.conditioned(y, sigma)


def train(steps: int, seed: int = 0) -> torch.nn.Module:
    """Train the inner network and return it."""
    import deepinv

    torch.manual_seed(seed)
    slices = training_slices()

    physics = deepinv.physics.Denoising(deepinv.physics.GaussianNoise())
    low, high = SIGMA_RANGE
    generator = deepinv.physics.generator.SigmaGenerator(
        sigma_min=low, sigma_max=high
    )

    loader = torch.utils.data.DataLoader(
        PatchDataset(slices, length=steps * BATCH), batch_size=BATCH
    )
    inner = deepinv.models.DnCNN(**ARCHITECTURE_KWARGS)
    model = DenoiserUnderTraining(inner)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    started = time.perf_counter()
    deepinv.Trainer(
        model=model,
        physics=physics,
        optimizer=optimizer,
        train_dataloader=loader,
        physics_generator=generator,
        online_measurements=True,
        losses=deepinv.loss.SupLoss(metric=deepinv.metric.MSE()),
        metrics=deepinv.metric.PSNR(),
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps
        ),
        epochs=1,
        device="cpu",
        non_blocking_transfers=False,
        save_path=None,
        plot_images=False,
        show_progress_bar=False,
        verbose=True,
    ).train()
    print(f"{steps} steps in {time.perf_counter() - started:.0f}s")
    return inner


def report(inner: torch.nn.Module) -> None:
    """Print what the network does, and check the noise level reaches it."""
    from pulserver.recon import NoiseConditioned

    model = NoiseConditioned(inner).eval()
    dataset = PatchDataset(training_slices(), length=32)
    with torch.no_grad():
        clean = torch.stack([dataset[index] for index in range(32)])
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
        sample = clean[:4]
        if torch.allclose(model(sample, 0.01), model(sample, 0.2)):
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
            "trained_on": (
                "fastMRI demo subset, knee and brain, less the slice the "
                "learned figures reconstruct"
            ),
            "patch": PATCH,
            "steps": arguments.steps,
            "sigma_range": list(SIGMA_RANGE),
            "conditioning": "noise level as a third input channel",
            "trainer": "deepinv.Trainer, SupLoss(MSE), PSNR",
        },
        promote=True,
    )
    print(f"wrote {bundle.manifest_path.parent}")


if __name__ == "__main__":
    main()
