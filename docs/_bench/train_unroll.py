#!/usr/bin/env python3
"""Train the unrolled network the MoDL figure reconstructs with.

Documentation-only tooling, run by hand. It writes a second bundle beside the
plug-and-play denoiser, so the two learned figures show two different things:
a denoiser trained on its own and dropped into an optimizer, and an unroll
whose prior and algorithm parameters were trained together against the
reconstruction they produce.

The network is DeepInverse's proximal gradient descent taken ``unfold=True``
over Pulserver's :class:`~pulserver.recon.NormalEquationL2` data fidelity, so
each step costs one normal-operator apply rather than a transform pair, with a
DnCNN prior. Trainer, loss and metric are DeepInverse's.

The scan it is trained against is the one the figure reconstructs: Cartesian,
four-fold undersampled with a fully sampled centre, through the same analytic
coil ring, at the same matrix size. An unroll learns how much to trust its
prior against a particular sampling pattern, so training and inference are
held at one geometry.

The training set is the fastMRI demo subset less the slice the figure
reconstructs, so what the figure reports is what the unroll does on data it
has not seen. Three slices make this a worked example of the path rather than
a reconstruction anyone should scan with.

Usage::

    python docs/_bench/train_unroll.py [--steps N] [--size N] [--device cuda]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# The training set, its augmentation and its patch sampler are the denoiser
# run's; this trains a different network on the same data.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_denoiser import PatchDataset, training_slices

#: Matrix, batch, unrolled depth, and the acceleration the network is trained
#: at -- the geometry the learned figure reconstructs.
SIZE = 160
BATCH = 4
STEPS_UNROLLED = 6
ACCELERATION = 4
CENTRE = 16
COILS = 4

#: What the manifest records, so the bundle rebuilds itself at load time.
ARCHITECTURE = "deepinv.models.DnCNN"
ARCHITECTURE_KWARGS = {
    "in_channels": 2,
    "out_channels": 2,
    "depth": 8,
    "nf": 32,
    "pretrained": None,
}


def sampling_mask(size: int) -> torch.Tensor:
    """Uniform phase-encode undersampling with a fully sampled centre."""
    mask = torch.zeros(1, 1, size, size)
    mask[..., ::ACCELERATION, :] = 1.0
    first = (size - CENTRE) // 2
    mask[..., first : first + CENTRE, :] = 1.0
    return mask


def coil_maps(size: int, coils: int = COILS) -> torch.Tensor:
    """The analytic receive array the figures measure through."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _figures import brain

    return brain(size, coils=coils).coil_maps


def encoding(size: int, device: str):
    """The physics both this training run and the figure reconstruct through."""
    from pulserver.recon import Cartesian2D

    return Cartesian2D(
        sampling_mask(size).to(device),
        coil_maps(size).to(device),
        viewed_as_real=True,
    )


def build(device: str):
    """Return the unfolded optimizer to train."""
    import deepinv

    from pulserver.recon import NormalEquationL2

    prior = deepinv.optim.PnP(deepinv.models.DnCNN(**ARCHITECTURE_KWARGS))
    return deepinv.optim.PGD(
        data_fidelity=NormalEquationL2(),
        prior=prior,
        params_algo={"stepsize": 1.0, "g_param": 0.05},
        max_iter=STEPS_UNROLLED,
        unfold=True,
    ).to(device)


def train(steps: int, device: str, size: int = SIZE, seed: int = 0) -> torch.nn.Module:
    """Train the unroll end to end and return it."""
    import deepinv

    torch.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        PatchDataset(training_slices(), length=steps * BATCH, patch=size),
        batch_size=BATCH,
    )
    model = build(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    started = time.perf_counter()
    deepinv.Trainer(
        model=model,
        physics=encoding(size, device),
        optimizer=optimizer,
        train_dataloader=loader,
        online_measurements=True,
        losses=deepinv.loss.SupLoss(metric=deepinv.metric.MSE()),
        metrics=deepinv.metric.PSNR(),
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps
        ),
        epochs=1,
        device=device,
        save_path=None,
        plot_images=False,
        show_progress_bar=False,
        non_blocking_transfers=False,
        verbose=True,
    ).train()
    print(f"{steps} steps in {time.perf_counter() - started:.0f}s")
    return model


def report(model: torch.nn.Module, device: str, size: int = SIZE) -> None:
    """Print what the unroll recovers, against the adjoint it starts from."""
    physics = encoding(size, device)
    dataset = PatchDataset(training_slices(), length=4, patch=size)
    truth = torch.stack([dataset[index] for index in range(4)]).to(device)
    with torch.no_grad():
        measured = physics.A(truth)
        adjoint = physics.A_adjoint(measured)
        reconstructed = model(measured, physics)

    def peak(value):
        error = torch.nn.functional.mse_loss(value, truth)
        return float(10 * torch.log10(truth.abs().max() ** 2 / error))

    print(f"  adjoint {peak(adjoint):5.2f} dB -> unrolled {peak(reconstructed):5.2f} dB")
    if peak(reconstructed) <= peak(adjoint):
        raise SystemExit("the unroll did not improve on the adjoint it starts from")


def trained_parts(model) -> tuple[torch.nn.Module, dict[str, float]]:
    """Split the trained unroll into the network and the scalars.

    A bundle records an architecture its manifest can construct, and an
    unfolded optimizer is assembled rather than constructed. What is deployed
    is therefore the prior network, with the algorithm parameters the unroll
    learned alongside it recorded in the manifest, so rebuilding the same
    optimizer needs the bundle and nothing else.

    Returns
    -------
    tuple
        The prior's network, and the learned ``params_algo`` scalars.
    """
    network = model.prior[0].denoiser
    learned = {
        name.split(".")[1]: float(value.detach().reshape(-1)[0])
        for name, value in model.named_parameters()
        if name.startswith("params_algo.")
    }
    return network, learned


def main() -> None:
    """Train and write the bundle."""
    from pulserver.recon import save_bundle

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "_models",
    )
    parser.add_argument("--version", default="1.0")
    arguments = parser.parse_args()

    model = train(arguments.steps, arguments.device, arguments.size)
    report(model, arguments.device, arguments.size)
    network, learned = trained_parts(model.cpu())
    print("  learned " + ", ".join(f"{k}={v:.4f}" for k, v in sorted(learned.items())))
    bundle = save_bundle(
        network,
        arguments.out,
        name="fastmri-unroll",
        version=arguments.version,
        architecture=ARCHITECTURE,
        kwargs=ARCHITECTURE_KWARGS,
        metadata={
            "trained_on": (
                "fastMRI demo subset, knee and brain, less the slice the "
                "learned figures reconstruct"
            ),
            "size": arguments.size,
            "steps": arguments.steps,
            "acceleration": ACCELERATION,
            "calibration_lines": CENTRE,
            "coils": COILS,
            "algorithm": "deepinv.optim.PGD",
            "data_fidelity": "pulserver.recon.NormalEquationL2",
            "max_iter": STEPS_UNROLLED,
            "params_algo": learned,
            "trainer": "deepinv.Trainer, SupLoss(MSE), PSNR",
        },
        promote=True,
    )
    print(f"wrote {bundle.manifest_path.parent}")


if __name__ == "__main__":
    main()
