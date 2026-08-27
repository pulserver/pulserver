#!/usr/bin/env python3
"""Derive the contiguous brain slices the context-adapter figure denoises.

Documentation-only tooling, run by hand. TorchIO fetches IXITiny from a
third-party host, and every ``.. plot::`` executes on every documentation
build, so the download happens once here and the result is committed beside
the model bundles. The figure then reads a file and the build touches no
network.

What it writes is one subject's central slices, intensity-normalized and
cropped square, as a real tensor plus a manifest naming the subject and the
preprocessing, so the derivation can be repeated or argued with.

Usage::

    python docs/_bench/make_ixi_stack.py [--root DIR] [--slices N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

#: Where the stack lands, relative to the documentation root, and the name
#: :func:`_figures.ixi_stack` reads it back by.
NAME = "ixi-stack"


def _subject_volume(root: Path, download: bool) -> tuple[torch.Tensor, str]:
    """Return one IXITiny subject's volume and the identifier it came from."""
    from pulserver.recon import IXITiny

    dataset = IXITiny(root, download=download)
    volume = dataset[0]
    if isinstance(volume, (tuple, list)):
        volume = volume[0]
    volume = torch.as_tensor(volume).float().squeeze()
    if volume.ndim != 3:
        raise SystemExit(f"expected a volume, got shape {tuple(volume.shape)}")
    identifier = getattr(dataset.subjects[0], "get", lambda *_: None)("subject_id")
    return volume, str(identifier or "IXITiny subject 0")


def _central_slices(volume: torch.Tensor, count: int) -> torch.Tensor:
    """Return ``count`` central slices, stacked along the thinnest axis.

    IXITiny ships heavily downsampled volumes -- 83 by 44 by 55 -- so the
    slices are kept at the resolution they arrive in. Cropping them square
    would throw away most of a brain that is already small.
    """
    axis = int(torch.tensor(volume.shape).argmin())
    volume = volume.movedim(axis, 0)
    first = max((volume.shape[0] - count) // 2, 0)
    return volume[first : first + count]


def _normalized(slices: torch.Tensor) -> torch.Tensor:
    """Scale to a unit peak, which is the range the denoisers were trained on."""
    peak = slices.amax()
    if peak <= 0:
        raise SystemExit("the selected slices are empty")
    return (slices / peak).clamp(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.home() / ".cache" / "pulserver" / "ixi-tiny",
        help="where TorchIO keeps IXITiny",
    )
    parser.add_argument("--slices", type=int, default=16)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "_models",
    )
    arguments = parser.parse_args()

    volume, subject = _subject_volume(arguments.root, not arguments.no_download)
    slices = _normalized(_central_slices(volume, arguments.slices))

    destination = arguments.out / NAME
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(slices.to(torch.float16), destination / "slices.pt")
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "source": "TorchIO IXITiny through pulserver.recon.IXITiny",
                "subject": subject,
                "slices": int(slices.shape[0]),
                "shape": [int(size) for size in slices.shape[-2:]],
                "preprocessing": (
                    "central slices along the thinnest axis, kept at the "
                    "resolution IXITiny ships, scaled to a unit peak, "
                    "stored float16"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    written = (destination / "slices.pt").stat().st_size
    print(f"wrote {destination} ({written / 1024:.0f} KiB, {tuple(slices.shape)})")


if __name__ == "__main__":
    main()
