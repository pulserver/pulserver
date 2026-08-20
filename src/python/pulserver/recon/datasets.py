"""TorchIO's volumetric MRI subjects through the DeepInverse dataset contract.

Every dataset here yields the sample layout
:class:`deepinv.datasets.ImageDataset` defines, so it drops into DeepInverse
training and evaluation loops unchanged. DeepInverse's own datasets -- the
MRI slice sets, the generic containers and samplers, and the natural-image
benchmarks -- are imported directly from :mod:`deepinv.datasets`.
"""

from __future__ import annotations

__all__ = [
    "IXI",
    "IXITiny",
    "TorchIODataset",
]

import copy
from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from os import PathLike
from typing import Any

import torch
from deepinv.datasets import ImageDataset as _ImageDataset

_Selector = str | Sequence[str] | Callable[[Any], torch.Tensor]


class TorchIODataset(_ImageDataset):
    """Expose TorchIO subjects through the DeepInverse dataset contract.

    Parameters
    ----------
    subjects
        Indexable TorchIO subjects, including the lists returned by
        ``torchio.datasets.ixi`` and ``torchio.datasets.ixi_tiny``.
    x
        Image name, image names concatenated along the channel dimension, or
        a callable extracting the DeepInverse ground truth from a subject.
    y
        Optional selector for precomputed measurements or targets.
    params
        Optional callable returning a string-keyed physics-parameter mapping.
    transform
        Optional TorchIO transform applied to the loaded subject.
    patch_sampler
        Optional TorchIO random patch sampler. It must accept a subject and
        ``num_patches`` in its call method.
    samples_per_volume
        Number of samples exposed for each subject in one epoch.
    """

    def __init__(
        self,
        subjects: Any,
        *,
        x: _Selector,
        y: _Selector | None = None,
        params: Callable[[Any], Mapping[str, Any]] | None = None,
        transform: Callable[[Any], Any] | None = None,
        patch_sampler: Callable[..., Any] | None = None,
        samples_per_volume: int = 1,
    ) -> None:
        if not hasattr(subjects, "__len__") or not hasattr(subjects, "__getitem__"):
            raise TypeError("subjects must be an indexable collection")
        if len(subjects) < 1:
            raise ValueError("subjects must contain at least one item")
        if (
            not isinstance(samples_per_volume, int)
            or isinstance(samples_per_volume, bool)
            or samples_per_volume < 1
        ):
            raise ValueError("samples_per_volume must be a positive integer")
        self.subjects = subjects
        self.x = _selector(x, name="x")
        self.y = None if y is None else _selector(y, name="y")
        self.params = params
        self.transform = transform
        self.patch_sampler = patch_sampler
        self.samples_per_volume = samples_per_volume

    def __len__(self) -> int:
        """Return the number of samples exposed in one epoch."""
        return len(self.subjects) * self.samples_per_volume

    def __getitem__(self, index: int) -> Any:
        """Return a tensor or tuple accepted by DeepInverse."""
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("dataset indices must be integers")
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError("dataset index out of range")

        subject = copy.deepcopy(self.subjects[index // self.samples_per_volume])
        if self.transform is not None:
            subject = self.transform(subject)
        if self.patch_sampler is not None:
            patches = iter(self.patch_sampler(subject, num_patches=1))
            try:
                subject = next(patches)
            except StopIteration as error:
                raise RuntimeError("patch_sampler returned no patches") from error

        ground_truth = _extract(subject, self.x, name="x")
        measurement = None if self.y is None else _extract(subject, self.y, name="y")
        physics_params = None
        if self.params is not None:
            physics_params = self.params(subject)
            if not isinstance(physics_params, Mapping) or any(
                not isinstance(key, str) for key in physics_params
            ):
                raise TypeError("params must return a string-keyed mapping")
            physics_params = dict(physics_params)

        if measurement is None:
            return (
                ground_truth
                if physics_params is None
                else (ground_truth, physics_params)
            )
        return (
            (ground_truth, measurement)
            if physics_params is None
            else (ground_truth, measurement, physics_params)
        )


class IXI(TorchIODataset):
    """TorchIO IXI dataset with the native DeepInverse sample contract.

    Parameters
    ----------
    root
        IXI dataset directory.
    download
        Download missing data through TorchIO.
    modalities
        IXI modalities to load. All selected modalities form ``x`` by
        channel-wise concatenation unless an explicit selector is supplied.
    x, y, params, transform, patch_sampler, samples_per_volume
        Arguments forwarded to :class:`TorchIODataset`.

    Raises
    ------
    ValueError
        If no modality is selected.
    ImportError
        If TorchIO is unavailable or does not provide IXI.
    """

    def __init__(
        self,
        root: str | PathLike[str],
        *,
        download: bool = False,
        modalities: Sequence[str] = ("T1", "T2"),
        x: _Selector | None = None,
        y: _Selector | None = None,
        params: Callable[[Any], Mapping[str, Any]] | None = None,
        transform: Callable[[Any], Any] | None = None,
        patch_sampler: Callable[..., Any] | None = None,
        samples_per_volume: int = 1,
    ) -> None:
        selected_modalities = tuple(modalities)
        if not selected_modalities:
            raise ValueError("modalities must contain at least one item")
        torchio = _torchio()
        datasets = torchio.datasets
        loader = getattr(datasets, "ixi", None)
        if callable(loader):
            subjects = loader(
                root,
                download=download,
                modalities=selected_modalities,
            )
        else:
            loader = getattr(datasets, "IXI", None)
            if loader is None:
                raise ImportError("installed TorchIO does not provide the IXI dataset")
            subjects = loader(
                root,
                download=download,
                modalities=selected_modalities,
                load_getitem=False,
            )
        super().__init__(
            subjects,
            x=selected_modalities if x is None else x,
            y=y,
            params=params,
            transform=transform,
            patch_sampler=patch_sampler,
            samples_per_volume=samples_per_volume,
        )


class IXITiny(TorchIODataset):
    """TorchIO IXITiny dataset with the native DeepInverse sample contract.

    Parameters
    ----------
    root
        IXITiny dataset directory.
    download
        Download missing data through TorchIO.
    x, y, params, transform, patch_sampler, samples_per_volume
        Arguments forwarded to :class:`TorchIODataset`. The default ``x`` is
        the anatomical image; ``y="label"`` selects its segmentation when
        needed for a supervised task.

    Raises
    ------
    ImportError
        If TorchIO is unavailable or does not provide IXITiny.
    """

    def __init__(
        self,
        root: str | PathLike[str],
        *,
        download: bool = False,
        x: _Selector = "image",
        y: _Selector | None = None,
        params: Callable[[Any], Mapping[str, Any]] | None = None,
        transform: Callable[[Any], Any] | None = None,
        patch_sampler: Callable[..., Any] | None = None,
        samples_per_volume: int = 1,
    ) -> None:
        torchio = _torchio()
        datasets = torchio.datasets
        loader = getattr(datasets, "ixi_tiny", None)
        if callable(loader):
            subjects = loader(root, download=download)
        else:
            loader = getattr(datasets, "IXITiny", None)
            if loader is None:
                raise ImportError(
                    "installed TorchIO does not provide the IXITiny dataset"
                )
            subjects = loader(root, download=download, load_getitem=False)
        super().__init__(
            subjects,
            x=x,
            y=y,
            params=params,
            transform=transform,
            patch_sampler=patch_sampler,
            samples_per_volume=samples_per_volume,
        )


def _selector(value: _Selector, *, name: str) -> _Selector:
    if callable(value) or isinstance(value, str):
        return value
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise TypeError(f"{name} must be an image name, image names, or callable")
    selected = tuple(value)
    if not selected or not all(isinstance(item, str) and item for item in selected):
        raise ValueError(f"{name} image names must be non-empty strings")
    return selected


def _extract(subject: Any, selector: _Selector, *, name: str) -> torch.Tensor:
    if callable(selector):
        value = selector(subject)
        return _tensor(value, name=name)
    names = (selector,) if isinstance(selector, str) else selector
    tensors = [_tensor(subject[key], name=f"{name}.{key}") for key in names]
    if len(tensors) == 1:
        return tensors[0]
    try:
        return torch.cat(tensors, dim=0)
    except RuntimeError as error:
        raise ValueError(f"{name} images cannot be concatenated by channel") from error


def _tensor(value: Any, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    data = getattr(value, "data", None)
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(value, Mapping) and isinstance(value.get("data"), torch.Tensor):
        return value["data"]
    raise TypeError(f"{name} must resolve to a Torch tensor or TorchIO image")


def _torchio() -> Any:
    try:
        return import_module("torchio")
    except ImportError as error:
        raise ImportError(
            "TorchIO datasets require TorchIO, which ships with "
            "pulserver; reinstall the package to restore it."
        ) from error
