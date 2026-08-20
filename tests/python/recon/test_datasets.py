"""Tests for the TorchIO to DeepInverse dataset boundary."""

from __future__ import annotations

from types import SimpleNamespace

import deepinv
import torch

import pulserver.recon.datasets as recon_datasets
from pulserver.recon.datasets import IXI, IXITiny, TorchIODataset


class _Image:
    def __init__(self, data):
        self.data = data


class _Subject(dict):
    def __init__(self, **items):
        super().__init__(items)
        self.load_calls = 0

    def load(self):
        self.load_calls += 1


def _subject(offset: float = 0.0) -> _Subject:
    shape = (1, 4, 5, 6)
    return _Subject(
        t1=_Image(torch.full(shape, 1.0 + offset)),
        t2=_Image(torch.full(shape, 2.0 + offset)),
        measurement=_Image(torch.full(shape, 3.0 + offset)),
        mask=_Image(torch.ones(shape)),
    )


def test_the_datasets_module_exposes_only_original_code():
    """DeepInverse's own datasets are imported from deepinv, never re-exported."""
    assert recon_datasets.__all__ == ["IXI", "IXITiny", "TorchIODataset"]
    assert not hasattr(recon_datasets, "FastMRISliceDataset")
    assert not hasattr(recon_datasets, "check_dataset")


def test_natural_image_benchmarks_stay_in_deepinverse():
    """This namespace is for MRI; a super-resolution set has no place in it."""
    benchmarks = {
        "BSDS500",
        "CBSD68",
        "DIV2K",
        "FMD",
        "Flickr2kHR",
        "Kohler",
        "LsdirHR",
        "NBUDataset",
        "Set14HR",
        "Urban100HR",
    }
    assert benchmarks.isdisjoint(recon_datasets.__all__)
    assert all(hasattr(deepinv.datasets, name) for name in benchmarks)


def test_multimodal_subjects_follow_native_deepinverse_tuple_contract():
    subject = _subject()
    dataset = TorchIODataset(
        [subject],
        x=("t1", "t2"),
        y="measurement",
        params=lambda item: {"mask": item["mask"].data},
    )

    ground_truth, measurement, params = dataset[0]

    assert ground_truth.shape == (2, 4, 5, 6)
    torch.testing.assert_close(ground_truth[0], torch.ones(4, 5, 6))
    torch.testing.assert_close(ground_truth[1], torch.full((4, 5, 6), 2.0))
    torch.testing.assert_close(measurement, torch.full((1, 4, 5, 6), 3.0))
    assert set(params) == {"mask"}
    assert subject.load_calls == 0
    deepinv.datasets.check_dataset(dataset, allow_non_tensor=False)


def test_torchio_transform_runs_before_deepinverse_extraction():
    def transform(subject):
        subject["t1"] = _Image(subject["t1"].data + 4.0)
        return subject

    dataset = TorchIODataset([_subject()], x="t1", transform=transform)

    result = dataset[0]

    torch.testing.assert_close(result, torch.full((1, 4, 5, 6), 5.0))


def test_patch_sampler_exposes_map_style_samples_for_deepinverse():
    calls = []

    def sampler(subject, *, num_patches):
        calls.append((subject, num_patches))
        yield _Subject(t1=_Image(subject["t1"].data[:, :2, :3, :4]))

    subjects = [_subject(), _subject(10.0)]
    dataset = TorchIODataset(
        subjects,
        x="t1",
        patch_sampler=sampler,
        samples_per_volume=3,
    )

    first = dataset[2]
    second = dataset[3]

    assert len(dataset) == 6
    assert first.shape == (1, 2, 3, 4)
    torch.testing.assert_close(second, torch.full((1, 2, 3, 4), 11.0))
    assert [num_patches for _, num_patches in calls] == [1, 1]
    assert all(
        sample is not source
        for (sample, _), source in zip(calls, subjects, strict=True)
    )


def test_callable_selector_can_define_custom_deepinverse_sample():
    dataset = TorchIODataset(
        [_subject()],
        x=lambda subject: subject["t1"].data - subject["t2"].data,
    )

    result = dataset[0]

    torch.testing.assert_close(result, -torch.ones(1, 4, 5, 6))


def test_ixi_uses_torchio_v2_loader_and_concatenates_selected_modalities(
    monkeypatch,
):
    calls = {}

    def load_ixi(root, *, download, modalities):
        calls.update(root=root, download=download, modalities=modalities)
        return [_subject()]

    torchio = SimpleNamespace(
        datasets=SimpleNamespace(ixi=load_ixi),
    )
    monkeypatch.setattr(recon_datasets, "_torchio", lambda: torchio)

    dataset = IXI(
        "/datasets/ixi",
        download=True,
        modalities=("t1", "t2"),
        samples_per_volume=2,
    )
    result = dataset[0]

    assert calls == {
        "root": "/datasets/ixi",
        "download": True,
        "modalities": ("t1", "t2"),
    }
    assert len(dataset) == 2
    assert result.shape == (2, 4, 5, 6)
    deepinv.datasets.check_dataset(dataset, allow_non_tensor=False)


def test_ixi_tiny_supports_torchio_v1_and_optional_segmentation_target(
    monkeypatch,
):
    calls = {}
    subject = _Subject(
        image=_Image(torch.ones(1, 4, 5, 6)),
        label=_Image(torch.zeros(1, 4, 5, 6)),
    )

    def load_ixi_tiny(root, *, download, load_getitem):
        calls.update(
            root=root,
            download=download,
            load_getitem=load_getitem,
        )
        return [subject]

    torchio = SimpleNamespace(
        datasets=SimpleNamespace(IXITiny=load_ixi_tiny),
    )
    monkeypatch.setattr(recon_datasets, "_torchio", lambda: torchio)

    dataset = IXITiny("/datasets/ixi-tiny", y="label")
    image, label = dataset[0]

    assert calls == {
        "root": "/datasets/ixi-tiny",
        "download": False,
        "load_getitem": False,
    }
    torch.testing.assert_close(image, torch.ones(1, 4, 5, 6))
    torch.testing.assert_close(label, torch.zeros(1, 4, 5, 6))
