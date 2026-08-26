"""Tests for portable learned-reconstruction model bundles."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
import deepinv

from pulserver.recon.weights import (
    MODEL_PATH_ENV,
    ModelBundle,
    ModelStore,
    default_model_paths,
    load_model,
    save_bundle,
)


def _linear(weight: tuple[float, float]) -> torch.nn.Linear:
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([weight]))
    return model


def _factory(**kwargs):
    return torch.nn.Linear(**kwargs)


def _save(
    root: Path,
    *,
    name: str = "subspace-recon",
    version: str = "1.0",
    weight: tuple[float, float] = (2.0, -1.0),
) -> ModelBundle:
    return save_bundle(
        _linear(weight),
        root,
        name=name,
        version=version,
        architecture="test.linear",
        kwargs={"in_features": 2, "out_features": 1, "bias": False},
        metadata={"basis_size": 5, "complex_layout": "paired-real"},
    )


def test_bundle_round_trip_constructs_model_with_custom_factory(tmp_path):
    written = _save(tmp_path)

    bundle = ModelBundle.from_path(written.root)
    model = bundle.load(factory=_factory)

    assert bundle.identifier == "subspace-recon@1.0"
    assert bundle.metadata["basis_size"] == 5
    assert not model.training
    torch.testing.assert_close(model.weight, torch.tensor([[2.0, -1.0]]))


def test_bundle_can_populate_existing_model_and_preserve_training_mode(tmp_path):
    bundle = _save(tmp_path)
    model = _linear((0.0, 0.0))

    loaded = bundle.load(model, evaluation=False)

    assert loaded is model
    assert model.training
    torch.testing.assert_close(model.weight, torch.tensor([[2.0, -1.0]]))


def test_an_architecture_outside_the_allowed_roots_is_refused(tmp_path):
    """A manifest names what to build; it cannot reach outside the stack."""
    bundle = _save(tmp_path)

    with pytest.raises(ValueError, match="not an importable path"):
        bundle.load()


def test_a_manifest_naming_an_importable_class_builds_it_without_a_factory(tmp_path):
    bundle = save_bundle(
        _linear((2.0, -1.0)),
        tmp_path,
        name="linear",
        version="1.0",
        architecture="torch.nn.Linear",
        kwargs={"in_features": 2, "out_features": 1, "bias": False},
    )

    model = bundle.load()

    torch.testing.assert_close(model.weight, torch.tensor([[2.0, -1.0]]))


def test_promoting_a_version_makes_the_bare_name_resolve_to_it(tmp_path):
    _save(tmp_path, version="1.0", weight=(2.0, -1.0))
    save_bundle(
        _linear((3.0, 4.0)),
        tmp_path,
        name="subspace-recon",
        version="2.0",
        architecture="torch.nn.Linear",
        kwargs={"in_features": 2, "out_features": 1, "bias": False},
        promote=True,
    )

    resolved = ModelStore([tmp_path]).resolve("subspace-recon")

    assert resolved.version == "2.0"


def test_bundle_constructs_a_native_deepinverse_model(tmp_path):
    kwargs = {
        "in_channels": 2,
        "out_channels": 2,
        "channels_per_scale": [4, 8],
        "batch_norm": False,
    }
    model = deepinv.models.UNet(**kwargs)
    bundle = save_bundle(
        model,
        tmp_path,
        name="deepinv-unet",
        version="1.0",
        architecture="deepinv.models.UNet",
        kwargs=kwargs,
    )

    loaded = bundle.load(factory=deepinv.models.UNet)

    assert type(loaded) is deepinv.models.UNet
    for expected, actual in zip(
        model.state_dict().values(),
        loaded.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_store_resolves_exact_versions_and_rejects_ambiguous_names(tmp_path):
    first = _save(tmp_path, version="1.0")
    second = _save(tmp_path, version="2.0", weight=(3.0, 4.0))
    store = ModelStore([tmp_path])

    assert store.resolve("subspace-recon@1.0").root == first.root.resolve()
    assert store.resolve("subspace-recon@2.0").root == second.root.resolve()
    with pytest.raises(ValueError, match="ambiguous"):
        store.resolve("subspace-recon")


def test_store_uses_current_bundle_for_unversioned_name(tmp_path):
    _save(tmp_path, version="1.0")
    selected = _save(tmp_path, version="2.0", weight=(3.0, 4.0))
    current = tmp_path / "subspace-recon" / "current"
    current.symlink_to("2.0", target_is_directory=True)

    resolved = ModelStore([tmp_path]).resolve("subspace-recon")

    assert resolved.root == selected.root.resolve()
    assert resolved.version == "2.0"


def test_default_paths_and_loading_honor_environment_precedence(
    tmp_path,
    monkeypatch,
):
    configured = tmp_path / "configured"
    installed_prefix = tmp_path / "installation"
    installed = installed_prefix / "share" / "pulserver" / "models"
    _save(configured, weight=(5.0, 6.0))
    _save(installed, weight=(7.0, 8.0))
    monkeypatch.setenv(MODEL_PATH_ENV, str(configured))
    monkeypatch.setattr(sys, "prefix", str(installed_prefix))

    paths = default_model_paths()
    model = load_model("subspace-recon@1.0", factory=_factory)

    assert paths == (configured, installed)
    torch.testing.assert_close(model.weight, torch.tensor([[5.0, 6.0]]))


def test_bundle_detects_checkpoint_corruption(tmp_path):
    bundle = _save(tmp_path)
    bundle.checkpoint_path.write_bytes(bundle.checkpoint_path.read_bytes() + b"x")

    with pytest.raises(ValueError, match="digest mismatch"):
        bundle.state_dict()


def test_manifest_rejects_checkpoint_paths_outside_bundle(tmp_path):
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"checkpoint")
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    payload = {
        "schema_version": 1,
        "name": "unsafe",
        "version": "1.0",
        "architecture": "test.linear",
        "kwargs": {},
        "checkpoint": "../outside.pt",
        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        "metadata": {},
    }
    (bundle_root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="contained"):
        ModelBundle.from_path(bundle_root)


def test_bundle_identity_uses_single_path_components(tmp_path):
    with pytest.raises(ValueError, match="one path component"):
        save_bundle(
            _linear((1.0, 1.0)),
            tmp_path / "bundle",
            name="models/unsafe",
            version="1.0",
            architecture="test.linear",
        )


def test_lightning_style_state_dict_key_and_prefix_are_explicit(tmp_path):
    bundle_root = tmp_path / "lightning"
    bundle_root.mkdir()
    checkpoint = bundle_root / "weights.pt"
    torch.save(
        {"state_dict": {"model.weight": torch.tensor([[9.0, 10.0]])}},
        checkpoint,
    )
    payload = {
        "schema_version": 1,
        "name": "lightning-export",
        "version": "1.0",
        "architecture": "test.linear",
        "kwargs": {"in_features": 2, "out_features": 1, "bias": False},
        "checkpoint": checkpoint.name,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "metadata": {},
        "state_dict_key": "state_dict",
        "strip_prefix": "model.",
    }
    (bundle_root / "manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    model = ModelBundle.from_path(bundle_root).load(factory=_factory)

    torch.testing.assert_close(model.weight, torch.tensor([[9.0, 10.0]]))
