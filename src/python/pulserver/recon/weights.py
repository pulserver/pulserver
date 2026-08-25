"""Versioned model bundles for workstation and scanner deployment."""

from __future__ import annotations

__all__ = [
    "MODEL_PATH_ENV",
    "ModelBundle",
    "ModelStore",
    "default_model_paths",
    "load_model",
    "save_bundle",
]

import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The environment variable naming extra directories to search for a model,
#: separated the way the platform separates a path. What it names is looked in
#: before the defaults :func:`default_model_paths` returns.
MODEL_PATH_ENV = "PULSERVER_MODEL_PATH"
_MANIFEST = "manifest.json"
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModelBundle:
    """Validated model architecture and state-dict bundle.

    Parameters
    ----------
    manifest_path
        Path to the bundle manifest.
    name
        Deployment name used by reconstruction applications.
    version
        Application-defined model version.
    architecture
        Application-defined architecture identifier.
    kwargs
        Arguments used to construct the architecture.
    checkpoint_path
        Weights-only Torch checkpoint.
    sha256
        Expected checkpoint digest.
    metadata
        Application metadata retained without interpretation.
    state_dict_key
        Optional key containing the state dict in a checkpoint mapping.
    strip_prefix
        Optional prefix removed from every state-dict key.

    Examples
    --------
    What :func:`load_model` resolves a name to before it builds anything: where
    the manifest is, what architecture it describes, and the checkpoint it is
    paired with.

    A bundle is read off disk rather than constructed by hand::

        import pulserver.recon as recon

        bundle = recon.ModelStore().resolve("unrolled-vn:1.0")
        model = recon.load_model(bundle.manifest_path)
    """

    manifest_path: Path
    name: str
    version: str
    architecture: str
    kwargs: Mapping[str, Any]
    checkpoint_path: Path
    sha256: str
    metadata: Mapping[str, Any]
    state_dict_key: str | None = None
    strip_prefix: str | None = None

    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> ModelBundle:
        """Read and validate a bundle directory or manifest path.

        Parameters
        ----------
        path
            Bundle directory or ``manifest.json`` path.

        Returns
        -------
        ModelBundle
            Validated bundle description.
        """
        selected = Path(path).expanduser()
        manifest_path = selected / _MANIFEST if selected.is_dir() else selected
        if not manifest_path.is_file():
            raise FileNotFoundError(f"model manifest not found: {manifest_path}")
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid model manifest JSON: {manifest_path}") from error
        if not isinstance(payload, Mapping):
            raise TypeError("model manifest must contain a JSON object")
        return cls._from_mapping(manifest_path, payload)

    @classmethod
    def _from_mapping(
        cls,
        manifest_path: Path,
        payload: Mapping[str, Any],
    ) -> ModelBundle:
        allowed = {
            "schema_version",
            "name",
            "version",
            "architecture",
            "kwargs",
            "checkpoint",
            "sha256",
            "metadata",
            "state_dict_key",
            "strip_prefix",
        }
        unknown = set(payload) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown model manifest fields: {names}")
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(f"model manifest schema_version must be {_SCHEMA_VERSION}")
        name = _nonempty_string(payload.get("name"), name="name")
        version = _nonempty_string(payload.get("version"), name="version")
        _safe_component(name, name="name")
        _safe_component(version, name="version")
        architecture = _nonempty_string(
            payload.get("architecture"),
            name="architecture",
        )
        kwargs = payload.get("kwargs", {})
        metadata = payload.get("metadata", {})
        if not isinstance(kwargs, Mapping):
            raise TypeError("model manifest kwargs must be an object")
        if not isinstance(metadata, Mapping):
            raise TypeError("model manifest metadata must be an object")
        checkpoint_name = _nonempty_string(
            payload.get("checkpoint"),
            name="checkpoint",
        )
        checkpoint_path = _contained_path(manifest_path.parent, checkpoint_name)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"model checkpoint not found: {checkpoint_path}")
        digest = _digest_string(payload.get("sha256"))
        state_dict_key = _optional_string(
            payload.get("state_dict_key"),
            name="state_dict_key",
        )
        strip_prefix = _optional_string(
            payload.get("strip_prefix"),
            name="strip_prefix",
        )
        return cls(
            manifest_path=manifest_path.resolve(),
            name=name,
            version=version,
            architecture=architecture,
            kwargs=dict(kwargs),
            checkpoint_path=checkpoint_path,
            sha256=digest,
            metadata=dict(metadata),
            state_dict_key=state_dict_key,
            strip_prefix=strip_prefix,
        )

    @property
    def root(self) -> Path:
        """Return the bundle directory."""
        return self.manifest_path.parent

    @property
    def identifier(self) -> str:
        """Return the stable ``name@version`` deployment identifier."""
        return f"{self.name}@{self.version}"

    def verify(self) -> None:
        """Verify checkpoint integrity against the manifest digest.

        Raises
        ------
        ValueError
            If the checkpoint digest does not match the manifest.
        """
        actual = _sha256(self.checkpoint_path)
        if actual != self.sha256:
            raise ValueError(
                f"checkpoint digest mismatch for {self.identifier}: "
                f"expected {self.sha256}, got {actual}"
            )

    def state_dict(
        self,
        *,
        map_location: Any = "cpu",
        verify: bool = True,
    ) -> dict[str, Any]:
        """Load a weights-only state dict from the bundle.

        Parameters
        ----------
        map_location
            Torch checkpoint target device.
        verify
            Validate the checkpoint digest before loading.

        Returns
        -------
        dict[str, Any]
            State dict accepted by ``torch.nn.Module.load_state_dict``.
        """
        if verify:
            self.verify()
        try:
            import torch
        except ImportError as error:
            raise ImportError(
                "Loading model bundles requires Torch, which ships with "
                "pulserver; reinstall the package to restore it."
            ) from error
        loaded = torch.load(
            self.checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
        if self.state_dict_key is not None:
            if not isinstance(loaded, Mapping) or self.state_dict_key not in loaded:
                raise KeyError(
                    f"checkpoint does not contain state_dict_key "
                    f"{self.state_dict_key!r}"
                )
            loaded = loaded[self.state_dict_key]
        elif isinstance(loaded, Mapping) and isinstance(
            loaded.get("state_dict"), Mapping
        ):
            loaded = loaded["state_dict"]
        if not isinstance(loaded, Mapping) or not all(
            isinstance(key, str) for key in loaded
        ):
            raise TypeError("model checkpoint must contain a string-keyed state dict")
        state = dict(loaded)
        if self.strip_prefix is not None:
            if not all(key.startswith(self.strip_prefix) for key in state):
                raise ValueError(
                    f"every state-dict key must start with {self.strip_prefix!r}"
                )
            state = {
                key[len(self.strip_prefix) :]: value for key, value in state.items()
            }
        return state

    def load(
        self,
        model: Any | None = None,
        *,
        factory: Callable[..., Any] | None = None,
        device: Any | None = None,
        map_location: Any = "cpu",
        strict: bool = True,
        verify: bool = True,
        evaluation: bool = True,
    ) -> Any:
        """Construct and populate a model from this bundle.

        Parameters
        ----------
        model
            Existing native Torch or DeepInverse model.
        factory
            Native model factory used when ``model`` is ``None``. It receives
            the manifest ``kwargs`` without modification.
        device
            Device receiving the populated model.
        map_location
            Device used while deserializing checkpoint tensors.
        strict
            Require an exact state-dict match.
        verify
            Validate the checkpoint digest before loading.
        evaluation
            Put the populated model in evaluation mode.

        Returns
        -------
        torch.nn.Module
            Populated model.
        """
        if model is not None and factory is not None:
            raise ValueError("provide model or factory, not both")
        if model is None:
            if factory is None:
                raise ValueError(
                    "provide a native model or factory when loading a bundle"
                )
            model = factory(**dict(self.kwargs))
        load_state_dict = getattr(model, "load_state_dict", None)
        if not callable(load_state_dict):
            raise TypeError("model factories must return a Torch-compatible module")
        load_state_dict(
            self.state_dict(map_location=map_location, verify=verify),
            strict=strict,
        )
        if device is not None:
            model = model.to(device)
        if evaluation:
            model.eval()
        return model


class ModelStore:
    """Resolve exact model versions from scanner deployment directories.

    Parameters
    ----------
    paths
        Search roots. ``None`` uses :func:`default_model_paths`.

    Examples
    --------
    >>> import pulserver.recon as recon
    >>> store = recon.ModelStore()
    >>> tuple(store.paths) == tuple(recon.default_model_paths())
    True
    """

    def __init__(
        self,
        paths: Sequence[str | os.PathLike[str]] | None = None,
    ) -> None:
        selected = (
            default_model_paths() if paths is None else tuple(Path(p) for p in paths)
        )
        self.paths = _unique_paths(selected)

    def resolve(self, spec: str | os.PathLike[str]) -> ModelBundle:
        """Resolve a path, ``name@version``, or unambiguous model name.

        Parameters
        ----------
        spec
            Bundle path, model name, or exact ``name@version`` identifier.

        Returns
        -------
        ModelBundle
            Resolved bundle.
        """
        explicit = Path(spec).expanduser()
        if explicit.exists():
            return ModelBundle.from_path(explicit)
        text = os.fspath(spec)
        if _looks_like_path(text):
            raise FileNotFoundError(f"model bundle not found: {explicit}")
        name, separator, version = text.rpartition("@")
        if not separator:
            name, version = text, None
        elif not name or not version:
            raise ValueError("model identifiers use the form name@version")
        _safe_component(name, name="model name")
        if version is not None:
            _safe_component(version, name="model version")
        unique: tuple[Path, ...] = ()
        for root in self.paths:
            unique = _unique_paths(self._matches(root, name, version))
            if unique:
                break
        if not unique:
            roots = ", ".join(str(path) for path in self.paths) or "<none>"
            raise FileNotFoundError(f"model {text!r} not found in {roots}")
        if len(unique) > 1:
            options = ", ".join(str(path.parent) for path in unique)
            raise ValueError(f"model {text!r} is ambiguous: {options}")
        bundle = ModelBundle.from_path(unique[0])
        if bundle.name != name:
            raise ValueError(
                f"resolved manifest name {bundle.name!r} does not match {name!r}"
            )
        if version is not None and bundle.version != version:
            raise ValueError(
                f"resolved manifest version {bundle.version!r} does not match "
                f"{version!r}"
            )
        return bundle

    def load(self, spec: str | os.PathLike[str], **kwargs: Any) -> Any:
        """Resolve and load a model bundle."""
        return self.resolve(spec).load(**kwargs)

    @staticmethod
    def _matches(root: Path, name: str, version: str | None) -> list[Path]:
        model_root = root.expanduser() / name
        if version is not None:
            manifest = model_root / version / _MANIFEST
            return [manifest] if manifest.is_file() else []
        direct = model_root / _MANIFEST
        if direct.is_file():
            return [direct]
        current = model_root / "current" / _MANIFEST
        if current.is_file():
            return [current]
        if not model_root.is_dir():
            return []
        return sorted(
            path
            for path in model_root.glob(f"*/{_MANIFEST}")
            if path.parent.name != "current" and path.is_file()
        )


def default_model_paths() -> tuple[Path, ...]:
    """Return environment and installation model search paths.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Search paths in precedence order.

    Examples
    --------
    Where a model is looked for when a caller names no directory of its own.

    >>> import pulserver.recon as recon
    >>> len(recon.default_model_paths()) >= 1
    True
    """
    configured = os.environ.get(MODEL_PATH_ENV, "")
    environment = [
        Path(item).expanduser() for item in configured.split(os.pathsep) if item
    ]
    installed = Path(sys.prefix) / "share" / "pulserver" / "models"
    return _unique_paths([*environment, installed])


def load_model(
    spec: str | os.PathLike[str],
    *,
    paths: Sequence[str | os.PathLike[str]] | None = None,
    **kwargs: Any,
) -> Any:
    """Resolve and load a deployed model.

    Parameters
    ----------
    spec
        Bundle path, model name, or exact ``name@version`` identifier.
    paths
        Explicit search paths. ``None`` uses scanner defaults.
    **kwargs
        Arguments forwarded to :meth:`ModelBundle.load`.

    Returns
    -------
    torch.nn.Module
        Populated model.

    Examples
    --------
    A model is named, and found on the search path unless a directory is given::

        import pulserver.recon as recon

        model = recon.load_model("unrolled-vn:1.0")
    """
    return ModelStore(paths).load(spec, **kwargs)


def save_bundle(
    model: Any,
    destination: str | os.PathLike[str],
    *,
    name: str,
    version: str,
    architecture: str,
    kwargs: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ModelBundle:
    """Write a weights-only model bundle for scanner deployment.

    Parameters
    ----------
    model
        Torch module providing ``state_dict``.
    destination
        New bundle directory.
    name
        Deployment name used by reconstruction applications.
    version
        Application-defined model version.
    architecture
        Application-defined architecture identifier.
    kwargs
        JSON-compatible architecture arguments.
    metadata
        JSON-compatible application metadata.

    Returns
    -------
    ModelBundle
        Written and validated bundle.

    Raises
    ------
    FileExistsError
        If ``destination`` already exists.

    Examples
    --------
    Write a trained model where :func:`load_model` will find it::

        import pulserver.recon as recon

        recon.save_bundle(
            model,
            "weights/unrolled-vn",
            name="unrolled-vn",
            version="1.0",
            architecture="UnrolledReconstructor",
        )
    """
    selected_name = _nonempty_string(name, name="name")
    selected_version = _nonempty_string(version, name="version")
    _safe_component(selected_name, name="name")
    _safe_component(selected_version, name="version")
    selected_architecture = _nonempty_string(architecture, name="architecture")
    selected_kwargs = {} if kwargs is None else dict(kwargs)
    selected_metadata = {} if metadata is None else dict(metadata)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "name": selected_name,
        "version": selected_version,
        "architecture": selected_architecture,
        "kwargs": selected_kwargs,
        "checkpoint": "weights.pt",
        "sha256": "",
        "metadata": selected_metadata,
    }
    json.dumps(payload)
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("model must provide state_dict")
    destination_path = Path(destination).expanduser()
    destination_path.mkdir(parents=True, exist_ok=False)
    checkpoint_path = destination_path / payload["checkpoint"]
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "Saving model bundles requires Torch, which ships with "
            "pulserver; reinstall the package to restore it."
        ) from error
    state = {
        key: value.detach().to("cpu") if isinstance(value, torch.Tensor) else value
        for key, value in state_dict().items()
    }
    torch.save(state, checkpoint_path)
    payload["sha256"] = _sha256(checkpoint_path)
    manifest_path = destination_path / _MANIFEST
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ModelBundle.from_path(manifest_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_string(value: Any) -> str:
    selected = _nonempty_string(value, name="sha256").lower()
    if len(selected) != 64 or any(
        character not in "0123456789abcdef" for character in selected
    ):
        raise ValueError("sha256 must contain 64 hexadecimal characters")
    return selected


def _contained_path(root: Path, value: str) -> Path:
    selected = (root / value).resolve()
    resolved_root = root.resolve()
    if not selected.is_relative_to(resolved_root):
        raise ValueError("checkpoint must be contained in the bundle directory")
    return selected


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _optional_string(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, name=name)


def _safe_component(value: str, *, name: str) -> None:
    if value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{name} must be one path component")


def _looks_like_path(value: str) -> bool:
    return (
        value.endswith(".json")
        or value.startswith(".")
        or os.path.sep in value
        or (os.path.altsep is not None and os.path.altsep in value)
    )


def _unique_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    result = []
    seen = set()
    for path in paths:
        selected = Path(path).expanduser()
        key = selected.resolve()
        if key not in seen:
            seen.add(key)
            result.append(selected)
    return tuple(result)
