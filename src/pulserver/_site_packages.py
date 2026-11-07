"""Persistent site packages, plugin requirements and reconstruction provenance."""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import site
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename

_configured: Path | None = None
_active = None
_STAMP = ".pulserver-install.json"


def runtime():
    return {
        "python": platform.python_version(),
        "soabi": sysconfig.get_config_var("SOABI"),
        "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
    }


def directory():
    named = os.environ.get("PULSERVER_SITE_PACKAGES")
    return Path(named).expanduser().resolve() if named else _configured


def activate(root=None):
    """Prepend the site library before importing plugins or warming workers."""
    global _configured, _active
    if root is not None:
        _configured = Path(root).resolve()
    path = directory()
    if path is None or not path.is_dir():
        return
    stamp = path / _STAMP
    if stamp.is_file():
        recorded = json.loads(stamp.read_text())["runtime"]
        current = runtime()
        for key in ("soabi", "machine"):
            if recorded[key] != current[key]:
                raise RuntimeError(
                    f"site packages in {path} require {key}={recorded[key]}, "
                    f"this interpreter has {current[key]}; reinstall compatible wheels"
                )
        native_torch = json.loads(stamp.read_text()).get("torch")
        if native_torch is not None and metadata.version("torch") != native_torch:
            raise RuntimeError(
                f"native site packages were installed with torch {native_torch}; installed torch {metadata.version('torch')}; reinstall matching wheels"
            )
    signature = (
        str(path),
        path.stat().st_mtime_ns,
        stamp.stat().st_mtime_ns if stamp.is_file() else 0,
    )
    if _active == signature and str(path) in sys.path:
        return
    before = sys.path.copy()
    site.addsitedir(str(path))
    added = [p for p in sys.path if p not in before]
    promoted = list(dict.fromkeys([str(path), *added]))
    sys.path[:] = promoted + [p for p in sys.path if p not in promoted]
    importlib.invalidate_caches()
    _active = signature


def declarations(path):
    found = {"REQUIRES": (), "REQUIRES_IMPORTS": ()}
    for node in ast.parse(Path(path).read_text(), filename=str(path)).body:
        if isinstance(node, ast.Assign):
            names = [n.id for n in node.targets if isinstance(n, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        for name in names:
            if name not in found:
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"{path}: {name} must be a literal tuple/list of strings"
                ) from error
            if not isinstance(value, (list, tuple)) or not all(
                isinstance(v, str) for v in value
            ):
                raise ValueError(
                    f"{path}: {name} must be a literal tuple/list of strings"
                )
            found[name] = tuple(value)
    return found


def requirements(path):
    """Check literal requirements before executing a plugin; return resolved versions."""
    activate()
    declared = declarations(path)
    versions = {}
    seen = set()

    def resolve(req):
        key = (canonicalize_name(req.name), tuple(sorted(req.extras)))
        if req.url:
            raise ValueError(
                f"{path}: REQUIRES uses installed name/version constraints, not URLs"
            )
        try:
            dist = metadata.distribution(req.name)
        except metadata.PackageNotFoundError:
            raise ValueError(
                f"{Path(path).name} needs {req}; not installed (site library: {directory()})"
            ) from None
        if not req.specifier.contains(dist.version, prereleases=None):
            raise ValueError(
                f"{Path(path).name} needs {req}; installed {req.name}=={dist.version} at {dist.locate_file('')}"
            )
        versions[key[0]] = {
            "version": dist.version,
            "location": str(Path(dist.locate_file("")).resolve()),
        }
        if key in seen:
            return
        seen.add(key)
        for text in dist.requires or ():
            child = Requirement(text)
            if child.marker is None or any(
                child.marker.evaluate({"extra": extra}) for extra in ("", *req.extras)
            ):
                resolve(child)

    for text in declared["REQUIRES"]:
        req = Requirement(text)
        if req.marker is None or req.marker.evaluate():
            resolve(req)

    return versions


def manifest(path):
    path = Path(path)
    versions = requirements(path)
    root = directory()
    site_versions = (
        {
            canonicalize_name(dist.metadata["Name"]): dist.version
            for dist in metadata.distributions(path=[str(root)])
            if dist.metadata["Name"]
        }
        if root is not None and root.is_dir()
        else {}
    )
    if (
        not versions
        and not site_versions
        and not declarations(path)["REQUIRES_IMPORTS"]
    ):
        return {}
    record = {
        "plugin": path.name,
        "plugin_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "runtime": runtime(),
        "requirements": versions,
        "site_packages": site_versions,
    }
    for name in ("pulserver", "pypulseqpp", "torch", "numpy"):
        with contextlib.suppress(metadata.PackageNotFoundError):
            record.setdefault("environment", {})[name] = metadata.version(name)
    path = directory()
    if path is not None and (path / _STAMP).is_file():
        record["site_install"] = json.loads((path / _STAMP).read_text())
    return record


def reconstruction_record(plugin, context):
    record = getattr(plugin, "_dependency_manifest", {})
    if not record:
        return None
    record = dict(record)
    info = getattr(context.header, "measurementInformation", None)
    record["measurement_id"] = getattr(info, "measurementID", None)
    encoded = json.dumps(record, sort_keys=True)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    named = os.environ.get("PULSERVER_PROVENANCE_DIR")
    root = directory()
    destination = Path(named) if named else root.parent / "manifests" if root else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"{digest}.json"
        with tempfile.NamedTemporaryFile(
            mode="w", dir=destination, delete=False
        ) as stream:
            stream.write(encoded + "\n")
        Path(stream.name).replace(path)
    return digest, encoded


def install(root, wheels):
    wheels = [Path(p).resolve() for p in wheels]
    if any(not p.is_file() or p.suffix != ".whl" for p in wheels):
        raise ValueError(
            "site install takes local .whl files; build/download wheels first"
        )
    root.mkdir(parents=True, exist_ok=True)
    stamp = root / _STAMP
    previous = json.loads(stamp.read_text()) if stamp.is_file() else {}
    # Never mix native packages from different Python ABIs in one library.
    if previous and any(
        previous["runtime"][k] != runtime()[k] for k in ("soabi", "machine")
    ):
        raise ValueError("use an empty site library for this Python ABI/architecture")
    subprocess.run(  # noqa: S603 -- wheel paths are explicit local installation inputs
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--upgrade",
            "--target",
            str(root),
            *map(str, wheels),
        ],
        check=True,
        stdout=sys.stderr,
    )
    entries = previous.get("wheels", {})
    for wheel in wheels:
        name = canonicalize_name(parse_wheel_filename(wheel.name)[0])
        entries[name] = {
            "filename": wheel.name,
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        }
    record = {"runtime": runtime(), "wheels": entries}
    native = any(
        tag.abi != "none" or tag.platform != "any"
        for p in wheels
        for tag in parse_wheel_filename(p.name)[3]
    )
    if native or previous.get("torch"):
        with contextlib.suppress(metadata.PackageNotFoundError):
            record["torch"] = metadata.version("torch")
    stamp.write_text(json.dumps(record, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pulserver site")
    parser.add_argument(
        "--root",
        type=Path,
        help="persistent site library; otherwise configured by the installer or PULSERVER_SITE_PACKAGES",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("path")
    installing = sub.add_parser("install")
    installing.add_argument("wheels", nargs="+")
    check = sub.add_parser("check")
    check.add_argument("--plugin", type=Path, required=True)
    check.add_argument("--import", dest="imports", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        if args.root is not None:
            os.environ["PULSERVER_SITE_PACKAGES"] = str(args.root.resolve())
        activate(args.root)
        root = directory()
        if args.command == "path":
            if root is None:
                raise ValueError("configure PULSERVER_SITE_PACKAGES or use --root")
            print(root)
        elif args.command == "install":
            if root is None:
                raise ValueError("configure PULSERVER_SITE_PACKAGES or use --root")
            install(root, args.wheels)
        else:
            record = manifest(args.plugin)
            imported = {}
            for name in (*declarations(args.plugin)["REQUIRES_IMPORTS"], *args.imports):
                module = importlib.import_module(name)
                imported[name] = str(getattr(module, "__file__", "namespace"))
            print(json.dumps({**record, "imports": imported}, indent=2))
    except (
        ValueError,
        RuntimeError,
        ImportError,
        OSError,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(1, f"{error}\n")
    return 0
