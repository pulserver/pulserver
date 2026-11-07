"""Persistent private wheels, early requirement failures and native import probes."""

import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from pulserver import _site_packages as packages
from pulserver.recon import load_plugin


@pytest.fixture(autouse=True)
def isolated_site(monkeypatch):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(packages, "_configured", None)
    monkeypatch.delenv("PULSERVER_SITE_PACKAGES", raising=False)
    monkeypatch.delenv("PULSERVER_PROVENANCE_DIR", raising=False)


def distribution(root, name="private_backend", version="1.2", extra=""):
    dist = root / f"{name}-{version}.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{extra}"
    )


def test_site_distribution_precedes_a_managed_distribution(tmp_path, monkeypatch):
    managed, private = tmp_path / "managed", tmp_path / "private"
    distribution(managed, version="1.0")
    distribution(private, version="1.2")
    monkeypatch.syspath_prepend(str(managed))
    packages.activate(private)
    plugin = tmp_path / "recon50.py"
    plugin.write_text('REQUIRES = ("private-backend>=1.2",)\n')
    assert packages.requirements(plugin)["private-backend"]["version"] == "1.2"
    assert metadata.version("private-backend") == "1.2"


def test_missing_requirements_fail_before_plugin_side_effects(tmp_path):
    plugin = tmp_path / "recon50.py"
    plugin.write_text(
        'REQUIRES = ("nonexistent-site-test-package>=1.2",)\nraise RuntimeError("executed")\n'
    )
    with pytest.raises(ValueError, match=r"recon50\.py needs .*not installed"):
        load_plugin(plugin)


def test_requirement_versions_markers_and_extras_are_checked(tmp_path):
    distribution(
        tmp_path,
        extra='Provides-Extra: gpu\nRequires-Dist: absent-gpu-backend>=2; extra == "gpu"\n',
    )
    packages.activate(tmp_path)
    plugin = tmp_path / "demo.py"
    plugin.write_text('REQUIRES = ("private-backend>=2",)\n')
    with pytest.raises(ValueError, match=r"installed private-backend==1\.2"):
        packages.requirements(plugin)
    plugin.write_text('REQUIRES = ("private-backend[gpu]>=1",)\n')
    with pytest.raises(ValueError, match="absent-gpu-backend"):
        packages.requirements(plugin)
    plugin.write_text("REQUIRES = (\"absent; python_version < '2'\",)\n")
    assert packages.requirements(plugin) == {}


def test_repeated_distribution_constraints_are_all_checked(tmp_path):
    distribution(tmp_path)
    packages.activate(tmp_path)
    plugin = tmp_path / "demo.py"
    plugin.write_text('REQUIRES = ("private-backend>=1", "private-backend>=2")\n')
    with pytest.raises(ValueError, match="needs private-backend>=2"):
        packages.requirements(plugin)


def test_computed_declarations_are_rejected_without_executing_them(tmp_path):
    plugin = tmp_path / "demo.py"
    plugin.write_text("REQUIRES = tuple(run_a_command())\n")
    with pytest.raises(ValueError, match="literal tuple/list"):
        packages.requirements(plugin)


def test_a_native_install_for_another_python_abi_is_refused(tmp_path):
    record = {"runtime": packages.runtime(), "wheels": {}}
    record["runtime"]["soabi"] = "another-python-abi"
    (tmp_path / packages._STAMP).write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="reinstall compatible wheels"):
        packages.activate(tmp_path)


def test_reconstruction_manifest_records_versions_and_measurement(
    tmp_path, monkeypatch
):
    distribution(tmp_path)
    packages.activate(tmp_path)
    plugin = tmp_path / "recon50.py"
    plugin.write_text('REQUIRES = ("private-backend>=1",)\n')
    record = packages.manifest(plugin)
    context = SimpleNamespace(
        header=SimpleNamespace(
            measurementInformation=SimpleNamespace(measurementID="5186_13")
        )
    )
    destination = tmp_path / "provenance"
    monkeypatch.setenv("PULSERVER_PROVENANCE_DIR", str(destination))
    digest, encoded = packages.reconstruction_record(
        SimpleNamespace(_dependency_manifest=record), context
    )
    saved = json.loads((destination / f"{digest}.json").read_text())
    assert saved == json.loads(encoded)
    assert saved["requirements"]["private-backend"]["version"] == "1.2"
    assert saved["measurement_id"] == "5186_13"


def test_preflight_imports_are_explicit_and_report_native_import_failure(tmp_path):
    plugin = tmp_path / "demo.py"
    plugin.write_text(
        'REQUIRES_IMPORTS = ("absent_native_extension",)\nraise RuntimeError("plugin must not execute")\n'
    )
    with pytest.raises(SystemExit) as error:
        packages.main(["check", "--plugin", str(plugin)])
    assert error.value.code == 1


def test_native_wheel_with_bundled_shared_library_installs_outside_the_environment(
    tmp_path,
):
    compiler = shutil.which("c++")
    if compiler is None or sys.platform != "linux":
        pytest.skip("requires a Linux C++ compiler")
    native = tmp_path / "native"
    native.mkdir()
    (native / "helper.cpp").write_text(
        'extern "C" int private_answer() { return 42; }\n'
    )
    (native / "extension.cpp").write_text("""#include <Python.h>
extern "C" int private_answer();
static PyObject* answer(PyObject*, PyObject*) { return PyLong_FromLong(private_answer()); }
static PyMethodDef methods[] = {{"answer", answer, METH_NOARGS, "value"}, {nullptr, nullptr, 0, nullptr}};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__native() { return PyModule_Create(&module); }
""")
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            str(native / "helper.cpp"),
            "-o",
            str(native / "libprivate.so"),
        ],
        check=True,
    )
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    extension = native / f"_native{suffix}"
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            f"-I{sysconfig.get_path('include')}",
            str(native / "extension.cpp"),
            f"-L{native}",
            "-lprivate",
            "-Wl,-rpath,$ORIGIN/.libs",
            "-o",
            str(extension),
        ],
        check=True,
    )
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    wheel = tmp_path / f"private_native-1.2-{tag}-{tag}-linux_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("private_native/__init__.py", "from ._native import answer\n")
        archive.write(extension, f"private_native/{extension.name}")
        archive.write(native / "libprivate.so", "private_native/.libs/libprivate.so")
        archive.writestr(
            "private_native-1.2.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: private-native\nVersion: 1.2\n",
        )
        archive.writestr(
            "private_native-1.2.dist-info/WHEEL",
            f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {tag}-{tag}-linux_x86_64\n",
        )
        archive.writestr("private_native-1.2.dist-info/RECORD", "")
    root = tmp_path / "persistent"
    packages.install(root, [wheel])
    stamp = json.loads((root / packages._STAMP).read_text())
    assert stamp["wheels"]["private-native"]["sha256"]
    found = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pulserver._site_packages import activate; activate(); import private_native; print(private_native.answer())",
        ],
        env={**os.environ, "PULSERVER_SITE_PACKAGES": str(root)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert found.stdout.strip() == "42"
    assert not (Path(sys.prefix) / "private_native").exists()


def test_a_native_install_is_rejected_after_a_torch_build_change(tmp_path):
    record = {"runtime": packages.runtime(), "wheels": {}, "torch": "0.0+other-cuda"}
    (tmp_path / packages._STAMP).write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="installed with torch"):
        packages.activate(tmp_path)


def test_pip_rejects_a_wheel_for_another_python_abi(tmp_path):
    wheel = tmp_path / "private_wrong-1.0-cp999-cp999-linux_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "private_wrong-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nTag: cp999-cp999-linux_x86_64\n",
        )
    root = tmp_path / "site"
    with pytest.raises(subprocess.CalledProcessError):
        packages.install(root, [wheel])
    assert not (root / packages._STAMP).exists()


def test_runtime_emits_reconstruction_provenance_in_mrd_images(tmp_path, monkeypatch):
    import ismrmrd
    import numpy as np

    from pulserver.recon import ReconResult
    from pulserver.recon._runtime import application

    distribution(tmp_path)
    packages.activate(tmp_path)
    path = tmp_path / "recon50.py"
    path.write_text('REQUIRES = ("private-backend>=1",)\n')

    class Plugin:
        _dependency_manifest = packages.manifest(path)

        def spawn(self):
            return self

        def startup(self, context):
            pass

        def flush(self, context):
            return [(None, ReconResult(np.ones((4, 4))))]

    class Connection:
        def __iter__(self):
            return iter(())

        def send(self, image):
            emitted.append(image)

    image = ismrmrd.Image.from_array(np.ones((4, 4), dtype=np.float32))
    image.attribute_string = ismrmrd.Meta({}).serialize()
    monkeypatch.setattr(application, "_make_images", lambda *args: ([image], 2))
    monkeypatch.setenv("PULSERVER_PROVENANCE_DIR", str(tmp_path / "records"))
    emitted = []
    application.run_application(
        Plugin(), Connection(), SimpleNamespace(header=SimpleNamespace())
    )
    attributes = ismrmrd.Meta.deserialize(emitted[0].attribute_string)
    saved = json.loads(attributes["PulserverReconstructionDependencies"])
    assert saved["requirements"]["private-backend"]["version"] == "1.2"
    assert attributes["PulserverReconstructionManifest"]
