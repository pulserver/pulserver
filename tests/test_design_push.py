"""Designs carried to a reconstruction computer: bundles, the intake and the push."""

import hashlib
import http.client
import io
import json
import os
import socket
import subprocess
import sys
import tarfile

import pytest
from _host import FIXTURE_LIMITS, FIXTURES, LIMITS, PLUGINS, generate

from pulserver.host import DesignStore, call
from pulserver.host._blocks import format_import
from pulserver.host._push import push
from pulserver.proxy import DesignIntake


@pytest.fixture
def host(tmp_path):
    return DesignStore(tmp_path / "host")


@pytest.fixture
def intake(tmp_path):
    running = DesignIntake(tmp_path / "recon")
    running.start()
    yield running
    running.close()


def url(intake):
    return f"http://127.0.0.1:{intake.port}"


def _files(directory):
    return {
        path.name: path.readlink() if path.is_symlink() else path.read_bytes()
        for path in directory.iterdir()
    }


def _tar(entries):
    """A gzip tar of ``(name, bytes)`` files and ``(name, None, target)`` links."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content, *target in entries:
            info = tarfile.TarInfo(name)
            if target:
                info.type = tarfile.SYMTYPE
                info.linkname = target[0]
                archive.addfile(info)
            else:
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def _bundle_of(store, design, **changes):
    """The files of a stored design as a bundle, with ``changes`` replacing files."""
    directory = store.directory(design)
    entries = [(p.name, p.read_bytes()) for p in sorted(directory.iterdir())]
    entries = [(name, changes.get(name, content)) for name, content in entries]
    return _tar(entries)


def test_a_bundle_carries_a_design_to_another_store_unchanged(host, tmp_path):
    design = generate(host, "tiny", {"TE": 8000})
    other = DesignStore(tmp_path / "other")
    assert other.receive(host.pack(design)) == design
    assert _files(other.directory(design)) == _files(host.directory(design))


def test_an_imported_design_keeps_its_link_through_a_bundle(host, tmp_path):
    status, reply = call(
        "import",
        limits=FIXTURE_LIMITS,
        block=format_import(FIXTURES / "dedup_gre_pair.seq"),
        store=host,
    )
    assert status == 0, reply
    design = reply.split()[1]
    other = DesignStore(tmp_path / "other")
    other.receive(host.pack(design))
    link = other.directory(design) / "sequence.seq"
    assert link.readlink().as_posix() == "dedup_gre_pair.seq"


def test_a_bundle_whose_file_differs_from_its_manifest_is_refused(host, tmp_path):
    design = generate(host, "tiny", {"TE": 8000})
    other = DesignStore(tmp_path / "other")
    with pytest.raises(ValueError, match=r"sequence\.seq differs from the manifest"):
        other.receive(_bundle_of(host, design, **{"sequence.seq": b"altered"}))
    assert not any(other.root.iterdir())


@pytest.mark.parametrize(
    ("entries", "reason"),
    [
        ([("../escape", b"x")], "an entry named '../escape'"),
        ([("sub/file", b"x")], "an entry named 'sub/file'"),
        ([("sequence.seq", None, "/etc/passwd")], "links outside its bundle"),
        ([("sequence.seq", b"x")], "holds no manifest"),
    ],
)
def test_a_bundle_holding_anything_but_a_flat_design_is_refused(
    tmp_path, entries, reason
):
    store = DesignStore(tmp_path / "store")
    with pytest.raises(ValueError, match=reason):
        store.receive(_tar(entries))
    assert not any(store.root.iterdir())


def test_a_bundle_with_a_file_its_manifest_does_not_record_is_refused(host, tmp_path):
    design = generate(host, "tiny", {"TE": 8000})
    bundle = io.BytesIO(host.pack(design))
    with tarfile.open(fileobj=bundle, mode="r:gz") as archive:
        entries = [(m.name, archive.extractfile(m).read()) for m in archive]
    with pytest.raises(ValueError, match="not those its manifest records"):
        DesignStore(tmp_path / "other").receive(_tar([*entries, ("extra", b"x")]))


def test_a_bundle_whose_manifest_names_another_identifier_is_refused(host, tmp_path):
    design = generate(host, "tiny", {"TE": 8000})
    manifest = json.loads((host.directory(design) / "manifest.json").read_text())
    manifest["id"] = "0" * 18
    bundle = _bundle_of(
        host, design, **{"manifest.json": json.dumps(manifest).encode()}
    )
    with pytest.raises(ValueError, match="names no design"):
        DesignStore(tmp_path / "other").receive(bundle)


@pytest.mark.parametrize("identity", ["../escaped", "../../escaped", "/escaped"])
def test_a_manifest_naming_a_path_places_nothing_outside_the_store(tmp_path, identity):
    content = b"a sequence"
    manifest = {
        "identity": identity,
        "id": identity[:18],
        "files": {"sequence.seq": hashlib.sha256(content).hexdigest()},
    }
    bundle = _tar(
        [("manifest.json", json.dumps(manifest).encode()), ("sequence.seq", content)]
    )
    store = DesignStore(tmp_path / "a" / "store")
    with pytest.raises(ValueError, match="names no design"):
        store.receive(bundle)
    assert not any(store.root.iterdir())
    assert not (tmp_path / "a" / "escaped").exists()
    assert not (tmp_path / "escaped").exists()


def test_bytes_that_are_not_a_bundle_are_refused(tmp_path):
    with pytest.raises(ValueError, match="cannot be read"):
        DesignStore(tmp_path / "store").receive(b"not a tar")


def test_a_pushed_design_is_stored_by_the_intake_once(host, intake):
    design = generate(host, "tiny", {"TE": 8000})
    assert push(host, design, url(intake)) is True
    assert push(host, design, url(intake)) is False
    assert _files(intake.store.directory(design)) == _files(host.directory(design))


def _request(intake, method, path, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", intake.port, timeout=10)
    try:
        connection.request(method, path, body=body)
        response = connection.getresponse()
        return response.status, response.read().decode()
    finally:
        connection.close()


def _put(intake, design, bundle):
    return _request(intake, "PUT", f"/designs/{design}", bundle)


def test_the_intake_refuses_a_bundle_under_another_design_identifier(host, intake):
    design = generate(host, "tiny", {"TE": 8000})
    other = "f" * 18
    status, text = _put(intake, other, host.pack(design))
    assert status == 400
    assert f"holds design {design}, not {other}" in text
    assert not intake.store.directory(design).exists()


def test_the_intake_refuses_a_tampered_bundle_with_the_reason(host, intake):
    design = generate(host, "tiny", {"TE": 8000})
    status, text = _put(
        intake, design, _bundle_of(host, design, **{"sequence.seq": b"altered"})
    )
    assert status == 400
    assert "differs from the manifest" in text
    assert not intake.store.directory(design).exists()


@pytest.mark.parametrize("path", ["/designs/xyz", "/other"])
def test_the_intake_serves_designs_only(intake, path):
    assert _request(intake, "HEAD", path)[0] == 404


def test_asking_the_intake_for_a_design_it_holds_marks_the_design_used(host, intake):
    design = generate(host, "tiny", {"TE": 8000})
    push(host, design, url(intake))
    manifest = intake.store.directory(design) / "manifest.json"
    os.utime(manifest, (0, 0))
    assert _request(intake, "HEAD", f"/designs/{design}")[0] == 200
    assert intake.store.prune(max_age=86400.0) == []


def test_a_generation_pushes_the_design_it_replies(host, intake):
    status, reply = call(
        "generate",
        plugins=PLUGINS,
        plugin="tiny",
        limits=LIMITS,
        block="[NimPulseqGUI Protocol]\nTE: 8000\n[NimPulseqGUI Protocol End]\n",
        store=host,
        push=url(intake),
    )
    assert status == 0, reply
    design = reply.split()[1]
    assert (intake.store.directory(design) / "manifest.json").is_file()


def _closed_port():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def test_a_design_that_cannot_be_pushed_is_an_error_and_stays_stored(host, tmp_path):
    unreachable = f"http://127.0.0.1:{_closed_port()}"
    block = "[NimPulseqGUI Protocol]\nTE: 8000\n[NimPulseqGUI Protocol End]\n"
    inputs = {
        "plugins": PLUGINS,
        "plugin": "tiny",
        "limits": LIMITS,
        "block": block,
        "store": host,
    }
    status, reply = call("generate", **inputs, push=unreachable)
    assert status == 1
    assert "is stored but not pushed" in reply
    (design,) = list(host)
    intake = DesignIntake(tmp_path / "recon")
    intake.start()
    try:
        status, reply = call("generate", **inputs, push=url(intake))
        assert (status, reply) == (0, f"GENERATED {design}\n")
        assert (intake.store.directory(design) / "manifest.json").is_file()
    finally:
        intake.close()


def test_the_push_call_sends_each_design_its_intake_lacks(host, intake):
    design = generate(host, "tiny", {"TE": 8000})
    command = [
        sys.executable,
        "-m",
        "pulserver._cli",
        "design",
        "push",
        "--store",
        str(host.root),
        "--to",
        url(intake),
        design,
    ]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    again = subprocess.run(command, capture_output=True, text=True, check=False)
    assert (first.returncode, first.stdout) == (0, "PUSHED 1\n")
    assert (again.returncode, again.stdout) == (0, "PUSHED 0\n")


def test_a_design_is_pushed_only_to_an_http_intake(host):
    design = generate(host, "tiny", {"TE": 8000})
    with pytest.raises(ValueError, match="http or https"):
        push(host, design, "file:///tmp/intake")


def test_a_bundle_names_every_file_by_its_digest(host):
    design = generate(host, "tiny", {"TE": 8000})
    with tarfile.open(fileobj=io.BytesIO(host.pack(design)), mode="r:gz") as archive:
        contents = {m.name: archive.extractfile(m).read() for m in archive}
    manifest = json.loads(contents.pop("manifest.json"))
    assert {
        name: hashlib.sha256(content).hexdigest() for name, content in contents.items()
    } == manifest["files"]
