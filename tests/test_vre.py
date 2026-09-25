"""The reconstruction proxy: a series streamed in, images streamed back."""

import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import ismrmrd
import ismrmrd.xsd
import numpy as np
import pytest
from _host import generate

from pulserver.host import DesignStore
from pulserver.host._push import push
from pulserver.recon._runtime import concurrency
from pulserver.recon._runtime.connection import Connection
from pulserver.recon._runtime.mrd2dicom import DicomWithName
from pulserver.vre import (
    DesignCache,
    DesignIntake,
    ReconProxy,
    ReconServer,
    SequenceTable,
    _designs,
)

RECON_PLUGINS = Path(__file__).parent / "recon_plugins"
FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
MATRIX = {"nx": 32, "ny": 16, "TE": 5000}
CHANNELS = 2
# Long enough that a stall fails the run instead of hanging it.
DEADLINE = 180.0

HEADER = """<?xml version="1.0"?>
<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
  <acquisitionSystemInformation><receiverChannels>{channels}</receiverChannels></acquisitionSystemInformation>
  <experimentalConditions><H1resonanceFrequency_Hz>63500000</H1resonanceFrequency_Hz></experimentalConditions>
  <encoding>
    <encodedSpace><matrixSize><x>1</x><y>1</y><z>1</z></matrixSize><fieldOfView_mm><x>1</x><y>1</y><z>1</z></fieldOfView_mm></encodedSpace>
    <reconSpace><matrixSize><x>1</x><y>1</y><z>1</z></matrixSize><fieldOfView_mm><x>1</x><y>1</y><z>1</z></fieldOfView_mm></reconSpace>
    <encodingLimits/>
    <trajectory>cartesian</trajectory>
  </encoding>
  <userParameters>
    <userParameterString><name>pulserver_design</name><value>{design}</value></userParameterString>{exam}
  </userParameters>
</ismrmrdHeader>
"""


@dataclass(frozen=True)
class Series:
    """A stored design and the readouts a client of it plays."""

    design: str
    table: SequenceTable


@pytest.fixture(scope="module")
def bucket(tmp_path_factory):
    """A store holding one design bound to a reconstruction and one unbound."""
    store = DesignStore(tmp_path_factory.mktemp("vre") / "designs")
    series = {}
    for plugin, name in (("gre2d", "bound"), ("gre2d_raw", "raw")):
        design = generate(store, plugin, MATRIX)
        table = SequenceTable.read(store.directory(design) / "sequence.seq")
        series[name] = Series(design, table)
    return store.root, series


@pytest.fixture
def start_proxy(bucket):
    """Start proxies serving the store; every one is closed when the test ends."""
    root, _ = bucket
    running = []

    def start(**options):
        proxy = ReconProxy(root, RECON_PLUGINS, **options)
        proxy.bind(0)
        thread = threading.Thread(target=proxy.serve, daemon=True)
        thread.start()
        running.append((proxy, thread))
        return proxy

    yield start
    for proxy, thread in running:
        proxy.close()
        thread.join(timeout=DEADLINE)


@pytest.fixture
def start_server():
    """Start reconstruction servers over the test plugins; each is closed when the test ends."""
    running = []

    def start(**options):
        server = ReconServer(RECON_PLUGINS, **options)
        server.bind(0)
        thread = threading.Thread(target=server.serve, daemon=True)
        thread.start()
        running.append((server, thread))
        return server

    yield start
    for server, thread in running:
        server.close()
        thread.join(timeout=DEADLINE)


class _RecordingServer:
    """An MRD server that keeps what one forwarded series carries, then runs ``answer``."""

    def __init__(self, answer):
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.port = self._listener.getsockname()[1]
        self.received = []
        self._answer = answer
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        stream, _ = self._listener.accept()
        self._listener.close()
        stream.settimeout(DEADLINE)
        connection = Connection(stream)
        for item in connection:
            if isinstance(item, ismrmrd.Acquisition) and not item.number_of_samples:
                break
            self.received.append(item)
        self._answer(connection, stream)
        connection.shutdown_close()

    def join(self):
        self._thread.join(timeout=DEADLINE)


def _texts(connection, _stream):
    connection.send("forwarded")


def header_xml(series, exam=None, design=None):
    return HEADER.format(
        channels=CHANNELS,
        design=series.design if design is None else design,
        exam=""
        if exam is None
        else f"<userParameterString><name>ExamID</name><value>{exam}</value>"
        "</userParameterString>",
    )


def flat(table, index):
    return np.ones((CHANNELS, int(table.num_samples[index])), dtype=np.complex64)


def point(table, index, position_m):
    """k-space of a point object at ``position_m``, in metres along the gradient axes."""
    samples = np.exp(-2j * np.pi * (np.asarray(position_m) @ table.readout_k(index)))
    return np.broadcast_to(samples, (CHANNELS, samples.size)).astype(np.complex64)


def stream(
    port,
    series,
    *,
    config="",
    data=flat,
    counters=None,
    exam=None,
    headers=1,
    readouts=None,
    last=None,
    design=None,
):
    """Play one series' readouts as the scanner client does; return what came back.

    ``config`` ``None`` sends no config message. ``counters`` numbers the
    acquisitions' ``scan_counter``; unnumbered by default. ``exam`` is the
    header's ``ExamID``; none by default. ``headers`` is how many times the
    header is sent, ``readouts`` how many readouts are, all of them by default,
    ``last`` the acquisition flagged ``LAST_IN_MEASUREMENT``, none by default,
    and ``design`` the header's ``pulserver_design``, the series' by default.
    """
    stream = socket.create_connection(("127.0.0.1", port), timeout=DEADLINE)
    connection = Connection(stream)
    stream.settimeout(DEADLINE)
    if config is not None:
        connection.send_config(config)
    for _ in range(headers):
        connection.send_header(header_xml(series, exam, design))
    for index in range(len(series.table) if readouts is None else readouts):
        acquisition = ismrmrd.Acquisition.from_array(data(series.table, index))
        if counters is not None:
            acquisition.scan_counter = counters[index]
        if index == last:
            acquisition.setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
        connection.send(acquisition)
    connection.send_close()
    received = list(connection)
    connection.shutdown_close()
    return received


def peak(image):
    picture = np.squeeze(np.asarray(image.data))
    return np.unravel_index(int(np.argmax(picture)), picture.shape)


def images(received):
    return [item for item in received if isinstance(item, ismrmrd.Image)]


def closed(received):
    return any(
        isinstance(item, ismrmrd.Acquisition)
        and item.isFlagSet(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
        for item in received
    )


def test_a_series_returns_images_then_closes(start_proxy, bucket):
    _, series = bucket
    proxy = start_proxy(slots=1)
    received = stream(proxy.port, series["bound"])
    assert closed(received)
    assert [np.squeeze(image.data).shape for image in images(received)] == [
        (MATRIX["ny"], MATRIX["nx"])
    ]


def test_a_second_series_waits_for_a_slot_and_still_returns_images(
    start_proxy, bucket, tmp_path
):
    _, series = bucket
    proxy = start_proxy(slots=1)
    trace = tmp_path / "trace"
    trace.mkdir()
    config = json.dumps({"parameters": {"config": "gre2d", "trace": str(trace)}})
    received = {}

    def play(name):
        received[name] = stream(proxy.port, series["bound"], config=config)

    threads = [threading.Thread(target=play, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=DEADLINE)

    assert set(received) == {"a", "b"}
    assert all(len(images(items)) == 1 for items in received.values())
    first, second = sorted(
        (json.loads(path.read_text()) for path in trace.glob("*.json")),
        key=lambda run: run["start"],
    )
    assert first["end"] <= second["start"]


def test_a_reconstruction_outlasting_the_worker_timeout_still_returns_its_image(
    start_proxy, bucket, monkeypatch
):
    _, series = bucket
    monkeypatch.setattr("pulserver.vre._proxy._WORKER_TIMEOUT", 1.0)
    proxy = start_proxy(slots=1)
    config = json.dumps({"parameters": {"config": "gre2d", "delay": 3.0}})
    received = stream(proxy.port, series["bound"], config=config)
    assert closed(received)
    assert len(images(received)) == 1


def test_a_reconstruction_past_the_recon_timeout_is_stopped_and_reported(
    start_proxy, bucket
):
    _, series = bucket
    proxy = start_proxy(slots=1, recon_timeout=1.0)
    config = json.dumps({"parameters": {"config": "gre2d", "delay": 60.0}})
    started = time.monotonic()
    received = stream(proxy.port, series["bound"], config=config)
    assert time.monotonic() - started < 30
    assert not images(received)
    assert any(isinstance(item, str) and "did not finish" in item for item in received)
    assert len(images(stream(proxy.port, series["bound"]))) == 1


def test_a_stream_numbered_without_gaps_is_reconstructed(start_proxy, bucket):
    _, series = bucket
    proxy = start_proxy(slots=1)
    counters = [index + 1 for index in range(len(series["bound"].table))]
    assert len(images(stream(proxy.port, series["bound"], counters=counters))) == 1


def test_a_gap_in_the_scan_counters_stops_the_series_unreconstructed(
    start_proxy, bucket
):
    _, series = bucket
    proxy = start_proxy(slots=1)
    readouts = len(series["bound"].table)
    counters = [index + 1 + (index >= readouts // 2) for index in range(readouts)]
    received = stream(proxy.port, series["bound"], counters=counters)
    assert not images(received)
    assert any(isinstance(item, str) and "scan counter" in item for item in received)
    assert len(images(stream(proxy.port, series["bound"]))) == 1


def _refused(received, reason):
    return not images(received) and any(
        isinstance(item, str) and reason in item for item in received
    )


def test_a_series_opening_with_its_header_is_reconstructed_as_its_design_names(
    start_proxy, bucket
):
    _, series = bucket
    proxy = start_proxy(slots=1)
    assert len(images(stream(proxy.port, series["bound"], config=None))) == 1


def test_a_series_that_ends_before_its_last_readout_is_refused(start_proxy, bucket):
    _, series = bucket
    proxy = start_proxy(slots=1)
    readouts = len(series["bound"].table)
    received = stream(proxy.port, series["bound"], readouts=readouts - 1)
    assert _refused(received, f"ended after {readouts - 1} of the {readouts} readouts")
    assert len(images(stream(proxy.port, series["bound"]))) == 1


@pytest.mark.parametrize("position", ["middle", "end"])
def test_only_the_last_readout_may_be_flagged_last_in_measurement(
    start_proxy, bucket, position
):
    _, series = bucket
    proxy = start_proxy(slots=1)
    readouts = len(series["bound"].table)
    last = readouts // 2 if position == "middle" else readouts - 1
    received = stream(proxy.port, series["bound"], last=last)
    if position == "middle":
        assert _refused(received, f"acquisition {last} is flagged LAST_IN_MEASUREMENT")
    else:
        assert len(images(received)) == 1


@pytest.mark.parametrize(
    ("headers", "reason"),
    [
        (0, "a message of type Acquisition where its MRD header belongs"),
        (2, "a second MRD header"),
    ],
)
def test_a_series_carrying_no_header_or_two_is_refused(
    start_proxy, bucket, headers, reason
):
    _, series = bucket
    proxy = start_proxy(slots=1)
    assert _refused(stream(proxy.port, series["bound"], headers=headers), reason)


def test_a_stream_closed_before_its_header_is_refused(start_proxy, bucket):
    _, series = bucket
    proxy = start_proxy(slots=1)
    received = stream(proxy.port, series["bound"], config=None, headers=0, readouts=0)
    assert _refused(received, "where its MRD header belongs")


def test_a_crashing_plugin_closes_its_series_and_frees_the_slot(start_proxy, bucket):
    _, series = bucket
    proxy = start_proxy(slots=1)
    crashed = stream(proxy.port, series["raw"], config="crash")
    assert closed(crashed)
    assert not images(crashed)
    assert any(isinstance(item, str) and "always fails" in item for item in crashed)
    assert len(images(stream(proxy.port, series["bound"]))) == 1


def test_the_spare_worker_is_replaced_after_each_series(start_proxy, bucket):
    _, series = bucket
    proxy = start_proxy(slots=1, spares=1)
    warm = proxy.workers.spare_pids()
    assert len(warm) == 1
    assert len(images(stream(proxy.port, series["bound"]))) == 1
    replaced = proxy.workers.spare_pids()
    assert len(replaced) == 1
    assert replaced != warm
    assert len(images(stream(proxy.port, series["bound"]))) == 1
    assert proxy.workers.spare_pids() not in (warm, replaced)


def test_a_series_with_every_slot_busy_is_held_on_disk_until_one_frees(
    start_proxy, bucket, tmp_path
):
    _, series = bucket
    proxy = start_proxy(slots=1)
    gate = tmp_path / "go"
    trace = tmp_path / "trace"
    trace.mkdir()
    held = json.dumps(
        {"parameters": {"config": "gre2d", "gate": str(gate), "trace": str(trace)}}
    )
    received = {}

    def play(name, config):
        received[name] = stream(proxy.port, series["bound"], config=config)

    first = threading.Thread(target=play, args=("held", held))
    first.start()
    queue = proxy.queue
    _wait_until(gate.with_name("go.waiting").exists, "the held series never started")

    second = threading.Thread(
        target=play,
        args=(
            "queued",
            json.dumps({"parameters": {"config": "gre2d", "trace": str(trace)}}),
        ),
    )
    second.start()
    _wait_until(lambda: len(list(queue.glob("*.h5"))) == 1, "no series was queued")

    gate.touch()
    for thread in (first, second):
        thread.join(timeout=DEADLINE)

    assert set(received) == {"held", "queued"}
    assert all(len(images(items)) == 1 for items in received.values())
    assert not list(queue.glob("*.h5"))


def _wait_until(condition, message, timeout=DEADLINE):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError(message)


def test_a_point_off_the_centre_reconstructs_where_it_was_acquired(start_proxy, bucket):
    """The readouts arrive demodulated by the playout, and the proxy leaves them so."""
    _, series = bucket
    proxy = start_proxy(slots=1)
    table = series["bound"].table
    fov_mm = table.spaces[0].fov_mm
    pixel_mm = (fov_mm[0] / MATRIX["nx"], fov_mm[1] / MATRIX["ny"])
    position = 1e-3 * np.array((2 * pixel_mm[0], 3 * pixel_mm[1], 0.0))

    def kspace(table, index):
        return point(table, index, position)

    received = stream(proxy.port, series["bound"], data=kspace)

    centre = (MATRIX["ny"] // 2, MATRIX["nx"] // 2)
    assert peak(images(received)[0]) == (centre[0] + 3, centre[1] + 2)


def test_a_stopped_proxy_stops_accepting_within_one_poll(tmp_path):
    proxy = ReconProxy(tmp_path, RECON_PLUGINS)
    proxy.bind(0)
    thread = threading.Thread(target=proxy.serve, daemon=True)
    thread.start()
    try:
        proxy.stop()
        thread.join(timeout=5)
        assert not thread.is_alive()
    finally:
        proxy.close()


def test_a_terminated_proxy_process_exits_cleanly(tmp_path):
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pulserver.vre",
            "--store",
            str(tmp_path),
            "--port",
            "0",
            "--plugins",
            str(RECON_PLUGINS),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for line in process.stderr:
            if "serving" in line:
                break
        process.terminate()
        assert process.wait(timeout=30) == 0
    finally:
        process.kill()
        process.stderr.close()


def test_the_proxy_process_does_not_import_the_reconstruction_engine(tmp_path):
    """Only a worker imports bartorch; the proxy that spawns it does not."""
    (tmp_path / "bartorch.py").write_text("raise SystemExit('imported bartorch')\n")
    probe = "import sys, pulserver.vre; print('bartorch' in sys.modules)"
    environment = {**os.environ, "PYTHONPATH": str(tmp_path)}
    found = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=environment,
        check=True,
    )
    assert found.stdout.strip() == "False"


def test_the_proxy_listens_on_the_loopback_interface_unless_told_otherwise(tmp_path):
    proxy = ReconProxy(tmp_path, tmp_path, slots=1, spares=1)
    try:
        port = proxy.bind(0)
        assert proxy._server.getsockname() == ("127.0.0.1", port)
    finally:
        proxy.close()


def test_the_series_of_one_exam_share_its_exam_cache(start_proxy, bucket):
    _, series = bucket
    port = start_proxy(slots=1).port
    counts = [
        [
            int(np.abs(image.data).max())
            for image in images(stream(port, series["raw"], config="exam", exam=exam))
        ]
        for exam in ("E1", "E1", "E2", "E1")
    ]
    assert counts == [[1], [2], [1], [1]]


def test_a_closed_proxy_leaves_no_exam_directory(tmp_path):
    proxy = ReconProxy(tmp_path, tmp_path, slots=1, spares=1)
    root = proxy._reconstruction._exam_root
    assert root.is_dir()
    proxy.close()
    assert not root.exists()


def test_a_reconstruction_may_start_processes_of_its_own(start_proxy, bucket):
    _, series = bucket
    received = stream(start_proxy(slots=1).port, series["raw"], config="children")
    assert [np.abs(image.data).max() for image in images(received)] == [1.0]


@pytest.mark.parametrize(("listed", "read"), [(0, [0, 0]), (2, [1, 2])])
def test_a_series_reads_the_gpu_its_slot_holds(
    start_proxy, bucket, monkeypatch, listed, read
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(concurrency, "_listed_gpus", lambda: listed)
    _, series = bucket
    port = start_proxy(slots=2).port
    values = [
        int(np.abs(image.data).max())
        for _ in range(2)
        for image in images(stream(port, series["raw"], config="device"))
    ]
    assert values == read


def test_a_design_pushed_to_the_intake_is_reconstructed_from_its_store(tmp_path):
    host = DesignStore(tmp_path / "host")
    design = generate(host, "gre2d", MATRIX)
    intake = DesignIntake(tmp_path / "recon")
    intake.start()
    proxy = ReconProxy(tmp_path / "recon", RECON_PLUGINS, slots=1)
    proxy.bind(0)
    thread = threading.Thread(target=proxy.serve, daemon=True)
    thread.start()
    try:
        assert push(host, design, f"http://127.0.0.1:{intake.port}")
        table = SequenceTable.read(host.directory(design) / "sequence.seq")
        received = stream(proxy.port, Series(design, table))
        assert len(images(received)) == 1
    finally:
        proxy.close()
        thread.join(timeout=DEADLINE)
        intake.close()


def test_the_least_recently_read_design_is_read_again_when_next_named(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(_designs, "_KEPT", 2)
    cache = DesignCache(tmp_path)
    directories = []
    for number in range(3):
        directory = tmp_path / f"{number:018x}"
        directory.mkdir(parents=True)
        shutil.copy(FIXTURES / "gre_2d_3sl.seq", directory / "sequence.seq")
        (directory / "manifest.json").write_text("{}")
        directories.append(directory)
    first, second = cache.read(directories[0]), cache.read(directories[1])
    assert cache.read(directories[0]) is first
    cache.read(directories[2])
    assert cache.read(directories[0]) is first
    assert cache.read(directories[1]) is not second


@pytest.mark.parametrize(
    ("design", "reason"),
    [
        ("", "carries no pulserver_design"),
        ("rev-1", "is not a design identifier"),
        ("0" * 18, "no design 000000000000000000"),
    ],
)
def test_a_series_naming_no_stored_design_is_refused(
    start_proxy, bucket, design, reason
):
    _, series = bucket
    proxy = start_proxy(slots=1)
    received = stream(proxy.port, series["bound"], design=design)
    assert _refused(received, reason)


def test_a_forwarded_series_returns_the_image_a_local_worker_returns(
    start_proxy, start_server, bucket
):
    _, series = bucket
    server = start_server(slots=1)
    forwarding = start_proxy(forward=("127.0.0.1", server.port))
    local = start_proxy(slots=1)

    def kspace(table, index):
        return point(table, index, (0.004, -0.002, 0.0))

    forwarded = stream(forwarding.port, series["bound"], data=kspace)
    reconstructed = stream(local.port, series["bound"], data=kspace)

    assert closed(forwarded)
    assert forwarding.workers is None
    assert len(images(forwarded)) == len(images(reconstructed)) == 1
    np.testing.assert_array_equal(
        images(forwarded)[0].data, images(reconstructed)[0].data
    )


@pytest.mark.parametrize("configured", [None, "default.xml"])
def test_a_forwarded_series_names_its_reconstruction_in_a_config_file_message(
    start_proxy, bucket, configured
):
    _, series = bucket
    recording = _RecordingServer(_texts)
    proxy = start_proxy(
        forward=("127.0.0.1", recording.port), forward_config=configured
    )

    received = stream(proxy.port, series["bound"], config='{"parameters": {}}')
    recording.join()

    name, header, *acquisitions = recording.received
    assert name == (configured or "gre2d")
    assert header.encoding[0].encodedSpace.matrixSize.x == MATRIX["nx"]
    assert len(acquisitions) == len(series["bound"].table)
    assert "forwarded" in received
    assert closed(received)


def test_a_message_the_proxy_cannot_read_ends_the_output_with_its_type(
    start_proxy, bucket
):
    _, series = bucket

    def unreadable(connection, stream):
        connection.send("first")
        stream.sendall(struct.pack("<H", 1030) + bytes(64))

    recording = _RecordingServer(unreadable)
    proxy = start_proxy(forward=("127.0.0.1", recording.port))
    received = stream(proxy.port, series["bound"])
    recording.join()

    assert "first" in received
    assert any(
        isinstance(item, str) and "message of type 1030" in item for item in received
    )
    assert closed(received)


def test_forwarded_images_come_back_as_dicom_when_asked(
    start_proxy, start_server, bucket
):
    _, series = bucket
    server = start_server(slots=1)
    proxy = start_proxy(forward=("127.0.0.1", server.port), forward_dicom=True)
    received = stream(proxy.port, series["bound"])

    assert not images(received)
    assert sum(isinstance(item, DicomWithName) for item in received) == 1
    assert closed(received)


def test_a_forwarded_series_past_the_recon_timeout_is_stopped_and_reported(
    start_proxy, bucket
):
    _, series = bucket
    released = threading.Event()
    recording = _RecordingServer(lambda *_: released.wait(DEADLINE))
    proxy = start_proxy(forward=("127.0.0.1", recording.port), recon_timeout=1.0)
    received = stream(proxy.port, series["bound"])
    released.set()
    recording.join()
    assert _refused(
        received,
        f"the reconstruction server at 127.0.0.1:{recording.port} did not finish "
        "within 1 s",
    )
    assert closed(received)


@pytest.mark.parametrize("arguments", [[], ["--forward", "no-port"]])
def test_the_proxy_command_needs_plugins_or_a_server_address(tmp_path, arguments):
    from pulserver.vre.__main__ import main

    with pytest.raises(SystemExit) as stopped:
        main(["--store", str(tmp_path), "--port", "0", *arguments])
    assert stopped.value.code == 2


def test_a_series_forwarded_to_no_server_is_refused(start_proxy, bucket):
    _, series = bucket
    with socket.create_server(("127.0.0.1", 0)) as unused:
        port = unused.getsockname()[1]
    proxy = start_proxy(forward=("127.0.0.1", port))
    received = stream(proxy.port, series["bound"])
    assert _refused(received, f"the reconstruction server at 127.0.0.1:{port}")


def test_a_series_whose_config_names_no_plugin_is_refused_by_the_server(
    start_server, bucket
):
    _, series = bucket
    server = start_server(slots=1)
    received = stream(server.port, series["bound"], config="")
    assert _refused(received, "the config names no reconstruction")


def test_a_proxy_that_neither_forwards_nor_has_plugins_is_refused(tmp_path):
    with pytest.raises(ValueError, match="needs a plugin directory"):
        ReconProxy(tmp_path)


def test_a_terminated_server_process_exits_cleanly():
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pulserver.recon",
            "--plugins",
            str(RECON_PLUGINS),
            "--port",
            "0",
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for line in process.stderr:
            if "serving" in line:
                break
        process.terminate()
        assert process.wait(timeout=30) == 0
    finally:
        process.kill()
        process.stderr.close()
