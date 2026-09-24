"""The reconstruction proxy: a series streamed in, images streamed back."""

import json
import os
import socket
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
from _host import DAY, LIMITS, Daemon

from pulserver.recon._runtime.connection import Connection
from pulserver.vre import ReconProxy, SequenceTable

RECON_PLUGINS = Path(__file__).parent / "recon_plugins"
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
    <userParameterLong><name>pulserver_revision</name><value>{revision}</value></userParameterLong>
    <userParameterString><name>pulserver_session</name><value>{session}</value></userParameterString>{exam}
  </userParameters>
</ismrmrdHeader>
"""


@dataclass(frozen=True)
class Series:
    """A generated revision and the readouts a client of it plays."""

    session: str
    revision: int
    table: SequenceTable


@pytest.fixture(scope="module")
def bucket(tmp_path_factory):
    """A bucket holding one revision bound to a reconstruction and one unbound."""
    base = tmp_path_factory.mktemp("vre") / "base"
    daemon = Daemon(base)
    daemon.start()
    try:
        series = {}
        for pid, plugin, name in ((901, "gre2d", "bound"), (902, "gre2d_raw", "raw")):
            client = daemon.client(pid=pid)
            client.open(plugin, LIMITS)
            revision = client.generate(MATRIX)
            session = f"{pid}-{DAY}"
            table = SequenceTable.read(
                base / "bucket" / session / "rev" / str(revision) / "sequence.seq"
            )
            series[name] = Series(session, revision, table)
    finally:
        daemon.cleanup()
    return base, series


@pytest.fixture
def start_proxy(bucket):
    """Start proxies serving the bucket; every one is closed when the test ends."""
    base, _ = bucket
    running = []

    def start(**options):
        proxy = ReconProxy(base, RECON_PLUGINS, **options)
        proxy.bind(0)
        thread = threading.Thread(target=proxy.serve, daemon=True)
        thread.start()
        running.append((proxy, thread))
        return proxy

    yield start
    for proxy, thread in running:
        proxy.close()
        thread.join(timeout=DEADLINE)


def header_xml(series, exam=None):
    return HEADER.format(
        channels=CHANNELS,
        session=series.session,
        revision=series.revision,
        exam=""
        if exam is None
        else f"<userParameterString><name>ExamID</name><value>{exam}</value>"
        "</userParameterString>",
    )


def flat(table, index):
    return np.ones((CHANNELS, int(table.num_samples[index])), dtype=np.complex64)


def point(table, index, position_m):
    """k-space of a point object at ``position_m``, in metres along the gradient axes."""
    start = int(table.sample_offset[index])
    k = table.k[:, start : start + int(table.num_samples[index])].astype(np.float64)
    samples = np.exp(-2j * np.pi * (np.asarray(position_m) @ k))
    return np.broadcast_to(samples, (CHANNELS, samples.size)).astype(np.complex64)


def stream(port, series, *, config="", data=flat, counters=None, exam=None):
    """Play one series' readouts as the scanner client does; return what came back.

    ``counters`` numbers the acquisitions' ``scan_counter``; unnumbered by default.
    ``exam`` is the header's ``ExamID``; none by default.
    """
    stream = socket.create_connection(("127.0.0.1", port), timeout=DEADLINE)
    connection = Connection(stream)
    stream.settimeout(DEADLINE)
    connection.send_config(config)
    connection.send_header(header_xml(series, exam))
    for index in range(len(series.table)):
        acquisition = ismrmrd.Acquisition.from_array(data(series.table, index))
        if counters is not None:
            acquisition.scan_counter = counters[index]
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


def queue_directory(bucket_base, series):
    return bucket_base / "bucket" / series.session / "queue"


def test_a_series_with_every_slot_busy_is_held_on_disk_until_one_frees(
    start_proxy, bucket, tmp_path
):
    base, series = bucket
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
    queue = queue_directory(base, series["bound"])
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
            "--base",
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
    root = proxy._exam_root
    assert root.is_dir()
    proxy.close()
    assert not root.exists()
