"""The reconstruction proxy: a series streamed in, images streamed back."""

import json
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
    <userParameterString><name>pulserver_session</name><value>{session}</value></userParameterString>
    {offset}
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


def header_xml(series, offset_mm=None):
    offset = ""
    if offset_mm is not None:
        value = " ".join(f"{component:g}" for component in offset_mm)
        offset = (
            "<userParameterString><name>pulserver_fov_offset_mm</name>"
            f"<value>{value}</value></userParameterString>"
        )
    return HEADER.format(
        channels=CHANNELS,
        session=series.session,
        revision=series.revision,
        offset=offset,
    )


def flat(table, index):
    return np.ones((CHANNELS, int(table.num_samples[index])), dtype=np.complex64)


def point(table, index, position_m):
    """k-space of a point object at ``position_m``, in metres along the gradient axes."""
    start = int(table.sample_offset[index])
    k = table.k[:, start : start + int(table.num_samples[index])].astype(np.float64)
    samples = np.exp(-2j * np.pi * (np.asarray(position_m) @ k))
    return np.broadcast_to(samples, (CHANNELS, samples.size)).astype(np.complex64)


def stream(port, series, *, config="", offset_mm=None, data=flat):
    """Play one series' readouts as the scanner client does; return what came back."""
    stream = socket.create_connection(("127.0.0.1", port), timeout=DEADLINE)
    connection = Connection(stream)
    stream.settimeout(DEADLINE)
    connection.send_config(config)
    connection.send_header(header_xml(series, offset_mm))
    for index in range(len(series.table)):
        connection.send(ismrmrd.Acquisition.from_array(data(series.table, index)))
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


def test_a_point_at_the_prescription_centre_reconstructs_at_the_image_centre(
    start_proxy, bucket
):
    _, series = bucket
    proxy = start_proxy(slots=1)
    table = series["bound"].table
    fov_mm = table.spaces[0].fov_mm
    pixel_mm = (fov_mm[0] / MATRIX["nx"], fov_mm[1] / MATRIX["ny"])
    offset_mm = (2 * pixel_mm[0], 3 * pixel_mm[1], 0.0)
    position = 1e-3 * np.array(offset_mm)

    def kspace(table, index):
        return point(table, index, position)

    centred = stream(proxy.port, series["bound"], offset_mm=offset_mm, data=kspace)
    uncentred = stream(proxy.port, series["bound"], data=kspace)

    centre = (MATRIX["ny"] // 2, MATRIX["nx"] // 2)
    assert peak(images(centred)[0]) == centre
    assert peak(images(uncentred)[0]) == (centre[0] + 3, centre[1] + 2)


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
