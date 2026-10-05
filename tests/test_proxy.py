"""The reconstruction proxy: a series streamed in, images streamed back."""

import json
import os
import runpy
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
from pulserver.proxy import (
    DesignCache,
    DesignIntake,
    LocalReconstruction,
    ReconProxy,
    ReconServer,
    SequenceTable,
    _designs,
    _held,
)
from pulserver.proxy._proxy import _config_plugin
from pulserver.proxy._seqdesc import is_message
from pulserver.recon import load_plugin
from pulserver.recon._runtime import concurrency
from pulserver.recon._runtime.connection import Connection
from pulserver.recon._runtime.mrd2dicom import DicomWithName

RECON_PLUGINS = Path(__file__).parent / "recon_plugins"
FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
MATRIX = {"nx": 32, "ny": 16, "TE": 5000}
CHANNELS = 2
# Long enough that a stall fails the run instead of hanging it.
DEADLINE = 180.0

HEADER = """<?xml version="1.0"?>
<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">{measurement}
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
    """A store holding the designs of the plugins ``gre2d`` and ``gre2d_raw``."""
    store = DesignStore(tmp_path_factory.mktemp("proxy") / "designs")
    series = {}
    for plugin in ("gre2d", "gre2d_raw"):
        design = generate(store, plugin, MATRIX)
        table = SequenceTable.read(store.directory(design) / "sequence.seq")
        series[plugin] = Series(design, table)
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


def header_xml(series, exam=None, design=None, measurement=None):
    return HEADER.format(
        measurement=""
        if measurement is None
        else f"<measurementInformation><measurementID>{measurement}</measurementID>"
        "<patientPosition>HFS</patientPosition></measurementInformation>",
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
    config="gre2d",
    data=flat,
    counters=None,
    exam=None,
    headers=1,
    readouts=None,
    last=None,
    design=None,
    measurement=None,
    leave=False,
):
    """Play one series' readouts as the scanner client does; return what came back.

    ``config`` is the text of the config message, which names the
    reconstruction plugin: the traced FFT ``gre2d`` by default; ``None`` sends
    no config message. ``counters`` numbers the acquisitions' ``scan_counter``;
    unnumbered by default. ``exam`` is the header's ``ExamID``; none by
    default. ``headers`` is how many times the header is sent, ``readouts`` how
    many readouts are, all of them by default, ``last`` the acquisition flagged
    ``LAST_IN_MEASUREMENT``, none by default, ``design`` the header's
    ``pulserver_design``, the series' by default, and ``measurement`` its
    ``measurementID``, none by default. With ``leave`` the client closes its
    connection once it has sent the series, reading nothing.
    """
    stream = socket.create_connection(("127.0.0.1", port), timeout=DEADLINE)
    connection = Connection(stream)
    stream.settimeout(DEADLINE)
    if config is not None:
        connection.send_config(config)
    for _ in range(headers):
        connection.send_header(header_xml(series, exam, design, measurement))
    for index in range(len(series.table) if readouts is None else readouts):
        acquisition = ismrmrd.Acquisition.from_array(data(series.table, index))
        if counters is not None:
            acquisition.scan_counter = counters[index]
        if index == last:
            acquisition.setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
        connection.send(acquisition)
    connection.send_close()
    if leave:
        stream.close()
        return []
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
    received = stream(proxy.port, series["gre2d"])
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
        received[name] = stream(proxy.port, series["gre2d"], config=config)

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
    monkeypatch.setattr("pulserver.proxy._proxy._WORKER_TIMEOUT", 1.0)
    proxy = start_proxy(slots=1)
    config = json.dumps({"parameters": {"config": "gre2d", "delay": 3.0}})
    received = stream(proxy.port, series["gre2d"], config=config)
    assert closed(received)
    assert len(images(received)) == 1


def test_a_reconstruction_past_the_recon_timeout_is_stopped_and_reported(
    start_proxy, bucket
):
    _, series = bucket
    proxy = start_proxy(slots=1, recon_timeout=1.0)
    config = json.dumps({"parameters": {"config": "gre2d", "delay": 60.0}})
    started = time.monotonic()
    received = stream(proxy.port, series["gre2d"], config=config)
    assert time.monotonic() - started < 30
    assert not images(received)
    assert any(isinstance(item, str) and "did not finish" in item for item in received)
    assert len(images(stream(proxy.port, series["gre2d"]))) == 1


def test_a_stream_numbered_without_gaps_is_reconstructed(start_proxy, bucket):
    _, series = bucket
    proxy = start_proxy(slots=1)
    counters = [index + 1 for index in range(len(series["gre2d"].table))]
    assert len(images(stream(proxy.port, series["gre2d"], counters=counters))) == 1


def test_a_gap_in_the_scan_counters_stops_the_series_unreconstructed(
    start_proxy, bucket
):
    _, series = bucket
    proxy = start_proxy(slots=1)
    readouts = len(series["gre2d"].table)
    counters = [index + 1 + (index >= readouts // 2) for index in range(readouts)]
    received = stream(proxy.port, series["gre2d"], counters=counters)
    assert not images(received)
    assert any(isinstance(item, str) and "scan counter" in item for item in received)
    assert len(images(stream(proxy.port, series["gre2d"]))) == 1


def _refused(received, reason):
    return not images(received) and any(
        isinstance(item, str) and reason in item for item in received
    )


def test_a_series_opening_with_its_header_names_no_reconstruction_and_is_refused(
    start_proxy, bucket
):
    _, series = bucket
    proxy = start_proxy(slots=1)
    received = stream(proxy.port, series["gre2d"], config=None)
    assert _refused(received, "the config names no reconstruction")


@pytest.mark.parametrize(
    ("config", "plugin"),
    [
        ("recon3", "recon3"),
        ("/any/dir/recon3.py", "recon3"),
        ("recon3.py\n", "recon3"),
        ('{"parameters": {"config": "/any/dir/recon3.py"}}', "recon3"),
        ('{\n  "parameters": {\n    "config": "/any/dir/recon3.py"\n  }\n}', "recon3"),
        ('{"parameters": {}}', ""),
        ("", ""),
    ],
)
def test_the_proxy_resolves_a_plugin_file_path_to_its_stem(config, plugin):
    assert _config_plugin(config) == plugin


def test_a_series_whose_config_is_the_path_of_a_plugin_file_is_reconstructed_by_it(
    start_proxy, bucket, tmp_path
):
    _, series = bucket
    proxy = start_proxy(slots=1)
    path = tmp_path / "elsewhere" / "gre2d.py"
    assert len(images(stream(proxy.port, series["gre2d"], config=str(path)))) == 1


def test_a_series_that_ends_before_its_last_readout_is_refused(start_proxy, bucket):
    _, series = bucket
    proxy = start_proxy(slots=1)
    readouts = len(series["gre2d"].table)
    received = stream(proxy.port, series["gre2d"], readouts=readouts - 1)
    assert _refused(received, f"ended after {readouts - 1} of the {readouts} readouts")
    assert len(images(stream(proxy.port, series["gre2d"]))) == 1


@pytest.mark.parametrize("position", ["middle", "end"])
def test_only_the_last_readout_may_be_flagged_last_in_measurement(
    start_proxy, bucket, position
):
    _, series = bucket
    proxy = start_proxy(slots=1)
    readouts = len(series["gre2d"].table)
    last = readouts // 2 if position == "middle" else readouts - 1
    received = stream(proxy.port, series["gre2d"], last=last)
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
    assert _refused(stream(proxy.port, series["gre2d"], headers=headers), reason)


def test_a_stream_closed_before_its_header_is_refused(start_proxy, bucket):
    _, series = bucket
    proxy = start_proxy(slots=1)
    received = stream(proxy.port, series["gre2d"], config=None, headers=0, readouts=0)
    assert _refused(received, "where its MRD header belongs")


def test_a_crashing_plugin_closes_its_series_and_frees_the_slot(start_proxy, bucket):
    _, series = bucket
    proxy = start_proxy(slots=1)
    crashed = stream(proxy.port, series["gre2d_raw"], config="crash")
    assert closed(crashed)
    assert not images(crashed)
    assert any(isinstance(item, str) and "always fails" in item for item in crashed)
    assert len(images(stream(proxy.port, series["gre2d"]))) == 1


def test_the_spare_worker_is_replaced_after_each_series(start_proxy, bucket):
    _, series = bucket
    proxy = start_proxy(slots=1, spares=1)
    warm = proxy.workers.spare_pids()
    assert len(warm) == 1
    assert len(images(stream(proxy.port, series["gre2d"]))) == 1
    replaced = proxy.workers.spare_pids()
    assert len(replaced) == 1
    assert replaced != warm
    assert len(images(stream(proxy.port, series["gre2d"]))) == 1
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
        received[name] = stream(proxy.port, series["gre2d"], config=config)

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
    table = series["gre2d"].table
    fov_mm = table.spaces[0].fov_mm
    pixel_mm = (fov_mm[0] / MATRIX["nx"], fov_mm[1] / MATRIX["ny"])
    position = 1e-3 * np.array((2 * pixel_mm[0], 3 * pixel_mm[1], 0.0))

    def kspace(table, index):
        return point(table, index, position)

    received = stream(proxy.port, series["gre2d"], data=kspace)

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
            "pulserver.proxy",
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
    probe = "import sys, pulserver.proxy; print('bartorch' in sys.modules)"
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
            for image in images(
                stream(port, series["gre2d_raw"], config="exam", exam=exam)
            )
        ]
        for exam in ("E1", "E1", "E2", "E1")
    ]
    assert counts == [[1], [2], [1], [1]]


def test_a_closed_proxy_leaves_the_exam_root_for_the_proxies_still_running(tmp_path):
    """The root is the host's: a proxy closing is not every proxy closing.

    What a closing proxy takes with it is each exam nothing else is on, which
    :mod:`tests.test_exam_sharing` states; the root it shares stays.
    """
    root = tmp_path / "exams"
    proxy = ReconProxy(tmp_path, tmp_path, slots=1, spares=1, exam_directory=root)
    assert root.is_dir()
    proxy.close()
    assert root.is_dir()


def test_a_reconstruction_may_start_processes_of_its_own(start_proxy, bucket):
    _, series = bucket
    received = stream(start_proxy(slots=1).port, series["gre2d_raw"], config="children")
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
        for image in images(stream(port, series["gre2d_raw"], config="device"))
    ]
    assert values == read


def _published(path):
    """Return the version and rotation of the pose a reader of ``path`` takes."""
    from pulserver.proxy._motion import SLOTS, _check_of

    raw = path.read_bytes()
    poses = []
    for slot in range(SLOTS):
        at = 16 + slot * 48
        words = list(struct.unpack("=12I", raw[at : at + 48]))
        if words[0] and words[2] == _check_of(words):
            rotation = struct.unpack("=9f", struct.pack("=9I", *words[3:]))
            poses.append((words[0], rotation))
    return max(poses)


@pytest.mark.parametrize("forwarded", [False, True])
def test_a_motion_corrected_series_publishes_its_poses_to_the_scan(
    start_proxy, start_server, bucket, monkeypatch, forwarded
):
    """The scan reads the last pose stated, as stated: absolute, not composed."""
    from pulserver.proxy._motion import FILENAME

    poses = runpy.run_path(str(RECON_PLUGINS / "pose.py"))["POSES"]

    monkeypatch.setattr(_designs, "_asks_for_motion_correction", lambda _: True)
    root, series = bucket
    path = Path(root) / series["gre2d_raw"].design / FILENAME
    path.unlink(missing_ok=True)
    if forwarded:
        server = start_server(slots=1)
        proxy = start_proxy(forward=("127.0.0.1", server.port), forward_config="pose")
    else:
        proxy = start_proxy(slots=1)
    received = stream(proxy.port, series["gre2d_raw"], config="pose")

    assert closed(received)
    assert len(images(received)) == 1
    assert not any(isinstance(item, ismrmrd.Waveform) for item in received)
    version, rotation = _published(path)
    assert version == len(poses)
    assert rotation == pytest.approx(poses[-1])


def test_a_series_not_corrected_for_motion_writes_no_pose_file(start_proxy, bucket):
    from pulserver.proxy._motion import FILENAME

    root, series = bucket
    path = Path(root) / series["gre2d"].design / FILENAME
    stream(start_proxy(slots=1).port, series["gre2d"])
    assert not path.exists()


def _at(position_m):
    def data(table, index):
        return point(table, index, position_m)

    return data


def test_images_whose_client_has_gone_reach_it_when_it_returns(
    start_proxy, bucket, monkeypatch, tmp_path
):
    """Each returning client gets its own series, whichever order they return in."""
    monkeypatch.setattr(_held, "DEFAULT_HELD_DIRECTORY", tmp_path)
    _, series = bucket
    port = start_proxy(slots=2).port
    config = json.dumps({"parameters": {"config": "gre2d", "delay": 2.0}})
    positions = {"M1": (0.004, -0.002, 0.0), "M2": (-0.006, 0.004, 0.0)}
    for measurement, position in positions.items():
        stream(
            port,
            series["gre2d_raw"],
            config=config,
            data=_at(position),
            measurement=measurement,
            leave=True,
        )
    _wait_until(
        lambda: len([p for p in tmp_path.iterdir() if p.suffix != ".part"]) == 2,
        "the outputs of the series whose clients left were not held",
    )
    for measurement in reversed(positions):
        received = stream(
            port, series["gre2d_raw"], readouts=0, measurement=measurement
        )
        assert closed(received)
        (image,) = images(received)
        (live,) = images(
            stream(
                port,
                series["gre2d_raw"],
                config="gre2d",
                data=_at(positions[measurement]),
            )
        )
        np.testing.assert_array_equal(image.data, live.data)
    assert list(tmp_path.iterdir()) == []


def test_a_client_that_stays_leaves_nothing_held(
    start_proxy, bucket, monkeypatch, tmp_path
):
    monkeypatch.setattr(_held, "DEFAULT_HELD_DIRECTORY", tmp_path)
    _, series = bucket
    received = stream(
        start_proxy(slots=1).port, series["gre2d_raw"], config="gre2d", measurement="M3"
    )
    assert len(images(received)) == 1
    assert list(tmp_path.iterdir()) == []


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
    received = stream(proxy.port, series["gre2d"], design=design)
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

    forwarded = stream(forwarding.port, series["gre2d"], data=kspace)
    reconstructed = stream(local.port, series["gre2d"], data=kspace)

    assert closed(forwarded)
    assert forwarding.workers is None
    assert len(images(forwarded)) == len(images(reconstructed)) == 1
    np.testing.assert_array_equal(
        images(forwarded)[0].data, images(reconstructed)[0].data
    )


def _in_process(
    root,
    series,
    *,
    data=flat,
    readouts=None,
    design=None,
    config="gre2d",
    plugins=RECON_PLUGINS,
    fov_offset_mm=None,
):
    """Reconstruct one series with :class:`LocalReconstruction`; return whether it was, and what it sent."""
    header = ismrmrd.xsd.CreateFromDocument(header_xml(series, design=design))
    if fov_offset_mm is not None:
        if header.userParameters is None:
            header.userParameters = ismrmrd.xsd.userParametersType()
        header.userParameters.userParameterString.append(
            ismrmrd.xsd.userParameterStringType(
                name="fov_offset_mm", value=" ".join(map(str, fov_offset_mm))
            )
        )
    count = len(series.table) if readouts is None else readouts
    acquisitions = (
        ismrmrd.Acquisition.from_array(data(series.table, index))
        for index in range(count)
    )
    received = []
    done = LocalReconstruction(root, plugins).run(
        header, acquisitions, received.append, config
    )
    return done, received


def test_a_series_reconstructed_in_this_process_is_centred_at_the_offset_its_header_states(
    bucket,
):
    root, series = bucket
    position_m = (0.004, -0.002, 0.0)

    def displaced(table, index):
        return point(table, index, position_m)

    def centred(table, index):
        return point(table, index, (0.0, 0.0, 0.0))

    _, shifted = _in_process(
        root,
        series["gre2d"],
        data=displaced,
        fov_offset_mm=[1e3 * value for value in position_m],
    )
    _, reference = _in_process(root, series["gre2d"], data=centred)

    np.testing.assert_allclose(
        images(shifted)[0].data, images(reference)[0].data, atol=1e-4
    )


def test_a_series_reconstructed_in_this_process_returns_the_image_a_worker_returns(
    start_proxy, bucket
):
    root, series = bucket
    proxy = start_proxy(slots=1)

    def kspace(table, index):
        return point(table, index, (0.004, -0.002, 0.0))

    done, received = _in_process(root, series["gre2d"], data=kspace)
    reconstructed = stream(proxy.port, series["gre2d"], data=kspace)

    assert done
    assert len(images(received)) == len(images(reconstructed)) == 1
    np.testing.assert_array_equal(
        images(received)[0].data, images(reconstructed)[0].data
    )


def test_in_this_process_a_reconstruction_is_found_in_any_of_the_plugin_directories(
    bucket, tmp_path
):
    root, series = bucket

    alone, received_alone = _in_process(root, series["gre2d"])
    searched, received = _in_process(
        root, series["gre2d"], plugins=[tmp_path / "absent", RECON_PLUGINS]
    )

    assert alone and searched
    np.testing.assert_array_equal(
        images(received)[0].data, images(received_alone)[0].data
    )


@pytest.mark.parametrize(
    ("options", "reason"),
    [
        ({"readouts": 3}, "pulserver: the stream ended after 3 of the"),
        ({"design": "0" * 18}, "pulserver: no design 000000000000000000"),
        (
            {"config": "crash"},
            "pulserver: crash failed: this reconstruction always fails",
        ),
        ({"config": ""}, "pulserver: the config names no reconstruction"),
        (
            {"config": '{"parameters": {"config": "../gre2d"}}'},
            "pulserver: invalid plugin name '../gre2d'",
        ),
        ({"config": "absent"}, "pulserver: no plugin 'absent'"),
    ],
)
def test_in_this_process_a_series_is_refused_or_fails_with_the_proxys_text(
    bucket, options, reason
):
    root, series = bucket

    done, received = _in_process(root, series["gre2d"], **options)

    assert not done
    assert not images(received)
    assert [item for item in received if isinstance(item, str)][-1].startswith(reason)


def test_a_manifest_naming_a_reconstruction_is_read_and_the_name_ignored(
    bucket, tmp_path
):
    """A store written when a design named its reconstruction still reads."""
    root, series = bucket
    design = series["gre2d"].design
    shutil.copytree(Path(root) / design, tmp_path / design)
    manifest = tmp_path / design / "manifest.json"
    manifest.write_text(
        json.dumps({**json.loads(manifest.read_text()), "recon": "crash"})
    )

    done, received = _in_process(tmp_path, series["gre2d"])

    assert done
    assert len(images(received)) == 1


@pytest.mark.parametrize("configured", [None, "default.xml"])
def test_a_forwarded_series_names_its_reconstruction_in_a_config_file_message(
    start_proxy, bucket, configured
):
    _, series = bucket
    recording = _RecordingServer(_texts)
    proxy = start_proxy(
        forward=("127.0.0.1", recording.port), forward_config=configured
    )

    config = '{"parameters": {"config": "gre2d"}}'
    received = stream(proxy.port, series["gre2d"], config=config)
    recording.join()

    name, header, description, *acquisitions = recording.received
    assert name == (configured or "gre2d")
    assert is_message(description)
    encoding = header.encoding[0]
    assert encoding.reconSpace.matrixSize.x == MATRIX["nx"]
    oversampling = encoding.encodedSpace.matrixSize.x / MATRIX["nx"]
    assert oversampling >= 1
    assert encoding.encodedSpace.fieldOfView_mm.x == pytest.approx(
        oversampling * encoding.reconSpace.fieldOfView_mm.x
    )
    assert len(acquisitions) == len(series["gre2d"].table)
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
    received = stream(proxy.port, series["gre2d"])
    recording.join()

    assert "first" in received
    assert any(
        isinstance(item, str) and "message of type 1030" in item for item in received
    )
    assert closed(received)


def test_the_reconstruction_reads_the_sequence_description_before_any_acquisition(
    start_proxy, bucket
):
    _, series = bucket
    recording = _RecordingServer(_texts)
    proxy = start_proxy(forward=("127.0.0.1", recording.port))
    stream(proxy.port, series["gre2d"])
    recording.join()

    kinds = [
        "description" if is_message(item) else type(item).__name__
        for item in recording.received
    ]
    assert kinds.count("description") == 1
    assert kinds.index("description") < kinds.index("Acquisition")


def test_the_sequence_description_is_not_sent_back_to_the_client(start_proxy, bucket):
    """A reconstruction echoes what it does not read, and the scanner did not ask for it."""
    _, series = bucket
    proxy = start_proxy(slots=1)
    received = stream(proxy.port, series["gre2d"])
    assert images(received)
    assert not any(is_message(item) for item in received)


def test_images_come_back_as_dicom_when_asked(start_proxy, bucket):
    """A client that reads DICOM alone, such as a scanner's, gets DICOM."""
    _, series = bucket
    proxy = start_proxy(slots=1, dicom=True)
    received = stream(proxy.port, series["gre2d"])

    assert not images(received)
    assert sum(isinstance(item, DicomWithName) for item in received) == 1
    assert closed(received)


def test_forwarded_images_come_back_as_dicom_when_asked(
    start_proxy, start_server, bucket
):
    _, series = bucket
    server = start_server(slots=1)
    proxy = start_proxy(forward=("127.0.0.1", server.port), dicom=True)
    received = stream(proxy.port, series["gre2d"])

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
    received = stream(proxy.port, series["gre2d"])
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
    from pulserver.proxy.__main__ import main

    with pytest.raises(SystemExit) as stopped:
        main(["--store", str(tmp_path), "--port", "0", *arguments])
    assert stopped.value.code == 2


def test_a_series_forwarded_to_no_server_is_refused(start_proxy, bucket):
    _, series = bucket
    with socket.create_server(("127.0.0.1", 0)) as unused:
        port = unused.getsockname()[1]
    proxy = start_proxy(forward=("127.0.0.1", port))
    received = stream(proxy.port, series["gre2d"])
    assert _refused(received, f"the reconstruction server at 127.0.0.1:{port}")


def test_a_series_whose_config_names_no_plugin_is_refused_by_the_server(
    start_server, bucket
):
    _, series = bucket
    server = start_server(slots=1)
    received = stream(server.port, series["gre2d"], config="")
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


def test_a_recorded_series_run_offline_against_its_store_gives_the_proxys_image(
    bucket, tmp_path
):
    root, series = bucket
    played = series["gre2d"]

    def kspace(table, index):
        return point(table, index, (0.004, -0.002, 0.0))

    path = str(tmp_path / "scan.h5")
    dataset = ismrmrd.Dataset(path, "dataset", create_if_needed=True)
    dataset.write_xml_header(header_xml(played))
    for index in range(len(played.table)):
        dataset.append_acquisition(
            ismrmrd.Acquisition.from_array(kspace(played.table, index))
        )
    dataset.close()

    offline = load_plugin(RECON_PLUGINS / "gre2d.py").run(path, store=root)
    _, received = _in_process(root, played, data=kspace)

    np.testing.assert_array_equal(images(offline)[0].data, images(received)[0].data)


def test_a_kept_series_runs_offline_and_gives_the_image_the_proxy_returned(
    start_proxy, bucket, tmp_path
):
    """What is kept is a scan a plugin can be run on again, against the store."""
    root, series = bucket
    played = series["gre2d"]

    def kspace(table, index):
        return point(table, index, (0.004, -0.002, 0.0))

    kept = tmp_path / "kept"
    proxy = start_proxy(slots=1, save_to=kept)
    received = stream(proxy.port, played, data=kspace)

    written = sorted(kept.glob("*.h5"))
    assert len(written) == 1, f"kept {written}"
    offline = load_plugin(RECON_PLUGINS / "gre2d.py").run(str(written[0]), store=root)
    np.testing.assert_array_equal(images(offline)[0].data, images(received)[0].data)


def test_what_a_proxy_keeps_is_the_header_the_scanner_sent(
    start_proxy, bucket, tmp_path
):
    """The proxy enriches what it passes on, and keeps what it was given."""
    _, series = bucket
    kept = tmp_path / "kept"
    proxy = start_proxy(slots=1, save_to=kept)

    received = stream(proxy.port, series["gre2d"])

    assert [np.squeeze(image.data).shape for image in images(received)] == [
        (MATRIX["ny"], MATRIX["nx"])
    ]
    written = sorted(kept.glob("*.h5"))
    held = ismrmrd.Dataset(str(written[0]), "dataset", create_if_needed=False)
    try:
        header = ismrmrd.xsd.CreateFromDocument(held.read_xml_header())
    finally:
        held.close()
    assert header.encoding[0].encodedSpace.matrixSize.x == 1
    assert header.sequenceParameters is None


def test_two_series_at_once_are_kept_in_files_of_their_own(
    start_proxy, bucket, tmp_path
):
    """Two series start within one second, and one file cannot hold both."""
    _, series = bucket
    kept = tmp_path / "kept"
    proxy = start_proxy(slots=2, save_to=kept)
    played = [series["gre2d"], series["gre2d_raw"]]
    received = {}

    def play(index):
        received[index] = stream(proxy.port, played[index], config="gre2d")

    threads = [threading.Thread(target=play, args=(index,)) for index in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=DEADLINE)

    assert all(closed(items) for items in received.values())
    written = sorted(kept.glob("*.h5"))
    assert len(written) == 2, f"kept {written}"
    counts = []
    for path in written:
        held = ismrmrd.Dataset(str(path), "dataset", create_if_needed=False)
        try:
            counts.append(int(held.number_of_acquisitions()))
        finally:
            held.close()
    assert counts == [len(played[0].table), len(played[1].table)] or counts == [
        len(played[1].table),
        len(played[0].table),
    ], counts


def test_a_forwarded_series_is_kept_by_the_server_it_reaches(
    start_proxy, start_server, bucket, tmp_path
):
    """A server keeps the series a proxy forwards, enriched as it arrives."""
    _, series = bucket
    kept = tmp_path / "kept"
    server = start_server(slots=1, save_to=kept)
    forwarding = start_proxy(forward=("127.0.0.1", server.port))

    received = stream(forwarding.port, series["gre2d"])

    assert closed(received)
    written = sorted(kept.glob("*.h5"))
    assert len(written) == 1, f"kept {written}"
    held = ismrmrd.Dataset(str(written[0]), "dataset", create_if_needed=False)
    try:
        assert int(held.number_of_acquisitions()) == len(series["gre2d"].table)
        header = ismrmrd.xsd.CreateFromDocument(held.read_xml_header())
    finally:
        held.close()
    # The readout is widened to the oversampling of the full echo, the phase
    # encodes are not.
    assert header.encoding[0].encodedSpace.matrixSize.y == MATRIX["ny"]
    assert pytest.approx([MATRIX["TE"] / 1e3]) == header.sequenceParameters.TE


def test_a_series_is_kept_nowhere_where_nowhere_is_asked_for(
    start_proxy, bucket, tmp_path
):
    _, series = bucket
    proxy = start_proxy(slots=1)
    stream(proxy.port, series["gre2d"])
    assert not list(tmp_path.rglob("*.h5"))


@pytest.mark.parametrize("service", ("proxy", "server"))
def test_a_service_nothing_connects_to_closes_on_its_own(service, bucket):
    """One started for a scan outlives whoever started it; this is how it ends."""
    root, _ = bucket
    running = (
        ReconProxy(root, RECON_PLUGINS, slots=1, spares=0, idle_timeout=0.5)
        if service == "proxy"
        else ReconServer(RECON_PLUGINS, slots=1, spares=0, idle_timeout=0.5)
    )
    running.bind(0)
    thread = threading.Thread(target=running.serve, daemon=True)
    thread.start()
    try:
        thread.join(timeout=20)
        assert not thread.is_alive(), f"the {service} was still waiting"
    finally:
        running.close()


def test_the_idle_timeout_counts_from_the_last_client_leaving(bucket):
    """A series longer than the timeout must not close the proxy under its client."""
    root, _ = bucket
    proxy = ReconProxy(root, RECON_PLUGINS, slots=1, spares=0, idle_timeout=1.0)
    port = proxy.bind(0)
    thread = threading.Thread(target=proxy.serve, daemon=True)
    thread.start()
    try:
        held = socket.create_connection(("127.0.0.1", port), timeout=DEADLINE)
        try:
            thread.join(timeout=4.0)
            assert thread.is_alive(), "the proxy closed while a client was connected"
        finally:
            held.close()
        thread.join(timeout=20)
        assert not thread.is_alive(), "the proxy was still waiting"
    finally:
        proxy.close()
