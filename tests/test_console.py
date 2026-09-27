"""The console gateway: a console's design calls, its exam's localizer and its scans, in process and over a WebSocket."""

import asyncio
import base64
import io
import json
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pydicom
import pytest
from _host import ANY_ORIENTATION, PLUGINS

from pulserver import virtual
from pulserver.host import DesignStore
from pulserver.host._blocks import format_limits
from pulserver.protocol import FOV_OFFSET, FOV_ROTATION, PROTOCOL_BEGIN, PROTOCOL_END
from pulserver.proxy import ReconProxy
from pulserver.virtual._console import Console, _connection
from pulserver.virtual._localizer import PLANES

websockets = pytest.importorskip("websockets")

RECON_PLUGINS = Path(__file__).parent / "recon_plugins"
DEADLINE = 180.0


def _block(**values):
    prescribed = {
        **dict.fromkeys(FOV_OFFSET, 0.0),
        **dict(zip(FOV_ROTATION, np.eye(3).ravel().tolist(), strict=True)),
        **values,
    }
    lines = [f"{name}: {value!r}" for name, value in prescribed.items()]
    return "\n".join([PROTOCOL_BEGIN, *lines, PROTOCOL_END]) + "\n"


def _console(tmp_path, **options):
    return Console(
        plugins=PLUGINS,
        limits=format_limits(ANY_ORIENTATION),
        store=tmp_path / "designs",
        spacing=2e-3,
        **options,
    )


@pytest.fixture
def proxy(tmp_path):
    running = ReconProxy(tmp_path / "designs", RECON_PLUGINS, slots=1)
    running.bind(0)
    thread = threading.Thread(target=running.serve, daemon=True)
    thread.start()
    yield running
    running.close()
    thread.join(timeout=DEADLINE)


def test_a_console_lists_and_generates_as_the_design_command_does(tmp_path):
    console = _console(tmp_path)

    listing = console.design("list", "gre2d")
    generated = console.design("generate", "gre2d", _block(TE=5000, nx=32, ny=32))

    assert "gre2d" in console.plugin_names()
    assert listing["status"] == 0
    assert listing["reply"].splitlines()[:2] == ["PROTOCOL", PROTOCOL_BEGIN]
    assert generated["status"] == 0
    assert generated["reply"].startswith("GENERATED ")
    assert (DesignStore(tmp_path / "designs").directory(generated["design"])).is_dir()


def test_a_console_makes_only_the_design_calls(tmp_path):
    with pytest.raises(ValueError, match="design calls"):
        _console(tmp_path).design("prune")


def test_an_exam_returns_its_subjects_three_plane_localizer_as_dicom(tmp_path):
    console = _console(tmp_path)

    files = console.exam("vials")

    datasets = [pydicom.dcmread(io.BytesIO(f)) for f in files]
    assert len(datasets) == 3
    assert {str(d.PatientName) for d in datasets} == {"vials"}
    assert all(d.pixel_array.max() > 0 for d in datasets[:1])
    orientations = [
        tuple(float(v) for v in d.ImageOrientationPatient) for d in datasets
    ]
    assert orientations == [
        tuple(np.concatenate(directions)) for directions in PLANES.values()
    ]


def test_the_subject_brainweb_starts_an_exam_on_brainweb(tmp_path, monkeypatch):
    fractions = np.zeros((4, 4, 4, 10), dtype=np.float32)
    monkeypatch.setitem(
        sys.modules,
        "brainweb_dl",
        types.SimpleNamespace(get_mri=lambda *a, **k: fractions),
    )
    console = _console(tmp_path)

    console.exam("BrainWeb")

    assert isinstance(console.phantom, virtual.BrainWeb)


def test_a_scan_streams_its_clock_and_returns_the_reconstruction_as_dicom(
    tmp_path, proxy
):
    console = _console(tmp_path, recon=("127.0.0.1", proxy.port))
    design = console.design("generate", "gre2d", _block(TE=5000, nx=32, ny=32))[
        "design"
    ]
    console.exam("vials")
    messages = []

    status = console.scan(
        design, rotation=np.eye(3), centre_mm=(0.0, 0.0, 0.0), emit=messages.append
    )

    assert status == 0
    clocks = [m for m in messages if "clock" in m]
    assert clocks
    assert clocks[-1]["clock"] == pytest.approx(clocks[-1]["duration"])
    images = [
        pydicom.dcmread(io.BytesIO(base64.b64decode(m["dicom"])))
        for m in messages
        if "dicom" in m
    ]
    assert images
    assert images[0].pixel_array.max() > 0


def test_a_cancelled_scan_stops_and_reports_it(tmp_path):
    console = _console(tmp_path)
    design = console.design("generate", "gre2d", _block(TE=5000, nx=32, ny=32))[
        "design"
    ]
    messages = []

    status = console.scan(
        design,
        rotation=np.eye(3),
        centre_mm=(0.0, 0.0, 0.0),
        emit=messages.append,
        cancelled=lambda: bool(messages),
    )

    assert status == 1
    assert len(messages) == 1


def test_a_scan_asked_for_its_sound_streams_it_with_its_clock(tmp_path):
    console = _console(tmp_path)
    design = console.design("generate", "gre2d", _block(TE=5000, nx=32, ny=32))[
        "design"
    ]
    messages = []

    status = console.scan(
        design,
        rotation=np.eye(3),
        centre_mm=(0.0, 0.0, 0.0),
        emit=messages.append,
        sound=True,
    )

    samples = np.concatenate(
        [
            np.frombuffer(base64.b64decode(m["sound"]), dtype="<i2").reshape(-1, 2)
            for m in messages
        ]
    )
    assert status == 0
    assert {m["rate"] for m in messages} == {virtual.SAMPLE_RATE}
    expected = messages[-1]["duration"] * virtual.SAMPLE_RATE
    assert abs(len(samples) - expected) <= len(messages)
    assert np.abs(samples).max() > 0


def test_the_gateway_answers_a_consoles_calls_over_a_websocket(tmp_path):
    console = _console(tmp_path)
    loop = asyncio.new_event_loop()
    started = threading.Event()
    server = {}

    async def run():
        server["it"] = await websockets.serve(
            lambda ws: _connection(console, ws), "127.0.0.1", 0, max_size=None
        )
        started.set()
        await server["it"].wait_closed()

    thread = threading.Thread(
        target=loop.run_until_complete, args=(run(),), daemon=True
    )
    thread.start()
    assert started.wait(DEADLINE)
    port = server["it"].sockets[0].getsockname()[1]
    try:
        from websockets.sync.client import connect

        with connect(f"ws://127.0.0.1:{port}", max_size=None) as client:
            client.send(json.dumps({"id": 1, "call": "plugins"}))
            plugins = json.loads(client.recv(timeout=DEADLINE))
            client.send(json.dumps({"id": 2, "call": "exam", "subject": "vials"}))
            exam = json.loads(client.recv(timeout=DEADLINE))
            client.send(json.dumps({"id": 3, "call": "reboot"}))
            refused = json.loads(client.recv(timeout=DEADLINE))
    finally:
        loop.call_soon_threadsafe(server["it"].close)
        thread.join(timeout=DEADLINE)

    assert plugins["id"] == 1
    assert "gre2d" in plugins["plugins"]
    assert exam["id"] == 2
    assert len(exam["localizer"]) == 3
    assert refused == {"id": 3, "error": "unknown call 'reboot'"}


def test_the_console_command_serves_a_console_of_its_options(tmp_path, monkeypatch):
    from pulserver import _cli
    from pulserver.virtual import _console

    limits = tmp_path / "limits.txt"
    limits.write_text(format_limits(ANY_ORIENTATION))
    served = {}

    async def serve(console, host, port):
        served.update(console=console, host=host, port=port)

    monkeypatch.setattr(_console, "serve", serve)

    status = _cli.main(
        [
            "console",
            "--plugins",
            str(PLUGINS),
            "--limits",
            str(limits),
            "--store",
            str(tmp_path / "designs"),
            "--recon",
            "recon.local:9020",
            "--port",
            "9876",
            "--spacing",
            "2",
            "--coils",
            "4",
        ]
    )

    assert status == 0
    assert (served["host"], served["port"]) == ("127.0.0.1", 9876)
    console = served["console"]
    assert console.recon == ("recon.local", 9020)
    assert console.spacing == pytest.approx(2e-3)
    assert console.coils == 4
    assert console.field_t == ANY_ORIENTATION["B0"]
