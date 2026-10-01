"""The console gateway: a console's design calls, its exam's localizer and its scans, in process and over a WebSocket."""

import asyncio
import base64
import contextlib
import io
import json
import socket
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pydicom
import pypulseqpp as pp
import pytest
from _host import ANY_ORIENTATION, LIMITS, PLUGINS
from _virtual import FIELD_CHANNELS, write_fields

from pulserver import ir, virtual
from pulserver.host import DesignStore
from pulserver.host._blocks import format_limits
from pulserver.protocol import FOV_OFFSET, FOV_ROTATION, PROTOCOL_BEGIN, PROTOCOL_END
from pulserver.proxy import ReconProxy
from pulserver.virtual._command import ORIENTATIONS
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


def _console(tmp_path, spacing=2e-3, **options):
    return Console(
        plugins=PLUGINS,
        limits=format_limits(ANY_ORIENTATION),
        store=tmp_path / "designs",
        spacing=spacing,
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


def test_a_console_loads_brainweb_once_and_computes_its_field_after_the_localizer(
    tmp_path, monkeypatch
):
    fractions = np.zeros((4, 4, 4, 10), dtype=np.float32)
    fractions[..., 3] = 1.0
    loads = []

    def get_mri(*args, **kwargs):
        loads.append(args)
        return fractions

    monkeypatch.setitem(
        sys.modules, "brainweb_dl", types.SimpleNamespace(get_mri=get_mri)
    )
    console = _console(tmp_path)

    console.exam("brainweb")
    brain = console.phantom
    console._warming.join(timeout=60.0)
    computed = "field_ppm" in vars(brain)
    console.exam("vials")
    console.exam("brainweb")

    assert computed
    assert console.phantom is brain
    assert len(loads) == 1


@pytest.fixture
def brainweb(monkeypatch):
    """BrainWeb's model, empty, in place of the one brainweb-dl downloads."""
    fractions = np.zeros((4, 4, 4, 10), dtype=np.float32)
    monkeypatch.setitem(
        sys.modules,
        "brainweb_dl",
        types.SimpleNamespace(get_mri=lambda *a, **k: fractions),
    )


@pytest.fixture
def fields(tmp_path):
    """A directory of every coil's field maps, and of VOPs for those that transmit."""
    directory = tmp_path / "fields"
    directory.mkdir()
    return write_fields(directory)


def test_a_console_with_field_maps_examines_brainweb_whatever_the_subject(
    tmp_path, brainweb, fields
):
    console = _console(tmp_path, fields=fields)

    files = console.exam("vials")

    assert isinstance(console.phantom, virtual.BrainWeb)
    assert {str(pydicom.dcmread(io.BytesIO(f)).PatientName) for f in files} == {"vials"}


def test_a_console_with_field_maps_lists_the_channels_its_maps_hold(tmp_path, fields):
    coils = _console(tmp_path, fields=fields).coils()

    assert coils == [
        {"name": "body", "transmit": FIELD_CHANNELS["body"], "receive": 2},
        {"name": "body/head48", "transmit": 2, "receive": FIELD_CHANNELS["head48"]},
        {"name": "head8/head32", "transmit": 8, "receive": FIELD_CHANNELS["head32"]},
    ]


def test_a_design_is_made_under_the_vops_of_the_exams_transmit_coil(
    tmp_path, brainweb, fields
):
    console = _console(tmp_path, fields=fields, coil="head8/head32")
    store = DesignStore(tmp_path / "designs")
    block = _block(TE=5000, nx=32, ny=32)

    head8 = console.design("generate", "gre2d", block)["design"]
    console.exam("brainweb", "body")
    body = console.design("generate", "gre2d", block)["design"]

    assert head8 != body
    for design, transmit in ((head8, "head8"), (body, "body")):
        limits = store.manifest(design)["limits"]
        assert limits["vop_file"] == str(fields / f"{transmit}_vops.npz")
        (cached,) = ir.summary(
            store.directory(design) / "sequence.seq",
            pp.Opts(**LIMITS),
            cache_ext=".pseg",
        )["subsequences"]
        assert cached["vop_sar_ratio"] > 0.0
        assert cached["vop_global_sar_ratio"] > 0.0


def test_a_console_lists_each_coil_an_exam_can_start_with_and_its_channels(tmp_path):
    coils = _console(tmp_path).coils()

    assert coils == [
        {
            "name": name,
            "transmit": coil.transmit_channels,
            "receive": coil.receive_channels,
        }
        for name, coil in virtual.COILS.items()
    ]


def test_an_exam_starts_in_the_coil_it_names_and_keeps_it_until_another_is_named(
    tmp_path,
):
    console = _console(tmp_path)

    console.exam("vials", "body/head48")
    console.exam("vials")

    assert console.coil is virtual.COILS["body/head48"]
    with pytest.raises(ValueError, match="coils are"):
        console.exam("vials", "knee")


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


def _images(messages):
    return [
        pydicom.dcmread(io.BytesIO(base64.b64decode(m["dicom"]))).pixel_array
        for m in messages
        if "dicom" in m
    ]


def test_a_scan_reconstructed_in_this_process_returns_the_images_a_proxy_returns(
    tmp_path, proxy
):
    proxied = _console(tmp_path, recon=("127.0.0.1", proxy.port))
    local = _console(tmp_path, recon_plugins=RECON_PLUGINS)
    design = proxied.design("generate", "gre2d", _block(TE=5000, nx=32, ny=32))[
        "design"
    ]
    returned = {}
    for name, console in (("proxied", proxied), ("local", local)):
        console.exam("vials")
        messages = []
        status = console.scan(
            design, rotation=np.eye(3), centre_mm=(0.0, 0.0, 0.0), emit=messages.append
        )
        assert status == 0
        returned[name] = _images(messages)

    assert len(returned["local"]) == len(returned["proxied"]) == 1
    np.testing.assert_array_equal(returned["local"][0], returned["proxied"][0])


def test_an_exams_scans_play_on_its_isochromats_each_from_equilibrium(tmp_path):
    console = _console(tmp_path, recon_plugins=RECON_PLUGINS)
    design = console.design("generate", "gre2d", _block(TE=5000, nx=32, ny=32))[
        "design"
    ]
    built = []

    def scanned():
        messages = []
        status = console.scan(
            design, rotation=np.eye(3), centre_mm=(0.0, 0.0, 0.0), emit=messages.append
        )
        assert status == 0
        return _images(messages)[0]

    def exam():
        console.exam("vials")
        build = console.phantom.isochromats
        console.phantom.isochromats = lambda *a, **k: built.append(1) or build(*a, **k)

    exam()
    first, second = scanned(), scanned()
    exam()
    third = scanned()

    assert len(built) == 2
    np.testing.assert_array_equal(second, first)
    np.testing.assert_array_equal(third, first)


def _scanned(console, design, rotation):
    status = console.scan(
        design, rotation=rotation, centre_mm=(0.0, 0.0, 0.0), emit=lambda _: None
    )
    assert status == 0


def _builds(console):
    """The spacings the exam's phantom is sampled at, one per set of isochromats built."""
    spacings = []
    build = console.phantom.isochromats
    console.phantom.isochromats = lambda spacing, **k: (
        spacings.append(spacing) or build(spacing, **k)
    )
    return spacings


def test_a_scan_plays_on_the_isochromats_in_the_slabs_its_excitations_excite(
    tmp_path,
):
    console = _console(tmp_path)
    design = console.design("generate", "gre2d", _block(TE=5000, nx=32, ny=32))[
        "design"
    ]
    console.exam("vials")
    coronal = ORIENTATIONS["coronal"]

    _scanned(console, design, coronal)

    region, isochromats = console._isochromats
    sequence = DesignStore(tmp_path / "designs").directory(design) / "sequence.seq"
    assert region == virtual.excited(sequence, coronal)
    kept = console.phantom.count(2e-3, field_t=console.field_t, region=region)
    assert len(isochromats) == kept
    assert 0 < kept < console.phantom.count(2e-3, field_t=console.field_t)


def test_scans_that_excite_other_slabs_play_on_isochromats_of_their_own(tmp_path):
    console = _console(tmp_path)
    design = console.design("generate", "gre2d", _block(TE=5000, nx=32, ny=32))[
        "design"
    ]
    console.exam("vials")
    spacings = _builds(console)

    _scanned(console, design, np.eye(3))
    _scanned(console, design, np.eye(3))
    _scanned(console, design, ORIENTATIONS["coronal"])

    assert len(spacings) == 2


def test_a_console_coarsens_its_spacing_until_a_scan_keeps_no_more_isochromats_than_it_may(
    tmp_path,
):
    console = _console(tmp_path, spacing=1e-3)
    design = console.design("generate", "gre2d", _block(TE=5000, nx=32, ny=32))[
        "design"
    ]
    console.exam("vials")
    console.max_isochromats = console.phantom.count(2e-3, field_t=console.field_t)
    spacings = _builds(console)

    _scanned(console, design, np.eye(3))

    assert spacings == [pytest.approx(2e-3)]


def test_a_scan_no_spacing_keeps_within_the_consoles_isochromats_is_refused(tmp_path):
    console = _console(tmp_path, max_isochromats=0)
    design = console.design("generate", "gre2d", _block(TE=5000, nx=32, ny=32))[
        "design"
    ]
    console.exam("vials")

    with pytest.raises(ValueError, match="at every spacing"):
        _scanned(console, design, np.eye(3))


def test_a_console_counts_its_spins_per_voxel_in_keeping_within_its_isochromats(
    tmp_path,
):
    console = _console(tmp_path, spacing=1e-3, spins=4, voxel="box")
    design = console.design("generate", "gre2d", _block(TE=5000, nx=32, ny=32))[
        "design"
    ]
    console.exam("vials")
    console.max_isochromats = console.phantom.count(
        2e-3, field_t=console.field_t, spins=4
    )
    spacings = _builds(console)

    _scanned(console, design, np.eye(3))

    region, isochromats = console._isochromats
    assert spacings == [pytest.approx(2e-3)]
    single = console.phantom.count(2e-3, field_t=console.field_t, region=region)
    assert len(isochromats) == 4 * single


def test_a_console_moves_its_subject_from_rest_at_the_start_of_each_scan(tmp_path):
    times = []

    def still(t, positions):
        times.append(t)
        return positions

    console = _console(tmp_path, motion=still)
    design = console.design("generate", "gre2d", _block(TE=5000, nx=32, ny=32))[
        "design"
    ]
    console.exam("vials")

    _scanned(console, design, np.eye(3))
    first = list(times)
    times.clear()
    _scanned(console, design, np.eye(3))

    assert console._isochromats[1].moving
    assert first[0] == times[0] == 0.0
    assert times == first


def test_a_console_examines_a_brainweb_whose_tissues_diffuse_when_asked(
    tmp_path, brainweb
):
    still = _console(tmp_path)
    diffusing = _console(tmp_path, diffusion=True)

    still.exam("brainweb")
    diffusing.exam("brainweb")

    assert not still.phantom.diffusion
    assert dict(diffusing.phantom.diffusion) == dict(virtual.BrainWeb.DIFFUSION)


def test_a_console_reconstructs_through_a_proxy_or_in_process_not_both(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        _console(tmp_path, recon=("127.0.0.1", 9), recon_plugins=RECON_PLUGINS)


def test_a_request_answered_in_process_carries_what_the_websocket_carries(tmp_path):
    console = _console(tmp_path, recon_plugins=RECON_PLUGINS)
    answers = []

    def answer(call, **fields):
        messages = []
        console.answer({"call": call, **fields}, messages.append)
        answers.append(messages)
        return messages

    [generated] = answer(
        "generate", plugin="gre2d", block=_block(TE=5000, nx=32, ny=32)
    )
    answer("exam", subject="vials")
    scanned = answer("scan", design=generated["design"])
    [refused] = answer("reboot")
    [failed] = answer("scan")

    assert [len(messages) for messages in answers[:2]] == [1, 1]
    assert scanned[-1] == {"done": 0}
    assert "clock" in scanned[0]
    assert len(_images(scanned)) == 1
    assert refused == {"error": "unknown call 'reboot'"}
    assert failed == {"error": "KeyError: 'design'"}


def test_a_scan_whose_reconstruction_is_refused_returns_the_reason_and_fails(
    tmp_path,
):
    console = _console(tmp_path, recon_plugins=RECON_PLUGINS)
    design = console.design("generate", "gre2d_raw", _block(TE=5000, nx=32, ny=32))[
        "design"
    ]
    messages = []

    status = console.scan(
        design, rotation=np.eye(3), centre_mm=(0.0, 0.0, 0.0), emit=messages.append
    )

    assert status == 1
    assert not _images(messages)
    assert [m["text"] for m in messages if "text" in m] == [
        "pulserver: neither the design nor the config names a reconstruction"
    ]


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


def test_a_scan_cancelled_while_it_prepares_stops_before_its_clock_starts(
    tmp_path, monkeypatch
):
    simulate = virtual.Scan._readouts

    def slow(self, first, last):
        time.sleep(1.0)
        return simulate(self, first, last)

    # Four times as long to simulate as to play: the clock would wait about
    # three quarters of the simulation.
    monkeypatch.setattr(virtual.Scan, "_readouts", slow)
    console = _console(tmp_path, speed=1.0)
    design = console.design("generate", "gre2d", _block(TE=5000, nx=32, ny=32))[
        "design"
    ]
    messages = []
    started = time.monotonic()

    status = console.scan(
        design,
        rotation=np.eye(3),
        centre_mm=(0.0, 0.0, 0.0),
        emit=messages.append,
        cancelled=lambda: bool(messages),
    )

    assert status == 1
    assert time.monotonic() - started < 10.0
    assert messages and all("preparing" in m for m in messages)


def test_a_scan_at_a_speed_reports_its_preparation_before_its_clock_starts(tmp_path):
    console = _console(tmp_path, speed=50.0)
    design = console.design("generate", "gre2d", _block(TE=5000, nx=32, ny=32))[
        "design"
    ]
    messages = []

    status = console.scan(
        design, rotation=np.eye(3), centre_mm=(0.0, 0.0, 0.0), emit=messages.append
    )

    assert status == 0
    preparing = [m for m in messages if "preparing" in m]
    assert preparing and messages[: len(preparing)] == preparing
    assert preparing[0] == {"preparing": None}


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
            client.send(json.dumps({"id": 2, "call": "coils"}))
            coils = json.loads(client.recv(timeout=DEADLINE))
            request = {
                "id": 3,
                "call": "exam",
                "subject": "vials",
                "coil": "body/head48",
            }
            client.send(json.dumps(request))
            exam = json.loads(client.recv(timeout=DEADLINE))
            examined_in = console.coil
            client.send(json.dumps({"id": 4, "call": "reboot"}))
            refused = json.loads(client.recv(timeout=DEADLINE))
            block = _block(TE=5000, nx=32, ny=32)
            client.send(json.dumps({"id": 5, "call": "exam", "coil": "body"}))
            client.recv(timeout=DEADLINE)
            request = {"id": 6, "call": "generate", "plugin": "gre2d", "block": block}
            client.send(json.dumps(request))
            design = json.loads(client.recv(timeout=DEADLINE))["design"]
            # A cancel that arrives before a scan does not stop it.
            client.send(json.dumps({"call": "cancel"}))
            client.send(json.dumps({"id": 7, "call": "scan", "design": design}))
            scanned = [json.loads(client.recv(timeout=DEADLINE))]
            while "done" not in scanned[-1]:
                scanned.append(json.loads(client.recv(timeout=DEADLINE)))
    finally:
        loop.call_soon_threadsafe(server["it"].close)
        thread.join(timeout=DEADLINE)

    assert plugins["id"] == 1
    assert "gre2d" in plugins["plugins"]
    assert coils == {"id": 2, "coils": console.coils()}
    assert exam["id"] == 3
    assert len(exam["localizer"]) == 3
    assert examined_in is virtual.COILS["body/head48"]
    assert refused == {"id": 4, "error": "unknown call 'reboot'"}
    assert {message["id"] for message in scanned} == {7}
    assert "clock" in scanned[0]
    assert scanned[-1]["done"] == 0


def test_a_console_serves_the_pages_of_its_origins_and_clients_that_send_none(
    tmp_path,
):
    from websockets.exceptions import InvalidStatus
    from websockets.sync.client import connect

    from pulserver.virtual._console import serve

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    loop = asyncio.new_event_loop()
    task = loop.create_task(
        serve(_console(tmp_path), "127.0.0.1", port, ["https://pulserver.github.io"])
    )

    def run():
        with contextlib.suppress(asyncio.CancelledError):
            loop.run_until_complete(task)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    address = f"ws://127.0.0.1:{port}"

    def answers(**options):
        deadline = time.monotonic() + DEADLINE
        while True:
            try:
                with connect(address, open_timeout=DEADLINE, **options) as client:
                    client.send(json.dumps({"id": 1, "call": "plugins"}))
                    return "plugins" in json.loads(client.recv(timeout=DEADLINE))
            except InvalidStatus as refused:
                return refused.response.status_code
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.1)

    try:
        assert answers(origin="https://pulserver.github.io") is True
        assert answers() is True
        assert answers(origin="https://elsewhere.example") == 403
    finally:
        loop.call_soon_threadsafe(task.cancel)
        thread.join(timeout=DEADLINE)


def test_the_console_command_serves_a_console_of_its_options(tmp_path, monkeypatch):
    from pulserver import _cli
    from pulserver.virtual import _console

    limits = tmp_path / "limits.txt"
    limits.write_text(format_limits(ANY_ORIENTATION))
    served = {}

    async def serve(console, host, port, origins):
        served.update(console=console, host=host, port=port, origins=origins)

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
            "--max-isochromats",
            "500000",
            "--coil",
            "head8/head32",
            "--origin",
            "https://pulserver.github.io",
            "--spins",
            "8",
            "--voxel",
            "box",
            "--diffusion",
            "--nod",
            "2",
            "4",
        ]
    )

    assert status == 0
    assert (served["host"], served["port"]) == ("127.0.0.1", 9876)
    assert served["origins"] == ["https://pulserver.github.io"]
    console = served["console"]
    assert console.recon == ("recon.local", 9020)
    assert console.local is None
    assert console.spacing == pytest.approx(2e-3)
    assert console.max_isochromats == 500_000
    assert console.coil is virtual.COILS["head8/head32"]
    assert console.field_t == ANY_ORIENTATION["B0"]
    assert (console.spins, console.voxel, console.diffusion) == (8, "box", True)
    turned = console.motion(1.0, np.array([[0.0, 0.1, 0.0]]))
    angle = np.radians(2.0)
    np.testing.assert_allclose(
        turned, [[0.0, 0.1 * np.cos(angle), 0.1 * np.sin(angle)]]
    )


def test_the_console_command_scans_in_the_coils_of_the_field_maps_it_names(
    tmp_path, monkeypatch, fields
):
    from pulserver import _cli
    from pulserver.virtual import _console

    limits = tmp_path / "limits.txt"
    limits.write_text(format_limits(ANY_ORIENTATION))
    served = {}

    async def serve(console, host, port, origins):
        served["console"] = console

    monkeypatch.setattr(_console, "serve", serve)
    options = ["--plugins", str(PLUGINS), "--limits", str(limits)]
    options += ["--store", str(tmp_path / "designs"), "--coil", "head8/head32"]

    status = _cli.main(["console", *options, "--fields", str(fields)])

    assert status == 0
    console = served["console"]
    assert console.fields == fields
    assert console.max_isochromats == 2_000_000
    assert console.coil.transmit_model == fields / "head8.npz"
    assert console.coil.receive_model == fields / "head32.npz"


def test_the_console_command_reconstructs_in_process_with_recon_plugins(
    tmp_path, monkeypatch
):
    from pulserver import _cli
    from pulserver.virtual import _console

    limits = tmp_path / "limits.txt"
    limits.write_text(format_limits(ANY_ORIENTATION))
    served = {}

    async def serve(console, host, port, origins):
        served["console"] = console

    monkeypatch.setattr(_console, "serve", serve)
    options = [
        "console",
        "--plugins",
        str(tmp_path / "sequences"),
        "--plugins",
        str(PLUGINS),
        "--limits",
        str(limits),
        "--store",
        str(tmp_path / "designs"),
        "--recon-plugins",
        str(tmp_path / "recon"),
        "--recon-plugins",
        str(RECON_PLUGINS),
    ]

    assert _cli.main(options) == 0
    console = served["console"]
    assert console.recon is None
    assert console.plugins == (tmp_path / "sequences", PLUGINS)
    assert console.local.plugins == (tmp_path / "recon", RECON_PLUGINS)
    with pytest.raises(SystemExit):
        _cli.main([*options, "--recon", "recon.local:9020"])


def test_a_console_lists_every_directorys_plugins_and_designs_from_the_first_holding_one(
    tmp_path,
):
    own = tmp_path / "own"
    own.mkdir()
    (own / "gre2d.py").write_text((PLUGINS / "tiny.py").read_text())
    (own / "alias.py").symlink_to(PLUGINS / "gre2d.py")
    shipped = _console(tmp_path)
    console = Console(
        plugins=[own, PLUGINS],
        limits=format_limits(ANY_ORIENTATION),
        store=tmp_path / "designs",
        spacing=2e-3,
    )

    generated = console.design("generate", "alias", _block(TE=5000, nx=32, ny=32))

    assert console.plugin_names() == sorted({"alias", *shipped.plugin_names()})
    assert console.design("list", "gre2d") == shipped.design("list", "tiny")
    assert generated["status"] == 0
