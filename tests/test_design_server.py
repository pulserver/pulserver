"""The warm design server: forwarded calls, each answered in a child of its own."""

import os
import signal
import subprocess
import sys
import threading
import time

import pytest
from _host import LIMITS, PLUGINS

from pulserver.host._blocks import format_limits
from pulserver.protocol import PROTOCOL_BEGIN, PROTOCOL_END

DEADLINE = 60.0


def block(values):
    lines = [f"{name}: {value}" for name, value in values.items()]
    return "\n".join([PROTOCOL_BEGIN, *lines, PROTOCOL_END]) + "\n"


def command(*args, stdin="", env=None):
    return subprocess.run(
        [sys.executable, "-m", "pulserver._cli", "design", *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=DEADLINE,
    )


@pytest.fixture
def limits_file(tmp_path):
    path = tmp_path / "limits"
    path.write_text(format_limits(LIMITS))
    return path


@pytest.fixture
def server(tmp_path_factory):
    """A warm server over the test plugins, whose stalling plugin marks a file."""
    directory = tmp_path_factory.mktemp("server")
    socket_path = directory / "design.sock"
    marker = directory / "stalled"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pulserver._cli",
            "design",
            "serve",
            "--plugins",
            str(PLUGINS),
            "--socket",
            str(socket_path),
        ],
        env={**os.environ, "PULSERVER_STALL_MARKER": str(marker)},
    )
    deadline = time.monotonic() + DEADLINE
    while not socket_path.exists():
        if process.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError("the design server did not start")
        time.sleep(0.05)
    process.socket = socket_path
    process.marker = marker
    yield process
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=DEADLINE)


def calls(limits_file, store=None):
    common = ["--plugins", str(PLUGINS), "--plugin", "gre2d"]
    with_limits = [*common, "--limits", str(limits_file)]
    stored = [*with_limits, "--store", str(store)] if store is not None else None
    return common, with_limits, stored


def test_a_forwarded_call_replies_as_the_call_answered_in_its_own_process(
    server, limits_file, tmp_path
):
    common, with_limits, stored = calls(limits_file, tmp_path / "designs")
    request = block({"nx": 32, "ny": 32, "TE": 5000})
    forwarded = ["--socket", str(server.socket)]
    for args, stdin in (
        (["list", *common], ""),
        (["validate", *with_limits], request),
        (["generate", *stored], request),
    ):
        direct = command(*args, stdin=stdin)
        through = command(*args, *forwarded, stdin=stdin)
        assert (through.returncode, through.stdout) == (
            direct.returncode,
            direct.stdout,
        )
        assert direct.returncode == 0


def test_a_crashing_plugin_fails_only_its_call(server, limits_file):
    forwarded = ["--limits", str(limits_file), "--socket", str(server.socket)]
    crashed = command(
        "validate",
        "--plugins",
        str(PLUGINS),
        "--plugin",
        "crash",
        *forwarded,
        stdin=block({"TE": 8000}),
    )
    assert (crashed.returncode, crashed.stdout) == (
        1,
        "ERROR the design call ended with exit status 1\n",
    )
    healthy = command(
        "validate",
        "--plugins",
        str(PLUGINS),
        "--plugin",
        "tiny",
        *forwarded,
        stdin=block({"TE": 8000}),
    )
    assert healthy.stdout.startswith("VALID ")


def _stall(server, limits_file):
    """Start a call that does not return; return its thread once it has started."""
    stalled = {}

    def call():
        stalled["reply"] = command(
            "validate",
            "--plugins",
            str(PLUGINS),
            "--plugin",
            "stall",
            "--limits",
            str(limits_file),
            "--socket",
            str(server.socket),
            stdin=block({"TE": 8000}),
        )

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    deadline = time.monotonic() + DEADLINE
    while not server.marker.exists():
        assert time.monotonic() < deadline, "the stalling call never started"
        time.sleep(0.05)
    return thread, stalled


def test_a_call_is_answered_while_another_runs(server, limits_file):
    thread, _ = _stall(server, limits_file)
    answered = command(
        "list",
        "--plugins",
        str(PLUGINS),
        "--plugin",
        "tiny",
        "--socket",
        str(server.socket),
    )
    assert answered.stdout.startswith("PROTOCOL\n")
    assert thread.is_alive()


def test_a_stopped_server_ends_its_running_calls(server, limits_file):
    thread, stalled = _stall(server, limits_file)
    started = time.monotonic()
    server.send_signal(signal.SIGTERM)
    assert server.wait(timeout=DEADLINE) == 0
    assert time.monotonic() - started < 15
    thread.join(timeout=DEADLINE)
    assert stalled["reply"].returncode == 1
    assert stalled["reply"].stdout.startswith("ERROR ")
    assert not server.socket.exists()


def test_a_call_naming_a_socket_no_server_listens_on_is_answered_in_its_process(
    tmp_path,
):
    listed = command(
        "list",
        "--plugins",
        str(PLUGINS),
        "--plugin",
        "tiny",
        "--socket",
        str(tmp_path / "absent.sock"),
    )
    assert listed.returncode == 0
    assert listed.stdout.startswith("PROTOCOL\n")


PROBE = """
import sys
from pulserver._cli import main
status = main(sys.argv[1:])
loaded = sorted(m for m in ("numpy", "pypulseq", "pypulseqpp") if m in sys.modules)
print(status, loaded, file=sys.stderr)
"""


@pytest.mark.parametrize("named_by", ["option", "environment"])
def test_a_forwarded_call_loads_no_design_engine(server, named_by):
    args = ["design", "list", "--plugins", str(PLUGINS), "--plugin", "tiny"]
    env = dict(os.environ)
    if named_by == "option":
        args += ["--socket", str(server.socket)]
    else:
        env["PULSERVER_DESIGN_SOCKET"] = str(server.socket)
    probe = subprocess.run(
        [sys.executable, "-c", PROBE, *args],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        timeout=DEADLINE,
    )
    assert probe.stdout.startswith("PROTOCOL\n")
    assert probe.stderr.strip().splitlines()[-1] == "0 []"
