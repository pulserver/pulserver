import contextlib
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pypulseqpp as pp
import pytest
from _host import FIXTURE_LIMITS, GE_IR, LIMITS, PLUGINS, Daemon

from pulserver.host._blocks import format_import, parse_import
from pulserver.host.client import HostError
from pulserver.protocol import TEPreset

FIXTURES = Path(__file__).parent / "fixtures" / "sequences"


@pytest.fixture
def daemon(tmp_path):
    running = Daemon(tmp_path / "base")
    running.start()
    yield running
    running.cleanup()


def _session_dir(daemon, client):
    return daemon.base / "bucket" / str(client.session)


def test_repeated_predownloads_generate_one_revision(daemon):
    client = daemon.client(pid=101)
    client.open("tiny", LIMITS)
    assert client.list_protocol()["TE"].value == 8000
    assert client.validate({"TE": TEPreset.MINIMUM}).values["TE"] == 2500
    revisions = [client.generate({"TE": TEPreset.MINIMUM}) for _ in range(3)]
    assert revisions == [1, 1, 1]
    assert client.generate({"TE": 10000}) == 2
    directory = _session_dir(daemon, client)
    assert (directory / "current").readlink().as_posix() == "rev/2"
    assert sorted(p.name for p in (directory / "rev" / "1").iterdir()) == [
        "meta.json",
        "resolved.protocol",
        "sequence.pseg",
        "sequence.seq",
    ]
    assert "TE: 2500" in (directory / "rev" / "1" / "resolved.protocol").read_text()


def test_an_edited_plugin_generates_a_new_revision_for_the_same_protocol(tmp_path):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    plugin = plugins / "tiny.py"
    shutil.copyfile(PLUGINS / "tiny.py", plugin)
    daemon = Daemon(tmp_path / "base", plugins)
    daemon.start()
    try:
        client = daemon.client(pid=151)
        client.open("tiny", LIMITS)
        assert client.generate({"TE": 8000}) == 1
        # Two delays per repetition: the resolved protocol stays the same.
        edited = plugin.read_text().replace(
            "self.seq.add_block(pp.make_delay(self.te))",
            "self.seq.add_block(pp.make_delay(self.te))\n"
            "        self.seq.add_block(pp.make_delay(self.te))",
        )
        plugin.write_text(edited)
        stat = plugin.stat()
        os.utime(plugin, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        assert client.generate({"TE": 8000}) == 2
        revisions = _session_dir(daemon, client) / "rev"
        first = pp.Sequence()
        first.read(revisions / "1" / "sequence.seq")
        second = pp.Sequence()
        second.read(revisions / "2" / "sequence.seq")
        assert second.num_blocks == 2 * first.num_blocks
    finally:
        daemon.cleanup()


def test_two_sessions_interleave_without_sharing_state(daemon):
    first, second = daemon.client(pid=201), daemon.client(pid=202)
    first.open("tiny", LIMITS)
    second.open("tiny", LIMITS)
    assert first.generate({"TE": 8000}) == 1
    assert second.validate({"TE": 12000}).values["TE"] == 12000
    assert second.generate({"TE": 12000}) == 1
    assert first.generate({"TE": 8000}) == 1
    first_protocol = _session_dir(daemon, first) / "current" / "resolved.protocol"
    second_protocol = _session_dir(daemon, second) / "current" / "resolved.protocol"
    assert "TE: 8000" in first_protocol.read_text()
    assert "TE: 12000" in second_protocol.read_text()


def test_a_crashing_plugin_fails_only_its_command(daemon):
    crashing, healthy = daemon.client(pid=301), daemon.client(pid=302)
    crashing.open("crash", LIMITS)
    healthy.open("tiny", LIMITS)
    with pytest.raises(HostError, match="worker exited"):
        crashing.validate({"TE": 8000})
    assert healthy.validate({"TE": 8000}).valid


def test_a_terminated_daemon_exits_without_waiting_for_a_running_design(
    tmp_path, monkeypatch
):
    marker = tmp_path / "started"
    monkeypatch.setenv("PULSERVER_STALL_MARKER", str(marker))
    daemon = Daemon(tmp_path / "base")
    daemon.start()
    try:
        client = daemon.client(pid=351)
        client.open("stall", LIMITS)

        def validate():
            with contextlib.suppress(OSError, HostError):
                client.validate({"TE": 8000})

        threading.Thread(target=validate, daemon=True).start()
        deadline = time.monotonic() + 30
        while not marker.exists():
            assert time.monotonic() < deadline, "the design call never started"
            time.sleep(0.05)
        started = time.monotonic()
        daemon.stop()
        assert time.monotonic() - started < 10
    finally:
        daemon.cleanup()


def test_a_restarted_daemon_serves_an_open_session(daemon):
    client = daemon.client(pid=401)
    client.open("tiny", LIMITS)
    assert client.generate({"TE": 9000}) == 1
    daemon.stop()
    daemon.start()
    assert client.validate({"TE": 9000}).valid
    assert client.generate({"TE": 9000}) == 1


def test_an_invalid_protocol_generates_nothing(daemon):
    client = daemon.client(pid=501)
    client.open("tiny", LIMITS)
    with pytest.raises(HostError, match="shorter than"):
        client.generate({"TE": 1000})
    assert not (_session_dir(daemon, client) / "rev").exists()


def test_a_command_for_a_session_never_opened_is_an_error(daemon):
    with pytest.raises(HostError, match="not open"):
        daemon.client(pid=601).list_protocol()


def test_a_generated_revision_carries_its_cache(daemon):
    client = daemon.client(pid=701)
    client.open("tiny", {**LIMITS, **GE_IR})
    assert client.generate({"TE": 8000}) == 1
    revision = _session_dir(daemon, client) / "rev" / "1"
    assert sorted(p.name for p in revision.iterdir()) == [
        "meta.json",
        "resolved.protocol",
        "sequence.pge",
        "sequence.seq",
    ]
    vendor = struct.unpack("<6i", (revision / "sequence.pge").read_bytes()[:24])[4]
    assert vendor == 2


def test_a_generated_sequence_is_written_in_the_binary_form(daemon):
    client = daemon.client(pid=702)
    client.open("tiny", LIMITS)
    assert client.generate({"TE": 8000}) == 1
    written = _session_dir(daemon, client) / "rev" / "1" / "sequence.seq"
    assert b"[BLOCKS]" not in written.read_bytes()
    sequence = pp.Sequence()
    sequence.read(written, verify=True)
    assert sequence.num_blocks


def test_an_imported_chain_is_staged_and_converted(daemon):
    client = daemon.client(pid=801)
    client.open(None, FIXTURE_LIMITS)
    assert client.import_sequence(FIXTURES / "dedup_gre_pair.seq") == 1
    current = _session_dir(daemon, client) / "current"
    assert sorted(p.name for p in current.iterdir()) == [
        "dedup_gre_pair.seq",
        "dedup_gre_pair_b.seq",
        "meta.json",
        "sequence.pseg",
        "sequence.seq",
    ]
    assert (current / "sequence.seq").readlink().as_posix() == "dedup_gre_pair.seq"


def test_importing_the_same_file_reuses_its_revision(daemon):
    client = daemon.client(pid=802)
    client.open(None, FIXTURE_LIMITS)
    first = client.import_sequence(FIXTURES / "dedup_gre_pair.seq")
    assert client.import_sequence(FIXTURES / "dedup_gre_pair.seq") == first == 1


def test_a_session_without_a_plugin_refuses_design_commands(daemon):
    client = daemon.client(pid=803)
    client.open(None, FIXTURE_LIMITS)
    with pytest.raises(HostError, match="no plugin"):
        client.list_protocol()


def test_a_generated_revision_names_the_reconstruction_its_sequence_binds(daemon):
    bound, unbound = daemon.client(pid=901), daemon.client(pid=902)
    bound.open("gre2d", LIMITS)
    unbound.open("gre2d_raw", LIMITS)
    values = {"nx": 32, "ny": 16, "TE": 5000}
    assert _meta(daemon, bound, bound.generate(values))["recon"] == "gre2d"
    assert _meta(daemon, unbound, unbound.generate(values))["recon"] == ""


def test_a_prescribed_offset_is_a_revision_of_its_own(daemon):
    client = daemon.client(pid=903)
    client.open("gre2d", LIMITS)
    values = {"nx": 32, "ny": 32, "TE": 5000}
    assert client.generate(values) == 1
    assert client.generate({**values, "fov_offset_y": 20.0}) == 2
    assert client.generate({**values, "fov_offset_y": 0.0}) == 1
    revisions = _session_dir(daemon, client) / "rev"
    designed, moved = ((revisions / r / "sequence.pseg").read_bytes() for r in "12")
    assert designed != moved
    assert (revisions / "1" / "sequence.seq").read_bytes() == (
        revisions / "2" / "sequence.seq"
    ).read_bytes()


def test_an_import_block_carries_the_offset_it_was_given():
    moved = format_import("a.seq", (1.5, -2.0, 0.25))
    assert parse_import(moved) == (Path("a.seq"), (1.5, -2.0, 0.25))
    assert parse_import(format_import("a.seq")) == (Path("a.seq"), (0.0, 0.0, 0.0))


def test_an_import_at_another_offset_is_a_revision_of_its_own(daemon):
    client = daemon.client(pid=804)
    client.open(None, FIXTURE_LIMITS)
    assert client.import_sequence(FIXTURES / "dedup_gre_pair.seq") == 1
    moved = client.import_sequence(FIXTURES / "dedup_gre_pair.seq", (0.0, 0.0, 5.0))
    assert moved == 2
    assert _meta(daemon, client, moved)["fov_offset_mm"] == [0.0, 0.0, 5.0]


def _meta(daemon, client, revision):
    directory = daemon.base / "bucket" / str(client.session) / "rev" / str(revision)
    return json.loads((directory / "meta.json").read_text())


def test_a_client_process_does_not_import_the_design_engine():
    """A command sent from a shell is not charged the design engine's import."""
    probe = (
        "import sys\n"
        "from pulserver.host import HostClient, SessionKey\n"
        "print(sorted(m for m in ('pypulseq', 'pypulseqpp') if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"
