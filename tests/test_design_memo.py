"""What a warm design server keeps for the calls after a call: replies, the RF a plugin lists and the slew rate of a request."""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from _host import LIMITS

from pulserver.host import DesignStore, _memo, _slew
from pulserver.host import _service as service
from pulserver.host._blocks import format_limits
from pulserver.protocol import PROTOCOL_BEGIN, PROTOCOL_END

#: A nerve model under which a 36 mT/m step ramped at the design's full slew
#: rate stimulates past the limit, and ramped at about a third of it does not.
SITE = {
    **LIMITS,
    "design_max_grad": LIMITS["max_grad"],
    "design_max_slew": LIMITS["max_slew"],
    "pns_chronaxie": 360e-6,
    "pns_rheobase": 38.0,
    "pns_limit": 1.0,
}

#: Bipolar lobes ramped as fast as the design's slew rate allows, behind one
#: RF pulse. Each evaluation appends the slew rate it is made under to the
#: file ``PULSERVER_EVALUATIONS`` names. The representative sequence plays
#: its lobes at ``{representative}`` of the amplitude the design plays.
PLUGIN = '''"""Bipolar lobes ramped at the design's slew rate, whose evaluations are counted."""

import os

import pypulseqpp as pp

from pulserver.design import Evaluation, IntParam, RfLayout, SequencePlugin
from pulserver.protocol import UIParam


def lobes(system, nx=8, scale=1.0):
    seq = pp.Sequence(system)
    seq.add_block(pp.make_block_pulse(0.5, duration=1e-3, system=system, use="excitation"))
    lobe = pp.make_trapezoid(
        "x", amplitude=0.9 * scale * system.max_grad, flat_time=1e-3, system=system
    )
    for _ in range(nx // 2):
        seq.add_block(lobe)
        seq.add_block(pp.scale_grad(lobe, -1.0))
    seq.add_block(pp.make_delay(5e-3))
    return seq


class Lobes(SequencePlugin):
    app = lobes
    protocol = {{UIParam.NX: IntParam("nx", range_max=100)}}

    def evaluate(self, system, protocol):
        with open(os.environ["PULSERVER_EVALUATIONS"], "a") as log:
            log.write(f"{{protocol[UIParam.NX]}} {{system.max_slew / system.gamma:.3f}}\\n")
        one = lobes(system, **protocol.arguments, scale={representative})
        layout = RfLayout(one.rf_instances(), (None,), one.duration()[0])
        return Evaluation(protocol, one.duration()[0], rf_layout=layout, sequence=one)
'''


def block(nx=8):
    return f"{PROTOCOL_BEGIN}\nnx: {nx}\n{PROTOCOL_END}\n"


@pytest.fixture
def plugins(tmp_path, monkeypatch):
    folder = tmp_path / "plugins"
    folder.mkdir()
    (folder / "bound.py").write_text(PLUGIN.format(representative=1.0))
    (folder / "milder.py").write_text(PLUGIN.format(representative=0.7))
    _log(folder).touch()
    monkeypatch.setenv("PULSERVER_EVALUATIONS", str(_log(folder)))
    return folder


def _log(plugins):
    """The file the evaluations of the plugins in ``plugins`` are counted in."""
    return plugins.parent / "evaluations"


def _evaluations(plugins):
    """The size and slew rate in T/m/s of each evaluation made so far."""
    lines = _log(plugins).read_text().splitlines()
    return [(int(nx), float(slew)) for nx, slew in map(str.split, lines)]


def _child():
    """Forget what a call holds in memory, as the child of the next call has."""
    service._LOWERED = _slew.Remembered()
    service._RESPONSES = _slew.Remembered(kind="nerve")
    service._default_validation.cache_clear()
    service._cached.cache_clear()


@pytest.fixture
def serving():
    """The entries of a warm server, kept for calls made here as its children make them."""
    _child()
    root = _memo.start()
    yield root
    _memo.stop()
    _child()


def _call(name, plugins, plugin="bound", **inputs):
    _child()
    status, reply = service.call(
        name, plugins=plugins, plugin=plugin, limits=SITE, **inputs
    )
    assert status == 0, reply
    return reply


def test_nothing_is_kept_without_a_server():
    assert not _memo.kept()
    _memo.write("reply", "key", "value")
    assert _memo.read("reply", "key") == (False, None)


def test_an_entry_is_read_back_by_its_key_until_the_server_stops():
    root = _memo.start()
    try:
        assert _memo.kept()
        assert _memo.read("reply", "key") == (False, None)
        _memo.write("reply", "key", {"reply": "VALID", "slew": [True, None]})
        assert _memo.read("reply", "key") == (
            True,
            {"reply": "VALID", "slew": [True, None]},
        )
        assert _memo.read("reply", "another key") == (False, None)
        assert _memo.read("slew", "key") == (False, None)
        _memo.write("reply", "key", 2.5)
        assert _memo.read("reply", "key") == (True, 2.5)
    finally:
        _memo.stop()
    assert not root.exists()
    assert _memo.read("reply", "key") == (False, None)


def test_an_entry_half_written_or_of_another_key_is_not_found():
    _memo.start()
    try:
        _memo.write("reply", "key", 1)
        path = _memo._path("reply", "key")
        path.write_text(path.read_text()[:-3])
        assert _memo.read("reply", "key") == (False, None)
        path.write_text(json.dumps({"key": "another", "value": 1}))
        assert _memo.read("reply", "key") == (False, None)
    finally:
        _memo.stop()


def test_the_least_recently_written_entries_are_removed_past_the_size(monkeypatch):
    root = _memo.start()
    try:
        monkeypatch.setattr(_memo, "SIZE", 4)
        for number in range(12):
            _memo.write("reply", f"key {number}", number)
            os.utime(_memo._path("reply", f"key {number}"), (number, number))
        _memo._trim(root / "reply")
        kept = [n for n in range(12) if _memo.read("reply", f"key {n}")[0]]
        assert kept == [8, 9, 10, 11]
    finally:
        _memo.stop()


def test_a_validation_is_evaluated_each_time_without_a_server(plugins):
    first = _call("validate", plugins, block=block())
    made = len(_evaluations(plugins))
    assert _call("validate", plugins, block=block()) == first
    assert len(_evaluations(plugins)) == 2 * made


def test_a_kept_validation_is_replied_without_evaluating_again(plugins, serving):
    _call("list", plugins, rf_definitions=True)
    listed = len(_evaluations(plugins))
    first = _call("validate", plugins, block=block(), rf_layout=True)
    made = _evaluations(plugins)
    assert first.startswith("VALID")
    assert "[RfLayout]" in first
    # The request was lowered: evaluated at the design limit, then below it.
    assert len(made) > listed + 1
    assert made[-1][1] < made[listed][1]

    assert _call("validate", plugins, block=block(), rf_layout=True) == first
    assert _evaluations(plugins) == made
    # Another request is evaluated once, where the search of the last one
    # puts its readings on target: at or above the slew rate it ended at.
    other = _call("validate", plugins, block=block(10), rf_layout=True)
    assert other.startswith("VALID")
    ((size, slew),) = _evaluations(plugins)[len(made) :]
    assert size == 10
    assert made[-1][1] <= slew < made[listed][1]
    # The reply without the RF layout is another reply.
    plain = _call("validate", plugins, block=block())
    assert "[RfLayout]" not in plain
    assert len(_evaluations(plugins)) == len(made) + 2


def test_the_rf_a_plugin_lists_is_evaluated_once_for_every_call(plugins, serving):
    listed = _call("list", plugins, rf_definitions=True)
    assert "[RfDefinitions]" in listed
    # The default protocol, evaluated once at the design limits.
    default = _evaluations(plugins)
    assert len(default) == 1

    assert _call("list", plugins, rf_definitions=True) == listed
    _call("validate", plugins, block=block(4), rf_layout=True)
    _call("validate", plugins, block=block(6), rf_layout=True)
    sizes = [nx for nx, _ in _evaluations(plugins)[1:]]
    assert sizes and set(sizes) == {4, 6}


def test_an_edited_plugin_is_listed_and_validated_again(plugins, serving):
    listed = _call("list", plugins, rf_definitions=True)
    first = _call("validate", plugins, block=block())
    made = len(_evaluations(plugins))
    path = plugins / "bound.py"
    path.write_text(path.read_text().replace("range_max=100", "range_max=64"))
    os.utime(path, ns=(time.time_ns(), path.stat().st_mtime_ns + 1_000_000))

    assert _call("list", plugins, rf_definitions=True) != listed
    assert _call("validate", plugins, block=block()) == first
    assert len(_evaluations(plugins)) > made


def test_a_plugin_relinked_with_the_same_modification_time_is_listed_again(
    plugins, serving
):
    # An installer gives every plugin file it extracts the same modification
    # time; a slot linked to another of them holds another plugin.
    other = plugins / "other.py"
    other.write_text(
        PLUGIN.format(representative=1.0).replace("range_max=100", "range_max=64")
    )
    shipped = (plugins / "bound.py").stat().st_mtime_ns
    os.utime(other, ns=(shipped, shipped))
    slot = plugins / "sequence5.py"
    slot.symlink_to(plugins / "bound.py")
    listed = _call("list", plugins, plugin="sequence5", rf_definitions=True)

    slot.unlink()
    slot.symlink_to(other)

    relisted = _call("list", plugins, plugin="sequence5", rf_definitions=True)
    assert relisted != listed
    assert relisted == _call("list", plugins, plugin="other", rf_definitions=True)


def test_a_plugin_staged_where_another_was_starts_at_the_design_limit(plugins, serving):
    # A plugin's next request starts at the slew rate its last one ended at;
    # another plugin copied to the same path is not that plugin.
    slot = plugins / "sequence30.py"
    slot.write_text((plugins / "bound.py").read_text())
    _call("validate", plugins, plugin="sequence30", block=block())
    limit = _evaluations(plugins)[0][1]
    assert _evaluations(plugins)[-1][1] < 0.9 * limit
    made = len(_evaluations(plugins))

    slot.write_text((plugins / "milder.py").read_text())
    _call("validate", plugins, plugin="sequence30", block=block(10))

    (size, first), *_ = _evaluations(plugins)[made:]
    assert size == 10
    assert first == pytest.approx(limit, rel=1e-3)


def test_generate_starts_at_the_slew_rate_another_call_validated(
    plugins, serving, tmp_path
):
    _call("validate", plugins, block=block())
    validated = _evaluations(plugins)[-1][1]
    store = DesignStore(tmp_path / "designs", "scan")

    reply = _call("generate", plugins, block=block(), store=store)

    directory = store.directory(reply.split()[1])
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["slew_passes"] == 0
    assert manifest["design_max_slew_t_per_m_per_s"] == pytest.approx(
        validated, rel=1e-3
    )


def test_generate_searches_a_request_no_validation_searched(plugins, serving, tmp_path):
    store = DesignStore(tmp_path / "designs", "scan")

    reply = _call("generate", plugins, block=block(), store=store)

    directory = store.directory(reply.split()[1])
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["slew_passes"] == 0
    assert manifest["design_max_slew_t_per_m_per_s"] < 0.9 * LIMITS["max_slew"]
    # The validation of the request is then evaluated once, there.
    made = len(_evaluations(plugins))
    assert _call("validate", plugins, block=block()).startswith("VALID")
    assert _evaluations(plugins)[made:] == [
        (8, pytest.approx(manifest["design_max_slew_t_per_m_per_s"], rel=1e-3))
    ]


def test_a_design_lowered_further_drops_the_kept_validation(plugins, serving, tmp_path):
    """The representative sequence of ``milder`` asks for less than its design."""
    first = _call("validate", plugins, "milder", block=block())
    validated = _evaluations(plugins)[-1][1]
    store = DesignStore(tmp_path / "designs", "scan")
    reply = _call("generate", plugins, "milder", block=block(), store=store)
    directory = store.directory(reply.split()[1])
    manifest = json.loads((directory / "manifest.json").read_text())
    designed = manifest["design_max_slew_t_per_m_per_s"]
    assert manifest["slew_passes"] >= 1
    assert designed < validated

    made = len(_evaluations(plugins))
    again = _call("validate", plugins, "milder", block=block())
    # Evaluated once more, at the slew rate the design ended at, and kept.
    assert _evaluations(plugins)[made:] == [(8, pytest.approx(designed, rel=1e-3))]
    # Slower ramps: the scan time the console shows is the design's.
    assert float(again.split()[1]) > float(first.split()[1])
    assert _call("validate", plugins, "milder", block=block()) == again
    assert len(_evaluations(plugins)) == made + 1
    # The plugin's next request is tried where the design ended.
    _call("validate", plugins, "milder", block=block(10))
    assert _evaluations(plugins)[made + 1] == (10, pytest.approx(designed, rel=1e-3))


EXAM_PLUGIN = '''"""A delay as long as the exam's cache holds files, in ms."""

import pypulseqpp as pp

from pulserver.design import Evaluation, SequencePlugin


def delay(system):
    seq = pp.Sequence(system)
    seq.add_block(pp.make_delay(1e-3))
    return seq


class Examined(SequencePlugin):
    app = delay

    def evaluate(self, system, protocol, exam=None):
        files = 0 if exam is None else len(list(exam.glob("*")))
        return Evaluation(protocol, 1e-3 * files)
'''


def test_a_kept_validation_of_a_plugin_reading_the_exam_follows_its_files(
    tmp_path, serving
):
    folder = tmp_path / "plugins"
    folder.mkdir()
    (folder / "examined.py").write_text(EXAM_PLUGIN)
    exam = tmp_path / "exam"
    exam.mkdir()
    (exam / "b0_map.npy").write_bytes(b"1")
    empty = f"{PROTOCOL_BEGIN}\n{PROTOCOL_END}\n"

    def validate(directory):
        return _call("validate", folder, "examined", block=empty, exam=directory)

    assert validate(str(exam)).startswith("VALID 0.001")
    assert validate(str(exam)).startswith("VALID 0.001")
    (exam / "b1_map.npy").write_bytes(b"2")
    assert validate(str(exam)).startswith("VALID 0.002")
    assert validate(None).startswith("VALID ?")


def _ask(socket_path, request):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(socket_path))
        connection.sendall((json.dumps(request) + "\n").encode())
        with connection.makefile("rb") as reader:
            reply = json.loads(reader.readline())
    assert reply["status"] == 0, reply["output"]
    return reply["output"]


def test_the_children_of_a_warm_server_share_what_they_keep(plugins, tmp_path):
    directory = Path(tempfile.mkdtemp(prefix="ps"))
    socket_path = directory / "s"
    log = tmp_path / "server.log"
    with log.open("w") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pulserver._cli",
                "design",
                "serve",
                "--plugins",
                str(plugins),
                "--socket",
                str(socket_path),
            ],
            env={**os.environ, "PULSERVER_EVALUATIONS": str(_log(plugins))},
            stderr=stream,
        )
    try:
        deadline = time.monotonic() + 60.0
        while not socket_path.exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        common = {
            "plugins": [str(plugins)],
            "plugin": "bound",
            "limits": format_limits(SITE),
        }
        validation = {**common, "call": "validate", "input": block(), "rf_layout": True}
        listing = {**common, "call": "list", "rf_definitions": True}
        listed = _ask(socket_path, listing)
        first = _ask(socket_path, validation)
        made = _evaluations(plugins)
        assert _ask(socket_path, validation) == first
        assert _ask(socket_path, listing) == listed
        assert _evaluations(plugins) == made
        generated = _ask(
            socket_path,
            {
                **common,
                "call": "generate",
                "input": block(),
                "store": str(tmp_path / "designs"),
                "scan": "scan",
            },
        )
        store = DesignStore(tmp_path / "designs", "scan")
        manifest = json.loads(
            (store.directory(generated.split()[1]) / "manifest.json").read_text()
        )
        assert manifest["slew_passes"] == 0
        assert manifest["design_max_slew_t_per_m_per_s"] == pytest.approx(
            made[-1][1], rel=1e-3
        )
        kept = [
            line.split(" in ")[-1].strip()
            for line in log.read_text().splitlines()
            if "keeping what the calls share in" in line
        ]
        assert len(kept) == 1 and Path(kept[0]).is_dir()
    finally:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=60.0)
    assert not Path(kept[0]).exists()
    assert "of processor time" in log.read_text()


def test_a_pulse_design_is_kept_for_the_calls_after_the_one_that_made_it(
    serving, monkeypatch
):
    import numpy as np
    from pypulseqpp import _slr

    from pulserver.host import _server

    made = []
    design = _slr._design

    def counted(*arguments):
        made.append(arguments)
        return design(*arguments)

    # pypulseqpp caches the design per process; a child starts without it.
    monkeypatch.setattr(_slr, "_design", counted)
    _server.keep_pulse_designs()
    arguments = (128, 4.0, "ex", "ls", 0.01, 0.01, True, False)
    first = _slr._design(*arguments)
    again = _slr._design(*arguments)
    assert len(made) == 1
    assert again.dtype == first.dtype
    assert np.array_equal(again, first)
    again[0] += 1.0
    assert np.array_equal(_slr._design(*arguments), first)
    # Other arguments are another design.
    other = _slr._design(128, 6.0, "ex", "ls", 0.01, 0.01, True, False)
    assert len(made) == 2
    assert not np.array_equal(other, first)
    # Kept once: a second call wraps nothing.
    kept = _slr._design
    _server.keep_pulse_designs()
    assert _slr._design is kept


def test_a_pulse_designed_through_pypulseqpp_is_the_same_kept_or_not(
    serving, monkeypatch
):
    """Through pypulseqpp's own entry to the design, however it passes its arguments on."""
    import numpy as np
    from pypulseqpp import _slr

    from pulserver.host import _server

    def designed():
        return _slr.design_slr(256, 4.0, pulse_type="ex")

    plain = designed()
    made = []
    design = _slr._design

    def counted(*arguments, **named):
        made.append((arguments, named))
        return design(*arguments, **named)

    monkeypatch.setattr(_slr, "_design", counted)
    _server.keep_pulse_designs()
    first = designed()
    again = designed()
    assert len(made) == 1
    assert np.array_equal(first, plain)
    assert np.array_equal(again, plain)
    # Named arguments reach the design, and are another key than none.
    named = _slr._design(*made[0][0][:-1], root_flip=made[0][0][-1])
    assert np.array_equal(named, plain)
    assert len(made) == 2


def test_pypulseqpp_caches_the_design_the_server_keeps():
    """The server keeps what ``pypulseqpp._slr._design`` returns: the name and the cache are pypulseqpp's."""
    from pypulseqpp import _slr

    assert callable(getattr(_slr, "_design", None))
    assert hasattr(_slr._design, "cache_info")
