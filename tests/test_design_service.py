"""The stateless design calls and the store of the designs they write."""

import hashlib
import logging
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest
from _host import ANY_ORIENTATION, FIXTURE_LIMITS, LIMITS, PLUGINS, VENDOR_IR
from _virtual import OBLIQUE

from pulserver.host import DesignStore, design_id, design_identity
from pulserver.host import _service as service
from pulserver.host._blocks import format_import, format_limits
from pulserver.protocol import (
    FOV_ROTATION,
    PROTOCOL_BEGIN,
    PROTOCOL_END,
    TEPreset,
    parse_listing,
    parse_validation,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
GRE = {"nx": 32, "ny": 32, "TE": 5000}

COUNTED = '''"""A function app that appends the TE it is called with to a file; its evaluation states the shortest TE without calling it."""

import os

import pypulseqpp as pp

from pulserver.design import Evaluation, SequencePlugin, TimeParam
from pulserver.protocol import TEPreset, UIParam

SHORTEST_TE = 2.5e-3


def counted(system, te: float | None = 8e-3):
    with open(os.environ["PULSERVER_CALLS"], "a") as log:
        log.write(f"{te!r}\\n")
    te = SHORTEST_TE if te is None else te
    seq = pp.Sequence(system)
    seq.add_block(pp.make_delay(te))
    return seq


class Counted(SequencePlugin):
    app = counted
    protocol = {
        UIParam.TE: TimeParam(
            "te", range_min=1000, range_max=80000, presets={TEPreset.MINIMUM: None}
        )
    }

    def evaluate(self, system, protocol):
        te = protocol[UIParam.TE]
        shortest = SHORTEST_TE if te is None else te
        return Evaluation(protocol.replace({UIParam.TE: shortest}))
'''

EVALUATING = '''"""A function app; its evaluation rejects 80 lines, fails on 13 and states a scan time for 20."""

import pypulseqpp as pp

from pulserver.design import Evaluation, IntParam, SequencePlugin
from pulserver.protocol import UIParam


def delays(system, nx=8):
    seq = pp.Sequence(system)
    for _ in range(nx):
        seq.add_block(pp.make_delay(1e-3))
    return seq


class Evaluating(SequencePlugin):
    app = delays
    protocol = {UIParam.NX: IntParam("nx", range_max=100)}

    def evaluate(self, system, protocol):
        lines = protocol[UIParam.NX]
        if lines == 80:
            raise ValueError("80 lines do not fit the receiver buffer")
        if lines == 13:
            raise TypeError("a defect in the evaluation")
        if lines == 20:
            return Evaluation(protocol, 20e-3, "a note shown with the protocol")
        return super().evaluate(system, protocol)
'''

CHAINED = '''"""A function app returning a calibration and the scan: a chain of two sequences."""

import pypulseqpp as pp

from pulserver.design import IntParam, SequencePlugin
from pulserver.protocol import UIParam


def chain(system, nx=8):
    calibration, scan = pp.Sequence(system), pp.Sequence(system)
    calibration.add_block(pp.make_delay(1e-3))
    for _ in range(nx):
        scan.add_block(pp.make_delay(2e-3))
    return [calibration, scan]


class Chained(SequencePlugin):
    app = chain
    protocol = {UIParam.NX: IntParam("nx", range_max=100)}
'''

HELPER = """import pypulseqpp as pp


def delays(system, nx=8):
    seq = pp.Sequence(system)
    for _ in range(nx):
        seq.add_block(pp.make_delay(1e-3))
    return seq
"""

PARTIAL = '''"""A function app that is a partial of a function in another file."""

import functools
import importlib.util
from pathlib import Path

from pulserver.design import IntParam, SequencePlugin
from pulserver.protocol import UIParam

spec = importlib.util.spec_from_file_location(
    "delay_helper", Path(__file__).with_name("delay_helper.py")
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class Partial(SequencePlugin):
    app = functools.partial(helper.delays, nx=4)
    protocol = {UIParam.NX: IntParam("nx", range_max=100)}
'''


EXAM_READING = '''"""A function app whose evaluation and design read the B0 map of the exam."""

import pypulseqpp as pp

from pulserver.design import Evaluation, IntParam, SequencePlugin, load_exam
from pulserver.protocol import UIParam


def delays(system, nx=8):
    seq = pp.Sequence(system)
    for _ in range(nx):
        seq.add_block(pp.make_delay(1e-3))
    return seq


class ExamReading(SequencePlugin):
    app = delays
    protocol = {UIParam.NX: IntParam("nx", range_max=100)}

    def evaluate(self, system, protocol, exam=None):
        note = "no exam" if exam is None else f"b0 {load_exam(exam)['b0_map']}"
        return Evaluation(protocol, 0.0, note)

    def generate(self, system, protocol, exam=None):
        nx = protocol.arguments["nx"]
        if exam is not None:
            nx += load_exam(exam)["b0_map"]
        return delays(system, nx)
'''


def block(values):
    lines = [f"{name}: {value}" for name, value in values.items()]
    return "\n".join([PROTOCOL_BEGIN, *lines, PROTOCOL_END]) + "\n"


@pytest.fixture
def store(tmp_path):
    return DesignStore(tmp_path / "designs")


@pytest.fixture
def function_plugins(tmp_path):
    """A plugin directory holding the function-app plugins ``evaluating`` and ``chained``."""
    plugins = tmp_path / "function_plugins"
    plugins.mkdir()
    (plugins / "evaluating.py").write_text(EVALUATING)
    (plugins / "chained.py").write_text(CHAINED)
    return plugins


def generate(store, plugin, values, limits=LIMITS, plugins=PLUGINS):
    return service.call(
        "generate",
        plugins=plugins,
        plugin=plugin,
        limits=limits,
        block=block(values),
        store=store,
    )


def validate(plugins, plugin, values, limits=LIMITS):
    return service.call(
        "validate",
        plugins=plugins,
        plugin=plugin,
        limits=limits,
        block=block(values),
    )


def imported(store, path, offset_mm=None, rotation=None, limits=FIXTURE_LIMITS):
    return service.call(
        "import",
        limits=limits,
        block=format_import(path, offset_mm, rotation),
        store=store,
    )


def generated(reply):
    status, text = reply
    assert status == 0, text
    word, design = text.split()
    assert word in ("GENERATED", "IMPORTED")
    return design


def test_one_resolved_protocol_is_one_design(store):
    designs = [
        generated(generate(store, "tiny", {"TE": value}))
        for value in (TEPreset.MINIMUM, TEPreset.MINIMUM, 2500)
    ]
    assert len(set(designs)) == 1
    assert generated(generate(store, "tiny", {"TE": 10000})) != designs[0]
    directory = store.directory(designs[0])
    assert sorted(p.name for p in directory.iterdir()) == [
        "manifest.json",
        "resolved.protocol",
        "sequence.pseg",
        "sequence.seq",
    ]
    assert "TE: 2500" in (directory / "resolved.protocol").read_text()


def test_a_request_is_generated_once_whether_or_not_it_resolves_to_itself(
    tmp_path, store, monkeypatch
):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "counted.py").write_text(COUNTED)
    log = tmp_path / "calls"
    monkeypatch.setenv("PULSERVER_CALLS", str(log))

    def calls(values):
        log.write_text("")
        generated(generate(store, "counted", values, plugins=plugins))
        return log.read_text().split()

    assert calls({"TE": 5000}) == ["0.005"]
    # A preset resolves to a time, and the design is made from the time.
    assert calls({"TE": TEPreset.MINIMUM}) == ["0.0025"]


def test_a_manifest_records_what_the_design_depends_on(store):
    design = generated(generate(store, "gre2d", GRE))
    manifest = store.manifest(design)
    assert manifest["id"] == design == design_id(manifest["identity"])
    assert manifest["plugin"] == "gre2d"
    assert manifest["limits"] == LIMITS
    assert manifest["versions"]["pypulseqpp"] == pp.__version__
    assert manifest["scan_time"] > 0
    directory = store.directory(design)
    for name, digest in manifest["files"].items():
        assert digest == hashlib.sha256((directory / name).read_bytes()).hexdigest()
    assert sorted(manifest["files"]) == [
        "resolved.protocol",
        "sequence.pseg",
        "sequence.seq",
    ]


def test_a_stored_manifest_names_no_reconstruction(store):
    designed = generated(generate(store, "gre2d", {"nx": 32, "ny": 16, "TE": 5000}))
    copied = generated(imported(store, FIXTURES / "dedup_gre_pair.seq"))
    assert "recon" not in store.manifest(designed)
    assert "recon" not in store.manifest(copied)


def test_a_plugin_declaring_recon_is_warned_and_ignored(tmp_path, store):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "declared.py").write_text(
        (PLUGINS / "tiny.py")
        .read_text()
        .replace("    app = tiny\n", "    app = tiny\n    recon = 'crash'\n")
    )
    with pytest.warns(DeprecationWarning, match="recon"):
        design = generated(generate(store, "declared", {"TE": 5000}, plugins=plugins))
    assert "recon" not in store.manifest(design)


def test_an_edited_plugin_is_another_design_of_the_same_protocol(tmp_path, store):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    plugin = plugins / "tiny.py"
    shutil.copyfile(PLUGINS / "tiny.py", plugin)
    first = generated(generate(store, "tiny", {"TE": 8000}, plugins=plugins))
    plugin.write_text(
        plugin.read_text().replace(
            "seq.add_block(pp.make_delay(te))",
            "seq.add_block(pp.make_delay(te))\n        seq.add_block(pp.make_delay(te))",
        )
    )
    stat = plugin.stat()
    os.utime(plugin, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    second = generated(generate(store, "tiny", {"TE": 8000}, plugins=plugins))
    assert second != first
    written = []
    for design in (first, second):
        sequence = pp.Sequence()
        sequence.read(store.directory(design) / "sequence.seq")
        written.append(sequence.num_blocks)
    assert written[1] == 2 * written[0]


def test_an_invalid_protocol_is_an_error_and_stores_nothing(store):
    status, text = generate(store, "tiny", {"TE": 1000})
    assert status == 1
    assert text.startswith("ERROR ") and "shorter than" in text
    assert not list(store)
    assert not any(store.root.iterdir())


@pytest.mark.parametrize(
    ("nx", "level", "info", "logged"),
    [
        pytest.param(
            80,
            logging.WARNING,
            "80 lines do not fit the receiver buffer",
            "ValueError: 80 lines do not fit the receiver buffer",
            id="expected-rejection",
        ),
        pytest.param(
            13,
            logging.ERROR,
            "TypeError in Evaluating.evaluate",
            "TypeError: a defect in the evaluation",
            id="unexpected-error",
        ),
    ],
)
def test_an_evaluate_error_is_an_invalid_protocol(
    function_plugins, caplog, nx, level, info, logged
):
    with caplog.at_level(logging.WARNING, logger="pulserver.design"):
        status, reply = validate(function_plugins, "evaluating", {"nx": nx})

    assert status == 0
    assert reply.startswith(f"INVALID\nINFO {info}\n")
    assert f"nx: {nx}\n" in reply
    [record] = caplog.records
    assert (record.name, record.levelno) == ("pulserver.design", level)
    assert logged in caplog.text


def test_a_zero_duration_is_valid_and_sent_as_unknown(function_plugins, store):
    status, reply = validate(function_plugins, "evaluating", {"nx": 8})
    assert (status, reply.split("\n")[:2]) == (0, ["VALID ?", "INFO "])

    design = generated(
        generate(store, "evaluating", {"nx": 8}, plugins=function_plugins)
    )
    assert store.manifest(design)["scan_time"] is None


def test_a_valid_reply_carries_the_duration_and_note_of_the_evaluation(
    function_plugins,
):
    status, reply = validate(function_plugins, "evaluating", {"nx": 20})
    assert status == 0
    assert reply.startswith("VALID 0.02\nINFO a note shown with the protocol\n")


def test_a_function_app_is_designed_into_a_stored_design(function_plugins, store):
    def designed(nx):
        return generated(
            generate(store, "evaluating", {"nx": nx}, plugins=function_plugins)
        )

    design = designed(8)
    assert designed(8) == design
    assert designed(9) != design
    written = store.directory(design) / "sequence.seq"
    assert b"[BLOCKS]" not in written.read_bytes()
    assert pp.io.read(written).duration()[0] == pytest.approx(8e-3)


def test_a_chain_returned_by_a_function_app_is_one_stored_design(
    function_plugins, store
):
    design = generated(generate(store, "chained", {"nx": 4}, plugins=function_plugins))
    directory = store.directory(design)
    assert sorted(store.manifest(design)["files"]) == [
        "resolved.protocol",
        "sequence.pseg",
        "sequence.seq",
        "sequence_main.seq",
    ]
    first = pp.io.read(directory / "sequence.seq")
    assert first.get_definition("NextSequence") == "sequence_main.seq"
    scan = pp.io.read(directory / "sequence_main.seq")
    assert scan.duration()[0] == pytest.approx(4 * 2e-3)


def test_a_partial_app_is_covered_by_the_source_file_of_its_function(tmp_path, store):
    plugins = tmp_path / "partial_plugins"
    plugins.mkdir()
    helper, plugin = plugins / "delay_helper.py", plugins / "partial.py"
    helper.write_text(HELPER)
    plugin.write_text(PARTIAL)
    first = generated(generate(store, "partial", {}, plugins=plugins))

    helper.write_text(HELPER.replace("1e-3", "0.002"))
    stat = plugin.stat()
    os.utime(plugin, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    second = generated(generate(store, "partial", {}, plugins=plugins))

    assert second != first
    durations = [
        pp.io.read(store.directory(design) / "sequence.seq").duration()[0]
        for design in (first, second)
    ]
    assert durations == [pytest.approx(4e-3), pytest.approx(8e-3)]


def test_a_design_carries_its_cache_tagged_with_the_vendor(store):
    design = generated(
        generate(store, "tiny", {"TE": 8000}, limits={**LIMITS, **VENDOR_IR})
    )
    directory = store.directory(design)
    assert (directory / "sequence.cache").is_file()
    vendor = struct.unpack("<6i", (directory / "sequence.cache").read_bytes()[:24])[4]
    assert vendor == 5


def test_a_design_is_written_in_the_binary_form(store):
    design = generated(generate(store, "tiny", {"TE": 8000}))
    written = store.directory(design) / "sequence.seq"
    assert b"[BLOCKS]" not in written.read_bytes()
    sequence = pp.Sequence()
    sequence.read(written, verify=True)
    assert sequence.num_blocks


def test_a_prescribed_offset_is_a_design_of_its_own(store):
    designed = generated(generate(store, "gre2d", GRE))
    moved = generated(generate(store, "gre2d", {**GRE, "fov_offset_y": 20.0}))
    assert moved != designed
    assert generated(generate(store, "gre2d", {**GRE, "fov_offset_y": 0.0})) == designed
    one, other = (store.directory(d) for d in (designed, moved))
    assert (one / "sequence.seq").read_bytes() == (other / "sequence.seq").read_bytes()
    assert (one / "sequence.pseg").read_bytes() != (
        other / "sequence.pseg"
    ).read_bytes()


def test_a_design_beyond_the_scanner_limits_is_refused_and_stores_nothing(store):
    assert service.call(
        "validate", plugins=PLUGINS, plugin="strong", limits=LIMITS, block=block({})
    )[1].startswith("VALID")
    status, text = generate(store, "strong", {})
    assert status == 1
    assert "gradient amplitude of 60.0 mT/m on x" in text
    assert not any(store.root.iterdir())


def test_an_oblique_design_is_held_under_the_design_limits_and_checked_against_the_scanners(
    store,
):
    oblique = {**GRE, **dict(zip(FOV_ROTATION, OBLIQUE.ravel(), strict=True))}
    derated = {
        "max_grad": ANY_ORIENTATION["design_max_grad"],
        "max_slew": ANY_ORIENTATION["design_max_slew"],
    }
    for limits in (LIMITS, {**LIMITS, **derated}):
        status, text = generate(store, "gre2d", oblique, limits=limits)
        assert status == 1
        assert "slew rate of" in text
    design = generated(generate(store, "gre2d", oblique, limits=ANY_ORIENTATION))
    assert store.manifest(design)["limits"] == ANY_ORIENTATION


def test_a_request_is_validated_under_the_design_limits(store):
    listed = service.call("list", plugins=PLUGINS, plugin="gre2d")[1]
    listing = parse_listing(listed.split("\n", 1)[1])
    request = block({**GRE, "TE": TEPreset.MINIMUM})
    shortest = {}
    for name, limits in (("scanner", LIMITS), ("design", ANY_ORIENTATION)):
        reply = service.call(
            "validate", plugins=PLUGINS, plugin="gre2d", limits=limits, block=request
        )[1]
        shortest[name] = parse_validation(reply, listing).values["TE"]
    assert shortest["design"] > shortest["scanner"]
    generated(
        generate(
            store, "gre2d", {**GRE, "TE": shortest["design"]}, limits=ANY_ORIENTATION
        )
    )


def test_a_design_in_a_forbidden_band_is_refused(store):
    limits = {**LIMITS, "forbidden_band_1": "all 1 5000 0.001"}
    status, text = generate(store, "gre2d", GRE, limits=limits)
    assert status == 1
    assert "of the forbidden band 1-5000 Hz" in text


def test_limits_that_cannot_be_read_are_refused(store):
    limits = {**LIMITS, "forbidden_band_1": "w 590 650"}
    for call in ("validate", "generate"):
        inputs = {"plugins": PLUGINS, "plugin": "tiny", "limits": limits}
        inputs["block"] = block({"TE": 8000})
        if call == "generate":
            inputs["store"] = store
        status, text = service.call(call, **inputs)
        assert (status, text.split()[0]) == (1, "ERROR")
        assert "forbidden band" in text


def test_a_design_is_one_of_the_vop_file_contents_its_sar_ratios_came_from(
    tmp_path, store
):
    vops = tmp_path / "vops.npz"
    one = np.ones((1, 1, 1), dtype=complex)
    np.savez(vops, vops=one, global_matrix=one)
    checked = vops.read_bytes()
    limits = {
        **LIMITS,
        "vop_file": str(vops),
        "vop_head_limit": 3.2,
        "vop_local_limit": 10.0,
    }
    first = generated(generate(store, "tiny", {}, limits=limits))
    np.savez(vops, vops=2 * one, global_matrix=one)
    assert generated(generate(store, "tiny", {}, limits=limits)) != first
    vops.write_bytes(checked)
    assert generated(generate(store, "tiny", {}, limits=limits)) == first


def test_a_design_is_converted_under_the_vendor_file_its_limits_name(tmp_path, store):
    vendor = tmp_path / "vendor.txt"
    vendor.write_text(
        "[Grouping]\nboundary_gradient_hz_per_m: 100\nsplit_by_pulses: true\n"
        "split_by_readouts: true\nsplit_navigators: true\n"
        "split_edge_delays: false\n[Grouping End]\n[VendorProfile]\n"
        + "".join(
            f"{name}: float32 0\n"
            for name in (
                "grad_sample",
                "grad_amplitude",
                "rf_sample",
                "rf_amplitude",
                "rf_phase",
                "rf_frequency",
            )
        )
        + "[VendorProfile End]\n"
    )
    limits = {**LIMITS, "ir_vendor_file": str(vendor)}
    first = generated(generate(store, "gre2d", GRE, limits=limits))
    plain = generated(generate(store, "gre2d", GRE))
    assert first != plain
    cache = store.directory(first) / "sequence.pseg"
    seq = tmp_path / "sequence.seq"
    shutil.copy(store.directory(plain) / "sequence.seq", seq)
    system = pp.Opts(**LIMITS)
    expected = service.ir.convert(
        seq, system, grouping=service.ir.Grouping(split_edge_delays=False)
    )
    assert cache.read_bytes() == expected.read_bytes()
    vendor.write_text(vendor.read_text().replace("100", "200"))
    assert generated(generate(store, "gre2d", GRE, limits=limits)) != first


def test_an_unreadable_vendor_file_is_refused_and_stores_nothing(tmp_path, store):
    vendor = tmp_path / "vendor.txt"
    vendor.write_text("[Grouping]\n[Grouping End]\n")
    limits = {**LIMITS, "ir_vendor_file": str(vendor)}
    status, reply = generate(store, "tiny", {}, limits=limits)
    assert status == 1
    assert reply.startswith("ERROR ")
    assert list(store) == []


def test_an_imported_chain_is_copied_checked_and_converted(store):
    design = generated(imported(store, FIXTURES / "dedup_gre_pair.seq"))
    directory = store.directory(design)
    assert sorted(p.name for p in directory.iterdir()) == [
        "dedup_gre_pair.seq",
        "dedup_gre_pair_b.seq",
        "manifest.json",
        "sequence.pseg",
        "sequence.seq",
    ]
    assert (directory / "sequence.seq").readlink().as_posix() == "dedup_gre_pair.seq"
    assert store.manifest(design)["plugin"] == ""


def test_importing_the_same_files_at_the_same_prescription_is_one_design(store):
    path = FIXTURES / "dedup_gre_pair.seq"
    first = generated(imported(store, path))
    assert generated(imported(store, path)) == first
    moved = generated(imported(store, path, (0.0, 0.0, 5.0)))
    quarter = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    turned = generated(imported(store, path, rotation=quarter))
    assert len({first, moved, turned}) == 3
    assert store.manifest(moved)["fov_offset_mm"] == [0.0, 0.0, 5.0]
    np.testing.assert_allclose(
        np.reshape(store.manifest(turned)["fov_rotation"], (3, 3)), quarter, atol=1e-12
    )


def test_an_import_beyond_the_scanner_limits_is_refused(store):
    limits = {**FIXTURE_LIMITS, "max_grad": 5.0}
    status, text = imported(store, FIXTURES / "dedup_gre_pair.seq", limits=limits)
    assert status == 1
    assert "gradient amplitude" in text
    assert not any(store.root.iterdir())


def test_a_design_identifier_is_three_integers_a_float32_holds_exactly():
    design = design_id(design_identity("tiny", LIMITS, {"TE": 8000}))
    assert len(design) == 18
    parts = [int(design[start : start + 6], 16) for start in (0, 6, 12)]
    assert all(int(np.float32(part)) == part for part in (*parts, 0xFFFFFF))


def test_a_design_identifier_holding_another_identity_is_refused(store):
    identity = design_identity("tiny", LIMITS, {"TE": 8000})
    staged = store.stage()
    (staged / "sequence.seq").write_bytes(b"designed")
    store.commit(identity, staged, {})
    colliding = identity[:18] + ("0" if identity[18] != "0" else "1") + identity[19:]
    with pytest.raises(ValueError, match="another design with the same identifier"):
        store.find(colliding)


def test_a_design_committed_twice_is_stored_once(store):
    identity = design_identity("tiny", LIMITS, {"TE": 8000})
    stages = [store.stage() for _ in range(2)]
    for stage in stages:
        (stage / "sequence.seq").write_bytes(b"designed")
    designs = [store.commit(identity, stage, {}) for stage in stages]
    assert designs[0] == designs[1] == design_id(identity)
    assert [p.name for p in store.root.iterdir()] == [designs[0]]


def _stored(store, values):
    identity = design_identity("tiny", LIMITS, values)
    staged = store.stage()
    (staged / "sequence.seq").write_bytes(b"x" * 1000)
    return store.commit(identity, staged, {}), identity


def test_pruning_removes_the_designs_unused_for_longest_first(store):
    old, _ = _stored(store, {"TE": 8000})
    new, _ = _stored(store, {"TE": 9000})
    now = time.time()
    os.utime(store.directory(old) / "manifest.json", (now - 10 * 86400,) * 2)
    assert store.prune(max_age=86400, now=now) == [old]
    assert list(store) == [new]


def test_a_design_found_again_is_used_again(store):
    design, identity = _stored(store, {"TE": 8000})
    other, _ = _stored(store, {"TE": 9000})
    now = time.time()
    for name in (design, other):
        os.utime(store.directory(name) / "manifest.json", (now - 10 * 86400,) * 2)
    assert store.find(identity) == design
    assert store.prune(max_age=86400) == [other]


def test_pruning_to_a_size_keeps_the_most_recently_used(store):
    designs = [_stored(store, {"TE": te})[0] for te in (8000, 9000, 10000)]
    now = time.time()
    for age, design in zip((4, 3, 2), designs, strict=True):
        os.utime(store.directory(design) / "manifest.json", (now - age * 86400,) * 2)
    budget = sum(p.stat().st_size for p in store.directory(designs[2]).iterdir())
    assert store.prune(max_bytes=budget, now=now) == designs[:2]
    assert list(store) == [designs[2]]


def test_pruning_removes_a_stage_a_process_left(store):
    stage = store.stage()
    now = time.time()
    os.utime(stage, (now - 7200,) * 2)
    store.prune(now=now)
    assert not stage.exists()


def test_pruning_keeps_a_stage_younger_than_an_hour(store):
    """A stage expires by its age, so no reused process id keeps or removes one."""
    stage = store.stage()
    store.prune(now=time.time() + 600)
    assert stage.is_dir()


def test_pruning_keeps_a_design_used_within_the_day_whatever_the_limits(store):
    """A cleaner cannot remove a design a scan may be playing or reconstructing."""
    design, _ = _stored(store, {"TE": 8000})
    assert store.prune(max_age=0, max_bytes=0) == []
    assert list(store) == [design]


def test_a_design_being_written_is_invisible_until_it_is_whole(store):
    """A reader on a shared filesystem sees no directory, or the whole design."""
    identity = design_identity("tiny", LIMITS, {"TE": 8000})
    staged = store.stage()
    (staged / "sequence.seq").write_bytes(b"designed")
    assert store.find(identity) is None
    assert list(store) == []
    assert not store.directory(design_id(identity)).exists()
    design = store.commit(identity, staged, {})
    assert store.find(identity) == design


def test_a_stored_design_holds_the_files_its_manifest_records_and_no_other(store):
    design = generated(generate(store, "gre2d", GRE))
    files = {p.name for p in store.directory(design).iterdir()}
    assert files == {"manifest.json", *store.manifest(design)["files"]}


def test_two_processes_designing_one_protocol_at_once_store_it_once(
    tmp_path, limits_file
):
    args = [
        "--plugins",
        str(PLUGINS),
        "--plugin",
        "tiny",
        "--limits",
        str(limits_file),
        "--store",
        str(tmp_path / "designs"),
    ]
    command = [sys.executable, "-m", "pulserver._cli", "design", "generate", *args]
    running = [
        subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    replies = [p.communicate(block({"TE": 2500})) for p in running]
    assert [p.returncode for p in running] == [0, 0], replies
    assert replies[0][0] == replies[1][0]
    design = replies[0][0].split()[1]
    assert [p.name for p in (tmp_path / "designs").iterdir()] == [design]


def _command(*args, stdin=""):
    return subprocess.run(
        [sys.executable, "-m", "pulserver._cli", "design", *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def limits_file(tmp_path):
    path = tmp_path / "limits"
    path.write_text(format_limits(LIMITS))
    return path


def test_the_command_replies_on_standard_output_with_its_exit_status(
    tmp_path, limits_file
):
    common = [
        "--plugins",
        str(PLUGINS),
        "--plugin",
        "tiny",
        "--limits",
        str(limits_file),
    ]
    valid = _command("validate", *common, stdin=block({"TE": TEPreset.MINIMUM}))
    assert valid.returncode == 0
    assert valid.stdout.startswith("VALID ")
    store = ["--store", str(tmp_path / "designs")]
    created = _command("generate", *common, *store, stdin=block({"TE": 2500}))
    assert created.returncode == 0
    assert created.stdout.startswith("GENERATED ")
    refused = _command("generate", *common, *store, stdin=block({"TE": 1000}))
    assert refused.returncode == 1
    assert refused.stdout.startswith("ERROR ")
    pruned = _command("prune", *store, "--max-bytes", "0")
    assert (pruned.returncode, pruned.stdout) == (0, "PRUNED 0\n")


def test_a_listing_needs_neither_limits_nor_a_store():
    listed = _command("list", "--plugins", str(PLUGINS), "--plugin", "tiny")
    assert listed.returncode == 0
    header, listing = listed.stdout.split("\n", 1)
    assert header == "PROTOCOL"
    assert parse_listing(listing)["TE"].value == 8000


def test_a_plugin_that_ends_its_process_ends_the_command_without_a_reply(
    limits_file,
):
    crashed = _command(
        "validate",
        "--plugins",
        str(PLUGINS),
        "--plugin",
        "crash",
        "--limits",
        str(limits_file),
        stdin=block({"TE": 8000}),
    )
    assert crashed.returncode != 0
    assert crashed.stdout == ""


def test_the_command_names_an_unknown_plugin(limits_file):
    unknown = _command("list", "--plugins", str(PLUGINS), "--plugin", "absent")
    assert (unknown.returncode, unknown.stdout) == (
        1,
        f"ERROR no plugin 'absent' in {PLUGINS} or among the shipped plugins\n",
    )


def test_the_command_takes_a_plugin_from_the_first_of_its_directories_holding_it(
    tmp_path,
):
    own = tmp_path / "own"
    own.mkdir()
    (own / "gre2d.py").write_text((PLUGINS / "tiny.py").read_text())
    searched = ("--plugins", str(own), "--plugins", str(PLUGINS))

    shadowed = _command("list", *searched, "--plugin", "gre2d")
    shipped = _command("list", *searched, "--plugin", "gre2d_raw")

    assert (
        shadowed.stdout
        == _command("list", "--plugins", str(PLUGINS), "--plugin", "tiny").stdout
    )
    assert (
        shipped.stdout
        == _command("list", "--plugins", str(PLUGINS), "--plugin", "gre2d_raw").stdout
    )
    assert shadowed.returncode == shipped.returncode == 0


def test_the_top_level_command_names_its_commands():
    bare = subprocess.run(
        [sys.executable, "-m", "pulserver._cli"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bare.returncode == 2
    assert "pulserver design" in bare.stderr


@pytest.fixture
def exam(tmp_path):
    """The directory of an exam whose cache holds a B0 map of 3."""
    from pulserver.recon import ExamCache

    directory = tmp_path / "exam"
    ExamCache("exam", directory)["b0_map"] = 3
    return directory


def test_a_hook_taking_exam_reads_the_exam_cache_it_is_given(tmp_path, exam):
    (tmp_path / "exam_reading.py").write_text(EXAM_READING)
    plain = service.call(
        "validate",
        plugins=tmp_path,
        plugin="exam_reading",
        limits=LIMITS,
        block=block({"nx": 4}),
    )
    read = service.call(
        "validate",
        plugins=tmp_path,
        plugin="exam_reading",
        limits=LIMITS,
        block=block({"nx": 4}),
        exam=str(exam),
    )
    assert "no exam" in plain[1]
    assert "b0 3" in read[1]


def test_a_hook_not_taking_exam_is_called_as_without_one(function_plugins, exam):
    plain = validate(function_plugins, "evaluating", {"nx": 20})
    given = service.call(
        "validate",
        plugins=function_plugins,
        plugin="evaluating",
        limits=LIMITS,
        block=block({"nx": 20}),
        exam=str(exam),
    )
    assert given == plain


def test_an_exam_is_part_of_a_design_only_for_a_plugin_reading_it(
    tmp_path, store, exam, function_plugins
):
    from pulserver.recon import ExamCache

    (tmp_path / "exam_reading.py").write_text(EXAM_READING)

    def designed(plugins, plugin, values, exam):
        return generated(
            service.call(
                "generate",
                plugins=plugins,
                plugin=plugin,
                limits=LIMITS,
                block=block(values),
                store=store,
                exam=exam,
            )
        )

    without = designed(tmp_path, "exam_reading", {"nx": 4}, None)
    first = designed(tmp_path, "exam_reading", {"nx": 4}, str(exam))
    assert first != without
    assert designed(tmp_path, "exam_reading", {"nx": 4}, str(exam)) == first
    ExamCache("exam", exam)["b0_map"] = 5
    assert designed(tmp_path, "exam_reading", {"nx": 4}, str(exam)) != first

    ignored = designed(function_plugins, "chained", {"nx": 4}, None)
    assert designed(function_plugins, "chained", {"nx": 4}, str(exam)) == ignored
