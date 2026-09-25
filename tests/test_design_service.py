"""The stateless design calls and the store of the designs they write."""

import hashlib
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
from _host import ANY_ORIENTATION, FIXTURE_LIMITS, GE_IR, LIMITS, PLUGINS
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

COUNTED = '''"""A delay per repetition; every construction appends to a file."""

import os

import pypulseqpp as pp
from pypulseqpp import sequences

from pulserver.design import ScannerSequence, TimeParam
from pulserver.protocol import TEPreset, UIParam


class CountedApp(sequences.SequenceApp):
    MAX_GRAD = 40.0
    MAX_SLEW = 150.0

    def init_sequence(self, te: float | None = 8e-3) -> None:
        with open(os.environ["PULSERVER_CONSTRUCTIONS"], "a") as log:
            log.write("constructed\\n")
        self.te = 2.5e-3 if te is None else te
        self.resolve(te=self.te)

    def loop(self) -> None:
        self.kernel()

    def kernel(self) -> None:
        self.seq.add_block(pp.make_delay(self.te))


class Counted(ScannerSequence):
    app = CountedApp
    ui = {
        UIParam.TE: TimeParam(
            "te", range_min=1000, range_max=80000, presets={TEPreset.MINIMUM: None}
        )
    }
'''


def block(values):
    lines = [f"{name}: {value}" for name, value in values.items()]
    return "\n".join([PROTOCOL_BEGIN, *lines, PROTOCOL_END]) + "\n"


@pytest.fixture
def store(tmp_path):
    return DesignStore(tmp_path / "designs")


def generate(store, plugin, values, limits=LIMITS, plugins=PLUGINS):
    return service.call(
        "generate",
        plugins=plugins,
        plugin=plugin,
        limits=limits,
        block=block(values),
        store=store,
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


def test_a_request_resolving_to_itself_constructs_its_application_once(
    tmp_path, store, monkeypatch
):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "counted.py").write_text(COUNTED)
    log = tmp_path / "constructions"
    monkeypatch.setenv("PULSERVER_CONSTRUCTIONS", str(log))

    def constructions(values):
        log.write_text("")
        generated(generate(store, "counted", values, plugins=plugins))
        return len(log.read_text().splitlines())

    assert constructions({"TE": 5000}) == 1
    # A preset resolves to a time, and the design is made from the time.
    assert constructions({"TE": TEPreset.MINIMUM}) == 2


def test_a_manifest_records_what_the_design_depends_on(store):
    design = generated(generate(store, "gre2d", GRE))
    manifest = store.manifest(design)
    assert manifest["id"] == design == design_id(manifest["identity"])
    assert (manifest["plugin"], manifest["recon"]) == ("gre2d", "gre2d")
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


def test_a_design_names_the_reconstruction_its_sequence_binds(store):
    bound = generated(generate(store, "gre2d", {"nx": 32, "ny": 16, "TE": 5000}))
    unbound = generated(generate(store, "gre2d_raw", {"nx": 32, "ny": 16, "TE": 5000}))
    assert store.manifest(bound)["recon"] == "gre2d"
    assert store.manifest(unbound)["recon"] == ""


def test_an_edited_plugin_is_another_design_of_the_same_protocol(tmp_path, store):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    plugin = plugins / "tiny.py"
    shutil.copyfile(PLUGINS / "tiny.py", plugin)
    first = generated(generate(store, "tiny", {"TE": 8000}, plugins=plugins))
    plugin.write_text(
        plugin.read_text().replace(
            "self.seq.add_block(pp.make_delay(self.te))",
            "self.seq.add_block(pp.make_delay(self.te))\n"
            "        self.seq.add_block(pp.make_delay(self.te))",
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


def test_a_design_carries_its_cache_tagged_with_the_vendor(store):
    design = generated(
        generate(store, "tiny", {"TE": 8000}, limits={**LIMITS, **GE_IR})
    )
    directory = store.directory(design)
    assert (directory / "sequence.pge").is_file()
    vendor = struct.unpack("<6i", (directory / "sequence.pge").read_bytes()[:24])[4]
    assert vendor == 2


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
    np.savez(vops, vops=np.ones((1, 1, 1), dtype=complex))
    checked = vops.read_bytes()
    limits = {**LIMITS, "vop_file": str(vops)}
    first = generated(generate(store, "tiny", {}, limits=limits))
    np.savez(vops, vops=2 * np.ones((1, 1, 1), dtype=complex))
    assert generated(generate(store, "tiny", {}, limits=limits)) != first
    vops.write_bytes(checked)
    assert generated(generate(store, "tiny", {}, limits=limits)) == first


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
    for age, design in zip((3, 2, 1), designs, strict=True):
        os.utime(store.directory(design) / "manifest.json", (now - age * 60,) * 2)
    budget = sum(p.stat().st_size for p in store.directory(designs[2]).iterdir())
    assert store.prune(max_bytes=budget, now=now) == designs[:2]
    assert list(store) == [designs[2]]


def test_pruning_removes_a_stage_a_process_left(store):
    stage = store.stage()
    now = time.time()
    os.utime(stage, (now - 7200,) * 2)
    store.prune(now=now)
    assert not stage.exists()


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
    assert (pruned.returncode, pruned.stdout) == (0, "PRUNED 1\n")


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
        f"ERROR no plugin 'absent' in {PLUGINS}\n",
    )


def test_the_top_level_command_names_its_commands():
    bare = subprocess.run(
        [sys.executable, "-m", "pulserver._cli"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bare.returncode == 2
    assert "pulserver design" in bare.stderr
