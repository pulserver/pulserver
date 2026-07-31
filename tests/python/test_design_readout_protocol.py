"""Collection and state protocol shared by class-based readouts."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pp = pytest.importorskip("pypulseq")

from pulserver.design import _readout as readout

OPTS_KW = {"max_grad": 40, "grad_unit": "mT/m", "max_slew": 150, "slew_unit": "T/m/s"}


def _opts():
    return pp.Opts(**OPTS_KW)


def test_collection_access_requires_state():
    module = readout.Line2D(_opts(), (0.22, 0.22), (32, 32), spoil_position="none")

    with pytest.raises(RuntimeError, match=r"set_state\(\)"):
        len(module)
    with pytest.raises(RuntimeError, match=r"set_state\(\)"):
        module.blocks[0]
    with pytest.raises(RuntimeError, match=r"set_state\(\)"):
        list(module.blocks)


def test_line_supports_len_index_slice_iteration_and_get():
    opts = _opts()
    module = readout.Line2D(opts, (0.22, 0.22), (32, 32), spoil_position="none")
    assert isinstance(module.set_state(lin_idx=7, phase_offset_rad=0.25), readout.Readout)

    blocks = tuple(module.blocks)
    assert len(module) == module.num_blocks == len(blocks) == 3
    assert module.blocks[0] is blocks[0]
    assert module.blocks[-1] is blocks[-1]
    assert module.blocks[:] == blocks
    assert all(isinstance(block, tuple) and block for block in blocks)

    seq = pp.Sequence(opts)
    for block in module.blocks:
        seq.add_block(*block)
    assert len(seq.block_events) == len(module)

    standalone = module.get()
    assert len(standalone.block_events) == len(module)


def test_set_state_retunes_the_block_snapshot_in_place():
    """A new state re-points the same events; it does not rebuild them.

    The snapshot is the module's blocks *for its current state*, and a readout
    keeps one layout across states on purpose: the structure is fixed by the
    design and only a few payload numbers move, so rebuilding the events would
    re-solve no gradient and re-register no shape. The consequence a caller can
    observe is that a snapshot held across ``set_state`` is a live view, not a
    copy -- read it for the state that produced it.
    """
    module = readout.Line2D(_opts(), (0.22, 0.22), (32, 32), spoil_position="none")
    module.set_state(lin_idx=3, phase_offset_rad=0.1)
    old_blocks = tuple(module.blocks)
    assert all(event.phase_offset == pytest.approx(0.1) for event in _adcs(old_blocks))

    module.set_state(lin_idx=4, phase_offset_rad=0.9)
    new_blocks = tuple(module.blocks)

    # Same events, new numbers -- including on the snapshot taken earlier.
    assert new_blocks[0] is old_blocks[0]
    adc_events = _adcs(new_blocks)
    assert adc_events and all(event.phase_offset == pytest.approx(0.9) for event in adc_events)
    assert all(event.phase_offset == pytest.approx(0.9) for event in _adcs(old_blocks))


def test_retuned_snapshot_registers_the_same_blocks_as_a_rebuild():
    """Retuning is only an optimisation: the emitted sequence cannot tell.

    Two shots played through one module, against the same two shots played
    through a module that has never seen another state.
    """
    from pulserver.pypulseq import Sequence as PulserverSequence

    fov, matrix = (0.22, 0.22), (32, 32)
    shared = readout.Line2D(_opts(), fov, matrix, spoil_position="none")

    retuned = PulserverSequence._unstructured(_opts())
    for lin_idx, phase in ((3, 0.1), (4, 0.9)):
        shared.set_state(lin_idx=lin_idx, phase_offset_rad=phase)
        for _block in shared.blocks:
            retuned.add_block(*_block)
    rebuilt = PulserverSequence._unstructured(_opts())
    for lin_idx, phase in ((3, 0.1), (4, 0.9)):
        fresh = readout.Line2D(_opts(), fov, matrix, spoil_position="none")
        fresh.set_state(lin_idx=lin_idx, phase_offset_rad=phase)
        for _block in fresh.blocks:
            rebuilt.add_block(*_block)
    assert retuned.block_events == rebuilt.block_events
    assert retuned.block_durations == rebuilt.block_durations


def test_a_structural_change_rebuilds_rather_than_retunes():
    """A rotation adds an event to every gradient block, so it cannot be a retune."""
    module = readout.Line2D(_opts(), (0.22, 0.22), (32, 32), spoil_position="none")
    plain = module.set_state(lin_idx=3).blocks
    rotated = module.set_state(lin_idx=3, rotation=Rotation.from_euler("z", 30, degrees=True)).blocks

    assert rotated[0] is not plain[0]
    assert sum(getattr(event, "type", None) == "rot3D" for block in rotated for event in block) > 0
    assert module.set_state(lin_idx=3).blocks[0] is not rotated[0]


def _adcs(blocks):
    return [event for block in blocks for event in block if getattr(event, "type", None) == "adc"]


def test_epi_and_fse_use_the_common_protocol():
    epi = readout.Epi2D(_opts(), (0.22, 0.22), (32, 32), 1, np.array([0, 1, 2]))
    epi.set_state(lin_idx=10)
    assert len(epi) == 4
    assert len(epi.get().block_events) == len(epi)

    opts = _opts()
    rf, gz = readout.build_refocusing_pulse(opts, thickness_m=5e-3)
    fse = readout.Fse2D(opts, (0.22, 0.22), (32, 32), 2, rf, gz)
    indices = np.array([10, 11])
    fse.set_state(lin_idx=indices, freq_offset_hz=25.0)
    indices[:] = 0  # set_state() snapshots caller-owned index schedules.
    assert len(fse) == 10
    assert len(fse.get().block_events) == len(fse)


def test_noncartesian_train_uses_the_common_protocol():
    arm = readout.Radial(_opts(), 0.22, 16)
    module = readout.NonCartesian2D(arm).set_state(lin_idx=2, phase_offset_rad=0.4)

    assert isinstance(module, readout.Readout)
    assert len(module) == 3
    assert len(module.get().block_events) == len(module)


def test_modules_are_not_callable():
    module = readout.Line2D(_opts(), (0.22, 0.22), (32, 32), spoil_position="none")

    assert not callable(module)
    with pytest.raises(TypeError):
        module(pe_idx=5)


def test_a_counter_merges_into_the_first_block_only():
    module = readout.Line2D(_opts(), (0.22, 0.22), (32, 32), spoil_position="none")
    module.set_state(lin_idx=7)
    plain = module.num_blocks

    assert module.set_state(SLC=3) is module
    assert module.num_blocks == plain
    slc = module.blocks[0][-1]
    assert (slc.type, slc.label, slc.value) == ("labelset", "SLC", 3)
    assert all(slc not in block for block in module.blocks[1:])


def test_a_counter_keeps_the_event_it_was_declared_with():
    """A counter is structure once named, and only its value moves after that.

    This is what lets a TR template recognise the block as its own: the module
    hands back the *same* label object every shot, so the block the template
    recorded stays the block the loop adds.
    """
    module = readout.Line2D(_opts(), (0.22, 0.22), (32, 32), spoil_position="none")
    module.set_state(lin_idx=7, SLC=3)
    declared = module.blocks[0][-1]

    module.set_state(lin_idx=8, SLC=4)
    assert module.blocks[0][-1] is declared
    assert declared.value == 4
    # ...and a shot that does not mention it leaves it where it was.
    module.set_state(lin_idx=9)
    assert module.blocks[0][-1] is declared
    assert declared.value == 4


def test_duration_is_the_snapshot_duration():
    opts = _opts()
    module = readout.Line2D(opts, (0.22, 0.22), (32, 32), spoil_position="none")
    module.set_state(lin_idx=7)

    # Line readouts publish an exact analytic duration; it must agree with the
    # blocks actually emitted.
    assert module.duration == pytest.approx(sum(pp.calc_duration(*block) for block in module.blocks))

    # A module that publishes nothing falls back to summing its snapshot.
    rf, _ = readout.build_refocusing_pulse(opts, thickness_m=5e-3)
    excitation = readout.Fse2D(opts, (0.22, 0.22), (32, 32), 2, rf, _).set_state(lin_idx=np.array([1, 2]))
    assert excitation.duration > 0.0


def _label_pairs(block, name):
    return [
        (event.type, event.value)
        for event in block
        if getattr(event, "type", "").startswith("label") and event.label == name
    ]


def test_counters_and_numbers_travel_in_one_call():
    module = readout.Line2D(_opts(), (0.22, 0.22), (32, 32), spoil_position="none")
    module.set_state(lin_idx=7, REP=2, SLC=3, PHS=1)

    assert _label_pairs(module.blocks[0], "REP") == [("labelset", 2)]
    assert _label_pairs(module.blocks[0], "SLC") == [("labelset", 3)]
    assert _label_pairs(module.blocks[0], "PHS") == [("labelset", 1)]


def test_flags_are_scoped_to_the_module_and_sticky_ones_left_open():
    module = readout.Line2D(_opts(), (0.22, 0.22), (32, 32), spoil_position="none")
    module.set_state(lin_idx=7)
    plain = module.num_blocks

    assert module.set_state(OFF=1, NAV=True, ONCE=1, MODULE=4) is module
    assert module.num_blocks == plain
    assert module.flags == {"OFF": 1, "NAV": 1, "ONCE": 1, "MODULE": 4}

    # Scoped flags open on the first block and reset on the last, so they
    # cannot leak into whatever the sequence plays next...
    for name in ("OFF", "NAV"):
        assert _label_pairs(module.blocks[0], name) == [("labelset", 1)]
        assert _label_pairs(module.blocks[-1], name) == [("labelset", 0)]
    # ...while ONCE (a whole prep section) and MODULE (a group id) stay set.
    assert _label_pairs(module.blocks[0], "ONCE") == [("labelset", 1)]
    assert _label_pairs(module.blocks[0], "MODULE") == [("labelset", 4)]
    assert _label_pairs(module.blocks[-1], "ONCE") == _label_pairs(module.blocks[-1], "MODULE") == []

    # Clearing a flag is setting it to zero: the event stays, because it is
    # part of the structure the template recorded.
    module.set_state(OFF=0)
    assert module.flags["OFF"] == 0
    assert _label_pairs(module.blocks[0], "OFF") == [("labelset", 0)]


def test_adc_flag_is_the_off_flag_spelled_the_way_a_loop_reads():
    module = readout.Line2D(_opts(), (0.22, 0.22), (32, 32), spoil_position="none")
    module.set_state(lin_idx=7, adc_flag=False)
    assert _label_pairs(module.blocks[0], "OFF") == [("labelset", 1)]
    assert _label_pairs(module.blocks[-1], "OFF") == [("labelset", 0)]

    module.set_state(lin_idx=7, adc_flag=True)
    assert _label_pairs(module.blocks[0], "OFF") == [("labelset", 0)]

    # None means "leave it as it is", so a loop can pass the same keyword
    # whether or not this shot has an opinion about it.
    module.set_state(lin_idx=7, adc_flag=None, once=None)
    assert _label_pairs(module.blocks[0], "OFF") == [("labelset", 0)]
    assert module.flags == {"OFF": 0}


def test_flag_scope_is_declared_with_the_flag():
    module = readout.Line2D(
        _opts(), (0.22, 0.22), (32, 32), spoil_position="none",
        flags=("NOROT",), flag_scope="sticky",
    )
    module.set_state(lin_idx=7, NOROT=1)
    assert _label_pairs(module.blocks[-1], "NOROT") == []

    scoped = readout.Line2D(
        _opts(), (0.22, 0.22), (32, 32), spoil_position="none",
        flags={"ONCE": 1}, flag_scope="module",
    )
    scoped.set_state(lin_idx=7)
    assert _label_pairs(scoped.blocks[-1], "ONCE") == [("labelset", 0)]

    with pytest.raises(ValueError, match="flag_scope"):
        readout.Line2D(
            _opts(), (0.22, 0.22), (32, 32), spoil_position="none",
            flags=("NOROT",), flag_scope="block",
        )


def test_flags_and_counters_are_independent_states():
    module = readout.Line2D(_opts(), (0.22, 0.22), (32, 32), spoil_position="none")
    module.set_state(lin_idx=7, OFF=1, SLC=2)

    # Moving a counter must not disturb a sticky flag.
    module.set_state(SLC=5)
    assert _label_pairs(module.blocks[0], "OFF") == [("labelset", 1)]
    assert _label_pairs(module.blocks[0], "SLC") == [("labelset", 5)]


def test_labels_are_emitted_in_the_order_they_were_declared():
    """The file a module writes follows its declarations, not an internal split.

    A flag and a counter are the same kind of event to Pulseq, and the order
    they appear in decides the bytes, so it must be the order the module was
    told about them -- not "flags first" or any other rule the caller cannot
    see.
    """
    module = readout.Line2D(_opts(), (0.22, 0.22), (32, 32), spoil_position="none")
    module.set_state(lin_idx=7, SLC=1, ONCE=0, REP=2)
    assert [
        event.label for event in module.blocks[0] if getattr(event, "type", "") == "labelset"
    ] == ["SLC", "ONCE", "REP"]


def test_triggers_are_declared_with_the_module():
    gate = pp.make_trigger("physio1", duration=100e-6)
    sync = pp.make_digital_output_pulse("osc0", duration=100e-6)
    plain = readout.Line2D(_opts(), (0.22, 0.22), (32, 32), spoil_position="none")
    plain.set_state(lin_idx=7)

    module = readout.Line2D(
        _opts(), (0.22, 0.22), (32, 32), spoil_position="none",
        triggers={0: (gate,), -1: (sync,)},
    )
    module.set_state(lin_idx=7)

    assert module.num_blocks == plain.num_blocks
    assert gate in module.blocks[0] and sync not in module.blocks[0]
    assert sync in module.blocks[-1] and gate not in module.blocks[-1]

    # A flat sequence of events arms the first block, which is where a gating
    # trigger belongs.
    first = readout.Line2D(
        _opts(), (0.22, 0.22), (32, 32), spoil_position="none", triggers=(gate,)
    )
    first.set_state(lin_idx=7)
    assert gate in first.blocks[0]


def test_payloads_report_the_dynamic_fields_of_every_block():
    """One dict per block, keyed the way a pulseq block is."""
    module = readout.Line2D(_opts(), (0.22, 0.22), (32, 32), spoil_position="none")
    payloads = module.set_state(lin_idx=3).payloads()

    assert len(payloads) == len(module.blocks)
    assert all(set(payload) <= {"duration", "rf", "gx", "gy", "gz", "adc", "ext"} for payload in payloads)
    # Whatever axis carries the phase encode, changing lin_idx has to move the
    # payload -- that is the whole contract the fast path relies on. Snapshotted
    # because a module rewrites its payloads in place; see the test below.
    before = _snapshot(payloads)
    assert before != _snapshot(module.set_state(lin_idx=9).payloads())


def _snapshot(payloads):
    """A payload's values, frozen -- they are otherwise a live view."""
    return tuple(
        {key: (tuple(value) if isinstance(value, list | tuple) else value) for key, value in payload.items()}
        for payload in payloads
    )


def test_payloads_are_rewritten_in_place_not_rebuilt():
    """A module hands back the same payload objects every shot, by design.

    Rebuilding a dict per block per shot is the cost the whole fast path exists
    to avoid, so ``set_state`` rewrites the entries that moved and hands the
    same objects back. Anything that keeps a payload must copy it --
    :meth:`pulserver.pypulseq.Sequence._template_set_block` does.
    """
    module = readout.Line2D(_opts(), (0.22, 0.22), (32, 32), spoil_position="none")
    first = module.set_state(lin_idx=3).payloads()
    second = module.set_state(lin_idx=9).payloads()

    assert [id(p) for p in first] == [id(p) for p in second]
    assert _snapshot(first) == _snapshot(second), "the same objects, so the same values"


def test_the_peak_cache_survives_a_re_state_and_is_dropped_by_a_rebuild():
    """Peaks are keyed by sample-array identity, which a retune preserves.

    This is what makes ``payloads`` cheaper than rebuilding each event's
    library row rather than twice its price -- a module replays the same
    arrays every shot, so their peaks are derived once, not once per shot.
    """
    module = readout.Line2D(_opts(), (0.22, 0.22), (32, 32), spoil_position="none")
    module.set_state(lin_idx=0).payloads()
    entries = len(module._peak_cache)

    for index in range(1, 32):
        module.set_state(lin_idx=index).payloads()
    assert len(module._peak_cache) == entries, "a re-state must not mint new peaks"

    module._invalidate()
    assert module._peak_cache is None
