"""``add_range`` must be the per-shot loop, only faster.

Every test here compares a batched build against the loop it replaces and
demands the *bytes* match, not just the structure: event IDs are handed out in
visit order and ``remove_duplicates`` picks representatives by first
occurrence, so a batching that reorders registration would still produce a
physically correct sequence while changing the file and stale-ing every ``.pge``
truth fixture.
"""

from __future__ import annotations

import numpy as np
import pulserver.design as design
import pulserver.io as pio
import pulserver.pypulseq as pp
import pytest

OPTS_KW = {"max_grad": 40, "grad_unit": "mT/m", "max_slew": 150, "slew_unit": "T/m/s"}


@pytest.fixture
def system():
    return pp.Opts(**OPTS_KW)


def _payload(seq):
    return pio.write(seq, output=None, check_timing=False)


def _by_shot(system, items, length):
    """Replay a group one shot at a time — the loop ``add_range`` stands in for."""
    seq = pp.Sequence(system)
    for index in range(length):
        for module, states in items:
            if states is None:
                seq.add_block(module)
                continue
            module.set_state(**{name: values[index] for name, values in states.items()})
            for block in module:
                seq.add_block(*block)
    return seq


def _batched(system, items):
    seq = pp.Sequence(system)
    seq.add_range(*[module if states is None else (module, states) for module, states in items])
    return seq


def test_add_range_matches_the_per_shot_loop_byte_for_byte(system):
    ny, nz = 16, 8
    lin = np.repeat(np.arange(ny), nz)
    par = np.tile(np.arange(nz), ny)
    phases = np.asarray(design.make_rf_spoiling_schedule(len(lin)), dtype=float)

    # One shared module instance for both paths: the factories derate `system`
    # in place, so building them twice would design against different limits.
    pulse = design.make_slice_selective_pulse(np.deg2rad(10), 0.12, system=system)
    line = design.make_line_readout(system, (0.22, 0.22, 0.12), (32, ny, nz), spoil_position="post")
    items = [
        (pulse, {"phase_offset_rad": phases}),
        (pp.make_delay(2e-3), None),
        (line, {"lin_idx": lin, "par_idx": par, "phase_offset_rad": phases}),
        (pp.make_delay(3e-3), None),
    ]

    looped = _by_shot(system, items, len(lin))
    batched = _batched(system, items)

    assert len(batched.block_events) == len(looped.block_events)
    assert _payload(batched) == _payload(looped)


def test_add_range_scalars_broadcast_and_module_shorthand_agrees(system):
    ny = 24
    lin = np.arange(ny)
    line = design.make_line_readout(system, (0.22, 0.22, 0.12), (32, ny, 4), spoil_position="post")

    # par_idx held constant as a scalar must mean the same as spelling it out.
    looped = _by_shot(system, [(line, {"lin_idx": lin, "par_idx": np.zeros(ny, dtype=int)})], ny)
    broadcast = pp.Sequence(system)
    broadcast.add_range((line, {"lin_idx": lin, "par_idx": 0}))
    assert _payload(broadcast) == _payload(looped)

    # ...and the one-module shorthand is the same call.
    shorthand = pp.Sequence(system)
    line.add_range(shorthand, lin_idx=lin, par_idx=0)
    assert _payload(shorthand) == _payload(looped)


def test_add_range_falls_back_when_a_module_cannot_be_batched(system):
    """An unbatchable state must still be emitted, by the per-shot path."""
    ny = 16
    lin = np.arange(ny)
    # A varying amplitude scale rewrites the RF waveform per shot, so there is
    # nothing to batch; RfModule._bulk_plan declines and the loop takes over.
    pulse = design.make_slice_selective_pulse(np.deg2rad(10), 0.12, system=system)
    scales = np.linspace(0.5, 1.0, ny)

    looped = _by_shot(system, [(pulse, {"amplitude_scale": scales})], ny)
    batched = pp.Sequence(system)
    batched.add_range((pulse, {"amplitude_scale": scales}))
    assert _payload(batched) == _payload(looped)
    del lin


def test_add_range_below_the_batch_threshold_still_matches(system):
    """Short ranges take the per-shot path; the result must not depend on that."""
    from pulserver.pypulseq._sequence import _MIN_BULK_SHOTS

    ny = _MIN_BULK_SHOTS - 1
    assert ny >= 1
    lin = np.arange(ny)
    line = design.make_line_readout(system, (0.22, 0.22, 0.12), (32, 16, 4), spoil_position="post")

    looped = _by_shot(system, [(line, {"lin_idx": lin, "par_idx": 0 * lin})], ny)
    batched = pp.Sequence(system)
    batched.add_range((line, {"lin_idx": lin, "par_idx": 0 * lin}))
    assert _payload(batched) == _payload(looped)


def test_add_range_skips_none_items_and_rejects_mismatched_lengths(system):
    ny = 12
    lin = np.arange(ny)
    line = design.make_line_readout(system, (0.22, 0.22, 0.12), (32, ny, 4), spoil_position="post")

    # None is how an optional delay is passed without a branch at the call site.
    with_none = pp.Sequence(system)
    with_none.add_range(None, (line, {"lin_idx": lin, "par_idx": 0}), None)
    without = pp.Sequence(system)
    without.add_range((line, {"lin_idx": lin, "par_idx": 0}))
    assert _payload(with_none) == _payload(without)

    with pytest.raises(ValueError, match="expected"):
        pp.Sequence(system).add_range((line, {"lin_idx": lin, "par_idx": np.arange(ny + 1)}))


def test_add_range_labels_come_from_the_module_state_not_the_range(system):
    """Counters set before the call are constant across it, and must land on every shot."""
    ny = 16
    lin = np.arange(ny)
    line = design.make_line_readout(system, (0.22, 0.22, 0.12), (32, ny, 4), spoil_position="post")
    line.set_labels(SLC=3)

    looped = _by_shot(system, [(line, {"lin_idx": lin, "par_idx": 0 * lin})], ny)
    batched = pp.Sequence(system)
    batched.add_range((line, {"lin_idx": lin, "par_idx": 0 * lin}))
    assert _payload(batched) == _payload(looped)
    line.set_labels()


def test_add_range_batches_a_variable_flip_angle_schedule(system):
    """A per-shot flip angle is a payload column, not a shape.

    ``Sequence._compress_rf`` normalises the magnitude envelope by its own
    peak, so scaling a pulse leaves every shape ID alone and moves the
    amplitude field only -- exactly like the phase field beside it.
    """
    ny = 24
    lin = np.arange(ny)
    flips = np.linspace(0.2, 1.0, ny)  # a VFA ramp, no zero
    phases = np.asarray(design.make_rf_spoiling_schedule(ny), dtype=float)

    pulse = design.make_slice_selective_pulse(np.deg2rad(10), 0.12, system=system)
    line = design.make_line_readout(system, (0.22, 0.22, 0.12), (32, ny, 1), spoil_position="post")
    items = [
        (pulse, {"phase_offset_rad": phases, "amplitude_scale": flips}),
        (pp.make_delay(2e-3), None),
        (line, {"lin_idx": lin, "phase_offset_rad": phases}),
    ]

    looped = _by_shot(system, items, ny)
    batched = _batched(system, items)
    assert _payload(batched) == _payload(looped)

    # One magnitude shape for the whole ramp, on *both* paths: an RF module
    # publishes the envelope it scaled and the factor (``shape_source``), and
    # the writer normalises the envelope rather than the scaled array. That is
    # a property of how the pulse is described, not of batching -- so the range
    # and the loop register the same six shapes, not 6 against 24.
    assert len(batched.shape_library.data) == len(looped.shape_library.data) == 6


def test_add_range_falls_back_on_a_zero_flip_angle(system):
    """Scaling to zero normalises to an all-zero envelope -- a different shape."""
    ny = 16
    flips = np.linspace(0.0, 1.0, ny)
    pulse = design.make_slice_selective_pulse(np.deg2rad(10), 0.12, system=system)
    line = design.make_line_readout(system, (0.22, 0.22, 0.12), (32, ny, 1), spoil_position="post")
    items = [
        (pulse, {"amplitude_scale": flips}),
        (line, {"lin_idx": np.arange(ny)}),
    ]

    assert _payload(_batched(system, items)) == _payload(_by_shot(system, items, ny))


# ---------------------------------------------------------------------------
# Insert-time deduplication
#
# A range collapses repeated payloads as it registers them, which is only safe
# because it is a *refinement* of what write-time deduplication does anyway.
# The tests above already prove the important half -- they build the same
# sequence through ``add_block``, which does not collapse, and demand the same
# bytes -- so these cover the mechanism itself and the two places it must
# refuse to collapse.
# ---------------------------------------------------------------------------


def _library(profile=(6, -6), **kwargs):
    from pulserver.pypulseq._sequence import _AppendOnlyLibrary

    return _AppendOnlyLibrary(dedup_profile=profile, **kwargs)


def test_a_range_registers_one_entry_per_distinct_event_not_one_per_shot(system):
    """The point of the whole thing: the libraries stay the size of the vocabulary."""
    ny, nz = 32, 8
    shots = ny * nz
    pulse = design.make_slice_selective_pulse(np.deg2rad(10), 0.12, system=system)
    line = design.make_line_readout(system, (0.22, 0.22, 0.12), (32, ny, nz), spoil_position="post")
    items = [
        (pulse, {"phase_offset_rad": np.zeros(shots)}),
        (line, {"lin_idx": np.repeat(np.arange(ny), nz), "par_idx": np.tile(np.arange(nz), ny)}),
    ]

    batched = _batched(system, items)
    assert len(batched.rf_library) == 1, "one flip angle, one phase -- one RF entry"
    assert len(batched.trap_library) < shots
    assert len(batched.grad_library) < shots

    # And it is the same file the uncollapsed path emits.
    assert _payload(batched) == _payload(_by_shot(system, items, shots))


def test_extension_referenced_libraries_are_left_one_entry_per_event(system):
    """Collapsing these would reorder extension chains, which changes the file.

    A block sorts its extension chain by reference ID. Today's references climb
    with the shot, so one shot's chain order is every shot's; small shared IDs
    would order differently shot to shot, which the batched path cannot express
    and the per-shot path would resolve differently.
    """
    ny = 32
    line = design.make_line_readout(system, (0.22, 0.22, 0.12), (32, ny, 1), spoil_position="post")
    items = [(line, {"lin_idx": np.arange(ny), "par_idx": np.zeros(ny, dtype=int)})]

    batched = _batched(system, items)
    assert len(batched.label_set_library) >= ny
    assert _payload(batched) == _payload(_by_shot(system, items, ny))


def test_a_repeat_gets_the_first_occurrences_id_and_leaves_its_payload_alone():
    """What is stored is the original first row, never the rounded key.

    Rounding is not idempotent across a decade boundary, so storing the key
    would let the write-time pass round a second time and land somewhere the
    single-pass baseline never would.
    """
    lib = _library()
    ids = lib.extend(np.array([[9.9999999, 1.0], [2.0, 2.0], [9.99999985, 1.0]]))

    assert ids.tolist() == [1, 2, 1]
    assert len(lib) == 2
    assert lib.data[1] == (9.9999999, 1.0), "the first occurrence's own payload"


def test_ids_continue_across_calls_and_repeats_resolve_backwards():
    lib = _library()
    first = lib.extend(np.array([[1.0, 1.0], [2.0, 2.0]]))
    second = lib.extend(np.array([[3.0, 3.0], [1.0, 1.0], [3.0, 3.0]]))

    assert first.tolist() == [1, 2]
    assert second.tolist() == [3, 1, 3]
    assert len(lib) == 3


def test_the_entry_type_is_part_of_the_key():
    """``grad_library`` holds a bare reference into either of two libraries."""
    lib = _library(profile=(0,))
    ids = lib.extend(np.array([[7.0], [7.0], [7.0]]), np.array(["t", "g", "t"], dtype=object))

    assert ids.tolist() == [1, 2, 1]
    assert len(lib) == 2


def test_signed_zero_is_not_collapsed_into_zero():
    """``-0.0`` and ``0.0`` are distinct payloads and the file must keep them apart."""
    lib = _library()
    ids = lib.extend(np.array([[0.0, 1.0], [-0.0, 1.0], [0.0, 1.0]]))

    assert ids.tolist() == [1, 2, 1]
    assert np.signbit(lib.matrix()[1, 0])


def test_a_library_that_gives_up_on_its_index_still_registers_every_row():
    """The index is an optimisation; losing it must cost entries, never rows."""
    lib = _library()
    lib.extend(np.array([[1.0, 1.0], [2.0, 2.0]]))
    lib._dedup_off = True
    ids = lib.extend(np.array([[1.0, 1.0], [3.0, 3.0]]))

    assert ids.tolist() == [3, 4]
    assert len(lib) == 4
    assert lib.data[3] == (1.0, 1.0)


def test_without_a_profile_the_library_only_appends():
    lib = _library(profile=None)
    ids = lib.extend(np.array([[1.0, 1.0], [1.0, 1.0]]))

    assert ids.tolist() == [1, 2]
    assert len(lib) == 2


def test_a_tiled_type_pattern_must_cover_the_batch():
    lib = _library(profile=(0,))
    with pytest.raises(ValueError, match="cannot cover"):
        lib.extend(np.zeros((3, 1)), np.array(["t", "g"], dtype=object), tile=1)


def test_a_range_split_across_chunks_collapses_across_them_too(system, monkeypatch):
    """Chunking bounds memory; it must not bound what the index can see."""
    import pulserver.pypulseq._sequence as fast

    ny = 64
    line = design.make_line_readout(system, (0.22, 0.22, 0.12), (32, ny, 1), spoil_position="post")
    items = [(line, {"lin_idx": np.arange(ny), "par_idx": np.zeros(ny, dtype=int)})]

    whole = _batched(system, items)
    monkeypatch.setattr(fast, "_BULK_CHUNK_SHOTS", 8)
    chunked = _batched(system, items)

    assert len(chunked.trap_library) == len(whole.trap_library)
    assert _payload(chunked) == _payload(whole)


def test_ranges_and_single_blocks_interleave_without_disturbing_each_other(system):
    """The two paths collapse differently, and must still agree on the file.

    ``add_range`` collapses what it registers and ``add_block`` does not, so a
    sequence built from both holds a library whose front is collapsed and whose
    middle is not. Write-time deduplication has to reach the same answer over
    that as over a library where nothing was collapsed -- which is the whole
    claim insert-time deduplication rests on, exercised the way ``mprage_3d``
    exercises it: a prep block by block, then a train as a range, repeatedly.
    """
    ny, shots_per_segment = 16, 8
    lin = np.arange(ny)
    par = np.zeros(ny, dtype=int)
    phases = np.asarray(design.make_rf_spoiling_schedule(ny), dtype=float)
    pulse = design.make_slice_selective_pulse(np.deg2rad(10), 0.12, system=system)
    line = design.make_line_readout(system, (0.22, 0.22, 0.12), (32, ny, 1), spoil_position="post")
    inversion = design.make_inversion_pulse(adiabatic=False, system=system)
    recovery = pp.make_delay(5e-3)

    def build(batched):
        seq = pp.Sequence(system)
        for start in range(0, ny, shots_per_segment):
            stop = start + shots_per_segment
            for block in inversion:
                seq.add_block(*block)
            if batched:
                seq.add_range(
                    (pulse, {"phase_offset_rad": phases[start:stop]}),
                    (line, {"lin_idx": lin[start:stop], "par_idx": par[start:stop]}),
                )
            else:
                for index in range(start, stop):
                    pulse.set_state(phase_offset_rad=phases[index])
                    for block in pulse:
                        seq.add_block(*block)
                    line.set_state(lin_idx=lin[index], par_idx=par[index])
                    for block in line:
                        seq.add_block(*block)
            seq.add_block(recovery)
        return seq

    mixed = build(batched=True)
    looped = build(batched=False)
    # Every shot encodes a different line, so the prewinders cannot collapse --
    # the readout and spoiler trapezoids, which repeat, are what does.
    assert len(mixed.trap_library) < len(looped.trap_library)
    assert _payload(mixed) == _payload(looped)
