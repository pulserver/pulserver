"""``tr_struct`` must be the ordinary ``add_block`` loop, only faster.

Every test here builds the same sequence twice — once from events, once from
module payloads against a declared TR template — and demands the *bytes* match.
That is the only check worth making: event IDs are handed out in visit order and
``remove_duplicates`` picks representatives by first occurrence, so a template
that registers the same events in a different order produces a physically
identical sequence and a different file, which stales every ``.pge`` fixture.

Three bugs during development were caught here and nowhere else, all of them at
the seam where blocks *outside* the template meet the template: shape IDs
numbered ahead of a preparation pulse, event IDs interleaved across a flush, and
a stride writing into the wrong rows. The mixed test is the important one.
"""

from __future__ import annotations

import numpy as np
import pulserver.design as design
import pulserver.io as pio
import pulserver.pypulseq as pp
import pytest

OPTS_KW = {"max_grad": 40, "grad_unit": "mT/m", "max_slew": 150, "slew_unit": "T/m/s"}
FOV = (0.22, 0.22, 0.12)


@pytest.fixture
def system():
    return pp.Opts(**OPTS_KW)


def _payload(seq):
    return pio.write(seq, output=None, check_timing=False)


@pytest.fixture
def modules(system):
    """One excitation and one Cartesian readout, shared by both build paths.

    Shared on purpose: the factories derate ``system`` in place, so building
    them twice would design against different limits.
    """
    pulse = design.make_slice_selective_pulse(np.deg2rad(10), 0.12, system=system)
    line = design.make_line_readout(system, FOV, (32, 16, 4), slice_rephasing=pulse.rephasers)
    return pulse.without_rephasers(), line


def _states(pulse, line, lin, par, phases, index):
    pulse.set_state(phase_offset_rad=phases[index])
    line.set_state(lin_idx=int(lin[index]), par_idx=int(par[index]), phase_offset_rad=phases[index])


def _build(system, modules, lin, par, phases, *, template, loop_size=None):
    loop_size = len(lin) if loop_size is None else loop_size
    pulse, line = modules
    if template:
        _states(pulse, line, lin, par, phases, 0)
        seq = pp.Sequence(system, loop_size, *pulse.blocks, *line.blocks)
    else:
        seq = pp.Sequence._unstructured(system)

    for index in range(len(lin)):
        _states(pulse, line, lin, par, phases, index)
        for module in (pulse, line):
            if template:
                for payload in module.payloads():
                    seq.add_block(payload)
            else:
                for block in module.blocks:
                    seq.add_block(*block)
    return seq


@pytest.fixture
def encoding():
    ny, nz = 16, 4
    lin = np.repeat(np.arange(ny), nz)
    par = np.tile(np.arange(nz), ny)
    return lin, par, np.asarray(design.make_rf_spoiling_schedule(len(lin)))


def test_a_sequence_cannot_be_built_without_declaring_its_structure(system, modules):
    """The public contract: there is no block-at-a-time builder any more.

    Every library row is claimed from the declared structure up front, which is
    what makes ``add_block`` a handful of scalar writes. A sequence with nothing
    declared has nothing to claim from, so it is refused at construction rather
    than half-working.

    The period need not be the TR -- it is whatever actually repeats, and a scan
    with a once-only lead-in or tail is one pass over the whole thing.
    """
    pulse, _ = modules
    for attempt in (lambda: pp.Sequence(system), lambda: pp.Sequence(), lambda: pp.Sequence(system, 4)):
        with pytest.raises(TypeError, match="structure that repeats"):
            attempt()

    # Declared, and it builds.
    pulse.set_state(phase_offset_rad=0.0)
    assert pp.Sequence(system, 1, pulse)._template is not None

    # The reference builder is still reachable privately -- io.read needs it, and
    # so do the byte comparisons below, which are the whole point of it.
    assert pp.Sequence._unstructured(system)._template is None


def test_template_path_matches_the_event_path_byte_for_byte(system, modules, encoding):
    lin, par, phases = encoding
    plain = _build(system, modules, lin, par, phases, template=False)
    fast = _build(system, modules, lin, par, phases, template=True)
    assert _payload(fast) == _payload(plain)


def test_every_row_the_scan_will_use_is_claimed_before_the_first_block(system, modules, encoding):
    """The whole design in one assertion: construction hands out every ID.

    ``loop_size`` and the TR's structure between them say exactly how many
    library rows the scan can want, so they are all claimed here -- and
    ``add_block`` becomes a write into a row that already exists at an ID
    already settled. Nothing is buffered, nothing is interned later, and no ID
    is decided while the scan is being filled. That is what lets the block
    table be written up front too.

    One row per TR per slot, deliberately: a payload may move any of them, so
    none can be shared, and whatever turns out to repeat is collapsed once by
    the write-time pass rather than guessed at here.
    """
    lin, par, phases = encoding
    pulse, line = modules
    shots = len(lin)

    _states(pulse, line, lin, par, phases, 0)
    fast = pp.Sequence(system, shots, *pulse.blocks, *line.blocks)
    # Six trapezoids, one RF and one ADC per TR -- claimed before a single
    # block has been added.
    claimed = {name: len(getattr(fast, name)) for name in ("trap_library", "rf_library", "adc_library")}
    assert claimed == {"trap_library": 6 * shots, "rf_library": shots, "adc_library": shots}

    for index in range(shots):
        _states(pulse, line, lin, par, phases, index)
        for module in (pulse, line):
            for payload in module.payloads():
                fast.add_block(payload)

    # Filling the scan registered nothing: the rows were already there.
    assert {name: len(getattr(fast, name)) for name in claimed} == claimed

    plain = _build(system, modules, lin, par, phases, template=False)
    assert _payload(fast) == _payload(plain)


def test_a_scan_that_plays_less_than_it_claimed_gives_the_rest_back(system, modules, encoding):
    """An entry nothing points at is still an entry in the file, so it has to go.

    Rows are claimed in the order the blocks fill them, so what a short scan
    leaves unused is always a *tail* -- which is why giving it back is a
    truncation and not a renumbering.
    """
    lin, par, phases = encoding
    pulse, line = modules
    played = len(lin) // 2

    _states(pulse, line, lin, par, phases, 0)
    short = pp.Sequence(system, len(lin), *pulse.blocks, *line.blocks)
    assert len(short.trap_library) == 6 * len(lin)
    for index in range(played):
        _states(pulse, line, lin, par, phases, index)
        for module in (pulse, line):
            for payload in module.payloads():
                short.add_block(payload)

    exact = _build(system, modules, lin[:played], par[:played], phases[:played], template=True)
    assert _payload(short) == _payload(exact)
    assert len(short.trap_library) == 6 * played


def test_a_delay_between_whole_passes_lands_where_it_was_added(system, modules, encoding):
    """The seam a segmented acquisition sits on, with the pass boundary exact.

    The recovery delay owns no library row, so it may sit between passes -- but
    it joins the block table the moment it is added, while the template's own
    rows were written when the sequence was built and are handed over in runs.
    If the run is not closed here, every one of these delays ends up in the
    table *ahead* of the TRs it was supposed to follow.

    Deliberately an even division, so every delay falls on a completed pass and
    nothing is cut short: that is the case a "the pass stopped early" guard
    skips, and it was wrong for exactly that reason.
    """
    pulse, line = modules
    lin, par, phases = encoding
    recovery = pp.make_delay(5e-3)
    etl = 8
    assert len(lin) % etl == 0, "this test is about the segment that fills exactly"

    def build(template):
        if template:
            _states(pulse, line, lin, par, phases, 0)
            view = [*pulse.blocks, *line.blocks]
            seq = pp.Sequence(system, len(lin) // etl, *(etl * view))
        else:
            seq = pp.Sequence._unstructured(system)
        for start in range(0, len(lin), etl):
            for index in range(start, start + etl):
                _states(pulse, line, lin, par, phases, index)
                for module in (pulse, line):
                    if template:
                        for payload in module.payloads():
                            seq.add_block(payload)
                    else:
                        for block in module.blocks:
                            seq.add_block(*block)
            seq.add_block(recovery)
        return seq

    assert _payload(build(True)) == _payload(build(False))


def test_a_preparation_belongs_inside_the_template_and_a_short_segment_is_fine(
    system, modules, encoding
):
    """The MPRAGE shape again, but with the whole *segment* declared as the unit.

    A preparation is a module like any other, so it goes in ``tr_struct`` and
    takes the fast path with everything else — there is no such thing as a block
    the template merely tolerates. The train length need not divide the view
    count, and here deliberately does not: the last segment stops part way
    through the template, is ended by the recovery delay, and its blocks have to
    be registered against the prefix the pass reached rather than left with the
    IDs they were born with.
    """
    pulse, line = modules
    lin, par, phases = encoding
    inversion = design.make_inversion_pulse(adiabatic=False, system=system)
    ti_delay, recovery = pp.make_delay(20e-3), pp.make_delay(5e-3)
    etl = 10
    assert len(lin) % etl, "this test is about the segment that does not fill"

    def build(template):
        if template:
            _states(pulse, line, lin, par, phases, 0)
            view = [*pulse.blocks, *line.blocks]
            seq = pp.Sequence(system, -(-len(lin) // etl), inversion, ti_delay, *(etl * view))
        else:
            seq = pp.Sequence._unstructured(system)
        for start in range(0, len(lin), etl):
            for payload in inversion.payloads():
                seq.add_block(payload) if template else seq.add_block(*payload.events)
            seq.add_block(ti_delay)
            for index in range(start, min(start + etl, len(lin))):
                _states(pulse, line, lin, par, phases, index)
                for module in (pulse, line):
                    if template:
                        for payload in module.payloads():
                            seq.add_block(payload)
                    else:
                        for block in module.blocks:
                            seq.add_block(*block)
            seq.add_block(recovery)
        return seq

    fast = build(True)
    assert _payload(fast) == _payload(build(False))
    # The preparation is a template block like any other, so its rows were
    # claimed with everything else: one per segment, and none left over for the
    # segment that did not fill.
    segments = -(-len(lin) // etl)
    assert len(fast.rf_library) == len(lin) + segments


def test_a_template_may_not_be_added_to_after_it_has_been_written(system, modules, encoding):
    """Writing gives the unclaimed rows back, so there is nothing left to write into.

    This is the cost of claiming every row up front: the rows a scan did not
    reach are counted off the end when the block table is read, and a block
    added after that would land on an ID something else now holds. Refused
    rather than silently emitted wrong.
    """
    pulse, line = modules
    lin, par, phases = encoding
    _states(pulse, line, lin, par, phases, 0)
    seq = pp.Sequence(system, len(lin), *pulse.blocks, *line.blocks)

    seq.add_block(pulse.payloads()[0])
    _payload(seq)  # finishes the template and truncates what it did not reach
    with pytest.raises(RuntimeError, match="cannot be added to afterwards"):
        seq.add_block(line.payloads()[0])


def test_a_run_split_across_flushes_is_the_same_file(system, modules, encoding, monkeypatch):
    """Flushing bounds memory; it must not bound what the file looks like."""
    import pulserver.pypulseq._sequence as fast_sequence

    lin, par, phases = encoding
    whole = _build(system, modules, lin, par, phases, template=True)
    monkeypatch.setattr(fast_sequence, "_TEMPLATE_FLUSH_ROWS", 4)
    chunked = _build(system, modules, lin, par, phases, template=True)

    assert _payload(chunked) == _payload(whole)
    assert len(chunked.trap_library) == len(whole.trap_library)


def test_a_block_outside_the_template_may_own_no_library_row(system, modules, encoding):
    """Every ID is handed out at construction, so a foreign row would shift them all.

    A pure delay is the one thing that may sit between passes -- it owns
    nothing -- and anything that would register a row of its own is refused
    where it stands rather than quietly renumbering the file behind it.
    """
    pulse, line = modules
    lin, par, phases = encoding
    _states(pulse, line, lin, par, phases, 0)
    inversion = design.make_inversion_pulse(adiabatic=False, system=system)
    seq = pp.Sequence(system, len(lin), *pulse.blocks, *line.blocks)

    seq.add_block(pulse.payloads()[0])
    with pytest.raises(RuntimeError, match="put this one in tr_struct"):
        seq.add_block(*inversion.blocks[0])


def test_a_payload_cache_rebuilt_mid_scan_does_not_apply_the_rf_offsets_twice(
    system, modules, encoding
):
    """Anything that drops a module's payload cache mid-scan must not double its state.

    ``RfModule._direct_payloads`` caches the *template's* frequency and phase and
    adds the state's offsets to them. If the cache is dropped at a shot whose
    offsets are already applied, rebuilding it against the *rendered* event
    rather than the template event adds them a second time.

    This was silent: every pass ran, the block count was right, and the file
    carried doubled RF phase and doubled slice frequency offset -- which puts a
    multi-slice 2D acquisition's slices in the wrong place. Both offsets are
    varied here because they are separate entries of the same tuple.

    A per-shot counter is what used to trigger it, once per shot; it no longer
    does, because a label event is declared once and only its value moves. What
    still can is naming a *new* label mid-scan, or any change of structure, so
    the drop is provoked directly here rather than through whichever caller
    happens to cause it today.
    """
    pulse, line = modules
    lin, par, phases = encoding
    offsets = np.linspace(-2000.0, 2000.0, len(lin))

    def build(templated):
        pulse.set_state(phase_offset_rad=0.0, freq_offset_hz=0.0, LIN=0)
        line.set_state(lin_idx=0, par_idx=0, phase_offset_rad=0.0)
        seq = (
            pp.Sequence(system, len(lin), *pulse.blocks, *line.blocks)
            if templated
            else pp.Sequence._unstructured(system)
        )
        for index in range(len(lin)):
            pulse.set_state(
                phase_offset_rad=phases[index],
                freq_offset_hz=float(offsets[index]),
                LIN=int(lin[index]),
            )
            pulse._payload_cache = None
            for block in pulse:
                seq.add_block(*block)
            line.set_state(
                lin_idx=int(lin[index]), par_idx=int(par[index]), phase_offset_rad=phases[index]
            )
            for block in line:
                seq.add_block(*block)
        return seq

    assert _payload(build(True)) == _payload(build(False))


def test_loop_size_is_a_ceiling_not_a_hint(system, modules, encoding):
    lin, par, phases = encoding
    with pytest.raises(RuntimeError, match="loop_size"):
        _build(system, modules, lin, par, phases, template=True, loop_size=len(lin) - 1)


def test_every_event_type_can_live_inside_the_template(system, modules, encoding):
    """There is no block a TR template cannot express -- including a transmit shim.

    A shim is the same shape of thing as a rotation: *whether* a block carries
    one is structural, the numbers in it are not. So it is a payload, its row is
    claimed with everything else, and each pass writes its own vector into it.
    The width is the transmit channel count, which a pulse knows at construction.

    This test used to assert the opposite -- that a shim, and before that a
    trigger, could not be expressed and was refused. Both were writer gaps, not
    facts about sequences.
    """
    pulse, line = modules
    lin, par, phases = encoding
    vectors = [
        np.exp(1j * np.linspace(0.0, np.pi, 4)) * (0.5 + 0.1 * index) for index in range(4)
    ]
    # One shim *event*, re-pointed per pass: recognition is by identity, so the
    # block has to hand back the object the template recorded.
    shim = pp.make_rf_shim(vectors[0])

    def build(templated):
        _states(pulse, line, lin, par, phases, 0)
        shim.shim_vector = vectors[0]
        seq = (
            pp.Sequence(system, len(lin), (*pulse.blocks[0], shim), *pulse.blocks[1:], *line.blocks)
            if templated
            else pp.Sequence._unstructured(system)
        )
        for index in range(len(lin)):
            _states(pulse, line, lin, par, phases, index)
            shim.shim_vector = vectors[index % len(vectors)]
            seq.add_block(*pulse.blocks[0], shim)
            for block in pulse.blocks[1:]:
                seq.add_block(*block)
            for block in line.blocks:
                seq.add_block(*block)
        return seq

    fast = build(True)
    assert _payload(fast) == _payload(build(False))
    assert fast._template_pass == len(lin)
    # Four distinct vectors over the whole scan, whatever the pass count.
    assert len({tuple(row) for row in fast.rf_shim_library.data.values()}) == len(vectors)


def test_a_trigger_may_live_inside_the_template(system, modules, encoding):
    """A once-per-frame trigger owns a row, so the template has to be able to hold it.

    Which is what makes the *frame* a usable pass for a cine or a dynamic scan:
    the trigger cannot sit outside the template, because every library row is
    claimed up front and a foreign row would shift every ID after it.
    """
    pulse, line = modules
    lin, par, phases = encoding
    frames = 4
    per_frame = len(lin) // frames
    trigger = pp.make_trigger("physio1", duration=1e-3, system=system)

    def build(templated):
        _states(pulse, line, lin, par, phases, 0)
        view = [*pulse.blocks, *line.blocks]
        seq = (
            pp.Sequence(system, frames, trigger, *(per_frame * view))
            if templated
            else pp.Sequence._unstructured(system)
        )
        for frame in range(frames):
            seq.add_block(trigger)
            for index in range(frame * per_frame, (frame + 1) * per_frame):
                _states(pulse, line, lin, par, phases, index)
                for module in (pulse, line):
                    for block in module:
                        seq.add_block(*block)
        return seq

    fast = build(True)
    assert _payload(fast) == _payload(build(False))
    assert fast._template_pass == frames


def test_a_repeatedly_interrupted_template_stays_linear_in_the_block_count(system, modules):
    """The flush must patch its own run, not re-collapse the whole table.

    A preparation between passes forces a flush, so a segmented acquisition
    flushes once per segment. Rebuilding the block matrix each time is
    quadratic, and the emitted file is identical either way -- so nothing but a
    cost measurement can catch it. Measured in *rows copied* rather than
    seconds, which is the thing that actually grows.

    This was real: inversion-prepared MPRAGE at 192**3 spent 4.3 s of 9.9 s
    rebuilding the table.
    """
    import pulserver.pypulseq._sequence as fast_sequence

    pulse, line = modules
    copied = []
    real = fast_sequence.Sequence._raw_block_row_matrix

    def counting(self):
        rows = real(self)
        copied.append(len(rows))
        return rows

    def build(segments, etl=4):
        # The encode is irrelevant here -- only the block count is -- so it
        # simply cycles inside the readout's own matrix.
        lin = np.arange(segments * etl) % 16
        par = np.zeros_like(lin)
        phases = np.zeros(len(lin))
        _states(pulse, line, lin, par, phases, 0)
        seq = pp.Sequence(system, len(lin), *pulse.blocks, *line.blocks)
        for start in range(0, len(lin), etl):
            seq.add_block(pp.make_delay(5e-3))
            for index in range(start, start + etl):
                _states(pulse, line, lin, par, phases, index)
                for module in (pulse, line):
                    for payload in module.payloads():
                        seq.add_block(payload)
        return seq

    fast_sequence.Sequence._raw_block_row_matrix = counting
    try:
        copied.clear()
        build(8)
        small = sum(copied)
        copied.clear()
        build(32)
        large = sum(copied)
    finally:
        fast_sequence.Sequence._raw_block_row_matrix = real

    # Four times the segments must not cost sixteen times the copying. The
    # quadratic version scored ratios above 10 here; linear scores about 4.
    assert large <= 6 * max(small, 1), f"copying grew {large / max(small, 1):.1f}x for 4x the blocks"


def test_a_moving_waveform_with_no_registered_shape_is_refused(system):
    """Samples are data, but the shape they name has to exist before the shot runs.

    An FSE partition encode rides the slice crushers, and two gradients summed
    on one axis are an arbitrary waveform that moves with the encode. Nothing
    about that stops it being templated -- the TR is still the template, and the
    samples are a number like any other. What stops *this* module is that it
    publishes no waveform inventory, so there is no registered shape for the
    later partition to point at, and the only alternatives are compressing
    samples inside ``add_block`` (re-deriving an ID the module could have named)
    or emitting the first partition's shape under the later one's peak.

    That last one is what used to happen, silently: a 4-partition FSE wrote
    42,007 bytes against 42,291. Refusing is the floor, not the goal -- the goal
    is the module handing every shot's waveform to its constructor.
    """
    refocusing = design.make_refocusing_pulse(slice_thickness=0.16, system=system)
    etl, ny = 8, 128
    fse = design.make_fse_readout(system, (0.24, 0.24, 0.24), (128, ny, 32), etl, refocusing)
    lin = np.arange(etl) * (ny // etl)

    fse.set_state(lin_idx=lin, par_idx=0)
    seq = pp.Sequence(system, 2, *fse.blocks)

    # The partition it was built at is fine, even though the retune hands back
    # freshly built arrays: they hold the same samples, and the check is on the
    # shape moving rather than on the object being new.
    fse.set_state(lin_idx=lin, par_idx=0)
    for _block in fse:
        seq.add_block(*_block)
    fse.set_state(lin_idx=lin, par_idx=3)
    with pytest.raises(NotImplementedError, match="published no waveform inventory"):
        for _block in fse:
            seq.add_block(*_block)
def test_a_fixed_waveform_still_goes_through_the_template(system):
    """The check is on the samples moving, not on the gradient being arbitrary.

    A spiral is nothing but arbitrary gradients, and it is the case the whole
    non-Cartesian half of the zoo rests on: one designed interleaf replayed
    under a rotation or a phase. If the check cost that the fast path it would
    be worse than the bug it prevents.
    """
    arm = design.make_spiral_readout(system, 0.22, 32, 8, variant="in_out", num_points=256)
    assert any(
        getattr(event, "type", None) == "grad" for block in arm.set_state(lin_idx=0).blocks
        for event in block
    ), "this test is only meaningful if the readout carries an arbitrary gradient"

    def build(template):
        seq = (
            pp.Sequence(system, 8, *arm.set_state(lin_idx=0).blocks)
            if template
            else pp.Sequence._unstructured(system)
        )
        for shot in range(8):
            arm.set_state(lin_idx=shot, phase_offset_rad=0.1 * shot)
            for _block in arm:
                seq.add_block(*_block)
        return seq

    assert _payload(build(True)) == _payload(build(False))


def test_the_ordinary_event_loop_is_the_fast_path_on_a_templated_sequence(system, modules, encoding):
    """``for block in module.blocks: seq.add_block(*block)`` — unchanged, and templated.

    The loop the whole design exists to keep. Nothing at the call site says
    "use the template": a module hands back the event objects the template was
    built from, so the sequence recognises them by identity and takes them
    through it. This is checked on the *libraries*, because the bytes are
    identical either way and would not show whether the template was used.
    """
    pulse, line = modules
    lin, par, phases = encoding

    def build(templated):
        if templated:
            _states(pulse, line, lin, par, phases, 0)
            seq = pp.Sequence(system, len(lin), *pulse.blocks, *line.blocks)
        else:
            seq = pp.Sequence._unstructured(system)
        for index in range(len(lin)):
            _states(pulse, line, lin, par, phases, index)
            for module in (pulse, line):
                for block in module.blocks:  # the canonical loop, verbatim
                    seq.add_block(*block)
        return seq

    plain, fast = build(False), build(True)
    assert _payload(fast) == _payload(plain)
    # Checked on the cursor rather than on the libraries: both paths claim the
    # same number of rows now, so only the pass count says whether the blocks
    # went *through* the template or merely alongside it.
    assert fast._template_pass == len(lin), "the event loop must reach the template"
    assert fast._template_written == len(lin) * len(fast._template)


def test_a_block_that_is_not_the_templates_own_events_is_not_absorbed(system, modules, encoding):
    """Recognition is by identity, so a lookalike preparation cannot be swallowed.

    An inversion pulse is an RF event and a gradient, exactly like the
    excitation at template position 0. Matching on *structure* would take it for
    that block and write its numbers into the excitation's rows -- a wrong file,
    not a slow one. Matching on identity cannot, so it reaches the boundary
    check and is refused there.
    """
    pulse, line = modules
    lin, par, phases = encoding
    inversion = design.make_inversion_pulse(adiabatic=False, system=system)
    _states(pulse, line, lin, par, phases, 0)
    seq = pp.Sequence(system, len(lin), *pulse.blocks, *line.blocks)

    with pytest.raises(RuntimeError, match="put this one in tr_struct"):
        for block in inversion.blocks:
            seq.add_block(*block)


def test_a_pure_delay_block_takes_its_duration_from_the_payload(system, modules):
    """A delay is an event type like any other: prepared once, moved per shot.

    A TE or TR delay that varies -- a TI schedule, a variable recovery -- is
    one number, so it belongs in the payload rather than forcing the block out
    of the template.
    """
    pulse, _ = modules
    pulse.set_state(phase_offset_rad=0.0)
    tr_struct = [*pulse.blocks, pp.make_delay(1e-3)]
    durations = (1e-3, 4e-3, 2.5e-3)

    def build(templated):
        seq = pp.Sequence(system, len(durations), *tr_struct) if templated else pp.Sequence._unstructured(system)
        for duration in durations:
            pulse.set_state(phase_offset_rad=0.0)
            for block in pulse.blocks:
                seq.add_block(*block)
            seq.add_block({"duration": duration} if templated else pp.make_delay(duration))
        return seq

    fast = build(True)
    assert _payload(fast) == _payload(build(False))
    assert list(fast.block_durations.values())[1::2] == pytest.approx(list(durations))
