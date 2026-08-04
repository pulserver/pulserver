"""The compiled deduplication kernels against the NumPy ones they replace.

Every assertion here is bit-for-bit rather than approximate, and that is the
point: these two functions decide what numbers reach the .seq file and how many
distinct events it declares. A kernel that is right to fifteen digits would
silently split one gradient into two.

The tests are skipped, not failed, when the extension was not built -- it is
optional, and `fastseq.modules` falls back to the NumPy versions.
"""

import numpy as np
import pypulseq as pp
import pytest

from fastseq import modules as M
from fastseq.modules import (
    PeriodicSequence,
    SequenceModule,
    _collapse_gathered,
    _collapse_gathered_py,
    _rounded,
    _rounded_py,
    _unique_rows,
    _unique_rows_py,
)
from fastseq.sequence import _render_int_rows, _render_int_rows_py

pytestmark = pytest.mark.skipif(
    M._accel is None, reason='the fastseq accelerator was not built'
)


def identical(a, b):
    """Equal as bit patterns, so that -0.0 and NaN are not glossed over."""
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


# ---------------------------------------------------------------------------
# round_rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'digits',
    [
        (),
        M._GRAD_DIGITS,
        M._RF_DIGITS,
        M._ADC_DIGITS,
        (3,),                       # fewer digits than the table has columns
        (6, -6, -6, -6, -6, -6, 9, 9, 9, 9, 9, 9),   # more
        (0, 0, 0, 0, 0, 0),         # zero means round to the integer
    ],
)
def test_round_rows_matches_numpy(digits):
    rng = np.random.default_rng(7)
    table = np.column_stack([
        rng.normal(0.0, 1e5, 20000),          # gradient amplitudes
        rng.random(20000) * 2 * np.pi,        # phases
        rng.integers(0, 4, 20000) * 1e-5,     # exact multiples, incl. zero
        rng.normal(0.0, 1e-9, 20000),         # things that round to nothing
        10.0 ** rng.integers(-9, 7, 20000),   # exact powers of ten
        rng.integers(0, 12, 20000).astype(float),   # shape IDs
    ])
    assert identical(_rounded(table, digits), _rounded_py(table, digits))


def test_round_rows_folds_negative_zero():
    table = np.array([[-0.0, -0.0, 0.0], [0.0, 1.0, -1e-20]])
    out = _rounded(table, (6, -6, -6))
    assert identical(out, _rounded_py(table, (6, -6, -6)))
    assert not np.signbit(out).any()


def test_round_rows_rounds_halves_the_way_numpy_does():
    # np.round is round-half-to-even, and so is C's rint under the default
    # rounding mode. 0.5 and 1.5 must land on 0.0 and 2.0, not on 1.0 and 2.0.
    table = np.array([[0.5], [1.5], [2.5], [-0.5], [-1.5], [-2.5]])
    assert identical(_rounded(table, (0,)), _rounded_py(table, (0,)))
    assert list(_rounded(table, (0,)).ravel()) == [0.0, 2.0, 2.0, -0.0 + 0.0, -2.0, -2.0]


@pytest.mark.parametrize('shape', [(0, 5), (1, 1), (3, 0)])
def test_round_rows_degenerate_shapes(shape):
    table = np.zeros(shape)
    assert identical(_rounded(table, M._GRAD_DIGITS), _rounded_py(table, M._GRAD_DIGITS))


def test_round_rows_leaves_the_input_alone():
    table = np.array([[1.234567891, 2.0]])
    before = table.copy()
    _rounded(table, (6, -6))
    assert identical(table, before)


# ---------------------------------------------------------------------------
# unique_rows
# ---------------------------------------------------------------------------


def same_partition(rows, mine, theirs):
    """The two answers name the same groups, in the same order."""
    (unique_a, code_a), (unique_b, code_b) = mine, theirs
    assert identical(unique_a, unique_b)
    assert identical(code_a, code_b)
    # And the answer is actually right: each row equals its representative.
    assert identical(unique_a[code_a], np.ascontiguousarray(rows))


@pytest.mark.parametrize('distinct', [1, 2, 97, 5000, 20000])
def test_unique_rows_matches_numpy(distinct):
    rng = np.random.default_rng(11)
    base = rng.normal(0.0, 1e3, (distinct, 6))
    rows = np.ascontiguousarray(base[rng.integers(0, distinct, 20000)])
    same_partition(rows, _unique_rows(rows), _unique_rows_py(rows))


@pytest.mark.parametrize('distinct', [1, 900, 300000])
def test_unique_rows_over_the_threshold_that_splits_it(distinct):
    """Long enough to be deduplicated on threads, which is a different code path.

    The chunks each collapse against a table of their own and are merged in
    order afterwards, so the codes and the order of first appearance have to
    come out as a single pass would have written them -- whatever the machine
    running this decided about how many threads to use.
    """
    rng = np.random.default_rng(23)
    base = rng.normal(0.0, 1e3, (distinct, 4))
    rows = np.ascontiguousarray(base[rng.integers(0, distinct, 300000)])
    same_partition(rows, _unique_rows(rows), _unique_rows_py(rows))


def test_unique_rows_split_when_almost_every_row_is_distinct():
    """The case where merging the chunks has as much to do as collapsing them."""
    rng = np.random.default_rng(29)
    rows = np.ascontiguousarray(rng.integers(0, 1 << 62, (200000, 2)).astype(np.int64))
    same_partition(rows, _unique_rows(rows), _unique_rows_py(rows))


def test_unique_rows_on_integer_keys():
    # The extension chain table is int64, not float: a chain is a type, a
    # reference and a successor.
    rng = np.random.default_rng(3)
    rows = np.ascontiguousarray(rng.integers(0, 40, (20000, 3)).astype(np.int64))
    mine, theirs = _unique_rows(rows), _unique_rows_py(rows)
    same_partition(rows, mine, theirs)
    assert mine[0].dtype == rows.dtype


def test_unique_rows_first_appearance_order():
    rows = np.array([[3.0], [1.0], [3.0], [2.0], [1.0]])
    unique, code = _unique_rows(rows)
    assert list(unique.ravel()) == [3.0, 1.0, 2.0]
    assert list(code) == [0, 1, 0, 2, 1]


def test_unique_rows_separates_rows_that_share_a_hash():
    # A row's identity is the row, not its digest. Sixty thousand distinct rows
    # in a table sized for them is where an open-addressed table would show a
    # collision if it resolved one carelessly.
    rng = np.random.default_rng(5)
    rows = np.ascontiguousarray(rng.integers(0, 2**62, (60000, 2)).astype(np.int64))
    unique, code = _unique_rows(rows)
    assert unique.shape[0] == len(set(map(tuple, rows.tolist())))
    assert identical(unique[code], rows)


def test_unique_rows_treats_negative_zero_as_its_own_row():
    # Which is why `_rounded` folds it first. Documented here so that the
    # ordering of the two steps is not an accident anyone can undo.
    rows = np.array([[0.0], [-0.0]])
    assert _unique_rows(rows)[0].shape[0] == 2
    assert _unique_rows(_rounded(rows, (-6,)))[0].shape[0] == 1


@pytest.mark.parametrize('shape', [(0, 3), (1, 1)])
def test_unique_rows_degenerate_shapes(shape):
    rows = np.zeros(shape)
    mine, theirs = _unique_rows(rows), _unique_rows_py(rows)
    assert identical(mine[0], theirs[0])
    assert identical(mine[1], theirs[1])


def test_unique_rows_accepts_a_non_contiguous_view():
    rng = np.random.default_rng(13)
    wide = rng.integers(0, 5, (2000, 8)).astype(float)
    view = wide[::2, 1:5]
    assert not view.flags['C_CONTIGUOUS']
    same_partition(view, _unique_rows(view), _unique_rows_py(view))


# ---------------------------------------------------------------------------
# collapse_rows
# ---------------------------------------------------------------------------
#
# Rounding and deduplicating fused into one pass, so that the gathered table
# they would otherwise pass between them never exists. What it answers has to be
# what the two of them answer separately.


def collapse_agrees(table, take, digits, extra=None):
    fast_rows, fast_code = _collapse_gathered(table, take, digits, extra)
    slow_rows, slow_code = _collapse_gathered_py(table, take, digits, extra)
    assert identical(fast_rows, slow_rows)
    assert identical(fast_code, slow_code)


@pytest.mark.parametrize('digits', [(), (6,), (6, -6, -6, 0), (0, -9, 6, 6, 6, 6)])
def test_collapse_rows_matches_the_two_steps_it_replaces(digits):
    rng = np.random.default_rng(21)
    table = rng.normal(scale=1e4, size=(300, 6))
    take = rng.integers(0, 300, size=5000)
    collapse_agrees(table, take, digits)


@pytest.mark.parametrize('distinct', [3, 400000])
def test_collapse_rows_over_the_threshold_that_splits_it(distinct):
    """Long enough to be collapsed on threads, merged back in chunk order.

    The second case is the awkward one: nearly every row is its own, so the
    merge has as much to do as the chunks did, and a scan whose events really
    are all different is exactly when that has to stay correct.
    """
    rng = np.random.default_rng(24)
    table = rng.normal(scale=1e4, size=(distinct, 5))
    take = rng.integers(0, distinct, size=400000)
    collapse_agrees(table, take, (6, -6, -6, 0, 6))


def test_collapse_rows_split_with_an_extra_key_column():
    rng = np.random.default_rng(25)
    table = rng.normal(size=(700, 3))
    take = rng.integers(0, 700, size=250000)
    collapse_agrees(table, take, (6, 6, 6), rng.integers(0, 4, size=700).astype(float))


def test_collapse_rows_keys_on_the_extra_column_without_rounding_it():
    """An RF pulse's `use` decides identity but is not part of its data."""
    rng = np.random.default_rng(22)
    table = rng.normal(size=(40, 4))
    take = rng.integers(0, 40, size=900)
    extra = rng.integers(0, 3, size=40).astype(float)
    collapse_agrees(table, take, (6, 0, 0, 0), extra)


def test_collapse_rows_treats_negative_zero_as_its_own_row():
    table = np.array([[0.0, 1.0], [-0.0, 1.0]])
    collapse_agrees(table, np.array([0, 1, 0, 1]), (-6, -6))


@pytest.mark.parametrize('n', [0, 1])
def test_collapse_rows_degenerate_selections(n):
    table = np.arange(12, dtype=float).reshape(4, 3)
    collapse_agrees(table, np.zeros(n, dtype=np.int64), (0, 0, 0))


def test_collapse_rows_accepts_an_integer_selection_of_any_width():
    """The gradient index arrives as int32; nothing should have to say so."""
    rng = np.random.default_rng(23)
    table = rng.normal(size=(50, 5))
    take = rng.integers(0, 50, size=400).astype(np.int32)
    collapse_agrees(table, take, (6, -6, -6, -6, -6))


def test_collapse_rows_refuses_a_row_that_is_not_in_the_table():
    table = np.zeros((3, 2))
    with pytest.raises(IndexError):
        _collapse_gathered(table, np.array([0, 7]), (0, 0))


# ---------------------------------------------------------------------------
# render_int_rows
# ---------------------------------------------------------------------------
#
# `[BLOCKS]` and `[EXTENSIONS]` as text. Both are one row per block or per
# chain, so this is most of a large .seq file and every character of it has to
# be the character the NumPy version would have written.


def render_agrees(matrix, widths):
    assert _render_int_rows(matrix, widths) == _render_int_rows_py(matrix, widths)


def test_render_int_rows_matches_numpy():
    rng = np.random.default_rng(31)
    for _ in range(50):
        columns = int(rng.integers(1, 9))
        matrix = (rng.integers(0, 1000, size=(int(rng.integers(1, 80)), columns))
                  * 10 ** rng.integers(0, 6, size=columns)).astype(np.int64)
        render_agrees(matrix, tuple(int(w) for w in rng.integers(1, 8, size=columns)))


def test_render_int_rows_pads_to_the_field_and_widens_past_it():
    matrix = np.array([[1, 100000], [123456, 2]], dtype=np.int64)
    render_agrees(matrix, (3, 3))


def test_render_int_rows_refuses_negative_values():
    """`%3d` widens rather than truncating, so the caller formats those itself."""
    matrix = np.array([[1, -2]], dtype=np.int64)
    assert _render_int_rows(matrix, (2, 2)) is None
    assert _render_int_rows_py(matrix, (2, 2)) is None


def test_render_int_rows_on_an_empty_table():
    matrix = np.zeros((0, 3), dtype=np.int64)
    assert _render_int_rows(matrix, (1, 2, 3)) is None
    assert _render_int_rows_py(matrix, (1, 2, 3)) is None


# ---------------------------------------------------------------------------
# tile_rows
# ---------------------------------------------------------------------------
#
# The dense per-occurrence libraries, which is one seeded copy of the period per
# iteration. `np.tile` is the reference and the fallback.


def tiled_agrees(period, loop_size):
    mine = M._accel.tile_rows(period, loop_size)
    theirs = np.tile(period, (loop_size, 1))
    assert mine.shape == theirs.shape
    assert identical(mine, theirs)


@pytest.mark.parametrize('loop_size', [0, 1, 2, 37])
def test_tile_rows_matches_numpy(loop_size):
    rng = np.random.default_rng(17)
    tiled_agrees(rng.normal(size=(9, 4)), loop_size)


def test_tile_rows_over_the_threshold_that_splits_it():
    """Long enough to be laid out on threads, which is a different code path."""
    rng = np.random.default_rng(19)
    tiled_agrees(rng.normal(size=(701, 5)), 500)


@pytest.mark.parametrize('shape', [(0, 4), (5, 0), (0, 0)])
def test_tile_rows_degenerate_shapes(shape):
    tiled_agrees(np.zeros(shape), 6)


def test_tile_rows_refuses_a_negative_repeat():
    with pytest.raises(ValueError):
        M._accel.tile_rows(np.zeros((2, 2)), -1)


# ---------------------------------------------------------------------------
# block_columns
# ---------------------------------------------------------------------------
#
# What the loop's slot offsets and switches say the block table is. Every number
# here is a library ID, so a kernel that miscounts by one does not produce a
# wrong waveform -- it produces a different one, out of the same file.


def random_slots(rng, blocks):
    """A slot table shaped the way the constructor lays one out.

    Offsets are one-based and dense per kind, in block order, and a gradient's
    is signed: positive into the trapezoid table, negative into the arbitrary
    one. Some blocks get an event and some do not, which is the whole point --
    a zero column is what the template never gave that block.
    """
    slots = np.zeros((blocks, 5), dtype=np.int32)
    taken = {'rf': 0, 'adc': 0, 'trap': 0, 'arb': 0}
    for block in range(blocks):
        if rng.random() < 0.6:
            taken['rf'] += 1
            slots[block, 0] = (taken['rf'] - 1) * M._RF_WIDTH + 1
        if rng.random() < 0.5:
            taken['adc'] += 1
            slots[block, 4] = (taken['adc'] - 1) * M._ADC_WIDTH + 1
        for axis in range(3):
            draw = rng.random()
            if draw < 0.4:
                taken['trap'] += 1
                slots[block, 1 + axis] = (taken['trap'] - 1) * M._TRAP_WIDTH + 1
            elif draw < 0.7:
                taken['arb'] += 1
                slots[block, 1 + axis] = -((taken['arb'] - 1) * M._ARB_WIDTH + 1)
    return slots


def columns_agree(slots, on):
    widths = (M._RF_WIDTH, M._ADC_WIDTH, M._TRAP_WIDTH, M._ARB_WIDTH)
    fast = M._accel.block_columns(slots, on, *widths)
    slow = M._block_columns_py(slots, on, *widths)
    assert len(fast) == len(slow) == 4
    for mine, theirs in zip(fast, slow):
        assert identical(mine, theirs)
    return fast


def test_block_columns_matches_numpy():
    rng = np.random.default_rng(101)
    for _ in range(60):
        slots = random_slots(rng, int(rng.integers(1, 90)))
        on = bytearray(bytes(rng.integers(0, 2, size=slots.size, dtype=np.uint8)))
        columns_agree(slots, on)


@pytest.mark.parametrize('switch', [0, 1])
def test_block_columns_with_everything_off_and_everything_on(switch):
    """Off is a scan that played nothing; on is one that played every slot it had.

    The second is the interesting one: a switch set where no slot exists must
    still read as no event, rather than as row zero of a table.
    """
    slots = random_slots(np.random.default_rng(5), 40)
    columns_agree(slots, bytearray([switch]) * slots.size)


def test_block_columns_ranks_allocated_rows_but_indexes_played_ones():
    """The two halves of the bargain, and they count differently.

    A slot the loop skipped still consumes its rank -- the row behind it exists,
    which is what makes skipping it free, and pruning is what drops it. The
    index is the other way round: it is what deduplication reads, so it holds
    only the slots that were played.
    """
    slots = np.array([[0, 1, 0, 0, 0], [0, 6, 0, 0, 0], [0, 11, 0, 0, 0]], dtype=np.int32)
    on = bytearray(5 * 3)
    on[5 * 2 + 1] = 1                       # only the third block plays its gx
    _, grad, _, index = columns_agree(slots, on)
    assert grad.tolist() == [0, 0, 0, 0, 0, 0, 3, 0, 0]
    assert index.shape == (1, 2)
    assert index[0].tolist() == [0, 2]      # a trapezoid, the third row of its table


def test_block_columns_on_an_empty_scan():
    columns_agree(np.zeros((0, 5), dtype=np.int32), bytearray())


def test_block_columns_refuses_a_mismatched_switch_table():
    slots = np.zeros((4, 5), dtype=np.int32)
    with pytest.raises(ValueError):
        M._accel.block_columns(slots, bytearray(3), 1, 1, 1, 1)


# ---------------------------------------------------------------------------
# On a real scan
# ---------------------------------------------------------------------------


def test_a_built_sequence_is_the_same_with_and_without_the_kernels():
    """The whole pipeline, kernels on and off, compared library by library."""
    system = pp.Opts(rf_raster_time=2e-6, adc_raster_time=2e-6,
                     grad_raster_time=4e-6, block_duration_raster=4e-6)

    class Readout(SequenceModule):
        def init_module(self, system):
            rf = pp.make_block_pulse(flip_angle=np.deg2rad(10), duration=1e-3,
                                     system=system, use='excitation')
            gx_pre = pp.make_trapezoid(channel='x', area=-100.0, system=system)
            gy_pre = pp.make_trapezoid(channel='y', area=100.0, system=system)
            gx = pp.make_trapezoid(channel='x', flat_area=200.0, flat_time=400e-6,
                                   system=system)
            adc = pp.make_adc(num_samples=64, duration=400e-6, delay=gx.rise_time,
                              system=system)
            lin = pp.make_label(type='SET', label='LIN', value=0)
            self.seq = pp.Sequence(system=system)
            self.seq.add_block(rf)
            self.seq.add_block(gx_pre, gy_pre)
            self.seq.add_block(gx, adc, lin)

    def build(accelerated):
        saved = M._accel
        M._accel = saved if accelerated else None
        try:
            readout = Readout(system)
            seq = PeriodicSequence(system, 32, readout)
            scale = ((2 * np.arange(32) - 32) / 32).tolist()
            gy = readout.gy_pre
            amplitude = float(gy.amplitude)
            phase = 0.0
            for n in range(32):
                gy.amplitude = scale[n] * amplitude
                readout.rf.phase_offset = phase
                readout.adc.phase_offset = phase
                readout.lin.value = n
                seq.add_block(readout.rf)
                seq.add_block(readout.gx_pre, gy)
                seq.add_block(readout.gx, readout.adc, readout.lin)
                phase = (phase + 0.7) % (2 * np.pi)
            return seq.seq
        finally:
            M._accel = saved

    fast, slow = build(True), build(False)

    for name in ('rf_library', 'grad_library', 'adc_library', 'shape_library',
                 'label_set_library', 'extensions_library'):
        mine = getattr(fast, name).data
        theirs = getattr(slow, name).data
        assert mine.keys() == theirs.keys(), name
        for key in mine:
            assert identical(np.asarray(mine[key], dtype=float),
                             np.asarray(theirs[key], dtype=float)), (name, key)

    assert fast.block_events.keys() == slow.block_events.keys()
    for key in fast.block_events:
        assert identical(fast.block_events[key], slow.block_events[key]), key
    assert fast.block_durations == slow.block_durations
