"""Collection and state protocol shared by class-based readouts."""

from __future__ import annotations

import numpy as np
import pytest

pp = pytest.importorskip("pypulseq")

from pulserver.pypulseq import _readout as readout  # noqa: E402

OPTS_KW = dict(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


def _opts():
    return pp.Opts(**OPTS_KW)


def test_collection_access_requires_state():
    module = readout.Line2D(_opts(), (0.22, 0.22), (32, 32), spoil_position="none")

    with pytest.raises(RuntimeError, match=r"set_state\(\)"):
        len(module)
    with pytest.raises(RuntimeError, match=r"set_state\(\)"):
        module[0]
    with pytest.raises(RuntimeError, match=r"set_state\(\)"):
        list(module)


def test_line_supports_len_index_slice_iteration_add_to_and_get():
    opts = _opts()
    module = readout.Line2D(opts, (0.22, 0.22), (32, 32), spoil_position="none")
    assert isinstance(module.set_state(lin_idx=7, adc_phase_rad=0.25), readout.Readout)

    blocks = tuple(module)
    assert len(module) == module.num_blocks == len(blocks) == 3
    assert module[0] is blocks[0]
    assert module[-1] is blocks[-1]
    assert module[:] == blocks
    assert all(isinstance(block, tuple) and block for block in blocks)

    seq = pp.Sequence(opts)
    assert module.add_to(seq) is seq
    assert len(seq.block_events) == len(module)

    standalone = module.get()
    assert len(standalone.block_events) == len(module)


def test_set_state_replaces_cached_block_snapshot():
    module = readout.Line2D(_opts(), (0.22, 0.22), (32, 32), spoil_position="none")
    module.set_state(lin_idx=3, adc_phase_rad=0.1)
    old_blocks = tuple(module)

    module.set_state(lin_idx=4, adc_phase_rad=0.9)
    new_blocks = tuple(module)

    assert new_blocks is not old_blocks
    assert new_blocks[0] is not old_blocks[0]
    adc_events = [event for block in new_blocks for event in block if getattr(event, "type", None) == "adc"]
    assert adc_events and all(event.phase_offset == pytest.approx(0.9) for event in adc_events)


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
    module = readout.NonCartesian2D(arm).set_state(lin_idx=2, adc_phase_rad=0.4)

    assert isinstance(module, readout.Readout)
    assert len(module) == 3
    assert len(module.get().block_events) == len(module)


def test_callable_api_remains_compatible():
    opts = _opts()
    module = readout.Line2D(opts, (0.22, 0.22), (32, 32), spoil_position="none")
    seq = pp.Sequence(opts)

    assert module(seq=seq, pe_idx=5, rf_phase_rad=0.3) is seq
    assert len(seq.block_events) == len(module)
