"""Test automatic segment identification subroutines."""

import itertools

import numpy as np
import numpy.testing as npt

import pytest

from pulserver._core import _autoseg

# events
rf = 8
slice_reph = 9
phase_enc = 3
freq_enc = 1
read_adc = 4
read = 5
delay = 0
spoil = 7
prep = 13
crush = 11

isbal = [True, False]
Ny = [4, 5]
Nz = [4, 5]


def gre(balanced, ny, nz):
    # build single readout
    if balanced:
        segment = np.asarray(
            [
                rf,
                slice_reph,
                delay,
                phase_enc,
                freq_enc,
                read_adc,
                freq_enc,
                phase_enc,
                delay,
            ]
        )
    else:
        segment = np.asarray(
            [
                rf,
                slice_reph,
                delay,
                phase_enc,
                freq_enc,
                read_adc,
                freq_enc,
                phase_enc,
                spoil,
                delay,
            ]
        )

    # build main loop
    main_loop = np.concatenate([segment for n in range(ny * nz)])

    return main_loop


def ir_gre(balanced, ny, nz):
    # build prep
    prep_segment = np.asarray([prep, crush])

    # build readout
    read_block = gre(balanced, 1, nz)

    # build main loop
    main_block = np.concatenate([prep_segment, read_block])
    main_loop = np.concatenate([main_block for n in range(ny)])

    return main_loop


def ssfp_mrf():
    seq = ir_gre(False, 2, 2)
    seq[seq == freq_enc] *= -1  # rotate readout
    seq[seq == read_adc] *= -1
    return seq


def _calc_segment_idx(loop, segments):
    segments_idx = np.zeros(len(loop))
    for n in range(len(segments)):
        tmp = _autoseg.find_segments(loop, segments[n])
        segments_idx[tmp] = n
    segments_idx += 1
    return segments_idx


@pytest.mark.parametrize("isbal, Ny, Nz", list(itertools.product(*[isbal, Ny, Nz])))
def test_find_segment_definitions(isbal, Ny, Nz):
    # case 1: 3D GRE
    loop = gre(isbal, Ny, Nz)
    segments = _autoseg.find_segment_definitions(loop)

    assert len(segments) == 1
    if isbal:
        npt.assert_allclose(
            segments[0],
            [
                rf,
                slice_reph,
                delay,
                phase_enc,
                freq_enc,
                read_adc,
                freq_enc,
                phase_enc,
                delay,
            ],
        )
    else:
        npt.assert_allclose(
            segments[0],
            [
                rf,
                slice_reph,
                delay,
                phase_enc,
                freq_enc,
                read_adc,
                freq_enc,
                phase_enc,
                spoil,
                delay,
            ],
        )

    # case 2: 3D IR GRE
    loop = ir_gre(isbal, Ny, Nz)
    segments = _autoseg.find_segment_definitions(loop)

    assert len(segments) == 1
    if isbal:
        readout = [
            rf,
            slice_reph,
            delay,
            phase_enc,
            freq_enc,
            read_adc,
            freq_enc,
            phase_enc,
            delay,
        ]
        readout = np.concatenate([readout for n in range(Nz)]).tolist()
        npt.assert_allclose(segments[0], [prep, crush] + readout)
    else:
        readout = [
            rf,
            slice_reph,
            delay,
            phase_enc,
            freq_enc,
            read_adc,
            freq_enc,
            phase_enc,
            spoil,
            delay,
        ]
        readout = np.concatenate([readout for n in range(Nz)]).tolist()
        npt.assert_allclose(segments[0], [prep, crush] + readout)


def test_split_rotated_segments():
    loop = ssfp_mrf()
    segments = _autoseg.find_segment_definitions(loop)
    segments = _autoseg.split_rotated_segments(segments)

    # expected segments
    init = [prep, crush, rf, slice_reph, delay, phase_enc]
    readout = [freq_enc, read_adc, freq_enc]
    post_read = [phase_enc, spoil, delay, rf, slice_reph, delay, phase_enc]
    end = [phase_enc, spoil, delay]

    npt.assert_allclose(segments[0], init)
    npt.assert_allclose(segments[1], readout)
    npt.assert_allclose(segments[2], post_read)
    npt.assert_allclose(segments[3], end)


@pytest.mark.parametrize("isbal, Ny, Nz", list(itertools.product(*[isbal, Ny, Nz])))
def test_find_segments(isbal, Ny, Nz):
    # case 1: 3D GRE
    loop = gre(isbal, Ny, Nz)
    segments = _autoseg.find_segment_definitions(loop)
    segments_idx = _calc_segment_idx(loop, segments)

    expected = np.ones(len(loop))
    npt.assert_allclose(segments_idx, expected)

    # case 2: 3D IR GRE
    loop = ir_gre(isbal, Ny, Nz)
    segments = _autoseg.find_segment_definitions(loop)
    segments_idx = _calc_segment_idx(loop, segments)

    expected = np.ones(len(loop))
    npt.assert_allclose(segments_idx, expected)
