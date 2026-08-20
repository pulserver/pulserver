"""The claim the acoustic gate rests on: one TR's answer covers the scan.

``pulseg_check_safety`` judges a single canonical TR and calls the result
safe for every repetition of it. That reasoning is sound only if the number
it judges is at least what any instance of the TR really drives, at every
frequency the bands guard. These tests are that assertion, not a regression
baseline: multishot readouts, phase encodes and rotation extensions all make
the instances differ, and each of them is checked here against the bound
that stands in for them.
"""

import numpy as np
import pytest
from pulserver._ext.pulseg import _calc_mech_resonances, _check_safety

from .conftest import CORPUS

#: ``pulseg_types.h``'s amplitude modes: the bound over every instance, and
#: one instance exactly as it plays.
BOUND, ACTUAL = 0, 2

GAMMA_HZ_PER_MT_PER_M = 42.576e3

#: Fixtures whose canonical TR is played more than once, which is where the
#: bound has anything to do. Covers a Cartesian phase encode, an echo train,
#: rotated arms and arms written out as their own waveforms.
REPEATED = [
    "gre_2d.seq",
    "gre_2d_3sl.seq",
    "gre_3d.seq",
    "se_2d.seq",
    "fse_2d.seq",
    "fse_3d.seq",
    "mprage_3d.seq",
    "gre_radial_2d.seq",
    "gre_spiral_2d.seq",
    "gre_stack_of_stars_3d.seq",
    "gre_stack_of_spirals_3d.seq",
    "mprage_stack_of_spirals_3d.seq",
    "se_propeller_2d.seq",
]

#: One canonical TR, so there is nothing to bound and the sum is the whole
#: answer.
SINGLE = ["epi_2d.seq", "bssfp_2d.seq"]

#: Wide enough to hold the corpus's dominant lines, so the comparison is not
#: made only where every sequence is quiet.
BANDS = [(500.0, 600.0, 0.0)]
MAX_FREQ_HZ = 3000.0


def _lines(collection, index, mode, tr_duration_s):
    """``(frequencies, A_eq per axis in mT/m)`` for one TR."""
    spectra = _calc_mech_resonances(
        collection,
        0,
        index,
        mode,
        target_resolution_hz=1.0 / tr_duration_s,
        max_freq_hz=MAX_FREQ_HZ,
        forbidden_bands=BANDS,
    )
    amps = np.stack(
        [np.asarray(spectra[f"analytical_peak_amp_g{a}"], float) for a in "xyz"],
        axis=-1,
    )
    return np.asarray(spectra["analytical_peak_freqs"], float), (
        amps / GAMMA_HZ_PER_MT_PER_M
    )


def _structure(name):
    import pulserver.io as pio

    sequence = pio.read(CORPUS / name)
    return sequence._structure_for("bound")


@pytest.mark.parametrize("name", REPEATED, ids=lambda n: n[:-4])
def test_the_canonical_tr_bounds_every_instance_it_stands_for(name):
    structure = _structure(name)
    _, bound = _lines(structure.collection, 0, BOUND, structure.tr_duration)

    for index in range(structure.num_trs):
        _, played = _lines(structure.collection, index, ACTUAL, structure.tr_duration)
        assert np.all(played <= bound + 1e-6), f"{name} instance {index}"


@pytest.mark.parametrize("name", SINGLE, ids=lambda n: n[:-4])
def test_a_sequence_of_one_tr_is_summed_coherently_and_nothing_is_added(name):
    """Nothing varies, so the bound has no term of its own to add.

    This is what keeps the bound from being a blanket margin: it appears
    only where the instances actually differ.
    """
    structure = _structure(name)
    _, bound = _lines(structure.collection, 0, BOUND, structure.tr_duration)
    _, played = _lines(structure.collection, 0, ACTUAL, structure.tr_duration)

    np.testing.assert_array_equal(bound, played)


def test_a_multishot_scan_reads_the_same_however_its_arms_are_encoded():
    """A rotation extension and a written-out arm are the same gradient.

    The arms of a spiral can be one waveform under a rotation or one
    waveform each; a scanner plays the identical field either way, so the
    acoustic verdict may not depend on which the author wrote.
    """
    from pulserver.app import gre_spiral2D_sequence

    built = {}
    for rotated in (True, False):
        sequence = gre_spiral2D_sequence.main(
            n_x=32,
            n_arms=3,
            n_dummy=0,
            tr=20e-3,
            readout_bandwidth_hz=125e3,
            angle_scheme="uniform",
            use_rotation_ext=rotated,
        )
        structure = sequence._structure_for("bound")
        built[rotated] = [
            _lines(structure.collection, 0, BOUND, structure.tr_duration)[1],
            *(
                _lines(structure.collection, arm, ACTUAL, structure.tr_duration)[1]
                for arm in range(structure.num_trs)
            ),
        ]

    for turned, written in zip(built[True], built[False], strict=True):
        np.testing.assert_allclose(turned, written, rtol=1e-4, atol=1e-6)


def test_a_scan_with_more_encodes_than_are_enumerated_is_still_bounded():
    """Past the point where the combinations are worth listing one by one.

    A position that varies is bounded over the combinations it really plays
    while there are few of them, and by its largest amplitude per axis once
    there are many. Both are upper bounds; this holds the second one, which
    a 320-line phase encode is well past.
    """
    from pulserver.app import gre2D_sequence

    sequence = gre2D_sequence.main(
        plot=False, write_seq=False, n_x=64, n_y=320, n_dummy=0, tr=12e-3
    )
    structure = sequence._structure_for("bound")
    assert structure.num_trs > 256

    _, bound = _lines(structure.collection, 0, BOUND, structure.tr_duration)
    for index in range(0, structure.num_trs, 17):
        _, played = _lines(structure.collection, index, ACTUAL, structure.tr_duration)
        assert np.all(played <= bound + 1e-6), f"instance {index}"


def test_the_gate_refuses_a_band_that_only_a_later_instance_drives():
    """The instance that violates need not be the one the TR is walked at.

    The first instances of this GRE are dummies with no phase encode, so at
    2533 Hz they drive 0.11 mT/m while instance 40 -- a real encode -- drives
    0.15. A gate that read the first TR would pass a 0.13 mT/m band the scan
    goes on to break.
    """
    tolerance_mt_per_m = 0.13
    structure = _structure("gre_2d.seq")

    freqs, first = _lines(structure.collection, 0, ACTUAL, structure.tr_duration)
    played = np.stack(
        [
            _lines(structure.collection, index, ACTUAL, structure.tr_duration)[1]
            for index in range(structure.num_trs)
        ]
    )
    line = int(np.argmin(np.abs(freqs - 2533.0)))

    assert first[line].max() < tolerance_mt_per_m < played[:, line].max()

    with pytest.raises(RuntimeError, match="mech-res"):
        _check_safety(
            structure.collection,
            forbidden_bands=[
                (
                    freqs[line] - 10.0,
                    freqs[line] + 10.0,
                    tolerance_mt_per_m * GAMMA_HZ_PER_MT_PER_M,
                )
            ],
            skip_pns=True,
        )
