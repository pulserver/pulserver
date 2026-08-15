"""The EPI modules of the sequence zoo.

What the family owes: valid trains with the blips on the ramps, ``REV`` on
every reversed line, counters that agree with k-space, and -- the export
this slot exists to exercise -- the navigator and main written as a
``NextSequence``-linked pair the interpreter's collection reader follows.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver.seqzoo import epi_2d, epi_3d

BANDWIDTH = 250e3


def design_2d(**kwargs):
    kwargs.setdefault("readout_bandwidth_hz", BANDWIDTH)
    return epi_2d.main(n_x=32, n_y=16, **kwargs)


def design_3d(**kwargs):
    kwargs.setdefault("readout_bandwidth_hz", BANDWIDTH)
    return epi_3d.main(n_x=32, n_y=16, n_z=4, slab_thickness=32e-3, **kwargs)


@pytest.mark.parametrize("design", [design_2d, design_3d], ids=["2d", "3d"])
def test_the_sequence_is_valid_pulseq(design):
    is_ok, error_report = design().check_timing()
    assert is_ok, error_report


def test_the_line_counter_agrees_with_where_the_line_actually_is():
    seq = design_2d()
    labels = seq.evaluate_labels(evolution="adc")
    k_adc, *_ = seq.calculate_kspace(dense=False)
    n_samples = k_adc.shape[1] // len(labels["LIN"])
    ky = k_adc[1].reshape(-1, n_samples).mean(axis=1)
    lines = np.rint(ky * 0.22).astype(int) + 8
    assert np.array_equal(lines, labels["LIN"])


@pytest.mark.parametrize("design", [design_2d, design_3d], ids=["2d", "3d"])
def test_every_other_line_is_marked_reversed(design):
    labels = design().evaluate_labels(evolution="adc")
    rev = labels["REV"].reshape(-1, 16)
    assert np.array_equal(rev[0], np.arange(16) % 2)


def test_a_3d_train_covers_every_partition():
    labels = design_3d().evaluate_labels(evolution="adc")
    assert sorted(set(labels["PAR"].tolist())) == [0, 1, 2, 3]


def test_the_multiband_epi_builds_a_valid_blipped_caipi_acquisition(monkeypatch):
    """With SMS_EXCITATION set, epi_2d becomes a multiband acquisition: a
    single-band reference pass (REF, one shot per slice), blip-nulled phase
    navigator (NAV), and blipped-CAIPI multiband shots (SMS, one per group)
    whose partition label walks the CAIPI slice-phase sawtooth."""
    monkeypatch.setattr(epi_2d, "SMS_EXCITATION", True)
    n_slices, n_bands, n_y = 9, 3, 15
    seq = epi_2d.main(
        n_x=32,
        n_y=n_y,
        n_slices=n_slices,
        slice_thickness=3e-3,
        n_bands=n_bands,
        readout_bandwidth_hz=250e3,
    )
    is_ok, error_report = seq.check_timing()
    assert is_ok, error_report
    assert seq.get_definition("MultibandFactor") == n_bands

    labels = seq.evaluate_labels(evolution="adc")
    sms = np.asarray(labels["SMS"])
    ref = np.asarray(labels["REF"])
    par = np.asarray(labels["PAR"])
    # single-band reference: one full train per slice; multiband: one per group.
    assert int(ref.sum()) == n_slices * n_y
    assert int(sms.sum()) == (n_slices // n_bands) * n_y
    # the CAIPI slice-phase index on the multiband lines is the sawtooth.
    assert par[sms == 1][: 2 * n_bands].tolist() == [k % n_bands for k in range(2 * n_bands)]


def test_the_water_only_excitation_builds_a_valid_train(monkeypatch):
    """With SPSP_EXCITATION set, the slab pulse becomes spectral-spatial
    (water only): the train stays valid pulseq and, the pulse being longer,
    the repetition is longer than the plain slab excitation's."""
    plain = design_3d()
    monkeypatch.setattr(epi_3d, "SPSP_EXCITATION", True)
    water = design_3d()

    is_ok, error_report = water.check_timing()
    assert is_ok, error_report
    assert water.duration()[0] > plain.duration()[0]


@pytest.mark.parametrize(
    "acceleration,acceleration_z,caipi_shift",
    [(1, 2, 1), (2, 2, 1), (1, 4, 1), (2, 2, 0)],
)
def test_blipped_caipi_tiles_the_caipirinha_lattice(
    acceleration, acceleration_z, caipi_shift
):
    """Above ``acceleration_z = 1`` each shot walks a CAIPI shell, and the
    ``n_z // Rz`` shells tile exactly the lattice ``make_caipirinha_mask``
    describes -- the built-in mask and the sequence cannot drift apart."""
    n_y, n_z = 24, 8
    seq = epi_3d.main(
        n_x=48,
        n_y=n_y,
        n_z=n_z,
        slab_thickness=32e-3,
        acceleration=acceleration,
        acceleration_z=acceleration_z,
        caipi_shift=caipi_shift,
        n_acs=0,  # isolate the imaging lattice from the ACS rectangle
        n_acs_z=0,
        readout_bandwidth_hz=180e3,
    )
    labels = seq.evaluate_labels(evolution="adc")
    sampled = np.zeros((n_y, n_z), dtype=bool)
    sampled[labels["LIN"], labels["PAR"]] = True
    expected = pp.make_caipirinha_mask(
        (n_y, n_z), acceleration, acceleration_z, delta=caipi_shift
    )
    assert np.array_equal(sampled, expected)


@pytest.mark.parametrize(
    "acceleration,acceleration_z", [(2, 1), (1, 2), (2, 2)]
)
def test_an_accelerated_scan_lays_down_a_full_calibration_rectangle(
    acceleration, acceleration_z
):
    """Whenever the imaging lattice is undersampled, a short linear train fills
    the central ACS rectangle -- the fully sampled block ``calc_calibration_lines``
    describes and :mod:`pulserver.reczoo.epi_3d` calibrates coil maps from."""
    n_y, n_z, n_acs, n_acs_z = 24, 8, 8, 4
    seq = epi_3d.main(
        n_x=48,
        n_y=n_y,
        n_z=n_z,
        slab_thickness=32e-3,
        acceleration=acceleration,
        acceleration_z=acceleration_z,
        caipi_shift=1,
        n_acs=n_acs,
        n_acs_z=n_acs_z,
        readout_bandwidth_hz=180e3,
    )
    labels = seq.evaluate_labels(evolution="adc")
    sampled = np.zeros((n_y, n_z), dtype=bool)
    sampled[labels["LIN"], labels["PAR"]] = True
    acs_y = pp.calc_calibration_lines(n_y, n_acs)
    acs_z = pp.calc_calibration_lines(n_z, n_acs_z)
    assert sampled[np.ix_(acs_y, acs_z)].all()


@pytest.mark.parametrize(
    "partial_fourier,partial_fourier_z", [(0.75, 1.0), (1.0, 0.75), (0.75, 0.75)]
)
def test_partial_fourier_drops_the_leading_edge(partial_fourier, partial_fourier_z):
    """Partial Fourier keeps the trailing fraction of each phase-encode axis and
    the centre, leaving the leading edge for conjugate symmetry -- and still
    lays down the calibration rectangle so the missing lines are fillable."""
    n_y, n_z, n_acs, n_acs_z = 32, 8, 8, 4
    seq = epi_3d.main(
        n_x=48,
        n_y=n_y,
        n_z=n_z,
        slab_thickness=32e-3,
        partial_fourier=partial_fourier,
        partial_fourier_z=partial_fourier_z,
        n_acs=n_acs,
        n_acs_z=n_acs_z,
        readout_bandwidth_hz=150e3,
    )
    labels = seq.evaluate_labels(evolution="adc")
    lin, par = np.asarray(labels["LIN"]), np.asarray(labels["PAR"])
    sampled = np.zeros((n_y, n_z), dtype=bool)
    sampled[lin, par] = True

    # The leading edge of each axis is dropped in proportion to the fraction,
    # the centre is kept, and the calibration rectangle is fully sampled.
    assert lin.min() >= round((1.0 - partial_fourier) * n_y) - 1
    assert par.min() >= round((1.0 - partial_fourier_z) * n_z) - 1
    assert sampled[n_y // 2, n_z // 2]
    acs_y = pp.calc_calibration_lines(n_y, n_acs)
    acs_z = pp.calc_calibration_lines(n_z, n_acs_z)
    assert sampled[np.ix_(acs_y, acs_z)].all()


def test_a_time_series_carries_its_repetition_counter():
    labels = design_2d(n_repetitions=3).evaluate_labels(evolution="adc")
    assert sorted(set(labels["REP"].tolist())) == [0, 1, 2]


@pytest.mark.parametrize("module,build", [(epi_2d, design_2d), (epi_3d, design_3d)], ids=["2d", "3d"])
def test_the_pair_is_written_linked_navigator_first(tmp_path, module, build):
    """The Sequence Collection contract: the navigator carries NextSequence,
    the main file sits beside it under that name."""
    path = tmp_path / "scan.seq"
    build(write_seq=True, seq_filename=str(path))

    main_path = tmp_path / "scan_main.seq"
    assert path.exists() and main_path.exists()

    lead = pp.Sequence()
    lead.read(str(path))
    assert lead.get_definition("NextSequence") == "scan_main.seq"

    labels = lead.evaluate_labels(evolution="adc")
    assert set(labels["NAV"].tolist()) <= {0, 1} and 1 in labels["NAV"].tolist()
    assert 1 in labels["SET"].tolist()

    body = pp.Sequence()
    body.read(str(main_path))
    assert body.get_definition("NextSequence") is None


def test_the_navigator_lines_are_blip_nulled():
    nav = epi_2d.navigator(n_x=32, n_y=16, readout_bandwidth_hz=BANDWIDTH)
    labels = nav.evaluate_labels(evolution="adc")
    k_adc, *_ = nav.calculate_kspace(dense=False)
    n_samples = k_adc.shape[1] // len(labels["NAV"])
    ky = k_adc[1].reshape(-1, n_samples)
    nav_rows = [row for row, flag in enumerate(labels["NAV"].tolist()) if flag == 1]
    for row in nav_rows:
        assert np.abs(ky[row]).max() == pytest.approx(0.0, abs=1e-9)


def test_the_opposite_reference_walks_k_space_backwards():
    nav = epi_2d.navigator(n_x=32, n_y=16, readout_bandwidth_hz=BANDWIDTH)
    labels = nav.evaluate_labels(evolution="adc")
    k_adc, *_ = nav.calculate_kspace(dense=False)
    n_samples = k_adc.shape[1] // len(labels["SET"])
    ky = k_adc[1].reshape(-1, n_samples).mean(axis=1)
    reference = ky[np.asarray(labels["SET"]) == 1]
    assert reference[0] > reference[-1]


def test_the_default_protocol_is_feasible():
    system = pp.Opts()
    for module in (epi_2d, epi_3d):
        report = module.PLUGIN.validate_protocol(
            system, module.PLUGIN.get_default_protocol(system)
        )
        assert report["valid"] is True, report["info"]
        assert "ESP" in report["info"]
