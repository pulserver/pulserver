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
from pulserver.app import epi2D_sequence, epi3D_sequence

BANDWIDTH = 250e3


def design_2d(**kwargs):
    kwargs.setdefault("readout_bandwidth_hz", BANDWIDTH)
    return epi2D_sequence.main(n_x=32, n_y=16, **kwargs)


def design_3d(**kwargs):
    kwargs.setdefault("readout_bandwidth_hz", BANDWIDTH)
    return epi3D_sequence.main(n_x=32, n_y=16, n_z=4, slab_thickness=32e-3, **kwargs)


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


def test_the_multiband_epi_splits_calibration_from_imaging(monkeypatch):
    """With SMS_EXCITATION set, epi_2d becomes a linked collection: a
    calibration file (blip-nulled phase navigator NAV, then the low-resolution
    GRE calibration REF, a central block per slice) and an imaging file of
    blipped-CAIPI multiband shots (SMS, one per group) whose partition label
    walks the CAIPI slice-phase sawtooth. Neither file mixes the two, so no
    repeating unit is malformed."""
    monkeypatch.setattr(epi2D_sequence, "SMS_EXCITATION", True)
    n_slices, n_bands, n_y, n_acs = 9, 3, 15, 8
    common = {
        "n_x": 32,
        "n_y": n_y,
        "n_slices": n_slices,
        "slice_thickness": 3e-3,
        "n_bands": n_bands,
        "n_acs": n_acs,
        "readout_bandwidth_hz": 250e3,
    }
    main_seq = epi2D_sequence.main(**common)
    calibration = epi2D_sequence.SmsCalibrationKernel(
        pp.Opts(),
        fov=220e-3,
        slice_gap=0.0,
        flip_angle_deg=70.0,
        **common,
    )
    for seq in (main_seq, calibration):
        is_ok, error_report = seq.check_timing()
        assert is_ok, error_report
    assert main_seq.get_definition("MultibandFactor") == n_bands

    # Imaging file: multiband shots only, no calibration lines.
    main_labels = main_seq.evaluate_labels(evolution="adc")
    sms = np.asarray(main_labels["SMS"])
    par = np.asarray(main_labels["PAR"])
    assert int(sms.sum()) == (n_slices // n_bands) * n_y
    assert int(np.asarray(main_labels.get("REF", np.zeros(1))).sum()) == 0
    # the CAIPI slice-phase index on the multiband lines is the sawtooth.
    assert par[sms == 1][: 2 * n_bands].tolist() == [
        k % n_bands for k in range(2 * n_bands)
    ]

    # Calibration file: the navigator, then the GRE calibration block per slice.
    cal_labels = calibration.evaluate_labels(evolution="adc")
    n_acs_lines = len(pp.calc_calibration_lines(n_y, n_acs))
    assert int(np.asarray(cal_labels["NAV"]).sum()) == 3
    assert int(np.asarray(cal_labels["REF"]).sum()) == n_slices * n_acs_lines
    assert sorted(set(cal_labels["SLC"].tolist())) == list(range(n_slices))
    assert int(np.asarray(cal_labels.get("SMS", np.zeros(1))).sum()) == 0


def test_the_water_only_excitation_builds_a_valid_train(monkeypatch):
    """With SPSP_EXCITATION set, the slab pulse becomes spectral-spatial
    (water only): the train stays valid pulseq and, the pulse being longer,
    the repetition is longer than the plain slab excitation's."""
    plain = design_3d()
    monkeypatch.setattr(epi3D_sequence, "SPSP_EXCITATION", True)
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
    seq = epi3D_sequence.main(
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


def test_the_calibration_is_a_full_low_res_rectangle_marked_reference():
    """The autocalibration is its own low-resolution gradient echo, a fully
    sampled central ``n_acs x n_acs_z`` block whose lines all carry ``REF``
    (coil calibration, never imaging) at the ``calc_calibration_lines`` window
    the reconstruction reads."""
    n_y, n_z, n_acs, n_acs_z = 24, 8, 8, 4
    seq = epi3D_sequence.CalibrationKernel(
        n_x=48,
        n_y=n_y,
        n_z=n_z,
        slab_thickness=32e-3,
        n_acs=n_acs,
        n_acs_z=n_acs_z,
        readout_bandwidth_hz=180e3,
    )
    is_ok, error_report = seq.check_timing()
    assert is_ok, error_report
    labels = seq.evaluate_labels(evolution="adc")
    acs_y = pp.calc_calibration_lines(n_y, n_acs)
    acs_z = pp.calc_calibration_lines(n_z, n_acs_z)
    # every acquired line is a reference line, and they tile the whole rectangle.
    assert (np.asarray(labels["REF"]) == 1).all()
    sampled = set(zip(labels["LIN"].tolist(), labels["PAR"].tolist(), strict=True))
    assert sampled == {(y, z) for y in acs_y for z in acs_z}


def test_the_imaging_file_carries_no_calibration():
    """The main file is imaging only -- no ``REF`` lines -- so it is one clean
    repeating unit; the calibration lives in its own linked sequence."""
    seq = epi3D_sequence.main(
        n_x=48,
        n_y=24,
        n_z=8,
        slab_thickness=32e-3,
        acceleration=2,
        acceleration_z=2,
        n_acs=8,
        n_acs_z=4,
        readout_bandwidth_hz=180e3,
    )
    labels = seq.evaluate_labels(evolution="adc")
    assert int(np.asarray(labels.get("REF", np.zeros(1))).sum()) == 0


def test_the_2d_calibration_is_a_per_slice_low_res_gradient_echo():
    """The 2D autocalibration is its own slice-selective gradient echo: a fully
    sampled central ``n_acs`` block per slice, every line ``REF`` (coil
    calibration, never imaging) and carrying its ``SLC`` counter, at the
    ``calc_calibration_lines`` window the reconstruction reads."""
    n_slices, n_y, n_acs = 3, 24, 8
    seq = epi2D_sequence.CalibrationKernel(
        n_x=32,
        n_y=n_y,
        n_slices=n_slices,
        n_acs=n_acs,
        readout_bandwidth_hz=BANDWIDTH,
    )
    is_ok, error_report = seq.check_timing()
    assert is_ok, error_report
    labels = seq.evaluate_labels(evolution="adc")
    acs_y = pp.calc_calibration_lines(n_y, n_acs)
    assert (np.asarray(labels["REF"]) == 1).all()
    assert sorted(set(labels["SLC"].tolist())) == list(range(n_slices))
    # every slice acquires exactly the calibration window, and nothing else.
    for slc in range(n_slices):
        window = np.asarray(labels["LIN"])[np.asarray(labels["SLC"]) == slc]
        assert sorted(window.tolist()) == sorted(acs_y)


def test_the_accelerated_2d_epi_leads_with_a_calibration(tmp_path):
    """An accelerated non-SMS 2D EPI writes a three-file collection --
    calibration, then navigator, then main -- each linked to the next; the
    imaging file itself carries no ``REF`` lines."""
    path = tmp_path / "scan.seq"
    design_2d(
        n_slices=2, acceleration=2, n_acs=8, write_seq=True, seq_filename=str(path)
    )

    calib_path = path
    nav_path = tmp_path / "scan_navigator.seq"
    main_path = tmp_path / "scan_main.seq"
    assert calib_path.exists() and nav_path.exists() and main_path.exists()

    calib = pp.Sequence()
    calib.read(str(calib_path))
    assert calib.get_definition("Name") == "epi_2d_calibration"
    assert calib.get_definition("NextSequence") == nav_path.name

    nav = pp.Sequence()
    nav.read(str(nav_path))
    assert nav.get_definition("NextSequence") == main_path.name

    body = pp.Sequence()
    body.read(str(main_path))
    assert body.get_definition("NextSequence") is None
    labels = body.evaluate_labels(evolution="adc")
    assert int(np.asarray(labels.get("REF", np.zeros(1))).sum()) == 0


def test_the_fully_sampled_2d_epi_writes_no_calibration(tmp_path):
    """A fully sampled scan reconstructs from itself, so the chain is the plain
    navigator-then-main pair with no calibration file."""
    path = tmp_path / "scan.seq"
    design_2d(n_slices=2, acceleration=1, write_seq=True, seq_filename=str(path))
    assert path.exists() and (tmp_path / "scan_main.seq").exists()
    assert not (tmp_path / "scan_navigator.seq").exists()
    lead = pp.Sequence()
    lead.read(str(path))
    assert lead.get_definition("Name") == "epi_2d_navigator"


@pytest.mark.parametrize(
    "partial_fourier,partial_fourier_z", [(0.75, 1.0), (1.0, 0.75), (0.75, 0.75)]
)
def test_partial_fourier_drops_the_leading_edge(partial_fourier, partial_fourier_z):
    """Partial Fourier keeps the trailing fraction of each phase-encode axis and
    the centre, leaving the leading edge for conjugate symmetry."""
    n_y, n_z = 32, 8
    seq = epi3D_sequence.main(
        n_x=48,
        n_y=n_y,
        n_z=n_z,
        slab_thickness=32e-3,
        partial_fourier=partial_fourier,
        partial_fourier_z=partial_fourier_z,
        readout_bandwidth_hz=150e3,
    )
    labels = seq.evaluate_labels(evolution="adc")
    lin, par = np.asarray(labels["LIN"]), np.asarray(labels["PAR"])
    sampled = np.zeros((n_y, n_z), dtype=bool)
    sampled[lin, par] = True

    # The leading edge of each axis is dropped in proportion to the fraction,
    # and the centre is kept.
    assert lin.min() >= round((1.0 - partial_fourier) * n_y) - 1
    assert par.min() >= round((1.0 - partial_fourier_z) * n_z) - 1
    assert sampled[n_y // 2, n_z // 2]


def test_a_time_series_carries_its_repetition_counter():
    labels = design_2d(n_repetitions=3).evaluate_labels(evolution="adc")
    assert sorted(set(labels["REP"].tolist())) == [0, 1, 2]


@pytest.mark.parametrize(
    "module,build",
    [(epi2D_sequence, design_2d), (epi3D_sequence, design_3d)],
    ids=["2d", "3d"],
)
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
    nav = epi2D_sequence.NavigatorKernel(n_x=32, n_y=16, readout_bandwidth_hz=BANDWIDTH)
    labels = nav.evaluate_labels(evolution="adc")
    k_adc, *_ = nav.calculate_kspace(dense=False)
    n_samples = k_adc.shape[1] // len(labels["NAV"])
    ky = k_adc[1].reshape(-1, n_samples)
    nav_rows = [row for row, flag in enumerate(labels["NAV"].tolist()) if flag == 1]
    for row in nav_rows:
        assert np.abs(ky[row]).max() == pytest.approx(0.0, abs=1e-9)


def test_the_opposite_reference_walks_k_space_backwards():
    nav = epi2D_sequence.NavigatorKernel(n_x=32, n_y=16, readout_bandwidth_hz=BANDWIDTH)
    labels = nav.evaluate_labels(evolution="adc")
    k_adc, *_ = nav.calculate_kspace(dense=False)
    n_samples = k_adc.shape[1] // len(labels["SET"])
    ky = k_adc[1].reshape(-1, n_samples).mean(axis=1)
    reference = ky[np.asarray(labels["SET"]) == 1]
    assert reference[0] > reference[-1]


def test_the_default_protocol_is_feasible():
    system = pp.Opts()
    for module in (epi2D_sequence, epi3D_sequence):
        report = module.PLUGIN.validate_protocol(
            system, module.PLUGIN.get_default_protocol(system)
        )
        assert report["valid"] is True, report["info"]
        assert "ESP" in report["info"]
