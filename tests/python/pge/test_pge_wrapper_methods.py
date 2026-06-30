import matplotlib.pyplot as plt
import numpy as np
import pytest

from pge import Opts, SequenceCollection
from pge.core._extension._pulseqlib_wrapper import _calc_mech_resonances
from pge.core._plot import plot as _plot_impl


def test_generated_sequences_wrapper_smoke(generated_seq_path):
    sc = SequenceCollection(str(generated_seq_path))
    report = sc.report()

    assert sc.num_sequences >= 1
    assert isinstance(report, list)
    assert len(report) >= 1
    assert report[0].num_blocks > 0


def test_grad_spectrum_api_smoke(generated_seq_path):
    sc = SequenceCollection(str(generated_seq_path))

    sc.grad_spectrum(
        sequence_idx=0,
        forbidden_bands=[(500.0, 600.0, 2000.0)],
        max_frequency=1200.0,
    )

    fig = plt.gcf()
    assert fig is not None
    # Single-panel: 1 main axis + 1 twin (echo spacing) = 2
    assert len(fig.axes) >= 1

    resonance_axes = [
        ax
        for ax in fig.axes
        if isinstance(ax.get_title(), str) and "Mechanical Resonances" in ax.get_title()
    ]
    assert len(resonance_axes) == 1

    plt.close(fig)


def test_grad_spectrum_peak_tuning_kwargs_smoke(generated_seq_path):
    sc = SequenceCollection(str(generated_seq_path))

    sc.grad_spectrum(
        sequence_idx=0,
        forbidden_bands=[(500.0, 600.0, 2000.0)],
        max_frequency=1200.0,
        peak_log10_threshold=1.8,
        peak_norm_scale=8.0,
        peak_eps=1e-20,
    )

    fig = plt.gcf()
    assert fig is not None
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_calc_mech_resonances_peak_default_parity(generated_seq_path):
    sc = SequenceCollection(str(generated_seq_path))
    forbidden = [(500.0, 600.0, 2000.0)]

    rd_default = _calc_mech_resonances(
        sc._cseq,
        subsequence_idx=0,
        target_resolution_hz=5.0,
        max_freq_hz=1200.0,
        forbidden_bands=forbidden,
    )

    rd_explicit = _calc_mech_resonances(
        sc._cseq,
        subsequence_idx=0,
        target_resolution_hz=5.0,
        max_freq_hz=1200.0,
        forbidden_bands=forbidden,
        peak_log10_threshold=2.25,
        peak_norm_scale=10.0,
        peak_eps=1e-30,
    )

    for key in (
        "spectrum_full_gx",
        "spectrum_full_gy",
        "spectrum_full_gz",
    ):
        np.testing.assert_allclose(
            np.asarray(rd_default[key], dtype=np.float32),
            np.asarray(rd_explicit[key], dtype=np.float32),
        )


def test_calc_mech_resonances_returns_candidate_grad_amps(generated_seq_path):
    sc = SequenceCollection(str(generated_seq_path))
    rd = _calc_mech_resonances(
        sc._cseq,
        subsequence_idx=0,
        target_resolution_hz=5.0,
        max_freq_hz=1200.0,
        forbidden_bands=[(500.0, 600.0, 2000.0)],
    )
    assert "candidate_grad_amps" in rd
    grad_amps = np.asarray(rd["candidate_grad_amps"], dtype=np.float32)
    freqs = np.asarray(rd.get("candidate_freqs", []), dtype=np.float32)
    assert len(grad_amps) == len(freqs)
    assert np.all(grad_amps >= 0.0)


def test_grad_spectrum_single_panel(generated_seq_path):
    sc = SequenceCollection(str(generated_seq_path))

    sc.grad_spectrum(
        sequence_idx=0,
        forbidden_bands=[(500.0, 600.0, 2000.0)],
        max_frequency=1200.0,
    )

    fig = plt.gcf()
    assert fig is not None
    resonance_axes = [
        ax
        for ax in fig.axes
        if isinstance(ax.get_title(), str) and "Mechanical Resonances" in ax.get_title()
    ]
    assert len(resonance_axes) == 1
    plt.close(fig)


def test_calculate_gradient_spectrum_alias(generated_seq_path):
    sc = SequenceCollection(str(generated_seq_path))

    sc.calculate_gradient_spectrum(
        sequence_idx=0,
        forbidden_bands=[(500.0, 600.0, 2000.0)],
        max_frequency=1200.0,
    )

    fig = plt.gcf()
    assert fig is not None
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_calculate_pns_alias(representative_generated_seq_path):
    sc = SequenceCollection(str(representative_generated_seq_path))

    sc.calculate_pns(
        sequence_idx=0,
        stim_threshold=23.4 / 0.333,
        decay_constant_us=360.0,
        threshold_percent=[80.0],
    )

    fig = plt.gcf()
    assert fig is not None
    assert len(fig.axes) >= 1
    plt.close(fig)


@pytest.mark.parametrize("threshold_percent", [80.0, [80.0, 100.0], (80.0, 100.0), []])
def test_pns_threshold_percent_variants(
    representative_generated_seq_path, threshold_percent
):
    sc = SequenceCollection(str(representative_generated_seq_path))

    sc.pns(
        sequence_idx=0,
        stim_threshold=23.4 / 0.333,
        decay_constant_us=360.0,
        threshold_percent=threshold_percent,
    )

    fig = plt.gcf()
    assert fig is not None
    assert len(fig.axes) >= 1

    expected_count = (
        1 if isinstance(threshold_percent, (int, float)) else len(threshold_percent)
    )
    labels = [ln.get_label() for ln in fig.axes[0].lines]
    threshold_labels = [
        lb for lb in labels if isinstance(lb, str) and lb.endswith("% threshold")
    ]
    assert len(threshold_labels) >= expected_count

    plt.close(fig)


def test_plot_accepts_pypulseq_without_fig(simple_gre_seq):
    sc = SequenceCollection(simple_gre_seq)
    _plot_impl(sc, subsequence_idx=0, tr_instance=0, time_unit="ms")

    fig = plt.gcf()
    assert fig is not None
    assert len(fig.axes) == 6  # (3, 2) subplot grid

    plt.close(fig)


def test_plot_sequence_collection_without_fig(simple_gre_seq):
    sc = SequenceCollection(simple_gre_seq)
    _plot_impl(sc, subsequence_idx=0, tr_instance=0, time_unit="ms")

    fig = plt.gcf()
    assert fig is not None
    assert len(fig.axes) == 6  # (3, 2) subplot grid

    plt.close(fig)


def test_plot_accepts_xml_without_fig(tmp_path):
    """plot() only accepts SequenceCollection; XML overlays belong in validate()."""
    xml_path = tmp_path / "wf.xml"
    xml_path.write_text(
        "<root>"
        "<gx>0 10 20 0.0 1.0 0.0</gx>"
        "<gy>0 10 20 0.0 0.5 0.0</gy>"
        "<gz>0 10 20 0.0 0.2 0.0</gz>"
        "<rf>0 10 20 0.0 0.1 0.0</rf>"
        "</root>",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="plot\\(\\) expects a SequenceCollection"):
        _plot_impl(str(xml_path), time_unit="us")


def test_validate_handles_unsorted_reference_times(simple_gre_seq, monkeypatch):
    import pge.core._validate as _validate_mod

    sc = SequenceCollection(simple_gre_seq)
    orig = _validate_mod._pypulseq_reference

    def _unsorted_reference(
        seq, sequence_idx, tr_idx, tr_info, num_averages=1, **kwargs
    ):
        ref = orig(seq, sequence_idx, tr_idx, tr_info, num_averages, **kwargs)
        out = {}
        for key, value in ref.items():
            if not isinstance(value, tuple) or len(value) != 2:
                out[key] = value
                continue
            t, a = value
            t_arr = np.asarray(t)
            a_arr = np.asarray(a)
            if t_arr.size > 1:
                idx = np.arange(t_arr.size)[::-1]
                out[key] = (t_arr[idx], a_arr[idx])
            else:
                out[key] = (t_arr, a_arr)
        return out

    monkeypatch.setattr(_validate_mod, "_pypulseq_reference", _unsorted_reference)

    result = sc.validate(subsequence_idx=0, do_plot=False)

    assert result is True


def test_validate_passes_for_supported_generated_sequences(
    validate_pass_seq_path, num_averages, capsys
):
    sc = SequenceCollection(str(validate_pass_seq_path), num_averages=num_averages)

    result = sc.validate()

    captured = capsys.readouterr()
    assert result is True
    assert "Validation passed." in captured.out


def test_validate_defaults_iterate_full_scope_for_multisubsequence(
    simple_gre_seq, capsys
):
    sc = SequenceCollection([simple_gre_seq, simple_gre_seq])
    assert sc.num_sequences == 2

    result = sc.validate(do_plot=False)

    captured = capsys.readouterr()
    assert result is True
    assert "Validation passed." in captured.out


def test_validate_plot_requires_subsequence_for_multisubsequence(simple_gre_seq):
    sc = SequenceCollection([simple_gre_seq, simple_gre_seq])
    assert sc.num_sequences == 2

    with pytest.raises(ValueError, match="subsequence_idx must be specified"):
        sc.validate(do_plot=True)


def test_validate_plot_requires_tr_for_multi_tr_subsequence(simple_gre_seq):
    sc = SequenceCollection(simple_gre_seq)

    with pytest.raises(ValueError, match="tr_instance must be specified"):
        sc.validate(do_plot=True, subsequence_idx=0)


def test_validate_plot_autoselects_single_subsequence(simple_gre_seq, capsys):
    sc = SequenceCollection(simple_gre_seq)

    result = sc.validate(do_plot=True, tr_instance=0)

    captured = capsys.readouterr()
    assert result is True
    assert "Validation passed." in captured.out
    fig = plt.gcf()
    assert fig is not None
    plt.close(fig)


def test_validate_plot_autoselects_single_tr_for_selected_subsequence(
    simple_gre_seq, monkeypatch, capsys
):
    from pge.core._extension import _pulseqlib_wrapper as _wrapper_mod

    sc = SequenceCollection(simple_gre_seq)

    tr_info = {
        "num_prep_trs": 0,
        "num_trs": 1,
        "num_cooldown_trs": 0,
        "num_prep_blocks": 0,
        "num_cooldown_blocks": 0,
        "tr_size": 1,
        "num_passes": 1,
        "degenerate_prep": True,
        "degenerate_cooldown": True,
        "tr_duration_us": 1000.0,
    }

    monkeypatch.setattr(_wrapper_mod, "_find_tr", lambda *args, **kwargs: tr_info)

    result = sc.validate(do_plot=True, subsequence_idx=0)

    captured = capsys.readouterr()
    assert result is True
    assert "Validation passed." in captured.out
    fig = plt.gcf()
    assert fig is not None
    plt.close(fig)


def test_validate_raises_on_failure(simple_gre_seq, monkeypatch):
    import pge.core._validate as _validate_mod

    sc = SequenceCollection(simple_gre_seq)
    orig = _validate_mod._pypulseq_reference

    def _bad_reference(seq, sequence_idx, tr_idx, tr_info, num_averages=1, **kwargs):
        ref = orig(seq, sequence_idx, tr_idx, tr_info, num_averages, **kwargs)
        out = {}
        for key, value in ref.items():
            if not isinstance(value, tuple) or len(value) != 2:
                out[key] = value
                continue
            t, a = value
            t_arr = np.asarray(t)
            a_arr = np.asarray(a)
            if t_arr.size > 0:
                out[key] = (t_arr + 500.0, a_arr)
            else:
                out[key] = (t_arr, a_arr)
        return out

    monkeypatch.setattr(_validate_mod, "_pypulseq_reference", _bad_reference)

    with pytest.raises(RuntimeError, match="Validation failed"):
        sc.validate(
            grad_atol=1e-9,
            rf_rms_percent=1e-9,
        )


def test_pns_uses_opts_defaults(representative_generated_seq_path, monkeypatch):
    opts = Opts(
        gamma=42.576e6,
        B0=3.0,
        max_grad=40.0,
        max_slew=150.0,
        chronaxie_us=360.0,
        rheobase=23.4,
        alpha=0.333,
    )
    sc = SequenceCollection(str(representative_generated_seq_path), system=opts)

    captured = {}

    def _fake_calc_pns(*args, **kwargs):
        captured.update(kwargs)
        return {
            "num_samples": 4,
            "slew_x": [0.0, 1.0, 0.5, 0.0],
            "slew_y": [0.0, 0.2, 0.1, 0.0],
            "slew_z": [0.0, 0.4, 0.2, 0.0],
        }

    import pge.core._pns as _pns_mod

    monkeypatch.setattr(_pns_mod, "_calc_pns", _fake_calc_pns)

    sc.pns(sequence_idx=0)

    assert np.isclose(captured["chronaxie_us"], 360.0)
    assert np.isclose(captured["rheobase"], 23.4 / 0.333)
    plt.close(plt.gcf())


def test_grad_spectrum_uses_opts_forbidden_bands_default(simple_gre_seq, monkeypatch):
    opts = Opts(
        gamma=float(simple_gre_seq.system.gamma),
        B0=float(simple_gre_seq.system.B0),
        max_grad=40.0,
        max_slew=150.0,
        forbidden_bands=[(100.0, 200.0, 1.0)],
        rf_raster_time=float(simple_gre_seq.system.rf_raster_time),
        grad_raster_time=float(simple_gre_seq.system.grad_raster_time),
        adc_raster_time=float(simple_gre_seq.system.adc_raster_time),
        block_duration_raster=float(simple_gre_seq.system.block_duration_raster),
    )
    sc = SequenceCollection(simple_gre_seq, system=opts)

    captured = {}

    def _fake_calc_acoustic(*args, **kwargs):
        captured.update(kwargs)
        return {
            "num_freq_bins": 4,
            "freq_min_hz": 0.0,
            "freq_spacing_hz": 50.0,
            "spectrum_full_gx": [0.0, 0.2, 0.3, 0.1],
            "spectrum_full_gy": [0.0, 0.2, 0.3, 0.1],
            "spectrum_full_gz": [0.0, 0.2, 0.3, 0.1],
            "num_instances": 1,
            "num_analytical_peaks": 4,
            "analytical_peak_freqs": [50.0, 100.0, 150.0, 200.0],
            "analytical_peak_amp_gx": [0.0, 0.1, 0.2, 0.1],
            "analytical_peak_amp_gy": [0.0, 0.1, 0.2, 0.1],
            "analytical_peak_amp_gz": [0.0, 0.1, 0.2, 0.1],
            "analytical_peak_widths_hz": [8.0, 8.0, 8.0, 8.0],
            "num_candidates": 0,
            "candidate_freqs": [],
            "candidate_amps_gx": [],
            "candidate_amps_gy": [],
            "candidate_amps_gz": [],
            "candidate_grad_amps": [],
            "candidate_violations": [],
            "candidate_num_contribs": [],
            "candidate_contrib_def_ids": [],
            "candidate_contrib_axes": [],
            "num_component_terms": 3,
            "component_freqs_hz": [100.0, 100.0, 100.0],
            "component_amps": [0.2, 0.2, 0.2],
            "component_phases_rad": [0.0, 0.0, 0.0],
            "component_widths_hz": [8.0, 8.0, 8.0],
            "component_axes": [0, 1, 2],
            "component_def_ids": [10, 11, 12],
            "component_contrib_ids": [0, 0, 0],
            "component_run_ids": [0, 0, 0],
            "num_surviving_freqs": 0,
            "surviving_freqs_hz": [],
        }

    import pge.core._mechresonance as _mr_mod

    monkeypatch.setattr(_mr_mod, "_calc_mech_resonances", _fake_calc_acoustic)

    sc.grad_spectrum(sequence_idx=0)

    bands = captured["forbidden_bands"]
    assert len(bands) == 1
    assert np.isclose(bands[0][0], 100.0)
    assert np.isclose(bands[0][1], 200.0)
    assert np.isclose(bands[0][2], 1.0e-3 * float(opts.gamma))
    plt.close(plt.gcf())


def test_grad_spectrum_explicit_bands_override_opts(simple_gre_seq, monkeypatch):
    opts = Opts(
        gamma=float(simple_gre_seq.system.gamma),
        B0=float(simple_gre_seq.system.B0),
        max_grad=40.0,
        max_slew=150.0,
        forbidden_bands=[(100.0, 200.0, 1.0)],
        rf_raster_time=float(simple_gre_seq.system.rf_raster_time),
        grad_raster_time=float(simple_gre_seq.system.grad_raster_time),
        adc_raster_time=float(simple_gre_seq.system.adc_raster_time),
        block_duration_raster=float(simple_gre_seq.system.block_duration_raster),
    )
    sc = SequenceCollection(simple_gre_seq, system=opts)

    captured = {}

    def _fake_calc_acoustic(*args, **kwargs):
        captured.update(kwargs)
        return {
            "num_freq_bins": 4,
            "freq_min_hz": 0.0,
            "freq_spacing_hz": 50.0,
            "spectrum_full_gx": [0.0, 0.2, 0.3, 0.1],
            "spectrum_full_gy": [0.0, 0.2, 0.3, 0.1],
            "spectrum_full_gz": [0.0, 0.2, 0.3, 0.1],
            "num_instances": 1,
            "num_analytical_peaks": 4,
            "analytical_peak_freqs": [50.0, 100.0, 150.0, 200.0],
            "analytical_peak_amp_gx": [0.0, 0.1, 0.2, 0.1],
            "analytical_peak_amp_gy": [0.0, 0.1, 0.2, 0.1],
            "analytical_peak_amp_gz": [0.0, 0.1, 0.2, 0.1],
            "analytical_peak_widths_hz": [8.0, 8.0, 8.0, 8.0],
            "num_candidates": 0,
            "candidate_freqs": [],
            "candidate_amps_gx": [],
            "candidate_amps_gy": [],
            "candidate_amps_gz": [],
            "candidate_grad_amps": [],
            "candidate_violations": [],
            "candidate_num_contribs": [],
            "candidate_contrib_def_ids": [],
            "candidate_contrib_axes": [],
            "num_component_terms": 3,
            "component_freqs_hz": [100.0, 100.0, 100.0],
            "component_amps": [0.2, 0.2, 0.2],
            "component_phases_rad": [0.0, 0.0, 0.0],
            "component_widths_hz": [8.0, 8.0, 8.0],
            "component_axes": [0, 1, 2],
            "component_def_ids": [10, 11, 12],
            "component_contrib_ids": [0, 0, 0],
            "component_run_ids": [0, 0, 0],
            "num_surviving_freqs": 0,
            "surviving_freqs_hz": [],
        }

    import pge.core._mechresonance as _mr_mod

    monkeypatch.setattr(_mr_mod, "_calc_mech_resonances", _fake_calc_acoustic)

    explicit = [(250.0, 300.0, 1234.0)]
    sc.grad_spectrum(sequence_idx=0, forbidden_bands=explicit)

    assert captured["forbidden_bands"] == explicit
    plt.close(plt.gcf())


def test_pns_iterates_all_subsequences_and_canonical_trs(
    representative_generated_seq_path,
    monkeypatch,
):
    sc = SequenceCollection(str(representative_generated_seq_path))

    calls = []

    def _fake_find_tr(*args, **kwargs):
        return {"num_canonical_trs": 2}

    def _fake_calc_pns(*args, **kwargs):
        calls.append(kwargs)
        return {
            "num_samples": 4,
            "slew_x": [0.0, 1.0, 0.5, 0.0],
            "slew_y": [0.0, 0.2, 0.1, 0.0],
            "slew_z": [0.0, 0.4, 0.2, 0.0],
        }

    import pge.core._pns as _pns_mod

    monkeypatch.setattr(_pns_mod, "_find_tr", _fake_find_tr)
    monkeypatch.setattr(_pns_mod, "_calc_pns", _fake_calc_pns)

    sc.pns(stim_threshold=120.0, decay_constant_us=360.0)

    expected_calls = sc.num_sequences * 2
    assert len(calls) == expected_calls
    assert {call["canonical_tr_idx"] for call in calls} == {0, 1}
    plt.close("all")


def test_grad_spectrum_iterates_all_subsequences_and_canonical_trs(
    representative_generated_seq_path,
    monkeypatch,
):
    sc = SequenceCollection(str(representative_generated_seq_path))

    calls = []

    def _fake_find_tr(*args, **kwargs):
        return {"num_canonical_trs": 2}

    def _fake_calc_acoustic(*args, **kwargs):
        calls.append(kwargs)
        return {
            "num_freq_bins": 4,
            "freq_min_hz": 0.0,
            "freq_spacing_hz": 50.0,
            "spectrum_full_gx": [0.0, 0.2, 0.3, 0.1],
            "spectrum_full_gy": [0.0, 0.2, 0.3, 0.1],
            "spectrum_full_gz": [0.0, 0.2, 0.3, 0.1],
            "num_instances": 1,
            "num_analytical_peaks": 4,
            "analytical_peak_freqs": [50.0, 100.0, 150.0, 200.0],
            "analytical_peak_amp_gx": [0.0, 0.1, 0.2, 0.1],
            "analytical_peak_amp_gy": [0.0, 0.1, 0.2, 0.1],
            "analytical_peak_amp_gz": [0.0, 0.1, 0.2, 0.1],
            "analytical_peak_widths_hz": [8.0, 8.0, 8.0, 8.0],
            "num_candidates": 0,
            "candidate_freqs": [],
            "candidate_amps_gx": [],
            "candidate_amps_gy": [],
            "candidate_amps_gz": [],
            "candidate_grad_amps": [],
            "candidate_violations": [],
            "candidate_num_contribs": [],
            "candidate_contrib_def_ids": [],
            "candidate_contrib_axes": [],
            "num_component_terms": 3,
            "component_freqs_hz": [100.0, 100.0, 100.0],
            "component_amps": [0.2, 0.2, 0.2],
            "component_phases_rad": [0.0, 0.0, 0.0],
            "component_widths_hz": [8.0, 8.0, 8.0],
            "component_axes": [0, 1, 2],
            "component_def_ids": [10, 11, 12],
            "component_contrib_ids": [0, 0, 0],
            "component_run_ids": [0, 0, 0],
            "num_surviving_freqs": 0,
            "surviving_freqs_hz": [],
        }

    import pge.core._mechresonance as _mr_mod

    monkeypatch.setattr(_mr_mod, "_find_tr", _fake_find_tr)
    monkeypatch.setattr(_mr_mod, "_calc_mech_resonances", _fake_calc_acoustic)

    sc.grad_spectrum(forbidden_bands=[(250.0, 300.0, 1234.0)])

    expected_calls = sc.num_sequences * 2
    assert len(calls) == expected_calls
    assert {call["canonical_tr_idx"] for call in calls} == {0, 1}
    plt.close("all")
