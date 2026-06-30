function report = truth_report_case(base)
%TRUTH_REPORT_CASE Parse and print a terminal report for one truth case.

    truth = testutils.truth_parse_case(base);

    report = struct();
    report.base_name = truth.base_name;
    report.data_dir = truth.data_dir;
    report.num_unique_adcs = truth.meta.num_unique_adcs;
    report.num_segments = truth.meta.num_segments;
    report.num_canonical_trs = truth.meta.num_canonical_trs;
    report.canonical_mode = truth.meta.canonical_mode;
    report.canonical_duration_us = truth.meta.canonical_duration_us;
    report.num_scan_entries = truth.scan_table.num_entries;
    report.num_freqmod_defs = truth.freqmod_def.num_defs;
    report.num_freqmod_plan_entries = truth.freqmod_plan.num_plans;
    report.num_freqmod_plan_probes = truth.freqmod_plan.num_probes;
    report.num_labels = truth.label_state.num_labels;
    report.num_label_scan_rows = truth.label_state.scan_rows;
    report.num_label_adc_rows = truth.label_state.adc_rows;
    report.validation_ok = truth.validation.ok;
    report.validation_warnings = truth.validation.warnings;

    adc_rows = 0;
    rf_rows = 0;
    trig_rows = 0;
    digital_rows = 0;
    freq_rows = 0;
    for i = 1:truth.scan_table.num_entries
        e = truth.scan_table.entries(i);
        adc_rows = adc_rows + (e.adc_flag ~= 0);
        rf_rows = rf_rows + (abs(e.rf_amp_hz) > 0);
        trig_rows = trig_rows + (e.trigger_flag ~= 0);
        digital_rows = digital_rows + (e.digitalout_flag ~= 0);
        freq_rows = freq_rows + (e.freq_mod_id > 0);
    end

    fprintf('\n=== Truth Report: %s ===\n', truth.base_name);
    fprintf('Data dir: %s\n', truth.data_dir);
    fprintf('TR duration (meta): %d us\n', truth.meta.tr_duration_us);
    fprintf('Unique ADCs: %d\n', truth.meta.num_unique_adcs);

    for i = 1:truth.meta.num_unique_adcs
        fprintf('  ADC %d: samples=%d dwell_ns=%d\n', ...
            i - 1, truth.meta.adc_samples(i), truth.meta.adc_dwell_ns(i));
    end

    fprintf('Segments: %d\n', truth.meta.num_segments);
    for s = 1:truth.segment_def.num_segments
        fprintf('  Segment %d: blocks=%d rf_adc_gap_us=%.3f adc_adc_gap_us=%.3f\n', ...
            s - 1, ...
            truth.segment_def.segments(s).num_blocks, ...
            truth.segment_def.segments(s).rf_adc_gap_us, ...
            truth.segment_def.segments(s).adc_adc_gap_us);
    end

    fprintf('Canonical TRs: %d\n', truth.tr_waveforms.num_trs);
    fprintf('Canonical mode: %s\n', truth.meta.canonical_mode);
    if ~isempty(truth.meta.canonical_duration_us)
        fprintf('Canonical duration (meta): %d us\n', truth.meta.canonical_duration_us);
    end
    for i = 1:truth.tr_waveforms.num_trs
        w = truth.tr_waveforms.waveforms(i);
        fprintf('  TR %d: samples=%d t_end_us=%.3f max|g|=[%.3f %.3f %.3f]\n', ...
            i - 1, w.num_samples, w.time_us(end), ...
            max(abs(w.gx)), max(abs(w.gy)), max(abs(w.gz)));
    end

    fprintf('Freqmod defs: %d\n', truth.freqmod_def.num_defs);
    for d = 1:truth.freqmod_def.num_defs
        def = truth.freqmod_def.defs(d);
        fprintf('  Def %d: type=%d samples=%d raster_us=%.3f duration_us=%.3f\n', ...
            d, def.type, def.num_samples, def.raster_us, def.duration_us);
    end

    fprintf('Freqmod plan entries: %d (probes=%d)\n', ...
        truth.freqmod_plan.num_plans, truth.freqmod_plan.num_probes);
    for p = 1:truth.freqmod_plan.num_plans
        pe = truth.freqmod_plan.plans(p);
        fprintf('  Plan %d: def=%d ns=%d phase_total=[%s]\n', ...
            p - 1, pe.def_id, pe.num_samples, num2str(pe.phase_total, '%.6g '));
    end

    fprintf('Label states: labels=%d scan_rows=%d adc_rows=%d\n', ...
        truth.label_state.num_labels, truth.label_state.scan_rows, truth.label_state.adc_rows);
    for li = 1:truth.label_state.num_labels
        fprintf('  Label %s: adc_min=%d adc_max=%d (int32=[%d,%d])\n', ...
            truth.label_state.label_names{li}, ...
            round(truth.label_state.adc_value_min(li)), ...
            round(truth.label_state.adc_value_max(li)), ...
            intmin('int32'), intmax('int32'));
    end

    fprintf('Scan table rows: %d\n', truth.scan_table.num_entries);
    fprintf('  RF rows=%d ADC rows=%d FreqMod rows=%d Trigger rows=%d DigitalOut rows=%d\n', ...
        rf_rows, adc_rows, freq_rows, trig_rows, digital_rows);

    if truth.validation.ok
        fprintf('Validation: OK\n');
    else
        fprintf('Validation: WARNINGS (%d)\n', numel(truth.validation.warnings));
        for i = 1:numel(truth.validation.warnings)
            fprintf('  - %s\n', truth.validation.warnings{i});
        end
    end

    fprintf('================================\n\n');
end
