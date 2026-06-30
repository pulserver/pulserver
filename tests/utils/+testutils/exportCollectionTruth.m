function exportCollectionTruth(out_dir, base_name, truths)
%EXPORTCOLLECTIONTRUTH Merge single-sequence truth artifacts into one collection case.

    if nargin < 3 || isempty(truths)
        error('truth:input', 'At least one truth case is required.');
    end
    if ~iscell(truths)
        truths = {truths};
    end

    merged = merge_truth_cases(truths);

    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    write_meta(fullfile(out_dir, [base_name '_meta.txt']), merged.meta);
    write_tr_waveforms(fullfile(out_dir, [base_name '_tr_waveform.truth']), merged.tr_waveforms);
    write_segment_defs(fullfile(out_dir, [base_name '_segment_def.truth']), merged.segment_def);
    write_freqmod_defs(fullfile(out_dir, [base_name '_freqmod_def.truth']), merged.freqmod_def);
    write_scan_table(fullfile(out_dir, [base_name '_scan_table.truth']), merged.scan_table);
    write_label_state(fullfile(out_dir, [base_name '_label_state.truth']), merged.label_state);
    write_freqmod_plan(fullfile(out_dir, [base_name '_freqmod_plan.truth']), merged.freqmod_plan);
end

function merged = merge_truth_cases(truths)
    merged = struct();
    merged.meta = merge_meta(truths);
    merged.tr_waveforms = merge_tr_waveforms(truths);
    merged.freqmod_def = merge_freqmod_defs(truths);
    merged.segment_def = merge_segment_defs(truths, merged.meta.num_unique_adcs, merged.freqmod_def.offsets);
    merged.scan_table = merge_scan_table(truths, merged.freqmod_def.offsets, compute_segment_offsets(truths));
    merged.label_state = merge_label_state(truths);
    merged.freqmod_plan = merge_freqmod_plan(truths, merged.freqmod_def.offsets);
end

function meta = merge_meta(truths)
    num_cases = numel(truths);
    adc_samples = [];
    adc_dwell_ns = [];
    adc_anchor = [];
    segment_num_blocks = [];
    segment_order = [];
    segment_offset = 0;
    num_canonical_trs = 0;
    total_scan_rows = 0;
    total_adc_rows = 0;
    num_plan_entries = 0;
    canonical_mode = 'tr';
    fmod_build_mode = 'full_collection';

    for i = 1:num_cases
        truth = truths{i};
        adc_samples = [adc_samples, reshape(truth.meta.adc_samples, 1, [])]; %#ok<AGROW>
        adc_dwell_ns = [adc_dwell_ns, reshape(truth.meta.adc_dwell_ns, 1, [])]; %#ok<AGROW>
        adc_anchor = [adc_anchor, reshape(truth.meta.adc_anchor, 1, [])]; %#ok<AGROW>
        segment_num_blocks = [segment_num_blocks, reshape(truth.meta.segment_num_blocks, 1, [])]; %#ok<AGROW>
        segment_order = [segment_order, segment_offset + reshape(truth.meta.segment_order, 1, [])]; %#ok<AGROW>
        segment_offset = segment_offset + truth.meta.num_segments;
        num_canonical_trs = num_canonical_trs + truth.meta.num_canonical_trs;
        total_scan_rows = total_scan_rows + truth.label_state.scan_rows;
        total_adc_rows = total_adc_rows + truth.label_state.adc_rows;
        if isfield(truth, 'freqmod_plan')
            num_plan_entries = num_plan_entries + truth.freqmod_plan.num_plans;
        end
        if isfield(truth.meta, 'canonical_mode') && strcmp(truth.meta.canonical_mode, 'pass_expanded')
            canonical_mode = 'pass_expanded';
        end
        if isfield(truth.meta, 'fmod_build_mode') && strcmp(truth.meta.fmod_build_mode, 'tr_scoped')
            fmod_build_mode = 'tr_scoped';
        end
    end

    label_names = collect_label_names(truths);

    meta = struct();
    meta.num_unique_adcs = numel(adc_samples);
    meta.adc_samples = adc_samples;
    meta.adc_dwell_ns = adc_dwell_ns;
    meta.adc_anchor = adc_anchor;
    meta.max_b1_subseq = 0;
    meta.tr_duration_us = sum(cellfun(@(t) double(t.meta.tr_duration_us), truths));
    meta.num_segments = numel(segment_num_blocks);
    meta.segment_num_blocks = segment_num_blocks;
    meta.num_canonical_trs = num_canonical_trs;
    meta.canonical_mode = canonical_mode;
    meta.fmod_build_mode = fmod_build_mode;
    meta.canonical_duration_us = first_canonical_duration(truths);
    meta.num_freqmod_plan_probes = merged_num_probes(truths);
    meta.num_freqmod_plan_entries = num_plan_entries;
    meta.num_labels = numel(label_names);
    meta.num_label_scan_rows = total_scan_rows;
    meta.num_label_adc_rows = total_adc_rows;
    meta.segment_order = segment_order;
    meta.num_subsequences = num_cases;
end

function out = merge_tr_waveforms(truths)
    waveforms = repmat(struct('num_samples', 0, 'time_us', [], 'gx', [], 'gy', [], 'gz', []), 0, 1);
    for i = 1:numel(truths)
        waveforms = [waveforms; truths{i}.tr_waveforms.waveforms(:)]; %#ok<AGROW>
    end
    out = struct('num_trs', numel(waveforms), 'waveforms', waveforms);
end

function out = merge_freqmod_defs(truths)
    defs = repmat(struct( ...
        'type', int32(0), ...
        'num_samples', int32(0), ...
        'raster_us', single(0), ...
        'duration_us', single(0), ...
        'ref_time_us', single(0), ...
        'ref_integral', single(zeros(1, 3)), ...
        'waveform', single(zeros(0, 3))), 0, 1);
    offsets = zeros(1, numel(truths));
    count = 0;

    for i = 1:numel(truths)
        offsets(i) = count;
        defs = [defs; truths{i}.freqmod_def.defs(:)]; %#ok<AGROW>
        count = count + truths{i}.freqmod_def.num_defs;
    end

    out = struct('num_defs', numel(defs), 'defs', defs, 'offsets', offsets);
end

function out = merge_segment_defs(truths, total_num_adcs, fmod_offsets)
    segments = repmat(struct('num_blocks', 0, 'blocks', [], 'rf_adc_gap_us', -1, 'adc_adc_gap_us', -1), 0, 1);
    adc_offset = 0;

    for i = 1:numel(truths)
        truth = truths{i};
        for s = 1:truth.segment_def.num_segments
            seg = truth.segment_def.segments(s);
            for b = 1:seg.num_blocks
                if seg.blocks(b).adc_def_id >= 0
                    seg.blocks(b).adc_def_id = int32(double(seg.blocks(b).adc_def_id) + adc_offset);
                end
                if seg.blocks(b).freq_mod_def_id >= 0
                    seg.blocks(b).freq_mod_def_id = int32(double(seg.blocks(b).freq_mod_def_id) + fmod_offsets(i));
                end
            end
            segments(end + 1, 1) = seg; %#ok<AGROW>
        end
        adc_offset = adc_offset + truth.meta.num_unique_adcs;
    end

    if adc_offset ~= total_num_adcs
        error('truth:consistency', 'ADC definition count mismatch while merging collection truth.');
    end

    out = struct('num_segments', numel(segments), 'segments', segments);
end

function out = merge_scan_table(truths, fmod_offsets, seg_offsets)
    entries = repmat(empty_scan_entry(), 0, 1);
    for i = 1:numel(truths)
        local_entries = truths{i}.scan_table.entries;
        for j = 1:numel(local_entries)
            if local_entries(j).freq_mod_id > 0
                local_entries(j).freq_mod_id = int32(double(local_entries(j).freq_mod_id) + fmod_offsets(i));
            end
            local_entries(j).segment_id = int32(double(local_entries(j).segment_id) + seg_offsets(i));
        end
        entries = [entries; local_entries(:)]; %#ok<AGROW>
    end
    out = struct('num_entries', numel(entries), 'entries', entries);
end

function offsets = compute_segment_offsets(truths)
    offsets = zeros(1, numel(truths));
    count = 0;
    for i = 1:numel(truths)
        offsets(i) = count;
        count = count + truths{i}.meta.num_segments;
    end
end

function out = merge_label_state(truths)
    label_names = collect_label_names(truths);
    nlabels = numel(label_names);
    scan_states = zeros(0, nlabels);
    adc_states = zeros(0, nlabels);
    adc_scan_row_idx = zeros(0, 1);
    adc_value_min = inf(1, nlabels);
    adc_value_max = -inf(1, nlabels);
    scan_row_offset = 0;

    for i = 1:numel(truths)
        labels = truths{i}.label_state;
        map = zeros(1, numel(labels.label_names));
        for j = 1:numel(labels.label_names)
            map(j) = find(strcmp(label_names, labels.label_names{j}), 1);
        end

        local_scan = zeros(labels.scan_rows, nlabels);
        if ~isempty(map) && ~isempty(labels.scan_states)
            local_scan(:, map) = labels.scan_states;
        end
        scan_states = [scan_states; local_scan]; %#ok<AGROW>

        local_adc = zeros(labels.adc_rows, nlabels);
        if ~isempty(map) && ~isempty(labels.adc_states)
            local_adc(:, map) = labels.adc_states;
            adc_value_min(map) = min(adc_value_min(map), labels.adc_value_min(:).');
            adc_value_max(map) = max(adc_value_max(map), labels.adc_value_max(:).');
        end
        adc_states = [adc_states; local_adc]; %#ok<AGROW>
        adc_scan_row_idx = [adc_scan_row_idx; labels.adc_scan_row_idx(:) + scan_row_offset]; %#ok<AGROW>
        scan_row_offset = scan_row_offset + labels.scan_rows;
    end

    adc_value_min(~isfinite(adc_value_min)) = 0;
    adc_value_max(~isfinite(adc_value_max)) = 0;

    out = struct( ...
        'num_labels', nlabels, ...
        'label_names', {label_names}, ...
        'scan_rows', size(scan_states, 1), ...
        'adc_rows', size(adc_states, 1), ...
        'scan_states', scan_states, ...
        'adc_scan_row_idx', adc_scan_row_idx, ...
        'adc_states', adc_states, ...
        'adc_value_min', adc_value_min, ...
        'adc_value_max', adc_value_max);
end

function out = merge_freqmod_plan(truths, fmod_offsets)
    probes = [];
    num_plans = 0;
    max_samples = 0;

    for i = 1:numel(truths)
        plan = truths{i}.freqmod_plan;
        if plan.num_probes > 0
            if isempty(probes)
                probes = plan.probes;
            elseif ~isequal(size(probes), size(plan.probes)) || any(abs(probes(:) - plan.probes(:)) > 1e-9)
                error('truth:consistency', 'Collection truth merge requires matching freqmod plan probes.');
            end
        end
        num_plans = num_plans + plan.num_plans;
        max_samples = max(max_samples, plan.max_samples);
    end

    if isempty(probes)
        out = struct('num_probes', 0, 'probes', [], 'num_plans', 0, ...
            'max_samples', 0, 'plans', [], 'scan_len', 0, 'scan_to_plan', []);
        return;
    end

    plan_template = struct( ...
        'def_id', 0, ...
        'num_samples', 0, ...
        'rotmat', zeros(3, 3), ...
        'phase_active', zeros(1, size(probes, 1)), ...
        'phase_extra', zeros(1, size(probes, 1)), ...
        'phase_total', zeros(1, size(probes, 1)), ...
        'waveforms', zeros(size(probes, 1), max_samples));
    plans = repmat(plan_template, num_plans, 1);
    scan_to_plan = zeros(0, 1);
    plan_offset = 0;

    for i = 1:numel(truths)
        plan = truths{i}.freqmod_plan;
        for j = 1:plan.num_plans
            merged_plan = plan_template;
            merged_plan.def_id = plan.plans(j).def_id;
            if merged_plan.def_id > 0
                merged_plan.def_id = merged_plan.def_id + fmod_offsets(i);
            end
            merged_plan.num_samples = plan.plans(j).num_samples;
            merged_plan.rotmat = plan.plans(j).rotmat;
            merged_plan.phase_active = plan.plans(j).phase_active;
            merged_plan.phase_extra = plan.plans(j).phase_extra;
            merged_plan.phase_total = plan.plans(j).phase_total;
            merged_plan.waveforms(:, 1:plan.max_samples) = plan.plans(j).waveforms;
            plans(plan_offset + j) = merged_plan;
        end

        local_scan_to_plan = plan.scan_to_plan(:);
        if ~isempty(local_scan_to_plan)
            mask = local_scan_to_plan >= 0;
            local_scan_to_plan(mask) = local_scan_to_plan(mask) + plan_offset;
            scan_to_plan = [scan_to_plan; local_scan_to_plan]; %#ok<AGROW>
        end
        plan_offset = plan_offset + plan.num_plans;
    end

    out = struct('num_probes', size(probes, 1), 'probes', probes, ...
        'num_plans', num_plans, 'max_samples', max_samples, ...
        'plans', plans, 'scan_len', numel(scan_to_plan), 'scan_to_plan', scan_to_plan);
end

function names = collect_label_names(truths)
    names = {};
    for i = 1:numel(truths)
        for j = 1:numel(truths{i}.label_state.label_names)
            name = truths{i}.label_state.label_names{j};
            if ~any(strcmp(names, name))
                names{end + 1} = name; %#ok<AGROW>
            end
        end
    end
end

function value = first_canonical_duration(truths)
    value = 0;
    for i = 1:numel(truths)
        if isfield(truths{i}.meta, 'canonical_duration_us') && ~isempty(truths{i}.meta.canonical_duration_us)
            value = truths{i}.meta.canonical_duration_us;
            return;
        end
    end
end

function value = merged_num_probes(truths)
    value = 0;
    for i = 1:numel(truths)
        if isfield(truths{i}, 'freqmod_plan')
            value = max(value, truths{i}.freqmod_plan.num_probes);
        end
    end
end

function e = empty_scan_entry()
    e = struct( ...
        'tr_start_flag', int32(0), ...
        'segment_id', int32(0), ...
        'within_segment_idx', int32(0), ...
        'rf_amp_hz', single(0), ...
        'rf_phase_rad', single(0), ...
        'rf_freq_hz', single(0), ...
        'gx_amp_hz_per_m', single(0), ...
        'gy_amp_hz_per_m', single(0), ...
        'gz_amp_hz_per_m', single(0), ...
        'adc_flag', int32(0), ...
        'adc_phase_rad', single(0), ...
        'adc_freq_hz', single(0), ...
        'digitalout_flag', int32(0), ...
        'trigger_flag', int32(0), ...
        'rotmat', single(zeros(1, 9)), ...
        'freq_mod_id', int32(0), ...
        'block_dur_us', int32(0));
end

function write_meta(path, meta)
    fid = fopen(path, 'w');
    if fid < 0
        error('truth:io', 'Failed to open %s for writing.', path);
    end
    cleanup = onCleanup(@() fclose(fid));

    fprintf(fid, 'num_unique_adcs %d\n', meta.num_unique_adcs);
    for a = 1:meta.num_unique_adcs
        fprintf(fid, 'adc_%d_samples %d\n', a - 1, meta.adc_samples(a));
        fprintf(fid, 'adc_%d_dwell_ns %d\n', a - 1, meta.adc_dwell_ns(a));
        fprintf(fid, 'adc_%d_anchor %.9g\n', a - 1, meta.adc_anchor(a));
    end
    fprintf(fid, 'max_b1_subseq %d\n', meta.max_b1_subseq);
    fprintf(fid, 'tr_duration_us %d\n', round(meta.tr_duration_us));
    fprintf(fid, 'num_segments %d\n', meta.num_segments);
    for s = 1:meta.num_segments
        fprintf(fid, 'segment_%d_num_blocks %d\n', s - 1, meta.segment_num_blocks(s));
    end
    fprintf(fid, 'num_canonical_trs %d\n', meta.num_canonical_trs);
    fprintf(fid, 'canonical_mode %s\n', meta.canonical_mode);
    fprintf(fid, 'fmod_build_mode %s\n', meta.fmod_build_mode);
    fprintf(fid, 'canonical_duration_us %d\n', round(meta.canonical_duration_us));
    fprintf(fid, 'num_freqmod_plan_probes %d\n', meta.num_freqmod_plan_probes);
    fprintf(fid, 'num_freqmod_plan_entries %d\n', meta.num_freqmod_plan_entries);
    fprintf(fid, 'num_labels %d\n', meta.num_labels);
    fprintf(fid, 'num_label_scan_rows %d\n', meta.num_label_scan_rows);
    fprintf(fid, 'num_label_adc_rows %d\n', meta.num_label_adc_rows);
    fprintf(fid, 'num_subsequences %d\n', meta.num_subsequences);
    fprintf(fid, 'segment_order');
    for k = 1:numel(meta.segment_order)
        fprintf(fid, ' %d', meta.segment_order(k));
    end
    fprintf(fid, '\n');
end

function write_tr_waveforms(path, tr)
    fid = fopen(path, 'wb');
    if fid < 0
        error('truth:io', 'Failed to open %s for writing.', path);
    end
    cleanup = onCleanup(@() fclose(fid));

    fwrite(fid, int32(tr.num_trs), 'int32');
    for i = 1:tr.num_trs
        fwrite(fid, int32(tr.waveforms(i).num_samples), 'int32');
        fwrite(fid, single(tr.waveforms(i).time_us), 'float32');
        fwrite(fid, single(tr.waveforms(i).gx), 'float32');
        fwrite(fid, single(tr.waveforms(i).gy), 'float32');
        fwrite(fid, single(tr.waveforms(i).gz), 'float32');
    end
end

function write_segment_defs(path, segdef)
    fid = fopen(path, 'wb');
    if fid < 0
        error('truth:io', 'Failed to open %s for writing.', path);
    end
    cleanup = onCleanup(@() fclose(fid));

    fwrite(fid, int32(segdef.num_segments), 'int32');
    for s = 1:segdef.num_segments
        seg = segdef.segments(s);
        fwrite(fid, int32(seg.num_blocks), 'int32');
        for b = 1:seg.num_blocks
            blk = seg.blocks(b);
            fwrite(fid, uint8(blk.flags), 'uint8');
            fwrite(fid, single(blk.rf_delay), 'float32');
            fwrite(fid, single(blk.rf_amp), 'float32');
            fwrite(fid, single(blk.rf_raster_us), 'float32');
            fwrite(fid, int32(blk.rf_n), 'int32');
            if blk.rf_n > 0
                fwrite(fid, single(blk.rf_rho), 'float32');
                fwrite(fid, single(blk.rf_time_s), 'float32');
            end
            for ax = 1:3
                fwrite(fid, single(blk.grad_delay(ax)), 'float32');
                fwrite(fid, single(blk.grad_amp(ax)), 'float32');
                fwrite(fid, int32(blk.grad_n(ax)), 'int32');
                if blk.grad_n(ax) > 0
                    fwrite(fid, single(blk.grad_wave{ax}), 'float32');
                    fwrite(fid, single(blk.grad_time_s{ax}), 'float32');
                end
            end
            fwrite(fid, single(blk.adc_delay), 'float32');
            fwrite(fid, int32(blk.adc_def_id), 'int32');
            fwrite(fid, single(blk.digital_out_delay), 'float32');
            fwrite(fid, single(blk.digital_out_duration), 'float32');
            fwrite(fid, int32(blk.freq_mod_num_samples), 'int32');
            fwrite(fid, int32(blk.freq_mod_def_id), 'int32');
            fwrite(fid, single(blk.rf_isocenter_us), 'float32');
            fwrite(fid, single(blk.adc_kzero_us), 'float32');
        end
        fwrite(fid, single(seg.rf_adc_gap_us), 'float32');
        fwrite(fid, single(seg.adc_adc_gap_us), 'float32');
    end
end

function write_freqmod_defs(path, fmod)
    fid = fopen(path, 'wb');
    if fid < 0
        error('truth:io', 'Failed to open %s for writing.', path);
    end
    cleanup = onCleanup(@() fclose(fid));

    fwrite(fid, int32(fmod.num_defs), 'int32');
    for i = 1:fmod.num_defs
        def = fmod.defs(i);
        fwrite(fid, int32(def.type), 'int32');
        fwrite(fid, int32(def.num_samples), 'int32');
        fwrite(fid, single(def.raster_us), 'float32');
        fwrite(fid, single(def.duration_us), 'float32');
        fwrite(fid, single(def.ref_time_us), 'float32');
        fwrite(fid, single(def.ref_integral), 'float32');
        fwrite(fid, single(def.waveform(:, 1)), 'float32');
        fwrite(fid, single(def.waveform(:, 2)), 'float32');
        fwrite(fid, single(def.waveform(:, 3)), 'float32');
    end
end

function write_scan_table(path, scan)
    fid = fopen(path, 'wb');
    if fid < 0
        error('truth:io', 'Failed to open %s for writing.', path);
    end
    cleanup = onCleanup(@() fclose(fid));

    fwrite(fid, int32(scan.num_entries), 'int32');
    for i = 1:scan.num_entries
        e = scan.entries(i);
        fwrite(fid, int32(e.tr_start_flag), 'int32');
        fwrite(fid, int32(e.segment_id), 'int32');
        fwrite(fid, int32(e.within_segment_idx), 'int32');
        fwrite(fid, single(e.rf_amp_hz), 'float32');
        fwrite(fid, single(e.rf_phase_rad), 'float32');
        fwrite(fid, single(e.rf_freq_hz), 'float32');
        fwrite(fid, single(e.gx_amp_hz_per_m), 'float32');
        fwrite(fid, single(e.gy_amp_hz_per_m), 'float32');
        fwrite(fid, single(e.gz_amp_hz_per_m), 'float32');
        fwrite(fid, int32(e.adc_flag), 'int32');
        fwrite(fid, single(e.adc_phase_rad), 'float32');
        fwrite(fid, single(e.adc_freq_hz), 'float32');
        fwrite(fid, int32(e.digitalout_flag), 'int32');
        fwrite(fid, int32(e.trigger_flag), 'int32');
        fwrite(fid, single(e.rotmat), 'float32');
        fwrite(fid, int32(e.freq_mod_id), 'int32');
        fwrite(fid, int32(e.block_dur_us), 'int32');
    end
end

function write_label_state(path, labels)
    fid = fopen(path, 'wb');
    if fid < 0
        error('truth:io', 'Failed to open %s for writing.', path);
    end
    cleanup = onCleanup(@() fclose(fid));

    fwrite(fid, int32(labels.num_labels), 'int32');
    for i = 1:labels.num_labels
        bytes = uint8(labels.label_names{i});
        fwrite(fid, int32(numel(bytes)), 'int32');
        fwrite(fid, bytes, 'uint8');
    end

    fwrite(fid, int32(labels.scan_rows), 'int32');
    fwrite(fid, int32(labels.adc_rows), 'int32');
    if labels.num_labels > 0 && labels.scan_rows > 0
        fwrite(fid, int32(labels.scan_states.'), 'int32');
    end
    if labels.adc_rows > 0
        fwrite(fid, int32(labels.adc_scan_row_idx(:)), 'int32');
    end
    if labels.num_labels > 0 && labels.adc_rows > 0
        fwrite(fid, int32(labels.adc_states.'), 'int32');
        fwrite(fid, int32(labels.adc_value_min(:)), 'int32');
        fwrite(fid, int32(labels.adc_value_max(:)), 'int32');
    elseif labels.num_labels > 0
        fwrite(fid, int32(zeros(labels.num_labels, 1)), 'int32');
        fwrite(fid, int32(zeros(labels.num_labels, 1)), 'int32');
    end
end

function write_freqmod_plan(path, plan)
    fid = fopen(path, 'wb');
    if fid < 0
        error('truth:io', 'Failed to open %s for writing.', path);
    end
    cleanup = onCleanup(@() fclose(fid));

    fwrite(fid, int32(plan.num_probes), 'int32');
    if plan.num_probes > 0
        fwrite(fid, single(plan.probes(:)), 'float32');
    end

    fwrite(fid, int32(plan.num_plans), 'int32');
    fwrite(fid, int32(plan.max_samples), 'int32');
    for i = 1:plan.num_plans
        p = plan.plans(i);
        fwrite(fid, int32(p.def_id), 'int32');
        fwrite(fid, int32(p.num_samples), 'int32');
        fwrite(fid, single(p.rotmat.'), 'float32');
        fwrite(fid, single(p.phase_active), 'float32');
        fwrite(fid, single(p.phase_extra), 'float32');
        fwrite(fid, single(p.phase_total), 'float32');
        for q = 1:plan.num_probes
            fwrite(fid, single(p.waveforms(q, :)), 'float32');
        end
    end

    fwrite(fid, int32(plan.scan_len), 'int32');
    if plan.scan_len > 0
        fwrite(fid, int32(plan.scan_to_plan(:)), 'int32');
    end
end