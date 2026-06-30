function truth = truth_parse_case(base)
%TRUTH_PARSE_CASE Parse all TruthBuilder outputs for one test case.
%
% base can be:
%   - basename (e.g. 'gre_2d_1sl_1avg')
%   - full path without suffix
%   - .seq path
%   - one of the exported artifact paths

    [data_dir, base_name] = normalize_case_path(base);

    truth = struct();
    truth.base_name = base_name;
    truth.data_dir = data_dir;

    truth.paths = struct( ...
        'meta', fullfile(data_dir, [base_name '_meta.txt']), ...
        'tr_waveform', fullfile(data_dir, [base_name '_tr_waveform.truth']), ...
        'segment_def', fullfile(data_dir, [base_name '_segment_def.truth']), ...
        'freqmod_def', fullfile(data_dir, [base_name '_freqmod_def.truth']), ...
        'freqmod_plan', fullfile(data_dir, [base_name '_freqmod_plan.truth']), ...
        'label_state', fullfile(data_dir, [base_name '_label_state.truth']), ...
        'scan_table', fullfile(data_dir, [base_name '_scan_table.truth']), ...
        'seq', fullfile(data_dir, [base_name '.seq']));

    required = {'meta', 'tr_waveform', 'segment_def', 'freqmod_def', 'scan_table'};
    for i = 1:numel(required)
        key = required{i};
        p = truth.paths.(key);
        if ~isfile(p)
            error('truth:io', 'Missing required truth artifact: %s', p);
        end
    end

    truth.meta = testutils.truth_parse_meta(truth.paths.meta);
    truth.tr_waveforms = testutils.truth_parse_tr_waveform_bin(truth.paths.tr_waveform);
    truth.segment_def = testutils.truth_parse_segment_def_bin(truth.paths.segment_def);
    truth.freqmod_def = testutils.truth_parse_freqmod_def_bin(truth.paths.freqmod_def);
    truth.scan_table = testutils.truth_parse_scan_table_bin(truth.paths.scan_table);
    if isfile(truth.paths.label_state)
        truth.label_state = testutils.truth_parse_label_state_bin(truth.paths.label_state);
    else
        truth.label_state = struct('num_labels', 0, 'label_names', {{}}, ...
            'scan_rows', 0, 'adc_rows', 0, 'scan_states', zeros(0, 0), ...
            'adc_scan_row_idx', [], 'adc_states', zeros(0, 0), ...
            'adc_value_min', zeros(1, 0), 'adc_value_max', zeros(1, 0));
    end
    if isfile(truth.paths.freqmod_plan)
        truth.freqmod_plan = testutils.truth_parse_freqmod_plan_bin(truth.paths.freqmod_plan);
    else
        truth.freqmod_plan = struct('num_probes', 0, 'probes', [], 'num_plans', 0, ...
            'max_samples', 0, 'plans', [], 'scan_len', 0, 'scan_to_plan', []);
    end

    truth = local_backfill_label_adc_rows(truth);

    truth.validation = local_validate(truth);
end

function truth = local_backfill_label_adc_rows(truth)
% Backfill legacy artifacts that stored no ADC rows when num_labels == 0.
    if ~isfield(truth, 'label_state') || ~isfield(truth, 'scan_table')
        return;
    end

    if truth.label_state.num_labels ~= 0
        return;
    end

    if truth.label_state.adc_rows > 0
        return;
    end

    adc_idx = zeros(0, 1);
    for i = 1:truth.scan_table.num_entries
        if truth.scan_table.entries(i).adc_flag ~= 0
            adc_idx(end+1, 1) = i - 1; %#ok<AGROW>
        end
    end

    truth.label_state.adc_rows = size(adc_idx, 1);
    truth.label_state.adc_scan_row_idx = adc_idx;
    truth.label_state.adc_states = zeros(truth.label_state.adc_rows, 0);
    truth.label_state.adc_value_min = zeros(1, 0);
    truth.label_state.adc_value_max = zeros(1, 0);

    if isfield(truth, 'meta') && truth.meta.num_label_adc_rows == 0
        truth.meta.num_label_adc_rows = truth.label_state.adc_rows;
    end
end

function [data_dir, base_name] = normalize_case_path(base)
    if nargin < 1 || isempty(base)
        error('truth:input', 'A base case name or path is required.');
    end

    if (exist('isstring', 'builtin') || exist('isstring', 'file')) && isstring(base)
        base = char(base);
    end

    [in_dir, in_name, in_ext] = fileparts(base);
    if isempty(in_dir)
        data_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'expected');
    else
        data_dir = in_dir;
    end

    if strcmp(in_ext, '.seq')
        base_name = in_name;
        return;
    end

    suffixes = {'_meta', '_tr_waveform', '_segment_def', '_freqmod_def', '_freqmod_plan', '_label_state', '_scan_table'};
    base_name = in_name;
    for i = 1:numel(suffixes)
        sfx = suffixes{i};
        n = length(sfx);
        if length(base_name) >= n && strcmp(base_name(end-n+1:end), sfx)
            base_name = base_name(1:(length(base_name) - n));
            return;
        end
    end

    if isempty(in_ext)
        base_name = in_name;
    else
        base_name = [in_name in_ext];
    end
end

function validation = local_validate(truth)
    warnings = {};

    if truth.meta.num_canonical_trs ~= truth.tr_waveforms.num_trs
        warnings{end + 1} = sprintf( ...
            'num_canonical_trs mismatch: meta=%d, tr_waveforms=%d', ...
            truth.meta.num_canonical_trs, truth.tr_waveforms.num_trs); %#ok<AGROW>
    end

    if isfield(truth.meta, 'canonical_mode') && strcmp(truth.meta.canonical_mode, 'pass_expanded')
        if truth.tr_waveforms.num_trs ~= 1
            warnings{end + 1} = sprintf( ...
                'pass_expanded mode expects one canonical waveform, got %d', ...
                truth.tr_waveforms.num_trs); %#ok<AGROW>
        end
    end

    if truth.meta.num_segments ~= truth.segment_def.num_segments
        warnings{end + 1} = sprintf( ...
            'num_segments mismatch: meta=%d, segment_def=%d', ...
            truth.meta.num_segments, truth.segment_def.num_segments); %#ok<AGROW>
    end

    if isfield(truth, 'freqmod_plan') && truth.freqmod_plan.num_plans > 0
        if truth.freqmod_plan.scan_len ~= truth.scan_table.num_entries
            warnings{end + 1} = sprintf( ...
                'freqmod plan scan length mismatch: plan=%d scan_table=%d', ...
                truth.freqmod_plan.scan_len, truth.scan_table.num_entries); %#ok<AGROW>
        end
        bad_plan = [];
        for i = 1:numel(truth.freqmod_plan.scan_to_plan)
            pid = truth.freqmod_plan.scan_to_plan(i);
            if pid < -1 || pid >= truth.freqmod_plan.num_plans
                bad_plan(end+1) = i; %#ok<AGROW>
            end
        end
        if ~isempty(bad_plan)
            warnings{end + 1} = sprintf( ...
                'freqmod plan has invalid scan_to_plan rows: %s', mat2str(bad_plan)); %#ok<AGROW>
        end
    end

    if isfield(truth, 'label_state')
        if truth.label_state.scan_rows ~= truth.scan_table.num_entries
            warnings{end + 1} = sprintf( ...
                'label_state scan rows mismatch: labels=%d scan_table=%d', ...
                truth.label_state.scan_rows, truth.scan_table.num_entries); %#ok<AGROW>
        end

        adc_rows = 0;
        for i = 1:truth.scan_table.num_entries
            adc_rows = adc_rows + (truth.scan_table.entries(i).adc_flag ~= 0);
        end
        if truth.label_state.adc_rows ~= adc_rows
            warnings{end + 1} = sprintf( ...
                'label_state ADC rows mismatch: labels=%d scan_table=%d', ...
                truth.label_state.adc_rows, adc_rows); %#ok<AGROW>
        end

        if truth.meta.num_labels ~= truth.label_state.num_labels
            warnings{end + 1} = sprintf( ...
                'num_labels mismatch: meta=%d label_state=%d', ...
                truth.meta.num_labels, truth.label_state.num_labels); %#ok<AGROW>
        end

        if truth.meta.num_label_scan_rows ~= truth.label_state.scan_rows
            warnings{end + 1} = sprintf( ...
                'num_label_scan_rows mismatch: meta=%d label_state=%d', ...
                truth.meta.num_label_scan_rows, truth.label_state.scan_rows); %#ok<AGROW>
        end

        if truth.meta.num_label_adc_rows ~= truth.label_state.adc_rows
            warnings{end + 1} = sprintf( ...
                'num_label_adc_rows mismatch: meta=%d label_state=%d', ...
                truth.meta.num_label_adc_rows, truth.label_state.adc_rows); %#ok<AGROW>
        end
    end

    nseg = min(numel(truth.meta.segment_num_blocks), truth.segment_def.num_segments);
    for s = 1:nseg
        expected = truth.meta.segment_num_blocks(s);
        got = truth.segment_def.segments(s).num_blocks;
        if expected ~= got
            warnings{end + 1} = sprintf( ...
                'segment %d block-count mismatch: meta=%d, segment_def=%d', ...
                s - 1, expected, got); %#ok<AGROW>
        end
    end

    bad_fmod = [];
    for i = 1:truth.scan_table.num_entries
        fid = truth.scan_table.entries(i).freq_mod_id;
        if fid < 0 || fid > truth.freqmod_def.num_defs
            bad_fmod(end + 1) = i; %#ok<AGROW>
        end
    end
    if ~isempty(bad_fmod)
        warnings{end + 1} = sprintf( ...
            'scan table has invalid freq_mod_id at rows: %s', ...
            mat2str(bad_fmod)); %#ok<AGROW>
    end

    validation = struct();
    validation.ok = isempty(warnings);
    validation.warnings = warnings;
end
