function meta = truth_parse_meta(path)
%TRUTH_PARSE_META Parse TruthBuilder _meta.txt file.

    fid = fopen(path, 'r');
    if fid < 0
        error('truth:io', 'Failed to open meta file: %s', path);
    end
    cleanup = onCleanup(@() fclose(fid));

    % Read all lines for line-by-line parsing (needed for segment_order).
    raw = {};
    while ~feof(fid)
        ln = fgetl(fid);
        if ischar(ln) && ~isempty(strtrim(ln))
            raw{end+1} = strtrim(ln); %#ok<AGROW>
        end
    end

    meta = struct();
    meta.num_unique_adcs = 0;
    meta.adc_samples = [];
    meta.adc_dwell_ns = [];
    meta.adc_anchor = [];
    meta.max_b1_subseq = 0;
    meta.tr_duration_us = 0;
    meta.num_segments = 0;
    meta.segment_num_blocks = [];
    meta.num_canonical_trs = 0;
    meta.canonical_mode = 'tr';
    meta.fmod_build_mode = 'full_collection';
    meta.canonical_duration_us = [];
    meta.num_freqmod_plan_probes = 0;
    meta.num_freqmod_plan_entries = 0;
    meta.num_labels = 0;
    meta.num_label_scan_rows = 0;
    meta.num_label_adc_rows = 0;
    meta.segment_order = [];

    for i = 1:numel(raw)
        parts = strsplit(raw{i});
        key = parts{1};

        if strcmp(key, 'segment_order')
            % Variable-length: segment_order 0 1 2 2 2 1
            meta.segment_order = cellfun(@str2double, parts(2:end));
            continue;
        end

        if strcmp(key, 'canonical_mode')
            if numel(parts) >= 2
                meta.canonical_mode = parts{2};
            end
            continue;
        end

        if strcmp(key, 'fmod_build_mode')
            if numel(parts) >= 2
                meta.fmod_build_mode = parts{2};
            end
            continue;
        end

        val = str2double(parts{2});

        if strcmp(key, 'num_unique_adcs')
            meta.num_unique_adcs = round(val);
        elseif strcmp(key, 'max_b1_subseq')
            meta.max_b1_subseq = round(val);
        elseif strcmp(key, 'tr_duration_us')
            meta.tr_duration_us = round(val);
        elseif strcmp(key, 'num_segments')
            meta.num_segments = round(val);
        elseif strcmp(key, 'num_canonical_trs')
            meta.num_canonical_trs = round(val);
        elseif strcmp(key, 'canonical_duration_us')
            meta.canonical_duration_us = round(val);
        elseif strcmp(key, 'num_freqmod_plan_probes')
            meta.num_freqmod_plan_probes = round(val);
        elseif strcmp(key, 'num_freqmod_plan_entries')
            meta.num_freqmod_plan_entries = round(val);
        elseif strcmp(key, 'num_labels')
            meta.num_labels = round(val);
        elseif strcmp(key, 'num_label_scan_rows')
            meta.num_label_scan_rows = round(val);
        elseif strcmp(key, 'num_label_adc_rows')
            meta.num_label_adc_rows = round(val);
        else
            tok = regexp(key, '^adc_(\d+)_(samples|dwell_ns|anchor)$', 'tokens', 'once');
            if ~isempty(tok)
                idx = str2double(tok{1}) + 1;
                suffix = tok{2};
                if strcmp(suffix, 'samples')
                    meta.adc_samples(idx) = round(val);
                elseif strcmp(suffix, 'anchor')
                    meta.adc_anchor(idx) = val;
                else
                    meta.adc_dwell_ns(idx) = round(val);
                end
                continue;
            end

            tok = regexp(key, '^segment_(\d+)_num_blocks$', 'tokens', 'once');
            if ~isempty(tok)
                idx = str2double(tok{1}) + 1;
                meta.segment_num_blocks(idx) = round(val);
            end
        end
    end

    if meta.num_unique_adcs ~= numel(meta.adc_samples)
        error('truth:consistency', ...
            'Meta mismatch in %s: num_unique_adcs=%d but adc_samples has %d entries.', ...
            path, meta.num_unique_adcs, numel(meta.adc_samples));
    end

    if meta.num_unique_adcs ~= numel(meta.adc_dwell_ns)
        error('truth:consistency', ...
            'Meta mismatch in %s: num_unique_adcs=%d but adc_dwell_ns has %d entries.', ...
            path, meta.num_unique_adcs, numel(meta.adc_dwell_ns));
    end

    if ~isempty(meta.adc_anchor) && meta.num_unique_adcs ~= numel(meta.adc_anchor)
        error('truth:consistency', ...
            'Meta mismatch in %s: num_unique_adcs=%d but adc_anchor has %d entries.', ...
            path, meta.num_unique_adcs, numel(meta.adc_anchor));
    end

    if isempty(meta.adc_anchor)
        meta.adc_anchor = 0.5 * ones(1, meta.num_unique_adcs);
    end

    if meta.num_segments ~= numel(meta.segment_num_blocks)
        error('truth:consistency', ...
            'Meta mismatch in %s: num_segments=%d but segment_num_blocks has %d entries.', ...
            path, meta.num_segments, numel(meta.segment_num_blocks));
    end
end
