function out = truth_parse_scan_table_bin(path)
%TRUTH_PARSE_SCAN_TABLE_BIN Parse TruthBuilder _scan_table.bin file.

    fid = fopen(path, 'rb');
    if fid < 0
        error('truth:io', 'Failed to open scan table file: %s', path);
    end
    cleanup = onCleanup(@() fclose(fid));

    out = struct();
    out.num_entries = read_scalar(fid, 'int32', path, 'num_entries');
    if out.num_entries < 0
        error('truth:format', 'Invalid num_entries=%d in %s', out.num_entries, path);
    end

    out.entries = repmat(empty_entry(), out.num_entries, 1);

    for i = 1:out.num_entries
        e = empty_entry();
        e.tr_start_flag = read_scalar(fid, 'int32', path, sprintf('entries(%d).tr_start_flag', i));
        e.segment_id = read_scalar(fid, 'int32', path, sprintf('entries(%d).segment_id', i));
        e.within_segment_idx = read_scalar(fid, 'int32', path, sprintf('entries(%d).within_segment_idx', i));
        e.rf_amp_hz = read_scalar(fid, 'single', path, sprintf('entries(%d).rf_amp_hz', i));
        e.rf_phase_rad = read_scalar(fid, 'single', path, sprintf('entries(%d).rf_phase_rad', i));
        e.rf_freq_hz = read_scalar(fid, 'single', path, sprintf('entries(%d).rf_freq_hz', i));
        e.gx_amp_hz_per_m = read_scalar(fid, 'single', path, sprintf('entries(%d).gx_amp_hz_per_m', i));
        e.gy_amp_hz_per_m = read_scalar(fid, 'single', path, sprintf('entries(%d).gy_amp_hz_per_m', i));
        e.gz_amp_hz_per_m = read_scalar(fid, 'single', path, sprintf('entries(%d).gz_amp_hz_per_m', i));
        e.adc_flag = read_scalar(fid, 'int32', path, sprintf('entries(%d).adc_flag', i));
        e.adc_phase_rad = read_scalar(fid, 'single', path, sprintf('entries(%d).adc_phase_rad', i));
        e.adc_freq_hz = read_scalar(fid, 'single', path, sprintf('entries(%d).adc_freq_hz', i));
        e.digitalout_flag = read_scalar(fid, 'int32', path, sprintf('entries(%d).digitalout_flag', i));
        e.trigger_flag = read_scalar(fid, 'int32', path, sprintf('entries(%d).trigger_flag', i));
        e.rotmat = read_array(fid, 9, 'single', path, sprintf('entries(%d).rotmat', i)).';
        e.freq_mod_id = read_scalar(fid, 'int32', path, sprintf('entries(%d).freq_mod_id', i));
        e.block_dur_us = read_scalar(fid, 'int32', path, sprintf('entries(%d).block_dur_us', i));
        out.entries(i) = e;
    end

    trailing = fread(fid, 1, '*uint8');
    if ~isempty(trailing)
        error('truth:format', 'Unexpected trailing bytes in %s', path);
    end
end

function e = empty_entry()
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

function v = read_scalar(fid, type, path, field)
    v = fread(fid, 1, ['*' type]);
    if isempty(v)
        error('truth:io', 'Unexpected EOF while reading %s (%s)', field, path);
    end
end

function a = read_array(fid, n, type, path, field)
    a = fread(fid, n, ['*' type]);
    if numel(a) ~= n
        error('truth:io', 'Unexpected EOF while reading %s (%s)', field, path);
    end
end
