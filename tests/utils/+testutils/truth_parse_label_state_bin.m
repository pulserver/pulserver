function out = truth_parse_label_state_bin(path)
%TRUTH_PARSE_LABEL_STATE_BIN Parse supplemental _label_state.bin artifact.

    fid = fopen(path, 'rb');
    if fid < 0
        error('truth:io', 'Failed to open label state file: %s', path);
    end
    cleanup = onCleanup(@() fclose(fid));

    out = struct();
    out.num_labels = read_scalar(fid, 'int32', path, 'num_labels');
    out.label_names = cell(1, out.num_labels);
    for i = 1:out.num_labels
        nbytes = read_scalar(fid, 'int32', path, sprintf('label_names(%d).len', i));
        if nbytes < 0
            error('truth:format', 'Negative label name length in %s', path);
        end
        raw = read_array(fid, nbytes, 'uint8', path, sprintf('label_names(%d)', i));
        out.label_names{i} = char(raw(:)).';
    end

    out.scan_rows = read_scalar(fid, 'int32', path, 'scan_rows');
    out.adc_rows = read_scalar(fid, 'int32', path, 'adc_rows');

    out.scan_states = zeros(out.scan_rows, out.num_labels);
    if out.scan_rows > 0 && out.num_labels > 0
        vals = read_array(fid, out.scan_rows * out.num_labels, 'int32', path, 'scan_states');
        out.scan_states = reshape(double(vals), [out.num_labels, out.scan_rows]).';
    end

    out.adc_scan_row_idx = [];
    if out.adc_rows > 0
        out.adc_scan_row_idx = double(read_array(fid, out.adc_rows, 'int32', path, 'adc_scan_row_idx'));
    end

    out.adc_states = zeros(out.adc_rows, out.num_labels);
    if out.adc_rows > 0 && out.num_labels > 0
        vals = read_array(fid, out.adc_rows * out.num_labels, 'int32', path, 'adc_states');
        out.adc_states = reshape(double(vals), [out.num_labels, out.adc_rows]).';
    end

    out.adc_value_min = zeros(1, out.num_labels);
    out.adc_value_max = zeros(1, out.num_labels);
    if out.num_labels > 0
        out.adc_value_min = double(read_array(fid, out.num_labels, 'int32', path, 'adc_value_min')).';
        out.adc_value_max = double(read_array(fid, out.num_labels, 'int32', path, 'adc_value_max')).';
    end

    trailing = fread(fid, 1, '*uint8');
    if ~isempty(trailing)
        error('truth:format', 'Unexpected trailing bytes in %s', path);
    end
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
