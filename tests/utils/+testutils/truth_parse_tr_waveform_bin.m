function out = truth_parse_tr_waveform_bin(path)
%TRUTH_PARSE_TR_WAVEFORM_BIN Parse TruthBuilder _tr_waveform.bin file.

    fid = fopen(path, 'rb');
    if fid < 0
        error('truth:io', 'Failed to open TR waveform file: %s', path);
    end
    cleanup = onCleanup(@() fclose(fid));

    out = struct();
    out.num_trs = read_scalar(fid, 'int32', path, 'num_trs');
    if out.num_trs < 0
        error('truth:format', 'Invalid num_trs=%d in %s', out.num_trs, path);
    end

    out.waveforms = repmat(struct( ...
        'num_samples', 0, ...
        'time_us', [], ...
        'gx', [], ...
        'gy', [], ...
        'gz', []), out.num_trs, 1);

    for i = 1:out.num_trs
        n = read_scalar(fid, 'int32', path, sprintf('waveforms(%d).num_samples', i));
        if n <= 0
            error('truth:format', 'Invalid num_samples=%d for TR %d in %s', n, i, path);
        end

        w = struct();
        w.num_samples = n;
        w.time_us = read_array(fid, n, 'single', path, sprintf('waveforms(%d).time_us', i));
        w.gx = read_array(fid, n, 'single', path, sprintf('waveforms(%d).gx', i));
        w.gy = read_array(fid, n, 'single', path, sprintf('waveforms(%d).gy', i));
        w.gz = read_array(fid, n, 'single', path, sprintf('waveforms(%d).gz', i));

        out.waveforms(i) = w;
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
