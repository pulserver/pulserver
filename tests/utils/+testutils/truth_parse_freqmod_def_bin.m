function out = truth_parse_freqmod_def_bin(path)
%TRUTH_PARSE_FREQMOD_DEF_BIN Parse TruthBuilder _freqmod_def.bin file.

    fid = fopen(path, 'rb');
    if fid < 0
        error('truth:io', 'Failed to open freqmod definition file: %s', path);
    end
    cleanup = onCleanup(@() fclose(fid));

    out = struct();
    out.num_defs = read_scalar(fid, 'int32', path, 'num_defs');
    if out.num_defs < 0
        error('truth:format', 'Invalid num_defs=%d in %s', out.num_defs, path);
    end

    out.defs = repmat(struct( ...
        'type', int32(0), ...
        'num_samples', int32(0), ...
        'raster_us', single(0), ...
        'duration_us', single(0), ...
        'ref_time_us', single(0), ...
        'ref_integral', single(zeros(1, 3)), ...
        'waveform', single(zeros(0, 3))), out.num_defs, 1);

    for d = 1:out.num_defs
        def = struct();
        def.type = read_scalar(fid, 'int32', path, sprintf('defs(%d).type', d));
        def.num_samples = read_scalar(fid, 'int32', path, sprintf('defs(%d).num_samples', d));
        if def.num_samples <= 0
            error('truth:format', 'Invalid num_samples=%d for def %d in %s', def.num_samples, d, path);
        end
        def.raster_us = read_scalar(fid, 'single', path, sprintf('defs(%d).raster_us', d));
        def.duration_us = read_scalar(fid, 'single', path, sprintf('defs(%d).duration_us', d));
        def.ref_time_us = read_scalar(fid, 'single', path, sprintf('defs(%d).ref_time_us', d));
        def.ref_integral = read_array(fid, 3, 'single', path, sprintf('defs(%d).ref_integral', d)).';

        gx = read_array(fid, def.num_samples, 'single', path, sprintf('defs(%d).waveform(:,1)', d));
        gy = read_array(fid, def.num_samples, 'single', path, sprintf('defs(%d).waveform(:,2)', d));
        gz = read_array(fid, def.num_samples, 'single', path, sprintf('defs(%d).waveform(:,3)', d));
        def.waveform = [gx, gy, gz];

        out.defs(d) = def;
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
