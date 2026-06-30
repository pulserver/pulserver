function out = truth_parse_segment_def_bin(path)
%TRUTH_PARSE_SEGMENT_DEF_BIN Parse TruthBuilder _segment_def.bin file.

    fid = fopen(path, 'rb');
    if fid < 0
        error('truth:io', 'Failed to open segment definition file: %s', path);
    end
    cleanup = onCleanup(@() fclose(fid));

    out = struct();
    out.num_segments = read_scalar(fid, 'int32', path, 'num_segments');
    if out.num_segments < 0
        error('truth:format', 'Invalid num_segments=%d in %s', out.num_segments, path);
    end

    out.segments = repmat(struct( ...
        'num_blocks', 0, ...
        'blocks', [], ...
        'rf_adc_gap_us', -1, ...
        'adc_adc_gap_us', -1), out.num_segments, 1);

    for s = 1:out.num_segments
        nblk = read_scalar(fid, 'int32', path, sprintf('segments(%d).num_blocks', s));
        if nblk < 0
            error('truth:format', 'Invalid num_blocks=%d for segment %d in %s', nblk, s, path);
        end

        seg = struct();
        seg.num_blocks = nblk;
        seg.blocks = repmat(empty_block(), nblk, 1);

        for b = 1:nblk
            flags = read_scalar(fid, 'uint8', path, sprintf('segments(%d).blocks(%d).flags', s, b));
            blk = empty_block();
            blk.flags = uint8(flags);

            blk.has_rf = logical(bitget(flags, 1));
            blk.has_grad = logical([bitget(flags, 2), bitget(flags, 3), bitget(flags, 4)]);
            blk.has_adc = logical(bitget(flags, 5));
            blk.has_rotation = logical(bitget(flags, 6));
            blk.has_digital_out = logical(bitget(flags, 7));
            blk.has_freq_mod = logical(bitget(flags, 8));

            blk.rf_delay = read_scalar(fid, 'single', path, sprintf('seg%d_blk%d.rf_delay', s, b));
            blk.rf_amp = read_scalar(fid, 'single', path, sprintf('seg%d_blk%d.rf_amp', s, b));
            blk.rf_raster_us = read_scalar(fid, 'single', path, sprintf('seg%d_blk%d.rf_raster_us', s, b));
            blk.rf_n = read_scalar(fid, 'int32', path, sprintf('seg%d_blk%d.rf_n', s, b));
            blk.rf_rho = read_array(fid, blk.rf_n, 'single', path, sprintf('seg%d_blk%d.rf_rho', s, b));
            blk.rf_time_s = read_array(fid, blk.rf_n, 'single', path, sprintf('seg%d_blk%d.rf_time_s', s, b));

            for ax = 1:3
                blk.grad_delay(ax) = read_scalar(fid, 'single', path, sprintf('seg%d_blk%d.grad_delay(%d)', s, b, ax));
                blk.grad_amp(ax) = read_scalar(fid, 'single', path, sprintf('seg%d_blk%d.grad_amp(%d)', s, b, ax));
                blk.grad_n(ax) = read_scalar(fid, 'int32', path, sprintf('seg%d_blk%d.grad_n(%d)', s, b, ax));
                blk.grad_wave{ax} = read_array(fid, blk.grad_n(ax), 'single', path, sprintf('seg%d_blk%d.grad_wave(%d)', s, b, ax));
                blk.grad_time_s{ax} = read_array(fid, blk.grad_n(ax), 'single', path, sprintf('seg%d_blk%d.grad_time_s(%d)', s, b, ax));
            end

            blk.adc_delay = read_scalar(fid, 'single', path, sprintf('seg%d_blk%d.adc_delay', s, b));
            blk.adc_def_id = read_scalar(fid, 'int32', path, sprintf('seg%d_blk%d.adc_def_id', s, b));
            blk.digital_out_delay = read_scalar(fid, 'single', path, sprintf('seg%d_blk%d.digital_out_delay', s, b));
            blk.digital_out_duration = read_scalar(fid, 'single', path, sprintf('seg%d_blk%d.digital_out_duration', s, b));
            blk.freq_mod_num_samples = read_scalar(fid, 'int32', path, sprintf('seg%d_blk%d.freq_mod_num_samples', s, b));
            blk.freq_mod_def_id = read_scalar(fid, 'int32', path, sprintf('seg%d_blk%d.freq_mod_def_id', s, b));

            blk.rf_isocenter_us = read_scalar(fid, 'single', path, sprintf('seg%d_blk%d.rf_isocenter_us', s, b));
            blk.adc_kzero_us = read_scalar(fid, 'single', path, sprintf('seg%d_blk%d.adc_kzero_us', s, b));

            seg.blocks(b) = blk;
        end

        seg.rf_adc_gap_us = read_scalar(fid, 'single', path, sprintf('segments(%d).rf_adc_gap_us', s));
        seg.adc_adc_gap_us = read_scalar(fid, 'single', path, sprintf('segments(%d).adc_adc_gap_us', s));

        out.segments(s) = seg;
    end

    trailing = fread(fid, 1, '*uint8');
    if ~isempty(trailing)
        error('truth:format', 'Unexpected trailing bytes in %s', path);
    end
end

function blk = empty_block()
    blk = struct( ...
        'flags', uint8(0), ...
        'has_rf', false, ...
        'has_grad', false(1, 3), ...
        'has_adc', false, ...
        'has_rotation', false, ...
        'has_digital_out', false, ...
        'has_freq_mod', false, ...
        'rf_delay', single(0), ...
        'rf_amp', single(0), ...
        'rf_raster_us', single(1), ...
        'rf_n', int32(0), ...
        'rf_rho', single([]), ...
        'rf_time_s', single([]), ...
        'grad_delay', single(zeros(1, 3)), ...
        'grad_amp', single(zeros(1, 3)), ...
        'grad_n', int32(zeros(1, 3)), ...
        'grad_wave', {{single([]), single([]), single([])}}, ...
        'grad_time_s', {{single([]), single([]), single([])}}, ...
        'adc_delay', single(0), ...
        'adc_def_id', int32(-1), ...
        'digital_out_delay', single(0), ...
        'digital_out_duration', single(0), ...
        'freq_mod_num_samples', int32(0), ...
        'freq_mod_def_id', int32(-1), ...
        'rf_isocenter_us', single(-1), ...
        'adc_kzero_us', single(-1));
end

function v = read_scalar(fid, type, path, field)
    v = fread(fid, 1, ['*' type]);
    if isempty(v)
        error('truth:io', 'Unexpected EOF while reading %s (%s)', field, path);
    end
end

function a = read_array(fid, n, type, path, field)
    if n < 0
        error('truth:format', 'Invalid array length %d for %s (%s)', n, field, path);
    end
    if n == 0
        a = cast([], type_to_cast(type));
        return;
    end
    a = fread(fid, n, ['*' type]);
    if numel(a) ~= n
        error('truth:io', 'Unexpected EOF while reading %s (%s)', field, path);
    end
end

function t = type_to_cast(type)
    switch type
        case 'single'
            t = 'single';
        case 'int32'
            t = 'int32';
        case 'uint8'
            t = 'uint8';
        otherwise
            t = 'double';
    end
end
