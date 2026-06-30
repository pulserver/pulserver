function out = truth_parse_freqmod_plan_bin(path)
%TRUTH_PARSE_FREQMOD_PLAN_BIN Parse supplemental _freqmod_plan.bin artifact.

    fid = fopen(path, 'rb');
    if fid < 0
        error('truth:io', 'Failed to open supplemental freqmod plan file: %s', path);
    end
    cleanup = onCleanup(@() fclose(fid));

    out = struct();
    out.num_probes = read_scalar(fid, 'int32', path, 'num_probes');
    out.probes = zeros(out.num_probes, 3);
    if out.num_probes > 0
        probe_vals = read_array(fid, out.num_probes * 3, 'single', path, 'probes');
        out.probes = reshape(double(probe_vals), [3, out.num_probes]).';
    end

    out.num_plans = read_scalar(fid, 'int32', path, 'num_plans');
    out.max_samples = read_scalar(fid, 'int32', path, 'max_samples');

    plan_template = struct( ...
        'def_id', 0, ...
        'num_samples', 0, ...
        'rotmat', zeros(3,3), ...
        'phase_active', zeros(1, out.num_probes), ...
        'phase_extra', zeros(1, out.num_probes), ...
        'phase_total', zeros(1, out.num_probes), ...
        'waveforms', zeros(out.num_probes, out.max_samples));
    out.plans = repmat(plan_template, out.num_plans, 1);

    for i = 1:out.num_plans
        p = plan_template;
        p.def_id = read_scalar(fid, 'int32', path, sprintf('plans(%d).def_id', i));
        p.num_samples = read_scalar(fid, 'int32', path, sprintf('plans(%d).num_samples', i));
        rv = read_array(fid, 9, 'single', path, sprintf('plans(%d).rotmat', i));
        p.rotmat = reshape(double(rv), [3, 3]).';

        p.phase_active = double(read_array(fid, out.num_probes, 'single', path, sprintf('plans(%d).phase_active', i))).';
        p.phase_extra = double(read_array(fid, out.num_probes, 'single', path, sprintf('plans(%d).phase_extra', i))).';
        p.phase_total = double(read_array(fid, out.num_probes, 'single', path, sprintf('plans(%d).phase_total', i))).';

        for q = 1:out.num_probes
            wf = read_array(fid, out.max_samples, 'single', path, sprintf('plans(%d).waveforms(%d)', i, q));
            p.waveforms(q, :) = double(wf(:)).';
        end

        out.plans(i) = p;
    end

    out.scan_len = read_scalar(fid, 'int32', path, 'scan_len');
    if out.scan_len > 0
        out.scan_to_plan = double(read_array(fid, out.scan_len, 'int32', path, 'scan_to_plan'));
    else
        out.scan_to_plan = [];
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
