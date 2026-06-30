function truth_print_scan_table(truth, varargin)
%TRUTH_PRINT_SCAN_TABLE Print a formatted scan table to the terminal.
%
%   truth_print_scan_table(truth)
%   truth_print_scan_table(truth, 'shift', [dx dy dz], 'adc_only', false)
%
%   Columns printed per row:
%     Row | RF: freq(Hz) phase(rad) amp(Hz) | Gx Gy Gz (Hz/m) | shot |
%     ADC: freq(Hz) phase(rad) | rot_id | fmod_id
%
%   If fmod_id > 0 and freqmod_plan data are present, four additional
%   columns show phase_total per probe direction (x / y / z / oblique).
%
%   If additionally a non-zero shift is supplied, per-axis scaling info
%   is printed:
%     dx dy dz (mm) | scale_x scale_y scale_z (rad/m) | prod_x prod_y prod_z (rad)
%   where prod_i = shift(i) * ref_integral(i).
%
%   Parameters
%   ----------
%   shift     : 1x3 double, isocenter offset in metres (default [0 0 0]).
%   adc_only  : logical, skip non-ADC rows (default false).

    p = inputParser;
    addParameter(p, 'shift',    [0 0 0], @(x) isnumeric(x) && numel(x) == 3);
    addParameter(p, 'adc_only', false,   @islogical);
    parse(p, varargin{:});

    shift    = double(p.Results.shift(:).');   % 1x3
    adc_only = p.Results.adc_only;

    st  = truth.scan_table;
    fp  = truth.freqmod_plan;
    fd  = truth.freqmod_def;
    lbs = truth.label_state;

    n = st.num_entries;

    % ── Unique rotation matrices (first-appearance) ────────────────────
    % rot_id=0 means identity (no rotation); others are 1-based unique rotations
    rot_ids = zeros(n, 1);
    rot_list = zeros(0, 9);
    
    % Identity matrix (tolerance 1e-5)
    I3 = eye(3);
    I3_flat = I3(:)';
    
    for i = 1:n
        rm = double(st.entries(i).rotmat(:)');     % ensure 1x9 row vector
        
        % Check if this is (approximately) identity
        if max(abs(rm - I3_flat)) < 1e-5
            rot_ids(i) = 0;  % rot_id=0 means identity/no rotation
            continue;
        end
        
        % Otherwise, find or create unique rotation ID
        found = false;
        for r = 1:size(rot_list, 1)
            if max(abs(rot_list(r,:) - rm)) < 1e-5
                rot_ids(i) = r;
                found = true;
                break;
            end
        end
        if ~found
            rot_list(end+1, :) = rm; %#ok<AGROW>
            rot_ids(i) = size(rot_list, 1);
        end
    end

    % ── Shot ID from label_state ("SEG" label, case-insensitive) ───────
    shot_ids = ones(n, 1);
    seg_col = find_label_col(lbs, 'SEG');
    if seg_col > 0 && lbs.scan_rows == n
        shot_ids = lbs.scan_states(:, seg_col) + 1;   % store 0-based, print 1-based
    end

    % ── Check freqmod_plan coverage ────────────────────────────────────
    has_plan = (fp.num_plans > 0) && ~isempty(fp.scan_to_plan) && ...
               (numel(fp.scan_to_plan) == n);
    print_shift = has_plan && (norm(shift) > 0) && (fd.num_defs > 0);

    % ── Header ─────────────────────────────────────────────────────────
    fprintf('\n=== Scan table: %s (%d rows%s) ===\n', ...
        truth.base_name, n, iif(adc_only, ', ADC only', ''));

    hdr_base = '  %5s  %10s %9s %9s   %10s %10s %10s   %4s   %10s %9s  %5s  %5s';
    fprintf([hdr_base '\n'], ...
        'Row', 'RF_freq', 'RF_phi', 'RF_amp', ...
        'Gx', 'Gy', 'Gz', 'shot', ...
        'ADC_freq', 'ADC_phi', 'rot', 'fmod');

    if has_plan
        fprintf('        %11s %11s %11s %11s\n', ...
            'p_x(rad)', 'p_y(rad)', 'p_z(rad)', 'p_obl(rad)');
    end
    if print_shift
        fprintf('        %10s %10s %10s  %11s %11s %11s  %11s %11s %11s\n', ...
            'dx(mm)', 'dy(mm)', 'dz(mm)', ...
            'sc_x(r/m)', 'sc_y(r/m)', 'sc_z(r/m)', ...
            'prod_x(r)', 'prod_y(r)', 'prod_z(r)');
    end

    sep = repmat('-', 1, 90 + iif(has_plan, 48, 0) + iif(print_shift, 108, 0));
    fprintf('%s\n', sep);

    % ── Row loop ───────────────────────────────────────────────────────
    for i = 1:n
        e = st.entries(i);
        if adc_only && (e.adc_flag == 0)
            continue;
        end

        fprintf(['  %5d  %10.3f %9.5f %9.3f   %10.3f %10.3f %10.3f   %4d' ...
                 '   %10.3f %9.5f  %5d  %5d\n'], ...
            i, e.rf_freq_hz, e.rf_phase_rad, e.rf_amp_hz, ...
            e.gx_amp_hz_per_m, e.gy_amp_hz_per_m, e.gz_amp_hz_per_m, ...
            shot_ids(i), ...
            e.adc_freq_hz, e.adc_phase_rad, ...
            rot_ids(i), e.freq_mod_id);

        if has_plan && (e.freq_mod_id > 0)
            pi_idx = fp.scan_to_plan(i) + 1;   % scan_to_plan is 0-based → 1-based
            if pi_idx >= 1 && pi_idx <= fp.num_plans
                pt = fp.plans(pi_idx).phase_total;   % 1 × num_probes
                % Pad to 4 columns in case num_probes < 4
                pt4 = [pt, zeros(1, max(0, 4 - numel(pt)))];
                fprintf('        %11.6f %11.6f %11.6f %11.6f\n', ...
                    pt4(1), pt4(2), pt4(3), pt4(4));
            else
                fprintf('        %11s %11s %11s %11s\n', 'n/a', 'n/a', 'n/a', 'n/a');
            end

            if print_shift
                fid_1 = double(e.freq_mod_id);   % 1-based
                if fid_1 >= 1 && fid_1 <= fd.num_defs
                    ri = double(fd.defs(fid_1).ref_integral);  % 1x3 rad/m
                    prod = shift .* ri;
                    fprintf('        %10.3f %10.3f %10.3f  %11.4f %11.4f %11.4f  %11.6f %11.6f %11.6f\n', ...
                        shift(1)*1e3, shift(2)*1e3, shift(3)*1e3, ...
                        ri(1), ri(2), ri(3), ...
                        prod(1), prod(2), prod(3));
                end
            end
        end
    end
    fprintf('%s\n', sep);
end

% ── Helpers ────────────────────────────────────────────────────────────
function col = find_label_col(lbs, name)
    col = -1;
    for k = 1:lbs.num_labels
        if strcmpi(lbs.label_names{k}, name)
            col = k;
            return;
        end
    end
end

function v = iif(cond, a, b)
    if cond; v = a; else; v = b; end
end
