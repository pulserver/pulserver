function truth_print_label_table(truth)
%TRUTH_PRINT_LABEL_TABLE Print ordered ADC label sequence to the terminal.
%
%   truth_print_label_table(truth)
%
%   For every ADC readout row the following is printed:
%     ADC# | ave=AVE | rot=rot_id | traj=traj_val | n_samp=NS | center=CI
%
%   ave      -- value of the "AVE" or "REP" label at this ADC row.
%   rot_id   -- 0 for identity rotation, else 1-based unique index
%              (first-appearance ordering).
%   traj_val -- value of the "SEG" or "SHT" label (= shot / trajectory
%              index); 0 if the label is absent.
%   n_samp   -- number of ADC samples for this readout (from meta).
%   center   -- 1-based sample index corresponding to k-space centre
%              (anchor fraction * (n_samp-1), rounded + 1).

    lbs  = truth.label_state;
    meta = truth.meta;
    st   = truth.scan_table;

    if lbs.adc_rows == 0
        adc_scan_idx = [];
        for i = 1:st.num_entries
            if double(st.entries(i).adc_flag) ~= 0
                adc_scan_idx(end+1, 1) = i - 1; %#ok<AGROW>
            end
        end
        if ~isempty(adc_scan_idx)
            lbs.adc_rows = numel(adc_scan_idx);
            lbs.adc_scan_row_idx = adc_scan_idx;
            lbs.adc_states = zeros(lbs.adc_rows, lbs.num_labels);
        end
    end

    if lbs.adc_rows == 0
        fprintf('\n=== Label table: %s -- no ADC rows ===\n', truth.base_name);
        return;
    end

    % -- Find label columns ---------------------------------------------------
    col_ave  = find_label_col(lbs, 'AVE', 'REP');
    col_traj = find_label_col(lbs, 'SEG', 'SHT');

    % -- Build unique rotation-matrix index (first-appearance) ----------------
    n_scan = st.num_entries;
    rot_ids = zeros(n_scan, 1);
    rot_list = zeros(0, 9);
    I3 = eye(3);  I3_flat = I3(:)';
    for i = 1:n_scan
        rm = double(st.entries(i).rotmat(:)');  % ensure 1x9 row
        if max(abs(rm - I3_flat)) < 1e-5
            rot_ids(i) = 0;
            continue;
        end
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

    % -- Build map: adc_def_id 1-based -> (n_samples, center_idx) ------------
    % Anchor sample index (0-based) = round(anchor_frac * (n_samp - 1))
    adc_n_samples  = meta.adc_samples(:);
    adc_center_idx = zeros(meta.num_unique_adcs, 1);
    for id1 = 1:meta.num_unique_adcs
        adc_center_idx(id1) = round(meta.adc_anchor(id1) * (meta.adc_samples(id1) - 1));
    end

    % -- Header ---------------------------------------------------------------
    fprintf('\n=== Label table: %s (%d ADC rows) ===\n', ...
        truth.base_name, lbs.adc_rows);
    fprintf('  %6s  %5s  %5s  %6s  %7s  %7s\n', ...
        'ADC#', 'ave', 'rot', 'traj', 'n_samp', 'center');
    sep = repmat('-', 1, 52);
    fprintf('%s\n', sep);

    % -- Row loop -------------------------------------------------------------
    for a = 1:lbs.adc_rows
        scan_row = lbs.adc_scan_row_idx(a) + 1;   % 0-based -> 1-based

        % Rotation id from scan table
        if scan_row >= 1 && scan_row <= n_scan
            rot_id = rot_ids(scan_row);
        else
            rot_id = 0;
        end

        % Label values
        if col_ave > 0
            ave_val = lbs.adc_states(a, col_ave);
        else
            ave_val = 0;
        end
        if col_traj > 0
            traj_val = lbs.adc_states(a, col_traj);
        else
            traj_val = 0;
        end

        % ADC def id from scan table entry
        adc_def_id1 = 1;   % default to first ADC def
        if scan_row >= 1 && scan_row <= n_scan
            flag = double(st.entries(scan_row).adc_flag);
            if flag > 0 && flag <= meta.num_unique_adcs
                adc_def_id1 = flag;
            end
        end
        n_samp = adc_n_samples(adc_def_id1);
        center_idx = adc_center_idx(adc_def_id1) + 1;

        fprintf('  %6d  %5d  %5d  %6d  %7d  %7d\n', ...
            a, ave_val, rot_id, traj_val, n_samp, center_idx);
    end
    fprintf('%s\n', sep);
end

% -- Helpers ------------------------------------------------------------------
function col = find_label_col(lbs, varargin)
    col = -1;
    for k = 1:lbs.num_labels
        for j = 1:numel(varargin)
            if strcmpi(lbs.label_names{k}, varargin{j})
                col = k;
                return;
            end
        end
    end
end
