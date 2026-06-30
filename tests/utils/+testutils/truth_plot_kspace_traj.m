function truth_plot_kspace_traj(truth)
%TRUTH_PLOT_KSPACE_TRAJ Plot non-Cartesian k-space trajectories.
%
%   truth_plot_kspace_traj(truth)
%
%   Computes the k-space trajectory for each unique (ADC-def, gradient-
%   amplitude) combination found in the segment definitions.  Straight-line
%   trajectories (Cartesian readout lines, radial spokes) are detected and
%   excluded by default.  Exception: if a readout block carries an explicit
%   rotation extension event, its trajectory is shown even when straight.
%   If every trajectory is suppressed, no figure is created and an
%   informational message is printed.
%
%   K-space is expressed in cycles/m (Hzs/m, standard Pulseq convention).
%   Sample axis is centred at 0 (= k-space centre, i.e. adc_kzero_us).
%
%   Requires MATLAB R2017b or later (vecnorm) and R2018b or later (sgtitle).

    meta = truth.meta;
    sdef = truth.segment_def;

    if meta.num_unique_adcs == 0
        return;
    end

    % ---- Collect unique (adc_def_id 0-based, grad_amp[3]) blocks ----------
    % Two blocks are considered identical when they have the same ADC
    % definition ID and identical (bitwise, after double conversion)
    % gradient-amplitude triple.  This matches the library's keying.
    key_list = zeros(0, 4);   % rows: [adc_def_id0, amp_x, amp_y, amp_z]
    blk_list = {};
    key_has_rotation = false(0, 1);

    for s = 1:sdef.num_segments
        seg = sdef.segments(s);
        for b = 1:seg.num_blocks
            blk = seg.blocks(b);
            if ~blk.has_adc
                continue;
            end
            key = [double(blk.adc_def_id), double(blk.grad_amp(:)')];
            % Exact-match deduplication (same binary representation)
            is_dup = false;
            for ki = 1:size(key_list, 1)
                if key_list(ki,1) == key(1) && all(key_list(ki,2:4) == key(2:4))
                    is_dup = true;
                    break;
                end
            end
            if ~is_dup
                key_list(end+1, :) = key; %#ok<AGROW>
                blk_list{end+1}    = blk; %#ok<AGROW>
                key_has_rotation(end+1, 1) = logical(blk.has_rotation); %#ok<AGROW>
            elseif logical(blk.has_rotation)
                % Preserve explicit rotation-extension presence for this key.
                key_has_rotation(ki, 1) = true;
            end
        end
    end

    if isempty(key_list)
        return;
    end

    % ---- Compute k-space trajectory per unique block ------------------------------------
    n_keys      = size(key_list, 1);
    kspace      = cell(n_keys, 3);   % kspace{i, ax}: n_samp x 1 (cycles/m)
    n_samp_arr  = zeros(n_keys, 1);  % sample count per key
    anchor_idx_arr = zeros(n_keys, 1); % 0-based sample index of anchor point
    is_trivial  = true(n_keys, 1);   % straight-line flag

    G_TOL = 1e-3;   % Hz/m tolerance for gradient constancy check (Cartesian detection)

    for i = 1:n_keys
        blk     = blk_list{i};
        adc_id0 = double(blk.adc_def_id);
        adc_id1 = adc_id0 + 1;
        if adc_id1 < 1 || adc_id1 > meta.num_unique_adcs
            continue;
        end
        dwell_s  = double(meta.adc_dwell_ns(adc_id1)) * 1e-9;
        n_samp   = double(meta.adc_samples(adc_id1));
        if n_samp < 2 || dwell_s <= 0
            continue;
        end

        % ADC window bounds (in seconds, within the block)
        adc_start_s = double(blk.adc_delay);
        adc_end_s   = adc_start_s + n_samp * dwell_s;
        
        % Anchor point (k-space center) time and its sample index
        anchor_time_s = double(blk.adc_kzero_us) * 1e-6;
        anchor_idx    = round((anchor_time_s - adc_start_s) / dwell_s);
        anchor_idx    = max(0, min(n_samp - 1, anchor_idx));  % clamp to valid range
        
        n_samp_arr(i)      = n_samp;
        anchor_idx_arr(i) = anchor_idx;

        % Time of each ADC sample within the ADC window (absolute time in block)
        j_vec = (0:n_samp-1)';
        t_s   = adc_start_s + j_vec * dwell_s;

        k_traj = zeros(n_samp, 3);
        is_cartesian = true;
        
        for ax = 1:3
            gt = double(blk.grad_time_s{ax}(:));
            gw = double(blk.grad_wave{ax}(:));
            
            if numel(gt) < 1 || numel(gw) < 1
                kspace{i, ax} = zeros(n_samp, 1);
                continue;
            end
            
            % Apply gradient amplitude
            gw = gw .* double(blk.grad_amp(ax));
            
            % Resample gradient onto ADC dwell grid (within ADC window)
            g_at_t = interp1(gt, gw, t_s, 'linear', 0);
            
            % Check Cartesian: is gradient constant within ADC window?
            g_min = min(g_at_t);
            g_max = max(g_at_t);
            if (g_max - g_min) > G_TOL
                is_cartesian = false;
            end
            
            % Integrate to get k-space (cumsum of gradient * dwell)
            k_ax = cumsum(g_at_t) * dwell_s;           % cycles/m
            
            % Center at anchor point: subtract k at anchor index
            k_at_anchor = k_ax(anchor_idx + 1);        % +1 for 1-based indexing
            k_ax = k_ax - k_at_anchor;
            
            kspace{i, ax}  = k_ax;
            k_traj(:, ax)  = k_ax;
        end

        has_rotation_evt = false;
        if i <= numel(key_has_rotation)
            has_rotation_evt = key_has_rotation(i);
        end

        % Keep straight-line trajectories when they come from explicit
        % rotation-extension events (even if the effective matrix is identity).
        if is_cartesian && ~has_rotation_evt
            is_trivial(i) = true;
        else
            is_trivial(i) = false;
        end
    end

    % ---- If everything is straight-line, skip --------------------------------------------------
    if all(is_trivial)
        fprintf(['truth_plot_kspace_traj: all %d trajectory(ies) were suppressed ' ...
                 '(straight-line without explicit rotation event) - no figure created.\n'], n_keys);
        return;
    end

    % ---- Plot non-trivial trajectories ----------------------------------------------------------------
    non_trivial = find(~is_trivial);
    n_plot      = numel(non_trivial);

    figure('Name', sprintf('K-space trajectories: %s', truth.base_name), ...
           'NumberTitle', 'off');

    ax_names = {'kx', 'ky', 'kz'};

    for row = 1:3
        subplot(3, 1, row);
        hold on;
        for jj = 1:n_plot
            i        = non_trivial(jj);
            blk      = blk_list{i};
            adc_id1  = double(blk.adc_def_id) + 1;
            n_samp   = n_samp_arr(i);
            anchor_idx = anchor_idx_arr(i);
            samp_ax  = (0:n_samp-1) - anchor_idx;   % center at anchor

            k_ax = kspace{i, row};
            if isempty(k_ax) || numel(k_ax) ~= n_samp
                k_ax = zeros(n_samp, 1);
            end
            plot(samp_ax, k_ax(:)', ...
                'DisplayName', sprintf('traj %d  (adc=%d)', jj, adc_id1-1));
        end
        hold off;
        xlabel('Sample (0 = k-anchor)');
        ylabel('cycles/m');
        title(ax_names{row});
        if n_plot > 1
            legend('Location', 'best');
        end
        grid on;
    end

    sgtitle(sprintf('K-space trajectories - %s', truth.base_name), ...
            'Interpreter', 'none');
end
