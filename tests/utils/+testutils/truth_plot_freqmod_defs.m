function fig = truth_plot_freqmod_defs(base_or_truth, varargin)
%TRUTH_PLOT_FREQMOD_DEFS Plot base and projected frequency modulation.
%
% Layout is 4x2:
%   Rows 1-3: base per-axis frequency modulation and cumulative phase.
%   Row 4: projected actual modulation and phase_total for x/y/z/oblique probes.

    p = inputParser;
    addParameter(p, 'def_idx',     [], @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'segment_idx', 1,  @(x) isnumeric(x) && isscalar(x) && x >= 1);
    parse(p, varargin{:});

    truth = coerce_truth(base_or_truth);

    if truth.freqmod_def.num_defs == 0
        error('truth:empty', 'No frequency modulation definitions found for %s', truth.base_name);
    end

    if isempty(p.Results.def_idx)
        ids = 1:truth.freqmod_def.num_defs;  % will be filtered to segment below
    else
        ids = p.Results.def_idx(:).';
    end

    adc_dwell_ms = double(truth.meta.adc_dwell_ns) * 1e-6;
    n_adc        = double(truth.meta.adc_samples);
    adc_dur_all  = n_adc .* adc_dwell_ms;  % per-definition durations (ms)

    % --- Build segment timing context (ms) ----------------------------
    s_idx = p.Results.segment_idx;
    if s_idx < 1 || s_idx > truth.segment_def.num_segments
        error('truth:input', 'segment_idx %d out of range', s_idx);
    end
    seg      = truth.segment_def.segments(s_idx);
    n_blocks = seg.num_blocks;

    blk_fmod_ids = zeros(1, n_blocks);  % 1-based freq_mod_id per block

    % --- Filter defs to those actually used by this segment -----------
    if isempty(p.Results.def_idx)
        % Use segment_order + segment_num_blocks to find which scan-table
        % entries belong to this segment, then collect their freq_mod_id.
        seg_order  = truth.meta.segment_order;   % 0-based segment IDs
        seg_nblk   = truth.meta.segment_num_blocks;  % blocks per segment def
        % Per-block freq_mod_id (OR-merged across all TRs)
        blocks_per_tr = sum(arrayfun(@(s) seg_nblk(s + 1), seg_order));
        n_scan = numel(truth.scan_table.entries);
        st_pos = 1;  % current position in scan table
        while st_pos + blocks_per_tr - 1 <= n_scan
            for k = 1:numel(seg_order)
                sid = seg_order(k);       % 0-based
                nb  = seg_nblk(sid + 1);  % 1-based index
                if sid == (s_idx - 1)     % this is our segment
                    for bb = 1:nb
                        fid_k = double(truth.scan_table.entries(st_pos + bb - 1).freq_mod_id);
                        blk_fmod_ids(bb) = max(blk_fmod_ids(bb), fid_k);
                    end
                end
                st_pos = st_pos + nb;
            end
        end
        ids = unique(blk_fmod_ids(blk_fmod_ids > 0));
        if isempty(ids)
            fig = [];
            return;
        end
    else
        for b = 1:n_blocks
            fid0 = double(seg.blocks(b).freq_mod_def_id);
            if fid0 >= 0
                blk_fmod_ids(b) = fid0 + 1;  % 0-based -> 1-based
            end
        end
    end

    block_end_ms = zeros(1, n_blocks);
    for b = 1:n_blocks
        blk   = seg.blocks(b);
        spans = zeros(1, 5);
        for ch = 1:3
            n = double(blk.grad_n(ch));
            if n > 0
                spans(ch) = double(blk.grad_delay(ch)) * 1e3 + ...
                            max(double(blk.grad_time_s{ch})) * 1e3;
            end
        end
        rf_n = double(blk.rf_n);
        if rf_n > 0
            spans(4) = double(blk.rf_delay) * 1e3 + ...
                       max(double(blk.rf_time_s)) * 1e3;
        end
        if blk.has_adc
            ad = double(blk.adc_def_id) + 1;  % 0-based -> 1-based
            spans(5) = double(blk.adc_delay) * 1e3 + adc_dur_all(ad);
        end
        block_end_ms(b) = max(spans);
    end
    block_start_ms = [0, cumsum(block_end_ms + 0.002)];
    total_dur_ms   = block_start_ms(end);

    % --- Match each def to its source block via blk_fmod_ids -----------
    def_t0_ms_list = cell(1, numel(ids));   % active-window start list per def (ms)
    for i = 1:numel(ids)
        def = truth.freqmod_def.defs(ids(i));
        t0_vals = [];
        for b = 1:n_blocks
            if blk_fmod_ids(b) ~= ids(i), continue; end
            blk = seg.blocks(b);
            if def.type == 0 && blk.has_rf
                t0_vals(end+1) = block_start_ms(b) + double(blk.rf_delay) * 1e3; %#ok<AGROW>
            elseif def.type == 1 && blk.has_adc
                t0_vals(end+1) = block_start_ms(b) + double(blk.adc_delay) * 1e3; %#ok<AGROW>
            end
        end
        def_t0_ms_list{i} = t0_vals;
    end

    % --- Plot (4 rows x 2 cols: col1=freq mod, col2=phase) -------------
    n_defs     = numel(ids);
    cmap       = lines(n_defs);
    bot_margin = 0.09;
    probe_colors = [0.85 0.33 0.10; 0.00 0.45 0.74; 0.47 0.67 0.19; 0.49 0.18 0.56];
    probe_labels = {'x', 'y', 'z', 'oblique'};

    fig    = figure('Name', sprintf('Freqmod defs: %s', truth.base_name), 'Color', 'w');
    labels = {'Gx', 'Gy', 'Gz'};
    
    % Create all 4x2 subplots upfront to avoid grid conflicts
    fm_ax  = gobjects(1, 4);
    ph_ax  = gobjects(1, 4);

    for ax = 1:3
        fm_ax(ax) = subplot(4, 2, 2*(ax-1) + 1);
        hold(fm_ax(ax), 'on');
        ylabel(fm_ax(ax), sprintf('%s (Hz/m)', labels{ax}));
        grid(fm_ax(ax), 'on');
        if ax == 1, title(fm_ax(ax), 'Freq Mod'); end

        ph_ax(ax) = subplot(4, 2, 2*(ax-1) + 2);
        hold(ph_ax(ax), 'on');
        ylabel(ph_ax(ax), sprintf('\\phi_{%s} (rad/m)', labels{ax}));
        grid(ph_ax(ax), 'on');
        if ax == 1, title(ph_ax(ax), 'Phase'); end
    end
    xlabel(fm_ax(3), 'Time (ms)');
    xlabel(ph_ax(3), 'Time (ms)');

    fm_ax(4) = subplot(4, 2, 7);
    hold(fm_ax(4), 'on');
    ylabel(fm_ax(4), 'f_{proj} (Hz)');
    xlabel(fm_ax(4), 'Time (ms)');
    grid(fm_ax(4), 'on');
    title(fm_ax(4), 'Projected Actual Modulation');

    ph_ax(4) = subplot(4, 2, 8);
    hold(ph_ax(4), 'on');
    ylabel(ph_ax(4), '\phi_{proj} (rad)');
    xlabel(ph_ax(4), 'Time (ms)');
    grid(ph_ax(4), 'on');
    title(ph_ax(4), 'Projected Phase Compensation');
    
    fm_actual_ax = fm_ax(4);
    ph_actual_ax = ph_ax(4);

    def_lbls = cell(1, n_defs);
    for i = 1:n_defs
        d    = ids(i);
        def  = truth.freqmod_def.defs(d);
        dt_s = double(def.raster_us) * 1e-6;
        kind = 'RF';
        if def.type == 1, kind = 'ADC'; end
        def_lbls{i} = sprintf('Def %d (%s)', d, kind);

        t0_list_ms = def_t0_ms_list{i};
        if isempty(t0_list_ms)
            continue;
        end
        dur_ms   = double(def.duration_us) * 1e-3;
        t_win_ms = (0:double(def.num_samples)-1).' * double(def.raster_us) * 1e-3;

        for ax = 1:3
            fm = double(def.waveform(:, ax));
            ph = cumsum(fm * 2 * pi * dt_s);
            for occ = 1:numel(t0_list_ms)
                t0_ms = t0_list_ms(occ);

                % Zero-padded envelope around each active window.
                t_plot  = [0;  t0_ms;  t0_ms + t_win_ms;  t0_ms + dur_ms;  total_dur_ms];
                fm_plot = [0;  0;      fm;                 0;               0            ];
                ph_plot = [0;  0;      ph;                 ph(end);         ph(end)      ];

                plot(fm_ax(ax), t_plot, fm_plot, 'Color', cmap(i,:), 'LineWidth', 1.2);
                plot(ph_ax(ax), t_plot, ph_plot, 'Color', cmap(i,:), 'LineWidth', 1.2);

                % Diamond at the interpolated cumulative phase at ref_time.
                ref_t_ms = double(def.ref_time_us) * 1e-3;
                ref_ph_interp = interp1(t_win_ms, ph, ref_t_ms, 'linear', ph(end));
                plot(ph_ax(ax), t0_ms + ref_t_ms, ref_ph_interp, 'd', ...
                    'Color', cmap(i,:), 'MarkerSize', 7, 'MarkerFaceColor', cmap(i,:), ...
                    'HandleVisibility', 'off');
            end
        end

        if isfield(truth, 'freqmod_plan') && truth.freqmod_plan.num_plans > 0
            plan_idx = find(arrayfun(@(p) p.def_id == d, truth.freqmod_plan.plans), 1, 'first');
            if ~isempty(plan_idx)
                pe = truth.freqmod_plan.plans(plan_idx);
                t_win_ms = (0:double(pe.num_samples)-1).' * double(def.raster_us) * 1e-3;
                dur_ms = double(def.duration_us) * 1e-3;
                for q = 1:min(4, truth.freqmod_plan.num_probes)
                    wf = pe.waveforms(q, 1:pe.num_samples).';
                    ph_tot = pe.phase_total(q);
                    for occ = 1:numel(t0_list_ms)
                        t0_ms = t0_list_ms(occ);
                        t_plot  = [0;  t0_ms;  t0_ms + t_win_ms;  t0_ms + dur_ms;  total_dur_ms];
                        wf_plot = [0;  0;      wf;                 0;               0            ];
                        ph_plot = [0; 0; ph_tot; ph_tot];
                        t_phase = [0; t0_ms + double(def.ref_time_us) * 1e-3; total_dur_ms; total_dur_ms];

                        if occ == 1
                            dname = sprintf('Def %d %s', d, probe_labels{q});
                        else
                            dname = '';
                        end

                        plot(fm_ax(4), t_plot, wf_plot, ...
                            'Color', probe_colors(q, :), 'LineWidth', 1.2, ...
                            'DisplayName', dname);
                        plot(ph_ax(4), t_phase, ph_plot, ...
                            'Color', probe_colors(q, :), 'LineWidth', 1.2, ...
                            'DisplayName', dname);

                        % Diamond at the reference timepoint on phase subplot.
                        ref_t_ms = double(def.ref_time_us) * 1e-3;
                        plot(ph_ax(4), t0_ms + ref_t_ms, ph_tot, 'd', ...
                            'Color', probe_colors(q, :), 'MarkerSize', 7, ...
                            'MarkerFaceColor', probe_colors(q, :), 'HandleVisibility', 'off');
                    end
                end
            end
        end
    end

    all_ax = [reshape(fm_ax(1:3), 1, 3), reshape(ph_ax(1:3), 1, 3), fm_ax(4), ph_ax(4)];
    linkaxes(all_ax, 'x');

    for k = 1:8
        pos    = get(all_ax(k), 'Position');
        pos(2) = pos(2) * (1 - bot_margin) + bot_margin;
        pos(4) = pos(4) * (1 - bot_margin);
        set(all_ax(k), 'Position', pos);
    end

    lgd_ax = axes('Parent', fig, ...
        'Position', [0.08, 0.005, 0.84, bot_margin - 0.01], ...
        'Visible', 'off');
    hold(lgd_ax, 'on');
    dummy_h = gobjects(1, n_defs);
    for i = 1:n_defs
        dummy_h(i) = plot(lgd_ax, NaN, NaN, '-', 'Color', cmap(i,:), 'LineWidth', 1.2);
    end
    legend(lgd_ax, dummy_h, def_lbls, ...
        'Orientation', 'horizontal', 'Location', 'north', 'Box', 'off');

    legend(fm_ax(4), 'show', 'Location', 'eastoutside');
    legend(ph_ax(4), 'show', 'Location', 'eastoutside');

    sgtitle(fig, sprintf('Frequency Modulation Definitions (%s) seg %d', ...
        truth.base_name, s_idx - 1), 'Interpreter', 'none');
end

function truth = coerce_truth(base_or_truth)
    if isstruct(base_or_truth)
        truth = base_or_truth;
    else
        truth = testutils.truth_parse_case(base_or_truth);
    end
end
