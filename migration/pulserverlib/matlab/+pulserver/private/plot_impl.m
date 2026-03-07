function handle = plot_impl(seq, varargin)
% plot - Plot waveforms for one TR of a loaded sequence.
%
%   pulserver.plot(seq)
%   pulserver.plot(seq, 'tr_idx', 'max_pos', 'collapse_delays', true)
%   handle = pulserver.plot(seq);
%
%   Layout is (3, 2): first column has RF magnitude (top), RF phase
%   (middle), ADC mask (bottom); second column has Gx, Gy, Gz.
%
% Inputs
%   seq     SequenceCollection | double   Loaded sequence collection.
%
% Name-value options (forwarded to get_tr_waveforms)
%   sequence_idx       double|char  Subsequence index (1-based). Default: 1
%   tr_idx             double|char  TR index (1-based) or 'max_pos'/'zero_var'.
%                                   Default: 'max_pos'
%   hide_prep          logical      Hide preparation blocks. Default: true
%   hide_cooldown      logical      Hide cooldown blocks. Default: true
%   collapse_delays    logical      Shrink pure-delay blocks at C level. Default: false
%
% Additional options
%   time_unit          char         'ms' (default) | 'us'
%   show_segments      logical      Colour-code by segment. Default: true
%   show_blocks        logical      Draw block boundaries. Default: true
%   show_slew          logical      Overlay slew rate. Default: false
%   show_rf_centers    logical      Mark RF iso-centres. Default: false
%   show_echoes        logical      Mark ADC echoes. Default: false
%   max_grad_mT_per_m  double       Max gradient reference line. Default: []
%   max_slew_T_per_m_per_s double   Max slew reference line. Default: []
%
% Output
%   handle  struct with .fig, .axes, .tr_duration_us, .num_trs,
%           .first_tr_start_us
%
% See also: pulserver.SequenceCollection

    p = inputParser;
    addParameter(p, 'sequence_idx',        1,         @isnumeric);
    addParameter(p, 'tr_idx',              'max_pos', @(x)isnumeric(x)||ischar(x)||isstring(x));
    addParameter(p, 'hide_prep',           true,      @islogical);
    addParameter(p, 'hide_cooldown',       true,      @islogical);
    addParameter(p, 'collapse_delays',     false,     @islogical);
    addParameter(p, 'time_unit',           'ms',      @ischar);
    addParameter(p, 'show_segments',       true,      @islogical);
    addParameter(p, 'show_blocks',         true,      @islogical);
    addParameter(p, 'show_slew',           false,     @islogical);
    addParameter(p, 'show_rf_centers',     false,     @islogical);
    addParameter(p, 'show_echoes',         false,     @islogical);
    addParameter(p, 'max_grad_mT_per_m',   [],        @isnumeric);
    addParameter(p, 'max_slew_T_per_m_per_s', [],     @isnumeric);
    parse(p, varargin{:});
    o = p.Results;

    % resolve handle
    h = to_handle(seq);
    ss_idx = o.sequence_idx - 1;  % convert to 0-based for C

    % Determine amplitude mode and TR index
    if ischar(o.tr_idx) || isstring(o.tr_idx)
        amplitude_mode = char(o.tr_idx);
        tr_index = 1;
    elseif o.tr_idx > 1
        amplitude_mode = 'actual';
        tr_index = o.tr_idx;
    else
        amplitude_mode = 'max_pos';
        tr_index = 1;
    end

    include_prep = ~o.hide_prep;
    include_cooldown = ~o.hide_cooldown;

    % get waveforms
    wf = get_tr_waveforms(h, ...
        'subsequence_idx',  ss_idx + 1, ...
        'amplitude_mode',   amplitude_mode, ...
        'tr_index',         tr_index, ...
        'include_prep',     include_prep, ...
        'include_cooldown', include_cooldown, ...
        'collapse_delays',  o.collapse_delays);

    % Get TR metadata
    tr_info = pulseqlib_mex('find_tr', h, ss_idx);

    % time scaling
    switch o.time_unit
        case 'ms'
            tscale = 1e-3;
            xlabel_str = 'time (ms)';
        case 'us'
            tscale = 1;
            xlabel_str = 'time (\mus)';
        otherwise
            error('pulserver:plot', 'time_unit must be ''ms'' or ''us''');
    end

    % gamma for unit conversion
    gamma = 42.576e6;

    fig_h = figure('Name', 'pulserver.plot');

    % Layout: (3, 2)
    % Col 1: rf_mag, rf_phase, adc
    % Col 2: gx, gy, gz
    axs = gobjects(3, 2);
    for r = 1:3
        for c = 1:2
            axs(r, c) = subplot(3, 2, (r-1)*2 + c);
        end
    end

    % Helper for time conversion
    t_fn = @(t) t * tscale;

    % ── RF magnitude (row 1, col 1) ──
    ax = axs(1, 1);
    ch = wf.rf_mag;
    t = t_fn(ch.time_us);
    a = ch.amplitude / gamma * 1e6;  % Hz -> uT
    if ~isempty(t)
        plot(ax, t, a, 'k-', 'LineWidth', 0.8);
    end
    ylabel(ax, '|RF| (\muT)');
    grid(ax, 'on');

    % ── RF phase (row 2, col 1) ──
    ax = axs(2, 1);
    ch = wf.rf_phase;
    t = t_fn(ch.time_us);
    if ~isempty(t)
        plot(ax, t, ch.amplitude, 'k-', 'LineWidth', 0.8);
    end
    ylabel(ax, '\angleRF (rad)');
    set(ax, 'YTick', [-pi 0 pi], 'YTickLabel', {'-\pi','0','\pi'});
    grid(ax, 'on');

    % ── ADC (row 3, col 1) ── placeholder
    ax = axs(3, 1);
    ylabel(ax, 'ADC');
    set(ax, 'YLim', [0 1], 'YTick', []);
    grid(ax, 'on');
    % ADC rectangles would require block-level info from MEX;
    % for now leave as empty panel

    % ── Gradients (col 2) ──
    grad_channels = {'gx', 'gy', 'gz'};
    grad_colors   = {'r', 'g', 'b'};
    grad_ylabels  = {'Gx (mT/m)', 'Gy (mT/m)', 'Gz (mT/m)'};
    for k = 1:3
        ax = axs(k, 2);
        ch = wf.(grad_channels{k});
        t = t_fn(ch.time_us);
        a = ch.amplitude / gamma * 1e3;  % Hz/m -> mT/m
        if ~isempty(t)
            plot(ax, t, a, [grad_colors{k} '-'], 'LineWidth', 0.8);
        end
        ylabel(ax, grad_ylabels{k});
        grid(ax, 'on');

        if ~isempty(o.max_grad_mT_per_m)
            hold(ax, 'on');
            yline(ax, o.max_grad_mT_per_m, '--', 'Color', [0.5 0.5 0.5], 'Alpha', 0.6);
            yline(ax, -o.max_grad_mT_per_m, '--', 'Color', [0.5 0.5 0.5], 'Alpha', 0.6);
        end
    end

    % ── RF centre markers ──
    if o.show_rf_centers && ~isempty(wf.rf_mag.time_us)
        ax = axs(1, 1);
        hold(ax, 'on');
        t_rf = wf.rf_mag.time_us;
        a_rf = abs(wf.rf_mag.amplitude / gamma * 1e6);
        % Find bursts
        if numel(t_rf) > 1
            gaps = find(diff(t_rf) > 10);
            starts = [1; gaps(:)+1];
            ends   = [gaps(:); numel(t_rf)];
            for j = 1:numel(starts)
                seg_t = t_rf(starts(j):ends(j));
                seg_a = a_rf(starts(j):ends(j));
                total = sum(seg_a);
                if total > 0
                    center_us = sum(seg_t .* seg_a) / total;
                    xline(ax, center_us * tscale, 'r--', 'Alpha', 0.7);
                end
            end
        end
    end

    % ── Echo markers ──
    if o.show_echoes
        ax = axs(3, 1);
        hold(ax, 'on');
        % Echo markers would need ADC events from MEX
    end

    % Link x axes
    linkaxes(axs(:), 'x');

    % X labels on bottom row
    xlabel(axs(3, 1), xlabel_str);
    xlabel(axs(3, 2), xlabel_str);

    title(axs(1, 1), sprintf('TR waveforms (mode = %s, tr\\_index = %d)', ...
          amplitude_mode, tr_index));

    % Build output handle
    handle_out = struct();
    handle_out.fig = fig_h;
    handle_out.axes = struct('rf_mag', axs(1,1), 'rf_phase', axs(2,1), ...
        'adc', axs(3,1), 'gx', axs(1,2), 'gy', axs(2,2), 'gz', axs(3,2));
    handle_out.tr_duration_us = tr_info.tr_duration_us;
    handle_out.num_trs = tr_info.num_trs;
    handle_out.first_tr_start_us = 0;

    if nargout > 0
        handle = handle_out;
    end
end

% ── Local helpers ────────────────────────────────────────────────────

function h = to_handle(seq)
% Accept either a numeric handle or a SequenceCollection object.
    if isa(seq, 'pulserver.SequenceCollection')
        h = seq.Handle;
    else
        h = seq;
    end
end
