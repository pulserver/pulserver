function fig = plot(seq, varargin)
% plot - Plot waveforms for one TR of a loaded sequence.
%
%   pulserver.plot(seq)
%   pulserver.plot(seq, 'amplitude_mode', 'actual', 'time_unit', 'us')
%   pulserver.plot(seq, 'collapse_delays', true, 'collapsed_duration_us', 50)
%   fig = pulserver.plot(seq);
%
% Inputs
%   seq     double | SequenceCollection   Collection handle or object.
%
% Name-value options (forwarded to get_tr_waveforms)
%   amplitude_mode     char     'max_pos' | 'actual' | 'first'. Default: 'max_pos'
%   tr_index           double   0-based TR index. Default: 0
%   include_prep       logical  Include preparation blocks. Default: false
%   include_cooldown   logical  Include cooldown blocks. Default: false
%
% Additional options
%   time_unit              char     'ms' (default) | 'us'
%   collapse_delays        logical  Shrink pure-delay blocks. Default: false
%   delay_threshold_us     double   Min block duration (us) to collapse. Default: 1000
%   collapsed_duration_us  double   Display duration (us) for collapsed blocks. Default: 100
%
% Output
%   fig     figure handle (optional)
%
% See also: pulserver.SequenceCollection

    p = inputParser;
    addParameter(p, 'amplitude_mode',       'max_pos', @ischar);
    addParameter(p, 'tr_index',             0,         @isnumeric);
    addParameter(p, 'include_prep',         false,     @islogical);
    addParameter(p, 'include_cooldown',     false,     @islogical);
    addParameter(p, 'time_unit',            'ms',      @ischar);
    addParameter(p, 'collapse_delays',      false,     @islogical);
    addParameter(p, 'delay_threshold_us',   1000,      @isnumeric);
    addParameter(p, 'collapsed_duration_us', 100,      @isnumeric);
    parse(p, varargin{:});
    o = p.Results;

    % resolve handle
    seq = to_handle(seq);

    % get waveforms
    wf = get_tr_waveforms(seq, ...
        'amplitude_mode',   o.amplitude_mode, ...
        'tr_index',         o.tr_index, ...
        'include_prep',     o.include_prep, ...
        'include_cooldown', o.include_cooldown);

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

    % --- Build delay-collapse mapping if requested ---
    breaks = [];   % empty => no remapping
    if o.collapse_delays
        breaks = build_collapse_map(wf, channels_list(), ...
            o.delay_threshold_us, o.collapsed_duration_us);
    end

    fig_h = figure('Name', 'pulserver.plot');
    channels = channels_list();
    ylabels  = {'|RF| (\muT)', '\angleRF (rad)', 'Gx (mT/m)', 'Gy (mT/m)', 'Gz (mT/m)'};
    colors   = {'k', 'k', 'r', 'g', 'b'};
    nPanels  = numel(channels);

    axs = gobjects(1, nPanels);
    for k = 1:nPanels
        axs(k) = subplot(nPanels, 1, k);
        ch = wf.(channels{k});
        t = remap_time(ch.time_us, breaks) * tscale;
        a = ch.amplitude;

        % unit conversions matching Python wrapper
        switch channels{k}
            case 'rf_mag'
                % C library outputs Hz; convert to µT: a / gamma * 1e6
                % gamma ≈ 42.576e6 Hz/T
                a = a / 42.576e6 * 1e6;
            case 'rf_phase'
                % already radians
            otherwise
                % gradients: C library outputs Hz/m; convert to mT/m
                a = a / 42.576e6 * 1e3;
        end

        plot(axs(k), t, a, [colors{k} '-'], 'LineWidth', 0.8);
        ylabel(axs(k), ylabels{k}, 'Rotation', 0, 'HorizontalAlignment', 'right');
        grid(axs(k), 'on');
    end

    xlabel(axs(end), xlabel_str);
    linkaxes(axs, 'x');

    title(axs(1), sprintf('TR waveforms (mode = %s, tr\\_index = %d)', ...
          o.amplitude_mode, o.tr_index));

    if nargout > 0
        fig = fig_h;
    end
end

% ── Local helpers ────────────────────────────────────────────────────

function ch = channels_list()
    ch = {'rf_mag', 'rf_phase', 'gx', 'gy', 'gz'};
end

function t_out = remap_time(t_us, breaks)
% Remap original time through the collapse mapping.
% If breaks is empty, return t_us unchanged.
    if isempty(breaks)
        t_out = t_us;
        return;
    end
    t_out = zeros(size(t_us));
    for i = 1:numel(t_us)
        t = t_us(i);
        mapped = false;
        for j = 1:size(breaks, 1)
            s   = breaks(j, 1);
            e   = breaks(j, 2);
            d   = breaks(j, 3);
            dur = breaks(j, 4);
            if t >= s && t <= e
                frac = (t - s) / max(e - s, 1e-12);
                t_out(i) = d + frac * dur;
                mapped = true;
                break;
            end
        end
        if ~mapped
            t_out(i) = t;
        end
    end
end

function breaks = build_collapse_map(wf, channels, threshold_us, collapsed_us)
% Build (N x 4) mapping: [orig_start, orig_end, disp_start, disp_dur].
% A block is "pure delay" if no channel has samples in its time range.
    % We need block info from the MEX struct.  The wf struct has
    % total_duration_us but no per-block descriptors from the MEX side,
    % so we infer from the waveform gaps.  For now, use a simple
    % approach: scan all channel time vectors and identify gaps.

    % Collect all event time stamps across channels
    all_t = [];
    for k = 1:numel(channels)
        ch = wf.(channels{k});
        if ~isempty(ch.time_us)
            all_t = [all_t; ch.time_us(:)]; %#ok<AGROW>
        end
    end
    all_t = sort(unique(all_t));

    total_dur = wf.total_duration_us;

    % Build list of occupied intervals from the event times
    % and identify gaps > threshold as pure delays
    if isempty(all_t)
        % entire duration is a delay
        breaks = [0, total_dur, 0, collapsed_us];
        return;
    end

    pieces = [];
    display_t = 0;

    % Gap before first event
    if all_t(1) > threshold_us
        pieces = [pieces; 0, all_t(1), display_t, collapsed_us];
        display_t = display_t + collapsed_us;
    elseif all_t(1) > 0
        pieces = [pieces; 0, all_t(1), display_t, all_t(1)];
        display_t = display_t + all_t(1);
    end

    % Occupied region
    event_start = all_t(1);
    event_end   = all_t(end);
    event_dur   = event_end - event_start;
    pieces = [pieces; event_start, event_end, display_t, event_dur];
    display_t = display_t + event_dur;

    % Gap after last event
    tail = total_dur - event_end;
    if tail > threshold_us
        pieces = [pieces; event_end, total_dur, display_t, collapsed_us];
    elseif tail > 0
        pieces = [pieces; event_end, total_dur, display_t, tail];
    end

    breaks = pieces;
end

function h = to_handle(seq)
% Accept either a numeric handle or a SequenceCollection object.
    if isa(seq, 'pulserver.SequenceCollection')
        h = seq.Handle;
    else
        h = seq;
    end
end
