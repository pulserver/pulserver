function result = validate_impl(seq, varargin)
% validate - Compare C-backend waveforms against a reference source.
%
%   result = pulserver.validate(sc)
%   result = pulserver.validate(sc, 'do_plot', true)
%   result = pulserver.validate(sc, 'xml_path', 'waveforms.xml')
%   result = pulserver.validate(sc, 'tr_range', [0 3], 'grad_atol', 0.05)
%
% Validates pulserver C-backend waveforms against a reference.  The
% reference is either the stored mr.Sequence objects (default) or an
% XML waveform file.  Gradient channels are compared by maximum
% absolute error (mT/m); RF magnitude by percent-RMS error.
%
% When do_plot is true the C-backend TR waveform plot is created and
% the reference is overlaid for visual comparison (uses pulserver.plot
% overlay machinery).
%
% Inputs
%   seq     SequenceCollection   A SequenceCollection object.
%
% Name-value options
%   sequence_idx       double   Subsequence index (1-based).  Default: 1
%   xml_path           char     Path to XML waveform file.    Default: ''
%   do_plot            logical  Show overlay comparison plot.  Default: false
%   tr_range           1x2      Inclusive TR range [first, last] (1-based). Default: [1 1]
%   hide_prep          logical  Hide prep blocks in plot.     Default: true
%   hide_cooldown      logical  Hide cooldown in plot.        Default: true
%   show_rf_centers    logical  Mark RF iso-centres.          Default: false
%   show_echoes        logical  Mark ADC centres.             Default: false
%   show_segments      logical  Colour by segment.            Default: true
%   show_blocks        logical  Block boundary lines.         Default: false
%   max_grad_mT_per_m  double|logical  Gradient limit line.  Default: true
%                               true = derive from system limits.
%   grad_atol          double   Abs gradient tolerance (mT/m).
%                               Default: [] (3 * maxSlew * gradRaster * 1e3)
%   rf_rms_percent     double   RF RMS error tolerance (%).   Default: 10
%
% Output
%   result   struct with fields:
%       ok        - logical, true if all checks pass
%       errors    - struct mapping channel names to max error values
%       messages  - cell array of error messages (empty if ok)
%
% See also: pulserver.SequenceCollection.validate, pulserver.check

    p = inputParser;
    addParameter(p, 'sequence_idx',      1,       @isnumeric);
    addParameter(p, 'xml_path',          '',      @(x)ischar(x)||isstring(x));
    addParameter(p, 'do_plot',           false,   @islogical);
    addParameter(p, 'tr_range',          [1 1],   @(x)isnumeric(x)&&numel(x)==2);
    addParameter(p, 'hide_prep',         true,    @islogical);
    addParameter(p, 'hide_cooldown',     true,    @islogical);
    addParameter(p, 'show_rf_centers',   false,   @islogical);
    addParameter(p, 'show_echoes',       false,   @islogical);
    addParameter(p, 'show_segments',     true,    @islogical);
    addParameter(p, 'show_blocks',       false,   @islogical);
    addParameter(p, 'max_grad_mT_per_m', true,    @(x)islogical(x)||isnumeric(x));
    addParameter(p, 'grad_atol',         [],      @isnumeric);
    addParameter(p, 'rf_rms_percent',    10,      @isnumeric);
    parse(p, varargin{:});
    o = p.Results;

    gamma = 42.576e6;  % Hz/T

    % --- resolve SequenceCollection ---
    if isa(seq, 'pulserver.SequenceCollection')
        sc = seq;
        h  = seq.Handle;
    else
        error('pulserver:validate', ...
              'First argument must be a SequenceCollection object.');
    end

    % --- system parameters ---
    seqObj = sc.Seqs{o.sequence_idx};
    sys = seqObj.sys;

    % default gradient tolerance: 3 slew steps (mT/m)
    grad_atol = o.grad_atol;
    if isempty(grad_atol)
        grad_atol = 3 * sys.maxSlew * sys.gradRasterTime * 1e3;
    end

    % resolve max_grad for plot
    if islogical(o.max_grad_mT_per_m) && o.max_grad_mT_per_m
        max_grad_val = sys.maxGrad * 1e3;  % T/m -> mT/m
    elseif isnumeric(o.max_grad_mT_per_m) && o.max_grad_mT_per_m > 0
        max_grad_val = double(o.max_grad_mT_per_m);
    else
        max_grad_val = [];
    end

    use_xml = ~isempty(o.xml_path);
    ss_idx  = o.sequence_idx - 1;  % 0-based for C

    % TR metadata
    tr_info = pulseqlib_mex('find_tr', h, ss_idx);

    % --- validate each TR in range (1-based inclusive) ---
    result.ok = true;
    result.errors = struct();
    result.messages = {};

    multi_tr = o.tr_range(2) > o.tr_range(1);

    for tr_idx = o.tr_range(1):o.tr_range(2)
        % C-backend waveforms (get_tr_waveforms expects 1-based)
        wf = get_tr_waveforms(h, ...
            'subsequence_idx', o.sequence_idx, ...
            'amplitude_mode', 'actual', ...
            'tr_index', tr_idx);

        % reference waveforms (pypulseq_reference uses 0-based internally)
        if ~use_xml
            ref = pypulseq_reference(seqObj, tr_idx - 1, tr_info, gamma);
        else
            ref = xml_reference(o.xml_path, gamma);
        end

        % compare gradients
        channels = {'gx', 'gy', 'gz'};
        for k = 1:3
            ch = channels{k};
            [ref_t, ref_a] = deal(ref.(ch).time_us, ref.(ch).amplitude);
            ps_t = wf.(ch).time_us;
            ps_a = wf.(ch).amplitude;

            if isempty(ref_t) || isempty(ps_t)
                err = 0;
            else
                a_interp = interp1(ps_t, ps_a, ref_t, 'linear', 0);
                err = max(abs(ref_a - a_interp));
            end

            % track worst across TRs
            if ~isfield(result.errors, ch) || err > result.errors.(ch)
                result.errors.(ch) = err;
            end

            if err > grad_atol
                if multi_tr
                    pfx = sprintf('TR %d: ', tr_idx);
                else
                    pfx = '';
                end
                msg = sprintf('%s%s mismatch: max diff %.4f mT/m (tol %.4f)', ...
                              pfx, upper(ch), err, grad_atol);
                result.messages{end+1} = msg;
                result.ok = false;
            end
        end

        % compare RF
        ref_t = ref.rf_mag.time_us;
        ref_a = ref.rf_mag.amplitude;
        ps_t  = wf.rf_mag.time_us;
        ps_a  = wf.rf_mag.amplitude;

        if ~isempty(ref_t) && ~isempty(ps_t) && max(abs(ref_a)) > 0
            a_interp = interp1(ps_t, ps_a, ref_t, 'linear', 0);
            rms_ref  = sqrt(mean(ref_a.^2));
            rms_err  = sqrt(mean((ref_a - a_interp).^2));
            rf_pct   = 100 * rms_err / rms_ref;
        else
            rf_pct = 0;
        end

        if ~isfield(result.errors, 'rf_mag') || rf_pct > result.errors.rf_mag
            result.errors.rf_mag = rf_pct;
        end

        if rf_pct > o.rf_rms_percent
            if multi_tr
                pfx = sprintf('TR %d: ', tr_idx);
            else
                pfx = '';
            end
            msg = sprintf('%sRF mismatch: %.1f%% RMS (tol %.1f%%)', ...
                          pfx, rf_pct, o.rf_rms_percent);
            result.messages{end+1} = msg;
            result.ok = false;
        end
    end

    % --- optional overlay plot ---
    if o.do_plot
        plot_handle = plot_impl(sc, ...
            'sequence_idx',      o.sequence_idx, ...
            'tr_idx',            o.tr_range(1), ...
            'hide_prep',         o.hide_prep, ...
            'hide_cooldown',     o.hide_cooldown, ...
            'show_segments',     o.show_segments, ...
            'show_blocks',       o.show_blocks, ...
            'show_slew',         false, ...
            'show_rf_centers',   o.show_rf_centers, ...
            'show_echoes',       o.show_echoes, ...
            'max_grad_mT_per_m', max_grad_val);

        % Overlay reference (simplified: plot ref traces on existing axes)
        if ~use_xml
            ref = pypulseq_reference(seqObj, o.tr_range(1) - 1, tr_info, gamma);
        else
            ref = xml_reference(o.xml_path, gamma);
        end
        overlay_ref(plot_handle, ref, use_xml);

        % annotate pass/fail
        if result.ok
            sgtitle(plot_handle.fig, 'validate() — PASS', ...
                    'FontWeight', 'bold', 'Color', [0 0.6 0]);
        else
            sgtitle(plot_handle.fig, 'validate() — FAIL', ...
                    'FontWeight', 'bold', 'Color', [0.8 0 0]);
        end
    end

    % --- print summary ---
    if result.ok
        fprintf('pulserver.validate: OK\n');
    else
        for m = 1:numel(result.messages)
            fprintf('pulserver.validate: %s\n', result.messages{m});
        end
    end
end


% ── Local helpers ────────────────────────────────────────────────────

function ref = pypulseq_reference(seqObj, tr_idx, tr_info, gamma)
% Extract reference waveforms from stored mr.Sequence for a single TR.
% Returns struct with gx/gy/gz/rf_mag, each containing .time_us and
% .amplitude in physical units (mT/m or µT).

    num_prep = tr_info.num_prep_blocks;
    tr_dur_s = tr_info.tr_duration_us * 1e-6;

    % compute prep duration from block_durations
    bd = seqObj.blockDurations;
    keys = sort(cell2mat(bd.keys()));
    prep_dur_s = 0;
    for k = 1:num_prep
        prep_dur_s = prep_dur_s + bd(keys(k));
    end
    t0 = prep_dur_s + tr_idx * tr_dur_s;
    t1 = t0 + tr_dur_s;

    % Get full waveforms (Matlab Pulseq toolbox may not support time_range)
    [wave_data, ~] = seqObj.waveforms_and_times(true);

    hz_to_mT_m = 1 / (gamma * 1e-3);
    hz_to_uT   = 1e6 / gamma;

    channels = {'gx', 'gy', 'gz'};
    for k = 1:3
        if k <= numel(wave_data) && ~isempty(wave_data{k}) && size(wave_data{k}, 2) > 0
            t_s = wave_data{k}(1,:);
            a   = wave_data{k}(2,:);
            mask = (t_s >= t0 - 1e-9) & (t_s <= t1 + 1e-9);
            ref.(channels{k}).time_us   = (t_s(mask) - t0) * 1e6;
            ref.(channels{k}).amplitude = a(mask) * hz_to_mT_m;
        else
            ref.(channels{k}).time_us   = [];
            ref.(channels{k}).amplitude = [];
        end
    end

    % RF (index 4 when append_RF=true)
    if numel(wave_data) >= 4 && ~isempty(wave_data{4}) && size(wave_data{4}, 2) > 0
        t_s = wave_data{4}(1,:);
        a   = wave_data{4}(2,:);
        mask = (t_s >= t0 - 1e-9) & (t_s <= t1 + 1e-9);
        ref.rf_mag.time_us   = (t_s(mask) - t0) * 1e6;
        ref.rf_mag.amplitude = abs(a(mask)) * hz_to_uT;
    else
        ref.rf_mag.time_us   = [];
        ref.rf_mag.amplitude = [];
    end
end


function ref = xml_reference(xml_path, gamma)  %#ok<INUSD>
% Extract reference waveforms from an XML file.

    doc = xmlread(xml_path);

    g_to_mT_m = 10;  % G/cm -> mT/m
    g_to_uT   = 100; % G -> µT

    channels = {'gx', 'gy', 'gz'};
    for k = 1:3
        ch = channels{k};
        nodes = doc.getElementsByTagName(ch);
        if nodes.getLength() > 0
            txt = char(nodes.item(0).getTextContent());
            data = str2num(txt);  %#ok<ST2NM>
            if numel(data) >= 2
                n = floor(numel(data) / 2);
                ref.(ch).time_us   = data(1:n);
                ref.(ch).amplitude = data(n+1:2*n) * g_to_mT_m;
            else
                ref.(ch).time_us   = [];
                ref.(ch).amplitude = [];
            end
        else
            ref.(ch).time_us   = [];
            ref.(ch).amplitude = [];
        end
    end

    nodes = doc.getElementsByTagName('rf');
    if nodes.getLength() > 0
        txt = char(nodes.item(0).getTextContent());
        data = str2num(txt);  %#ok<ST2NM>
        if numel(data) >= 2
            n = floor(numel(data) / 2);
            ref.rf_mag.time_us   = data(1:n);
            ref.rf_mag.amplitude = abs(data(n+1:2*n)) * g_to_uT;
        else
            ref.rf_mag.time_us   = [];
            ref.rf_mag.amplitude = [];
        end
    else
        ref.rf_mag.time_us   = [];
        ref.rf_mag.amplitude = [];
    end
end


function overlay_ref(plot_handle, ref, use_xml)
% Overlay reference waveforms onto an existing plot handle.
    if use_xml
        lbl = 'XML';
    else
        lbl = 'mr.Sequence';
    end

    tscale = 1e-3;  % us -> ms

    channels = {'gx', 'gy', 'gz'};
    ax_names = {'gx', 'gy', 'gz'};
    for k = 1:3
        ch = channels{k};
        ax = plot_handle.axes.(ax_names{k});
        t = ref.(ch).time_us;
        a = ref.(ch).amplitude;
        if ~isempty(t)
            hold(ax, 'on');
            plot(ax, t * tscale, a, 'r--', 'LineWidth', 0.7, ...
                 'DisplayName', lbl);
        end
    end

    % RF
    ax = plot_handle.axes.rf_mag;
    t = ref.rf_mag.time_us;
    a = ref.rf_mag.amplitude;
    if ~isempty(t)
        hold(ax, 'on');
        plot(ax, t * tscale, a, 'r--', 'LineWidth', 0.7, ...
             'DisplayName', lbl);
        legend(ax, 'Location', 'best', 'FontSize', 7);
    end
end
