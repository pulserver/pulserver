function result = validate(seq, seqObj, varargin)
% validate - Compare pulserver C-backend waveforms against a Pulseq sequence object.
%
%   result = pulserver.validate(seq, seqObj)
%   result = pulserver.validate(seq, seqObj, 'plot', true, 'grad_atol', 0.01)
%
% Compares gradient and RF waveforms extracted by the C backend against
% those produced by the Pulseq MATLAB toolbox waveforms_and_times().
%
% Inputs
%   seq       double | SequenceCollection   Collection handle or object.
%   seqObj    mr.Sequence   A Pulseq MATLAB sequence object (same .seq file).
%
% Name-value options
%   grad_atol        double   Abs. gradient tolerance (mT/m).
%                             Default: 3 * max_slew * grad_raster_time * 1e3
%   rf_rms_percent   double   RF RMS error tolerance (%). Default: 10
%   amplitude_mode   char     'actual' | 'max_pos' | 'first'. Default: 'actual'
%   tr_index         double   0-based TR index. Default: 0
%   doPlot           logical  Show overlay comparison plot. Default: false
%
% Output
%   result   struct with fields:
%       ok        - logical, true if all checks pass
%       errors    - struct mapping channel names to max error values
%       messages  - cell array of error messages (empty if ok)
%
% See also: pulserver.check, pulserver.get_tr_waveforms

    p = inputParser;
    addParameter(p, 'grad_atol',       [],       @isnumeric);
    addParameter(p, 'rf_rms_percent',  10,       @isnumeric);
    addParameter(p, 'amplitude_mode',  'actual', @ischar);
    addParameter(p, 'tr_index',        0,        @isnumeric);
    addParameter(p, 'doPlot',          false,    @islogical);
    parse(p, varargin{:});
    o = p.Results;

    gamma = 42.576e6;  % Hz/T

    % Default gradient tolerance: 3 slew steps
    if isempty(o.grad_atol)
        sys = seqObj.sys;
        o.grad_atol = 3 * sys.maxSlew * sys.gradRasterTime * 1e3;  % mT/m
    end

    % --- pulserver waveforms (from C backend) ---
    wf = pulserver.get_tr_waveforms(seq, ...
        'amplitude_mode', o.amplitude_mode, ...
        'tr_index',       o.tr_index);

    % --- reference waveforms from Pulseq MATLAB toolbox ---
    rpt = pulserver.report(seq);
    nBlocks = rpt(1).tr_size;
    nPrep   = rpt(1).num_prep_blocks;
    blk1 = nPrep + o.tr_index * nBlocks + 1;
    blk2 = blk1 + nBlocks - 1;
    [wave_data, ~] = seqObj.waveforms_and_times(true);

    result.ok = true;
    result.errors = struct();
    result.messages = {};

    % --- Compare gradients ---
    grad_channels = {'gx', 'gy', 'gz'};
    for k = 1:3
        ch = grad_channels{k};

        % pulserver
        t_ps  = wf.(ch).time_us * 1e-6;             % seconds
        a_ps  = wf.(ch).amplitude / gamma * 1e3;     % mT/m

        % reference (pypulseq-style: wave_data is cell {gx, gy, gz, rf, adc, ...})
        if size(wave_data, 2) >= k
            ref = wave_data{k};
            t_ref = ref(1, :);
            a_ref = ref(2, :) / gamma * 1e3;         % mT/m
        else
            t_ref = [];
            a_ref = [];
        end

        if isempty(t_ref)
            result.errors.(ch) = 0;
            continue;
        end

        % interpolate pulserver onto reference time grid
        a_interp = interp1(t_ps, a_ps, t_ref, 'linear', 0);
        err = max(abs(a_interp - a_ref));
        result.errors.(ch) = err;

        if err > o.grad_atol
            msg = sprintf('%s mismatch: max diff = %.4f mT/m (tol = %.4f)', ...
                          upper(ch), err, o.grad_atol);
            result.messages{end+1} = msg;
            result.ok = false;
        end
    end

    % --- Compare RF ---
    t_ps  = wf.rf_mag.time_us * 1e-6;
    a_ps  = wf.rf_mag.amplitude / gamma * 1e6;   % µT

    if size(wave_data, 2) >= 4
        ref = wave_data{4};
        t_ref = ref(1, :);
        a_ref = abs(ref(2, :)) / gamma * 1e6;    % µT
    else
        t_ref = [];
        a_ref = [];
    end

    if ~isempty(t_ref) && max(abs(a_ref)) > 0
        a_interp = interp1(t_ps, a_ps, t_ref, 'linear', 0);
        rms_ref = sqrt(mean(a_ref.^2));
        rms_err = sqrt(mean((a_interp - a_ref).^2));
        rf_pct  = 100 * rms_err / rms_ref;
        result.errors.rf = rf_pct;

        if rf_pct > o.rf_rms_percent
            msg = sprintf('RF mismatch: %.1f%% RMS error (tol = %.1f%%)', ...
                          rf_pct, o.rf_rms_percent);
            result.messages{end+1} = msg;
            result.ok = false;
        end
    else
        result.errors.rf = 0;
    end

    % --- Optional plot ---
    if o.doPlot
        plot_comparison(wf, wave_data, gamma);
    end

    % print summary
    if result.ok
        fprintf('pulserver.validate: OK\n');
    else
        for m = 1:numel(result.messages)
            fprintf('pulserver.validate: %s\n', result.messages{m});
        end
    end
end


function plot_comparison(wf, wave_data, gamma)
% plot_comparison - Overlay pulserver vs reference waveforms
    figure('Name', 'pulserver.validate');

    channels = {'rf_mag', 'gx', 'gy', 'gz'};
    ylabels  = {'|RF| (\muT)', 'Gx (mT/m)', 'Gy (mT/m)', 'Gz (mT/m)'};
    ref_idx  = [4, 1, 2, 3];  % mapping to wave_data cell indices

    axs = gobjects(1, 4);
    for k = 1:4
        axs(k) = subplot(4, 1, k);
        hold(axs(k), 'on');

        % pulserver
        ch = wf.(channels{k});
        t_ps = ch.time_us * 1e-3;  % ms
        a_ps = ch.amplitude;
        if k == 1
            a_ps = a_ps / gamma * 1e6;   % µT
        else
            a_ps = a_ps / gamma * 1e3;   % mT/m
        end
        plot(axs(k), t_ps, a_ps, 'b-', 'LineWidth', 1);

        % reference
        ri = ref_idx(k);
        if size(wave_data, 2) >= ri
            ref = wave_data{ri};
            t_ref = ref(1, :) * 1e3;  % ms
            a_ref = ref(2, :);
            if k == 1
                a_ref = abs(a_ref) / gamma * 1e6;
            else
                a_ref = a_ref / gamma * 1e3;
            end
            plot(axs(k), t_ref, a_ref, 'r--', 'LineWidth', 0.8);
        end

        ylabel(axs(k), ylabels{k}, 'Rotation', 0, 'HorizontalAlignment', 'right');
        grid(axs(k), 'on');
        if k == 1
            legend(axs(k), 'pulserver', 'reference', 'Location', 'best');
        end
    end

    xlabel(axs(end), 'time (ms)');
    linkaxes(axs, 'x');
    title(axs(1), 'pulserver.validate: waveform comparison');
end

function h = to_handle(seq)
    if isa(seq, 'pulserver.SequenceCollection')
        h = seq.Handle;
    else
        h = seq;
    end
end
