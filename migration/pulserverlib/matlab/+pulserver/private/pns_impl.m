function pns_impl(seq, varargin)
% pns - Plot convolved PNS waveforms for a representative TR.
%
%   pulserver.pns(seq, 'stim_threshold', 60, 'decay_constant_us', 360)
%   pulserver.pns(seq, 'stim_threshold', 60, 'decay_constant_us', 360, ...
%                'threshold_percent', 100, 'sequence_idx', 1)
%
% Displays per-axis and combined PNS waveforms with a threshold line.
% No pass/fail check is performed -- use check() for that.
%
% Inputs
%   seq     double | SequenceCollection   Collection handle or object.
%
% Required name-value parameters
%   stim_threshold     double   PNS threshold (Hz/m/s) = rheobase/alpha.
%   decay_constant_us  double   Chronaxie (us).
%
% Optional name-value parameters
%   sequence_idx       double   Subsequence (1-based). Default: 1.
%   threshold_percent  double   Threshold line (%). Default: 80.
%
% See also: pulserver.grad_spectrum, pulserver.SequenceCollection.check

    h = to_handle(seq);

    p = inputParser;
    addParameter(p, 'sequence_idx',       1,    @isnumeric);
    addParameter(p, 'stim_threshold',     [],   @isnumeric);
    addParameter(p, 'decay_constant_us',  [],   @isnumeric);
    addParameter(p, 'threshold_percent',  80,   @isnumeric);
    parse(p, varargin{:});
    o = p.Results;

    if isempty(o.stim_threshold) || isempty(o.decay_constant_us)
        error('pulserver:pns', ...
            'pns requires stim_threshold and decay_constant_us parameters.');
    end

    % Internally: alpha = 1, rheobase = stim_threshold
    result = pulseqlib_mex('pns', h, o.sequence_idx - 1, ...
                           o.decay_constant_us, o.stim_threshold, 1.0);

    n = result.num_samples;
    pns_x = result.slew_x;
    pns_y = result.slew_y;
    pns_z = result.slew_z;
    pns_total = sqrt(pns_x.^2 + pns_y.^2 + pns_z.^2);

    % Assume 10 us gradient raster (half-raster = 5 us)
    time_ms = (0:n-1) * 5e-3;

    figure;
    plot(time_ms, pns_total, 'Color', [0.886 0.290 0.200], 'LineWidth', 2); hold on;
    plot(time_ms, pns_x, 'Color', [0.122 0.467 0.706], 'LineWidth', 1.2);
    plot(time_ms, pns_y, 'Color', [1.000 0.498 0.055], 'LineWidth', 1.2);
    plot(time_ms, pns_z, 'Color', [0.173 0.627 0.173], 'LineWidth', 1.2);
    yline(o.threshold_percent, 'r--', 'LineWidth', 2);
    xlabel('Time (ms)');
    ylabel('PNS (%)');
    title('Peripheral Nerve Stimulation — Convolved Slew Rate');
    legend({'PNS Total', 'PNS X', 'PNS Y', 'PNS Z', ...
            sprintf('%.0f%% threshold', o.threshold_percent)}, ...
           'Location', 'northeast');
    grid on; set(gca, 'GridAlpha', 0.3);
    ylim([0, inf]);
    hold off;
end

function h = to_handle(seq)
    if isa(seq, 'pulserver.SequenceCollection')
        h = seq.Handle;
    else
        h = seq;
    end
end
