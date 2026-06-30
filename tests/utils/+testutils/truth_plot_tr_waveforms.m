function fig = truth_plot_tr_waveforms(base_or_truth, varargin)
%TRUTH_PLOT_TR_WAVEFORMS Plot canonical TR waveforms.
%
%   truth_plot_tr_waveforms('gre_2d_1sl_1avg')
%   truth_plot_tr_waveforms(truth_struct, 'overlay', true)

    p = inputParser;
    addParameter(p, 'overlay', true, @islogical);
    addParameter(p, 'show_slew', false, @islogical);
    parse(p, varargin{:});

    truth = coerce_truth(base_or_truth);

    GAMMA_HZ_PER_T = 42.577e6;
    GAMMA_GRAD     = GAMMA_HZ_PER_T * 1e-3;   % Hz/m -> mT/m

    fig = figure('Name', sprintf('Canonical waveforms: %s', truth.base_name), 'Color', 'w');

    labels = {'Gx (mT/m)', 'Gy (mT/m)', 'Gz (mT/m)'};
    fields = {'gx', 'gy', 'gz'};

    for ax = 1:3
        subplot(3, 1, ax);
        hold on;
        for i = 1:truth.tr_waveforms.num_trs
            w = truth.tr_waveforms.waveforms(i);
            t_ms = double(w.time_us) / 1e3;
            y = double(w.(fields{ax})) / GAMMA_GRAD;
            if p.Results.show_slew
                dt = diff(t_ms) * 1e-3;
                dy = diff(y);
                y = [0; dy ./ max(dt, eps)];
            end
            if p.Results.overlay
                plot(t_ms, y, 'LineWidth', 1.2, 'DisplayName', sprintf('TR %d', i - 1));
            else
                plot(t_ms, y, 'LineWidth', 1.2);
            end
        end
        hold off;
        ylabel(labels{ax});
        grid on;
        if ax == 1 && p.Results.overlay
            legend('show', 'Location', 'best');
        end
        if ax == 3
            xlabel('Time (ms)');
        end
    end

    if isfield(truth, 'meta') && isfield(truth.meta, 'canonical_mode')
        mode_lbl = truth.meta.canonical_mode;
    else
        mode_lbl = 'tr';
    end
    sgtitle(sprintf('Canonical waveforms (%s, mode=%s)', truth.base_name, mode_lbl), 'Interpreter', 'none');
end

function truth = coerce_truth(base_or_truth)
    if isstruct(base_or_truth)
        truth = base_or_truth;
    else
        truth = testutils.truth_parse_case(base_or_truth);
    end
end
