function figs = truth_plot_segments(base_or_truth, varargin)
%TRUTH_PLOT_SEGMENTS Plot segment block waveforms on a shared real-time axis.
%
% All five channels (Gx/Gy/Gz/RF/ADC) share the same x-axis in ms.
% Gradient x-axis uses grad_time_s (knot times for traps, uniform raster
% for arb), both stored in the segment_def binary by TruthBuilder.
% RF x-axis uses the per-block rf_raster_us field.
% Block start offsets are computed from actual event endpoints.
%
% Gradient amplitudes are in mT/m, RF |B1| in uT, ADC as a binary mask.
%
%   truth_plot_segments('gre_2d_1sl_1avg')
%   truth_plot_segments(truth_struct, 'segment_idx', [1 2])

    p = inputParser;
    addParameter(p, 'segment_idx', [], @(x) isempty(x) || isnumeric(x));
    parse(p, varargin{:});

    truth = coerce_truth(base_or_truth);

    GAMMA_HZ_PER_T = 42.577e6;
    GAMMA_GRAD     = GAMMA_HZ_PER_T * 1e-3;   % Hz/m -> mT/m
    GAMMA_RF       = GAMMA_HZ_PER_T * 1e-6;   % Hz   -> uT

    adc_dwell_ms = double(truth.meta.adc_dwell_ns) * 1e-6;  % ns→ms
    n_adc        = double(truth.meta.adc_samples);
    adc_dur_all  = n_adc .* adc_dwell_ms;  % per-definition durations (ms)

    if isempty(p.Results.segment_idx)
        seg_ids = 1:truth.segment_def.num_segments;
    else
        seg_ids = p.Results.segment_idx(:).';
    end

    figs       = gobjects(1, numel(seg_ids));
    bot_margin = 0.09;

    for ii = 1:numel(seg_ids)
        s = seg_ids(ii);
        if s < 1 || s > truth.segment_def.num_segments
            error('truth:input', 'segment_idx %d out of range', s);
        end

        seg      = truth.segment_def.segments(s);
        n_blocks = seg.num_blocks;
        cmap     = lines(n_blocks);
        fig      = figure('Name', sprintf('Segment %d: %s', s - 1, truth.base_name), 'Color', 'w');
        ax_h     = gobjects(1, 5);

        % Compute block start times (ms) from actual event endpoints.
        % grad_delay and rf_delay stored in seconds → convert to ms.
        % grad_time_s is 0-based local sample time in seconds.
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
        % Add a 0.002 ms gap between blocks for visual separation.
        block_start_ms = [0, cumsum(block_end_ms + 0.002)];

        % --- Gradient channels ---
        grad_ylabels = {'Gx (mT/m)', 'Gy (mT/m)', 'Gz (mT/m)'};
        for ch = 1:3
            ax_h(ch) = subplot(5, 1, ch);
            hold on;
            for b = 1:n_blocks
                blk = seg.blocks(b);
                n   = double(blk.grad_n(ch));
                if n > 0
                    y    = double(blk.grad_amp(ch)) * double(blk.grad_wave{ch}(:)) / GAMMA_GRAD;
                    t_ms = block_start_ms(b) + double(blk.grad_delay(ch)) * 1e3 + ...
                           double(blk.grad_time_s{ch}(:)) * 1e3;
                    plot(t_ms, y, 'Color', cmap(b,:), 'LineWidth', 1.1);
                end
            end
            hold off;
            ylabel(grad_ylabels{ch});
            grid on;
        end

        % --- RF |B1| ---
        ax_h(4) = subplot(5, 1, 4);
        hold on;
        for b = 1:n_blocks
            blk  = seg.blocks(b);
            rf_n = double(blk.rf_n);
            if rf_n > 0
                y    = double(blk.rf_amp) * double(blk.rf_rho(:)) / GAMMA_RF;
                t_ms = block_start_ms(b) + double(blk.rf_delay) * 1e3 + ...
                       double(blk.rf_time_s(:)) * 1e3;
                plot(t_ms, y, 'Color', cmap(b,:), 'LineWidth', 1.1);
            end
        end
        hold off;
        ylabel('RF |B1| (uT)');
        grid on;

        % --- ADC mask ---
        ax_h(5) = subplot(5, 1, 5);
        hold on;
        for b = 1:n_blocks
            blk = seg.blocks(b);
            if blk.has_adc
                ad = double(blk.adc_def_id) + 1;  % 0-based -> 1-based
                t0 = block_start_ms(b) + double(blk.adc_delay) * 1e3;
                t1 = t0 + adc_dur_all(ad);
                fill([t0, t1, t1, t0], [0, 0, 1, 1], cmap(b,:), ...
                    'FaceAlpha', 0.45, 'EdgeColor', 'none');
            end
        end
        hold off;
        ylim([-0.1, 1.5]);
        yticks([0, 1]);
        yticklabels({'0', '1'});
        ylabel('ADC');
        xlabel('Time (ms)');
        grid on;

        linkaxes(ax_h, 'x');

        % Compress subplots to leave room for the global legend.
        for ch = 1:5
            pos    = get(ax_h(ch), 'Position');
            pos(2) = pos(2) * (1 - bot_margin) + bot_margin;
            pos(4) = pos(4) * (1 - bot_margin);
            set(ax_h(ch), 'Position', pos);
        end

        % Global horizontal legend at bottom.
        lgd_ax = axes('Parent', fig, ...
            'Position', [0.08, 0.005, 0.84, bot_margin - 0.01], ...
            'Visible', 'off');
        hold(lgd_ax, 'on');
        dummy_h    = gobjects(1, n_blocks);
        block_lbls = cell(1, n_blocks);
        for b = 1:n_blocks
            block_lbls{b} = sprintf('B%d', b - 1);
            dummy_h(b)    = plot(lgd_ax, NaN, NaN, '-', 'Color', cmap(b,:), 'LineWidth', 1.1);
        end
        legend(lgd_ax, dummy_h, block_lbls, ...
            'Orientation', 'horizontal', 'Location', 'north', 'Box', 'off');

        sgtitle(fig, sprintf('Segment %d (%s)  rf->adc gap %.1fus, adc->adc gap %.1fus', ...
            s - 1, truth.base_name, seg.rf_adc_gap_us, seg.adc_adc_gap_us), 'Interpreter', 'none');
        figs(ii) = fig;
    end
end

function truth = coerce_truth(base_or_truth)
    if isstruct(base_or_truth)
        truth = base_or_truth;
    else
        truth = testutils.truth_parse_case(base_or_truth);
    end
end

