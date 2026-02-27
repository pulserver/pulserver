function grad_spectrum_impl(seq, varargin)
% grad_spectrum - Plot acoustic spectral analysis of gradient waveforms.
%
%   pulserver.grad_spectrum(seq)
%   pulserver.grad_spectrum(seq, 'forbidden_bands', [500 600 1e4; 1000 1200 5e3])
%
% Creates a two-row figure: sliding-window spectrograms (top),
% harmonic spectra with forbidden-band overlays (bottom).
% No pass/fail check -- use check() for that.
%
% Inputs
%   seq     double | SequenceCollection   Collection handle or object.
%
% Name-value options
%   sequence_idx       double  Subsequence (1-based). Default: 1.
%   forbidden_bands    Nx3     Each row [freq_min freq_max max_ampl] (Hz,Hz,Hz/m).
%   window_duration    double  Window size (s). Default: 25e-3.
%   spectral_resolution double Target freq resolution (Hz). Default: 5.
%   max_frequency      double  Max frequency (Hz). Default: 3000.
%
% See also: pulserver.pns, pulserver.SequenceCollection.check

    h = to_handle(seq);

    p = inputParser;
    addParameter(p, 'sequence_idx',        1,      @isnumeric);
    addParameter(p, 'forbidden_bands',     [],     @isnumeric);
    addParameter(p, 'window_duration',     25e-3,  @isnumeric);
    addParameter(p, 'spectral_resolution', 5.0,    @isnumeric);
    addParameter(p, 'max_frequency',       3000.0, @isnumeric);
    parse(p, varargin{:});
    o = p.Results;

    win_sz = round(o.window_duration / 10e-6);  % half-raster samples

    result = pulseqlib_mex('grad_spectrum', h, o.sequence_idx - 1, ...
        win_sz, o.spectral_resolution, o.max_frequency);

    nf = result.num_freq_bins;
    nw = result.num_windows;
    frequencies = result.freq_min_hz + (0:nf-1) * result.freq_spacing_hz;

    bands = o.forbidden_bands;
    if isempty(bands)
        bands = zeros(0, 3);
    end

    axis_names = {'gx', 'gy', 'gz'};
    axis_labels = {'Gx', 'Gy', 'Gz'};
    colors = {[0.122 0.467 0.706], [1.000 0.498 0.055], [0.173 0.627 0.173]};

    figure;

    % ── Top row: spectrograms ──
    for col = 1:3
        ax = subplot(2, 3, col);
        sg = result.(sprintf('spectrogram_%s', axis_names{col}));
        pk = result.(sprintf('peaks_%s', axis_names{col}));
        imagesc(ax, frequencies, 0:nw-1, sg);
        axis(ax, 'xy');
        colormap(ax, 'parula');

        hold(ax, 'on');
        [pr, pc_idx] = find(pk > 0);
        if ~isempty(pr)
            plot(ax, frequencies(pc_idx), pr-1, 'r*', 'MarkerSize', 8);
        end
        for bi = 1:size(bands, 1)
            xline(ax, bands(bi, 1), 'w--', 'LineWidth', 1.2, 'Alpha', 0.8);
            xline(ax, bands(bi, 2), 'w--', 'LineWidth', 1.2, 'Alpha', 0.8);
        end
        hold(ax, 'off');
        xlim(ax, [0 frequencies(end)]);
        xlabel(ax, 'Frequency (Hz)');
        ylabel(ax, 'Window Index');
        title(ax, sprintf('%s Spectrogram', axis_labels{col}));
    end

    % ── Bottom row: harmonic spectra + forbidden bands ──
    for col = 1:3
        ax = subplot(2, 3, 3 + col);
        sg = result.(sprintf('spectrogram_%s', axis_names{col}));
        spec_full = result.(sprintf('spectrum_full_%s', axis_names{col}));
        pk_full   = result.(sprintf('peaks_full_%s', axis_names{col}));
        max_env   = max(sg, [], 1);

        plot(ax, frequencies, max_env, 'Color', colors{col}, 'LineWidth', 1.5); hold(ax, 'on');
        plot(ax, frequencies, spec_full, 'Color', colors{col}, ...
             'LineWidth', 0.8, 'LineStyle', '-', 'Color', [colors{col} 0.5]);
        pk_idx = find(pk_full > 0);
        if ~isempty(pk_idx)
            plot(ax, frequencies(pk_idx), spec_full(pk_idx), 'o', ...
                 'Color', colors{col}, 'MarkerSize', 5, 'MarkerFaceColor', colors{col});
        end
        for bi = 1:size(bands, 1)
            patch(ax, [bands(bi,1) bands(bi,2) bands(bi,2) bands(bi,1)], ...
                  [0 0 1 1]*max(max_env)*1.1, 'r', ...
                  'FaceAlpha', 0.12, 'EdgeColor', 'none');
            yline(ax, bands(bi,3), 'r--', 'LineWidth', 1.2);
        end
        hold(ax, 'off');
        xlim(ax, [0 frequencies(end)]);
        xlabel(ax, 'Frequency (Hz)');
        ylabel(ax, 'Magnitude (a.u.)');
        title(ax, sprintf('%s Harmonic Spectrum', axis_labels{col}));
        legend(ax, {'Max envelope', 'Full TR'}, 'FontSize', 8, 'Location', 'northeast');
        grid(ax, 'on'); set(ax, 'GridAlpha', 0.3);
    end
end

function h = to_handle(seq)
    if isa(seq, 'pulserver.SequenceCollection')
        h = seq.Handle;
    else
        h = seq;
    end
end
