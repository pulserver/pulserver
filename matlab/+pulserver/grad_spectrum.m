function result = grad_spectrum(seq, varargin)
% grad_spectrum - Acoustic spectral analysis of gradient waveforms.
%
%   result = pulserver.grad_spectrum(seq)
%   result = pulserver.grad_spectrum(seq, 'max_frequency', 2000)
%
% Inputs
%   seq     double | SequenceCollection   Collection handle or object.
%
% Name-value options
%   window_size          double  Target window samples. Default: 0 (auto).
%   window_duration      double  Window duration in s.  Default: 25e-3
%   spectral_resolution  double  Target freq resolution (Hz). Default: 5
%   max_frequency        double  Max frequency (Hz). Default: 3000
%
% Output
%   result  struct with spectrogram / spectrum arrays per axis, including:
%       freq_min_hz, freq_spacing_hz, num_freq_bins, num_windows,
%       spectrogram_gx/gy/gz, peaks_gx/gy/gz,
%       spectrum_full_gx/gy/gz, peaks_full_gx/gy/gz,
%       and (when num_trs > 1) spectrum_seq_gx/gy/gz, peaks_seq_gx/gy/gz.
%
% See also: pulserver.pns, pulserver.check

    h = to_handle(seq);

    p = inputParser;
    addParameter(p, 'window_size',         0,      @isnumeric);
    addParameter(p, 'window_duration',     25e-3,  @isnumeric);
    addParameter(p, 'spectral_resolution', 5.0,    @isnumeric);
    addParameter(p, 'max_frequency',       3000.0, @isnumeric);
    parse(p, varargin{:});
    o = p.Results;

    win_sz = o.window_size;
    if win_sz == 0
        % Auto: derive from window_duration and grad_raster (10 us)
        win_sz = round(o.window_duration / 10e-6);
    end

    result = pulseqlib_mex('grad_spectrum', h, ...
        win_sz, o.spectral_resolution, o.max_frequency);
end

function h = to_handle(seq)
    if isa(seq, 'pulserver.SequenceCollection')
        h = seq.Handle;
    else
        h = seq;
    end
end
