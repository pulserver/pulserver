function check(seq, varargin)
% check - Run consistency and safety checks on a loaded sequence.
%
%   check(seq)
%   check(seq, 'stim_threshold', 23.4, 'decay_constant_us', 334)
%   check(seq, 'forbidden_bands', [500 600 1e4; 1000 1200 5e3])
%
% Inputs
%   seq     double | SequenceCollection   Collection handle or object.
%
% Name-value options
%   stim_threshold        double  PNS stimulation threshold
%                                 (Hz/m/s) = rheobase/alpha.
%                                 Default: 0 (skip PNS).
%   decay_constant_us     double  PNS chronaxie (us). Default: 0.
%   forbidden_bands       Nx3     [freq_min freq_max max_ampl] (Hz,Hz,Hz/m).
%   pns_threshold_percent double  PNS threshold (%). Default: 100.
%
% Raises an error if any check fails. Returns silently on success.

    h = to_handle(seq);

    p = inputParser;
    addParameter(p, 'stim_threshold',        0,   @isnumeric);
    addParameter(p, 'decay_constant_us',     0,   @isnumeric);
    addParameter(p, 'forbidden_bands',       [],  @isnumeric);
    addParameter(p, 'pns_threshold_percent', 100, @isnumeric);
    parse(p, varargin{:});
    o = p.Results;

    bands = o.forbidden_bands;
    if isempty(bands)
        bands = zeros(0, 3);
    end

    pulseqlib_mex('check', h, ...
        o.stim_threshold, o.decay_constant_us, ...
        o.pns_threshold_percent, bands);
end

function h = to_handle(seq)
    if isa(seq, 'pulserver.SequenceCollection')
        h = seq.Handle;
    else
        h = seq;
    end
end
