function result = pns(seq, varargin)
% pns - Peripheral nerve stimulation analysis.
%
%   result = pulserver.pns(seq, 'chronaxie_us', 360, 'rheobase', 20, 'alpha', 0.333)
%
% Inputs
%   seq     double | SequenceCollection   Collection handle or object.
%
% Required name-value parameters
%   chronaxie_us   double   Nerve time constant (us).
%   rheobase       double   Threshold slew rate (Hz/m/s).
%   alpha          double   Effective coil length (m).
%
% Output
%   result  struct with fields: num_samples, slew_x, slew_y, slew_z.
%
% See also: pulserver.grad_spectrum, pulserver.check

    h = to_handle(seq);

    p = inputParser;
    addParameter(p, 'chronaxie_us', [], @isnumeric);
    addParameter(p, 'rheobase',     [], @isnumeric);
    addParameter(p, 'alpha',        [], @isnumeric);
    parse(p, varargin{:});
    o = p.Results;

    if isempty(o.chronaxie_us) || isempty(o.rheobase) || isempty(o.alpha)
        error('pulserver:pns', ...
            'pns requires chronaxie_us, rheobase, and alpha parameters.');
    end

    result = pulseqlib_mex('pns', h, o.chronaxie_us, o.rheobase, o.alpha);
end

function h = to_handle(seq)
    if isa(seq, 'pulserver.SequenceCollection')
        h = seq.Handle;
    else
        h = seq;
    end
end
