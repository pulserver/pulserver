function check(seq)
% check - Run consistency and safety checks on a loaded sequence.
%
%   pulserver.check(seq)
%
% Inputs
%   seq     double | SequenceCollection   Collection handle or object.
%
% Raises an error if any check fails. Returns silently on success.
%
% See also: pulserver.SequenceCollection, pulserver.validate

    pulseqlib_mex('check', to_handle(seq));
end

function h = to_handle(seq)
    if isa(seq, 'pulserver.SequenceCollection')
        h = seq.Handle;
    else
        h = seq;
    end
end
