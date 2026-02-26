function deserialize(seq, path)
% deserialize - Restore a collection from a previously serialized binary cache.
%
%   pulserver.deserialize(seq, 'output.bin')
%
% Inputs
%   seq     double | SequenceCollection   Collection handle or object.
%   path    char        Path to the binary cache file.
%
% See also: pulserver.serialize, pulserver.SequenceCollection

    pulseqlib_mex('load_cache', to_handle(seq), path);
end

function h = to_handle(seq)
    if isa(seq, 'pulserver.SequenceCollection')
        h = seq.Handle;
    else
        h = seq;
    end
end
