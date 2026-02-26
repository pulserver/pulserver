function serialize(seq, path)
% serialize - Save the parsed collection to a binary cache file.
%
%   pulserver.serialize(seq, 'output.bin')
%
% Inputs
%   seq     double | SequenceCollection   Collection handle or object.
%   path    char        Output file path.
%
% See also: pulserver.deserialize, pulserver.SequenceCollection

    pulseqlib_mex('save_cache', to_handle(seq), path);
end

function h = to_handle(seq)
    if isa(seq, 'pulserver.SequenceCollection')
        h = seq.Handle;
    else
        h = seq;
    end
end
