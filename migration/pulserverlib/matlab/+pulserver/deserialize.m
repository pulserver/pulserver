function seqs = deserialize(path)
% deserialize - Read a linked chain of .seq files into a cell array.
%
%   seqs = pulserver.deserialize('scan_001.seq')
%
%   Starting from the given path, each file is read with seq.read().
%   If the [DEFINITIONS] section contains a 'next' key the referenced
%   file (resolved relative to the directory of the current file) is
%   loaded as well, and so on until no 'next' key is found.
%
% Inputs
%   path    char   Path to the first .seq file in the chain.
%
% Outputs
%   seqs    cell array of mr.Sequence   Ordered list of loaded sequences.
%
% See also: pulserver.serialize

    seqs = {};
    current = path;

    while ~isempty(current)
        if ~isfile(current)
            error('pulserver:deserialize', ...
                  'Sequence file not found: %s', current);
        end

        seq = mr.Sequence();
        seq.read(current);
        seqs{end+1} = seq; %#ok<AGROW>

        % Follow the "next" link if present
        if isfield(seq.definitions, 'next') || ...
           (isa(seq.definitions, 'containers.Map') && isKey(seq.definitions, 'next'))
            if isa(seq.definitions, 'containers.Map')
                nextName = seq.definitions('next');
            else
                nextName = seq.definitions.next;
            end
            folder = fileparts(current);
            current = fullfile(folder, nextName);
        else
            current = '';
        end
    end
end
