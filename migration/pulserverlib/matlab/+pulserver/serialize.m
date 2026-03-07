function filenames = serialize(seqs, path)
% serialize - Write a sequence collection as a linked chain of .seq files.
%
%   filenames = pulserver.serialize(seqs, 'scan.seq')
%   filenames = pulserver.serialize(seqs, 'scan')
%
%   Each sequence is written to its own .seq file.  A 'next' key is added
%   to the [DEFINITIONS] section of every file except the last, pointing
%   to the filename of the next file in the chain.
%
%   For a single sequence the output is <path>.seq.  For multiple
%   sequences the outputs are <path>_001.seq, <path>_002.seq, etc.
%
% Inputs
%   seqs    cell array of mr.Sequence objects.
%   path    char   Base file path (.seq suffix is optional).
%
% Outputs
%   filenames   cell array of char   Written file paths, in order.
%
% See also: pulserver.deserialize

    % Strip .seq extension if present
    [folder, stem, ext] = fileparts(path);
    if strcmpi(ext, '.seq')
        basepath = fullfile(folder, stem);
    else
        basepath = fullfile(folder, [stem ext]);
    end

    n = numel(seqs);
    if n == 0
        error('pulserver:serialize', 'seqs must be a non-empty cell array.');
    end

    % Build filenames
    if n == 1
        filenames = {[basepath '.seq']};
    else
        filenames = cell(1, n);
        for i = 1:n
            filenames{i} = sprintf('%s_%03d.seq', basepath, i);
        end
    end

    % Write each sequence with "next" link
    for i = 1:n
        seq = copy(seqs{i});  % deep copy to avoid mutating input
        if i < n
            [~, nextName, nextExt] = fileparts(filenames{i+1});
            seq.setDefinition('next', [nextName nextExt]);
        end
        seq.write(filenames{i});
    end
end
