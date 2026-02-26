function info = report(seq, varargin)
% report - Structured report of the sequence collection.
%
%   info = pulserver.report(seq)
%   str  = pulserver.report(seq, 'print', true)
%
% Inputs
%   seq     double | SequenceCollection   Collection handle or object.
%
% Name-value options
%   print   logical     If true, return a formatted string. Default: false
%
% Output
%   info    struct array (1 x num_subsequences) with fields:
%       tr_size              - blocks per TR
%       num_trs              - total TRs
%       tr_duration_us       - TR duration (us)
%       num_prep_blocks      - preparation blocks
%       num_cooldown_blocks  - cooldown blocks
%       num_unique_segments  - unique segments
%       segments             - struct array of (start_block, num_blocks)
%       prep_segment_table   - segment IDs for prep region
%       main_segment_table   - segment IDs for main TR
%       cooldown_segment_table - segment IDs for cooldown
%
% See also: pulserver.SequenceCollection, pulserver.get_block

    p = inputParser;
    addParameter(p, 'print', false, @islogical);
    parse(p, varargin{:});

    info = pulseqlib_mex('report', to_handle(seq));

    if p.Results.print
        nss = numel(info);
        lines = {};
        lines{end+1} = sprintf('Subsequences: %d', nss);
        lines{end+1} = '';
        for ss = 1:nss
            s = info(ss);
            lines{end+1} = sprintf('--- Subsequence %d ---', ss);
            lines{end+1} = sprintf('  TR size:            %d blocks', s.tr_size);
            lines{end+1} = sprintf('  TR duration:        %.3f ms', s.tr_duration_us / 1e3);
            lines{end+1} = sprintf('  Prep blocks:        %d', s.num_prep_blocks);
            lines{end+1} = sprintf('  Cooldown blocks:    %d', s.num_cooldown_blocks);
            lines{end+1} = sprintf('  Unique segments:    %d', s.num_unique_segments);
            lines{end+1} = sprintf('  Segment order (TR): [%s]', ...
                            num2str(s.main_segment_table));
            for si = 1:numel(s.segments)
                lines{end+1} = sprintf('    seg %d: start_block=%d, num_blocks=%d', ...
                    si, s.segments(si).start_block, s.segments(si).num_blocks);
            end
            lines{end+1} = '';
        end
        info = strjoin(lines, newline);
    end
end

function h = to_handle(seq)
    if isa(seq, 'pulserver.SequenceCollection')
        h = seq.Handle;
    else
        h = seq;
    end
end
