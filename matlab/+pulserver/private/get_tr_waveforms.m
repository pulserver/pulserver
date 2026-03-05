function wf = get_tr_waveforms(seq, varargin)
% get_tr_waveforms - Extract native-timing waveforms for one TR.
%
%   wf = get_tr_waveforms(seq)
%   wf = get_tr_waveforms(seq, 'amplitude_mode', 'actual', ...)
%
% Inputs
%   seq     double | SequenceCollection   Collection handle or object.
%
% Name-value options
%   subsequence_idx  double   Subsequence index (1-based). Default: 1
%   amplitude_mode   char     'max_pos' (0) | 'zero_var' (1) | 'actual' (2).
%                             Default: 'max_pos'
%   tr_index         double   TR index (1-based). Default: 1
%   include_prep     logical  Include preparation blocks. Default: false
%   include_cooldown logical  Include cooldown blocks. Default: false
%   collapse_delays  logical  Shrink pure delays. Default: false
%
% Output
%   wf      struct with fields: gx, gy, gz, rf_mag, rf_phase, total_duration_us.
%           Each channel field is a struct with .time_us and .amplitude.
%
% See also: pulserver.SequenceCollection, pulserver.plot

    p = inputParser;
    addParameter(p, 'subsequence_idx', 1,        @isnumeric);
    addParameter(p, 'amplitude_mode', 'max_pos', @ischar);
    addParameter(p, 'tr_index',       1,         @isnumeric);
    addParameter(p, 'include_prep',   false,     @islogical);
    addParameter(p, 'include_cooldown', false,    @islogical);
    addParameter(p, 'collapse_delays',  false,    @islogical);
    parse(p, varargin{:});
    o = p.Results;

    modes = containers.Map( ...
        {'max_pos','zero_var','actual'}, ...
        {0,        1,         2});
    if ~modes.isKey(o.amplitude_mode)
        error('pulserver:get_tr_waveforms', ...
              'amplitude_mode must be ''max_pos'', ''zero_var'', or ''actual''');
    end
    amp_mode = modes(o.amplitude_mode);

    wf = pulseqlib_mex('get_tr_waveforms', to_handle(seq), ...
                        o.subsequence_idx - 1, amp_mode, ...
                        o.tr_index - 1, o.include_prep, o.include_cooldown, ...
                        o.collapse_delays);
end

function h = to_handle(seq)
    if isa(seq, 'pulserver.SequenceCollection')
        h = seq.Handle;
    else
        h = seq;
    end
end
