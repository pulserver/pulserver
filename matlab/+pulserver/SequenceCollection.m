classdef SequenceCollection < handle
% SequenceCollection  Parsed Pulseq collection with analysis methods.
%
%   sc = pulserver.SequenceCollection(seq)
%   sc = pulserver.SequenceCollection({seq1, seq2, seq3})
%   sc = pulserver.SequenceCollection('scan_001.seq')
%   sc = pulserver.SequenceCollection(seq, 'parse_labels', true)
%
%   Accepts:
%     - A single mr.Sequence object (auto-wrapped in a cell).
%     - A cell array of mr.Sequence objects.
%     - A char/string path to the first .seq file in a linked chain
%       (files reference each other via the 'next' definition key).
%
%   System parameters (gamma, B0, gradient limits, raster times)
%   are read from the first sequence's seq.sys automatically.
%
%   This is a handle object: when the last reference goes out of scope
%   the underlying C collection is freed automatically.
%
% Construction arguments
%   seq    mr.Sequence | cell array of mr.Sequence | char/string
%
% Optional name-value pairs
%   parse_labels   logical  Parse label extensions.  Default: true
%   num_averages   double   Number of averages.      Default: 1
%
% Properties
%   NumSequences   Number of sequences in the collection.
%
% Methods
%   get_sequence       - Return the mr.Sequence at a given index.
%   get_block          - Per-block metadata access.
%   plot               - Plot waveforms for one TR.
%   check              - Run consistency and safety checks.
%   validate           - Compare against Pulseq MATLAB toolbox.
%   report             - Structured collection summary.
%   pns                - Peripheral nerve stimulation analysis.
%   grad_spectrum      - Acoustic spectral analysis.
%
% See also: pulserver.report, pulserver.serialize, pulserver.deserialize

    properties (SetAccess = private)
        Handle  double = 0     % 1-based MEX collection handle (0 = invalid)
    end

    properties (SetAccess = private, Hidden)
        Seqs  cell = {}        % Stored mr.Sequence objects
    end

    properties (Dependent)
        NumSequences           % Number of sequences in the collection
    end

    methods
        function obj = SequenceCollection(seq, varargin)
        % Construct a SequenceCollection from mr.Sequence object(s) or file.
        %
        %   sc = pulserver.SequenceCollection(seq)
        %   sc = pulserver.SequenceCollection({seq1, seq2})
        %   sc = pulserver.SequenceCollection('scan_001.seq')
        %   sc = pulserver.SequenceCollection(seq, 'num_averages', 2)

            p = inputParser;
            addRequired(p, 'seq');
            addParameter(p, 'parse_labels', true, @islogical);
            addParameter(p, 'num_averages', 1,    @isnumeric);
            parse(p, seq, varargin{:});
            o = p.Results;

            % ── Normalise input to cell array of mr.Sequence ────
            if ischar(seq) || isstring(seq)
                seqs = pulserver.deserialize(char(seq));
            elseif iscell(seq)
                seqs = seq;
            else
                seqs = {seq};
            end

            % Deep-copy so we never mutate the caller's objects
            for k = 1:numel(seqs)
                seqs{k} = copy(seqs{k});
            end
            obj.Seqs = seqs;

            % Read system from first sequence
            s = seqs{1}.sys;

            % Serialize each sequence via temp file → byte buffer
            bufs = cell(1, numel(seqs));
            for k = 1:numel(seqs)
                tmpFile = [tempname '.seq'];
                cleanObj = onCleanup(@() delete_if_exists(tmpFile));
                seqs{k}.write(tmpFile);
                fid = fopen(tmpFile, 'r');
                if fid < 0
                    error('pulserver:load', ...
                        'Failed to serialize sequence %d.', k);
                end
                bufs{k} = fread(fid, Inf, '*uint8');
                fclose(fid);
            end

            obj.Handle = pulseqlib_mex('load', bufs, ...
                s.gamma, s.B0, s.maxGrad, s.maxSlew, ...
                s.rfRasterTime, s.gradRasterTime, ...
                s.adcRasterTime, s.blockDurationRaster, ...
                o.parse_labels, o.num_averages);
        end

        function delete(obj)
        % Destructor — free the underlying C collection.
            if obj.Handle > 0
                pulseqlib_mex('free', obj.Handle);
                obj.Handle = 0;
            end
        end

        % ── Sequence-list accessors ──────────────────────────────

        function n = get.NumSequences(obj)
        % NumSequences  Number of sequences in the collection.
            n = numel(obj.Seqs);
        end

        function seq = get_sequence(obj, idx)
        % get_sequence  Return the mr.Sequence at a given index.
        %
        %   seq = sc.get_sequence(idx)
        %
        %   idx  1-based sequence index.
        %
        % Returns a deep copy of the stored sequence.
            if idx < 1 || idx > numel(obj.Seqs)
                error('pulserver:index', ...
                    'idx %d out of range (NumSequences=%d)', ...
                    idx, numel(obj.Seqs));
            end
            seq = copy(obj.Seqs{idx});
        end

        % ── Analysis methods ─────────────────────────────────────

        % ── Unique-block accessors ────────────────────────────

        function n = num_blocks(obj, sequence_idx)
        % num_blocks  Number of unique blocks in a subsequence.
        %
        %   n = sc.num_blocks(sequence_idx)
        %
        %   sequence_idx  1-based subsequence index (default 1).
            if nargin < 2; sequence_idx = 1; end
            n = pulseqlib_mex('num_unique_blocks', obj.Handle, sequence_idx - 1);
        end

        function blk = get_block(obj, sequence_idx, block_idx)
        % get_block  Return the block_idx-th unique block, normalised.
        %
        %   blk = sc.get_block(sequence_idx, block_idx)
        %
        %   sequence_idx  1-based subsequence index.
        %   block_idx     1-based unique-block index within sequence_idx.
        %
        % The block is fetched from the stored mr.Sequence and normalised:
        %   * RF freqOffset / phaseOffset set to 0; signal scaled to
        %     unit peak.
        %   * Gradient arbitrary waveforms scaled to unit peak;
        %     trapezoid amplitudes set to +/-1.
        %   * ADC freqOffset / phaseOffset set to 0.
            block_id = pulseqlib_mex('unique_block_id', obj.Handle, ...
                                     sequence_idx - 1, block_idx - 1);
            blk = obj.Seqs{sequence_idx}.getBlock(block_id);
            blk = normalize_block(blk);
        end

        % ── Segment / TR accessors ────────────────────────────

        function n = num_segments(obj, sequence_idx)
        % num_segments  Number of unique segments in a subsequence.
        %
        %   n = sc.num_segments(sequence_idx)
        %
        %   sequence_idx  1-based subsequence index (default 1).
            if nargin < 2; sequence_idx = 1; end
            r = report_impl(obj.Handle);
            ss = r(sequence_idx);
            n = ss.num_unique_segments;
        end

        function n = segment_size(obj, sequence_idx, segment_idx)
        % segment_size  Number of blocks in a segment.
        %
        %   n = sc.segment_size(sequence_idx, segment_idx)
        %
        %   sequence_idx  1-based subsequence index.
        %   segment_idx   1-based local segment index.
            r = report_impl(obj.Handle);
            ss = r(sequence_idx);
            segs = ss.segments;
            seg = segs(segment_idx);
            n = seg.num_blocks;
        end

        function seg_seq = get_segment(obj, sequence_idx, segment_idx)
        % get_segment  Extract a segment as a normalised mr.Sequence.
        %
        %   seg = sc.get_segment(sequence_idx, segment_idx)
        %
        %   sequence_idx  1-based subsequence index.
        %   segment_idx   1-based local segment index.
        %
        % Uses the C library's start_block and num_blocks for the
        % requested segment to parse consecutive blocks from the
        % stored mr.Sequence, normalises each block, and returns
        % a new mr.Sequence built via addBlock.
            r = report_impl(obj.Handle);
            ss = r(sequence_idx);
            segs = ss.segments;
            seg = segs(segment_idx);
            start_block = seg.start_block;  % 1-based from MEX
            nblk = seg.num_blocks;

            src_seq = obj.Seqs{sequence_idx};
            seg_seq = mr.Sequence(src_seq.sys);
            for i = 0:(nblk - 1)
                blk = src_seq.getBlock(start_block + i);
                seg_seq.addBlock(normalize_block(blk));
            end
        end

        function n = tr_size(obj, sequence_idx)
        % tr_size  Number of blocks per TR in a subsequence.
        %
        %   n = sc.tr_size(sequence_idx)
        %
        %   sequence_idx  1-based subsequence index (default 1).
            if nargin < 2; sequence_idx = 1; end
            r = report_impl(obj.Handle);
            ss = r(sequence_idx);
            n = ss.tr_size;
        end

        function plot(obj, varargin)
        % plot  Plot (3,2) TR waveforms with colour-coded segments.
        %
        %   sc.plot()
        %   sc.plot('sequence_idx', 1, 'tr_instance', 'max_pos')
        %   sc.plot('show_slew', true, 'max_grad_mT_per_m', 40)
        %
        % Displays Gx/Gy/Gz, RF magnitude/phase, and ADC events
        % for one TR, using waveforms from the C library.
        %
        % Name-value options
        %   sequence_idx              int     Subsequence (1-based). Default: 1
        %   tr_instance               int|char TR instance or 'max_pos'/'zero_var'.
        %   hide_prep                 logical Hide prep blocks. Default: true
        %   hide_cooldown             logical Hide cooldown blocks. Default: true
        %   collapse_delays           logical Shrink pure delays. Default: false
        %   show_segments             logical Colour by segment. Default: true
        %   show_blocks               logical Block boundaries. Default: true
        %   show_slew                 logical Overlay slew rate. Default: false
        %   show_rf_centers           logical RF iso-centres. Default: false
        %   show_echoes               logical Echo markers. Default: false
        %   max_grad_mT_per_m         double  Gradient ref line.
        %   max_slew_T_per_m_per_s    double  Slew ref line.
        %   time_unit                 char    'ms' (default) or 'us'.
        %
        % See also: pulserver.SequenceCollection.report
            plot_impl(obj, varargin{:});
        end

        function pns(obj, varargin)
        % pns  Plot convolved PNS waveforms for a representative TR.
        %
        %   sc.pns('stim_threshold', 60, 'decay_constant_us', 360)
        %   sc.pns('stim_threshold', 60, 'decay_constant_us', 360, ...
        %          'threshold_percent', 100, 'sequence_idx', 1)
        %
        % Displays per-axis and combined PNS percentage waveforms
        % with a threshold line.  No pass/fail check -- use check().
        %
        % Name-value options
        %   sequence_idx       int     Subsequence (1-based). Default: 1
        %   stim_threshold     double  PNS threshold (Hz/m/s) = rheobase/alpha.
        %   decay_constant_us  double  Chronaxie (us).
        %   threshold_percent  double  Threshold line (%). Default: 80.
        %
        % See also: pulserver.SequenceCollection.check
            pns_impl(obj, varargin{:});
        end

        function grad_spectrum(obj, varargin)
        % grad_spectrum  Plot acoustic spectra for gradient waveforms.
        %
        %   sc.grad_spectrum()
        %   sc.grad_spectrum('forbidden_bands', [500 600 1e4; 1000 1200 5e3])
        %
        % Creates a 2-row figure: spectrograms (top), harmonic
        % spectra with forbidden bands (bottom).  No pass/fail check.
        %
        % Name-value options
        %   sequence_idx        int     Subsequence (1-based). Default: 1
        %   forbidden_bands     Nx3     [freq_min freq_max max_ampl] (Hz).
        %   window_duration     double  Window size (s). Default: 25e-3.
        %   spectral_resolution double  Freq resolution (Hz). Default: 5.
        %   max_frequency       double  Max freq (Hz). Default: 3000.
        %
        % See also: pulserver.SequenceCollection.check
            grad_spectrum_impl(obj, varargin{:});
        end

        function check(obj, varargin)
        % check  Run consistency and safety checks.
        %
        %   sc.check()
        %   sc.check('stim_threshold', 23.4, 'decay_constant_us', 334)
        %   sc.check('forbidden_bands', [500 600 1e4; 1000 1200 5e3])
        %
        % Checks performed (in order):
        %   1. Consistency (segment boundaries, RF periodicity, labels)
        %   2. Peak gradient amplitude vs system limit
        %   3. Gradient continuity across block boundaries
        %   4. Peak slew-rate vs system limit
        %   5. Acoustic forbidden-band violations (per segment)
        %   6. PNS threshold check (if stim_threshold > 0)
        %
        % Name-value options
        %   stim_threshold        double  PNS stimulation threshold
        %                                 (Hz/m/s) = rheobase/alpha.
        %                                 Default: 0 (skip PNS).
        %   decay_constant_us     double  PNS chronaxie (us).
        %                                 Default: 0.
        %   forbidden_bands       Nx3     Each row is
        %                                 [freq_min freq_max max_ampl]
        %                                 in Hz, Hz, Hz/m.
        %   pns_threshold_percent double  PNS threshold percentage.
        %                                 Default: 100.
        %
        % Raises an error if any check fails.
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

            pulseqlib_mex('check', obj.Handle, ...
                o.stim_threshold, o.decay_constant_us, ...
                o.pns_threshold_percent, bands);
        end

        function result = validate(obj, varargin)
        % validate  Compare C-backend waveforms against reference.
        %
        %   result = sc.validate()
        %   result = sc.validate('do_plot', true)
        %   result = sc.validate('xml_path', 'waveforms.xml')
        %   result = sc.validate('tr_range', [1 3], 'grad_atol', 0.05)
        %
        % See also: pulserver.validate
            result = validate_impl(obj, varargin{:});
        end

        function info = report(obj, varargin)
        % report  Structured collection / subsequence summary.
        %
        %   info = sc.report()
        %   str  = sc.report('print', true)
        %
        % Returns a struct array (1 x num_subsequences) with fields:
        %   num_blocks, segments, num_prep_blocks, num_cooldown_blocks,
        %   tr_size, num_trs, tr_duration_us, num_unique_segments,
        %   segment_order, prep_segment_table, cooldown_segment_table.
        %
        % When 'print' is true, returns a formatted string instead.
        %
        % See also: pulserver.SequenceCollection.get_block,
        %           pulserver.SequenceCollection.num_blocks,
        %           pulserver.SequenceCollection.num_segments
            p = inputParser;
            addParameter(p, 'print', false, @islogical);
            parse(p, varargin{:});

            raw = pulseqlib_mex('report', obj.Handle);

            % Augment each subsequence entry with num_blocks from C getter
            for k = 1:numel(raw)
                raw(k).num_blocks = obj.num_blocks(k);  % 1-based
            end

            if ~p.Results.print
                info = raw;
                return;
            end

            % -- formatted output --
            nss = numel(raw);
            lines = {};
            lines{end+1} = sprintf('Subsequences: %d', nss);
            lines{end+1} = '';
            for ss = 1:nss
                s = raw(ss);
                lines{end+1} = sprintf('--- Subsequence %d ---', ss);
                lines{end+1} = sprintf('  TR size:            %d blocks', s.tr_size);
                lines{end+1} = sprintf('  TR duration:        %.3f ms', s.tr_duration_us / 1e3);
                lines{end+1} = sprintf('  Prep blocks:        %d', s.num_prep_blocks);
                lines{end+1} = sprintf('  Cooldown blocks:    %d', s.num_cooldown_blocks);
                lines{end+1} = sprintf('  Unique blocks:      %d', s.num_blocks);
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
end

function delete_if_exists(f)
% Clean up temp file if it still exists.
    if exist(f, 'file')
        delete(f);
    end
end

function blk = normalize_block(blk)
% normalize_block  Zero variable params, scale waveforms to unit peak.
%
%   RF: freqOffset/phaseOffset -> 0, signal -> signal/max(|signal|)
%   Gradients: arb waveform -> waveform/max(|waveform|),
%              trap amplitude -> sign(amplitude)
%   ADC: freqOffset/phaseOffset -> 0

    % -- RF -------------------------------------------------------
    if isfield(blk, 'rf') && ~isempty(blk.rf)
        blk.rf.freqOffset  = 0;
        blk.rf.phaseOffset = 0;
        peak = max(abs(blk.rf.signal));
        if peak > 0
            blk.rf.signal = blk.rf.signal / peak;
        end
    end

    % -- Gradients ------------------------------------------------
    for ax = {'gx', 'gy', 'gz'}
        a = ax{1};
        if ~isfield(blk, a) || isempty(blk.(a)); continue; end
        g = blk.(a);
        if isfield(g, 'waveform') && ~isempty(g.waveform)
            peak = max(abs(g.waveform));
            if peak > 0
                g.waveform = g.waveform / peak;
            end
        elseif isfield(g, 'amplitude') && g.amplitude ~= 0
            g.amplitude = sign(g.amplitude);
        end
        blk.(a) = g;
    end

    % -- ADC ------------------------------------------------------
    if isfield(blk, 'adc') && ~isempty(blk.adc)
        if isfield(blk.adc, 'freqOffset')
            blk.adc.freqOffset = 0;
        end
        if isfield(blk.adc, 'phaseOffset')
            blk.adc.phaseOffset = 0;
        end
    end
end
