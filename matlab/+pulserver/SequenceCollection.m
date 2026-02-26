classdef SequenceCollection < handle
% SequenceCollection  Parsed Pulseq collection with analysis methods.
%
%   sc = pulserver.SequenceCollection(seq)
%   sc = pulserver.SequenceCollection(seq, 'parse_labels', true)
%
%   Accepts an mr.Sequence object.  System parameters (gamma, B0,
%   gradient limits, raster times) are read from seq.sys automatically
%   — just like the Python constructor reads from seq.system.
%
%   This is a handle object: when the last reference goes out of scope
%   the underlying C collection is freed automatically.
%
% Construction arguments
%   seq            mr.Sequence  A Pulseq MATLAB toolbox sequence object.
%
% Optional name-value pairs
%   parse_labels   logical  Parse label extensions.  Default: true
%   num_averages   double   Number of averages.      Default: 1
%
% Methods
%   get_block          - Per-block metadata access.
%   plot               - Plot waveforms for one TR.
%   check              - Run consistency and safety checks.
%   validate           - Compare against Pulseq MATLAB toolbox.
%   report             - Structured collection summary.
%   pns                - Peripheral nerve stimulation analysis.
%   grad_spectrum      - Acoustic spectral analysis.
%   serialize          - Save to binary cache.
%   deserialize        - Restore from binary cache.
%
% See also: pulserver.report

    properties (SetAccess = private)
        Handle  double = 0     % 1-based MEX collection handle (0 = invalid)
    end

    methods
        function obj = SequenceCollection(seq, varargin)
        % Construct a SequenceCollection from an mr.Sequence object.
        %
        %   sc = pulserver.SequenceCollection(seq)
        %   sc = pulserver.SequenceCollection(seq, 'num_averages', 2)
        %
        %   System parameters are read from seq.sys (mr.opts struct).

            p = inputParser;
            addRequired(p, 'seq');
            addParameter(p, 'parse_labels', true, @islogical);
            addParameter(p, 'num_averages', 1,    @isnumeric);
            parse(p, seq, varargin{:});
            o = p.Results;

            % Read system from seq.sys (mr.opts struct)
            s = seq.sys;

            % Serialize via temp file
            tmpFile = [tempname '.seq'];
            cleanObj = onCleanup(@() delete_if_exists(tmpFile));
            seq.write(tmpFile);
            fid = fopen(tmpFile, 'r');
            if fid < 0
                error('pulserver:load', 'Failed to serialize sequence.');
            end
            buf = fread(fid, Inf, '*uint8');
            fclose(fid);

            obj.Handle = pulseqlib_mex('load', buf, ...
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

        % ── Analysis methods ─────────────────────────────────────

        function info = get_block(obj, segment_idx, block_idx)
        % get_block  Return metadata for a single base block.
        %
        %   info = seq.get_block(segment_idx, block_idx)
        %
        %   segment_idx  0-based segment index.
        %   block_idx    0-based block index within the segment.
        %
        % See also: pulserver.SequenceCollection.report
            info = pulseqlib_mex('get_block', obj.Handle, ...
                                 segment_idx, block_idx);
        end

        function fig = plot(obj, varargin)
        % plot  Plot waveforms for one TR.
        %
        %   seq.plot()
        %   seq.plot('collapse_delays', true, 'collapsed_duration_us', 50)
        %   fig = seq.plot('time_unit', 'us');
        %
        % See also: pulserver.SequenceCollection.report
            if nargout > 0
                fig = pulserver.plot(obj.Handle, varargin{:});
            else
                pulserver.plot(obj.Handle, varargin{:});
            end
        end

        function check(obj)
        % check  Run consistency and safety checks.
        %
        %   seq.check()
        %
        % Raises an error if any check fails.
            pulseqlib_mex('check', obj.Handle);
        end

        function result = validate(obj, seqObj, varargin)
        % validate  Compare C-backend waveforms against Pulseq MATLAB toolbox.
        %
        %   result = seq.validate(seqObj)
        %   result = seq.validate(seqObj, 'plot', true)
        %
        % See also: pulserver.validate
            result = pulserver.validate(obj.Handle, seqObj, varargin{:});
        end

        function info = report(obj, varargin)
        % report  Structured collection / subsequence summary.
        %
        %   info = seq.report()
        %   str  = seq.report('print', true)
        %
        % See also: pulserver.SequenceCollection.get_block
            info = pulserver.report(obj.Handle, varargin{:});
        end

        function result = pns(obj, varargin)
        % pns  Peripheral nerve stimulation analysis.
        %
        %   result = seq.pns('chronaxie_us', 360, 'rheobase', 20, 'alpha', 0.333)
        %
        % Required name-value parameters
        %   chronaxie_us   double   Nerve time constant (us).
        %   rheobase       double   Threshold slew rate (Hz/m/s).
        %   alpha          double   Effective coil length (m).
        %
        % Output
        %   result  struct with num_samples, slew_x, slew_y, slew_z.
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

            result = pulseqlib_mex('pns', obj.Handle, ...
                o.chronaxie_us, o.rheobase, o.alpha);
        end

        function result = grad_spectrum(obj, varargin)
        % grad_spectrum  Acoustic spectral analysis of gradient waveforms.
        %
        %   result = seq.grad_spectrum()
        %   result = seq.grad_spectrum('max_frequency', 2000)
        %
        % Name-value options
        %   window_size          double  Target window samples. Default: 0
        %                                (auto from window_duration).
        %   window_duration      double  Window duration in s.  Default: 25e-3
        %   spectral_resolution  double  Target freq resolution (Hz). Default: 5
        %   max_frequency        double  Max frequency (Hz). Default: 3000
        %
        % Output
        %   result  struct with spectrogram/spectrum arrays per axis.
            p = inputParser;
            addParameter(p, 'window_size',         0,      @isnumeric);
            addParameter(p, 'window_duration',     25e-3,  @isnumeric);
            addParameter(p, 'spectral_resolution', 5.0,    @isnumeric);
            addParameter(p, 'max_frequency',       3000.0, @isnumeric);
            parse(p, varargin{:});
            o = p.Results;

            win_sz = o.window_size;
            if win_sz == 0
                % Auto: derive from window_duration and grad_raster (10 us)
                win_sz = round(o.window_duration / 10e-6);
            end

            result = pulseqlib_mex('grad_spectrum', obj.Handle, ...
                win_sz, o.spectral_resolution, o.max_frequency);
        end

        function serialize(obj, path)
        % serialize  Save the collection to a binary cache file.
        %
        %   seq.serialize('output.bin')
        %
        % See also: pulserver.SequenceCollection.deserialize
            pulseqlib_mex('save_cache', obj.Handle, path);
        end

        function deserialize(obj, path)
        % deserialize  Restore collection from a binary cache file.
        %
        %   seq.deserialize('output.bin')
        %
        % See also: pulserver.SequenceCollection.serialize
            pulseqlib_mex('load_cache', obj.Handle, path);
        end
    end
end

function delete_if_exists(f)
% Clean up temp file if it still exists.
    if exist(f, 'file')
        delete(f);
    end
end
