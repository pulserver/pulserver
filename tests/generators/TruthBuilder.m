classdef TruthBuilder < handle
% TRUTHBUILDER  Automate ground-truth export for pulseqlib C-test sequences.
%
%   The user constructs an mr.Sequence object, then hands it to TruthBuilder.
%   The builder derives all bookkeeping (peak RF, canonical TR scale, segment
%   energy, freq-mod definitions, scan table) from the sequence blocks and a
%   small set of user-supplied hints.
%
%   Usage:
%       sys = mr.opts(...);
%       seq = mr.Sequence(sys);
%       % ... build sequence ...
%
%       tb = TruthBuilder(seq, sys);
%       tb.setBlocksPerTR(4);
%       tb.setSegments([4], [1]);          % sizes, reps
%       tb.setNumAverages(3);
%       tb.setBaseRotation(eye(3));        % optional, default eye(3)
%       tb.export(out_dir, 'gre_2d_1sl_1avg');

    % ----- public properties (read-only after export) -----
    properties (SetAccess = private)
        seq         % mr.Sequence object
        sys         % mr.opts system struct

        % User-supplied hints
        num_blocks_in_tr  = 0
        segment_sizes     = []   % [S1, S2, ...] blocks in each segment
        segment_reps      = []   % [R1, R2, ...] repetitions per segment
        num_averages      = 1
        base_rot          = eye(3)

        % Derived quantities (computed by prepare())
        peakRF            = 0
        canonical_scale   = []   % (num_blocks_in_tr, 3) signed worst-case
        canonical_seq     = []   % mr.Sequence for worst-case TR
        TR                = 0
        tr_times_us       = []
        tr_waveform       = []   % (N, 3) Hz/m
        segment_data      = []   % struct for binary export
        fmod_defs         = {}
        fmod_types        = []
        scan_table        = []
        rotmat_table      = []
        freq_mod_table    = []

        prepared          = false
    end

    methods
        % ---- constructor ----
        function obj = TruthBuilder(seq, sys)
            obj.seq = seq;
            obj.sys = sys;
        end

        % ---- setters ----
        function setBlocksPerTR(obj, n)
            obj.num_blocks_in_tr = n;
            obj.prepared = false;
        end

        function setSegments(obj, sizes, reps)
        % SETSEGMENTS  Define segment topology.
        %   sizes: row vector of block counts per segment
        %   reps:  row vector of repetition counts per segment
            assert(length(sizes) == length(reps), ...
                'sizes and reps must have the same length');
            obj.segment_sizes = sizes(:)';
            obj.segment_reps  = reps(:)';
            obj.prepared = false;
        end

        function setNumAverages(obj, n)
            obj.num_averages = n;
            obj.prepared = false;
        end

        function setBaseRotation(obj, R)
            obj.base_rot = R;
            obj.prepared = false;
        end

        % ---- main entry point ----
        function export(obj, out_dir, base_name)
        % EXPORT  Compute all derived quantities and write binary files.
            obj.validate();
            obj.prepare();

            if ~exist(out_dir, 'dir')
                mkdir(out_dir);
            end

            % Write .seq file
            obj.seq.write(fullfile(out_dir, [base_name '.seq']));

            % Meta text
            obj.exportMeta(fullfile(out_dir, [base_name '_meta.txt']));

            % TR waveform binary
            obj.exportTrWaveform(fullfile(out_dir, [base_name '_tr_waveform.bin']));

            % Segment definition binary
            obj.exportSegmentDef(fullfile(out_dir, [base_name '_segment_def.bin']));

            % Freq-mod definition binary
            obj.exportFreqModDefs(fullfile(out_dir, [base_name '_freqmod_def.bin']));

            % Scan table binary
            obj.exportScanTable(fullfile(out_dir, [base_name '_scan_table.bin']));
        end
    end

    methods (Access = private)
        % ---- validation ----
        function validate(obj)
            assert(obj.num_blocks_in_tr > 0, ...
                'Must call setBlocksPerTR before export');
            assert(~isempty(obj.segment_sizes), ...
                'Must call setSegments before export');
        end

        % ---- prepare all derived data ----
        function prepare(obj)
            if obj.prepared, return; end

            obj.computePeakRFAndCanonicalScale();
            obj.buildCanonicalTR();
            obj.buildSegmentData();
            obj.buildFreqModDefs();
            obj.buildScanTableData();

            % Store definitions on the sequence object.
            obj.seq.setDefinition('PeakRF', obj.peakRF);
            obj.seq.setDefinition('TR', obj.TR);
            obj.seq.setDefinition('TotalDuration', sum(obj.seq.blockDurations));

            obj.prepared = true;
        end

        % ---- Phase 1: peak RF + canonical scale + segment energy ----
        function computePeakRFAndCanonicalScale(obj)
            import mr.*

            nbt = obj.num_blocks_in_tr;
            num_total_blocks = length(obj.seq.blockEvents);

            % Track worst-case gradient amplitude per block position in TR.
            % tr_amp_max stores the signed value with the largest absolute amplitude.
            % tr_amp_sign stores the sign of the first nonzero occurrence.
            tr_amp_max  = zeros(nbt, 3);
            tr_amp_sign = zeros(nbt, 3);

            % Segment energy tracking: one energy value per TR, keep the best.
            best_energy = 0;
            best_energy_idx = 1;

            peak_rf = 0;

            for blk = 1:num_total_blocks
                block = obj.seq.getBlock(blk);

                % Position within TR (1-based)
                pos = mod(blk - 1, nbt) + 1;

                % Gradient amplitudes for this block position.
                amp_tmp = [0, 0, 0];
                ax_names = {'gx', 'gy', 'gz'};
                for a = 1:3
                    axn = ax_names{a};
                    if isfield(block, axn) && ~isempty(block.(axn))
                        amp_tmp(a) = TruthBuilder.gradPeakAmpSigned(block.(axn));
                    end
                end

                % Update max (by absolute value): keep the signed value with largest |amp|
                bigger = abs(amp_tmp) > abs(tr_amp_max(pos, :));
                tr_amp_max(pos, bigger) = amp_tmp(bigger);
                % Update sign (first nonzero)
                first_nz = (tr_amp_sign(pos, :) == 0) & (amp_tmp ~= 0);
                tr_amp_sign(pos, first_nz) = sign(amp_tmp(first_nz));

                % Peak RF
                if isfield(block, 'rf') && ~isempty(block.rf)
                    pk = max(abs(block.rf.signal));
                    if pk > peak_rf, peak_rf = pk; end
                end

                % Segment energy: accumulate per-TR and keep the maximum.
                if pos == 1
                    seg_start = blk;
                end
                if pos == nbt
                    energy = 0;
                    for sb = seg_start:blk
                        b2 = obj.seq.getBlock(sb);
                        for a2 = 1:3
                            axn2 = ax_names{a2};
                            if isfield(b2, axn2) && ~isempty(b2.(axn2))
                                energy = energy + TruthBuilder.gradEnergy(b2.(axn2));
                            end
                        end
                    end
                    if energy > best_energy
                        best_energy = energy;
                        best_energy_idx = seg_start;
                    end
                end
            end

            % Default unset sign entries to +1.
            tr_amp_sign(tr_amp_sign == 0) = 1;

            % Canonical amplitude: sign(first) * max(|amplitude|).
            obj.canonical_scale = tr_amp_sign .* abs(tr_amp_max);
            obj.peakRF = peak_rf;

            % Store the best-energy segment start index for segment def export.
            obj.segment_data = struct();
            obj.segment_data.max_seg_energy_idx = best_energy_idx;
        end

        % ---- Phase 2: canonical TR waveform ----
        function buildCanonicalTR(obj)
            import mr.*

            nbt = obj.num_blocks_in_tr;
            max_idx = obj.segment_data.max_seg_energy_idx;

            % Build canonical sequence with worst-case amplitudes.
            % For each block position, scale each gradient so its amplitude
            % matches the canonical amplitude: canonical_amp / ref_amp.
            cseq = mr.Sequence(obj.sys);
            tr_dur = 0;

            for pos = 1:nbt
                block = obj.seq.getBlock(max_idx + pos - 1);
                args = {};

                % RF: same shape in every TR, use as-is
                if isfield(block, 'rf') && ~isempty(block.rf)
                    args{end+1} = block.rf; %#ok<AGROW>
                end

                % Gradients: scale to canonical amplitude
                ax_names = {'gx', 'gy', 'gz'};
                for a = 1:3
                    axn = ax_names{a};
                    if isfield(block, axn) && ~isempty(block.(axn))
                        ref_amp = TruthBuilder.gradPeakAmpSigned(block.(axn));
                        can_amp = obj.canonical_scale(pos, a);
                        if ref_amp ~= 0
                            scale = can_amp / ref_amp;
                        else
                            scale = 0;
                        end
                        args{end+1} = mr.scaleGrad(block.(axn), scale); %#ok<AGROW>
                    end
                end

                % ADC: include if present
                if isfield(block, 'adc') && ~isempty(block.adc)
                    args{end+1} = block.adc; %#ok<AGROW>
                end

                cseq.addBlock(args{:});
                tr_dur = tr_dur + block.blockDuration;
            end

            obj.canonical_seq = cseq;
            obj.TR = tr_dur;

            % Resample to half-gradient raster (matches C library).
            wave_data = cseq.waveforms_and_times(false);
            raster = 0.5 * obj.sys.gradRasterTime;
            times = 0.0 : raster : cseq.duration;
            samples = zeros(length(times), 3);
            for c = 1:3
                if c <= length(wave_data) && ~isempty(wave_data{c})
                    samples(:, c) = interp1(wave_data{c}(1,:), wave_data{c}(2,:), times, 'linear', 0);
                end
            end
            obj.tr_times_us = times(:) * 1e6;
            obj.tr_waveform = samples;
        end

        % ---- Phase 3: segment definition ----
        function buildSegmentData(obj)
            import mr.*

            nbt = obj.num_blocks_in_tr;
            max_idx = obj.segment_data.max_seg_energy_idx;
            num_seg = length(obj.segment_sizes);

            obj.segment_data.num_segments = num_seg;
            obj.segment_data.segments = cell(1, num_seg);

            % Build a flat list of block indices for the max-energy segment group.
            % For multiple segments: segment 1 occupies blocks 1..S1, segment 2 occupies S1+1..S1+S2, etc.
            cum_offset = 0;
            for s = 1:num_seg
                seg_blocks = cell(1, obj.segment_sizes(s));
                block_start = 0.0;

                for b = 1:obj.segment_sizes(s)
                    block = obj.seq.getBlock(max_idx + cum_offset + b - 1);
                    seg_blocks{b} = obj.extractBlockData(block, block_start);
                    block_start = block_start + block.blockDuration;
                end

                % Compute segment-level gaps.
                [rf_adc_gap, adc_adc_gap] = obj.computeSegmentGaps(seg_blocks);

                seg = struct();
                seg.blocks = seg_blocks;
                seg.rf_adc_gap_us = rf_adc_gap;
                seg.adc_adc_gap_us = adc_adc_gap;
                obj.segment_data.segments{s} = seg;

                cum_offset = cum_offset + obj.segment_sizes(s);
            end
        end

        % ---- Phase 4: frequency modulation definitions ----
        function buildFreqModDefs(obj)
            import mr.*

            nbt = obj.num_blocks_in_tr;
            num_total_blocks = length(obj.seq.blockEvents);

            % Scan all unique block "shapes" to discover RF and ADC freq-mod defs.
            % RF: first block with RF + nonzero gradient in RF window.
            % ADC: first imaging (non-dummy) block with ADC + nonzero gradient in ADC window.
            rf_def_block  = [];
            adc_def_block = [];

            % Find number of dummy blocks: walk blocks until ONCE=0 label is seen.
            num_dummy_blocks = obj.findNumDummyBlocks();

            for blk = 1:num_total_blocks
                block = obj.seq.getBlock(blk);

                if isempty(rf_def_block) && isfield(block, 'rf') && ~isempty(block.rf)
                    % Check if any gradient overlaps the RF window.
                    rf_start = block.rf.delay;
                    rf_end   = block.rf.delay + block.rf.t(end);
                    if obj.anyGradNonzeroInWindow(block, rf_start, rf_end)
                        rf_def_block = block;
                    end
                end

                if isempty(adc_def_block) && blk > num_dummy_blocks && ...
                        isfield(block, 'adc') && ~isempty(block.adc)
                    adc_start = block.adc.delay;
                    adc_end   = block.adc.delay + block.adc.numSamples * block.adc.dwell;
                    if obj.anyGradNonzeroInWindow(block, adc_start, adc_end)
                        adc_def_block = block;
                    end
                end

                if ~isempty(rf_def_block) && ~isempty(adc_def_block)
                    break;
                end
            end

            defs = {};
            types = [];
            if ~isempty(rf_def_block)
                rf_active_start = rf_def_block.rf.delay;
                rf_active_end   = rf_def_block.rf.delay + rf_def_block.rf.t(end);
                rf_isodelay     = rf_def_block.rf.t(end) - mr.calcRfCenter(rf_def_block.rf);
                defs{end+1} = TruthBuilder.buildFreqModDefinition( ...
                    rf_def_block, rf_active_start, rf_active_end, rf_isodelay, ...
                    obj.sys.gradRasterTime, obj.sys.rfRasterTime);
                types(end+1) = 0;  %#ok<AGROW>
            end
            if ~isempty(adc_def_block)
                adc_dur = adc_def_block.adc.numSamples * adc_def_block.adc.dwell;
                adc_active_start = adc_def_block.adc.delay;
                adc_active_end   = adc_def_block.adc.delay + adc_dur;
                adc_ref_time     = 0.5 * adc_dur;
                defs{end+1} = TruthBuilder.buildFreqModDefinition( ...
                    adc_def_block, adc_active_start, adc_active_end, adc_ref_time, ...
                    obj.sys.gradRasterTime, obj.sys.adcRasterTime);
                types(end+1) = 1;  %#ok<AGROW>
            end

            obj.fmod_defs  = defs;
            obj.fmod_types = types;
        end

        % ---- Phase 5: scan table ----
        function buildScanTableData(obj)
            import mr.*

            num_blocks_per_pass = length(obj.seq.blockEvents);
            num_cols = 11;
            max_entries = obj.num_averages * num_blocks_per_pass;

            st  = zeros(max_entries, num_cols);
            rot = zeros(max_entries, 9);
            fmt = zeros(max_entries, 1);
            act = 1;

            ppm_to_hz = 1e-6 * obj.sys.gamma * obj.sys.B0;
            num_dummy_blocks = obj.findNumDummyBlocks();

            once  = 0;
            norot_active = 0;

            for avg = 1:obj.num_averages
                for b = 1:num_blocks_per_pass
                    block = obj.seq.getBlock(b);

                    % Read labels from sequence block.
                    if isfield(block, 'label') && ~isempty(block.label)
                        for lbl = 1:length(block.label)
                            lab = block.label(lbl);
                            if strcmp(lab.label, 'ONCE')
                                if strcmp(lab.type, 'labelset')
                                    once = lab.value;
                                elseif strcmp(lab.type, 'labelinc')
                                    once = once + lab.value;
                                end
                            elseif strcmp(lab.label, 'NOROT')
                                if strcmp(lab.type, 'labelset')
                                    norot_active = lab.value;
                                elseif strcmp(lab.type, 'labelinc')
                                    norot_active = norot_active + lab.value;
                                end
                            end
                        end
                    end

                    % ONCE filter: once==1 → first avg only; once==2 → last avg only.
                    if once == 0 || (once == 1 && avg == 1) || (once == 2 && avg == obj.num_averages)
                        % RF
                        if isfield(block, 'rf') && ~isempty(block.rf)
                            st(act, 1) = max(abs(block.rf.signal));
                            st(act, 2) = block.rf.phaseOffset + ppm_to_hz * block.rf.phasePPM;
                            st(act, 3) = block.rf.freqOffset  + ppm_to_hz * block.rf.freqPPM;
                            rf_idx = find(obj.fmod_types == 0, 1);
                            if ~isempty(rf_idx)
                                fmt(act) = rf_idx;
                            end
                        end

                        % Gradients
                        if isfield(block, 'gx') && ~isempty(block.gx)
                            st(act, 4) = TruthBuilder.gradPeakAmp(block.gx);
                        end
                        if isfield(block, 'gy') && ~isempty(block.gy)
                            st(act, 5) = TruthBuilder.gradPeakAmp(block.gy);
                        end
                        if isfield(block, 'gz') && ~isempty(block.gz)
                            st(act, 6) = TruthBuilder.gradPeakAmp(block.gz);
                        end

                        % ADC
                        if isfield(block, 'adc') && ~isempty(block.adc)
                            st(act, 7) = 1;
                            st(act, 8) = block.adc.phaseOffset + ppm_to_hz * block.adc.phasePPM;
                            st(act, 9) = block.adc.freqOffset  + ppm_to_hz * block.adc.freqPPM;
                            adc_idx = find(obj.fmod_types == 1, 1);
                            if ~isempty(adc_idx)
                                fmt(act) = adc_idx;
                            end
                        end

                        % Triggers
                        if isfield(block, 'trig') && ~isempty(block.trig)
                            for t = 1:length(block.trig)
                                if strcmp(block.trig(t).type, 'output')
                                    st(act, 10) = 1;
                                end
                                if strcmp(block.trig(t).type, 'trigger')
                                    st(act, 11) = 1;
                                end
                            end
                        end

                        % Rotation
                        if isfield(block, 'rotation')
                            rotmat = mr.aux.quat.toRotMat(block.rotation.rotQuaternion);
                        else
                            rotmat = eye(3);
                        end
                        if norot_active == 1
                            act_rotmat = rotmat;
                        else
                            act_rotmat = obj.base_rot * rotmat;
                        end
                        rot(act, :) = reshape(act_rotmat', 1, 9);

                        act = act + 1;
                    end
                end
            end

            n = act - 1;
            obj.scan_table    = st(1:n, :);
            obj.rotmat_table  = rot(1:n, :);
            obj.freq_mod_table = fmt(1:n);
        end

        % ---- helper: find number of dummy blocks via ONCE label ----
        function n = findNumDummyBlocks(obj)
            % Walk blocks until we find SET ONCE 0, which marks end of dummy region.
            n = 0;
            once_state = 0;
            for b = 1:length(obj.seq.blockEvents)
                block = obj.seq.getBlock(b);
                if isfield(block, 'label') && ~isempty(block.label)
                    for lbl = 1:length(block.label)
                        lab = block.label(lbl);
                        if strcmp(lab.label, 'ONCE') && strcmp(lab.type, 'labelset')
                            if lab.value == 1 && once_state == 0
                                once_state = 1;
                            elseif lab.value == 0 && once_state == 1
                                n = b - 1;
                                return;
                            end
                        end
                    end
                end
            end
            % If no ONCE=0 found, assume no dummies.
            n = 0;
        end

        % ---- helper: extract block data for segment def ----
        function bd = extractBlockData(obj, block, block_start)
            import mr.*

            bd = struct();

            % RF
            if isfield(block, 'rf') && ~isempty(block.rf)
                bd.has_rf = true;
                bd.rf_delay = block.rf.delay;
                rf_samples = block.rf.signal;
                bd.rf_amp = max(abs(rf_samples));
                if bd.rf_amp > 0
                    rf_samples = rf_samples / bd.rf_amp;
                end
                if ~any(imag(rf_samples))
                    bd.rf_rho = real(rf_samples);
                    bd.rf_theta = [];
                else
                    bd.rf_rho = abs(rf_samples);
                    bd.rf_theta = angle(rf_samples);
                end
                rf_time = block.rf.t;
                dt = rf_time(2) - rf_time(1);
                if length(unique(diff(rf_time))) == 1 && dt == obj.sys.rfRasterTime
                    bd.rf_time = [];
                else
                    bd.rf_time = rf_time;
                end
            else
                bd.has_rf = false;
                bd.rf_delay = 0;
                bd.rf_rho = [];
                bd.rf_theta = [];
                bd.rf_amp = 0;
                bd.rf_time = [];
            end

            % Gradients
            ax_names = {'gx', 'gy', 'gz'};
            for a = 1:3
                axn = ax_names{a};
                if isfield(block, axn) && ~isempty(block.(axn))
                    grad = block.(axn);
                    bd.([axn '_delay']) = grad.delay;
                    if strcmp(grad.type, 'trap')
                        bd.([axn '_amp']) = grad.amplitude;
                        if grad.flatTime == 0
                            bd.([axn '_wave']) = [0, 1, 0];
                            bd.([axn '_time']) = cumsum([0, grad.riseTime, grad.fallTime]);
                        else
                            bd.([axn '_wave']) = [0, 1, 1, 0];
                            bd.([axn '_time']) = cumsum([0, grad.riseTime, grad.flatTime, grad.fallTime]);
                        end
                    else
                        w = grad.waveform;
                        bd.([axn '_amp']) = max(abs(w));
                        if bd.([axn '_amp']) > 0
                            bd.([axn '_wave']) = w / bd.([axn '_amp']);
                        else
                            bd.([axn '_wave']) = w;
                        end
                        t = grad.tt;
                        dt = t(2) - t(1);
                        if length(unique(diff(t))) == 1 && dt == obj.sys.gradRasterTime
                            bd.([axn '_time']) = [];
                        else
                            bd.([axn '_time']) = t;
                        end
                    end
                else
                    bd.([axn '_delay']) = 0;
                    bd.([axn '_wave']) = [];
                    bd.([axn '_amp']) = 0;
                    bd.([axn '_time']) = [];
                end
            end

            % ADC
            has_grad = ~isempty(bd.gx_wave) || ~isempty(bd.gy_wave) || ~isempty(bd.gz_wave);
            if isfield(block, 'adc') && ~isempty(block.adc)
                bd.has_adc = 1;
                bd.adc_delay = block.adc.delay;
                bd.adc_id = 1;
            else
                bd.has_adc = 0;
                bd.adc_delay = 0;
                bd.adc_id = [];
            end

            % Rotation: derive from labels or block rotation presence.
            bd.rotate = isfield(block, 'rotation') && ~isempty(block.rotation);

            % Digital output
            bd.has_digital_out = 0;
            bd.digital_out_delay = 0;
            bd.digital_out_duration = 0;
            if isfield(block, 'trig') && ~isempty(block.trig)
                for t = 1:length(block.trig)
                    if strcmp(block.trig(t).type, 'output')
                        bd.has_digital_out = 1;
                        bd.digital_out_delay = block.trig(t).delay;
                        bd.digital_out_duration = block.trig(t).duration;
                    end
                end
            end

            % Frequency modulation
            bd.has_freq_mod = false;
            bd.num_freq_mod_samples = 0;
            if bd.has_rf && has_grad
                rf_ws = block.rf.delay;
                rf_we = block.rf.delay + block.rf.t(end);
                if obj.anyGradNonzeroInWindow(block, rf_ws, rf_we)
                    bd.has_freq_mod = true;
                    bd.num_freq_mod_samples = round(block.blockDuration / obj.sys.rfRasterTime);
                end
            end
            if bd.has_adc && has_grad
                adc_ws = block.adc.delay;
                adc_we = block.adc.delay + block.adc.numSamples * block.adc.dwell;
                if obj.anyGradNonzeroInWindow(block, adc_ws, adc_we)
                    bd.has_freq_mod = true;
                    bd.num_freq_mod_samples = round(block.blockDuration / obj.sys.adcRasterTime);
                end
            end

            % Anchor times (relative to segment start, in us)
            if bd.has_rf
                rf_iso = mr.calcRfCenter(block.rf);
                bd.rf_isocenter_us = (block_start + block.rf.delay + rf_iso) * 1e6;
                bd.rf_start_us = (block_start + block.rf.delay) * 1e6;
                bd.rf_end_us   = (block_start + block.rf.delay + block.rf.t(end)) * 1e6;
            else
                bd.rf_isocenter_us = -1;
                bd.rf_start_us = -1;
                bd.rf_end_us   = -1;
            end
            if bd.has_adc
                adc_dur_s = block.adc.numSamples * block.adc.dwell;
                bd.adc_kzero_us = (block_start + block.adc.delay + 0.5 * adc_dur_s) * 1e6;
                bd.adc_start_us = (block_start + block.adc.delay) * 1e6;
                bd.adc_end_us   = (block_start + block.adc.delay + adc_dur_s) * 1e6;
            else
                bd.adc_kzero_us = -1;
                bd.adc_start_us = -1;
                bd.adc_end_us   = -1;
            end
        end

        % ---- helper: compute segment-level RF->ADC and ADC->ADC gaps ----
        function [rf_adc_gap, adc_adc_gap] = computeSegmentGaps(~, seg_blocks)
            rf_ends = [];
            adc_starts = [];
            adc_ends = [];
            for b = 1:length(seg_blocks)
                bd = seg_blocks{b};
                if bd.rf_end_us >= 0
                    rf_ends = [rf_ends, bd.rf_end_us]; %#ok<AGROW>
                end
                if bd.adc_start_us >= 0
                    adc_starts = [adc_starts, bd.adc_start_us]; %#ok<AGROW>
                    adc_ends   = [adc_ends,   bd.adc_end_us];   %#ok<AGROW>
                end
            end

            rf_adc_gap = -1;
            for r = 1:length(rf_ends)
                candidates = adc_starts(adc_starts >= rf_ends(r));
                if ~isempty(candidates)
                    gap = min(candidates) - rf_ends(r);
                    if rf_adc_gap < 0 || gap < rf_adc_gap
                        rf_adc_gap = gap;
                    end
                end
            end

            adc_adc_gap = -1;
            if length(adc_starts) >= 2
                sorted_starts = sort(adc_starts);
                sorted_ends   = sort(adc_ends);
                for a = 2:length(sorted_starts)
                    gap = sorted_starts(a) - sorted_ends(a-1);
                    if adc_adc_gap < 0 || gap < adc_adc_gap
                        adc_adc_gap = gap;
                    end
                end
            end
        end

        % ---- helper: check if any gradient in block is nonzero in window ----
        function result = anyGradNonzeroInWindow(~, block, wstart, wend)
            result = false;
            ax_names = {'gx', 'gy', 'gz'};
            for a = 1:3
                axn = ax_names{a};
                if isfield(block, axn) && ~isempty(block.(axn))
                    if TruthBuilder.gradNonzeroInWindow(block.(axn), wstart, wend)
                        result = true;
                        return;
                    end
                end
            end
        end

        % ---- export: meta text ----
        function exportMeta(obj, path)
            % Find first ADC block to get samples/dwell.
            adc_info = [];
            for b = 1:length(obj.seq.blockEvents)
                block = obj.seq.getBlock(b);
                if isfield(block, 'adc') && ~isempty(block.adc)
                    adc_info = block.adc;
                    break;
                end
            end

            fid = fopen(path, 'w');
            if fid < 0, error('Failed to open %s', path); end

            fprintf(fid, 'num_unique_adcs %d\n', 1);
            if ~isempty(adc_info)
                fprintf(fid, 'adc_0_samples %d\n', adc_info.numSamples);
                fprintf(fid, 'adc_0_dwell_ns %d\n', round(adc_info.dwell * 1e9));
            end
            fprintf(fid, 'max_b1_subseq %d\n', 0);
            fprintf(fid, 'tr_duration_us %d\n', round(obj.TR * 1e6));
            fprintf(fid, 'num_segments %d\n', length(obj.segment_sizes));
            for s = 1:length(obj.segment_sizes)
                fprintf(fid, 'segment_%d_num_blocks %d\n', s - 1, obj.segment_sizes(s));
            end
            fprintf(fid, 'num_canonical_trs %d\n', 1);

            fclose(fid);
        end

        % ---- export: TR waveform binary ----
        function exportTrWaveform(obj, path)
            fid = fopen(path, 'w');
            if fid < 0, error('Failed to open %s', path); end

            N = length(obj.tr_times_us);
            fwrite(fid, N, 'int32');
            fwrite(fid, single(obj.tr_times_us), 'float32');
            fwrite(fid, single(obj.tr_waveform(:, 1)), 'float32');
            fwrite(fid, single(obj.tr_waveform(:, 2)), 'float32');
            fwrite(fid, single(obj.tr_waveform(:, 3)), 'float32');

            fclose(fid);
        end

        % ---- export: segment definition binary ----
        function exportSegmentDef(obj, path)
            fid = fopen(path, 'wb');
            if fid < 0, error('Failed to open %s', path); end

            fwrite(fid, obj.segment_data.num_segments, 'int32');

            for s = 1:obj.segment_data.num_segments
                seg = obj.segment_data.segments{s};
                blocks = seg.blocks;
                fwrite(fid, length(blocks), 'int32');

                for b = 1:length(blocks)
                    bd = blocks{b};

                    flags = 0;
                    flags = flags + (bd.has_rf * (2^0));
                    flags = flags + (~isempty(bd.gx_wave) * (2^1));
                    flags = flags + (~isempty(bd.gy_wave) * (2^2));
                    flags = flags + (~isempty(bd.gz_wave) * (2^3));
                    flags = flags + (bd.has_adc * (2^4));
                    flags = flags + (bd.rotate * (2^5));
                    flags = flags + (bd.has_digital_out * (2^6));
                    flags = flags + (bd.has_freq_mod * (2^7));
                    fwrite(fid, uint8(flags), 'uint8');

                    % RF
                    fwrite(fid, single(bd.rf_delay), 'float32');
                    fwrite(fid, single(bd.rf_amp), 'float32');
                    if bd.has_rf && ~isempty(bd.rf_rho)
                        fwrite(fid, int32(length(bd.rf_rho)), 'int32');
                        fwrite(fid, single(bd.rf_rho), 'float32');
                    else
                        fwrite(fid, int32(0), 'int32');
                    end

                    % Gradients
                    ax_names = {'gx', 'gy', 'gz'};
                    for a = 1:3
                        axn = ax_names{a};
                        wave = bd.([axn '_wave']);
                        delay = bd.([axn '_delay']);
                        amp = bd.([axn '_amp']);
                        fwrite(fid, single(delay), 'float32');
                        fwrite(fid, single(amp), 'float32');
                        if ~isempty(wave)
                            fwrite(fid, int32(length(wave)), 'int32');
                            fwrite(fid, single(wave), 'float32');
                        else
                            fwrite(fid, int32(0), 'int32');
                        end
                    end

                    % ADC
                    fwrite(fid, single(bd.adc_delay), 'float32');

                    % Digital output
                    fwrite(fid, single(bd.digital_out_delay), 'float32');
                    fwrite(fid, single(bd.digital_out_duration), 'float32');

                    % Freq-mod
                    fwrite(fid, int32(bd.num_freq_mod_samples), 'int32');

                    % Anchors
                    fwrite(fid, single(bd.rf_isocenter_us), 'float32');
                    fwrite(fid, single(bd.adc_kzero_us), 'float32');
                end

                % Segment-level gaps
                fwrite(fid, single(seg.rf_adc_gap_us), 'float32');
                fwrite(fid, single(seg.adc_adc_gap_us), 'float32');
            end

            fclose(fid);
        end

        % ---- export: freq-mod definitions binary ----
        function exportFreqModDefs(obj, path)
            fid = fopen(path, 'wb');
            if fid < 0, error('Failed to open %s', path); end

            nd = length(obj.fmod_defs);
            fwrite(fid, int32(nd), 'int32');

            for d = 1:nd
                fm = obj.fmod_defs{d};
                fwrite(fid, int32(obj.fmod_types(d)), 'int32');
                fwrite(fid, int32(fm.num_samples), 'int32');
                fwrite(fid, single(fm.raster_us), 'float32');
                fwrite(fid, single(fm.duration_us), 'float32');
                fwrite(fid, single(fm.ref_time_us), 'float32');
                fwrite(fid, single(fm.ref_integral), 'float32');
                fwrite(fid, single(fm.waveform(:, 1)), 'float32');
                fwrite(fid, single(fm.waveform(:, 2)), 'float32');
                fwrite(fid, single(fm.waveform(:, 3)), 'float32');
            end

            fclose(fid);
        end

        % ---- export: scan table binary ----
        function exportScanTable(obj, path)
            fid = fopen(path, 'wb');
            if fid < 0, error('Failed to open %s', path); end

            n = size(obj.scan_table, 1);
            fwrite(fid, int32(n), 'int32');

            for i = 1:n
                fwrite(fid, single(obj.scan_table(i, 1)), 'float32');
                fwrite(fid, single(obj.scan_table(i, 2)), 'float32');
                fwrite(fid, single(obj.scan_table(i, 3)), 'float32');
                fwrite(fid, single(obj.scan_table(i, 4)), 'float32');
                fwrite(fid, single(obj.scan_table(i, 5)), 'float32');
                fwrite(fid, single(obj.scan_table(i, 6)), 'float32');
                fwrite(fid, int32(obj.scan_table(i, 7)),  'int32');
                fwrite(fid, single(obj.scan_table(i, 8)), 'float32');
                fwrite(fid, single(obj.scan_table(i, 9)), 'float32');
                fwrite(fid, int32(obj.scan_table(i, 10)), 'int32');
                fwrite(fid, int32(obj.scan_table(i, 11)), 'int32');
                fwrite(fid, single(obj.rotmat_table(i, :)), 'float32');
                fwrite(fid, int32(obj.freq_mod_table(i)), 'int32');
            end

            fclose(fid);
        end
    end

    methods (Static)
        function e = gradEnergy(g)
        % GRADENERGY  Gradient energy: integral of amplitude^2 over time.
            if isfield(g, 'amplitude')
                e = (g.amplitude)^2 / 3 * g.riseTime ...
                  + (g.amplitude)^2 * g.flatTime ...
                  + (g.amplitude)^2 / 3 * g.fallTime;
            elseif isfield(g, 'waveform') && ~isempty(g.waveform)
                e = sum((g.waveform(1:end-1)).^2 .* diff(g.tt));
            else
                e = 0;
            end
        end

        function s = gradPeakAmpSigned(grad)
        % GRADPEAKAMPSIGNED  Return the signed amplitude with max absolute value.
        %   For trapezoid: amplitude (already signed).
        %   For arbitrary: the sample with max |value|.
            if strcmp(grad.type, 'trap') || strcmp(grad.type, 'trapezoid')
                s = grad.amplitude;
            else
                [~, idx] = max(abs(grad.waveform));
                s = grad.waveform(idx);
            end
        end

        function amp = gradPeakAmp(grad)
        % GRADPEAKAMP  Return the peak amplitude of a gradient (signed).
            if strcmp(grad.type, 'trap') || strcmp(grad.type, 'trapezoid')
                amp = grad.amplitude;
            else
                amp = max(abs(grad.waveform));
            end
        end

        function result = gradNonzeroInWindow(grad, wstart, wend)
        % GRADNONZEROINWINDOW  Check if gradient has nonzero samples in [wstart, wend].
            result = false;
            if isempty(grad), return; end

            if strcmp(grad.type, 'trap') || strcmp(grad.type, 'trapezoid')
                flat_start = grad.delay + grad.riseTime;
                flat_end   = grad.delay + grad.riseTime + grad.flatTime;
                result = (flat_start < wend) && (flat_end > wstart);
            else
                local_start = wstart - grad.delay;
                local_end   = wend   - grad.delay;
                in_window = (grad.tt >= local_start) & (grad.tt <= local_end);
                if any(in_window)
                    result = any(abs(grad.waveform(in_window)) > 0);
                end
            end
        end

        function [t, w] = gradToKnots(grad)
        % GRADTOKNOTS  Convert Pulseq gradient to time/amplitude knots.
            if strcmp(grad.type, 'trap') || strcmp(grad.type, 'trapezoid')
                d = grad.delay;
                r = grad.riseTime;
                f = grad.flatTime;
                l = grad.fallTime;
                a = grad.amplitude;
                if f > 0
                    t = [d; d+r; d+r+f; d+r+f+l];
                    w = [0; a; a; 0];
                else
                    t = [d; d+r; d+r+l];
                    w = [0; a; 0];
                end
            else
                t = grad.delay + grad.tt(:);
                w = grad.waveform(:);
            end
        end

        function fmod = buildFreqModDefinition(block, active_start_s, active_end_s, ref_time_s, grad_raster_s, target_raster_s)
        % BUILDFREQMODDEFINITION  Build freq-mod base definition (static).
            active_dur_s   = active_end_s - active_start_s;
            grad_raster_us = grad_raster_s * 1e6;
            active_dur_us  = active_dur_s * 1e6;
            ref_time_us    = ref_time_s * 1e6;

            num_samples = floor(active_dur_us / grad_raster_us) + 1;
            if num_samples < 2, num_samples = 2; end

            uniform_t = (0:num_samples-1)' * grad_raster_s;
            axes = {'gx', 'gy', 'gz'};
            waveform = zeros(num_samples, 3);
            ref_integral = zeros(1, 3);

            for ch = 1:3
                ax = axes{ch};
                if isfield(block, ax) && ~isempty(block.(ax))
                    [raw_t, raw_w] = TruthBuilder.gradToKnots(block.(ax));
                    raw_t = raw_t - active_start_s;
                    if raw_t(1) > 0
                        raw_t = [0; raw_t(:)]; %#ok<AGROW>
                        raw_w = [0; raw_w(:)]; %#ok<AGROW>
                    end
                    if raw_t(end) < active_dur_s
                        raw_t = [raw_t(:); active_dur_s]; %#ok<AGROW>
                        raw_w = [raw_w(:); 0];            %#ok<AGROW>
                    end
                    waveform(:, ch) = interp1(raw_t(:), raw_w(:), uniform_t, 'linear', 0);
                end

                ref_sample = floor(ref_time_us / grad_raster_us);
                ref_sample = max(0, min(ref_sample, num_samples - 1));
                if ref_sample > 0
                    ref_integral(ch) = 2 * pi * 1e-6 * ...
                        trapz(waveform(1:ref_sample+1, ch)) * grad_raster_us;
                else
                    ref_integral(ch) = 0;
                end
            end

            fmod.num_samples   = num_samples;
            fmod.raster_us     = grad_raster_us;
            fmod.duration_us   = active_dur_us;
            fmod.ref_time_us   = ref_time_us;
            fmod.ref_integral  = ref_integral;
            fmod.waveform      = waveform;

            if target_raster_s > 0 && target_raster_s < grad_raster_s - 1e-9
                target_raster_us = target_raster_s * 1e6;
                fine_num = floor(active_dur_us / target_raster_us) + 1;
                if fine_num < 2, fine_num = 2; end

                fine_waveform = zeros(fine_num, 3);
                for j = 1:fine_num
                    orig_idx = min(floor((j-1) * target_raster_us / grad_raster_us) + 1, num_samples);
                    fine_waveform(j, :) = waveform(orig_idx, :);
                end

                for ch = 1:3
                    ref_sample = floor(ref_time_us / target_raster_us);
                    ref_sample = max(0, min(ref_sample, fine_num - 1));
                    if ref_sample > 0
                        ref_integral(ch) = 2 * pi * 1e-6 * ...
                            trapz(fine_waveform(1:ref_sample+1, ch)) * target_raster_us;
                    else
                        ref_integral(ch) = 0;
                    end
                end

                fmod.num_samples  = fine_num;
                fmod.raster_us    = target_raster_us;
                fmod.ref_integral = ref_integral;
                fmod.waveform     = fine_waveform;
            end
        end
    end
end
