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
%       tb.setSegments([4]);               % unique segment sizes
%       tb.setSegmentOrder([1]);           % segment instances per TR
%       tb.setNumAverages(3);
%       tb.setBaseRotation(eye(3));        % optional, default eye(3)
%       tb.export(out_dir, 'gre_2d_1sl_1avg');

    % ----- public properties (read-only after export) -----
    properties (SetAccess = private)
        seq         % mr.Sequence object
        sys         % mr.opts system struct

        % User-supplied hints
        num_blocks_in_tr  = 0
        segment_sizes     = []   % [S1, S2, ...] blocks in each unique segment definition
        segment_reps      = []   % derived from segment_order
        segment_order     = []   % expanded segment instances per TR (1-based segment IDs)
        num_averages      = 1
        base_rot          = eye(3)

        % Derived quantities (computed by prepare())
        peakRF              = 0
        num_canonical_trs   = 0    % number of unique canonical TR groups
        tr_group_labels     = []   % (num_imaging_trs, 1) 0-based group IDs
        canonical_scales    = {}   % cell array of (nbt, 3) per group
        canonical_seqs      = {}   % cell array of mr.Sequence per group
        TR                  = 0
        tr_times_list       = {}   % cell array of time vectors per group
        tr_waveforms        = {}   % cell array of (N_i, 3) per group
        segment_data        = []   % struct for binary export
        unique_adcs         = []   % struct array: (numSamples, dwell)
        fmod_defs           = {}
        fmod_types          = []
        fmod_durations      = []   % blockDuration of each def (for scan-table matching)
        fmod_kinds          = []   % 0=RF, 1=ADC (same as fmod_types)
        fmod_gradsigs       = []   % Nx3 peak |grad amplitude| per axis (for dedup)
        scan_table          = []
        rotmat_table        = []
        freq_mod_table      = []

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

        function setSegments(obj, sizes)
        % SETSEGMENTS  Define unique segment definitions.
        %   sizes: row vector of block counts per unique segment definition
            assert(all(sizes > 0), ...
                'segment sizes must be positive');
            obj.segment_sizes = sizes(:)';
            obj.segment_reps  = ones(size(obj.segment_sizes));
            obj.segment_order = 1:length(obj.segment_sizes);
            obj.prepared = false;
        end

        function setSegmentOrder(obj, order)
        % SETSEGMENTORDER  Override segment instance order within a TR.
        %   order: row vector of 1-based segment IDs, e.g. [1 2 3 3 3 2]
            assert(~isempty(obj.segment_sizes), ...
                'Must call setSegments before setSegmentOrder');

            ord = order(:)';
            num_seg = length(obj.segment_sizes);
            assert(all(ord >= 1 & ord <= num_seg & floor(ord) == ord), ...
                'segment order must contain valid 1-based segment IDs');

            obj.segment_order = ord;

            % Keep reps consistent with effective order.
            reps = zeros(1, num_seg);
            for s = 1:num_seg
                reps(s) = sum(ord == s);
            end
            obj.segment_reps = reps;
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
            assert(~isempty(obj.segment_order), ...
                'Segment order is empty');

            blocks_in_pattern = sum(obj.segment_sizes(obj.segment_order));
            assert(blocks_in_pattern == obj.num_blocks_in_tr, ...
                'Segment topology does not match blocks-per-TR');
        end

        % ---- prepare all derived data ----
        function prepare(obj)
            if obj.prepared, return; end

            obj.discoverUniqueAdcs();
            obj.computePeakRFAndCanonicalScales();
            obj.buildCanonicalTRs();
            obj.buildSegmentData();
            obj.buildFreqModDefs();
            obj.buildScanTableData();

            % Store definitions on the sequence object.
            obj.seq.setDefinition('PeakRF', obj.peakRF);
            obj.seq.setDefinition('TR', obj.TR);
            obj.seq.setDefinition('TotalDuration', sum(obj.seq.blockDurations));

            obj.prepared = true;
        end

        % ---- ADC discovery ----
        function discoverUniqueAdcs(obj)
        % DISCOVERUNIQUEADCS  Scan all blocks, collect unique (numSamples, dwell) pairs.
            adcs = zeros(0, 2);  % each row: [numSamples, dwell]
            for b = 1:length(obj.seq.blockEvents)
                block = obj.seq.getBlock(b);
                if isfield(block, 'adc') && ~isempty(block.adc)
                    key = [block.adc.numSamples, block.adc.dwell];
                    if isempty(adcs) || ~any(adcs(:,1) == key(1) & abs(adcs(:,2) - key(2)) < 1e-15)
                        adcs(end+1, :) = key; %#ok<AGROW>
                    end
                end
            end
            obj.unique_adcs = adcs;
        end

        % ---- TR group discovery ----
        function discoverTRGroups(obj)
        % DISCOVERTGROUPS  Fingerprint imaging TRs by gradient shape;
        %   group TRs with identical shot-index patterns.
        %   Mirrors C library's pulseqlib__find_unique_shot_trs().
            nbt = obj.num_blocks_in_tr;
            num_dummy = obj.findNumDummyBlocks();
            num_imaging_blocks = length(obj.seq.blockEvents) - num_dummy;
            num_trs = num_imaging_blocks / nbt;
            assert(num_trs == floor(num_trs) && num_trs > 0, ...
                'Imaging block count is not a multiple of blocks-per-TR');

            % Shape cache: cell array of fingerprint vectors (one per unique shape).
            shapes = {};

            % Build per-TR fingerprint: one shape ID per (position, axis).
            fp_matrix = zeros(num_trs, nbt * 3);

            for tr_i = 1:num_trs
                base_blk = num_dummy + (tr_i - 1) * nbt;
                for pos = 1:nbt
                    block = obj.seq.getBlock(base_blk + pos);
                    ax_names = {'gx', 'gy', 'gz'};
                    for a = 1:3
                        axn = ax_names{a};
                        if isfield(block, axn) && ~isempty(block.(axn))
                            [sid, shapes] = TruthBuilder.matchOrAddShape(block.(axn), shapes);
                        else
                            sid = 0;
                        end
                        fp_matrix(tr_i, (pos - 1) * 3 + a) = sid;
                    end
                end
            end

            % Deduplicate fingerprints.
            [~, ia, ic] = unique(fp_matrix, 'rows', 'stable');

            obj.num_canonical_trs = length(ia);
            obj.tr_group_labels = ic(:) - 1;  % 0-based to match C convention
        end

        % ---- Phase 1: peak RF + per-group canonical scale + segment energy ----
        function computePeakRFAndCanonicalScales(obj)
            import mr.*

            obj.discoverTRGroups();

            nbt = obj.num_blocks_in_tr;
            num_total_blocks = length(obj.seq.blockEvents);
            num_dummy = obj.findNumDummyBlocks();
            num_imaging_blocks = num_total_blocks - num_dummy;
            num_trs = num_imaging_blocks / nbt;
            M = obj.num_canonical_trs;

            % Global peak RF (across all blocks including dummies).
            peak_rf = 0;
            for blk = 1:num_total_blocks
                block = obj.seq.getBlock(blk);
                if isfield(block, 'rf') && ~isempty(block.rf)
                    pk = max(abs(block.rf.signal));
                    if pk > peak_rf, peak_rf = pk; end
                end
            end
            obj.peakRF = peak_rf;

            % Per-group: canonical scale + max-energy representative.
            obj.canonical_scales = cell(M, 1);
            repr_energy_indices = zeros(M, 1);
            first_group_indices = zeros(M, 1);
            overall_best_energy = 0;
            overall_best_idx = num_dummy + 1;
            shapes = {};

            ax_names = {'gx', 'gy', 'gz'};

            for g = 1:M
                ref_shape_ids = zeros(nbt, 3);
                ref_grads = cell(nbt, 3);
                can_scale = zeros(nbt, 3);
                best_energy = 0;
                best_idx = 0;

                for tr_i = 1:num_trs
                    if obj.tr_group_labels(tr_i) ~= (g - 1)
                        continue;
                    end

                    base_blk = num_dummy + (tr_i - 1) * nbt;
                    tr_energy = 0;

                    if first_group_indices(g) == 0
                        first_group_indices(g) = base_blk + 1;
                    end

                    for pos = 1:nbt
                        block = obj.seq.getBlock(base_blk + pos);

                        for a = 1:3
                            axn = ax_names{a};
                            if isfield(block, axn) && ~isempty(block.(axn))
                                grad = block.(axn);
                                amp = TruthBuilder.gradPeakAmpSigned(grad);
                                [sid, shapes] = TruthBuilder.matchOrAddShape(grad, shapes);
                                tr_energy = tr_energy + TruthBuilder.gradEnergy(grad);

                                if ref_shape_ids(pos, a) == 0
                                    ref_shape_ids(pos, a) = sid;
                                    ref_grads{pos, a} = grad;
                                    can_scale(pos, a) = amp;
                                else
                                    same_state = false;
                                    if strcmp(grad.type, 'trap') || strcmp(grad.type, 'trapezoid')
                                        same_state = (sid == ref_shape_ids(pos, a));
                                    else
                                        same_state = TruthBuilder.gradArbStateMatches(ref_grads{pos, a}, grad);
                                    end
                                    if same_state && abs(amp) > abs(can_scale(pos, a))
                                        can_scale(pos, a) = sign(can_scale(pos, a)) * abs(amp);
                                    end
                                end
                            elseif ref_shape_ids(pos, a) == 0
                                can_scale(pos, a) = 0;
                            end
                        end
                    end

                    if tr_energy > best_energy
                        best_energy = tr_energy;
                        best_idx = base_blk + 1;
                    end
                end

                obj.canonical_scales{g} = can_scale;
                repr_energy_indices(g) = best_idx;

                if best_energy > overall_best_energy
                    overall_best_energy = best_energy;
                    overall_best_idx = best_idx;
                end
            end

            % Store the overall best-energy index for segment def export.
            obj.segment_data = struct();
            obj.segment_data.max_seg_energy_idx = overall_best_idx;
            obj.segment_data.repr_energy_indices = repr_energy_indices;
            obj.segment_data.first_group_indices = first_group_indices;
        end

        % ---- Phase 2: canonical TR waveforms (one per group) ----
        function buildCanonicalTRs(obj)
            import mr.*

            nbt = obj.num_blocks_in_tr;
            M = obj.num_canonical_trs;
            repr = obj.segment_data.first_group_indices;

            obj.canonical_seqs = cell(M, 1);
            obj.tr_waveforms   = cell(M, 1);
            obj.tr_times_list  = cell(M, 1);

            for g = 1:M
                max_idx = repr(g);
                can_scale = obj.canonical_scales{g};

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
                            can_amp = can_scale(pos, a);
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

                    if isempty(args)
                        % Delay-only blocks have no RF/grad/ADC events; emit explicit delay.
                        cseq.addBlock(mr.makeDelay(block.blockDuration));
                    else
                        cseq.addBlock(args{:});
                    end
                    tr_dur = tr_dur + block.blockDuration;
                end

                obj.canonical_seqs{g} = cseq;

                % Resample to half-gradient raster (matches C library).
                wave_data = cseq.waveforms_and_times(false);
                raster = 0.5 * obj.sys.gradRasterTime;
                times = 0.0 : raster : cseq.duration;
                samples = zeros(length(times), 3);
                for c = 1:3
                    if c <= length(wave_data) && ~isempty(wave_data{c})
                        t_raw = wave_data{c}(1, :);
                        w_raw = wave_data{c}(2, :);
                        % interp1 requires monotonic unique sample times.
                        % Keep the last value on duplicated timestamps.
                        [t_u, iu] = unique(t_raw, 'last');
                        w_u = w_raw(iu);
                        samples(:, c) = interp1(t_u, w_u, times, 'linear', 0);
                    end
                end
                obj.tr_times_list{g} = times(:) * 1e6;
                obj.tr_waveforms{g}  = samples;
            end

            % TR duration (same for all groups — same block structure).
            first_idx = obj.segment_data.repr_energy_indices(1);
            obj.TR = 0;
            for pos = 1:nbt
                block = obj.seq.getBlock(first_idx + pos - 1);
                obj.TR = obj.TR + block.blockDuration;
            end
        end

        % ---- Phase 3: segment definition ----
        function buildSegmentData(obj)
            import mr.*

            nbt = obj.num_blocks_in_tr;
            num_total_blocks = length(obj.seq.blockEvents);
            num_dummy = obj.findNumDummyBlocks();
            num_imaging_blocks = num_total_blocks - num_dummy;
            num_trs = num_imaging_blocks / nbt;
            num_seg = length(obj.segment_sizes);
            seg_order = obj.segment_order;

            obj.segment_data.num_segments = num_seg;
            obj.segment_data.segments = cell(1, num_seg);

            % Compute start offsets of each segment instance in one TR.
            num_inst = length(seg_order);
            inst_start = zeros(1, num_inst);
            cum_offset = 0;
            for k = 1:num_inst
                inst_start(k) = cum_offset;
                cum_offset = cum_offset + obj.segment_sizes(seg_order(k));
            end
            assert(cum_offset == nbt, ...
                'Expanded segment order does not span full TR block count');

            for s = 1:num_seg
                inst_ids = find(seg_order == s);
                assert(~isempty(inst_ids), ...
                    'Every segment definition must appear at least once in segment order');

                best_energy = -inf;
                best_start = -1;

                for tr_i = 1:num_trs
                    % MATLAB Sequence block indices are 1-based.
                    tr_base = num_dummy + (tr_i - 1) * nbt + 1;

                    for ii = 1:length(inst_ids)
                        inst = inst_ids(ii);
                        seg_offset = inst_start(inst);
                        e = 0;

                        for b = 1:obj.segment_sizes(s)
                            block = obj.seq.getBlock(tr_base + seg_offset + b - 1);
                            ax_names = {'gx', 'gy', 'gz'};
                            for a = 1:3
                                axn = ax_names{a};
                                if isfield(block, axn) && ~isempty(block.(axn))
                                    e = e + TruthBuilder.gradEnergy(block.(axn));
                                end
                            end
                        end

                        if e > best_energy
                            best_energy = e;
                            best_start = tr_base + seg_offset;
                        end
                    end
                end

                assert(best_start > 0, 'Could not find representative segment instance');

                seg_blocks = cell(1, obj.segment_sizes(s));
                block_start = 0.0;

                for b = 1:obj.segment_sizes(s)
                    block = obj.seq.getBlock(best_start + b - 1);
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
            end
        end

        % ---- Phase 4: frequency modulation definitions ----
        function buildFreqModDefs(obj)
            import mr.*

            num_total_blocks = length(obj.seq.blockEvents);
            num_dummy_blocks = obj.findNumDummyBlocks();

            % Collect ALL unique freq-mod block types.
            % Key: blockDuration (distinguishes shapes with different timing).
            % We keep separate lists for RF-type and ADC-type defs.
            defs = {};
            types = [];
            def_durations = [];   % blockDuration of each def (for matching)
            def_kinds     = [];   % 0=RF, 1=ADC
            def_gradsigs  = [];   % Nx3 peak |grad amplitude| per axis

            for blk = 1:num_total_blocks
                block = obj.seq.getBlock(blk);

                if isfield(block, 'rf') && ~isempty(block.rf)
                    rf_start = block.rf.delay;
                    rf_end   = block.rf.delay + block.rf.t(end);
                    if obj.anyGradNonzeroInWindow(block, rf_start, rf_end)
                        dur = block.blockDuration;
                        gsig = TruthBuilder.blockGradSig(block);
                        already = false;
                        for k = 1:length(def_durations)
                            if def_kinds(k) == 0 && abs(def_durations(k) - dur) < 1e-9 ...
                                    && max(abs(def_gradsigs(k,:) - gsig)) < 1
                                already = true;
                                break;
                            end
                        end
                        if ~already
                            rf_active_start = block.rf.delay;
                            rf_active_end   = block.rf.delay + block.rf.t(end);
                            rf_isodelay     = block.rf.t(end) - mr.calcRfCenter(block.rf);
                            defs{end+1} = TruthBuilder.buildFreqModDefinition( ...
                                block, rf_active_start, rf_active_end, rf_isodelay, ...
                                obj.sys.gradRasterTime, obj.sys.rfRasterTime); %#ok<AGROW>
                            types(end+1) = 0;         %#ok<AGROW>
                            def_durations(end+1) = dur; %#ok<AGROW>
                            def_kinds(end+1) = 0;      %#ok<AGROW>
                            def_gradsigs(end+1,:) = gsig; %#ok<AGROW>
                        end
                    end
                end

                if isfield(block, 'adc') && ~isempty(block.adc) && blk > num_dummy_blocks
                    adc_start = block.adc.delay;
                    adc_end   = block.adc.delay + block.adc.numSamples * block.adc.dwell;
                    if obj.anyGradNonzeroInWindow(block, adc_start, adc_end)
                        dur = block.blockDuration;
                        gsig = TruthBuilder.blockGradSig(block);
                        already = false;
                        for k = 1:length(def_durations)
                            if def_kinds(k) == 1 && abs(def_durations(k) - dur) < 1e-9 ...
                                    && max(abs(def_gradsigs(k,:) - gsig)) < 1
                                already = true;
                                break;
                            end
                        end
                        if ~already
                            adc_dur = block.adc.numSamples * block.adc.dwell;
                            adc_active_start = block.adc.delay;
                            adc_active_end   = adc_active_start + adc_dur;
                            adc_ref_time     = 0.5 * adc_dur;
                            defs{end+1} = TruthBuilder.buildFreqModDefinition( ...
                                block, adc_active_start, adc_active_end, adc_ref_time, ...
                                obj.sys.gradRasterTime, obj.sys.adcRasterTime); %#ok<AGROW>
                            types(end+1) = 1;           %#ok<AGROW>
                            def_durations(end+1) = dur;  %#ok<AGROW>
                            def_kinds(end+1) = 1;        %#ok<AGROW>
                            def_gradsigs(end+1,:) = gsig; %#ok<AGROW>
                        end
                    end
                end
            end

            obj.fmod_defs       = defs;
            obj.fmod_types      = types;
            obj.fmod_durations  = def_durations;
            obj.fmod_kinds      = def_kinds;
            obj.fmod_gradsigs   = def_gradsigs;
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
                            rf_start = block.rf.delay;
                            rf_end   = block.rf.delay + block.rf.t(end);
                            if obj.anyGradNonzeroInWindow(block, rf_start, rf_end)
                                dur = block.blockDuration;
                                gsig = TruthBuilder.blockGradSig(block);
                                for ki = 1:length(obj.fmod_durations)
                                    if obj.fmod_kinds(ki) == 0 && abs(obj.fmod_durations(ki) - dur) < 1e-9 ...
                                            && max(abs(obj.fmod_gradsigs(ki,:) - gsig)) < 1
                                        fmt(act) = ki;
                                        break;
                                    end
                                end
                            end
                        end

                        % Gradients
                        if isfield(block, 'gx') && ~isempty(block.gx)
                            st(act, 4) = TruthBuilder.gradPeakAmpSigned(block.gx);
                        end
                        if isfield(block, 'gy') && ~isempty(block.gy)
                            st(act, 5) = TruthBuilder.gradPeakAmpSigned(block.gy);
                        end
                        if isfield(block, 'gz') && ~isempty(block.gz)
                            st(act, 6) = TruthBuilder.gradPeakAmpSigned(block.gz);
                        end

                        % ADC
                        if isfield(block, 'adc') && ~isempty(block.adc)
                            st(act, 7) = 1;
                            st(act, 8) = block.adc.phaseOffset + ppm_to_hz * block.adc.phasePPM;
                            st(act, 9) = block.adc.freqOffset  + ppm_to_hz * block.adc.freqPPM;
                            adc_start = block.adc.delay;
                            adc_end   = block.adc.delay + block.adc.numSamples * block.adc.dwell;
                            if obj.anyGradNonzeroInWindow(block, adc_start, adc_end)
                                dur = block.blockDuration;
                                gsig = TruthBuilder.blockGradSig(block);
                                for ki = 1:length(obj.fmod_durations)
                                    if obj.fmod_kinds(ki) == 1 && abs(obj.fmod_durations(ki) - dur) < 1e-9 ...
                                            && max(abs(obj.fmod_gradsigs(ki,:) - gsig)) < 1
                                        fmt(act) = ki;
                                        break;
                                    end
                                end
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
                        bd.([axn '_amp']) = TruthBuilder.gradPeakAmpSigned(grad);
                        if bd.([axn '_amp']) ~= 0
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
                % Look up 0-based unique ADC index.
                key = [block.adc.numSamples, block.adc.dwell];
                match = find(obj.unique_adcs(:,1) == key(1) & ...
                             abs(obj.unique_adcs(:,2) - key(2)) < 1e-15, 1);
                bd.adc_id = match - 1;  % 0-based
            else
                bd.has_adc = 0;
                bd.adc_delay = 0;
                bd.adc_id = -1;
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
            fid = fopen(path, 'w');
            if fid < 0, error('Failed to open %s', path); end

            num_adcs = size(obj.unique_adcs, 1);
            fprintf(fid, 'num_unique_adcs %d\n', num_adcs);
            for a = 1:num_adcs
                fprintf(fid, 'adc_%d_samples %d\n', a - 1, obj.unique_adcs(a, 1));
                fprintf(fid, 'adc_%d_dwell_ns %d\n', a - 1, round(obj.unique_adcs(a, 2) * 1e9));
            end
            fprintf(fid, 'max_b1_subseq %d\n', 0);
            fprintf(fid, 'tr_duration_us %d\n', round(obj.TR * 1e6));
            fprintf(fid, 'num_segments %d\n', length(obj.segment_sizes));
            for s = 1:length(obj.segment_sizes)
                fprintf(fid, 'segment_%d_num_blocks %d\n', s - 1, obj.segment_sizes(s));
            end
            fprintf(fid, 'num_canonical_trs %d\n', obj.num_canonical_trs);

            fclose(fid);
        end

        % ---- export: TR waveform binary ----
        function exportTrWaveform(obj, path)
            fid = fopen(path, 'w');
            if fid < 0, error('Failed to open %s', path); end

            M = obj.num_canonical_trs;
            fwrite(fid, int32(M), 'int32');

            for g = 1:M
                times = obj.tr_times_list{g};
                wf    = obj.tr_waveforms{g};
                N = length(times);
                fwrite(fid, int32(N), 'int32');
                fwrite(fid, single(times), 'float32');
                fwrite(fid, single(wf(:, 1)), 'float32');
                fwrite(fid, single(wf(:, 2)), 'float32');
                fwrite(fid, single(wf(:, 3)), 'float32');
            end

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
        % GRADPEAKAMP  Return the peak magnitude of a gradient.
            if strcmp(grad.type, 'trap') || strcmp(grad.type, 'trapezoid')
                amp = grad.amplitude;
            else
                amp = max(abs(grad.waveform));
            end
        end

        function sig = blockGradSig(block)
        % BLOCKGRADSIG  Return [|peak_gx|, |peak_gy|, |peak_gz|] for a block.
            sig = [0, 0, 0];
            axes = {'gx', 'gy', 'gz'};
            for ch = 1:3
                ax = axes{ch};
                if isfield(block, ax) && ~isempty(block.(ax))
                    sig(ch) = abs(TruthBuilder.gradPeakAmp(block.(ax)));
                end
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

        function [sid, shapes] = matchOrAddShape(grad, shapes)
        % MATCHORADDSHAPE  Return shape ID for a gradient, adding to cache if new.
        %   For trapezoids: fingerprint = (riseTime, flatTime, fallTime)
        %   For arbitrary:  fingerprint = normalized waveform (waveform / peak)
            if strcmp(grad.type, 'trap') || strcmp(grad.type, 'trapezoid')
                fp = [grad.riseTime, grad.flatTime, grad.fallTime];
            else
                w = grad.waveform(:)';
                pk = max(abs(w));
                if pk > 0
                    fp = w / pk;
                else
                    fp = w;
                end
            end

            for i = 1:length(shapes)
                if length(fp) == length(shapes{i}) && ...
                        max(abs(fp - shapes{i})) < 1e-8
                    sid = i;
                    return;
                end
            end

            sid = length(shapes) + 1;
            shapes{sid} = fp;
        end

        function tf = gradArbStateMatches(ref_grad, cand_grad)
        % GRADARBSTATEMATCHES  True when arbitrary gradients share the same raw state.
            tol = 1e-9;
            if isempty(ref_grad) || isempty(cand_grad)
                tf = false;
                return;
            end
            if ~(strcmp(ref_grad.type, cand_grad.type) && ...
                    ~(strcmp(ref_grad.type, 'trap') || strcmp(ref_grad.type, 'trapezoid')))
                tf = false;
                return;
            end
            if length(ref_grad.waveform) ~= length(cand_grad.waveform) || ...
                    length(ref_grad.tt) ~= length(cand_grad.tt)
                tf = false;
                return;
            end
            tf = max(abs(ref_grad.waveform(:) - cand_grad.waveform(:))) < 1e-8 && ...
                 max(abs(ref_grad.tt(:) - cand_grad.tt(:))) < tol && ...
                 abs(ref_grad.delay - cand_grad.delay) < tol;
        end
    end
end
