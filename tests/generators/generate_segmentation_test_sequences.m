%% generate_segmentation_test_sequences.m (v2 reset)
%
% Iterative rebuild of segmentation test generation.
% Phase 1 scope:
%   - Build one basic GRE 2D case
%   - Export minimal segmentation-focused truth
%   - Keep placeholders for later TR/safety/freqmod truth

clear; clc;
import mr.*

write_gre_2d_base_case(1, 1);
fprintf('\n SPGR segmentation case generated.\n');


function write_gre_2d_base_case(num_slices, num_averages)
    sys = make_system();

    % Basic GRE geometry intentionally small for fast iteration.
    fov = 0.22;
    Nx = 64;
    Ny = 8;
    slice_thickness = 5e-3;
    ndummy = 5;

    alpha = 10 * pi / 180;
    rf_spoil_inc = 84.0; % degrees

    % RF and slice-select
    [rf, gz] = mr.makeSincPulse(alpha, ...
        'Duration', 2.0e-3, ...
        'SliceThickness', slice_thickness, ...
        'timeBwProduct', 4, ...
        'apodization', 0.5, ...
        'use', 'excitation', ...
        'system', sys);
    gz_reph = mr.makeTrapezoid('z', 'Area', -gz.area/2, 'Duration', 1.0e-3, 'system', sys);
    gz_spoil = mr.makeTrapezoid('z', 'Area', 4 / slice_thickness, 'Duration', 1.0e-3, 'system', sys);

    % Readout and ADC
    readout_time = 2.56e-3;
    gx_full = mr.makeTrapezoid('x', 'FlatArea', Nx/fov, 'FlatTime', readout_time, 'system', sys);
    gx_parts = mr.splitGradientAt(gx_full, gx_full.riseTime + gx_full.flatTime);
    gx = gx_parts(1); % truncate at end of flat
    adc = mr.makeAdc(Nx, 'Duration', gx_full.flatTime, 'Delay', gx_full.riseTime, 'system', sys);
    dummy_adc = mr.makeDelay(mr.calcDuration(adc));

    % Pre/rewinder templates
    gx_pre = mr.makeTrapezoid('x', 'Area', -gx_full.area/2, 'Duration', 1.0e-3, 'system', sys);
    gx_spoil_area = 4 / slice_thickness;

    % Bridged spoiler that starts at gx flat amplitude for continuity.
    gx_spoil = mr.makeExtendedTrapezoidArea('x', gx_full.amplitude, 0, gx_spoil_area, sys);

    pe_areas = ((0:Ny-1) - floor(Ny/2)) / fov;
    max_pe_area = max(abs(pe_areas));
    gy_phase = mr.makeTrapezoid('y', 'Area', max_pe_area, 'Duration', 1.0e-3, 'system', sys);

    % ONCE labels for prep/main semantics.
    lbl_once1 = mr.makeLabel('SET', 'ONCE', 1);
    lbl_once0 = mr.makeLabel('SET', 'ONCE', 0);

    seq = mr.Sequence(sys);

    rf_center = mr.calcRfCenter(rf);
    rf_phase = 0.0;
    rf_inc = 0.0;

    % Bookkeeping per TR block: [x_scale, y_scale, z_scale].
    num_blocks_in_tr = 4;
    tr_scale_tmp = zeros(num_blocks_in_tr, 3);
    tr_scale_max = zeros(num_blocks_in_tr, 3);
    tr_scale_min = zeros(num_blocks_in_tr, 3) + inf;
    tr_scale_sign = zeros(num_blocks_in_tr, 3); % track sign of each block's gradient amplitudes for later TR safety analysis.
    
    % Segment bookkeeping: track energy of each segment and which has the max. Energy is
    max_seg_energy_idx = 1;
    seg_energy = 0.0;
    seg_size = 4;
    seg_idx = 1; % initialize to first block of first segment

    % RF bookkeeping
    peakRF = 0.0;

    % Dummy TRs (once region): same 4-block TR shape with PE=0.
    for d = 1:ndummy
        rf_inc = mod(rf_inc + rf_spoil_inc, 360.0);
        rf_phase = mod(rf_phase + rf_inc, 360.0);

        rf_curr = rf;
        rf_curr.phaseOffset = rf_phase / 180 * pi;

        gy_pre = mr.scaleGrad(gy_phase, 0.0);
        gy_rew = mr.scaleGrad(gy_phase, 0.0);

        if d == 1
            seq.addBlock(rf_curr, gz, lbl_once1);
        else
            seq.addBlock(rf_curr, gz);
        end
        seq.addBlock(gx_pre, gy_pre, gz_reph);
        seq.addBlock(gx, dummy_adc);
        seq.addBlock(gx_spoil, gy_rew, gz_spoil);

        % Bookkeeping for TR scale: track max/min gradient amplitude across each block position in the TR, to determine later which blocks are truly "varying" vs. just numerical noise.
        tr_scale_tmp(1, :) = [0, 0, 1];
        tr_scale_tmp(2, :) = [1, 0, 1];
        tr_scale_tmp(3, :) = [1, 0, 0];
        tr_scale_tmp(4, :) = [1, 0, 1];
        tr_scale_max(abs(tr_scale_tmp) > abs(tr_scale_max)) = tr_scale_tmp(abs(tr_scale_tmp) > abs(tr_scale_max));
        tr_scale_min(abs(tr_scale_tmp) < abs(tr_scale_min)) = tr_scale_tmp(abs(tr_scale_tmp) < abs(tr_scale_min));
        sign_mask = (tr_scale_sign == 0) & (tr_scale_tmp ~= 0);
        tr_scale_sign(sign_mask) = sign(tr_scale_tmp(sign_mask));

        % Bookkeeping for segment energy.
        seg_energy_tmp = grad_energy(gx_pre) + grad_energy(gz_reph) + ...
                         grad_energy(gx) + ...
                         grad_energy(gx_spoil) + grad_energy(gz_spoil);
        if seg_energy_tmp > seg_energy
            seg_energy = seg_energy_tmp;
            max_seg_energy_idx = seg_idx;
        end
        seg_idx = seg_idx + seg_size;

        % Bookkeeping for RF: track peak RF amplitude across the sequence
        if max(abs(rf_curr.signal)) > peakRF
            peakRF = max(abs(rf_curr.signal));
        end
    end

    % Main imaging loop: averages -> PE -> slices (slice is inner loop).
    first_main = true;
    for pe = 1:Ny
        if max_pe_area > 0
            yscale = pe_areas(pe) / max_pe_area;
        else
            yscale = 0;
        end
        for sl = 1:num_slices
            rf_inc = mod(rf_inc + rf_spoil_inc, 360.0);
            rf_phase = mod(rf_phase + rf_inc, 360.0);

            rf_curr = rf;
            slc_shift = (sl - 1 - (num_slices - 1) / 2);

            rf_curr.freqOffset = gz.amplitude * slice_thickness * slc_shift;
            rf_curr.phaseOffset = rf_phase / 180 * pi - 2 * pi * rf_curr.freqOffset * rf_center;

            adc_curr = adc;
            adc_curr.freqOffset = rf_curr.freqOffset;
            adc_curr.phaseOffset = rf_phase / 180 * pi;

            gy_pre = mr.scaleGrad(gy_phase, yscale);
            gy_rew = mr.scaleGrad(gy_phase, -yscale);

            if first_main
                seq.addBlock(rf_curr, gz, lbl_once0);
                first_main = false;
            else
                seq.addBlock(rf_curr, gz);
            end
            seq.addBlock(gx_pre, gy_pre, gz_reph);
            seq.addBlock(gx, adc_curr);
            seq.addBlock(gx_spoil, gy_rew, gz_spoil);

            % Bookkeeping for TR scale.
            tr_scale_tmp(1, :) = [0, 0, 1];
            tr_scale_tmp(2, :) = [1, yscale, 1];
            tr_scale_tmp(3, :) = [1, 0, 0];
            tr_scale_tmp(4, :) = [1, -yscale, 1];
            tr_scale_max(abs(tr_scale_tmp) > abs(tr_scale_max)) = tr_scale_tmp(abs(tr_scale_tmp) > abs(tr_scale_max));
            tr_scale_min(abs(tr_scale_tmp) < abs(tr_scale_min)) = tr_scale_tmp(abs(tr_scale_tmp) < abs(tr_scale_min));
            sign_mask = (tr_scale_sign == 0) & (tr_scale_tmp ~= 0);
            tr_scale_sign(sign_mask) = sign(tr_scale_tmp(sign_mask));

            % Bookkeeping for segment energy.
            seg_energy_tmp = grad_energy(gz) + ...
                            grad_energy(gx_pre) + grad_energy(gy_pre) + grad_energy(gz_reph) + ...
                            grad_energy(gx) + ...
                            grad_energy(gx_spoil) + grad_energy(gy_rew) + grad_energy(gz_spoil);
            if seg_energy_tmp > seg_energy
                seg_energy = seg_energy_tmp;
                max_seg_energy_idx = seg_idx;
            end
            seg_idx = seg_idx + seg_size;

            % Bookkeeping for RF: track peak RF amplitude across the sequence
            if max(abs(rf_curr.signal)) > peakRF
                peakRF = max(abs(rf_curr.signal));
            end
        end
    end

    % Default unset sign entries to +1 (for always-zero gradient positions).
    tr_scale_sign(tr_scale_sign == 0) = 1;

    % Compute signed worst-case scale per block position.
    % tr_scale_max stores the signed value with max absolute amplitude;
    % tr_scale_sign stores the sign of the first nonzero occurrence.
    % The C library AMP_MAX_POS mode uses: sign(first) × max(|amp|).
    canonical_scale = tr_scale_sign .* abs(tr_scale_max);

    % Build canonical sequence using worst-case scaled gradients.
    gy_pre  = mr.scaleGrad(gy_phase, canonical_scale(2, 2));
    gy_rew  = mr.scaleGrad(gy_phase, canonical_scale(4, 2));

    canonical_seq = mr.Sequence(sys);
    canonical_seq.addBlock(rf, gz);
    canonical_seq.addBlock(gx_pre, gy_pre, gz_reph);
    canonical_seq.addBlock(gx, adc);
    canonical_seq.addBlock(gx_spoil, gy_rew, gz_spoil);

    % Compute TR duration from block durations.
    TR = 0;
    TR = TR + mr.calcDuration(rf, gz);
    TR = TR + mr.calcDuration(gx_pre, gy_pre, gz_reph);
    TR = TR + mr.calcDuration(gx, adc);
    TR = TR + mr.calcDuration(gx_spoil, gy_rew, gz_spoil);

    % Build canonical (worst-case) TR waveform for safety checks.
    [times_us, waveform_samples] = build_canonical_tr(canonical_seq, sys);

    seq.setDefinition('FOV', [fov fov slice_thickness * num_slices]);
    seq.setDefinition('NumSlices', num_slices);
    seq.setDefinition('TotalDuration', sum(seq.blockDurations));
    seq.setDefinition('PeakRF', peakRF);
    seq.setDefinition('TR', TR);

    [ok, err] = seq.checkTiming;
    if ~ok
        error('Timing check failed:\n%s', strjoin(err, '\n'));
    end

    out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    base = sprintf('gre_2d_%dsl_%davg', num_slices, num_averages);
    seq_path = fullfile(out_dir, [base '.seq']);
    seq.write(seq_path);

    % --- exports: meta ---
    export_meta(fullfile(out_dir, [base '_meta.txt']), adc, TR, num_blocks_in_tr);

    % --- exports: TR waveform (binary float32) ---
    export_tr_waveform(fullfile(out_dir, [base '_tr_waveform.bin']), times_us, waveform_samples);

    % --- exports: segment-level ground truth (phase 3) ---
    num_segments = 1;
    segment_size = [num_blocks_in_tr]; % single segment with all blocks (worst-case energy)
    has_rot = zeros(1, num_blocks_in_tr);
    norot = zeros(1, num_blocks_in_tr);
    adc_id = zeros(1, num_blocks_in_tr);
    adc_id(1, 3) = 1;

    % ------ gaps
    rf_adc_gap_value = -rf.delay + mr.calcDuration(rf, gz) + mr.calcDuration(gx_pre, gy_pre, gz_reph) + adc.delay;

    rf_adc_gap = zeros(1, num_blocks_in_tr);
    rf_adc_gap(1, 1) = rf_adc_gap_value;
    adc_rf_gap = zeros(1, num_blocks_in_tr);
    adc_rf_gap(1, 3) = rf_adc_gap_value;
    adc_adc_gap = zeros(1, num_blocks_in_tr); % no back-to-back ADCs in this sequence

    % Prepare segment definition data structure
    segment_data = struct();
    segment_data.num_segments = num_segments;
    segment_data.segments = {};

    % Write segment-def header and blocks
    for s = 1:num_segments
        block_start = 0.0;
        has_digital_out = 0;
        digital_out_delay = 0.0;
        digital_out_duration = 0.0;

        segment_data.segments{s} = struct();
        segment_data.segments{s}.blocks = {};

        for b = 1:segment_size(s)
            block = seq.getBlock(max_seg_energy_idx + b - 1);
            block_dur = block.blockDuration;

            % get RF
            if isfield(block, 'rf') && ~isempty(block.rf)
                has_rf = true;
                rf_delay = block.rf.delay; % RF delay within block (s)
                rf_samples = block.rf.signal;
                rf_amp = max(abs(rf_samples));
                if rf_amp > 0
                    rf_samples = rf_samples / rf_amp; % normalize to max amplitude 1
                end

                % if it imaginary part is 0, make it real-valued
                if ~any(imag(rf_samples))
                    rf_rho = real(rf_samples);
                    rf_theta = [];                    
                else
                    rf_rho = abs(rf_samples);
                    rf_theta = angle(rf_samples);
                end

                % get time points for rf samples
                rf_time = block.rf.t; % time points for RF samples
                dt = rf_time(2) - rf_time(1);
                if length(unique(diff(rf_time))) == 1 && dt == sys.rfRasterTime
                    rf_time = []; % uniform sampling, can be inferred from start time and dwell
                end
            else
                has_rf = false;
                rf_delay = 0;
                rf_time = [];
                rf_rho = [];
                rf_theta = [];
                rf_amp = 0.0;
            end

            % get gradients
            has_grad = 0;
            if isfield(block, 'gx') && ~isempty(block.gx)
                has_grad = 1;
                gx_delay = block.gx.delay; % Gx delay within block (s)
                if strcmp(block.gx.type, 'trap')
                    gx_amp = block.gx.amplitude;
                    gx_rise = block.gx.riseTime;
                    gx_flat = block.gx.flatTime;
                    gx_fall = block.gx.fallTime;
                    if gx_flat == 0
                        gx_time = cumsum([0, gx_rise, gx_fall]);
                        gx_wave = [0, 1, 0]; % normalized trapezoid waveform
                    else
                        gx_time = cumsum([0, gx_rise, gx_flat, gx_fall]);
                        gx_wave = [0, 1, 1, 0]; % normalized trapezoid waveform
                    end
                else
                    gx_wave = block.gx.waveform;
                    gx_amp = max(abs(gx_wave));
                    if gx_amp > 0
                        gx_wave = gx_wave / gx_amp; % normalize to max amplitude 1
                    end

                     % get time points for gx samples
                    gx_time = block.gx.tt;
                    dt = gx_time(2) - gx_time(1);
                    if length(unique(diff(gx_time))) == 1 && dt == sys.gradRasterTime
                        gx_time = []; % uniform sampling, can be inferred from start time and dwell
                    end
                end
            else
                gx_delay = 0;
                gx_time = [];
                gx_wave = [];
                gx_amp = 0.0;
            end

            if isfield(block, 'gy') && ~isempty(block.gy)
                has_grad = 1;
                gy_delay = block.gy.delay; % Gy delay within block (s)
                if strcmp(block.gy.type, 'trap')
                    gy_amp = block.gy.amplitude;
                    gy_rise = block.gy.riseTime;
                    gy_flat = block.gy.flatTime;
                    gy_fall = block.gy.fallTime;
                    if gy_flat == 0
                        gy_time = cumsum([0, gy_rise, gy_fall]);
                        gy_wave = [0, 1, 0]; % normalized trapezoid waveform
                    else
                        gy_time = cumsum([0, gy_rise, gy_flat, gy_fall]);
                        gy_wave = [0, 1, 1, 0]; % normalized trapezoid waveform
                    end
                else
                    gy_wave = block.gy.waveform;
                    gy_amp = max(abs(gy_wave));
                    if gy_amp > 0
                        gy_wave = gy_wave / gy_amp; % normalize to max amplitude 1
                    end

                     % get time points for gy samples
                    gy_time = block.gy.tt;
                    dt = gy_time(2) - gy_time(1);
                    if length(unique(diff(gy_time))) == 1 && dt == sys.gradRasterTime
                        gy_time = []; % uniform sampling, can be inferred from start time and dwell
                    end
                end
            else
                gy_delay = 0;
                gy_time = [];
                gy_wave = [];
                gy_amp = 0.0;
            end

            if isfield(block, 'gz') && ~isempty(block.gz)
                has_grad = 1;
                gz_delay = block.gz.delay; % Gz delay within block (s)
                if strcmp(block.gz.type, 'trap')
                    gz_amp = block.gz.amplitude;
                    gz_rise = block.gz.riseTime;
                    gz_flat = block.gz.flatTime;
                    gz_fall = block.gz.fallTime;
                    if gz_flat == 0
                        gz_time = cumsum([0, gz_rise, gz_fall]);
                        gz_wave = [0, 1, 0]; % normalized trapezoid waveform
                    else
                        gz_time = cumsum([0, gz_rise, gz_flat, gz_fall]);
                        gz_wave = [0, 1, 1, 0]; % normalized trapezoid waveform
                    end
                else
                    gz_wave = block.gz.waveform;
                    gz_amp = max(abs(gz_wave));
                    if gz_amp > 0
                        gz_wave = gz_wave / gz_amp; % normalize to max amplitude 1
                    end

                     % get time points for gz samples
                    gz_time = block.gz.tt;
                    dt = gz_time(2) - gz_time(1);
                    if length(unique(diff(gz_time))) == 1 && dt == sys.gradRasterTime
                        gz_time = []; % uniform sampling, can be inferred from start time and dwell
                    end
                end
            else
                gz_delay = 0;
                gz_time = [];
                gz_wave = [];
                gz_amp = 0.0;
            end

            % get adc
            if isfield(block, 'adc') && ~isempty(block.adc)
                has_adc = adc_id(s, b);
                adc_delay = block.adc.delay; % ADC delay within block (s)
                adc_id_value = 1; % we have only a single ADC definition in this sequence
            else
                has_adc = 0;
                adc_delay = 0;
                adc_id_value = [];
            end

            % get rotation flag
            rotate = has_rot(s, b) || norot(s, b);

            % get trigger
            if isfield(block, 'trig') && ~isempty(block.trig)
                for t = 1:length(block.trig)
                    if strcmp(block.trig.type, 'output')
                        has_digital_out = 1;
                        digital_out_delay = block.trig(t).delay; % digital output delay within block (s)
                        digital_out_duration = block.trig(t).duration; % digital output duration (s)
                    end
                end
            end

            % get frequency modulation
            has_freq_mod = false;
            num_freq_mod_samples = 0;
            
            % Define RF window: [rf.delay, rf.delay + rf.tt(end)]
            rf_window_start = rf.delay;
            rf_window_end = rf.delay + rf.t(end);
            
            % Define ADC window: [adc.delay, adc.delay + adc.numSamples * adc.dwell]
            adc_window_start = adc.delay;
            adc_window_end = adc.delay + adc.numSamples * adc.dwell;
            
            if has_rf
                % Check if any gradient has nonzero samples in RF window
                gx_nonzero_in_rf = has_grad && grad_nonzero_in_window(block.gx, rf_window_start, rf_window_end);
                gy_nonzero_in_rf = has_grad && grad_nonzero_in_window(block.gy, rf_window_start, rf_window_end);
                gz_nonzero_in_rf = has_grad && grad_nonzero_in_window(block.gz, rf_window_start, rf_window_end);
                
                if gx_nonzero_in_rf || gy_nonzero_in_rf || gz_nonzero_in_rf
                    has_freq_mod = true;
                    num_freq_mod_samples = round(block.blockDuration / sys.rfRasterTime);
                end
            end
            
            if has_adc
                % Check if any gradient has nonzero samples in ADC window
                gx_nonzero_in_adc = has_grad && grad_nonzero_in_window(block.gx, adc_window_start, adc_window_end);
                gy_nonzero_in_adc = has_grad && grad_nonzero_in_window(block.gy, adc_window_start, adc_window_end);
                gz_nonzero_in_adc = has_grad && grad_nonzero_in_window(block.gz, adc_window_start, adc_window_end);
                
                if gx_nonzero_in_adc || gy_nonzero_in_adc || gz_nonzero_in_adc
                    has_freq_mod = true;
                    num_freq_mod_samples = round(block.blockDuration / sys.adcRasterTime);
                end
            end

            % Store block data in segment structure
            block_data = struct();
            block_data.has_rf = has_rf;
            block_data.rf_delay = rf_delay;
            block_data.rf_rho = rf_rho;
            block_data.rf_theta = rf_theta;
            block_data.rf_amp = rf_amp;
            block_data.rf_time = rf_time;
            block_data.gx_delay = gx_delay;
            block_data.gx_wave = gx_wave;
            block_data.gx_amp = gx_amp;
            block_data.gx_time = gx_time;
            block_data.gy_delay = gy_delay;
            block_data.gy_wave = gy_wave;
            block_data.gy_amp = gy_amp;
            block_data.gy_time = gy_time;
            block_data.gz_delay = gz_delay;
            block_data.gz_wave = gz_wave;
            block_data.gz_amp = gz_amp;
            block_data.gz_time = gz_time;
            block_data.has_adc = has_adc;
            block_data.adc_delay = adc_delay;
            block_data.adc_id = adc_id_value;
            block_data.rotate = rotate;
            block_data.has_digital_out = has_digital_out;
            block_data.digital_out_delay = digital_out_delay;
            block_data.digital_out_duration = digital_out_duration;
            block_data.has_freq_mod = has_freq_mod;
            block_data.num_freq_mod_samples = num_freq_mod_samples;
            
            segment_data.segments{s}.blocks{b} = block_data;

            block_start = block_start + block_dur;
        end
    end

    % Export segment definition binary file
    export_segment_def(fullfile(out_dir, [base '_segment_def.bin']), segment_data, sys);

    % --- placeholders for upcoming phases ---
    % TODO(phase4): export scan table truth with ONCE + ignoreRepetitions behavior.
    % TODO(phase5): export frequency-modulation ground truth and k-space crossings.

    fprintf('Wrote %s and minimal truth files.\n', [base '.seq']);
end


function sys = make_system()
    sys = mr.opts( ...
        'MaxGrad',   28,   'GradUnit', 'mT/m', ...
        'MaxSlew',   150,  'SlewUnit', 'T/m/s', ...
        'rfRasterTime',         2e-6, ...
        'gradRasterTime',      20e-6, ...
        'adcRasterTime',        2e-6, ...
        'blockDurationRaster', 20e-6);
end


function export_meta(path, adc, TR, num_blocks_in_tr)
    fid = fopen(path, 'w');
    if fid < 0, error('Failed to open %s', path); end

    % Quantities from example_check.c step 6.
    fprintf(fid, 'num_unique_adcs %d\n', 1);
    fprintf(fid, 'adc_0_samples %d\n', adc.numSamples);
    fprintf(fid, 'adc_0_dwell_ns %d\n', round(adc.dwell * 1e9));
    fprintf(fid, 'max_b1_subseq %d\n', 0);
    fprintf(fid, 'tr_duration_us %d\n', round(TR * 1e6));

    % Segment structure (example_check.c step 5).
    fprintf(fid, 'num_segments %d\n', 1);
    fprintf(fid, 'segment_0_num_blocks %d\n', num_blocks_in_tr);

    % Canonical TR count (1 for single-shot trajectories).
    fprintf(fid, 'num_canonical_trs %d\n', 1);

    fclose(fid);
end


function e = grad_energy(g)
% Gradient energy: integral of amplitude^2 over time [(Hz/m)^2 * s].
    if isfield(g, 'amplitude')
        % Trapezoid: piecewise linear ramp-flat-ramp
        e = (g.amplitude)^2 / 3 * g.riseTime ...
          + (g.amplitude)^2 * g.flatTime ...
          + (g.amplitude)^2 / 3 * g.fallTime;
    elseif isfield(g, 'waveform') && ~isempty(g.waveform)
        % Arbitrary / extended trapezoid
        e = sum((g.waveform(1:end-1)).^2 .* diff(g.tt));
    else
        e = 0;
    end
end


function [times_us, samples] = build_canonical_tr(canonical_seq, sys)
% BUILD_CANONICAL_TR  Construct worst-case TR and resample to uniform raster.
%   Returns times in microseconds, gradient samples in Hz/m (Nx3), and TR in seconds.

    % Extract waveform data and interpolate to uniform half-gradient-raster grid.
    wave_data = canonical_seq.waveforms_and_times(false);
    raster = 0.5 * sys.gradRasterTime;  % 10 us — matches C library
    times = 0.0 : raster : canonical_seq.duration;
    samples = zeros(length(times), 3);
    for c = 1:3
        if c <= length(wave_data) && ~isempty(wave_data{c})
            samples(:, c) = interp1(wave_data{c}(1,:), wave_data{c}(2,:), times, 'linear', 0);
        end
    end

    % Convert times from seconds to microseconds.
    times_us = times(:) * 1e6;
end


function export_tr_waveform(path, times_us, samples)
% EXPORT_TR_WAVEFORM  Write canonical TR waveform as binary float32.
%   Layout: int32 num_samples, then float32 arrays: time_us[N], gx[N], gy[N], gz[N].
    fid = fopen(path, 'w');
    if fid < 0, error('Failed to open %s', path); end

    N = length(times_us);
    fwrite(fid, N, 'int32');
    fwrite(fid, single(times_us), 'float32');
    fwrite(fid, single(samples(:, 1)), 'float32');
    fwrite(fid, single(samples(:, 2)), 'float32');
    fwrite(fid, single(samples(:, 3)), 'float32');

    fclose(fid);
end


function export_segment_def(path, segment_data, sys)
% EXPORT_SEGMENT_DEF  Write segment definition as binary file.
%   Structure: 
%   int32 num_segments
%   For each segment:
%     int32 num_blocks
%     For each block:
%       uint8 flags (bit-packed: has_rf, gx, gy, gz, adc, rotation, digital_out, freq_mod)
%       float32 rf_delay, rf_amp, rf samples
%       float32 gx/gy/gz delays, amplitudes, waveform samples
%       float32 adc_delay, digital_out_delay, digital_out_duration
%       int32 freq_mod_sample_count
    fid = fopen(path, 'wb');
    if fid < 0, error('Failed to open %s', path); end

    % Write header
    fwrite(fid, segment_data.num_segments, 'int32');
    
    % Write per-segment data
    for s = 1:segment_data.num_segments
        blocks = segment_data.segments{s}.blocks;
        fwrite(fid, length(blocks), 'int32');
        
        % Write per-block data
        for b = 1:length(blocks)
            block_data = blocks{b};
            
            % Pack flags as single byte (bit 0-7: rf, gx, gy, gz, adc, rotation, digital_out, freq_mod)
            flags = 0;
            flags = flags + (block_data.has_rf * (2^0));
            flags = flags + (~isempty(block_data.gx_wave) * (2^1));
            flags = flags + (~isempty(block_data.gy_wave) * (2^2));
            flags = flags + (~isempty(block_data.gz_wave) * (2^3));
            flags = flags + (block_data.has_adc * (2^4));
            flags = flags + (block_data.rotate * (2^5));
            flags = flags + (block_data.has_digital_out * (2^6));
            flags = flags + (block_data.has_freq_mod * (2^7));
            fwrite(fid, uint8(flags), 'uint8');
            
            % RF parameters
            fwrite(fid, single(block_data.rf_delay), 'float32');
            fwrite(fid, single(block_data.rf_amp), 'float32');
            if block_data.has_rf && ~isempty(block_data.rf_rho)
                fwrite(fid, int32(length(block_data.rf_rho)), 'int32');
                fwrite(fid, single(block_data.rf_rho), 'float32');
            else
                fwrite(fid, int32(0), 'int32');
            end
            
            % Gradient parameters (x, y, z)
            for axis = 1:3
                switch axis
                    case 1, wave = block_data.gx_wave; delay = block_data.gx_delay; amp = block_data.gx_amp;
                    case 2, wave = block_data.gy_wave; delay = block_data.gy_delay; amp = block_data.gy_amp;
                    case 3, wave = block_data.gz_wave; delay = block_data.gz_delay; amp = block_data.gz_amp;
                end
                fwrite(fid, single(delay), 'float32');
                fwrite(fid, single(amp), 'float32');
                if ~isempty(wave)
                    fwrite(fid, int32(length(wave)), 'int32');
                    fwrite(fid, single(wave), 'float32');
                else
                    fwrite(fid, int32(0), 'int32');
                end
            end
            
            % ADC parameters
            fwrite(fid, single(block_data.adc_delay), 'float32');
            
            % Digital output parameters
            fwrite(fid, single(block_data.digital_out_delay), 'float32');
            fwrite(fid, single(block_data.digital_out_duration), 'float32');
            
            % Frequency modulation parameters
            fwrite(fid, int32(block_data.num_freq_mod_samples), 'int32');
        end
    end
    
    fclose(fid);
end


function has_nonzero = grad_nonzero_in_window(grad, window_start, window_end)
% GRAD_NONZERO_IN_WINDOW  Check if gradient has nonzero samples within [window_start, window_end].
%   For trapezoid: check if flat region overlaps with window.
%   For arbitrary: check if samples within window (after accounting for delay) are nonzero.
    
    has_nonzero = false;
    
    if isempty(grad)
        return;
    end
    
    if strcmp(grad.type, 'trap') || strcmp(grad.type, 'trapezoid')
        % Trapezoid gradient: nonzero region is [delay + riseTime, delay + riseTime + flatTime]
        flat_start = grad.delay + grad.riseTime;
        flat_end = grad.delay + grad.riseTime + grad.flatTime;
        
        % Check if flat region overlaps with [window_start, window_end]
        has_nonzero = (flat_start < window_end) && (flat_end > window_start);
    else
        % Arbitrary waveform: check if samples within window are nonzero
        % Convert window bounds to local time (relative to gradient delay)
        local_window_start = window_start - grad.delay;
        local_window_end = window_end - grad.delay;
        
        % Find indices where gradient time samples fall within the window
        time_samples = grad.tt;
        waveform_samples = grad.waveform;
        
        % Find samples within the window
        in_window = (time_samples >= local_window_start) & (time_samples <= local_window_end);
        
        if any(in_window)
            % Check if any sample in the window is nonzero
            has_nonzero = any(abs(waveform_samples(in_window)) > 0);
        end
    end
end
