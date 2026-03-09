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

    % --- exports: block-level ground truth (phase 3) ---
    export_block_meta(fullfile(out_dir, [base '_block_meta.txt']), ...
                      seq, sys, rf, adc, gz, gx_pre, gz_reph, gx, gx_spoil, gy_phase, gz_spoil, ...
                      max_seg_energy_idx, num_blocks_in_tr, canonical_scale);
    export_rf_magnitude(fullfile(out_dir, [base '_rf_mag.bin']), rf);
    export_arb_grad(fullfile(out_dir, [base '_arb_grad_b2_x.bin']), gx);
    export_arb_grad(fullfile(out_dir, [base '_arb_grad_b3_x.bin']), gx_spoil);

    % --- placeholders for upcoming phases ---
    % TODO(phase4): export scan table truth with ONCE + ignoreRepetitions behavior.
    % TODO(phase5): export frequency-modulation ground truth and k-space crossings.

    fprintf('Wrote %s and minimal truth files.\n', [base '.seq']);
end


function export_tr_scales(path, scale_max, scale_min)
    fid = fopen(path, 'w');
    if fid < 0, error('Failed to open %s', path); end

    fprintf(fid, 'tr_block_idx,x_max,y_max,z_max,x_min,y_min,z_min\n');
    for b = 1:size(scale_max, 1)
        fprintf(fid, '%d,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n', b - 1, ...
            scale_max(b, 1), scale_max(b, 2), scale_max(b, 3), ...
            scale_min(b, 1), scale_min(b, 2), scale_min(b, 3));
    end

    fclose(fid);
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


function export_blocks_csv(path, seq)
    fid = fopen(path, 'w');
    if fid < 0, error('Failed to open %s', path); end

    fprintf(fid, 'idx,duration_us,rf_amp_hz,rf_freq_hz,rf_phase_rad,gx_amp,gy_amp,gz_amp,adc_flag,adc_freq_hz,adc_phase_rad\n');

    for n = 1:length(seq.blockDurations)
        blk = seq.getBlock(n);
        dur_us = round(seq.blockDurations(n) * 1e6);

        [rf_amp, rf_freq, rf_phase] = extract_rf(blk);
        gx_amp = extract_grad_amp(blk, 'gx');
        gy_amp = extract_grad_amp(blk, 'gy');
        gz_amp = extract_grad_amp(blk, 'gz');
        [adc_flag, adc_freq, adc_phase] = extract_adc(blk);

        fprintf(fid, '%d,%d,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%d,%.8g,%.8g\n', ...
            n - 1, dur_us, rf_amp, rf_freq, rf_phase, gx_amp, gy_amp, gz_amp, adc_flag, adc_freq, adc_phase);
    end

    fclose(fid);
end


function export_segments_stub(path, num_blocks_in_tr)
    % Phase-1 stub: one segment line. Will be replaced with exact unique
    % block-definition IDs when segment truth tests are added.
    fid = fopen(path, 'w');
    if fid < 0, error('Failed to open %s', path); end
    fprintf(fid, '0');
    for b = 2:num_blocks_in_tr
        fprintf(fid, ' %d', b - 1);
    end
    fprintf(fid, '\n');
    fclose(fid);
end


function n = count_adcs(seq)
    n = 0;
    for ii = 1:length(seq.blockDurations)
        b = seq.getBlock(ii);
        if isfield(b, 'adc') && ~isempty(b.adc)
            n = n + 1;
        end
    end
end


function [rf_amp, rf_freq, rf_phase] = extract_rf(blk)
    rf_amp = 0; rf_freq = 0; rf_phase = 0;
    if isfield(blk, 'rf') && ~isempty(blk.rf)
        rf_amp = blk.rf.signal(1);
        if isfield(blk.rf, 'freqOffset'),  rf_freq  = blk.rf.freqOffset;  end
        if isfield(blk.rf, 'phaseOffset'), rf_phase = blk.rf.phaseOffset; end
    end
end


function amp = extract_grad_amp(blk, axis)
    amp = 0;
    if isfield(blk, axis)
        g = blk.(axis);
        if ~isempty(g)
            if isfield(g, 'amplitude')
                amp = g.amplitude;
            elseif isfield(g, 'waveform') && ~isempty(g.waveform)
                amp = max(abs(g.waveform));
            end
        end
    end
end


function [adc_flag, adc_freq, adc_phase] = extract_adc(blk)
    adc_flag = 0; adc_freq = 0; adc_phase = 0;
    if isfield(blk, 'adc') && ~isempty(blk.adc)
        adc_flag = 1;
        if isfield(blk.adc, 'freqOffset'),  adc_freq  = blk.adc.freqOffset;  end
        if isfield(blk.adc, 'phaseOffset'), adc_phase = blk.adc.phaseOffset; end
    end
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


function export_block_meta(path, seq, sys, rf, adc, gz, gx_pre, gz_reph, gx, gx_spoil, gy_phase, gz_spoil, ...
                           max_seg_energy_idx, num_blocks, canonical_scale)
% EXPORT_BLOCK_META  Write per-block ground truth for geninstruction tests.
%   Extracts timing, gradient, RF, and ADC metadata from the max-energy
%   segment instance into a key-value text file.
    fid = fopen(path, 'w');
    if fid < 0, error('Failed to open %s', path); end

    % Block durations and cumulative start times.
    start_us = 0;
    for b = 1:num_blocks
        blk = seq.getBlock(max_seg_energy_idx + b - 1);
        dur_us = round(seq.blockDurations(max_seg_energy_idx + b - 1) * 1e6);
        fprintf(fid, 'block_%d_duration_us %d\n', b - 1, dur_us);
        fprintf(fid, 'block_%d_start_time_us %d\n', b - 1, start_us);
        start_us = start_us + dur_us;
    end

    % RF (block 0): always the rf+gz block.
    fprintf(fid, 'block_0_rf_delay_us %d\n', round(rf.delay * 1e6));
    fprintf(fid, 'block_0_rf_num_samples %d\n', length(rf.signal));
    % Windowed sinc with negative sidelobes: phase shape encodes sign flips.
    rf_is_complex = any(rf.signal < 0);
    fprintf(fid, 'block_0_rf_is_complex %d\n', rf_is_complex);
    fprintf(fid, 'block_0_rf_num_channels %d\n', 1);

    % Trap gradients — write corner parameters for each trapezoid.
    % Block 0: Gz (slice-select trap)
    write_trap_meta(fid, 0, 'z', gz);

    % Block 1: Gx_pre, Gy_pre (scaled), Gz_reph — all traps.
    write_trap_meta(fid, 1, 'x', gx_pre);
    % Gy is scaled by canonical_scale(2,2) — amplitude is template × scale.
    gy_pre_scaled = mr.scaleGrad(gy_phase, canonical_scale(2, 2));
    write_trap_meta(fid, 1, 'y', gy_pre_scaled);
    write_trap_meta(fid, 1, 'z', gz_reph);

    % Block 2: Gx is arbitrary (split trap), ADC.
    fprintf(fid, 'block_2_gx_is_arb %d\n', 1);
    fprintf(fid, 'block_2_gx_num_samples %d\n', length(gx.waveform));
    fprintf(fid, 'block_2_gx_delay_us %d\n', round(gx.delay * 1e6));
    fprintf(fid, 'block_2_adc_delay_us %d\n', round(adc.delay * 1e6));

    % Block 3: Gx_spoil (arb extended trap), Gy_rew (scaled trap), Gz_spoil (trap).
    fprintf(fid, 'block_3_gx_is_arb %d\n', 1);
    fprintf(fid, 'block_3_gx_num_samples %d\n', length(gx_spoil.waveform));
    fprintf(fid, 'block_3_gx_delay_us %d\n', round(gx_spoil.delay * 1e6));
    gy_rew_scaled = mr.scaleGrad(gy_phase, canonical_scale(4, 2));
    write_trap_meta(fid, 3, 'y', gy_rew_scaled);
    write_trap_meta(fid, 3, 'z', gz_spoil);

    % RF-ADC gap: ADC_start − RF_end (within segment, in us).
    block_durs_s = seq.blockDurations(max_seg_energy_idx : max_seg_energy_idx + num_blocks - 1);
    rf_end_us = round((rf.delay + length(rf.signal) * sys.rfRasterTime) * 1e6);
    adc_start_us = round((block_durs_s(1) + block_durs_s(2) + adc.delay) * 1e6);
    fprintf(fid, 'rf_adc_gap_us %d\n', adc_start_us - rf_end_us);

    fclose(fid);
end


function write_trap_meta(fid, block_idx, axis, g)
% WRITE_TRAP_META  Write trapezoid corner parameters for one gradient.
    prefix = sprintf('block_%d_g%s', block_idx, axis);
    fprintf(fid, '%s_amplitude_hz_m %.8g\n', prefix, g.amplitude);
    fprintf(fid, '%s_rise_us %d\n',  prefix, round(g.riseTime * 1e6));
    fprintf(fid, '%s_flat_us %d\n',  prefix, round(g.flatTime * 1e6));
    fprintf(fid, '%s_fall_us %d\n',  prefix, round(g.fallTime * 1e6));
    fprintf(fid, '%s_delay_us %d\n', prefix, round(g.delay * 1e6));
end


function export_rf_magnitude(path, rf)
% EXPORT_RF_MAGNITUDE  Write normalized RF magnitude as binary float32.
%   Layout: int32 num_samples, then float32[N] normalized magnitude.
    fid = fopen(path, 'w');
    if fid < 0, error('Failed to open %s', path); end

    mag = abs(rf.signal(:));
    mag_norm = mag / max(mag);

    fwrite(fid, length(mag_norm), 'int32');
    fwrite(fid, single(mag_norm), 'float32');

    fclose(fid);
end


function export_arb_grad(path, g)
% EXPORT_ARB_GRAD  Write arbitrary gradient waveform as binary float32.
%   Layout: int32 num_samples, then float32[N] amplitude (Hz/m),
%   then float32[N] time (us).
    fid = fopen(path, 'w');
    if fid < 0, error('Failed to open %s', path); end

    amp = g.waveform(:);
    t_us = g.tt(:) * 1e6;
    N = length(amp);

    fwrite(fid, N, 'int32');
    fwrite(fid, single(amp), 'float32');
    fwrite(fid, single(t_us), 'float32');

    fclose(fid);
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
