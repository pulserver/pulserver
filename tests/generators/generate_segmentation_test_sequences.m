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
        tr_scale_max(tr_scale_tmp > tr_scale_max) = tr_scale_tmp(tr_scale_tmp > tr_scale_max);
        tr_scale_min(tr_scale_tmp < tr_scale_min) = tr_scale_tmp(tr_scale_tmp < tr_scale_min);
        
        % Bookkeeping for segment energy: track which segment has the highest total gradient energy, as a proxy for which will be most important to get right in segmentation. This is a heuristic to help guide the design of segmentation test cases and their expected outputs, and is not meant to be a perfect measure of "segment importance" in general.
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
            tr_scale_max(tr_scale_tmp > tr_scale_max) = tr_scale_tmp(tr_scale_tmp > tr_scale_max);
            tr_scale_min(tr_scale_tmp < tr_scale_min) = tr_scale_tmp(tr_scale_tmp < tr_scale_min);

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

    % Compute tr duration
    TR = 0;
    TR = TR + mr.calcDuration(rf, gz);
    TR = TR + mr.calcDuration(gx_pre, gy_pre, gz_reph);
    TR = TR + mr.calcDuration(gx, adc);
    TR = TR + mr.calcDuration(gx_spoil, gy_rew, gz_spoil);

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
    export_meta(fullfile(out_dir, [base '_meta.txt']), adc, TR);

    % --- placeholders for upcoming phases ---
    % TODO(phase2): export scan table truth with ONCE + ignoreRepetitions behavior.
    % TODO(phase3): export waveform truth for max-energy segment instance.
    % TODO(phase4): export TR safety waveforms (max_pos_amp, zero_var).
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


function export_meta(path, adc, TR)
    fid = fopen(path, 'w');
    if fid < 0, error('Failed to open %s', path); end

    % Only the quantities needed by example_check.c step 6.
    fprintf(fid, 'num_unique_adcs %d\n', 1);
    fprintf(fid, 'adc_0_samples %d\n', adc.numSamples);
    fprintf(fid, 'adc_0_dwell_us %d\n', round(adc.dwell * 1e6));
    fprintf(fid, 'max_b1_subseq %d\n', 0);
    fprintf(fid, 'tr_duration_us %d\n', round(TR * 1e6));

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
