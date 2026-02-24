%% generate_segmentation_test_sequences.m
%
% Generates Pulseq .seq files to verify:
%   - TR detection and segmentation
%   - Multi-shot pattern detection
%   - Min / max gradient amplitudes
%   - Block cursor (per-block ground truth)
%
% Design rules
%   1. All events created OUTSIDE the acquisition loop; only scaleGrad /
%      scalar property changes (rf.phaseOffset, etc.) inside.
%   2. Dummy / preparation scans marked ONCE=1; exit blocks ONCE=2.
%   3. Raster times chosen for GE + Siemens compatibility:
%        rf = 2 us, grad = 20 us, adc = 2 us, block = 20 us
%   4. Ground truth exported per-sequence as CSV + metadata text files.
%   5. Each generator accepts num_slices and num_averages as input params.

clear; clc;
import mr.*

%% --- run all generators -------------------------------------------------

write_bssfp(1);
write_bssfp(3);

write_spgr(1, 1);
write_spgr(1, 3);
write_spgr(3, 1);
write_spgr(3, 3);

write_fse(1, 1);
write_fse(1, 3);
write_fse(3, 1);
write_fse(3, 3);

write_epi(1, 1);
write_epi(1, 3);
write_epi(3, 1);
write_epi(3, 3);

write_mprage(1, 1);
write_mprage(1, 3);
write_mprage(3, 1);
write_mprage(3, 3);

write_mprage_noncart(1, 1, 240, false);
write_mprage_noncart(1, 3, 240, false);
write_mprage_noncart(3, 1, 240, false);
write_mprage_noncart(3, 3, 240, false);

write_mprage_noncart(1, 1, 240, true);
write_mprage_noncart(1, 3, 240, true);
write_mprage_noncart(3, 1, 240, true);
write_mprage_noncart(3, 3, 240, true);

write_mprage_noncart(1, 1, 2048, true);

fprintf('\n=== All segmentation test sequences generated. ===\n');


%% ========================================================================
%  Shared utilities
%  ========================================================================

function sys = make_system()
% System limits compatible with both GE and Siemens scanners.
    sys = mr.opts( ...
        'MaxGrad',   28,   'GradUnit', 'mT/m', ...
        'MaxSlew',   150,  'SlewUnit', 'T/m/s', ...
        'rfRingdownTime',      20e-6, ...
        'rfDeadTime',         100e-6, ...
        'adcDeadTime',         10e-6, ...
        'rfRasterTime',         2e-6, ...
        'gradRasterTime',      20e-6, ...
        'adcRasterTime',        2e-6, ...
        'blockDurationRaster', 20e-6);
end

function fname = seq_filename(prefix, num_slices, num_averages, suffix)
% Build output filename: <prefix>_<Nsl>sl_<Navg>avg<suffix>.seq
    if nargin < 4, suffix = ''; end
    fname = sprintf('%s_%dsl_%davg%s.seq', prefix, num_slices, num_averages, suffix);
end

function check_and_write(seq, fname, fov, thick, num_slices, num_averages, gt)
% Timing check, definitions, write, ground truth.
%
% gt (optional struct) contains structural ground truth:
%   .tr_min          - mr.Sequence: representative TR with zero PE
%                      (matches C library amplitude mode 2 = definition-min)
%   .tr_max          - mr.Sequence: representative TR with max |PE|
%                      (matches C library amplitude mode 1 = position-max)
%   .rf_center_s     - RF isocenter offset within the shape (seconds)
%   .adc_num_samples - number of ADC samples per readout
%   .adc_dwell_s     - ADC dwell time (seconds)
%   .rf_refocus_center_s - RF refocusing isocenter offset (seconds)
%   .seg_unique_ids  - cell array of int arrays, each cell is
%                      unique block def IDs for one segment
%   .unique_blocks   - int array of unique block def IDs for the
%                      full TR (all segments concatenated)
%   .num_prep_blocks - number of preparation blocks (ONCE=1 region)
%   .num_cool_blocks - number of cooldown blocks (ONCE=2 region)
%   .degenerate_prep - 1 if prep pattern == main pattern, 0 otherwise
%   .degenerate_cool - 1 if cooldown pattern == main pattern, 0 otherwise

    if nargin < 7, gt = struct(); end

    [ok, err] = seq.checkTiming;
    if ok
        fprintf('  [%s] Timing OK\n', fname);
    else
        fprintf('  [%s] Timing FAILED:\n', fname);
        fprintf('  %s\n', err{:});
    end

    % Embed metadata in definitions
    total_dur_s = sum(seq.blockDurations);
    seq.setDefinition('TotalDuration', total_dur_s);
    seq.setDefinition('FOV', [fov fov thick * num_slices]);
    seq.setDefinition('NumSlices', num_slices);

    seq.write(fname);

    % Export ground truth
    export_ground_truth(seq, fname, num_averages, gt);
end

function export_ground_truth(seq, seq_fname, num_averages, gt)
% Write ground truth files for C library validation.
%
% Outputs (all <base>_* files):
%   _blocks.csv   - per-block event summary
%   _meta.txt     - key/value metadata (C-parseable)
%   _segments.txt - segment definitions as unique block IDs
%   _tr_gx.csv, _tr_gy.csv, _tr_gz.csv - TR gradient waveforms

    if nargin < 4, gt = struct(); end

    N = length(seq.blockDurations);
    [~, base, ~] = fileparts(seq_fname);

    % --- per-block data ---
    fid = fopen([base '_blocks.csv'], 'w');
    fprintf(fid, 'idx,duration_us,rf_amp_hz,rf_freq_hz,rf_phase_rad,gx_amp,gy_amp,gz_amp,adc_flag,adc_freq_hz,adc_phase_rad\n');

    num_adcs = 0;
    for n = 1:N
        blk = seq.getBlock(n);
        dur_us = round(seq.blockDurations(n) * 1e6);

        [rf_amp, rf_freq, rf_phase] = extract_rf(blk);
        gx_amp = extract_grad_amp(blk, 'gx');
        gy_amp = extract_grad_amp(blk, 'gy');
        gz_amp = extract_grad_amp(blk, 'gz');
        [adc_flag, adc_freq, adc_phase] = extract_adc(blk);
        num_adcs = num_adcs + adc_flag;

        fprintf(fid, '%d,%d,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%d,%.8g,%.8g\n', ...
            n - 1, dur_us, rf_amp, rf_freq, rf_phase, ...
            gx_amp, gy_amp, gz_amp, adc_flag, adc_freq, adc_phase);
    end
    fclose(fid);

    % --- metadata ---
    fid = fopen([base '_meta.txt'], 'w');
    fprintf(fid, 'num_blocks %d\n', N);
    fprintf(fid, 'num_averages %d\n', num_averages);
    fprintf(fid, 'total_duration_us %d\n', round(sum(seq.blockDurations) * 1e6));
    fprintf(fid, 'num_adcs %d\n', num_adcs);

    if isfield(gt, 'num_prep_blocks')
        fprintf(fid, 'num_prep_blocks %d\n', gt.num_prep_blocks);
    end
    if isfield(gt, 'num_cool_blocks')
        fprintf(fid, 'num_cool_blocks %d\n', gt.num_cool_blocks);
    end
    if isfield(gt, 'degenerate_prep')
        fprintf(fid, 'degenerate_prep %d\n', gt.degenerate_prep);
    end
    if isfield(gt, 'degenerate_cool')
        fprintf(fid, 'degenerate_cool %d\n', gt.degenerate_cool);
    end
    if isfield(gt, 'unique_blocks')
        tr_size = length(gt.unique_blocks);
        fprintf(fid, 'tr_size %d\n', tr_size);
    end
    if isfield(gt, 'seg_unique_ids')
        fprintf(fid, 'num_segments %d\n', length(gt.seg_unique_ids));
    end
    fclose(fid);

    % --- segment definitions (unique block IDs per segment) ---
    if isfield(gt, 'seg_unique_ids')
        fid = fopen([base '_segments.txt'], 'w');
        for s = 1:length(gt.seg_unique_ids)
            ids = gt.seg_unique_ids{s};
            fprintf(fid, '%d', ids(1));
            for k = 2:length(ids)
                fprintf(fid, ' %d', ids(k));
            end
            fprintf(fid, '\n');
        end
        fclose(fid);
    end

    % --- expected scan table ---
    if isfield(gt, 'unique_blocks') && isfield(gt, 'num_prep_blocks')
        export_scan_table(base, N, num_averages, gt);
    end

    % --- TR gradient waveforms (min amplitude = definition-min, mode 2) ---
    if isfield(gt, 'tr_min') && ~isempty(gt.tr_min)
        export_tr_waveforms(gt.tr_min, base, '_min', gt);
    end

    % --- TR gradient waveforms (max positional amplitude, mode 1) ---
    if isfield(gt, 'tr_max') && ~isempty(gt.tr_max)
        export_tr_waveforms(gt.tr_max, base, '_max', gt);
    end

    % --- prep TR waveforms (actual amplitude, mode 0) ---
    if isfield(gt, 'tr_prep') && ~isempty(gt.tr_prep)
        export_tr_waveforms(gt.tr_prep, base, '_prep', gt);
    end

    % --- cooldown TR waveforms (actual amplitude, mode 0) ---
    if isfield(gt, 'tr_cool') && ~isempty(gt.tr_cool)
        export_tr_waveforms(gt.tr_cool, base, '_cool', gt);
    end
end

function export_tr_waveforms(tr_seq, base, mode_suffix, gt)
% Export per-axis gradient waveforms from a representative TR sequence.
% Uses waveforms_and_times to get native-timing waveform points.
%
% Args:
%   tr_seq      - mr.Sequence with one TR worth of blocks
%   base        - output filename base (no extension)
%   mode_suffix - '_min' or '_max' appended to filenames
%   gt          - ground truth struct with rf_center_s, adc_num_samples,
%                 adc_dwell_s for proper anchor computation
%
% waveforms_and_times returns:
%   wave_data      - cell array {gx, gy, gz}, each 2xN (time;amplitude)
%   tfp_excitation - Nx3 [time, freq, phase] for excitation RF pulses
%                    time = block_start + rf.delay (shape onset, NOT isocenter)
%   tfp_refocusing - Nx3 [time, freq, phase] for refocusing RF pulses
%   t_adc          - vector of ALL ADC sample times (s)
%
% Output files:
%   _tr<mode>_gx.csv, _tr<mode>_gy.csv, _tr<mode>_gz.csv
%   _tr<mode>_anchors.txt  (RF isocenter + ADC k=0 center times)

    [wave, tfp_exc, tfp_ref, t_adc] = tr_seq.waveforms_and_times();

    axis_labels = {'gx', 'gy', 'gz'};
    for c = 1:3
        fname = sprintf('%s_tr%s_%s.csv', base, mode_suffix, axis_labels{c});
        fid = fopen(fname, 'w');
        fprintf(fid, 'time_us,amplitude_hz_per_m\n');

        if c <= length(wave) && ~isempty(wave{c})
            t = wave{c}(1,:) * 1e6;   % seconds -> us
            a = wave{c}(2,:);          % Hz/m (Pulseq native)
            for k = 1:length(t)
                fprintf(fid, '%.6f,%.8g\n', t(k), a(k));
            end
        end
        fclose(fid);
    end

    % --- RF/ADC timing anchors ---
    fid = fopen(sprintf('%s_tr%s_anchors.txt', base, mode_suffix), 'w');

    % RF isocenter = tfp_excitation time + rf.center
    rf_center = 0;
    if isfield(gt, 'rf_center_s'), rf_center = gt.rf_center_s; end

    if ~isempty(tfp_exc)
        for k = 1:size(tfp_exc, 1)
            isocenter_us = (tfp_exc(k, 1) + rf_center) * 1e6;
            fprintf(fid, 'rf_isocenter_us %.6f\n', isocenter_us);
        end
    end

    if ~isempty(tfp_ref)
        rf_center_ref = rf_center;
        if isfield(gt, 'rf_refocus_center_s')
            rf_center_ref = gt.rf_refocus_center_s;
        end
        for k = 1:size(tfp_ref, 1)
            isocenter_us = (tfp_ref(k, 1) + rf_center_ref) * 1e6;
            fprintf(fid, 'rf_refocus_isocenter_us %.6f\n', isocenter_us);
        end
    end

    % ADC k-space center = first_sample_time + ceil(N/2) * dwell
    if ~isempty(t_adc) && isfield(gt, 'adc_num_samples') && isfield(gt, 'adc_dwell_s')
        % t_adc contains ALL sample times; find center of each ADC event
        N_adc = gt.adc_num_samples;
        dwell = gt.adc_dwell_s;
        num_events = length(t_adc) / N_adc;
        for ev = 1:num_events
            first_idx = (ev - 1) * N_adc + 1;
            kzero_us = (t_adc(first_idx) + ceil(N_adc / 2) * dwell) * 1e6;
            fprintf(fid, 'adc_kzero_us %.6f\n', kzero_us);
        end
    end

    fclose(fid);
end

function export_scan_table(base, N, num_averages, gt)
% Build and export the expected scan table for the given number of averages.
% The scan table maps scan positions to 0-based block indices, accounting
% for prep (once), main (repeated num_averages times), and cooldown (once).
%
% Output: _scan_table.csv with columns (scan_pos, block_idx)

    num_prep = gt.num_prep_blocks;
    num_cool = gt.num_cool_blocks;
    num_main = N - num_prep - num_cool;

    prep_idx = 0:(num_prep - 1);
    main_idx = num_prep:(num_prep + num_main - 1);
    cool_idx = (num_prep + num_main):(N - 1);

    block_idx = [prep_idx, repmat(main_idx, 1, num_averages), cool_idx];

    fid = fopen([base '_scan_table.csv'], 'w');
    fprintf(fid, 'scan_pos,block_idx\n');
    for p = 1:length(block_idx)
        fprintf(fid, '%d,%d\n', p - 1, block_idx(p));
    end
    fclose(fid);
end

function [amp, freq, phase] = extract_rf(blk)
    if isfield(blk, 'rf') && ~isempty(blk.rf)
        amp   = max(abs(blk.rf.signal));
        freq  = blk.rf.freqOffset;
        phase = blk.rf.phaseOffset;
    else
        amp = 0; freq = 0; phase = 0;
    end
end

function amp = extract_grad_amp(blk, ch)
    if isfield(blk, ch) && ~isempty(blk.(ch))
        g = blk.(ch);
        if isfield(g, 'amplitude')
            amp = g.amplitude;        % trapezoid
        elseif isfield(g, 'waveform')
            amp = max(abs(g.waveform)); % arbitrary / extended
        else
            amp = 0;
        end
    else
        amp = 0;
    end
end

function [flag, freq, phase] = extract_adc(blk)
    if isfield(blk, 'adc') && ~isempty(blk.adc)
        flag  = 1;
        freq  = blk.adc.freqOffset;
        phase = blk.adc.phaseOffset;
    else
        flag = 0; freq = 0; phase = 0;
    end
end


%% ========================================================================
%  bSSFP  (True FISP)
%  ========================================================================

function write_bssfp(num_averages)
    fprintf('Generating bSSFP (1 slice, %d avg) ...\n', num_averages);

    sys   = make_system();
    seq   = mr.Sequence(sys);
    fov   = 220e-3;
    Nx    = 256;
    Ny    = 256;

    % RF / ADC parameters
    adc_dur  = 2560e-6;            % readout flat time
    alpha    = 40;                 % flip angle [deg]
    thick    = 4e-3;               % slice thickness
    rf_dur   = 600e-6;

    % --- create events ---
    [rf, gz, gzReph] = mr.makeSincPulse(alpha * pi / 180, ...
        'Duration', rf_dur, 'SliceThickness', thick, ...
        'apodization', 0.5, 'timeBwProduct', 1.5, ...
        'system', sys, 'use', 'excitation');

    deltak = 1 / fov;
    gx     = mr.makeTrapezoid('x', 'FlatArea', Nx * deltak, ...
                              'FlatTime', adc_dur, 'system', sys);
    adc    = mr.makeAdc(Nx, 'Duration', gx.flatTime, ...
                        'Delay', gx.riseTime, 'system', sys);
    gxPre  = mr.makeTrapezoid('x', 'Area', -gx.area / 2, 'system', sys);
    phaseAreas = ((0:Ny-1) - Ny/2) * deltak;

    % --- split & combine for bSSFP optimal timing ---
    gz_parts = mr.splitGradientAt(gz, mr.calcDuration(rf));
    gz_parts(1).delay = mr.calcDuration(gzReph);
    gz_1 = mr.addGradients({gzReph, gz_parts(1)}, 'system', sys);
    [rf, ~] = mr.align('right', rf, gz_1);
    gz_parts(2).delay = 0;
    gzReph.delay = mr.calcDuration(gz_parts(2));
    gz_2 = mr.addGradients({gz_parts(2), gzReph}, 'system', sys);

    gx_parts = mr.splitGradientAt(gx, ...
        ceil(mr.calcDuration(adc) / sys.gradRasterTime) * sys.gradRasterTime);
    gx_parts(1).delay = mr.calcDuration(gxPre);
    gx_1 = mr.addGradients({gxPre, gx_parts(1)}, 'system', sys);
    adc.delay = adc.delay + mr.calcDuration(gxPre);
    gx_parts(2).delay = 0;
    gxPre.delay = mr.calcDuration(gx_parts(2));
    gx_2 = mr.addGradients({gx_parts(2), gxPre}, 'system', sys);

    pe_dur = mr.calcDuration(gx_2);

    gz_1.delay = max(mr.calcDuration(gx_2) - rf.delay + rf.ringdownTime, 0);
    rf.delay   = rf.delay + gz_1.delay;

    TR = mr.calcDuration(gz_1) + mr.calcDuration(gx_1);
    TE = TR / 2;

    % --- phase-encode template (max area, will be scaled) ---
    maxPeArea = max(abs(phaseAreas));
    gyMax     = mr.makeTrapezoid('y', 'Area', maxPeArea, ...
                                'Duration', pe_dur, 'system', sys);

    % --- pre-create labels ---
    lblOnce1 = mr.makeLabel('SET', 'ONCE', 1);
    lblOnce0 = mr.makeLabel('SET', 'ONCE', 0);
    lblOnce2 = mr.makeLabel('SET', 'ONCE', 2);

    % --- alpha/2 prep (ONCE=1) ---
    rf05        = rf;
    rf05.signal = 0.5 * rf.signal;
    seq.addBlock(rf05, gz_1, lblOnce1);

    prepDelay = mr.makeDelay( ...
        round((TR/2 - mr.calcDuration(gz_1)) / sys.gradRasterTime) ...
        * sys.gradRasterTime);
    gx_1_1    = mr.makeExtendedTrapezoidArea('x', 0, gx_2.first, -gx_2.area, sys);
    gyPre_2   = mr.scaleGrad(gyMax, phaseAreas(end) / maxPeArea);
    seq.addBlock(mr.align('left', prepDelay, gz_2, gyPre_2, 'right', gx_1_1));

    seq.addBlock(lblOnce0);   % clear ONCE flag -> first main block

    % --- main loop ---
    for i = 1:Ny
        rf.phaseOffset  = pi * mod(i, 2);
        adc.phaseOffset = pi * mod(i, 2);

        gyPre_1 = mr.scaleGrad(gyPre_2, -1);             % undo previous PE
        gyPre_2 = mr.scaleGrad(gyMax, phaseAreas(i) / maxPeArea);  % new PE

        seq.addBlock(rf, gz_1, gyPre_1, gx_2);
        seq.addBlock(gx_1, gyPre_2, gz_2, adc);
    end

    % --- exit block (ONCE=2) ---
    seq.addBlock(gx_2, lblOnce2);

    % sanity: prep = 3 blocks (rf05+gz_1, align block, lblOnce0)
    % first main TR starts at block 4
    assert(abs(TR - (mr.calcDuration(seq.getBlock(4)) ...
                   + mr.calcDuration(seq.getBlock(5)))) < 1e-12);

    fprintf('  TR = %.3f ms   TE = %.3f ms\n', TR * 1e3, TE * 1e3);

    % --- structural ground truth ---
    % bSSFP uses split/merged gradients that don't start at 0, so
    % representative TR sequences cannot be built as standalone sequences.
    gt.tr_min          = [];
    gt.tr_max          = [];
    gt.tr_prep         = [];
    gt.tr_cool         = [];
    gt.rf_center_s     = rf.center;
    gt.adc_num_samples = adc.numSamples;
    gt.adc_dwell_s     = adc.dwell;
    gt.seg_unique_ids  = {[0, 1]};   % single segment, full 2-block TR
    gt.unique_blocks   = [0, 1];
    gt.num_prep_blocks = 3;          % alpha/2 + align + lblOnce0
    gt.num_cool_blocks = 1;          % exit gx_2 block
    gt.degenerate_prep = 0;          % alpha/2 prep ~= main pattern
    gt.degenerate_cool = 0;          % exit block ~= main pattern

    fname = sprintf('bssfp_2d_%davg.seq', num_averages);
    check_and_write(seq, fname, fov, thick, 1, num_averages, gt);
end


%% ========================================================================
%  SPGR  (spoiled GRE with labels)
%  ========================================================================

function write_spgr(num_slices, num_averages)
    fprintf('Generating SPGR (%d slice, %d avg) ...\n', num_slices, num_averages);

    sys = make_system();
    seq = mr.Sequence(sys);

    fov       = 224e-3;
    Nx        = 256;
    Ny        = Nx;
    alpha     = 15;                 % flip angle [deg]
    thick     = 5e-3;
    Nslices   = num_slices;
    TR        = 10e-3;
    TE        = 4.3e-3;
    Ndummy    = 5;                  % dummy TRs for steady state
    rfSpoilInc = 84;               % RF spoiling increment [deg]
    roDur     = 2.560e-3;          % readout flat time: dwell=10us (mult of adcRaster=2us), trap=2680us (mult of blockRaster=20us)

    % --- events ---
    [rf, gz] = mr.makeSincPulse(alpha * pi / 180, ...
        'Duration', 3e-3, 'SliceThickness', thick, ...
        'apodization', 0.42, 'timeBwProduct', 4, ...
        'use', 'excitation', 'system', sys);

    deltak  = 1 / fov;
    gx      = mr.makeTrapezoid('x', 'FlatArea', Nx * deltak, 'FlatTime', roDur, 'system', sys);
    adc     = mr.makeAdc(Nx, 'Duration', gx.flatTime, 'Delay', gx.riseTime, 'system', sys);
    gxPre   = mr.makeTrapezoid('x', 'Area', -gx.area / 2, 'Duration', 1e-3, 'system', sys);
    gzReph  = mr.makeTrapezoid('z', 'Area', -gz.area / 2, 'Duration', 1e-3, 'system', sys);
    gxSpoil = mr.makeTrapezoid('x', 'Area', 2 * Nx * deltak, 'system', sys);
    gzSpoil = mr.makeTrapezoid('z', 'Area', 4 / thick, 'system', sys);

    % Phase-encode template
    phaseAreas = -((0:Ny-1) - Ny/2) * deltak;
    maxPeArea  = max(abs(phaseAreas));
    gyMax      = mr.makeTrapezoid('y', 'Area', maxPeArea, ...
                                  'Duration', mr.calcDuration(gxPre), ...
                                  'system', sys);

    % Timing delays
    delayTE = ceil((TE - mr.calcDuration(gxPre) - gz.fallTime ...
              - gz.flatTime / 2 - mr.calcDuration(gx) / 2) ...
              / sys.gradRasterTime) * sys.gradRasterTime;
    delayTR = ceil((TR - mr.calcDuration(gz) - mr.calcDuration(gxPre) ...
              - mr.calcDuration(gx) - delayTE) ...
              / sys.gradRasterTime) * sys.gradRasterTime;
    assert(delayTE >= 0, 'TE too short');
    assert(delayTR >= mr.calcDuration(gxSpoil, gzSpoil), 'TR too short');
    evDelayTE = mr.makeDelay(delayTE);
    evDelayTR = mr.makeDelay(delayTR);

    % Pre-create labels
    lblOnce1  = mr.makeLabel('SET', 'ONCE', 1);
    lblOnce0  = mr.makeLabel('SET', 'ONCE', 0);
    lblIncLin = mr.makeLabel('INC', 'LIN', 1);
    lblSetLin = mr.makeLabel('SET', 'LIN', 0);
    lblIncSlc = mr.makeLabel('INC', 'SLC', 1);
    lblSetSlc = mr.makeLabel('SET', 'SLC', 0);
    lblIncRep = mr.makeLabel('INC', 'REP', 1);

    rf_phase = 0;
    rf_inc   = 0;

    % --- ground truth: segment defs as unique block IDs ---
    seg_unique_ids = {[0, 1, 2, 3, 4]};  % single segment = full TR

    % --- representative TRs for waveform ground truth ---
    tr_min = mr.Sequence(sys);   % zero PE  (C library mode 2: definition-min)
    tr_max = mr.Sequence(sys);   % max  |PE| (C library mode 1: position-max)

    % --- build representative TR: min amplitude (zero PE, mode 2) ---
    tr_min.addBlock(rf, gz);
    tr_min.addBlock(gxPre, mr.scaleGrad(gyMax, 0.0), gzReph);
    tr_min.addBlock(evDelayTE);
    tr_min.addBlock(gx, adc);
    tr_min.addBlock(gxSpoil, mr.scaleGrad(gyMax, 0.0), gzSpoil, evDelayTR);

    % --- build representative TR: max positional amplitude (full PE, mode 1) ---
    tr_max.addBlock(rf, gz);
    tr_max.addBlock(gxPre, gyMax, gzReph);
    tr_max.addBlock(evDelayTE);
    tr_max.addBlock(gx, adc);
    tr_max.addBlock(gxSpoil, mr.scaleGrad(gyMax, -1), gzSpoil, evDelayTR);

    % --- prep: dummy scans (ONCE=1) ---
    for d = 1:Ndummy
        rf.phaseOffset = rf_phase / 180 * pi;
        rf_inc   = mod(rf_inc + rfSpoilInc, 360.0);
        rf_phase = mod(rf_phase + rf_inc, 360.0);

        if d == 1
            seq.addBlock(rf, gz, lblOnce1);
        else
            seq.addBlock(rf, gz);
        end

        seq.addBlock(gxPre, mr.scaleGrad(gyMax, 0.0), gzReph);     % no PE during dummies
        seq.addBlock(evDelayTE);
        seq.addBlock(gx);                % no ADC
        seq.addBlock(gxSpoil, mr.scaleGrad(gyMax, 0.0), gzSpoil, evDelayTR);
    end

    % --- main imaging loop ---
    for i = 1:Ny
        for s = 1:Nslices
            rf.freqOffset  = gz.amplitude * thick * (s - 1 - (Nslices-1)/2);
            rf.phaseOffset = rf_phase / 180 * pi;
            adc.phaseOffset = rf_phase / 180 * pi;
            rf_inc   = mod(rf_inc + rfSpoilInc, 360.0);
            rf_phase = mod(rf_phase + rf_inc, 360.0);
            if maxPeArea > 0
                pe_scale = phaseAreas(i) / maxPeArea;
            else                
                pe_scale = 0;
            end

            if i == 1 && s == 1
                seq.addBlock(rf, gz, lblOnce0);
            else
                seq.addBlock(rf, gz);
            end
            seq.addBlock(gxPre, mr.scaleGrad(gyMax, pe_scale), gzReph);
            seq.addBlock(evDelayTE);
            seq.addBlock(gx, adc);

            % Spoiler + rewind PE
            gyRew = mr.scaleGrad(gyMax, -pe_scale);
            if i == Ny && s == Nslices
                seq.addBlock(gxSpoil, gyRew, gzSpoil, evDelayTR, lblSetLin, lblSetSlc);
            elseif s == Nslices
                seq.addBlock(gxSpoil, gyRew, gzSpoil, evDelayTR, lblIncLin, lblSetSlc);
            else
                seq.addBlock(gxSpoil, gyRew, gzSpoil, evDelayTR, lblIncSlc);
            end
        end
    end

    fprintf('  TR = %.3f ms   TE = %.3f ms   Ndummy = %d\n', ...
            TR * 1e3, TE * 1e3, Ndummy);

    % --- structural ground truth ---
    gt.tr_min          = tr_min;
    gt.tr_max          = tr_max;
    gt.rf_center_s     = rf.center;                         % RF isocenter offset (s)
    gt.adc_num_samples = adc.numSamples;                    % samples per readout
    gt.adc_dwell_s     = adc.dwell;                         % dwell time (s)
    gt.seg_unique_ids  = seg_unique_ids;
    gt.unique_blocks   = [0, 1, 2, 3, 4];
    gt.num_prep_blocks = Ndummy * 5;  % 5 blocks per dummy TR (no slice loop in dummies)
    gt.num_cool_blocks = 0;
    gt.degenerate_prep = 1;    % dummy TR pattern == imaging TR pattern
    gt.degenerate_cool = 0;    % no cooldown blocks

    fname = seq_filename('gre_2d', num_slices, num_averages);
    check_and_write(seq, fname, fov, thick, num_slices, num_averages, gt);
end


%% ========================================================================
%  FSE  (fast spin echo)
%  ========================================================================

function write_fse(num_slices, num_averages)
    fprintf('Generating FSE (%d slice, %d avg) ...\n', num_slices, num_averages);

    sys = make_system();
    seq = mr.Sequence(sys);

    fov     = 256e-3;
    Nx      = 256;
    Ny      = 256;
    necho   = 16;
    Nslices = num_slices;
    thick   = 5e-3;

    rflip  = 180 * ones(1, necho);
    TE1    = 12e-3;
    TR     = 2000e-3;
    TEeff  = 100e-3;
    Ndummy = 1;        % one dummy excitation

    samplingTime = 5.120e-3;       % dwell = 20 us (mult of adcRaster); samplingTime+2*adcDeadTime on gradRaster
    readoutTime  = samplingTime + 2 * sys.adcDeadTime;
    tEx   = 2.5e-3;
    tExwd = tEx + sys.rfRingdownTime + sys.rfDeadTime;
    tRef  = 3e-3;
    tRefwd = tRef + sys.rfRingdownTime + sys.rfDeadTime;
    tSp    = 0.5 * (TE1 - readoutTime - tRefwd);
    tSp    = sys.blockDurationRaster * round(tSp / sys.blockDurationRaster);
    tSpex  = 0.5 * (TE1 - tExwd - tRefwd);
    tSpex  = sys.blockDurationRaster * round(tSpex / sys.blockDurationRaster);
    fspR   = 1.0;
    fspS   = 0.5;
    dG     = 260e-6;    % ramp time (multiple of 20 us grad raster)

    rfex_phase  = pi / 2;
    rfref_phase = 0;

    % --- RF pulses ---
    flipex = 90 * pi / 180;
    [rfex, gz_ex] = mr.makeSincPulse(flipex, sys, ...
        'Duration', tEx, 'SliceThickness', thick, ...
        'apodization', 0.5, 'timeBwProduct', 4, ...
        'PhaseOffset', rfex_phase, 'use', 'excitation');

    flipref = rflip(1) * pi / 180;
    [rfref, ~] = mr.makeSincPulse(flipref, sys, ...
        'Duration', tRef, 'SliceThickness', thick, ...
        'apodization', 0.5, 'timeBwProduct', 4, ...
        'PhaseOffset', rfref_phase, 'use', 'refocusing');

    GSex  = mr.makeTrapezoid('z', sys, 'amplitude', gz_ex.amplitude, ...
                             'FlatTime', tExwd, 'riseTime', dG);
    GSref = mr.makeTrapezoid('z', sys, 'amplitude', GSex.amplitude, ...
                             'FlatTime', tRefwd, 'riseTime', dG);

    AGSex  = GSex.area / 2;
    GSspr  = mr.makeTrapezoid('z', sys, 'area', AGSex * (1 + fspS), ...
                              'duration', tSp, 'riseTime', dG);
    GSspex = mr.makeTrapezoid('z', sys, 'area', AGSex * fspS, ...
                              'duration', tSpex, 'riseTime', dG);

    % --- readout ---
    deltak  = 1 / fov;
    kWidth  = Nx * deltak;
    GRacq  = mr.makeTrapezoid('x', sys, 'FlatArea', kWidth, ...
                              'FlatTime', readoutTime, 'riseTime', dG);
    adc    = mr.makeAdc(Nx, 'Duration', samplingTime, 'Delay', sys.adcDeadTime);
    GRspr  = mr.makeTrapezoid('x', sys, 'area', GRacq.area * fspR, ...
                              'duration', tSp, 'riseTime', dG);
    GRpreph = mr.makeTrapezoid('x', sys, 'Area', ...
                               GRacq.area / 2 + GRspr.area, ...
                               'duration', tSpex, 'riseTime', dG);

    % --- phase-encode ordering ---
    nex = floor(Ny / necho);
    pe_steps = (1:(necho * nex)) - 0.5 * necho * nex - 1;
    if mod(necho, 2) == 0
        pe_steps = circshift(pe_steps, [0, -round(nex/2)]);
    end
    [~, iPEmin] = min(abs(pe_steps));
    k0curr      = floor((iPEmin - 1) / nex) + 1;
    k0prescr    = max(round(TEeff / TE1), 1);
    PEorder     = circshift(reshape(pe_steps, [nex, necho])', k0prescr - k0curr);
    phaseAreas  = PEorder * deltak;

    % --- phase-encode template (max area) ---
    maxPeArea = max(abs(phaseAreas(:)));
    gyMax     = mr.makeTrapezoid('y', sys, 'Area', maxPeArea, ...
                                'Duration', tSp, 'riseTime', dG);

    % --- split gradients for optimal timing ---
    % Slice-select splits
    GS1times = [0, GSex.riseTime];
    GS1amp   = [0, GSex.amplitude];
    GS1 = mr.makeExtendedTrapezoid('z', 'times', GS1times, 'amplitudes', GS1amp);

    GS2times = [0, GSex.flatTime];
    GS2amp   = [GSex.amplitude, GSex.amplitude];
    GS2 = mr.makeExtendedTrapezoid('z', 'times', GS2times, 'amplitudes', GS2amp);

    GS3times = [0, GSspex.riseTime, ...
                GSspex.riseTime + GSspex.flatTime, ...
                GSspex.riseTime + GSspex.flatTime + GSspex.fallTime];
    GS3amp   = [GSex.amplitude, GSspex.amplitude, GSspex.amplitude, GSref.amplitude];
    GS3 = mr.makeExtendedTrapezoid('z', 'times', GS3times, 'amplitudes', GS3amp);

    GS4times = [0, GSref.flatTime];
    GS4amp   = [GSref.amplitude, GSref.amplitude];
    GS4 = mr.makeExtendedTrapezoid('z', 'times', GS4times, 'amplitudes', GS4amp);

    GS5times = [0, GSspr.riseTime, ...
                GSspr.riseTime + GSspr.flatTime, ...
                GSspr.riseTime + GSspr.flatTime + GSspr.fallTime];
    GS5amp   = [GSref.amplitude, GSspr.amplitude, GSspr.amplitude, 0];
    GS5 = mr.makeExtendedTrapezoid('z', 'times', GS5times, 'amplitudes', GS5amp);

    GS7times = [0, GSspr.riseTime, ...
                GSspr.riseTime + GSspr.flatTime, ...
                GSspr.riseTime + GSspr.flatTime + GSspr.fallTime];
    GS7amp   = [0, GSspr.amplitude, GSspr.amplitude, GSref.amplitude];
    GS7 = mr.makeExtendedTrapezoid('z', 'times', GS7times, 'amplitudes', GS7amp);

    % Readout splits
    GR3 = GRpreph;
    GR5times = [0, GRspr.riseTime, ...
                GRspr.riseTime + GRspr.flatTime, ...
                GRspr.riseTime + GRspr.flatTime + GRspr.fallTime];
    GR5amp   = [0, GRspr.amplitude, GRspr.amplitude, GRacq.amplitude];
    GR5 = mr.makeExtendedTrapezoid('x', 'times', GR5times, 'amplitudes', GR5amp);

    GR6times = [0, readoutTime];
    GR6amp   = [GRacq.amplitude, GRacq.amplitude];
    GR6 = mr.makeExtendedTrapezoid('x', 'times', GR6times, 'amplitudes', GR6amp);

    GR7times = [0, GRspr.riseTime, ...
                GRspr.riseTime + GRspr.flatTime, ...
                GRspr.riseTime + GRspr.flatTime + GRspr.fallTime];
    GR7amp   = [GRacq.amplitude, GRspr.amplitude, GRspr.amplitude, 0];
    GR7 = mr.makeExtendedTrapezoid('x', 'times', GR7times, 'amplitudes', GR7amp);

    % Timing
    tex     = mr.calcDuration(GS1) + mr.calcDuration(GS2) + mr.calcDuration(GS3);
    tref    = mr.calcDuration(GS4) + mr.calcDuration(GS5) + mr.calcDuration(GS7) + readoutTime;
    tend    = mr.calcDuration(GS4) + mr.calcDuration(GS5);
    tETrain = tex + necho * tref + tend;
    TRfill  = (TR - Nslices * tETrain) / Nslices;
    TRfill  = sys.gradRasterTime * round(TRfill / sys.gradRasterTime);
    if TRfill < 0
        TRfill = 1e-3;
        fprintf('  Warning: TR too short, adapted to %.1f ms\n', ...
                1000 * Nslices * (tETrain + TRfill));
    end
    delayTR = mr.makeDelay(TRfill);

    % --- labels ---
    lblOnce1 = mr.makeLabel('SET', 'ONCE', 1);
    lblOnce0 = mr.makeLabel('SET', 'ONCE', 0);

    % --- ground truth: segment defs as unique block IDs ---
    % Block defs (based on gradient/RF structure, ADC not in key):
    %   0: GS1                          (slice-select ramp-up)
    %   1: GS2 + rfex                   (excitation)
    %   2: GS3 + GR3                    (transition + readout prephasing)
    %   3: GS4 + rfref                  (refocusing)
    %   4: GS5 + GR5 + GPpre            (spoiler + readout pre + PE)
    %   5: GR6                           (readout flat, +/- ADC)
    %   6: GS7 + GR7 + GPrew            (spoiler + readout post + PE rewind)
    %   7: GS4                           (end crusher, no RF)
    %   8: GS5                           (end spoiler)
    %   9: delayTR                       (TR fill delay)
    echo_pattern = repmat([3, 4, 5, 6], 1, necho);
    seg0_ids = [0, 1, 2, echo_pattern, 7, 8];  % echo train segment
    seg1_ids = 9;                                % delay segment
    seg_unique_ids = {seg0_ids, seg1_ids};
    unique_blocks  = [seg0_ids, seg1_ids];

    % --- representative TRs for waveform ground truth ---
    tr_min = mr.Sequence(sys);  % zero PE (C library mode 2: definition-min)
    tr_max = mr.Sequence(sys);  % max |PE| (C library mode 1: position-max)

    % Build tr_min (zero PE throughout)
    tr_min.addBlock(GS1);
    tr_min.addBlock(GS2, rfex);
    tr_min.addBlock(GS3, GR3);
    for kech = 1:necho
        tr_min.addBlock(GS4, rfref);
        tr_min.addBlock(GS5, GR5, mr.scaleGrad(gyMax, 0));
        tr_min.addBlock(GR6, adc);
        tr_min.addBlock(GS7, GR7, mr.scaleGrad(gyMax, 0));
    end
    tr_min.addBlock(GS4);
    tr_min.addBlock(GS5);
    tr_min.addBlock(delayTR);

    % Build tr_max (max |PE|)
    tr_max.addBlock(GS1);
    tr_max.addBlock(GS2, rfex);
    tr_max.addBlock(GS3, GR3);
    for kech = 1:necho
        phaseArea = phaseAreas(kech, 1);
        if maxPeArea > 0
            pe_scale = phaseArea / maxPeArea;
        else
            pe_scale = 0;
        end
        GPpre = mr.scaleGrad(gyMax, pe_scale);
        GPrew = mr.scaleGrad(gyMax, -pe_scale);
        tr_max.addBlock(GS4, rfref);
        tr_max.addBlock(GS5, GR5, GPpre);
        tr_max.addBlock(GR6, adc);
        tr_max.addBlock(GS7, GR7, GPrew);
    end
    tr_max.addBlock(GS4);
    tr_max.addBlock(GS5);
    tr_max.addBlock(delayTR);

    % --- main imaging loop ---
    for kex = 0:nex
        for s = 1:Nslices
            rfex.freqOffset  = GSex.amplitude * thick * (s - 1 - (Nslices-1)/2);
            rfref.freqOffset = GSref.amplitude * thick * (s - 1 - (Nslices-1)/2);
            rfex.phaseOffset  = rfex_phase - 2*pi * rfex.freqOffset * mr.calcRfCenter(rfex);
            rfref.phaseOffset = rfref_phase - 2*pi * rfref.freqOffset * mr.calcRfCenter(rfref);

            if kex == 0 && s == 1
                seq.addBlock(GS1, lblOnce1);  % start of prep (dummy excitation)
            elseif kex == Ndummy && s == 1
                seq.addBlock(GS1, lblOnce0);  % end prep, start of main
            else
                seq.addBlock(GS1);
            end
            seq.addBlock(GS2, rfex);
            seq.addBlock(GS3, GR3);

            for kech = 1:necho
                if kex > 0
                    phaseArea = phaseAreas(kech, kex);
                else
                    phaseArea = 0;
                end
                if maxPeArea > 0 && kex > 0
                    pe_scale = phaseArea / maxPeArea;
                else
                    pe_scale = 0;
                end
                GPpre = mr.scaleGrad(gyMax, pe_scale);
                GPrew = mr.scaleGrad(gyMax, -pe_scale);

                seq.addBlock(GS4, rfref);
                seq.addBlock(GS5, GR5, GPpre);
                if kex > 0
                    seq.addBlock(GR6, adc);
                else
                    seq.addBlock(GR6);
                end
                seq.addBlock(GS7, GR7, GPrew);
            end

            seq.addBlock(GS4);
            seq.addBlock(GS5);
            seq.addBlock(delayTR);
        end
    end

    blocks_per_tr = 3 + 4*necho + 3;  % excitation + echo train + end + delay

    % --- structural ground truth ---
    gt.tr_min              = tr_min;
    gt.tr_max              = tr_max;
    gt.rf_center_s         = rfex.center;
    gt.rf_refocus_center_s = rfref.center;
    gt.adc_num_samples     = adc.numSamples;
    gt.adc_dwell_s         = adc.dwell;
    gt.seg_unique_ids      = seg_unique_ids;
    gt.unique_blocks       = unique_blocks;
    gt.num_prep_blocks     = Ndummy * blocks_per_tr * Nslices;
    gt.num_cool_blocks     = 0;
    gt.degenerate_prep     = 1;  % dummy uses same block defs (ADC not in dedup key)
    gt.degenerate_cool     = 0;

    fname = seq_filename('fse_2d', num_slices, num_averages);
    check_and_write(seq, fname, fov, thick, num_slices, num_averages, gt);
end


%% ========================================================================
%  EPI (echo-planar imaging)
%  ========================================================================

function write_epi(num_slices, num_averages)
    fprintf('Generating EPI (%d slice, %d avg) ...\n', num_slices, num_averages);

    sys = make_system();
    seq = mr.Sequence(sys);

    fov       = 220e-3;
    Nx        = 96;
    Ny        = Nx;
    thick     = 3e-3;
    sliceGap  = 1.5e-3;
    Nslices   = num_slices;
    TR        = 3000e-3;
    ro_os     = 2;
    readoutTime = 580e-6;
    partFourierFactor = 1;
    Nnav      = 3;          % navigator echoes
    pe_enable = 1;

    % Fat-sat pulse
    sat_ppm = -3.35;
    rf_fs = mr.makeGaussPulse(110 * pi / 180, 'system', sys, ...
        'Duration', 8e-3, ...
        'bandwidth', abs(sat_ppm * 1e-6 * sys.B0 * sys.gamma), ...
        'freqPPM', sat_ppm, 'use', 'saturation');
    rf_fs.phasePPM = -2*pi * rf_fs.freqPPM * rf_fs.center;
    gz_fs = mr.makeTrapezoid('z', sys, 'delay', mr.calcDuration(rf_fs), 'Area', 0.1 / 1e-4);

    % Excitation
    [rf, gz, gzReph] = mr.makeSincPulse(pi / 2, 'system', sys, ...
        'Duration', 2e-3, 'SliceThickness', thick, ...
        'apodization', 0.42, 'timeBwProduct', 4, 'use', 'excitation');

    trig = mr.makeDigitalOutputPulse('osc0', 'duration', 100e-6);

    % Readout gradient
    deltak = 1 / fov;
    blip_dur = ceil(2 * sqrt(deltak / sys.maxSlew) / sys.gradRasterTime / 2) ...
               * sys.gradRasterTime * 2;
    gy = mr.makeTrapezoid('y', sys, 'Area', -deltak, 'Duration', blip_dur);

    extra_area = blip_dur/2 * blip_dur/2 * sys.maxSlew;
    gx = mr.makeTrapezoid('x', sys, 'Area', deltak * Nx + extra_area, 'Duration', readoutTime + blip_dur);
    actual_area = gx.area ...
        - gx.amplitude / gx.riseTime  * blip_dur/2 * blip_dur/2 / 2 ...
        - gx.amplitude / gx.fallTime  * blip_dur/2 * blip_dur/2 / 2;
    gx.amplitude = gx.amplitude / actual_area * (Nx * deltak);
    gx.area      = gx.amplitude * (gx.flatTime + gx.riseTime/2 + gx.fallTime/2);
    gx.flatArea  = gx.amplitude * gx.flatTime;
    assert(gx.amplitude <= sys.maxGrad, 'Readout gradient exceeds maxGrad');

    % ADC
    assert(ro_os >= 2);
    adcSamples = Nx * ro_os;
    adcDwell   = sys.adcRasterTime * floor(readoutTime / adcSamples / sys.adcRasterTime);
    adc = mr.makeAdc(adcSamples, 'Dwell', adcDwell, 'Delay', blip_dur / 2);
    time_to_center = adc.dwell * ((adcSamples - 1)/2 + 0.5);
    adc.delay = round((gx.riseTime + gx.flatTime/2 - time_to_center) * 1e6) * 1e-6;

    % Split blips
    gy_parts = mr.splitGradientAt(gy, blip_dur / 2, sys);
    [gy_blipup, gy_blipdown, ~] = mr.align('right', gy_parts(1), 'left', gy_parts(2), gx);
    gy_blipdownup = mr.addGradients({gy_blipdown, gy_blipup}, sys);

    gy_blipup.waveform     = gy_blipup.waveform * pe_enable;
    gy_blipdown.waveform   = gy_blipdown.waveform * pe_enable;
    gy_blipdownup.waveform = gy_blipdownup.waveform * pe_enable;

    % Phase encoding
    Ny_pre  = round(partFourierFactor * Ny / 2 - 1);
    Ny_post = round(Ny / 2 + 1);
    Ny_meas = Ny_pre + Ny_post;

    % Pre-phasing
    gxPre = mr.makeTrapezoid('x', sys, 'Area', -gx.area / 2);
    gyPre = mr.makeTrapezoid('y', sys, 'Area', Ny_pre * deltak);
    [gxPre, gyPre, gzReph] = mr.align('right', gxPre, 'left', gyPre, gzReph);
    gyPre = mr.makeTrapezoid('y', sys, 'Area', gyPre.area, ...
        'Duration', mr.calcDuration(gxPre, gyPre, gzReph));
    gyPre.amplitude = gyPre.amplitude * pe_enable;

    % Slice positions (interleaved)
    slicePositions = (thick + sliceGap) * ((0:(Nslices-1)) - (Nslices-1)/2);
    slicePositions = slicePositions([1:2:Nslices, 2:2:Nslices]);

    % TR timing
    minTR_1slice = mr.calcDuration(gz_fs) + mr.calcDuration(gz) ...
                 + mr.calcDuration(gzReph) + Nnav * mr.calcDuration(gx) ...
                 + mr.calcDuration(gyPre) + Ny_meas * mr.calcDuration(gx);
    TRdelay = TR - minTR_1slice * Nslices;
    TRdelay_perSlice = round(TRdelay / Nslices / sys.blockDurationRaster) ...
                       * sys.blockDurationRaster;
    assert(TRdelay_perSlice > 0, 'TR too short for EPI');

    ROpolarity = sign(gx.amplitude);

    % Labels
    lblOnce1  = mr.makeLabel('SET', 'ONCE', 1);
    lblOnce0  = mr.makeLabel('SET', 'ONCE', 0);
    lblSetSlc = mr.makeLabel('SET', 'SLC', 0);
    lblIncSlc = mr.makeLabel('INC', 'SLC', 1);
    lblIncRep = mr.makeLabel('INC', 'REP', 1);

    % --- prep (ONCE=1): one dummy volume ---
    seq.addBlock(lblOnce1);
    seq.addBlock(lblSetSlc);
    for s = 1:Nslices
        seq.addBlock(rf_fs, gz_fs);
        rf.freqOffset  = gz.amplitude * slicePositions(s);
        rf.phaseOffset = -2*pi * rf.freqOffset * rf.center;
        seq.addBlock(rf, gz, trig);

        % Prephasing (reverse for navigator)
        gxPre_nav = mr.scaleGrad(gxPre, -1);
        gx_nav    = mr.scaleGrad(gx, -1);
        seq.addBlock(gxPre_nav, gzReph);
        gxPre_nav = mr.scaleGrad(gxPre_nav, -1);  % restore

        for n = 1:Nnav
            seq.addBlock(gx_nav);     % no ADC during dummies
            gx_nav = mr.scaleGrad(gx_nav, -1);
        end

        seq.addBlock(gyPre);

        for i = 1:Ny_meas
            if i == 1
                seq.addBlock(gx, gy_blipup);
            elseif i == Ny_meas
                seq.addBlock(gx, gy_blipdown);
            else
                seq.addBlock(gx, gy_blipdownup);
            end
            gx = mr.scaleGrad(gx, -1);
        end

        seq.addBlock(lblIncSlc);
        if sign(gx.amplitude) ~= ROpolarity
            gx = mr.scaleGrad(gx, -1);
        end
        seq.addBlock(TRdelay_perSlice);
    end
    seq.addBlock(lblOnce0);  % end prep

    % --- main: imaging volume ---
    seq.addBlock(lblSetSlc);
    for s = 1:Nslices
        seq.addBlock(rf_fs, gz_fs);
        rf.freqOffset  = gz.amplitude * slicePositions(s);
        rf.phaseOffset = -2*pi * rf.freqOffset * rf.center;
        seq.addBlock(rf, gz, trig);

        if Nnav > 0
            gxPre_nav = mr.scaleGrad(gxPre, -1);
            gx_tmp    = mr.scaleGrad(gx, -1);
            seq.addBlock(gxPre_nav, gzReph, ...
                mr.makeLabel('SET', 'NAV', 1), ...
                mr.makeLabel('SET', 'LIN', floor(Ny/2)));
            gxPre_nav = mr.scaleGrad(gxPre_nav, -1);

            for n = 1:Nnav
                seq.addBlock( ...
                    mr.makeLabel('SET', 'REV', sign(gx_tmp.amplitude) ~= ROpolarity), ...
                    mr.makeLabel('SET', 'SEG', sign(gx_tmp.amplitude) ~= ROpolarity), ...
                    mr.makeLabel('SET', 'AVG', n == Nnav));
                seq.addBlock(gx_tmp, adc);
                gx_tmp = mr.scaleGrad(gx_tmp, -1);
            end

            seq.addBlock(gyPre, ...
                mr.makeLabel('SET', 'LIN', -1), ...
                mr.makeLabel('SET', 'NAV', 0), ...
                mr.makeLabel('SET', 'AVG', 0));
        else
            seq.addBlock(gxPre, gyPre, gzReph, ...
                mr.makeLabel('SET', 'LIN', -1), ...
                mr.makeLabel('SET', 'NAV', 0), ...
                mr.makeLabel('SET', 'AVG', 0));
        end

        for i = 1:Ny_meas
            lrev = mr.makeLabel('SET', 'REV', sign(gx.amplitude) ~= ROpolarity);
            lseg = mr.makeLabel('SET', 'SEG', sign(gx.amplitude) ~= ROpolarity);
            llin = mr.makeLabel('INC', 'LIN', 1);

            if i == 1
                seq.addBlock(gx, gy_blipup, adc, lrev, lseg, llin);
            elseif i == Ny_meas
                seq.addBlock(gx, gy_blipdown, adc, lrev, lseg, llin);
            else
                seq.addBlock(gx, gy_blipdownup, adc, lrev, lseg, llin);
            end
            gx = mr.scaleGrad(gx, -1);
        end

        seq.addBlock(lblIncSlc);
        if sign(gx.amplitude) ~= ROpolarity
            gx = mr.scaleGrad(gx, -1);
        end
        seq.addBlock(TRdelay_perSlice);
    end

    % Definitions
    seq.setDefinition('Name', 'epi');
    seq.setDefinition('SlicePositions', slicePositions);
    seq.setDefinition('SliceThickness', thick);
    seq.setDefinition('SliceGap', sliceGap);
    seq.setDefinition('ReadoutOversamplingFactor', ro_os);

    % --- representative TRs for waveform ground truth ---
    % EPI: each "TR" is one slice excitation through readout train.
    % Block structure per slice (main):
    %   0: rf_fs + gz_fs         (fat-sat)
    %   1: rf + gz + trig        (excitation)
    %   2: gxPre_nav + gzReph    (prephasing, reversed for nav)
    %   3..3+2*Nnav-1: label + gx_tmp+adc  (navigator pairs)
    %   3+2*Nnav: gyPre + labels (PE prephasing)
    %   then Ny_meas readout blocks: gx + blip + adc
    %   Ny_meas+...: lblIncSlc, TRdelay

    tr_min = mr.Sequence(sys);  % definition-min (mode 2)
    tr_max = mr.Sequence(sys);  % position-max (mode 1)

    % For EPI the readout gradient alternates polarity each line.
    % min-amplitude: just the readout train structure (no PE blips contribute)
    % max-amplitude: same structure (PE blips are same for all shots)
    % Both are identical for EPI since there's only one shot pattern.

    % Fat-sat + excitation
    tr_min.addBlock(rf_fs, gz_fs);
    tr_min.addBlock(rf, gz, trig);
    tr_max.addBlock(rf_fs, gz_fs);
    tr_max.addBlock(rf, gz, trig);

    % Navigator prephasing + navigator echoes
    gxPre_nav = mr.scaleGrad(gxPre, -1);
    gx_tmp    = mr.scaleGrad(gx, -1);
    tr_min.addBlock(gxPre_nav, gzReph);
    tr_max.addBlock(gxPre_nav, gzReph);
    for n = 1:Nnav
        tr_min.addBlock(gx_tmp, adc);
        tr_max.addBlock(gx_tmp, adc);
        gx_tmp = mr.scaleGrad(gx_tmp, -1);
    end

    % PE prephasing
    tr_min.addBlock(gyPre);
    tr_max.addBlock(gyPre);

    % Readout train
    gx_ro = gx;  % ensure positive polarity at start
    if sign(gx_ro.amplitude) ~= ROpolarity
        gx_ro = mr.scaleGrad(gx_ro, -1);
    end
    for i = 1:Ny_meas
        if i == 1
            tr_min.addBlock(gx_ro, gy_blipup, adc);
            tr_max.addBlock(gx_ro, gy_blipup, adc);
        elseif i == Ny_meas
            tr_min.addBlock(gx_ro, gy_blipdown, adc);
            tr_max.addBlock(gx_ro, gy_blipdown, adc);
        else
            tr_min.addBlock(gx_ro, gy_blipdownup, adc);
            tr_max.addBlock(gx_ro, gy_blipdownup, adc);
        end
        gx_ro = mr.scaleGrad(gx_ro, -1);
    end

    % TR delay
    tr_min.addBlock(TRdelay_perSlice);
    tr_max.addBlock(TRdelay_perSlice);

    % --- structural ground truth ---
    % Block defs (dedup key = duration, rf_def, gx_def, gy_def, gz_def;
    %             amplitude is scalar, NOT in key):
    %   0: label-only               (min-duration: ONCE, SLC, NAV, SEG, etc.)
    %   1: rf_fs + gz_fs            (fat-sat)
    %   2: rf + gz + trig           (excitation + slice-select)
    %   3: gxPre + gzReph           (nav/readout prephasing)
    %   4: gx + adc                 (nav readout, shape-only)
    %   5: gyPre                    (PE prephasing)
    %   6: gx + gy_blipup + adc     (first readout line)
    %   7: gx + gy_blipdownup + adc (middle readout lines)
    %   8: gx + gy_blipdown + adc   (last readout line)
    %   9: TRdelay                  (per-slice delay)

    % Prep: lblOnce1 + lblSetSlc + Nslices*(2+1+Nnav+1+Ny_meas+1+1) + lblOnce0
    blocks_per_slice_prep = 2 + 1 + Nnav + 1 + Ny_meas + 1 + 1;
    gt.tr_min          = tr_min;
    gt.tr_max          = tr_max;
    gt.rf_center_s     = rf.center;
    gt.adc_num_samples = adc.numSamples;
    gt.adc_dwell_s     = adc.dwell;
    gt.seg_unique_ids  = {0:9};       % single segment = full per-volume TR
    gt.unique_blocks   = 0:9;
    gt.num_prep_blocks = 2 + blocks_per_slice_prep * Nslices + 1;
    gt.num_cool_blocks = 0;
    gt.degenerate_prep = 0;           % nav structure differs (no ADC, no labels)
    gt.degenerate_cool = 0;

    fname = seq_filename('epi_2d', num_slices, num_averages);
    check_and_write(seq, fname, fov, thick, num_slices, num_averages, gt);
end


%% ========================================================================
%  MPRAGE (3D inversion-recovery GRE)
%  ========================================================================

function write_mprage(num_slices, num_averages)
    fprintf('Generating MPRAGE (%d slice, %d avg) ...\n', num_slices, num_averages);

    sys = make_system();
    seq = mr.Sequence(sys);

    alpha      = 7;             % flip angle [deg]
    ro_dur     = 5040e-6;       % RO duration (multiple of 20us)
    ro_os      = 1;
    ro_spoil   = 3;
    TI         = 1.1;
    TRout      = 2.5;
    rfSpoilInc = 84;
    rfLen      = 100e-6;

    fov = [256, 240, 192] * 1e-3;   % [x, y, z]
    Nx  = 256;                       % readout (x)
    Ny  = 240;                       % PE1 (y) — inner loop
    Nz  = 192;                       % partition (z) — outer loop

    % --- events ---
    rf180 = mr.makeBlockPulse(pi, sys, ...
        'Duration', 10e-3, 'use', 'excitation');
    rf = mr.makeBlockPulse(alpha * pi / 180, sys, ...
        'Duration', rfLen, 'use', 'excitation');
    
    deltak = 1 ./ fov;
    gro    = mr.makeTrapezoid('x', 'Amplitude', ...
        Nx * deltak(1) / ro_dur, ...
        'FlatTime', ceil((ro_dur + sys.adcDeadTime) / sys.gradRasterTime) ...
                    * sys.gradRasterTime, 'system', sys);
    adc    = mr.makeAdc(Nx * ro_os, 'Duration', ro_dur, ...
                        'Delay', gro.riseTime, 'system', sys);
    groPre = mr.makeTrapezoid('x', 'Area', ...
        -gro.amplitude * (adc.dwell * (adc.numSamples/2 + 0.5) ...
         + 0.5 * gro.riseTime), 'system', sys);
    gpe1   = mr.makeTrapezoid('y', 'Area', -deltak(2) * Ny / 2, 'system', sys);
    gpe2   = mr.makeTrapezoid('z', 'Area', -deltak(3) * Nz / 2, 'system', sys);
    gslSp  = mr.makeTrapezoid('z', 'Area', max(deltak .* [Nx Ny Nz]) * 4, ...
                              'Duration', 10e-3, 'system', sys);

    [gro1, groSp] = mr.splitGradientAt(gro, gro.riseTime + gro.flatTime);
    if ro_spoil > 0
        groSp = mr.makeExtendedTrapezoidArea('x', gro.amplitude, 0, ...
                    deltak(1) / 2 * Nx * ro_spoil, sys);
    end

    rf.delay = mr.calcDuration(groSp, gpe1, gpe2);
    [groPre, ~, ~] = mr.align('right', groPre, gpe1, gpe2);
    gro1.delay = mr.calcDuration(groPre);
    adc.delay  = gro1.delay + gro.riseTime;
    gro1 = mr.addGradients({gro1, groPre}, 'system', sys);

    TRinner = mr.calcDuration(rf) + mr.calcDuration(gro1);

    pe1Steps = ((0:Ny-1) - Ny/2) / Ny * 2;
    pe2Steps = ((0:Nz-1) - Nz/2) / Nz * 2;

    TIdelay = round((TI - (find(pe1Steps==0) - 1) * TRinner ...
              - (mr.calcDuration(rf180) - mr.calcRfCenter(rf180) - rf180.delay) ...
              - rf.delay - mr.calcRfCenter(rf)) / sys.blockDurationRaster) ...
              * sys.blockDurationRaster;
    TRoutDelay = TRout - TRinner * Ny - TIdelay - mr.calcDuration(rf180);

    % Pre-create labels
    lblIncLin   = mr.makeLabel('INC', 'LIN', 1);
    lblIncPar   = mr.makeLabel('INC', 'PAR', 1);
    lblResetPar = mr.makeLabel('SET', 'PAR', 0);
    lblOnce1    = mr.makeLabel('SET', 'ONCE', 1);
    lblOnce0    = mr.makeLabel('SET', 'ONCE', 0);

    % Pre-register unchanging events
    gslSp.id  = seq.registerGradEvent(gslSp);
    groSp.id  = seq.registerGradEvent(groSp);
    gro1.id   = seq.registerGradEvent(gro1);
    [~, rf.shapeIDs]           = seq.registerRfEvent(rf);
    [rf180.id, rf180.shapeIDs] = seq.registerRfEvent(rf180);
    lblIncPar.id = seq.registerLabelEvent(lblIncPar);

    % Build sequence
    % First inversion block (prep, ONCE=1)
    seq.addBlock(rf180, lblOnce1);
    seq.addBlock(TIdelay, gslSp);
    seq.addBlock(lblOnce0);  % end prep

    rf_phase = 0;
    rf_inc   = 0;

    for j = 1:Nz
        if j > 1
            seq.addBlock(rf180);
            seq.addBlock(TIdelay, gslSp);
        end

        gpe2je    = mr.scaleGrad(gpe2, pe2Steps(j));
        gpe2je.id = seq.registerGradEvent(gpe2je);
        gpe2jr    = mr.scaleGrad(gpe2, -pe2Steps(j));
        gpe2jr.id = seq.registerGradEvent(gpe2jr);

        for i = 1:Ny
            rf.phaseOffset  = rf_phase / 180 * pi;
            adc.phaseOffset = rf_phase / 180 * pi;
            rf_inc   = mod(rf_inc + rfSpoilInc, 360.0);
            rf_phase = mod(rf_phase + rf_inc, 360.0);

            if i == 1
                seq.addBlock(rf);
            else
                seq.addBlock(rf, groSp, mr.scaleGrad(gpe1, -pe1Steps(i-1)), gpe2jr, lblIncPar);
            end
            seq.addBlock(adc, gro1, ...
                mr.scaleGrad(gpe1, pe1Steps(i)), gpe2je);
        end
        seq.addBlock(groSp, mr.makeDelay(TRoutDelay), lblResetPar, lblIncLin);
    end

    seq.setDefinition('FOV', fov);
    seq.setDefinition('Name', 'mprage');
    seq.setDefinition('OrientationMapping', 'AX');

    % --- representative TRs for waveform ground truth ---
    tr_min = mr.Sequence(sys);  % zero PE (mode 2: definition-min)
    tr_max = mr.Sequence(sys);  % max |PE| (mode 1: position-max)

    tr_min.addBlock(rf);
    tr_min.addBlock(adc, gro1);

    tr_max.addBlock(rf, groSp, gpe1, gpe2);
    tr_max.addBlock(adc, gro1, mr.scaleGrad(gpe1, -1), mr.scaleGrad(gpe2, -1));

    % --- structural ground truth ---
    % Block defs (dedup key = duration, rf_def, gx_def, gy_def, gz_def;
    %             amplitude is scalar, NOT in key):
    %   0: rf180                         (inversion pulse)
    %   1: TIdelay + gslSp               (TI delay + z-axis slab spoiler)
    %   2: label-only                    (lblOnce0 in prep)
    %   3: rf                            (inner TR first block, rf only)
    %   4: rf + groSp + gpe1 + gpe2      (inner TR i>1, x-spoiler + y/z PE rewind)
    %   5: adc + gro1 + gpe1 + gpe2      (x-readout + y/z PE encode)
    %   6: groSp + TRoutDelay            (end-of-partition, x-spoiler + delay)
    gt.tr_min          = tr_min;
    gt.tr_max          = tr_max;
    gt.rf_center_s     = rf.center;
    gt.adc_num_samples = adc.numSamples;
    gt.adc_dwell_s     = adc.dwell;
    gt.seg_unique_ids  = {[0, 1, 3, 4, 5, 6]};  % outer TR (j>1 pattern)
    gt.unique_blocks   = 0:6;
    gt.num_prep_blocks = 3;   % rf180+lblOnce1, TIdelay+gslSp, lblOnce0
    gt.num_cool_blocks = 0;
    gt.degenerate_prep = 0;
    gt.degenerate_cool = 0;

    fname = seq_filename('mprage_3d', num_slices, num_averages);
    check_and_write(seq, fname, fov(1), fov(3), num_slices, num_averages, gt);
end

%% ========================================================================
%  Noncartesian MPRAGE (3D stack-of-stars inversion-recovery GRE)
%  ========================================================================

function write_mprage_noncart(num_slices, num_averages, num_shots, use_rotext)
    fprintf('Generating Noncartesian MPRAGE (%d shots, rotext=%d) ...\n', ...
            num_shots, use_rotext);

    sys = make_system();
    seq = mr.Sequence(sys);

    alpha      = 7;             % flip angle [deg]
    ro_dur     = 5040e-6;       % RO duration (multiple of 20us)
    ro_os      = 1;
    ro_spoil   = 3;
    TI         = 1.1;
    TRout      = 2.5;
    rfSpoilInc = 84;
    rfLen      = 100e-6;

    fov = [256, 240, 192] * 1e-3;   % [x, y, z]
    Nx  = 256;                       % readout samples (x)
    Nz  = 192;                       % partition (z) — outer loop

    % --- events ---
    rf180 = mr.makeBlockPulse(pi, sys, ...
        'Duration', 10e-3, 'use', 'excitation');
    rf = mr.makeBlockPulse(alpha * pi / 180, sys, ...
        'Duration', rfLen, 'use', 'excitation');

    deltak = 1 ./ fov;

    % Readout trapezoid template → arbitrary waveform for rotation
    groTrap = mr.makeTrapezoid('x', ...
        'Amplitude', Nx * deltak(1) / ro_dur, ...
        'FlatTime', ceil((ro_dur + sys.adcDeadTime) / sys.gradRasterTime) ...
                    * sys.gradRasterTime, 'system', sys);
    times    = cumsum([0, groTrap.riseTime, groTrap.flatTime, groTrap.fallTime]);
    amp      = [0, groTrap.amplitude, groTrap.amplitude, 0];
    waveform = mr.pts2waveform(times, amp, 'system', sys);
    groArbX  = mr.makeArbitraryGrad('x', waveform, 'system', sys);
    groArbY  = mr.makeArbitraryGrad('y', 0 * waveform, 'system', sys);

    % ADC
    adc = mr.makeAdc(Nx * ro_os, 'Duration', ro_dur, ...
                     'Delay', groTrap.riseTime, 'system', sys);

    % Readout spoiler (along x)
    groSp = mr.makeTrapezoid('x', ...
        'Area', deltak(1) / 2 * Nx * ro_spoil, 'system', sys);

    % Partition encoding (along z)
    gpe2 = mr.makeTrapezoid('z', ...
        'Area', -deltak(3) * Nz / 2, 'system', sys);

    % Slab spoiler (along z)
    gslSp = mr.makeTrapezoid('z', ...
        'Area', max(deltak .* [Nx 1 Nz]) * 4, 'Duration', 10e-3, 'system', sys);

    % RF delay to accommodate spoiler + partition rewind
    rf.delay = mr.calcDuration(groSp, gpe2);

    TRinner = mr.calcDuration(rf) + mr.calcDuration(groArbX);

    pe2Steps = ((0:Nz-1) - Nz/2) / Nz * 2;

    % TI delay — for radial, every spoke passes through k-center,
    % so TI targets the first excitation of each partition
    TIdelay = round((TI ...
              - (mr.calcDuration(rf180) - mr.calcRfCenter(rf180) - rf180.delay) ...
              - rf.delay - mr.calcRfCenter(rf)) / sys.blockDurationRaster) ...
              * sys.blockDurationRaster;
    TRoutDelay = TRout - TRinner * num_shots - TIdelay - mr.calcDuration(rf180);

    % Pre-create labels
    lblIncLin   = mr.makeLabel('INC', 'LIN', 1);
    lblIncPar   = mr.makeLabel('INC', 'PAR', 1);
    lblResetPar = mr.makeLabel('SET', 'PAR', 0);
    lblOnce1    = mr.makeLabel('SET', 'ONCE', 1);
    lblOnce0    = mr.makeLabel('SET', 'ONCE', 0);

    % Pre-register unchanging events
    gslSp.id  = seq.registerGradEvent(gslSp);
    groSp.id  = seq.registerGradEvent(groSp);
    [~, rf.shapeIDs]           = seq.registerRfEvent(rf);
    [rf180.id, rf180.shapeIDs] = seq.registerRfEvent(rf180);
    lblIncPar.id = seq.registerLabelEvent(lblIncPar);

    % Build sequence
    % First inversion block (prep, ONCE=1)
    seq.addBlock(rf180, lblOnce1);
    seq.addBlock(TIdelay, gslSp);
    seq.addBlock(lblOnce0);  % end prep

    rf_phase = 0;
    rf_inc   = 0;
    phi      = 0;
    dphi     = 137.51 * pi / 180;  % golden angle [rad]

    for j = 1:Nz
        if j > 1
            seq.addBlock(rf180);
            seq.addBlock(TIdelay, gslSp);
        end

        gpe2je    = mr.scaleGrad(gpe2, pe2Steps(j));
        gpe2je.id = seq.registerGradEvent(gpe2je);
        gpe2jr    = mr.scaleGrad(gpe2, -pe2Steps(j));
        gpe2jr.id = seq.registerGradEvent(gpe2jr);

        for i = 1:num_shots
            rf.phaseOffset  = rf_phase / 180 * pi;
            adc.phaseOffset = rf_phase / 180 * pi;
            rf_inc   = mod(rf_inc + rfSpoilInc, 360.0);
            rf_phase = mod(rf_phase + rf_inc, 360.0);

            % RF block: first shot = rf only; subsequent = rf + spoiler + PE2 rewind
            if i == 1
                seq.addBlock(rf);
            else
                seq.addBlock(rf, groSp, gpe2jr, lblIncPar);
            end

            % Readout block: rotated arbitrary gradients + PE2 encode + ADC
            if use_rotext
                seq.addBlock(adc, groArbX, gpe2je, ...
                    mr.makeRotation('axis', 'z', 'angle', phi));
            else
                seq.addBlock(adc, ...
                    mr.rotate('z', phi, groArbX, groArbY), gpe2je);
            end

            phi = phi + dphi;
        end
        seq.addBlock(groSp, mr.makeDelay(TRoutDelay), lblResetPar, lblIncLin);
    end

    seq.setDefinition('FOV', fov);
    seq.setDefinition('Name', 'mprage_noncart');
    seq.setDefinition('OrientationMapping', 'AX');

    % --- representative TRs for waveform ground truth ---
    tr_min = mr.Sequence(sys);  % zero PE2 (mode 2: definition-min)
    tr_max = mr.Sequence(sys);  % max |PE2| (mode 1: position-max)

    % tr_min: first shot, no spoiler, no PE2
    tr_min.addBlock(rf);
    tr_min.addBlock(adc, groArbX, groArbY);

    % tr_max: subsequent shot, with spoiler + full PE2
    tr_max.addBlock(rf, groSp, gpe2);
    tr_max.addBlock(adc, groArbX, groArbY, mr.scaleGrad(gpe2, -1));

    % --- structural ground truth ---
    % Block defs (dedup key = duration, rf_def, gx_def, gy_def, gz_def;
    %             amplitude is scalar, NOT in key):
    %   0: rf180                         (inversion pulse)
    %   1: TIdelay + gslSp               (TI delay + z-axis slab spoiler)
    %   2: label-only                    (lblOnce0 in prep)
    %   3: rf                            (inner TR first shot, rf only)
    %   4: rf + groSp + gpe2             (inner TR i>1, x-spoiler + z PE2 rewind)
    %   5: adc + groArb + gpe2           (x/y rotated readout + z PE2 encode;
    %                                     waveform shape varies with angle)
    %   6: groSp + TRoutDelay            (end-of-partition, x-spoiler + delay)
    gt.tr_min          = tr_min;
    gt.tr_max          = tr_max;
    gt.rf_center_s     = rf.center;
    gt.adc_num_samples = adc.numSamples;
    gt.adc_dwell_s     = adc.dwell;
    gt.seg_unique_ids  = {[0, 1, 3, 4, 5, 6]};  % outer TR (j>1 pattern)
    gt.unique_blocks   = 0:6;
    gt.num_prep_blocks = 3;   % rf180+lblOnce1, TIdelay+gslSp, lblOnce0
    gt.num_cool_blocks = 0;
    gt.degenerate_prep = 0;
    gt.degenerate_cool = 0;

    if use_rotext
        rotext_tag = '_rotext';
    else
        rotext_tag = '';
    end
    fname = seq_filename(sprintf('mprage_noncart_3d_%dshots%s', num_shots, rotext_tag), num_slices, num_averages);
    check_and_write(seq, fname, fov(1), fov(3), num_slices, num_averages, gt);
end