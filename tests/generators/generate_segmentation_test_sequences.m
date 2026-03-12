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

write_bssfp(1, 1);
write_bssfp(1, 3);
write_bssfp(3, 1);
write_bssfp(3, 3);

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

write_mprage(1);
write_mprage(3);

write_mprage_noncart(1, 240, false);
write_mprage_noncart(3, 240, false);

write_mprage_noncart(1, 240, true);
write_mprage_noncart(3, 240, true);

write_mprage_noncart(1, 2048, true);

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
% Output is written to ../data/ relative to this script.
%
% gt (optional struct) contains structural ground truth:
%   .tr_min          - mr.Sequence: representative TR with zero-var amplitudes
%                      (matches C library amplitude mode 1 = ZERO_VAR)
%   .tr_max          - mr.Sequence: representative TR with max |PE|
%                      (matches C library amplitude mode 0 = max positional)
%   .rf_center_s     - RF isocenter offset within the shape (seconds)
%   .adc_num_samples - number of ADC samples per readout
%   .adc_dwell_s     - ADC dwell time (seconds)
%   .rf_refocus_center_s - RF refocusing isocenter offset (seconds)
%   .seg_unique_ids  - cell array of int arrays, each cell is
%                      unique block def IDs for one segment
%   .unique_blocks   - int array of unique block def IDs for the
%                      full TR (all segments concatenated)
%   .num_prep_blocks - number of preparation blocks (ONCE=1 region)
%                      0 when degenerate (absorbed into main)
%   .num_cool_blocks - number of cooldown blocks (ONCE=2 region)
%                      0 when degenerate (absorbed into main)
%   .degenerate_prep - 1 if prep is absent or pattern == main, 0 otherwise
%   .degenerate_cool - 1 if cooldown is absent or pattern == main, 0 otherwise
%   .num_once1_blocks - (optional) total blocks with ONCE=1 flag; used by
%                       scan table export to exclude from replication even
%                       when degenerate.  Defaults to num_prep_blocks.
%   .num_once2_blocks - (optional) total blocks with ONCE=2 flag; defaults
%                       to num_cool_blocks.

    if nargin < 7, gt = struct(); end

    % Write to data directory
    dataDir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
    if ~exist(dataDir, 'dir'), mkdir(dataDir); end
    fname = fullfile(dataDir, fname);

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
    [fdir, base, ~] = fileparts(seq_fname);
    base = fullfile(fdir, base);  % preserve output directory in base path

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
    if isfield(gt, 'tr_size')
        fprintf(fid, 'tr_size %d\n', gt.tr_size);
    end
    if isfield(gt, 'seg_unique_ids')
        fprintf(fid, 'num_segments %d\n', length(gt.seg_unique_ids));
    end
    if isfield(gt, 'num_passes')
        fprintf(fid, 'num_passes %d\n', gt.num_passes);
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

    % --- TR gradient waveforms (ZERO_VAR amplitude, mode 1) ---
    if isfield(gt, 'tr_min') && ~isempty(gt.tr_min)
        export_tr_waveforms(gt.tr_min, base, '_min', gt);
    elseif isfield(gt, 'tr_min_range') && ~isempty(gt.tr_min_range)
        export_tr_waveforms(gt.tr_min_range{1}, base, '_min', gt, gt.tr_min_range{2});
    end

    % --- TR gradient waveforms (max positional amplitude, mode 1) ---
    if isfield(gt, 'tr_max') && ~isempty(gt.tr_max)
        export_tr_waveforms(gt.tr_max, base, '_max', gt);
    elseif isfield(gt, 'tr_max_range') && ~isempty(gt.tr_max_range)
        export_tr_waveforms(gt.tr_max_range{1}, base, '_max', gt, gt.tr_max_range{2});
    end

    % --- prep TR waveforms (actual amplitude) ---
    if isfield(gt, 'tr_prep') && ~isempty(gt.tr_prep)
        export_tr_waveforms(gt.tr_prep, base, '_prep', gt);
    elseif isfield(gt, 'tr_prep_range') && ~isempty(gt.tr_prep_range)
        export_tr_waveforms(gt.tr_prep_range{1}, base, '_prep', gt, gt.tr_prep_range{2});
    end

    % --- cooldown TR waveforms (actual amplitude) ---
    if isfield(gt, 'tr_cool') && ~isempty(gt.tr_cool)
        export_tr_waveforms(gt.tr_cool, base, '_cool', gt);
    elseif isfield(gt, 'tr_cool_range') && ~isempty(gt.tr_cool_range)
        export_tr_waveforms(gt.tr_cool_range{1}, base, '_cool', gt, gt.tr_cool_range{2});
    end
end

function export_tr_waveforms(pass_seq, base, mode_suffix, gt, blockRange)
% Export per-axis gradient waveforms from a canonical pass sequence.
%
% Approach: evaluate waveforms_and_times on the full canonical pass, then
% resample onto a uniform raster (0.5 * gradRasterTime) and extract
% the desired portion by simple time indexing.  The half-raster spacing
% ensures correct capture of arbitrary-gradient samples (center-of-raster)
% consistent with the C library's interpolation convention.
%
% Args:
%   pass_seq    - mr.Sequence with the full canonical pass (prep + TR + cool)
%                 or just one TR if no prep/cool.
%   base        - output filename base (no extension)
%   mode_suffix - '_min' or '_max' appended to filenames
%   gt          - ground truth struct with rf_center_s, adc_num_samples,
%                 adc_dwell_s for proper anchor computation
%   blockRange  - (optional) [start, end] 1-based block indices of the
%                 main TR within pass_seq.  If omitted, uses the entire
%                 sequence.  The waveform is ALWAYS evaluated on the full
%                 pass; blockRange only selects which time portion to export.
%
% Output files:
%   _tr<mode>_gx.csv, _tr<mode>_gy.csv, _tr<mode>_gz.csv
%   _tr<mode>_anchors.txt  (RF isocenter + ADC k=0 center times)

    % Always evaluate the full pass (no blockRange to waveforms_and_times)
    [wave, tfp_exc, tfp_ref, t_adc] = pass_seq.waveforms_and_times(true);

    % Uniform raster: 0.5 * gradRasterTime (matches C library convention;
    % arbitrary grads are sampled at center of raster intervals)
    raster = pass_seq.sys.gradRasterTime / 2;  % seconds
    t_end = sum(pass_seq.blockDurations);
    times = 0 : raster : t_end;
    if isempty(times), times = 0; end

    % Interpolate each gradient channel (1=gx, 2=gy, 3=gz) onto the raster
    samples = cell(1, 3);
    for c = 1:3
        if c <= length(wave) && ~isempty(wave{c})
            samples{c} = interp1(wave{c}(1,:), wave{c}(2,:), times, 'linear', 0);
        else
            samples{c} = zeros(size(times));
        end
    end

    % --- Determine TR time range ---
    if nargin >= 5 && ~isempty(blockRange)
        blk_start = blockRange(1);
        blk_end   = blockRange(2);
        durations = pass_seq.blockDurations;
        t_tr_start = sum(durations(1:blk_start-1));
        t_tr_end   = sum(durations(1:blk_end));
    else
        t_tr_start = 0;
        t_tr_end   = t_end;
    end

    % Extract TR portion (with small tolerance for floating point)
    idx = (times >= t_tr_start - 1e-9) & (times <= t_tr_end + 1e-9);
    t_tr = times(idx) - t_tr_start;  % shift to start at 0
    for c = 1:3
        samples{c} = samples{c}(idx);
    end

    % Also shift anchor times to be relative to TR start
    tfp_exc_shifted = tfp_exc;
    tfp_ref_shifted = tfp_ref;
    t_adc_shifted   = t_adc;
    if ~isempty(tfp_exc_shifted)
        tfp_exc_shifted(:,1) = tfp_exc_shifted(:,1) - t_tr_start;
        % Keep only anchors within the TR
        keep = tfp_exc_shifted(:,1) >= -1e-9 & tfp_exc_shifted(:,1) <= (t_tr_end - t_tr_start) + 1e-9;
        tfp_exc_shifted = tfp_exc_shifted(keep, :);
    end
    if ~isempty(tfp_ref_shifted)
        tfp_ref_shifted(:,1) = tfp_ref_shifted(:,1) - t_tr_start;
        keep = tfp_ref_shifted(:,1) >= -1e-9 & tfp_ref_shifted(:,1) <= (t_tr_end - t_tr_start) + 1e-9;
        tfp_ref_shifted = tfp_ref_shifted(keep, :);
    end
    if ~isempty(t_adc_shifted)
        t_adc_shifted = t_adc_shifted - t_tr_start;
        keep = t_adc_shifted >= -1e-9 & t_adc_shifted <= (t_tr_end - t_tr_start) + 1e-9;
        t_adc_shifted = t_adc_shifted(keep);
    end

    % --- Write gradient CSVs ---
    axis_labels = {'gx', 'gy', 'gz'};
    for c = 1:3
        fname = sprintf('%s_tr%s_%s.csv', base, mode_suffix, axis_labels{c});
        fid = fopen(fname, 'w');
        fprintf(fid, 'time_us,amplitude_hz_per_m\n');

        t = t_tr * 1e6;    % seconds -> us
        a = samples{c};    % Hz/m (Pulseq native)
        for k = 1:length(t)
            fprintf(fid, '%.6f,%.8g\n', t(k), a(k));
        end
        fclose(fid);
    end

    % --- RF/ADC timing anchors ---
    fid = fopen(sprintf('%s_tr%s_anchors.txt', base, mode_suffix), 'w');

    % RF isocenter = tfp_excitation time + rf.center
    rf_center = 0;
    if isfield(gt, 'rf_center_s'), rf_center = gt.rf_center_s; end

    if ~isempty(tfp_exc_shifted)
        for k = 1:size(tfp_exc_shifted, 1)
            isocenter_us = (tfp_exc_shifted(k, 1) + rf_center) * 1e6;
            fprintf(fid, 'rf_isocenter_us %.6f\n', isocenter_us);
        end
    end

    if ~isempty(tfp_ref_shifted)
        rf_center_ref = rf_center;
        if isfield(gt, 'rf_refocus_center_s')
            rf_center_ref = gt.rf_refocus_center_s;
        end
        for k = 1:size(tfp_ref_shifted, 1)
            isocenter_us = (tfp_ref_shifted(k, 1) + rf_center_ref) * 1e6;
            fprintf(fid, 'rf_refocus_isocenter_us %.6f\n', isocenter_us);
        end
    end

    % ADC k-space center = first_sample_time + ceil(N/2) * dwell
    if ~isempty(t_adc_shifted) && isfield(gt, 'adc_num_samples') && isfield(gt, 'adc_dwell_s')
        N_adc = gt.adc_num_samples;
        dwell = gt.adc_dwell_s;
        num_events = floor(length(t_adc_shifted) / N_adc);
        for ev = 1:num_events
            first_idx = (ev - 1) * N_adc + 1;
            kzero_us = (t_adc_shifted(first_idx) + ceil(N_adc / 2) * dwell) * 1e6;
            fprintf(fid, 'adc_kzero_us %.6f\n', kzero_us);
        end
    end

    fclose(fid);
end

function export_scan_table(base, N, num_averages, gt)
% Build and export the expected scan table for the given number of averages.
% The scan table maps scan positions to 0-based block indices, accounting
% for prep (once per pass), main (repeated num_averages times per pass),
% cooldown (once per pass), and multiple passes.
%
% Two modes:
%   Non-degenerate: num_prep/num_cool define contiguous prefix/suffix
%     prep_idx  = base_offset .. base_offset + num_prep - 1
%     main_idx  = base_offset + num_prep .. base_offset + pass_len - num_cool - 1
%     cool_idx  = base_offset + pass_len - num_cool .. base_offset + pass_len - 1
%
%   Degenerate: the ONCE-flagged blocks sit inside the main region,
%     but still play once only (first/last repetition).  The number of
%     such blocks is given by num_once1_blocks / num_once2_blocks.
%
% Per-pass scan order: prep on first avg, cooldown on last avg, main every avg.
%
% Output: _scan_table.csv with columns (scan_pos, block_idx)

    num_prep = gt.num_prep_blocks;
    num_cool = gt.num_cool_blocks;
    num_passes = 1;
    if isfield(gt, 'num_passes'), num_passes = gt.num_passes; end

    % For degenerate prep/cool the metadata num_prep/num_cool is 0, but
    % there are still ONCE-flagged blocks that play only once.  Use the
    % optional num_once1/once2_blocks fields when provided.
    num_once1 = num_prep;
    num_once2 = num_cool;
    if isfield(gt, 'num_once1_blocks'), num_once1 = gt.num_once1_blocks; end
    if isfield(gt, 'num_once2_blocks'), num_once2 = gt.num_once2_blocks; end

    pass_len = N / num_passes;
    num_main = pass_len - num_once1 - num_once2;

    block_idx = [];
    for pass = 0:(num_passes - 1)
        base_offset = pass * pass_len;
        prep_idx = base_offset:(base_offset + num_once1 - 1);
        main_idx = (base_offset + num_once1):(base_offset + num_once1 + num_main - 1);
        cool_idx = (base_offset + pass_len - num_once2):(base_offset + pass_len - 1);

        for avg = 0:(num_averages - 1)
            if avg == 0,              block_idx = [block_idx, prep_idx]; end %#ok<AGROW>
            block_idx = [block_idx, main_idx]; %#ok<AGROW>
            if avg == num_averages-1, block_idx = [block_idx, cool_idx]; end %#ok<AGROW>
        end
    end

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

function write_bssfp(num_averages, num_slices)
    if nargin < 2, num_slices = 1; end
    fprintf('Generating bSSFP (%d slice, %d avg) ...\n', num_slices, num_averages);

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
    gyMax     = mr.makeTrapezoid('y', 'Area', maxPeArea, 'Duration', pe_dur, 'system', sys);

    % --- pre-create labels ---
    lblOnce1 = mr.makeLabel('SET', 'ONCE', 1);
    lblOnce0 = mr.makeLabel('SET', 'ONCE', 0);
    lblOnce2 = mr.makeLabel('SET', 'ONCE', 2);

    % --- prep RF: half flip angle ---
    rf05        = rf;
    rf05.signal = 0.5 * rf.signal;

    % --- prep delay to center main acquisition around TR/2 ---
    prepDelay = mr.makeDelay( round((TR/2 - mr.calcDuration(gz_1)) / sys.gradRasterTime) * sys.gradRasterTime);
    gx_1_1    = mr.makeExtendedTrapezoidArea('x', 0, gx_2.first, -gx_2.area, sys);
    gyPre_2   = mr.scaleGrad(gyMax, phaseAreas(end) / maxPeArea);
    [prepDelay, gz_2, gyPre_2, gx_1_1] = mr.align('left', prepDelay, gz_2, gyPre_2, 'right', gx_1_1);

    for z = 1:num_slices
        rf05.freqOffset = gz.amplitude * thick * (z - 1 - (num_slices-1)/2);
        rf.freqOffset = gz.amplitude * thick * (z - 1 - (num_slices-1)/2);
        
        % --- alpha/2 prep (ONCE=1) ---
        seq.addBlock(rf05, gz_1, lblOnce1);
        seq.addBlock(prepDelay, gz_2, gyPre_2, gx_1_1);

        % --- main loop ---
        for i = 1:Ny
            rf.phaseOffset  = pi * mod(i, 2);
            adc.phaseOffset = pi * mod(i, 2);

            gyPre_1 = mr.scaleGrad(gyPre_2, -1);             % undo previous PE
            gyPre_2 = mr.scaleGrad(gyMax, phaseAreas(i) / maxPeArea);  % new PE

            if i == 1
                seq.addBlock(rf, gz_1, gyPre_1, gx_2, lblOnce0); % clear ONCE flag -> first main block
            else
                seq.addBlock(rf, gz_1, gyPre_1, gx_2);
            end
            seq.addBlock(gx_1, gyPre_2, gz_2, adc);
        end

        % --- exit block (ONCE=2) ---
        seq.addBlock(gx_2, lblOnce2);
    end

    % sanity: prep = 2 blocks (rf05+gz_1, align block).
    % Block 3 has lblOnce0 which sets once_flag=0 → first main block.
    % Main TR = blocks 4+5 (blocks of type [rf+gz_1+gy+gx_2] and
    %           [gx_1+gy+gz_2+adc]), verified below.
    assert(abs(TR - (mr.calcDuration(seq.getBlock(4)) + mr.calcDuration(seq.getBlock(5)))) < 1e-12);

    fprintf('  TR = %.3f ms   TE = %.3f ms\n', TR * 1e3, TE * 1e3);

    % --- structural ground truth ---
    % bSSFP uses split/merged gradients that connect across block
    % boundaries.  A standalone Sequence for just the main TR would fail
    % the MATLAB boundary check.  Instead, build full canonical pass
    % sequences (prep + 1 TR + cooldown) and extract the TR portion via
    % time indexing in export_tr_waveforms.
    %
    % For MAX_POS (mode 0): GY amplitude = max|PE| with sign from the
    %   first TR instance.  Other axes use actual amplitudes.
    % For ZERO_VAR (mode 1): GY = 0 (variable across TRs); GX/GZ at
    %   actual amplitudes (constant across TRs).

    % First TR instance PE signs (i=1 in main loop):
    %   pos 0 (block 3): undo previous PE = -phaseAreas(end)
    %   pos 1 (block 4): new PE = phaseAreas(1)
    pe_undo_sign = sign(-phaseAreas(end));
    if pe_undo_sign == 0, pe_undo_sign = 1; end
    pe_enc_sign  = sign(phaseAreas(1));
    if pe_enc_sign  == 0, pe_enc_sign  = 1; end

    % MAX_POS canonical pass: prep + 1 TR (full-scale GY) + cool
    pass_max = mr.Sequence(sys);
    pass_max.addBlock(rf05, gz_1);                                          % prep 1
    pass_max.addBlock(prepDelay, gz_2, ...                                  % prep 2
        mr.scaleGrad(gyMax, phaseAreas(end) / maxPeArea), gx_1_1);
    pass_max.addBlock(rf, gz_1, mr.scaleGrad(gyMax, pe_undo_sign), gx_2);  % TR pos 0
    pass_max.addBlock(gx_1, mr.scaleGrad(gyMax, pe_enc_sign), gz_2, adc);  % TR pos 1
    pass_max.addBlock(gx_2);                                                % cool

    % ZERO_VAR canonical pass: GY = 0 (variable across TRs); other axes actual
    pass_min = mr.Sequence(sys);
    pass_min.addBlock(rf05, gz_1);                                          % prep 1
    pass_min.addBlock(prepDelay, gz_2, ...                                  % prep 2
        mr.scaleGrad(gyMax, 0), gx_1_1);
    pass_min.addBlock(rf, gz_1, mr.scaleGrad(gyMax, 0), gx_2);             % TR pos 0
    pass_min.addBlock(gx_1, mr.scaleGrad(gyMax, 0), gz_2, adc);            % TR pos 1
    pass_min.addBlock(gx_2);                                                % cool

    % PREP canonical pass: actual amplitudes for first TR instance (i=1)
    pass_prep = mr.Sequence(sys);
    pass_prep.addBlock(rf05, gz_1);                                         % prep 1
    pass_prep.addBlock(prepDelay, gz_2, ...                                 % prep 2
        mr.scaleGrad(gyMax, phaseAreas(end) / maxPeArea), gx_1_1);
    pass_prep.addBlock(rf, gz_1, ...                                        % TR pos 0: undo prep PE
        mr.scaleGrad(gyMax, -phaseAreas(end) / maxPeArea), gx_2);
    pass_prep.addBlock(gx_1, ...                                            % TR pos 1: encode PE(1)
        mr.scaleGrad(gyMax, phaseAreas(1) / maxPeArea), gz_2, adc);
    pass_prep.addBlock(gx_2);                                               % cool

    % COOL canonical pass: actual amplitudes for last TR instance (i=Ny)
    pass_cool = mr.Sequence(sys);
    pass_cool.addBlock(rf05, gz_1);                                         % prep 1
    pass_cool.addBlock(prepDelay, gz_2, ...                                 % prep 2
        mr.scaleGrad(gyMax, phaseAreas(end) / maxPeArea), gx_1_1);
    pass_cool.addBlock(rf, gz_1, ...                                        % TR pos 0: undo PE(Ny-1)
        mr.scaleGrad(gyMax, -phaseAreas(Ny-1) / maxPeArea), gx_2);
    pass_cool.addBlock(gx_1, ...                                            % TR pos 1: encode PE(Ny)
        mr.scaleGrad(gyMax, phaseAreas(Ny) / maxPeArea), gz_2, adc);
    pass_cool.addBlock(gx_2);                                               % cool

    gt.tr_min          = [];
    gt.tr_max          = [];
    gt.tr_min_range    = {pass_min, [3, 4]};   % TR = blocks 3-4 of pass
    gt.tr_max_range    = {pass_max, [3, 4]};
    gt.tr_prep_range   = {pass_prep, [1, 4]};  % prep = blocks 1-4 (actual amps for i=1)
    gt.tr_cool_range   = {pass_cool, [3, 5]};  % cool = blocks 3-5 (actual amps for i=Ny)
    gt.rf_center_s     = rf.center;
    gt.adc_num_samples = adc.numSamples;
    gt.adc_dwell_s     = adc.dwell;
    gt.seg_unique_ids  = {[0, 1, repmat([2, 3], 1, Ny), 4]};
    gt.unique_blocks   = 0:4;
    gt.tr_size         = 2;           % 2 blocks per TR (rf+gz+gy+gx, gx+gy+gz+adc)
    gt.num_prep_blocks = 2;          % alpha/2 + align (lblOnce0 block is first main)
    gt.num_cool_blocks = 1;          % exit gx_2 block
    gt.degenerate_prep = 0;          % alpha/2 prep ~= main pattern
    gt.degenerate_cool = 0;          % exit block ~= main pattern
    if num_slices > 1
        gt.num_passes = num_slices;  % C library folds identical per-slice patterns
    end

    fname = sprintf('bssfp_2d_%dsl_%davg.seq', num_slices, num_averages);
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
    delayTR = 0.1e-3 + ceil((mr.calcDuration(gz) + mr.calcDuration(gxPre) ...
              + mr.calcDuration(gx) + delayTE) ...
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

    rf_phase = 0;
    rf_inc   = 0;

    % --- ground truth: segment defs as unique block IDs ---
    seg_unique_ids = {[0, 1, 2, 3, 4]};  % single segment = full TR

    % --- representative TRs for waveform ground truth ---
    [~, iPEmin] = min(abs(phaseAreas));
    pe_min_scale = phaseAreas(iPEmin) / maxPeArea;

    tr_min = mr.Sequence(sys);   % ZERO_VAR: GY=0 (variable), GX/GZ actual (constant)
    tr_max = mr.Sequence(sys);   % max positional amplitude (mode 0)

    % --- build representative TR: ZERO_VAR (mode 1) ---
    tr_min.addBlock(rf, gz);
    tr_min.addBlock(gxPre, mr.scaleGrad(gyMax, 0.0), gzReph);
    tr_min.addBlock(evDelayTE);
    tr_min.addBlock(gx, adc);
    tr_min.addBlock(gxSpoil, mr.scaleGrad(gyMax, 0.0), gzSpoil, evDelayTR);

    % --- build representative TR: max positional amplitude (mode 0) ---
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
    gt.tr_size         = 5;             % 5 blocks per TR (rf+gz, prewinder, delayTE, readout, spoiler)
    gt.num_prep_blocks = 0;            % degenerate: absorbed into main
    gt.num_cool_blocks = 0;
    gt.degenerate_prep = 1;            % dummy TR pattern == imaging TR pattern
    gt.degenerate_cool = 1;            % no cooldown blocks (absent = degenerate)
    gt.num_once1_blocks = Ndummy * 5;  % 5 blocks per dummy TR, once-flagged for scan table

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
    TRfill = 0.1e-3;  % override for short TR test (no fill, but still valid since echoes fit within TR)
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
    %       [same def as GS7 + GR7 + GPrew: timing-only dedup
    %        merges the time-reversed pre/post spoiler blocks]
    %   5: GR6                           (readout flat, +/- ADC)
    %   6: GS4                           (end crusher, no RF)
    %   7: GS5                           (end spoiler)
    %   8: delayTR                       (TR fill delay)
    echo_pattern = repmat([3, 4, 5, 4], 1, necho);
    seg0_ids = [0, 1, 2, echo_pattern, 6, 7];  % echo train segment
    seg1_ids = 8;                                % delay segment
    seg_unique_ids = {seg0_ids, seg1_ids};
    unique_blocks  = 0:8;

    % --- representative TRs for waveform ground truth ---
    % ZERO_VAR: GY = 0 (variable across excitations); GX/GZ at actual
    [~, iPEmin] = min(abs(phaseAreas(:)));
    [kech_min, kex_min] = ind2sub(size(phaseAreas), iPEmin);
    pe_min_scale = phaseAreas(kech_min, kex_min) / maxPeArea;

    tr_min = mr.Sequence(sys);  % ZERO_VAR: GY=0 (variable), GX/GZ actual (constant)
    tr_max = mr.Sequence(sys);  % max positional amplitude (mode 0)

    % Build tr_min (ZERO_VAR: all GY = 0)
    tr_min.addBlock(GS1);
    tr_min.addBlock(GS2, rfex);
    tr_min.addBlock(GS3, GR3);
    for kech = 1:necho
        tr_min.addBlock(GS4, rfref);
        tr_min.addBlock(GS5, GR5, mr.scaleGrad(gyMax, 0.0));
        tr_min.addBlock(GR6, adc);
        tr_min.addBlock(GS7, GR7, mr.scaleGrad(gyMax, 0.0));
    end
    tr_min.addBlock(GS4);
    tr_min.addBlock(GS5);
    tr_min.addBlock(delayTR);

    % Build tr_max (max |PE|)
    % For each echo position the C library computes
    %   sign(amp_first_imaging_TR) * max(abs(amp)) across all imaging TRs.
    % The first imaging TR uses phaseAreas(:, 1).
    tr_max.addBlock(GS1);
    tr_max.addBlock(GS2, rfex);
    tr_max.addBlock(GS3, GR3);
    for kech = 1:necho
        maxAbsArea = max(abs(phaseAreas(kech, :)));
        if maxPeArea > 0 && maxAbsArea > 0
            pe_sign = sign(phaseAreas(kech, 1));
            if pe_sign == 0; pe_sign = 1; end
            pe_scale = pe_sign * maxAbsArea / maxPeArea;
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
    gt.tr_size             = blocks_per_tr;  % 3 + 4*necho + 3 blocks per TR
    gt.num_prep_blocks     = 0;  % degenerate: absorbed into main
    gt.num_cool_blocks     = 0;
    gt.degenerate_prep     = 1;  % dummy uses same block defs (ADC not in dedup key)
    gt.degenerate_cool     = 1;  % no cooldown blocks (absent = degenerate)
    gt.num_once1_blocks    = Ndummy * blocks_per_tr * Nslices;

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
    TRdelay = 0.1e-3;
    TRdelay_perSlice = round(TRdelay / Nslices / sys.blockDurationRaster) * sys.blockDurationRaster;
    assert(TRdelay_perSlice > 0, 'TR too short for EPI');

    ROpolarity = sign(gx.amplitude);

    % Labels
    lblOnce1  = mr.makeLabel('SET', 'ONCE', 1);
    lblOnce0  = mr.makeLabel('SET', 'ONCE', 0);
    lblSetSlc = mr.makeLabel('SET', 'SLC', 0);
    lblIncSlc = mr.makeLabel('INC', 'SLC', 1);

    % Hardcoded volume number
    NDummyVolumes = 1; % in fMRI, dummy scans to reach steady state
    NVolumes = 1; % number of volumes to sample hemodynamics

    % --- prep (ONCE=1): one dummy volume ---
    % Structure mirrors main section (same block defs & ordering) so the C
    % library detects degenerate prep.  ADC is omitted because it does not
    % contribute to the block definition key; dummy data is not acquired.
    for v = 1:NDummyVolumes
        for s = 1:Nslices
            if v == 1 && s == 1
                seq.addBlock(rf_fs, gz_fs, lblOnce1, lblSetSlc);
            elseif s == 1
                seq.addBlock(rf_fs, gz_fs, lblSetSlc);
            else
                seq.addBlock(rf_fs, gz_fs);
            end
            rf.freqOffset  = gz.amplitude * slicePositions(s);
            seq.addBlock(rf, gz, trig);

            if Nnav > 0
                gxPre_nav = mr.scaleGrad(gxPre, -1);
                gx_nav    = mr.scaleGrad(gx, -1);
                seq.addBlock(gxPre_nav, gzReph, ...
                    mr.makeLabel('SET', 'NAV', 1), ...
                    mr.makeLabel('SET', 'LIN', floor(Ny/2)));

                for n = 1:Nnav
                    seq.addBlock( ...
                        gx_nav, ...
                        mr.makeLabel('SET', 'REV', sign(gx_nav.amplitude) ~= ROpolarity), ...
                        mr.makeLabel('SET', 'SEG', sign(gx_nav.amplitude) ~= ROpolarity), ...
                        mr.makeLabel('SET', 'AVG', n == Nnav));
                    gx_nav = mr.scaleGrad(gx_nav, -1);
                end

                seq.addBlock(mr.scaleGrad(gyPre, 0.0), ...
                    mr.makeLabel('SET', 'LIN', -1), ...
                    mr.makeLabel('SET', 'NAV', 0), ...
                    mr.makeLabel('SET', 'AVG', 0));
            else
                seq.addBlock(gxPre, mr.scaleGrad(gyPre, 0.0), gzReph, ...
                    mr.makeLabel('SET', 'LIN', -1), ...
                    mr.makeLabel('SET', 'NAV', 0), ...
                    mr.makeLabel('SET', 'AVG', 0));
            end

            for i = 1:Ny_meas
                lrev = mr.makeLabel('SET', 'REV', sign(gx.amplitude) ~= ROpolarity);
                lseg = mr.makeLabel('SET', 'SEG', sign(gx.amplitude) ~= ROpolarity);
                llin = mr.makeLabel('INC', 'LIN', 1);

                if i == 1
                    seq.addBlock(gx, gy_blipup, lrev, lseg, llin);
                elseif i == Ny_meas
                    seq.addBlock(gx, gy_blipdown, lrev, lseg, llin);
                else
                    seq.addBlock(gx, gy_blipdownup, lrev, lseg, llin);
                end
                gx = mr.scaleGrad(gx, -1);
            end

            if sign(gx.amplitude) ~= ROpolarity
                gx = mr.scaleGrad(gx, -1);
            end
            seq.addBlock(lblIncSlc, TRdelay_perSlice);
        end
    end

    % --- main: imaging volume ---
    for v = 1:NVolumes
        for s = 1:Nslices
            if v == 1 && s == 1
                seq.addBlock(rf_fs, gz_fs, lblOnce0, lblSetSlc);
            elseif s == 1
                seq.addBlock(rf_fs, gz_fs, lblSetSlc);
            else
                seq.addBlock(rf_fs, gz_fs);
            end
            rf.freqOffset  = gz.amplitude * slicePositions(s);
            rf.phaseOffset = -2*pi * rf.freqOffset * rf.center;
            seq.addBlock(rf, gz, trig);

            if Nnav > 0
                gxPre_nav = mr.scaleGrad(gxPre, -1);
                gx_tmp    = mr.scaleGrad(gx, -1);
                seq.addBlock(gxPre_nav, gzReph, ...
                    mr.makeLabel('SET', 'NAV', 1), ...
                    mr.makeLabel('SET', 'LIN', floor(Ny/2)));

                for n = 1:Nnav
                    seq.addBlock( ...
                        gx_tmp, adc, ...
                        mr.makeLabel('SET', 'REV', sign(gx_tmp.amplitude) ~= ROpolarity), ...
                        mr.makeLabel('SET', 'SEG', sign(gx_tmp.amplitude) ~= ROpolarity), ...
                        mr.makeLabel('SET', 'AVG', n == Nnav));
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

            if sign(gx.amplitude) ~= ROpolarity
                gx = mr.scaleGrad(gx, -1);
            end
            seq.addBlock(lblIncSlc, TRdelay_perSlice);
        end
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

    tr_min = mr.Sequence(sys);  % ZERO_VAR: all grads constant → same as MAX_POS
    tr_max = mr.Sequence(sys);  % max positional amplitude (mode 0)

    % For EPI the readout gradient alternates polarity each line, but the
    % pattern is identical in every TR instance (dummy and imaging use the
    % same gradient structure).  All gradients are constant across TRs,
    % so ZERO_VAR mode produces the same waveforms as MAX_POS.

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

    % PE prephasing (constant across TRs → kept at actual amplitude)
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
    %   0: rf_fs + gz_fs            (fat-sat)
    %   1: rf + gz + trig           (excitation + slice-select)
    %   2: gxPre + gzReph           (nav/readout prephasing)
    %   3: gx                       (nav readout — ADC not in def key)
    %   4: gyPre                    (PE prephasing)
    %   5: gx + gy_blipup           (first readout line)
    %   6: gx + gy_blipdownup       (middle readout lines)
    %   7: gx + gy_blipdown         (last readout line)
    %   8: TRdelay                  (per-slice delay)

    % Full per-slice segment pattern (expanded):
    nav_pattern = 3 * ones(1, Nnav);  % nav readout × Nnav
    readout_pattern = [5, repmat(6, 1, Ny_meas - 2), 7];
    seg0_ids = 0;  % Fat saturation
    seg1_ids = [1, 2, nav_pattern, 4, readout_pattern];  % main EPI readout
    seg2_ids = 8; % TR delay

    % Per-slice block count (prep mirrors main):
    %   fatsat + excite + prephase + 2*Nnav(label+nav) + gyPre + Ny_meas(ro) + lblIncSlc + delay
    blocks_per_slice = 2 + 1 + 2 * Nnav + 1 + Ny_meas + 1 + 1;
    gt.tr_min          = tr_min;
    gt.tr_max          = tr_max;
    gt.rf_center_s     = rf.center;
    gt.adc_num_samples = adc.numSamples;
    gt.adc_dwell_s     = adc.dwell;
    gt.seg_unique_ids  = {seg0_ids, seg1_ids, seg2_ids};  % 3 segments
    gt.unique_blocks   = 0:9;
    % lblOnce1/lblOnce0 merged into first fat-sat block of each section
    % (sticky flag propagates to all subsequent blocks until changed).
    gt.num_prep_blocks = 0;            % degenerate: absorbed into main
    gt.num_cool_blocks = 0;
    gt.degenerate_prep = 1;            % dummy volume structure == main (same block defs)
    gt.degenerate_cool = 1;            % no cooldown (absent = degenerate)
    gt.num_once1_blocks = blocks_per_slice * Nslices * NDummyVolumes;
    if Nslices > 1
        gt.tr_size = blocks_per_slice;   % per-slice period (library detects slice repeat)
    else
        gt.tr_size = blocks_per_slice;  % single-slice: library detects per-slice period
    end

    fname = seq_filename('epi_2d', num_slices, num_averages);
    check_and_write(seq, fname, fov, thick, num_slices, num_averages, gt);
end


%% ========================================================================
%  MPRAGE (3D inversion-recovery GRE)
%  ========================================================================

function write_mprage(num_averages)
    fprintf('Generating MPRAGE (%d avg) ...\n', num_averages);

    sys = make_system();
    seq = mr.Sequence(sys);

    alpha      = 7;             % flip angle [deg]
    roDur      = 2.560e-3;      % readout flat time (same constraints as SPGR)
    TI         = 1.1;
    TRout      = 2.5;
    rfSpoilInc = 84;

    fov = [256, 240, 192] * 1e-3;   % [x, y, z]
    Nx  = 256;                      % readout (x)
    Ny  = 240;                      % PE1 (y) — inner loop
    Nz  = 16;                       % partition (z) — outer loop (small for fast test generation)

    % --- events (SPGR-like FLASH inner shot) ---
    rf180 = mr.makeBlockPulse(pi, sys, ...
        'Duration', 10e-3, 'use', 'excitation');
    rf = mr.makeBlockPulse(alpha * pi / 180, sys, ...
        'Duration', 100e-6, 'use', 'excitation');

    deltak = 1 ./ fov;
    gx     = mr.makeTrapezoid('x', 'FlatArea', Nx * deltak(1), ...
                              'FlatTime', roDur, 'system', sys);
    adc    = mr.makeAdc(Nx, 'Duration', gx.flatTime, ...
                        'Delay', gx.riseTime, 'system', sys);
    gxPre  = mr.makeTrapezoid('x', 'Area', -gx.area / 2, 'system', sys);
    gxSpoil = mr.makeTrapezoid('x', 'Area', 3 * Nx * deltak(1), 'system', sys);

    % Extended trapezoids: connect readout → spoiler (no dead ramp time)
    transTime = max(ceil(abs(gxSpoil.amplitude - gx.amplitude) ...
                / sys.maxSlew / sys.gradRasterTime) * sys.gradRasterTime, ...
                sys.gradRasterTime);

    % Block 2: readout ramps into spoiler amplitude
    gx_ext_times = [0, gx.riseTime, gx.riseTime + gx.flatTime, ...
                    gx.riseTime + gx.flatTime + transTime];
    gx_ext_amps  = [0, gx.amplitude, gx.amplitude, gxSpoil.amplitude];
    gx_ext = mr.makeExtendedTrapezoid('x', 'times', gx_ext_times, ...
                                      'amplitudes', gx_ext_amps);

    % Block 3: spoiler continues from readout, then ramps down
    spFlatDur = gxSpoil.riseTime + gxSpoil.flatTime;  % riseTime repurposed as flat
    gxSp_ext_times = [0, spFlatDur, spFlatDur + gxSpoil.fallTime];
    gxSp_ext_amps  = [gxSpoil.amplitude, gxSpoil.amplitude, 0];
    gxSp_ext = mr.makeExtendedTrapezoid('x', 'times', gxSp_ext_times, ...
                                        'amplitudes', gxSp_ext_amps);

    % PE gradients
    gpe1   = mr.makeTrapezoid('y', 'Area', deltak(2) * Ny / 2, ...
                              'Duration', mr.calcDuration(gxPre), 'system', sys);
    gpe2   = mr.makeTrapezoid('z', 'Area', deltak(3) * Nz / 2, ...
                              'Duration', mr.calcDuration(gxPre), 'system', sys);
    gslSp  = mr.makeTrapezoid('z', 'Area', max(deltak .* [Nx Ny Nz]) * 4, ...
                              'Duration', 10e-3, 'system', sys);

    pe1Steps = ((0:Ny-1) - Ny/2) / Ny * 2;
    pe2Steps = ((0:Nz-1) - Nz/2) / Nz * 2;

    % Inner TR timing: rf | pre+PE | gx_ext+adc | gxSp_ext+PE_rewind
    TRinner = mr.calcDuration(rf) ...
            + max([mr.calcDuration(gxPre), mr.calcDuration(gpe1), mr.calcDuration(gpe2)]) ...
            + mr.calcDuration(gx_ext) ...
            + max([mr.calcDuration(gxSp_ext), mr.calcDuration(gpe1), mr.calcDuration(gpe2)]);

    TIdelay = mr.makeDelay(0.1e-3);
    TRoutDelay = 0.1e-3;
    TRoutDelay = round(TRoutDelay / sys.blockDurationRaster) * sys.blockDurationRaster;
    if TRoutDelay < sys.blockDurationRaster
        TRoutDelay = sys.blockDurationRaster;
    end

    % Pre-create labels
    lblIncLin   = mr.makeLabel('INC', 'LIN', 1);
    lblIncPar   = mr.makeLabel('INC', 'PAR', 1);
    lblResetLin = mr.makeLabel('SET', 'LIN', 0);
    lblResetPar = mr.makeLabel('SET', 'PAR', 0);
    lblOnce1    = mr.makeLabel('SET', 'ONCE', 1);
    lblOnce0    = mr.makeLabel('SET', 'ONCE', 0);

    % Build sequence
    rf_phase = 0;
    rf_inc   = 0;

    for j = 0:Nz
        if j == 0
            seq.addBlock(rf180, lblOnce1, lblResetLin, lblResetPar);
        elseif j == 1
            seq.addBlock(rf180, lblOnce0, lblResetLin, lblResetPar);
        else
            seq.addBlock(rf180, lblResetLin, lblIncPar);
        end
        seq.addBlock(TIdelay, gslSp);

        for i = 1:Ny
            rf.phaseOffset  = rf_phase / 180 * pi;
            adc.phaseOffset = rf_phase / 180 * pi;
            rf_inc   = mod(rf_inc + rfSpoilInc, 360.0);
            rf_phase = mod(rf_phase + rf_inc, 360.0);

            seq.addBlock(rf, lblIncLin);
            if j == 0
                seq.addBlock(gxPre, ...
                    mr.scaleGrad(gpe1, 0), ...
                    mr.scaleGrad(gpe2, 0));
                seq.addBlock(gx_ext);
                seq.addBlock(gxSp_ext, ...
                    mr.scaleGrad(gpe1, 0), ...
                    mr.scaleGrad(gpe2, 0));
            else
                seq.addBlock(gxPre, ...
                    mr.scaleGrad(gpe1, pe1Steps(i)), ...
                    mr.scaleGrad(gpe2, pe2Steps(j)));
                seq.addBlock(gx_ext, adc);
                seq.addBlock(gxSp_ext, ...
                    mr.scaleGrad(gpe1, -pe1Steps(i)), ...
                    mr.scaleGrad(gpe2, -pe2Steps(j)));
            end
        end
        seq.addBlock(mr.makeDelay(TRoutDelay));
    end

    seq.setDefinition('FOV', fov);
    seq.setDefinition('Name', 'mprage');
    seq.setDefinition('OrientationMapping', 'AX');

    % --- canonical TR = full inversion-to-inversion block range ---
    % Build standalone canonical pass sequences.  For MPRAGE with
    % degenerate prep (num_prep=0), the canonical TR spans one full
    % partition: [rf180, TIdelay, Ny*(rf+pre+readout+spoil), TRoutDelay]
    blocks_per_partition = 2 + 4 * Ny + 1;

    % MAX_POS: PE1 at actual per-line scale (constant across partitions),
    % PE2 at full scale with sign from first imaging partition (j=1).
    % pe2Steps(1) = -1 (most negative), so sign = -1.
    pe2_sign = sign(pe2Steps(1));
    if pe2_sign == 0; pe2_sign = 1; end
    pass_max = mr.Sequence(sys);
    pass_max.addBlock(rf180);
    pass_max.addBlock(TIdelay, gslSp);
    for i = 1:Ny
        pass_max.addBlock(rf);
        pass_max.addBlock(gxPre, mr.scaleGrad(gpe1, pe1Steps(i)), mr.scaleGrad(gpe2, pe2_sign));
        pass_max.addBlock(gx_ext, adc);
        pass_max.addBlock(gxSp_ext, mr.scaleGrad(gpe1, -pe1Steps(i)), mr.scaleGrad(gpe2, -pe2_sign));
    end
    pass_max.addBlock(mr.makeDelay(TRoutDelay));

    % ZERO_VAR: PE1 and PE2 = 0 (both variable across TRs because
    % j=0 dummy uses 0 while j>0 uses pe1Steps(i)/pe2Steps(j)).
    % All other gradients (gxPre, gx_ext, gxSp_ext, gslSp) are constant
    % across TRs → kept at actual amplitude.
    pass_min = mr.Sequence(sys);
    pass_min.addBlock(rf180);
    pass_min.addBlock(TIdelay, gslSp);
    for i = 1:Ny
        pass_min.addBlock(rf);
        pass_min.addBlock(gxPre, mr.scaleGrad(gpe1, 0), mr.scaleGrad(gpe2, 0));
        pass_min.addBlock(gx_ext, adc);
        pass_min.addBlock(gxSp_ext, mr.scaleGrad(gpe1, 0), mr.scaleGrad(gpe2, 0));
    end
    pass_min.addBlock(mr.makeDelay(TRoutDelay));

    % --- structural ground truth ---
    % Block defs:
    %   0: rf180 + labels                (inversion pulse; labels vary by j)
    %   1: TIdelay + gslSp               (TI delay + z-axis slab spoiler)
    %   2: rf + lblIncLin                (FLASH excitation)
    %   3: gxPre + gpe1 + gpe2           (prephaser + PE encode)
    %   4: gx_ext + adc                  (readout, ramps into spoiler amp)
    %   5: gxSp_ext + gpe1 + gpe2        (spoiler continuation + PE rewind)
    %   6: TRoutDelay                    (end-of-partition delay)
    gt.tr_min_range    = {pass_min, []};  % full canonical pass = canonical TR
    gt.tr_max_range    = {pass_max, []};
    gt.rf_center_s     = rf.center;
    gt.adc_num_samples = adc.numSamples;
    gt.adc_dwell_s     = adc.dwell;
    gt.seg_unique_ids  = {[0, 1], [2, 3, 4, 5], 6};  % 3 segments
    gt.unique_blocks   = 0:6;
    gt.num_prep_blocks = 0;            % degenerate: absorbed into main
    gt.num_cool_blocks = 0;
    gt.degenerate_prep = 1;            % j==0 TR treated as degenerate prep
    gt.degenerate_cool = 1;            % no cooldown (absent = degenerate)
    gt.num_once1_blocks = 2 + 4 * Ny + 1;  % j==0 iteration for scan table
    gt.tr_size = 2 + 4 * Ny + 1;

    fname = sprintf('mprage_3d_%davg.seq', num_averages);
    check_and_write(seq, fname, fov(1), fov(3), 1, num_averages, gt);
end

%% ========================================================================
%  Noncartesian MPRAGE (3D stack-of-stars inversion-recovery GRE)
%  ========================================================================

function write_mprage_noncart(num_averages, num_shots, use_rotext)
    fprintf('Generating Noncartesian MPRAGE (%d avg, %d shots, rotext=%d) ...\n', ...
            num_averages, num_shots, use_rotext);

    sys = make_system();
    seq = mr.Sequence(sys);

    alpha      = 7;             % flip angle [deg]
    ro_dur     = 5120e-6;       % RO duration (Nx * k * adcRasterTime, k=10)
    ro_os      = 1;
    TI         = 1.1;
    TRout      = 2.5;
    rfSpoilInc = 84;
    rfLen      = 100e-6;

    fov = [256, 240, 192] * 1e-3;   % [x, y, z]
    Nx  = 256;                       % readout samples (x)
    Nz  = 16;                        % partition (z) — outer loop (small for fast test generation)

    % --- events ---
    rf180 = mr.makeBlockPulse(pi, sys, ...
        'Duration', 10e-3, 'use', 'excitation');
    rf = mr.makeBlockPulse(alpha * pi / 180, sys, ...
        'Duration', rfLen, 'use', 'excitation');

    deltak = 1 ./ fov;

    % Readout trapezoid template → arbitrary waveform for rotation
    groTrap = mr.makeTrapezoid('x', ...
        'Amplitude', Nx * deltak(1) / ro_dur, ...
        'FlatTime', ceil((ro_dur + sys.adcDeadTime) / sys.gradRasterTime) * sys.gradRasterTime, ...
        'system', sys);
    gxPre  = mr.makeTrapezoid('x', 'Area', -groTrap.area / 2, 'system', sys);

    if gxPre.flatTime > 0
        times = cumsum([0, ...
            gxPre.riseTime, gxPre.flatTime, gxPre.fallTime, ...
            groTrap.riseTime, groTrap.flatTime, groTrap.fallTime, ...
            gxPre.riseTime, gxPre.flatTime, gxPre.fallTime]);
        amp = [0, ...
            gxPre.amplitude, gxPre.amplitude, 0, ...
            groTrap.amplitude, groTrap.amplitude, 0, ...
            gxPre.amplitude, gxPre.amplitude, 0];
    else
        times = cumsum([0, ...
            gxPre.riseTime, gxPre.fallTime, ...
            groTrap.riseTime, groTrap.flatTime, groTrap.fallTime, ...
            gxPre.riseTime, gxPre.fallTime]);
        amp = [0, ...
            gxPre.amplitude, 0, ...
            groTrap.amplitude, groTrap.amplitude, 0, ...
            gxPre.amplitude, 0];
    end
    waveform = mr.pts2waveform(times, amp, sys.gradRasterTime);
    groArbX  = mr.makeArbitraryGrad('x', waveform, 'system', sys, 'first', 0, 'last', 0);
    groArbY  = mr.makeArbitraryGrad('y', 0 * waveform, 'system', sys, 'first', 0, 'last', 0);

    % ADC
    prewind_duration = mr.calcDuration(gxPre);
    adc = mr.makeAdc(Nx * ro_os, 'Duration', ro_dur, 'Delay', prewind_duration+groTrap.riseTime, 'system', sys);

    % Partition encoding (along z)
    gpe = mr.makeTrapezoid('z', 'Area', -deltak(3) * Nz / 2, 'system', sys);
    [gpe, ~] = mr.align('right', gpe, gxPre);
    
    if gpe.flatTime > 0
        times = cumsum([0, ...,
            gpe.riseTime, gpe.flatTime, gpe.fallTime, ...
            mr.calcDuration(groTrap), ...
            gpe.riseTime, gpe.flatTime, gpe.fallTime]);
        amplitudes = [0, ...
            gpe.amplitude, gpe.amplitude, 0, ...
            0, ...
            -gpe.amplitude, -gpe.amplitude, 0];
    else
        times = cumsum([0, ...,
            gpe.riseTime, gpe.fallTime, ...
            mr.calcDuration(groTrap), ...
            gpe.riseTime, gpe.fallTime]);
        amplitudes = [0, ...
            gpe.amplitude, 0, ...
            0, ...
            -gpe.amplitude, 0];
    end
    delay = gpe.delay;
    gpe = mr.makeExtendedTrapezoid('z', 'times', times, 'amplitudes', amplitudes, 'system', sys);
    gpe.delay = delay;

    % Slab spoiler (along z)
    gslSp = mr.makeTrapezoid('z', ...
        'Area', max(deltak .* [Nx 1 Nz]) * 4, 'Duration', 10e-3, 'system', sys);

    pe_steps = ((0:Nz-1) - Nz/2) / Nz * 2;

    % TI delay — for radial, every spoke passes through k-center,
    % so TI targets the first excitation of each partition
    TIdelay = mr.makeDelay(0.1e-3);
    TRoutDelay = 0.1e-3;  % override for fast test generation (shorten end-of-partition delay)
    TRoutDelay = round(TRoutDelay / sys.blockDurationRaster) * sys.blockDurationRaster;
    if TRoutDelay < sys.blockDurationRaster
        TRoutDelay = sys.blockDurationRaster;
    end

    % Pre-create labels
    lblIncLin   = mr.makeLabel('INC', 'LIN', 1);
    lblIncPar   = mr.makeLabel('INC', 'PAR', 1);
    lblResetLin = mr.makeLabel('SET', 'LIN', 0);
    lblResetPar = mr.makeLabel('SET', 'PAR', 0);
    lblOnce1    = mr.makeLabel('SET', 'ONCE', 1);
    lblOnce0    = mr.makeLabel('SET', 'ONCE', 0);

    rf_phase = 0;
    rf_inc   = 0;
    phi      = 0;
    dphi     = 137.51 * pi / 180;  % golden angle [rad]

    % --- Pre-compute per-shot rotated readout gradients (non-rotext path) ---
    % For the non-rotext path we apply a 2-D rotation to the base
    % (waveform, 0*waveform) pair ourselves so that every shot produces
    % arbitrary-grad events with identical numSamples.  This keeps the C
    % library's dedup happy (grad definition = timing only; amplitude is
    % per-instance).
    if ~use_rotext
        % Total number of unique angles across all partitions (j>0).
        % j==0 uses unrotated grads, so we only rotate for j = 1..Nz.
        total_spokes = Nz * num_shots;
        groX_shots = cell(1, total_spokes);
        groY_shots = cell(1, total_spokes);
        spoke_phi  = 0;
        for s = 1:total_spokes
            c = cos(spoke_phi);
            sn = sin(spoke_phi);
            wx = c * waveform;   % rotated x component
            wy = sn * waveform;  % rotated y component
            groX_shots{s} = mr.makeArbitraryGrad('x', wx, 'system', sys, 'first', 0, 'last', 0);
            groY_shots{s} = mr.makeArbitraryGrad('y', wy, 'system', sys, 'first', 0, 'last', 0);
            spoke_phi = spoke_phi + dphi;
        end
        spoke_idx = 0;  % running index into groX/Y_shots
    end

    % Build sequence
    for j = 0:Nz
        if j == 0
            seq.addBlock(rf180, lblOnce1, lblResetLin, lblResetPar);
        elseif j == 1
            seq.addBlock(rf180, lblOnce0, lblResetLin, lblResetPar);
        else
            seq.addBlock(rf180, lblResetLin, lblIncPar);
        end
        seq.addBlock(TIdelay, gslSp);

        if j == 0
            gpeJ = mr.scaleGrad(gpe, 0);
        else
            gpeJ = mr.scaleGrad(gpe, peSteps(j)); 
        end

        for i = 1:num_shots
            rf.phaseOffset  = rf_phase / 180 * pi;
            adc.phaseOffset = rf_phase / 180 * pi;

            % RF block
            seq.addBlock(rf, lblIncLin);
            
            % Readout block: rotated arbitrary gradients + ADC
            if j == 0
                % Dummy partition: unrotated, no ADC
                seq.addBlock(groArbX, groArbY, gpeJ);
            elseif use_rotext
                % Rotation-extension path: hardware rotation descriptor
                seq.addBlock(adc, groArbX, groArbY, gpeJ, ...
                    mr.makeRotation('axis', 'z', 'angle', phi));
            else
                % Explicit rotation path: pre-computed per-shot grads
                spoke_idx = spoke_idx + 1;
                seq.addBlock(adc, groX_shots{spoke_idx}, groY_shots{spoke_idx}, gpeJ);
            end
            seq.addBlock(gslSp);
            
            rf_inc   = mod(rf_inc + rfSpoilInc, 360.0);
            rf_phase = mod(rf_phase + rf_inc, 360.0);
            
            if j > 0
                phi = phi + dphi;
            end
        end
        seq.addBlock(mr.makeDelay(TRoutDelay));
    end

    seq.setDefinition('FOV', fov);
    seq.setDefinition('Name', 'mprage_noncart');
    seq.setDefinition('OrientationMapping', 'AX');

    % --- canonical TR = full inversion-to-inversion block range ---
    % Build canonical pass sequences.  Structure per partition:
    %   [rf180, TIdelay+gslSp, num_shots*(rf+readout+gslSp), TRoutDelay]
    blocks_per_partition = 3 + 3 * num_shots;

    % MAX_POS: partition encode at full scale with sign from first
    % imaging partition (j=1).  peSteps(1) = -1, so sign = -1.
    % Readout rotation does NOT affect pos_max for the C library (all
    % shots share the same gradient definition; rotation only changes
    % amplitudes).
    pe_sign = sign(peSteps(1));
    if pe_sign == 0; pe_sign = 1; end
    pass_max = mr.Sequence(sys);
    pass_max.addBlock(rf180);
    pass_max.addBlock(TIdelay, gslSp);
    for i = 1:num_shots
        pass_max.addBlock(rf);
        pass_max.addBlock(adc, groArbX, groArbY, mr.scaleGrad(gpe, pe_sign));
        pass_max.addBlock(gslSp);
    end
    pass_max.addBlock(mr.makeDelay(TRoutDelay));

    % ZERO_VAR: For rotext path, only partition encode varies (gpe = 0);
    % readout base grads are constant in the block table (rotation stored
    % externally → C library does not detect gx/gy as variable).
    % For non-rotext path, readout grads also vary (different per-spoke
    % waveforms in the block table) → gx, gy, gz all zeroed at readout
    % positions.
    pass_min = mr.Sequence(sys);
    pass_min.addBlock(rf180);
    pass_min.addBlock(TIdelay, gslSp);
    if use_rotext
        % Rotext: gx/gy constant (same base), gz variable (gpe)
        for i = 1:num_shots
            pass_min.addBlock(rf);
            pass_min.addBlock(adc, groArbX, groArbY, mr.scaleGrad(gpe, 0));
            pass_min.addBlock(gslSp);
        end
    else
        % Non-rotext: gx/gy/gz all variable at readout positions
        for i = 1:num_shots
            pass_min.addBlock(rf);
            pass_min.addBlock(adc, ...
                mr.scaleGrad(groArbX, 0), mr.scaleGrad(groArbY, 0), ...
                mr.scaleGrad(gpe, 0));
            pass_min.addBlock(gslSp);
        end
    end
    pass_min.addBlock(mr.makeDelay(TRoutDelay));

    % --- structural ground truth ---
    % Block defs:
    %   0: rf180 + labels                (inversion pulse; labels vary by j)
    %   1: TIdelay + gslSp               (TI delay + z-axis slab spoiler)
    %   2: rf + lblIncLin                (FLASH excitation)
    %   3: groArbX + groArbY + gpe + adc (rotated readout + partition encode)
    %   1: gslSp                         (post-readout z-axis spoiler)
    %   4: TRoutDelay                    (end-of-partition delay)
    gt.tr_min_range    = {pass_min, []};  % full canonical pass = canonical TR
    gt.tr_max_range    = {pass_max, []};
    gt.rf_center_s     = rf.center;
    gt.adc_num_samples = adc.numSamples;
    gt.adc_dwell_s     = adc.dwell;
    gt.seg_unique_ids  = {[0, 1], [2, 3, 1], 4};  % 3 segments
    gt.unique_blocks   = 0:4;
    gt.num_prep_blocks = 0;             % degenerate: absorbed into main
    gt.num_cool_blocks = 0;
    gt.degenerate_prep = 1;             % j==0 TR treated as degenerate prep
    gt.degenerate_cool = 1;             % no cooldown (absent = degenerate)
    gt.num_once1_blocks = 3 + 3 * num_shots;  % j==0 iteration for scan table
    gt.tr_size = 3 + 3 * num_shots;

    if use_rotext
        rotext_tag = '_rotext';
    else
        rotext_tag = '';
    end
    fname = sprintf('mprage_noncart_3d_%dshots%s_%davg.seq', num_shots, rotext_tag, num_averages);
    check_and_write(seq, fname, fov(1), fov(3), 1, num_averages, gt);
end