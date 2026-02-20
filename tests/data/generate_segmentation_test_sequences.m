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
%   5. Each generator accepts num_slices (1 = single-slice, >1 = multi-slice).

clear; clc;
import mr.*

%% --- run all generators -------------------------------------------------

write_bssfp(1);
write_spgr(1);
write_spgr(3);
write_fse(1);
write_fse(3);
write_epi(1);
write_epi(3);
write_mprage();
write_radial(1);
write_radial(3);
write_radial_rotext(1);

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

function fname = seq_filename(prefix, num_slices, suffix)
% Build output filename: <prefix>_<Nsl>sl<suffix>.seq
    if nargin < 3, suffix = ''; end
    fname = sprintf('%s_%dsl%s.seq', prefix, num_slices, suffix);
end

function check_and_write(seq, fname, fov, thick, num_slices, num_averages)
% Timing check, definitions, write, ground truth.
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
    export_ground_truth(seq, fname, num_averages);
end

function export_ground_truth(seq, seq_fname, num_averages)
% Write per-block ground truth CSV and metadata text file.
%
% CSV columns (0-indexed block):
%   idx, duration_us, rf_amp_hz, rf_freq_hz, rf_phase_rad,
%   gx_amp, gy_amp, gz_amp, adc_flag, adc_freq_hz, adc_phase_rad
%
% The data corresponds to a single pass (num_averages == 1) through
% all blocks.  For num_averages > 1 the cursor replays main blocks,
% which the C test can derive from this base listing plus the
% prep / cooldown boundaries.

    N = length(seq.blockDurations);

    [~, base, ~] = fileparts(seq_fname);

    % --- per-block data ---
    fid = fopen([base '_blocks.csv'], 'w');
    fprintf(fid, 'idx,duration_us,rf_amp_hz,rf_freq_hz,rf_phase_rad,gx_amp,gy_amp,gz_amp,adc_flag,adc_freq_hz,adc_phase_rad\n');

    num_adcs = 0;
    for n = 1:N
        blk = seq.getBlock(n);
        dur_us = round(seq.blockDurations(n) * 1e6);

        % RF
        [rf_amp, rf_freq, rf_phase] = extract_rf(blk);

        % Gradients
        gx_amp = extract_grad_amp(blk, 'gx');
        gy_amp = extract_grad_amp(blk, 'gy');
        gz_amp = extract_grad_amp(blk, 'gz');

        % ADC
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

function write_bssfp(num_slices)
    fprintf('Generating bSSFP (%d slice) ...\n', num_slices);
    assert(num_slices == 1, 'bSSFP supports single-slice only (gradient continuity).');

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

    fname = seq_filename('bssfp_2d', num_slices);
    check_and_write(seq, fname, fov, thick, num_slices, 1);
end


%% ========================================================================
%  SPGR  (spoiled GRE with labels)
%  ========================================================================

function write_spgr(num_slices)
    fprintf('Generating SPGR (%d slice) ...\n', num_slices);

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
    roDur     = 3.2e-3;            % readout duration

    % --- events ---
    [rf, gz] = mr.makeSincPulse(alpha * pi / 180, ...
        'Duration', 3e-3, 'SliceThickness', thick, ...
        'apodization', 0.42, 'timeBwProduct', 4, ...
        'use', 'excitation', 'system', sys);

    deltak  = 1 / fov;
    gx      = mr.makeTrapezoid('x', 'FlatArea', Nx * deltak, ...
                               'FlatTime', roDur, 'system', sys);
    adc     = mr.makeAdc(Nx, 'Duration', gx.flatTime, ...
                         'Delay', gx.riseTime, 'system', sys);
    gxPre   = mr.makeTrapezoid('x', 'Area', -gx.area / 2, ...
                               'Duration', 1e-3, 'system', sys);
    gzReph  = mr.makeTrapezoid('z', 'Area', -gz.area / 2, ...
                               'Duration', 1e-3, 'system', sys);
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
    lblOnce2  = mr.makeLabel('SET', 'ONCE', 2);
    lblIncLin = mr.makeLabel('INC', 'LIN', 1);
    lblSetLin = mr.makeLabel('SET', 'LIN', 0);
    lblIncSlc = mr.makeLabel('INC', 'SLC', 1);
    lblSetSlc = mr.makeLabel('SET', 'SLC', 0);
    lblIncRep = mr.makeLabel('INC', 'REP', 1);

    rf_phase = 0;
    rf_inc   = 0;

    % --- prep: dummy scans (ONCE=1) ---
    seq.addBlock(lblOnce1);
    for d = 1:Ndummy
        for s = 1:Nslices
            rf.freqOffset  = gz.amplitude * thick * (s - 1 - (Nslices-1)/2);
            rf.phaseOffset = rf_phase / 180 * pi;
            rf_inc   = mod(rf_inc + rfSpoilInc, 360.0);
            rf_phase = mod(rf_phase + rf_inc, 360.0);

            seq.addBlock(rf, gz);
            seq.addBlock(gxPre, gzReph);     % no PE during dummies
            seq.addBlock(evDelayTE);
            seq.addBlock(gx);                % no ADC
            seq.addBlock(gxSpoil, gzSpoil, evDelayTR);
        end
    end
    seq.addBlock(lblOnce0);   % end prep

    % --- main imaging loop ---
    for i = 1:Ny
        for s = 1:Nslices
            rf.freqOffset  = gz.amplitude * thick * (s - 1 - (Nslices-1)/2);
            rf.phaseOffset = rf_phase / 180 * pi;
            adc.phaseOffset = rf_phase / 180 * pi;
            rf_inc   = mod(rf_inc + rfSpoilInc, 360.0);
            rf_phase = mod(rf_phase + rf_inc, 360.0);

            seq.addBlock(rf, gz);

            if maxPeArea > 0
                gyPre = mr.scaleGrad(gyMax, phaseAreas(i) / maxPeArea);
            else
                gyPre = mr.scaleGrad(gyMax, 0);
            end
            seq.addBlock(gxPre, gyPre, gzReph);
            seq.addBlock(evDelayTE);
            seq.addBlock(gx, adc);

            % Spoiler + rewind PE
            gyRew = mr.scaleGrad(gyPre, -1);
            if i == Ny && s == Nslices
                seq.addBlock(gxSpoil, gzSpoil, gyRew, evDelayTR, lblSetLin, lblSetSlc);
            elseif s == Nslices
                seq.addBlock(gxSpoil, gzSpoil, gyRew, evDelayTR, lblIncLin, lblSetSlc);
            else
                seq.addBlock(gxSpoil, gzSpoil, gyRew, evDelayTR, lblIncSlc);
            end
        end
    end

    % --- cooldown (ONCE=2) ---
    seq.addBlock(lblIncRep, lblOnce2);

    fprintf('  TR = %.3f ms   TE = %.3f ms   Ndummy = %d\n', ...
            TR * 1e3, TE * 1e3, Ndummy);

    fname = seq_filename('gre_2d', num_slices);
    check_and_write(seq, fname, fov, thick, num_slices, 1);
end


%% ========================================================================
%  FSE  (turbo spin echo)
%  ========================================================================

function write_fse(num_slices)
    fprintf('Generating FSE (%d slice) ...\n', num_slices);

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

    samplingTime = 6.4e-3;
    readoutTime  = samplingTime + 2 * sys.adcDeadTime;
    tEx   = 2.5e-3;
    tExwd = tEx + sys.rfRingdownTime + sys.rfDeadTime;
    tRef  = 2e-3;
    tRefwd = tRef + sys.rfRingdownTime + sys.rfDeadTime;
    tSp    = 0.5 * (TE1 - readoutTime - tRefwd);
    tSpex  = 0.5 * (TE1 - tExwd - tRefwd);
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

    % Labels
    lblOnce1 = mr.makeLabel('SET', 'ONCE', 1);
    lblOnce0 = mr.makeLabel('SET', 'ONCE', 0);
    lblOnce2 = mr.makeLabel('SET', 'ONCE', 2);

    % --- prep: dummy excitations (ONCE=1) ---
    seq.addBlock(lblOnce1);
    for kex = 0:(Ndummy - 1)
        for s = 1:Nslices
            rfex.freqOffset  = GSex.amplitude * thick * (s - 1 - (Nslices-1)/2);
            rfref.freqOffset = GSref.amplitude * thick * (s - 1 - (Nslices-1)/2);
            rfex.phaseOffset  = rfex_phase - 2*pi * rfex.freqOffset * mr.calcRfCenter(rfex);
            rfref.phaseOffset = rfref_phase - 2*pi * rfref.freqOffset * mr.calcRfCenter(rfref);

            seq.addBlock(GS1);
            seq.addBlock(GS2, rfex);
            seq.addBlock(GS3, GR3);
            for kech = 1:necho
                seq.addBlock(GS4, rfref);
                seq.addBlock(GS5, GR5);     % no PE during dummies
                seq.addBlock(GR6);           % no ADC during dummies
                seq.addBlock(GS7, GR7);
            end
            seq.addBlock(GS4);
            seq.addBlock(GS5);
            seq.addBlock(delayTR);
        end
    end
    seq.addBlock(lblOnce0);  % end prep

    % --- main imaging loop ---
    for kex = 1:nex
        for s = 1:Nslices
            rfex.freqOffset  = GSex.amplitude * thick * (s - 1 - (Nslices-1)/2);
            rfref.freqOffset = GSref.amplitude * thick * (s - 1 - (Nslices-1)/2);
            rfex.phaseOffset  = rfex_phase - 2*pi * rfex.freqOffset * mr.calcRfCenter(rfex);
            rfref.phaseOffset = rfref_phase - 2*pi * rfref.freqOffset * mr.calcRfCenter(rfref);

            seq.addBlock(GS1);
            seq.addBlock(GS2, rfex);
            seq.addBlock(GS3, GR3);

            for kech = 1:necho
                phaseArea = phaseAreas(kech, kex);
                if maxPeArea > 0
                    pe_scale = phaseArea / maxPeArea;
                else
                    pe_scale = 0;
                end
                GPpre = mr.scaleGrad(gyMax, pe_scale);
                GPrew = mr.scaleGrad(gyMax, -pe_scale);

                seq.addBlock(GS4, rfref);
                seq.addBlock(GS5, GR5, GPpre);
                seq.addBlock(GR6, adc);
                seq.addBlock(GS7, GR7, GPrew);
            end

            seq.addBlock(GS4);
            seq.addBlock(GS5);
            seq.addBlock(delayTR);
        end
    end

    % --- cooldown (ONCE=2) ---
    seq.addBlock(lblOnce2);

    fname = seq_filename('fse_2d', num_slices);
    check_and_write(seq, fname, fov, thick, num_slices, 1);
end


%% ========================================================================
%  EPI (echo-planar imaging)
%  ========================================================================

function write_epi(num_slices)
    fprintf('Generating EPI (%d slice) ...\n', num_slices);

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
    gz_fs = mr.makeTrapezoid('z', sys, 'delay', mr.calcDuration(rf_fs), ...
                             'Area', 0.1 / 1e-4);

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
    gx = mr.makeTrapezoid('x', sys, 'Area', deltak * Nx + extra_area, ...
                          'duration', readoutTime + blip_dur);
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
    adcDwell   = floor(readoutTime / adcSamples * 1e7) * 1e-7;
    adc = mr.makeAdc(adcSamples, 'Dwell', adcDwell, 'Delay', blip_dur / 2);
    time_to_center = adc.dwell * ((adcSamples - 1)/2 + 0.5);
    adc.delay = round((gx.riseTime + gx.flatTime/2 - time_to_center) * 1e6) * 1e-6;

    % Split blips
    gy_parts = mr.splitGradientAt(gy, blip_dur / 2, sys);
    [gy_blipup, gy_blipdown, ~] = mr.align('right', gy_parts(1), ...
                                            'left', gy_parts(2), gx);
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
    lblOnce2  = mr.makeLabel('SET', 'ONCE', 2);
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

    % --- cooldown (ONCE=2) ---
    seq.addBlock(lblIncRep, lblOnce2);

    % Definitions
    seq.setDefinition('Name', 'epi');
    seq.setDefinition('SlicePositions', slicePositions);
    seq.setDefinition('SliceThickness', thick);
    seq.setDefinition('SliceGap', sliceGap);
    seq.setDefinition('ReadoutOversamplingFactor', ro_os);

    fname = seq_filename('epi_2d', num_slices);
    check_and_write(seq, fname, fov, thick, num_slices, 1);
end


%% ========================================================================
%  MPRAGE (3D inversion-recovery GRE)
%  ========================================================================

function write_mprage()
    fprintf('Generating MPRAGE ...\n');

    sys = make_system();
    seq = mr.Sequence(sys);

    alpha   = 7;             % flip angle [deg]
    ro_dur  = 5040e-6;       % RO duration (multiple of 20us)
    ro_os   = 1;
    ro_spoil = 3;
    TI      = 1.1;
    TRout   = 2.5;
    rfSpoilInc = 84;
    rfLen   = 100e-6;

    fov = [192, 240, 256] * 1e-3;
    N   = [192, 240, 256];
    ax.d1 = 'z'; ax.d2 = 'x';
    ax.d3 = setdiff('xyz', [ax.d1, ax.d2]);
    ax.n1 = strfind('xyz', ax.d1);
    ax.n2 = strfind('xyz', ax.d2);
    ax.n3 = strfind('xyz', ax.d3);

    % --- events ---
    rf = mr.makeBlockPulse(alpha * pi / 180, sys, ...
                           'Duration', rfLen, 'use', 'excitation');
    rf180 = mr.makeAdiabaticPulse('hypsec', sys, ...
                                  'Duration', 10.24e-3, 'dwell', 1e-5, ...
                                  'use', 'excitation');

    deltak = 1 ./ fov;
    gro    = mr.makeTrapezoid(ax.d1, 'Amplitude', ...
        N(ax.n1) * deltak(ax.n1) / ro_dur, ...
        'FlatTime', ceil((ro_dur + sys.adcDeadTime) / sys.gradRasterTime) ...
                    * sys.gradRasterTime, 'system', sys);
    adc    = mr.makeAdc(N(ax.n1) * ro_os, 'Duration', ro_dur, ...
                        'Delay', gro.riseTime, 'system', sys);
    groPre = mr.makeTrapezoid(ax.d1, 'Area', ...
        -gro.amplitude * (adc.dwell * (adc.numSamples/2 + 0.5) ...
         + 0.5 * gro.riseTime), 'system', sys);
    gpe1   = mr.makeTrapezoid(ax.d2, 'Area', -deltak(ax.n2) * N(ax.n2) / 2, ...
                              'system', sys);
    gpe2   = mr.makeTrapezoid(ax.d3, 'Area', -deltak(ax.n3) * N(ax.n3) / 2, ...
                              'system', sys);
    gslSp  = mr.makeTrapezoid(ax.d3, 'Area', max(deltak .* N) * 4, ...
                              'Duration', 10e-3, 'system', sys);

    [gro1, groSp] = mr.splitGradientAt(gro, gro.riseTime + gro.flatTime);
    if ro_spoil > 0
        groSp = mr.makeExtendedTrapezoidArea(gro.channel, gro.amplitude, 0, ...
                    deltak(ax.n1) / 2 * N(ax.n1) * ro_spoil, sys);
    end

    rf.delay = mr.calcDuration(groSp, gpe1, gpe2);
    [groPre, ~, ~] = mr.align('right', groPre, gpe1, gpe2);
    gro1.delay = mr.calcDuration(groPre);
    adc.delay  = gro1.delay + gro.riseTime;
    gro1 = mr.addGradients({gro1, groPre}, 'system', sys);

    TRinner = mr.calcDuration(rf) + mr.calcDuration(gro1);

    pe1Steps = ((0:N(ax.n2)-1) - N(ax.n2)/2) / N(ax.n2) * 2;
    pe2Steps = ((0:N(ax.n3)-1) - N(ax.n3)/2) / N(ax.n3) * 2;

    TIdelay = round((TI - (find(pe1Steps==0) - 1) * TRinner ...
              - (mr.calcDuration(rf180) - mr.calcRfCenter(rf180) - rf180.delay) ...
              - rf.delay - mr.calcRfCenter(rf)) / sys.blockDurationRaster) ...
              * sys.blockDurationRaster;
    TRoutDelay = TRout - TRinner * N(ax.n2) - TIdelay - mr.calcDuration(rf180);

    % Pre-create labels
    lblIncLin  = mr.makeLabel('INC', 'LIN', 1);
    lblIncPar  = mr.makeLabel('INC', 'PAR', 1);
    lblResetPar = mr.makeLabel('SET', 'PAR', 0);
    lblOnce1   = mr.makeLabel('SET', 'ONCE', 1);
    lblOnce0   = mr.makeLabel('SET', 'ONCE', 0);
    lblOnce2   = mr.makeLabel('SET', 'ONCE', 2);

    % Pre-register unchanging events
    gslSp.id  = seq.registerGradEvent(gslSp);
    groSp.id  = seq.registerGradEvent(groSp);
    gro1.id   = seq.registerGradEvent(gro1);
    [~, rf.shapeIDs]      = seq.registerRfEvent(rf);
    [rf180.id, rf180.shapeIDs] = seq.registerRfEvent(rf180);
    lblIncPar.id = seq.registerLabelEvent(lblIncPar);

    % Build sequence
    % First inversion block (prep, ONCE=1)
    seq.addBlock(rf180, lblOnce1);
    seq.addBlock(TIdelay, gslSp);
    seq.addBlock(lblOnce0);  % end prep

    rf_phase = 0;
    rf_inc   = 0;

    for j = 1:N(ax.n3)
        if j > 1
            seq.addBlock(rf180);
            seq.addBlock(TIdelay, gslSp);
        end

        gpe2je = mr.scaleGrad(gpe2, pe2Steps(j));
        gpe2je.id = seq.registerGradEvent(gpe2je);
        gpe2jr = mr.scaleGrad(gpe2, -pe2Steps(j));
        gpe2jr.id = seq.registerGradEvent(gpe2jr);

        for i = 1:N(ax.n2)
            rf.phaseOffset  = rf_phase / 180 * pi;
            adc.phaseOffset = rf_phase / 180 * pi;
            rf_inc   = mod(rf_inc + rfSpoilInc, 360.0);
            rf_phase = mod(rf_phase + rf_inc, 360.0);

            if i == 1
                seq.addBlock(rf);
            else
                seq.addBlock(rf, groSp, ...
                    mr.scaleGrad(gpe1, -pe1Steps(i-1)), gpe2jr, lblIncPar);
            end
            seq.addBlock(adc, gro1, ...
                mr.scaleGrad(gpe1, pe1Steps(i)), gpe2je);
        end
        seq.addBlock(groSp, mr.makeDelay(TRoutDelay), lblResetPar, lblIncLin);
    end

    % Cooldown (ONCE=2)
    seq.addBlock(lblOnce2);

    seq.setDefinition('FOV', fov);
    seq.setDefinition('Name', 'mprage');
    seq.setDefinition('OrientationMapping', 'SAG');

    fname = 'mprage_3d.seq';
    check_and_write(seq, fname, fov(1), fov(3), 1, 1);
end


%% ========================================================================
%  Radial GRE (using mr.rotate)
%  ========================================================================

function write_radial(num_slices)
    fprintf('Generating Radial GRE (%d slice) ...\n', num_slices);

    sys = make_system();
    seq = mr.Sequence(sys);

    fov     = 260e-3;
    Nx      = 320;
    alpha   = 10;
    thick   = 3e-3;
    TE      = 8e-3;
    TR      = 20e-3;
    Nr      = 256;     % number of spokes
    Nslices = num_slices;
    Ndummy  = 20;
    delta   = pi / Nr;
    rfSpoilInc = 84;

    % --- events ---
    [rf, gz] = mr.makeSincPulse(alpha * pi / 180, 'Duration', 4e-3, ...
        'SliceThickness', thick, 'apodization', 0.5, 'timeBwProduct', 4, ...
        'system', sys, 'use', 'excitation');

    deltak  = 1 / fov;
    gx      = mr.makeTrapezoid('x', 'FlatArea', Nx * deltak, ...
                               'FlatTime', 6.4e-3/5, 'system', sys);
    adc     = mr.makeAdc(Nx, 'Duration', gx.flatTime, ...
                         'Delay', gx.riseTime, 'system', sys);
    gxPre   = mr.makeTrapezoid('x', 'Area', -gx.area/2 - deltak/2, ...
                               'Duration', 2e-3, 'system', sys);
    gzReph  = mr.makeTrapezoid('z', 'Area', -gz.area/2, ...
                               'Duration', 2e-3, 'system', sys);
    gxSpoil = mr.makeTrapezoid('x', 'Area', 0.5 * Nx * deltak, 'system', sys);
    gzSpoil = mr.makeTrapezoid('z', 'Area', 4 / thick, 'system', sys);

    delayTE = ceil((TE - mr.calcDuration(gxPre) - gz.fallTime ...
              - gz.flatTime/2 - mr.calcDuration(gx)/2) ...
              / seq.gradRasterTime) * seq.gradRasterTime;
    delayTR = ceil((TR - mr.calcDuration(gxPre) - mr.calcDuration(gz) ...
              - mr.calcDuration(gx) - delayTE) ...
              / seq.gradRasterTime) * seq.gradRasterTime;
    assert(delayTR >= mr.calcDuration(gxSpoil, gzSpoil), 'TR too short');
    evDelayTE = mr.makeDelay(delayTE);
    evDelayTR = mr.makeDelay(delayTR);

    lblOnce1 = mr.makeLabel('SET', 'ONCE', 1);
    lblOnce0 = mr.makeLabel('SET', 'ONCE', 0);
    lblOnce2 = mr.makeLabel('SET', 'ONCE', 2);

    rf_phase = 0;
    rf_inc   = 0;

    % --- prep: dummy spokes (ONCE=1) ---
    seq.addBlock(lblOnce1);
    for i = 1:Ndummy
        phi = delta * (i - 1);
        rf.phaseOffset = rf_phase / 180 * pi;
        rf_inc   = mod(rf_inc + rfSpoilInc, 360.0);
        rf_phase = mod(rf_phase + rf_inc, 360.0);

        for s = 1:Nslices
            rf.freqOffset = gz.amplitude * thick * (s - 1 - (Nslices-1)/2);
            seq.addBlock(rf, gz);
            seq.addBlock(mr.rotate('z', phi, gxPre, gzReph));
            seq.addBlock(evDelayTE);
            seq.addBlock(mr.rotate('z', phi, gx));  % no ADC
            seq.addBlock(mr.rotate('z', phi, gxSpoil, gzSpoil, evDelayTR));
        end
    end
    seq.addBlock(lblOnce0);  % end prep

    % --- main: imaging spokes ---
    for i = 1:Nr
        phi = delta * (i - 1);
        rf.phaseOffset  = rf_phase / 180 * pi;
        adc.phaseOffset = rf_phase / 180 * pi;
        rf_inc   = mod(rf_inc + rfSpoilInc, 360.0);
        rf_phase = mod(rf_phase + rf_inc, 360.0);

        for s = 1:Nslices
            rf.freqOffset = gz.amplitude * thick * (s - 1 - (Nslices-1)/2);
            seq.addBlock(rf, gz);
            seq.addBlock(mr.rotate('z', phi, gxPre, gzReph));
            seq.addBlock(evDelayTE);
            seq.addBlock(mr.rotate('z', phi, gx, adc));
            seq.addBlock(mr.rotate('z', phi, gxSpoil, gzSpoil, evDelayTR));
        end
    end

    % --- cooldown (ONCE=2) ---
    seq.addBlock(lblOnce2);

    seq.setDefinition('FOV', [fov, fov, thick * Nslices]);
    seq.setDefinition('Name', 'gre_rad');

    fname = seq_filename('gre_rad', num_slices);
    check_and_write(seq, fname, fov, thick, num_slices, 1);
end


%% ========================================================================
%  Radial GRE with rotation extension (uses mr.makeRotation)
%  ========================================================================

function write_radial_rotext(num_slices)
    fprintf('Generating Radial GRE rotext (%d slice) ...\n', num_slices);

    sys = make_system();
    seq = mr.Sequence(sys);

    fov     = 260e-3;
    Nx      = 320;
    alpha   = 10;
    thick   = 3e-3;
    TE      = 8e-3;
    TR      = 20e-3;
    Nr      = 256;
    Nslices = num_slices;
    Ndummy  = 20;
    delta   = pi / Nr;
    rfSpoilInc = 84;

    % --- events ---
    [rf, gz] = mr.makeSincPulse(alpha * pi / 180, 'Duration', 4e-3, ...
        'SliceThickness', thick, 'apodization', 0.5, 'timeBwProduct', 4, ...
        'system', sys, 'use', 'excitation');

    deltak  = 1 / fov;
    gx      = mr.makeTrapezoid('x', 'FlatArea', Nx * deltak, ...
                               'FlatTime', 6.4e-3/5, 'system', sys);
    adc     = mr.makeAdc(Nx, 'Duration', gx.flatTime, ...
                         'Delay', gx.riseTime, 'system', sys);
    gxPre   = mr.makeTrapezoid('x', 'Area', -gx.area/2 - deltak/2, ...
                               'Duration', 2e-3, 'system', sys);
    gzReph  = mr.makeTrapezoid('z', 'Area', -gz.area/2, ...
                               'Duration', 2e-3, 'system', sys);
    gxSpoil = mr.makeTrapezoid('x', 'Area', 0.5 * Nx * deltak, 'system', sys);
    gzSpoil = mr.makeTrapezoid('z', 'Area', 4 / thick, 'system', sys);

    delayTE = ceil((TE - mr.calcDuration(gxPre) - gz.fallTime ...
              - gz.flatTime/2 - mr.calcDuration(gx)/2) ...
              / seq.gradRasterTime) * seq.gradRasterTime;
    delayTR = ceil((TR - mr.calcDuration(gxPre) - mr.calcDuration(gz) ...
              - mr.calcDuration(gx) - delayTE) ...
              / seq.gradRasterTime) * seq.gradRasterTime;
    assert(delayTR >= mr.calcDuration(gxSpoil, gzSpoil), 'TR too short');
    evDelayTE = mr.makeDelay(delayTE);
    evDelayTR = mr.makeDelay(delayTR);

    lblOnce1 = mr.makeLabel('SET', 'ONCE', 1);
    lblOnce0 = mr.makeLabel('SET', 'ONCE', 0);
    lblOnce2 = mr.makeLabel('SET', 'ONCE', 2);

    rf_phase = 0;
    rf_inc   = 0;

    % --- prep: dummy spokes (ONCE=1) ---
    seq.addBlock(lblOnce1);
    for i = 1:Ndummy
        phi = delta * (i - 1);
        rot = mr.makeRotation(phi);
        rf.phaseOffset = rf_phase / 180 * pi;
        rf_inc   = mod(rf_inc + rfSpoilInc, 360.0);
        rf_phase = mod(rf_phase + rf_inc, 360.0);

        for s = 1:Nslices
            rf.freqOffset = gz.amplitude * thick * (s - 1 - (Nslices-1)/2);
            seq.addBlock(rf, gz);
            seq.addBlock(gxPre, gzReph, rot);
            seq.addBlock(evDelayTE);
            seq.addBlock(gx, rot);           % no ADC
            seq.addBlock(gxSpoil, gzSpoil, evDelayTR, rot);
        end
    end
    seq.addBlock(lblOnce0);  % end prep

    % --- main: imaging spokes ---
    for i = 1:Nr
        phi = delta * (i - 1);
        rot = mr.makeRotation(phi);
        rf.phaseOffset  = rf_phase / 180 * pi;
        adc.phaseOffset = rf_phase / 180 * pi;
        rf_inc   = mod(rf_inc + rfSpoilInc, 360.0);
        rf_phase = mod(rf_phase + rf_inc, 360.0);

        for s = 1:Nslices
            rf.freqOffset = gz.amplitude * thick * (s - 1 - (Nslices-1)/2);
            seq.addBlock(rf, gz);
            seq.addBlock(gxPre, gzReph, rot);
            seq.addBlock(evDelayTE);
            seq.addBlock(gx, adc, rot);
            seq.addBlock(gxSpoil, gzSpoil, evDelayTR, rot);
        end
    end

    % --- cooldown (ONCE=2) ---
    seq.addBlock(lblOnce2);

    seq.setDefinition('FOV', [fov, fov, thick * Nslices]);
    seq.setDefinition('Name', 'gre_rad_rotext');

    fname = seq_filename('gre_rad_rotext', num_slices);
    check_and_write(seq, fname, fov, thick, num_slices, 1);
end


