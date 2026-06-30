%% generate_test_sequences.m
%
% Iterative rebuild of segmentation test generation.
% Phase 1 scope:
%   - Build one basic GRE 2D case
%   - Export minimal segmentation-focused truth
%   - Keep placeholders for later TR/safety/freqmod truth

function generate_test_sequences()

clear; clc;

% Add testutils to path
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

% Add Pulseq MATLAB library to path
pulseq_path = fullfile(fileparts(mfilename('fullpath')), '..', 'pulseq', 'matlab');
if isfolder(pulseq_path) || isdir(pulseq_path)
    addpath(pulseq_path);
    fprintf('Added Pulseq path: %s\n', pulseq_path);
else
    error('Pulseq MATLAB library not found at: %s', pulseq_path);
end

write_bssfp(true, 1, 1);
write_bssfp(true, 3, 1);
write_bssfp(true, 1, 3);
write_bssfp(true, 3, 3);

write_gre(true, 1, 1);
write_gre(true, 3, 1);
write_gre(true, 1, 3);
write_gre(true, 3, 3);

write_fse(true, 1, 1);
write_fse(true, 3, 1);
write_fse(true, 1, 3);
write_fse(true, 3, 3);

write_epi(true, 1, 1);
write_epi(true, 3, 1);
write_epi(true, 1, 3);
write_epi(true, 3, 3);

write_epi_gre(true, 1);
write_epi_gre(true, 3);

write_mprage(true, 1, 1);
write_mprage(true, 3, 1);
write_mprage(true, 1, 3);
write_mprage(true, 3, 3);

write_mprage_nav(true, 1, 1);
write_mprage_nav(true, 3, 1);
write_mprage_nav(true, 1, 3);
write_mprage_nav(true, 3, 3);

write_mprage_noncart(true, 1, 1, false);
write_mprage_noncart(true, 3, 1, false);
write_mprage_noncart(true, 1, 3, false);
write_mprage_noncart(true, 3, 3, false);

write_mprage_noncart(true, 1, 1, true);
write_mprage_noncart(true, 3, 1, true);
write_mprage_noncart(true, 1, 3, true);
write_mprage_noncart(true, 3, 3, true);

write_qalas_noncart(true, 1, 1, true);
write_qalas_noncart(true, 3, 1, true);
write_qalas_noncart(true, 1, 3, true);
write_qalas_noncart(true, 3, 3, true);

end


function seq = write_bssfp(write, num_slices, num_averages)
    base = sprintf('bssfp_2d_%dsl_%davg', num_slices, num_averages);
    fprintf('Generating sequence: %s\n', base);

    sys = make_system();

    % Basic GRE geometry intentionally small for fast iteration.
    fov = 0.22;
    Nx = 64;
    Ny = 8;
    slice_thickness = 5e-3;

    alpha = 10 * pi / 180;

    % --- create events ---
    [rf, gz, gz_reph] = mr.makeSincPulse(alpha, sys, ...
        'Duration', 2.0e-3, 'SliceThickness', slice_thickness, ...
        'apodization', 0.5, 'timeBwProduct', 1.5, ...
        'use', 'excitation');

    deltak = 1 / fov;
    gx = mr.makeTrapezoid('x', sys, ...
        'FlatArea', Nx * deltak, ...
        'FlatTime', Nx * 4e-6);
    adc = mr.makeAdc(Nx, sys, ...
        'Duration', gx.flatTime, ...
        'Delay', gx.riseTime);
    gx_pre = mr.makeTrapezoid('x', sys, 'Area', -gx.area / 2);
    phase_areas = ((0:Ny-1) - Ny/2) * deltak;

    % --- split & combine for bSSFP optimal timing ---
    gz_parts = testutils.splitGradientAt(gz, mr.calcDuration(rf));
    gz_parts(1).delay = mr.calcDuration(gz_reph);
    gz_1 = testutils.addGradients({gz_reph, gz_parts(1)}, sys);
    [rf, ~] = mr.align('right', rf, gz_1);
    gz_parts(2).delay = 0;
    gz_reph.delay = mr.calcDuration(gz_parts(2));
    gz_2 = testutils.addGradients({gz_parts(2), gz_reph}, sys);

    gx_parts = testutils.splitGradientAt(gx, ...
        ceil(mr.calcDuration(adc) / sys.gradRasterTime) * sys.gradRasterTime);
    gx_parts(1).delay = mr.calcDuration(gx_pre);
    gx_1 = testutils.addGradients({gx_pre, gx_parts(1)}, sys);
    adc.delay = adc.delay + mr.calcDuration(gx_pre);
    gx_parts(2).delay = 0;
    gx_pre.delay = mr.calcDuration(gx_parts(2));
    gx_2 = testutils.addGradients({gx_parts(2), gx_pre}, sys);

    pe_dur = mr.calcDuration(gx_2);

    gz_1.delay = max(mr.calcDuration(gx_2) - rf.delay + rf.ringdownTime, 0);
    rf.delay = rf.delay + gz_1.delay;

    TR = mr.calcDuration(gz_1) + mr.calcDuration(gx_1);

    % --- phase-encode template (max area, will be scaled) ---
    max_pe_area = max(abs(phase_areas));
    gyMax = mr.makeTrapezoid('y', sys, 'Area', max_pe_area, 'Duration', pe_dur);

    % --- pre-create labels ---
    lblOnce1 = mr.makeLabel('SET', 'ONCE', 1);
    lblOnce0 = mr.makeLabel('SET', 'ONCE', 0);
    lblOnce2 = mr.makeLabel('SET', 'ONCE', 2);

    % --- prep RF: half flip angle ---
    rf05 = rf;
    rf05.signal = 0.5 * rf.signal;

    % --- prep delay to center main acquisition around TR/2 ---
    prep_delay = mr.makeDelay(0.0);
%     prep_delay = mr.makeDelay( round((TR/2 - mr.calcDuration(gz_1)) / sys.gradRasterTime) * sys.gradRasterTime);
    gx_1_1 = testutils.makeExtendedTrapezoidArea('x', 0, gx_2.first, -gx_2.area, sys);
    gy_pre_2 = mr.scaleGrad(gyMax, phase_areas(end) / max_pe_area);
    [prep_delay, gz_2, gy_pre_2, gx_1_1] = mr.align('left', prep_delay, gz_2, gy_pre_2, 'right', gx_1_1);

    seq = mr.Sequence(sys);

    for z = 1:num_slices
        rf05.freqOffset = gz.amplitude * slice_thickness * (z - 1 - (num_slices-1)/2);
        rf.freqOffset = gz.amplitude * slice_thickness * (z - 1 - (num_slices-1)/2);

        % --- alpha/2 prep (ONCE=1) ---
        seq.addBlock(rf05, gz_1, lblOnce1);
        seq.addBlock(prep_delay, gz_2, gy_pre_2, gx_1_1);

        % --- main loop ---
        for i = 1:Ny
            rf.phaseOffset = pi * mod(i, 2);
            adc.phaseOffset = pi * mod(i, 2);

            gy_pre_1 = mr.scaleGrad(gy_pre_2, -1);             % undo previous PE
            gy_pre_2 = mr.scaleGrad(gyMax, phase_areas(i) / max_pe_area);  % new PE

            if i == 1
                seq.addBlock(rf, gz_1, gy_pre_1, gx_2, lblOnce0); % clear ONCE flag -> first main block
            else
                seq.addBlock(rf, gz_1, gy_pre_1, gx_2);
            end
            seq.addBlock(gx_1, gy_pre_2, gz_2, adc);
        end

        % --- exit block (ONCE=2) ---
        seq.addBlock(gx_2, lblOnce2);
    end

    seq.setDefinition('FOV', [fov fov slice_thickness * num_slices]);
    seq.setDefinition('NumSlices', num_slices);

    [ok, err] = seq.checkTiming;
    if ~ok
        error('Timing check failed:\n%s', strjoin(err, '\n'));
    end

    if write == false
        return;
    end

    out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'expected');

    tb = testutils.TruthBuilder(seq, sys);
    tb.setBlocksPerTR(3 + 2*Ny);
    tb.setSegments([3 + 2*Ny]);
    tb.setSegmentOrder([1]);
    tb.setNumAverages(num_averages);
    tb.export(out_dir, base);
end


function seq = write_gre(write, num_slices, num_averages)
    base = sprintf('gre_2d_%dsl_%davg', num_slices, num_averages);
    fprintf('Generating sequence: %s\n', base);

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
    [rf, gz] = mr.makeSincPulse(alpha, sys, ...
        'Duration', 2.0e-3, ...
        'SliceThickness', slice_thickness, ...
        'timeBwProduct', 4, ...
        'apodization', 0.5, ...
        'use', 'excitation');
    gz_reph = mr.makeTrapezoid('z', sys, 'Area', -gz.area/2, 'Duration', 1.0e-3);
    gz_spoil = mr.makeTrapezoid('z', sys, 'Area', 4 / slice_thickness, 'Duration', 1.0e-3);

    % Readout and ADC
    readout_time = 2.56e-3;
    gx_full = mr.makeTrapezoid('x', sys, 'FlatArea', Nx/fov, 'FlatTime', readout_time);
    gx_parts = testutils.splitGradientAt(gx_full, gx_full.riseTime + gx_full.flatTime);
    gx = gx_parts(1); % truncate at end of flat
    adc = mr.makeAdc(Nx, sys, 'Duration', gx_full.flatTime, 'Delay', gx_full.riseTime);
    dummy_adc = mr.makeDelay(mr.calcDuration(adc));

    % Pre/rewinder templates
    gx_pre = mr.makeTrapezoid('x', sys, 'Area', -gx_full.area/2, 'Duration', 1.0e-3);
    gx_spoil_area = 4 / slice_thickness;

    % Bridged spoiler that starts at gx flat amplitude for continuity.
    gx_spoil = testutils.makeExtendedTrapezoidArea('x', gx_full.amplitude, 0, gx_spoil_area, sys);

    pe_areas = ((0:Ny-1) - floor(Ny/2)) / fov;
    max_pe_area = max(abs(pe_areas));
    gy_phase = mr.makeTrapezoid('y', sys, 'Area', max_pe_area, 'Duration', 1.0e-3);

    % ONCE labels for prep/main semantics.
    lbl_once1 = mr.makeLabel('SET', 'ONCE', 1);
    lbl_once0 = mr.makeLabel('SET', 'ONCE', 0);

    seq = mr.Sequence(sys);

    rf_center = mr.calcRfCenter(rf);
    rf_phase = 0.0;
    rf_inc = 0.0;

    % Dummy TRs (once region): same 4-block TR shape with PE=0.
    for d = 1:ndummy
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

        rf_inc = mod(rf_inc + rf_spoil_inc, 360.0);
    end

    % Main imaging loop: PE -> slices (slice is inner loop).
    first_main = true;
    for pe = 1:Ny
        if max_pe_area > 0
            yscale = pe_areas(pe) / max_pe_area;
        else
            yscale = 0;
        end
        for sl = 1:num_slices
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

            rf_inc = mod(rf_inc + rf_spoil_inc, 360.0);
        end
    end

    seq.setDefinition('FOV', [fov fov slice_thickness * num_slices]);
    seq.setDefinition('NumSlices', num_slices);

    [ok, err] = seq.checkTiming;
    if ~ok
        error('Timing check failed:\n%s', strjoin(err, '\n'));
    end

    if write == false
        return;
    end

    out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'expected');

    tb = testutils.TruthBuilder(seq, sys);
    tb.setBlocksPerTR(4);
    tb.setSegments([4]);
    tb.setSegmentOrder([1]);
    tb.setNumAverages(num_averages);
    tb.export(out_dir, base);
end


function seq = write_fse(write, num_slices, num_averages)
    base = sprintf('fse_2d_%dsl_%davg', num_slices, num_averages);
    fprintf('Generating sequence: %s\n', base);

    sys = make_system(true);

    fov = 256e-3;
    Nx = 256;
    Ny = 8;
    necho   = 16;
    Nslices = num_slices;
    slice_thickness = 5e-3;

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
        'Duration', tEx, 'SliceThickness', slice_thickness, ...
        'apodization', 0.5, 'timeBwProduct', 4, ...
        'PhaseOffset', rfex_phase, 'use', 'excitation');

    flipref = rflip(1) * pi / 180;
    [rfref, ~] = mr.makeSincPulse(flipref, sys, ...
        'Duration', tRef, 'SliceThickness', slice_thickness, ...
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
    nex = max(1, floor(Ny / necho));
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
    TRfill = 0.1e-3;  % override for short TR test (no fill, but still valid since echoes fit within TR)
    delayTR = mr.makeDelay(TRfill);

    % --- labels ---
    lblOnce1 = mr.makeLabel('SET', 'ONCE', 1);
    lblOnce0 = mr.makeLabel('SET', 'ONCE', 0);

    seq = mr.Sequence(sys);

    % --- main imaging loop ---
    for kex = 0:nex
        for s = 1:Nslices
            rfex.freqOffset  = GSex.amplitude * slice_thickness * (s - 1 - (Nslices-1)/2);
            rfref.freqOffset = GSref.amplitude * slice_thickness * (s - 1 - (Nslices-1)/2);
            rfex.phaseOffset  = rfex_phase - 2*pi * rfex.freqOffset * mr.calcRfCenter(rfex);
            rfref.phaseOffset = rfref_phase - 2*pi * rfref.freqOffset * mr.calcRfCenter(rfref);

            if kex == 0 && s == 1
                seq.addBlock(GS1, lblOnce1);  % start of prep (dummy excitation) -> Block 1
            elseif kex == Ndummy && s == 1
                seq.addBlock(GS1, lblOnce0);  % end prep, start of main -> Block 1
            else
                seq.addBlock(GS1); % Block 1
            end
            seq.addBlock(GS2, rfex); % Block 2
            seq.addBlock(GS3, GR3); % Block 3

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

                seq.addBlock(GS4, rfref); % Block 3 + 4 * (kech-1) + 1
                seq.addBlock(GS5, GR5, GPpre); % Block 3 + 4 * (kech-1) + 2
                if kex > 0
                    seq.addBlock(GR6, adc); % Block 3 + 4 * (kech-1) + 3
                else
                    seq.addBlock(GR6); % Block 3 + 4 * (kech-1) + 3
                end
                seq.addBlock(GS7, GR7, GPrew); % Block 3 + 4 * (kech-1) + 4
            end

            seq.addBlock(GS4); % Block 3 + 4 * necho + 1
            seq.addBlock(GS5); % Block 3 + 4 * necho + 2
            seq.addBlock(delayTR); % Block 3 + 4 * necho + 3
        end
    end

    [ok, err] = seq.checkTiming;
    if ~ok
        error('Timing check failed:\n%s', strjoin(err, '\n'));
    end

    if write == false
        return;
    end

    out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'expected');

    tb = testutils.TruthBuilder(seq, sys);
    tb.setBlocksPerTR(3 + 4 * necho + 3);
    tb.setSegments([3 + 4 * necho + 2, 1]);
    tb.setSegmentOrder([1, 2]);
    tb.setNumAverages(num_averages);
    tb.export(out_dir, base);
end


function seq = write_epi(write, num_slices, num_averages)
    base = sprintf('epi_2d_%dsl_%davg', num_slices, num_averages);
    fprintf('Generating sequence: %s\n', base);

    sys = make_system();

    % Basic EPI geometry intentionally small for fast iteration.
    fov = 0.22;
    Nx = 64;
    Ny = 128;
    slice_thickness = 5e-3;
    slice_gap  = 1.5e-3;
    TR        = 3000e-3;
    ro_os     = 2;
    readoutTime = 580e-6;
    partFourierFactor = 1;
    Nnav      = 3;          % navigator echoes
    pe_enable = 1;

     % Fat-sat pulse
    sat_ppm = -3.35;
    rf_fs = mr.makeGaussPulse(110 * pi / 180, sys, ...
        'Duration', 8e-3, ...
        'bandwidth', abs(sat_ppm * 1e-6 * sys.B0 * sys.gamma), ...
        'freqPPM', sat_ppm, 'use', 'saturation');
    rf_fs.phasePPM = -2*pi * rf_fs.freqPPM * rf_fs.center;
    gz_fs = mr.makeTrapezoid('z', sys, 'delay', mr.calcDuration(rf_fs), 'Area', 0.1 / 1e-4);

    % Excitation
    [rf, gz, gzReph] = mr.makeSincPulse(pi / 2, sys, ...
        'Duration', 2e-3, 'SliceThickness', slice_thickness, ...
        'apodization', 0.42, 'timeBwProduct', 4, 'use', 'excitation');

    trig = mr.makeDigitalOutputPulse('osc0', 'duration', 100e-6);

    % Readout gradient
    deltak = 1 / fov;
    blip_dur = ceil(2 * sqrt(deltak / sys.maxSlew) / sys.gradRasterTime / 2)  * sys.gradRasterTime * 2;
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
    gy_parts = testutils.splitGradientAt(gy, blip_dur / 2, sys);
    [gy_blipup, gy_blipdown, ~] = mr.align('right', gy_parts(1), 'left', gy_parts(2), gx);
    gy_blipdownup = testutils.addGradients({gy_blipdown, gy_blipup}, sys);

    gy_blipup.waveform     = gy_blipup.waveform * pe_enable;
    gy_blipdown.waveform   = gy_blipdown.waveform * pe_enable;
    gy_blipdownup.waveform = gy_blipdownup.waveform * pe_enable;

    % Phase encoding
    Ny_pre  = round(partFourierFactor * Ny / 2 - 1);
    Ny_post = round(Ny / 2 + 1);
    Ny_meas = Ny_pre + Ny_post;
    Nslices = num_slices;

    % Pre-phasing
    gxPre = mr.makeTrapezoid('x', sys, 'Area', -gx.area / 2);
    gyPre = mr.makeTrapezoid('y', sys, 'Area', Ny_pre * deltak);
    [gxPre, gyPre, gzReph] = mr.align('right', gxPre, 'left', gyPre, gzReph);
    gyPre = mr.makeTrapezoid('y', sys, 'Area', gyPre.area, ...
        'Duration', mr.calcDuration(gxPre, gyPre, gzReph));
    gyPre.amplitude = gyPre.amplitude * pe_enable;

    % Slice positions (interleaved)
    slicePositions = (slice_thickness + slice_gap) * ((0:(Nslices-1)) - (Nslices-1)/2);
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
    lblNoPos1 = mr.makeLabel('SET', 'NOPOS', 1);
    lblNoPos0 = mr.makeLabel('SET', 'NOPOS', 0);
    lblNoRot1 = mr.makeLabel('SET', 'NOROT', 1);
    lblNoRot0 = mr.makeLabel('SET', 'NOROT', 0);

    % Hardcoded volume number
    NDummyVolumes = 1; % in fMRI, dummy scans to reach steady state
    NVolumes = 1; % number of volumes to sample hemodynamics
    NSlices = num_slices; % number of slices per volume

    seq = mr.Sequence(sys);

    % --- prep (ONCE=1): one dummy volume ---
    % Structure mirrors main section (same block defs & ordering) so the C
    % library detects degenerate prep.  ADC is omitted because it does not
    % contribute to the block definition key; dummy data is not acquired.
    for v = 1:NDummyVolumes
        for s = 1:Nslices

            % Fat saturation
            if v == 1 && s == 1
                seq.addBlock(rf_fs, gz_fs, lblOnce1, lblSetSlc, lblNoPos1, lblNoRot1);
            elseif s == 1
                seq.addBlock(rf_fs, gz_fs, lblSetSlc, lblNoPos1, lblNoRot1);
            else
                seq.addBlock(rf_fs, gz_fs, lblNoPos1, lblNoRot1);
            end
            rf.freqOffset  = gz.amplitude * slicePositions(s);

            % Excitation
            seq.addBlock(rf, gz, trig, lblNoPos0, lblNoRot0);

            % Phase correction navigators
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

            % Phase encoding prephaser
            seq.addBlock(mr.scaleGrad(gyPre, 0.0), ...
                mr.makeLabel('SET', 'LIN', -1), ...
                mr.makeLabel('SET', 'NAV', 0), ...
                mr.makeLabel('SET', 'AVG', 0));

            % Readout train
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

            % Pause
            seq.addBlock(lblIncSlc, TRdelay_perSlice);
        end
    end

    % --- main: imaging volume ---
    for v = 1:NVolumes
        for s = 1:Nslices
            % Fat saturation (Segment 0)
            if v == 1 && s == 1
                seq.addBlock(rf_fs, gz_fs, lblOnce0, lblSetSlc, lblNoPos1, lblNoRot1); % Block 0
            elseif s == 1
                seq.addBlock(rf_fs, gz_fs, lblSetSlc, lblNoPos1, lblNoRot1); % Block 0
            else
                seq.addBlock(rf_fs, gz_fs, lblNoPos1, lblNoRot1); % Block 0
            end

            % Excitation (Segment 1)
            rf.freqOffset  = gz.amplitude * slicePositions(s);
            rf.phaseOffset = -2*pi * rf.freqOffset * rf.center;
            seq.addBlock(rf, gz, trig, lblNoPos0, lblNoRot0); % Block 1

            % Navigators for phase correction
            gxPre_nav = mr.scaleGrad(gxPre, -1);
            gx_tmp    = mr.scaleGrad(gx, -1);
            seq.addBlock(gxPre_nav, gzReph, ...
                mr.makeLabel('SET', 'NAV', 1), ...
                mr.makeLabel('SET', 'LIN', floor(Ny/2))); % Block 2

            for n = 1:Nnav
                seq.addBlock( ...
                    gx_tmp, adc, ...
                    mr.makeLabel('SET', 'REV', sign(gx_tmp.amplitude) ~= ROpolarity), ...
                    mr.makeLabel('SET', 'SEG', sign(gx_tmp.amplitude) ~= ROpolarity), ...
                    mr.makeLabel('SET', 'AVG', n == Nnav)); % Block 3..(3+Nnav-1)
                gx_tmp = mr.scaleGrad(gx_tmp, -1);
            end

            % Phase encoding prephaser
            seq.addBlock(gyPre, ...
                mr.makeLabel('SET', 'LIN', -1), ...
                mr.makeLabel('SET', 'NAV', 0), ...
                mr.makeLabel('SET', 'AVG', 0)); % Block 3+Nnav

            % Readout train
            for i = 1:Ny_meas
                lrev = mr.makeLabel('SET', 'REV', sign(gx.amplitude) ~= ROpolarity);
                lseg = mr.makeLabel('SET', 'SEG', sign(gx.amplitude) ~= ROpolarity);
                llin = mr.makeLabel('INC', 'LIN', 1);

                if i == 1
                    seq.addBlock(gx, gy_blipup, adc, lrev, lseg, llin); % Block (4+Nnav)..(4+Nnav+Ny_meas-1)
                elseif i == Ny_meas
                    seq.addBlock(gx, gy_blipdown, adc, lrev, lseg, llin); % Block (4+Nnav)..(4+Nnav+Ny_meas-1)
                else
                    seq.addBlock(gx, gy_blipdownup, adc, lrev, lseg, llin); % Block (4+Nnav)..(4+Nnav+Ny_meas-1)
                end
                gx = mr.scaleGrad(gx, -1);
            end

            if sign(gx.amplitude) ~= ROpolarity
                gx = mr.scaleGrad(gx, -1);
            end

            % Pause (Segment 2)
            seq.addBlock(lblIncSlc, TRdelay_perSlice); % Block (4+Nnav+Ny_meas)
        end
    end

    % Definitions
    seq.setDefinition('Name', 'epi');
    seq.setDefinition('SlicePositions', slicePositions);
    seq.setDefinition('SliceThickness', slice_thickness);
    seq.setDefinition('SliceGap', slice_gap);
    seq.setDefinition('ReadoutOversamplingFactor', ro_os);

    [ok, err] = seq.checkTiming;
    if ~ok
        error('Timing check failed:\n%s', strjoin(err, '\n'));
    end

    if write == false
        return;
    end

    out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'expected');

    tb = testutils.TruthBuilder(seq, sys);
    tb.setBlocksPerTR(5 + Nnav + Ny_meas);
    tb.setSegments([4 + Nnav + Ny_meas, 1]);
    tb.setSegmentOrder([1, 2]);
    tb.setNumAverages(num_averages);
    tb.export(out_dir, base);
end


function seqs = write_epi_gre(write, num_averages)
    base = sprintf('gre_epi_collection_2d_1sl_%davg', num_averages);
    fprintf('Generating sequence collection: %s\n', base);

    gre_seq = write_gre(false, 1, num_averages);
    epi_seq = write_epi(false, 1, num_averages);
    seqs = {gre_seq, epi_seq};

    if write == false
        return;
    end

    out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'expected');
    tmp_dir = tempname;
    mkdir(tmp_dir);
    cleanup = onCleanup(@() rmdir(tmp_dir, 's'));

    gre_seq.setDefinition('IgnoreAverages', 1);
    gre_seq.setDefinition('NextSequence', [base '_epi.seq']);

    gre_tb = testutils.TruthBuilder(gre_seq, make_system());
    gre_tb.setBlocksPerTR(4);
    gre_tb.setSegments([4]);
    gre_tb.setSegmentOrder([1]);
    gre_tb.setNumAverages(1);
    gre_tb.export(tmp_dir, [base '_gre_part']);

    epi_tb = testutils.TruthBuilder(epi_seq, make_system());
    Nnav_epi = 3; Ny_meas_epi = 128;
    nbt_epi = 5 + Nnav_epi + Ny_meas_epi;  % 136
    epi_tb.setBlocksPerTR(nbt_epi);
    epi_tb.setSegments([4 + Nnav_epi + Ny_meas_epi, 1]);
    epi_tb.setSegmentOrder([1, 2]);
    epi_tb.setNumAverages(num_averages);
    epi_tb.export(tmp_dir, [base '_epi_part']);

    gre_seq.write(fullfile(out_dir, [base '.seq']));
    epi_seq.write(fullfile(out_dir, [base '_epi.seq']));

    gre_truth = testutils.truth_parse_case(fullfile(tmp_dir, [base '_gre_part']));
    epi_truth = testutils.truth_parse_case(fullfile(tmp_dir, [base '_epi_part']));
    testutils.exportCollectionTruth(out_dir, base, {gre_truth, epi_truth});

        % Merge per-part Section 5 truths into one collection seq_desc.truth.
    testutils.mergeSeqDescBinaries( ...
        fullfile(out_dir, [base '_seq_desc.truth']), ...
        { fullfile(tmp_dir, [base '_gre_part_seq_desc.truth']), ...
          fullfile(tmp_dir, [base '_epi_part_seq_desc.truth']) });
end


function seq = write_mprage(write, num_slices, num_averages)
    base = sprintf('mprage_2d_%dsl_%davg', num_slices, num_averages);
    fprintf('Generating sequence: %s\n', base);

    sys = make_system();

    % Basic GRE geometry intentionally small for fast iteration.
    fov = 0.22;
    Nx = 64;
    Ny = 8;
    slice_thickness = 5e-3;

    alpha = 10 * pi / 180;
    rf_spoil_inc = 84.0; % degrees

    % Inversion
    % RF and slice-select
    rf180 = mr.makeBlockPulse(pi, sys, ...
        'Duration', 20.0e-3, ...
        'use', 'inversion');
    delayTI = mr.makeDelay(0.1e-3);

    % RF and slice-select
    [rf, gz] = mr.makeSincPulse(alpha, sys, ...
        'Duration', 2.0e-3, ...
        'SliceThickness', slice_thickness, ...
        'timeBwProduct', 4, ...
        'apodization', 0.5, ...
        'use', 'excitation');
    gz_reph = mr.makeTrapezoid('z', sys, 'Area', -gz.area/2, 'Duration', 1.0e-3);
    gz_spoil = mr.makeTrapezoid('z', sys, 'Area', 4 / slice_thickness, 'Duration', 1.0e-3);

    % Readout and ADC
    readout_time = 2.56e-3;
    gx_full = mr.makeTrapezoid('x', sys, 'FlatArea', Nx/fov, 'FlatTime', readout_time);
    gx_parts = testutils.splitGradientAt(gx_full, gx_full.riseTime + gx_full.flatTime);
    gx = gx_parts(1); % truncate at end of flat
    adc = mr.makeAdc(Nx, sys, 'Duration', gx_full.flatTime, 'Delay', gx_full.riseTime);

    % Pre/rewinder templates
    gx_pre = mr.makeTrapezoid('x', sys, 'Area', -gx_full.area/2, 'Duration', 1.0e-3);
    gx_spoil_area = 4 / slice_thickness;

    % Bridged spoiler that starts at gx flat amplitude for continuity.
    gx_spoil = testutils.makeExtendedTrapezoidArea('x', gx_full.amplitude, 0, gx_spoil_area, sys);
    delayTR = mr.makeDelay(0.5e-3);

    pe_areas = ((0:Ny-1) - floor(Ny/2)) / fov;
    max_pe_area = max(abs(pe_areas));
    gy_phase = mr.makeTrapezoid('y', sys, 'Area', max_pe_area, 'Duration', 1.0e-3);

    seq = mr.Sequence(sys);

    rf_center = mr.calcRfCenter(rf);
    rf_phase = 0.0;
    rf_inc = 0.0;

    % Main imaging loop: PE -> slices (slice is inner loop).
    for sl = 1:num_slices
        slc_shift = (sl - 1 - (num_slices - 1) / 2);

        seq.addBlock(rf180);
        seq.addBlock(gz_spoil);
        seq.addBlock(delayTI);

        for pe = 1:Ny
            if max_pe_area > 0
                yscale = pe_areas(pe) / max_pe_area;
            else
                yscale = 0;
            end

            rf_phase = mod(rf_phase + rf_inc, 360.0);

            rf_curr = rf;
            rf_curr.freqOffset = gz.amplitude * slice_thickness * slc_shift;
            rf_curr.phaseOffset = rf_phase / 180 * pi - 2 * pi * rf_curr.freqOffset * rf_center;

            adc_curr = adc;
            adc_curr.freqOffset = rf_curr.freqOffset;
            adc_curr.phaseOffset = rf_phase / 180 * pi;

            gy_pre = mr.scaleGrad(gy_phase, yscale);
            gy_rew = mr.scaleGrad(gy_phase, -yscale);

            seq.addBlock(rf_curr, gz);
            seq.addBlock(gx_pre, gy_pre, gz_reph);
            seq.addBlock(gx, adc_curr);
            seq.addBlock(gx_spoil, gy_rew, gz_spoil);

            rf_inc = mod(rf_inc + rf_spoil_inc, 360.0);
        end
        seq.addBlock(delayTR);
    end

    seq.setDefinition('FOV', [fov fov slice_thickness * num_slices]);
    seq.setDefinition('NumSlices', num_slices);

    [ok, err] = seq.checkTiming;
    if ~ok
        error('Timing check failed:\n%s', strjoin(err, '\n'));
    end

    if write == false
        return;
    end

    out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'expected');

    tb = testutils.TruthBuilder(seq, sys);
    tb.setBlocksPerTR(2 + 1 + 4 * Ny + 1);
    tb.setSegments([2, 1, 4]);
    tb.setSegmentOrder([1, 2, 3 * ones(1, Ny), 2]);
    tb.setNumAverages(num_averages);
    tb.export(out_dir, base);
end


function seq = write_mprage_nav(write, num_slices, num_averages)
    base = sprintf('mprage_nav_2d_%dsl_%davg', num_slices, num_averages);
    fprintf('Generating sequence: %s\n', base);

    sys = make_system();

    % Basic GRE geometry intentionally small for fast iteration.
    fov = 0.22;
    Nx = 64;
    Ny = 8;
    slice_thickness = 5e-3;

    alpha = 10 * pi / 180;
    rf_spoil_inc = 84.0; % degrees

    % Inversion
    % RF and slice-select
    rf180 = mr.makeBlockPulse(pi, sys, ...
        'Duration', 20.0e-3, ...
        'use', 'inversion');
    delayTI = mr.makeDelay(0.1e-3);

    % RF and slice-select
    [rf, gz] = mr.makeSincPulse(alpha, sys, ...
        'Duration', 2.0e-3, ...
        'SliceThickness', slice_thickness, ...
        'timeBwProduct', 4, ...
        'apodization', 0.5, ...
        'use', 'excitation');
    gz_reph = mr.makeTrapezoid('z', sys, 'Area', -gz.area/2, 'Duration', 1.0e-3);
    gz_spoil = mr.makeTrapezoid('z', sys, 'Area', 4 / slice_thickness, 'Duration', 1.0e-3);

    % Readout and ADC
    readout_time = 2.56e-3;
    gx_full = mr.makeTrapezoid('x', sys, 'FlatArea', Nx/fov, 'FlatTime', readout_time);
    gx_parts = testutils.splitGradientAt(gx_full, gx_full.riseTime + gx_full.flatTime);
    gx = gx_parts(1); % truncate at end of flat
    adc = mr.makeAdc(Nx, sys, 'Duration', gx_full.flatTime, 'Delay', gx_full.riseTime);

    % Navigator: 3 orthogonal single-shot spirals (16 samples)
    % Each nav has a slice-selective sinc pulse selecting the respective plane.
    Nx_nav = 16;
    nav_slice_thickness = 10e-3;
    [gx_nav_cells, gy_nav_cells, adc_nav] = testutils.makeTestSpiral(sys, 1, Nx_nav, 10.0*fov);

    gx_nav_ax = gx_nav_cells{1}; gx_nav_rew_ax = testutils.makeExtendedTrapezoidArea('x', gx_nav_ax.last, 0.0, -gx_nav_ax.area, sys);
    gy_nav_ax = gy_nav_cells{1}; gy_nav_rew_ax = testutils.makeExtendedTrapezoidArea('y', gy_nav_ax.last, 0.0, -gy_nav_ax.area, sys);
    gz_nav_spoil_ax = mr.makeTrapezoid('z', sys, 'Area', 4 / nav_slice_thickness, 'Duration', 1.0e-3);

    gx_nav_cor = gx_nav_ax; gx_nav_cor.channel = 'x'; gx_nav_rew_cor = testutils.makeExtendedTrapezoidArea('x', gx_nav_cor.last, 0.0, -gx_nav_cor.area, sys);
    gz_nav_cor = gy_nav_ax; gz_nav_cor.channel = 'z'; gz_nav_rew_cor = testutils.makeExtendedTrapezoidArea('z', gz_nav_cor.last, 0.0, -gz_nav_cor.area, sys);
    gy_nav_spoil_cor = mr.makeTrapezoid('y', sys, 'Area', 4 / nav_slice_thickness, 'Duration', 1.0e-3);

    gy_nav_sag = gx_nav_ax; gy_nav_sag.channel = 'y'; gy_nav_rew_sag = testutils.makeExtendedTrapezoidArea('y', gy_nav_sag.last, 0.0, -gy_nav_sag.area, sys);
    gz_nav_sag = gy_nav_ax; gz_nav_sag.channel = 'z'; gz_nav_rew_sag = testutils.makeExtendedTrapezoidArea('z', gz_nav_sag.last, 0.0, -gz_nav_sag.area, sys);
    gx_nav_spoil_sag = mr.makeTrapezoid('x', sys, 'Area', 4 / nav_slice_thickness, 'Duration', 1.0e-3);

    % Axial nav: slice-select on z
    [rf_nav_ax, gz_nav_ss] = mr.makeSincPulse(alpha, sys, ...
        'Duration', 0.5e-3, ...
        'SliceThickness', nav_slice_thickness, ...
        'timeBwProduct', 4, ...
        'apodization', 0.5, ...
        'use', 'excitation');

    % Coronal nav: slice-select on y
    gy_nav_ss = gz_nav_ss; gy_nav_ss.channel = 'y';
    rf_nav_cor = rf_nav_ax;

    % Sagittal nav: slice-select on x
    gx_nav_ss = gz_nav_ss; gx_nav_ss.channel = 'x';
    rf_nav_sag = rf_nav_ax;

    lblNav = mr.makeLabel('SET', 'NAV', 1);

    % Pre/rewinder templates
    gx_pre = mr.makeTrapezoid('x', sys, 'Area', -gx_full.area/2, 'Duration', 1.0e-3);
    gx_spoil_area = 4 / slice_thickness;

    % Bridged spoiler that starts at gx flat amplitude for continuity.
    gx_spoil = testutils.makeExtendedTrapezoidArea('x', gx_full.amplitude, 0, gx_spoil_area, sys);
    delayTR = mr.makeDelay(0.5e-3);

    pe_areas = ((0:Ny-1) - floor(Ny/2)) / fov;
    max_pe_area = max(abs(pe_areas));
    gy_phase = mr.makeTrapezoid('y', sys, 'Area', max_pe_area, 'Duration', 1.0e-3);

    seq = mr.Sequence(sys);

    rf_center = mr.calcRfCenter(rf);
    rf_phase = 0.0;
    rf_inc = 0.0;

    % Main imaging loop: PE -> slices (slice is inner loop).
    for sl = 1:num_slices
        slc_shift = (sl - 1 - (num_slices - 1) / 2);

        seq.addBlock(rf180);
        seq.addBlock(gz_spoil);
        seq.addBlock(delayTI);

        for pe = 1:Ny
            if max_pe_area > 0
                yscale = pe_areas(pe) / max_pe_area;
            else
                yscale = 0;
            end

            rf_phase = mod(rf_phase + rf_inc, 360.0);

            rf_curr = rf;
            rf_curr.freqOffset = gz.amplitude * slice_thickness * slc_shift;
            rf_curr.phaseOffset = rf_phase / 180 * pi - 2 * pi * rf_curr.freqOffset * rf_center;

            adc_curr = adc;
            adc_curr.freqOffset = rf_curr.freqOffset;
            adc_curr.phaseOffset = rf_phase / 180 * pi;

            gy_pre = mr.scaleGrad(gy_phase, yscale);
            gy_rew = mr.scaleGrad(gy_phase, -yscale);

            seq.addBlock(rf_curr, gz);
            seq.addBlock(gx_pre, gy_pre, gz_reph);
            seq.addBlock(gx, adc_curr);
            seq.addBlock(gx_spoil, gy_rew, gz_spoil);
            rf_inc = mod(rf_inc + rf_spoil_inc, 360.0);
        end
        % Navigator: 3 orthogonal single-shot spirals
        % Axial (xy plane): slice-select on z
        seq.addBlock(rf_nav_ax, gz_nav_ss, lblNav);
        seq.addBlock(gx_nav_ax, gy_nav_ax, adc_nav);
        seq.addBlock(gx_nav_rew_ax, gy_nav_rew_ax, gz_nav_spoil_ax);

        % Coronal (xz plane): slice-select on y
        seq.addBlock(rf_nav_cor, gy_nav_ss, lblNav);
        seq.addBlock(gx_nav_cor, gz_nav_cor, adc_nav);
        seq.addBlock(gx_nav_rew_cor, gy_nav_spoil_cor, gz_nav_rew_cor);

        % Sagittal (yz plane): slice-select on x
        seq.addBlock(rf_nav_sag, gx_nav_ss, lblNav);
        seq.addBlock(gy_nav_sag, gz_nav_sag, adc_nav);
        seq.addBlock(gx_nav_spoil_sag, gy_nav_rew_sag, gz_nav_rew_sag);

        seq.addBlock(delayTR);
    end

    seq.setDefinition('FOV', [fov fov slice_thickness * num_slices]);
    seq.setDefinition('NumSlices', num_slices);

    [ok, err] = seq.checkTiming;
    if ~ok
        error('Timing check failed:\n%s', strjoin(err, '\n'));
    end

    if write == false
        return;
    end

    out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'expected');

    tb = testutils.TruthBuilder(seq, sys);
    tb.setBlocksPerTR(2 + 1 + 4 * Ny + 3 * 3 + 1);
    tb.setSegments([2, 1, 4, 3, 3, 3]);
    tb.setSegmentOrder([1, 2, 3 * ones(1, Ny), 4, 5, 6, 2]);
    tb.setNumAverages(num_averages);
    tb.setFreqModBuildMode('tr_scoped');
    tb.anchorPoints.adc = [0.5, 0.0];
    tb.export(out_dir, base);
end


function seq = write_mprage_noncart(write, Nz, num_averages, use_rotext)
    base = sprintf('mprage_noncart_3d_%dsl_%davg_userotext%d', Nz, num_averages, use_rotext);
    fprintf('Generating sequence: %s\n', base);

    sys = make_system();

    % Basic geometry intentionally small for fast iteration.
    fov = 0.22;
    Nx = 64;
    num_shots = 2;
    slab_thickness = 15e-3;

    alpha = 10 * pi / 180;
    rf_spoil_inc = 84.0; % degrees

    % Inversion
    rf180 = mr.makeBlockPulse(pi, sys, ...
        'Duration', 20.0e-3, ...
        'use', 'inversion');
    delayTI = mr.makeDelay(0.1e-3);

    % Excitation
    rf = mr.makeBlockPulse(alpha, sys, ...
        'Duration', 2.0e-3, ...
        'use', 'excitation');
    gz_spoil = mr.makeTrapezoid('z', sys, 'Area', 4 / 5.0e-3, 'Duration', 1.0e-3);

    % --- Spiral readout (multi-interleave) ---
    if use_rotext
        % Rotext path: generate only first interleave; rotation events
        % will rotate it to the other interleave angles.
        [gx_cells, gy_cells, adc] = testutils.makeTestSpiral(sys, num_shots * Nz, Nx, fov);
        gx_base = gx_cells{1};
        gy_base = gy_cells{1};
        gx_rew = testutils.makeExtendedTrapezoidArea('x', gx_base.last, 0.0, -gx_base.area, sys);
        gy_rew = testutils.makeExtendedTrapezoidArea('y', gy_base.last, 0.0, -gy_base.area, sys);
        dphi = 2 * pi / (num_shots  * Nz);  % interleave angular step
    else
        % Explicit path: pre-compute all interleaves across all partitions.
        [gx_shots, gy_shots, adc] = testutils.makeTestSpiral(sys, num_shots * Nz, Nx, fov);
        % Build per-interleave rewinders
        gx_rews = cell(num_shots * Nz, 1);
        gy_rews = cell(num_shots * Nz, 1);
        for ii = 1:(num_shots * Nz)
            gx_rews{ii} = testutils.makeExtendedTrapezoidArea('x', gx_shots{ii}.last, 0.0, -gx_shots{ii}.area, sys);
            gy_rews{ii} = testutils.makeExtendedTrapezoidArea('y', gy_shots{ii}.last, 0.0, -gy_shots{ii}.area, sys);
        end
    end

    % Partition encoding (along z)
    gz_phase = mr.makeTrapezoid('z', sys, 'Area', -Nz / fov / 2);

    delayTR = mr.makeDelay(0.5e-3);

    if Nz > 1
        z_areas = ((0:Nz-1) - floor(Nz/2)) / slab_thickness;
        max_z_area = max(abs(z_areas));
    else
        z_areas = 0;
        max_z_area = 0;
    end

    seq = mr.Sequence(sys);

    rf_center = mr.calcRfCenter(rf);
    rf_phase = 0.0;
    rf_inc = 0.0;
    spoke_idx = 0;

    % Main imaging loop: partitions -> shots
    for z = 1:Nz
        if max_z_area > 0
            zscale = z_areas(z) / max_z_area;
        else
            zscale = 0;
        end

        seq.addBlock(rf180);
        seq.addBlock(gz_spoil);
        seq.addBlock(delayTI);

        for i = 1:num_shots
            rf_phase = mod(rf_phase + rf_inc, 360.0);
            spoke_idx = spoke_idx + 1;

            rf_curr = rf;
            rf_curr.phaseOffset = rf_phase / 180 * pi - 2 * pi * rf_curr.freqOffset * rf_center;

            adc_curr = adc;
            adc_curr.phaseOffset = rf_phase / 180 * pi;

            seq.addBlock(rf_curr);
            if use_rotext
                rot_angle = (spoke_idx-1) * dphi;
                seq.addBlock(adc_curr, gx_base, gy_base, ...
                    mr.scaleGrad(gz_phase, zscale), ...
                    mr.makeRotation('axis', 'z', 'angle', rot_angle));
                seq.addBlock(gx_rew, gy_rew, ...
                    mr.makeRotation('axis', 'z', 'angle', rot_angle), gz_spoil);
            else
                seq.addBlock(adc_curr, gx_shots{spoke_idx}, gy_shots{spoke_idx}, ...
                    mr.scaleGrad(gz_phase, zscale));
                seq.addBlock(gx_rews{spoke_idx}, gy_rews{spoke_idx}, gz_spoil);
            end
            rf_inc = mod(rf_inc + rf_spoil_inc, 360.0);
        end

        seq.addBlock(delayTR);
    end

    seq.setDefinition('FOV', [fov fov slab_thickness]);
    seq.setDefinition('NumPartitions', Nz);

    [ok, err] = seq.checkTiming;
    if ~ok
        error('Timing check failed:\n%s', strjoin(err, '\n'));
    end

    if write == false
        return;
    end

    out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'expected');

    tb = testutils.TruthBuilder(seq, sys);
    tb.setBlocksPerTR(2 + 1 + 3 * num_shots + 1);
    tb.setSegments([2, 1, 3]);
    tb.setSegmentOrder([1, 2, 3 * ones(1, num_shots), 2]);
    tb.setNumAverages(num_averages);
    tb.anchorPoints.adc = 0.0;
    tb.export(out_dir, base);
end


function seq = write_qalas_noncart(write, Nz, num_averages, use_rotext)
    base = sprintf('qalas_noncart_3d_%dsl_%davg_userotext%d', Nz, num_averages, use_rotext);
    fprintf('Generating sequence: %s\n', base);

    sys = make_system();

    % Basic geometry intentionally small for fast iteration.
    fov = 0.22;
    Nx = 64;
    num_shots = 2;
    slab_thickness = 15e-3;

    alpha = 10 * pi / 180;
    rf_spoil_inc = 84.0; % degrees

    % T2-prep
    rf90 = mr.makeBlockPulse(pi/2, sys, ...
        'Duration', 10.0e-3, ...
        'use', 'preparation');
    rf90_inv = mr.makeBlockPulse(pi/2, sys, ...
        'Duration', 10.0e-3, ...
        'use', 'preparation', ...
        'phaseOffset', pi); % 180 degree phase shift for preparation closeout
    t2prep_delay = mr.makeDelay(50e-3);

    % T2-prep refocusing pulse
    rf180_t2prep = mr.makeBlockPulse(pi, sys, ...
        'Duration', 20.0e-3, ...
        'use', 'refocusing');

    % Inversion
    rf180_inv = mr.makeBlockPulse(pi, sys, ...
        'Duration', 20.0e-3, ...
        'use', 'inversion');
    delayTI = mr.makeDelay(0.1e-3);

    % Excitation
    rf = mr.makeBlockPulse(alpha, sys, ...
        'Duration', 2.0e-3, ...
        'use', 'excitation');
    gz_spoil = mr.makeTrapezoid('z', sys, 'Area', 4 / 5.0e-3, 'Duration', 1.0e-3);

    % --- Spiral readout (multi-interleave) ---
    if use_rotext
        % Rotext path: generate only first interleave; rotation events
        % will rotate it to the other interleave angles.
        [gx_cells, gy_cells, adc] = testutils.makeTestSpiral(sys, num_shots * Nz, Nx, fov);
        gx_base = gx_cells{1};
        gy_base = gy_cells{1};
        gx_rew = testutils.makeExtendedTrapezoidArea('x', gx_base.last, 0.0, -gx_base.area, sys);
        gy_rew = testutils.makeExtendedTrapezoidArea('y', gy_base.last, 0.0, -gy_base.area, sys);
        dphi = 2 * pi / (num_shots  * Nz);  % interleave angular step
    else
        % Explicit path: pre-compute all interleaves across all partitions.
        [gx_shots, gy_shots, adc] = testutils.makeTestSpiral(sys, num_shots * Nz, Nx, fov);
        % Build per-interleave rewinders
        gx_rews = cell(num_shots * Nz, 1);
        gy_rews = cell(num_shots * Nz, 1);
        for ii = 1:(num_shots * Nz)
            gx_rews{ii} = testutils.makeExtendedTrapezoidArea('x', gx_shots{ii}.last, 0.0, -gx_shots{ii}.area, sys);
            gy_rews{ii} = testutils.makeExtendedTrapezoidArea('y', gy_shots{ii}.last, 0.0, -gy_shots{ii}.area, sys);
        end
    end

    % Partition encoding (along z)
    gz_phase = mr.makeTrapezoid('z', sys, 'Area', -Nz / fov / 2);

    delayTR = mr.makeDelay(0.5e-3);

    if Nz > 1
        z_areas = ((0:Nz-1) - floor(Nz/2)) / slab_thickness;
        max_z_area = max(abs(z_areas));
    else
        z_areas = 0;
        max_z_area = 0;
    end

    seq = mr.Sequence(sys);

    rf_center = mr.calcRfCenter(rf);
    rf_phase = 0.0;
    rf_inc = 0.0;
    spoke_idx = 0;

    % Main imaging loop: partitions -> shots
    for z = 1:Nz
        if max_z_area > 0
            zscale = z_areas(z) / max_z_area;
        else
            zscale = 0;
        end

        % T2-prep module
        seq.addBlock(rf90);
        seq.addBlock(t2prep_delay);
        seq.addBlock(rf180_t2prep);
        seq.addBlock(t2prep_delay);
        seq.addBlock(rf90_inv);
        seq.addBlock(gz_spoil);

        % Spiral FLASH readout
        rf_phase = mod(rf_phase + rf_inc, 360.0);
        spoke_idx = spoke_idx + 1;

        rf_curr = rf;
        rf_curr.phaseOffset = rf_phase / 180 * pi - 2 * pi * rf_curr.freqOffset * rf_center;

        adc_curr = adc;
        adc_curr.phaseOffset = rf_phase / 180 * pi;

        seq.addBlock(rf_curr);
        if use_rotext
            rot_angle = (spoke_idx-1) * dphi;
            seq.addBlock(adc_curr, gx_base, gy_base, ...
                mr.scaleGrad(gz_phase, zscale), ...
                mr.makeRotation('axis', 'z', 'angle', rot_angle));
            seq.addBlock(gx_rew, gy_rew, ...
                mr.makeRotation('axis', 'z', 'angle', rot_angle), gz_spoil);
        else
            seq.addBlock(adc_curr, gx_shots{spoke_idx}, gy_shots{spoke_idx}, ...
                mr.scaleGrad(gz_phase, zscale));
            seq.addBlock(gx_rews{spoke_idx}, gy_rews{spoke_idx}, gz_spoil);
        end
        rf_inc = mod(rf_inc + rf_spoil_inc, 360.0);

        %  Inversion module
        seq.addBlock(rf180_inv);
        seq.addBlock(gz_spoil);
        seq.addBlock(delayTI);

        % Multiple spiral FLASH shots
        for i = 1:num_shots
            rf_phase = mod(rf_phase + rf_inc, 360.0);
            spoke_idx = spoke_idx + 1;

            rf_curr = rf;
            rf_curr.phaseOffset = rf_phase / 180 * pi - 2 * pi * rf_curr.freqOffset * rf_center;

            adc_curr = adc;
            adc_curr.phaseOffset = rf_phase / 180 * pi;

            seq.addBlock(rf_curr);
            if use_rotext
                rot_angle = (spoke_idx-1) * dphi;
                seq.addBlock(adc_curr, gx_base, gy_base, ...
                    mr.scaleGrad(gz_phase, zscale), ...
                    mr.makeRotation('axis', 'z', 'angle', rot_angle));
                seq.addBlock(gx_rew, gy_rew, ...
                    mr.makeRotation('axis', 'z', 'angle', rot_angle), gz_spoil);
            else
                seq.addBlock(adc_curr, gx_shots{spoke_idx}, gy_shots{spoke_idx}, ...
                    mr.scaleGrad(gz_phase, zscale));
                seq.addBlock(gx_rews{spoke_idx}, gy_rews{spoke_idx}, gz_spoil);
            end
            rf_inc = mod(rf_inc + rf_spoil_inc, 360.0);
        end

        seq.addBlock(delayTR);
    end

    seq.setDefinition('FOV', [fov fov slab_thickness]);
    seq.setDefinition('NumPartitions', Nz);

    [ok, err] = seq.checkTiming;
    if ~ok
        error('Timing check failed:\n%s', strjoin(err, '\n'));
    end

    if write == false
        return;
    end

    out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'expected');

    tb = testutils.TruthBuilder(seq, sys);
    tb.setBlocksPerTR(6 + 3 + 2 + 1 + 3 * num_shots + 1);
    tb.setSegments([6, 3, 2, 1]);
    tb.setSegmentOrder([1, 2, 3, 4, 2 * ones(1, num_shots), 4]);
    tb.setNumAverages(num_averages);
    tb.anchorPoints.adc = 0.0;
    tb.export(out_dir, base);
end


function sys = make_system(fse_flag)
    if nargin < 1
        fse_flag = false;
    end
    if fse_flag
        sys = mr.opts( ...
            'MaxGrad',   28,   'GradUnit', 'mT/m', ...
            'MaxSlew',   200,  'SlewUnit', 'T/m/s', ...
            'B0', 3.0, ...
            'rfRasterTime', 2e-6, ...
            'gradRasterTime', 20e-6, ...
            'adcRasterTime', 2e-6, ...
            'blockDurationRaster', 20e-6, ...
            'rfRingdownTime', 20e-6, ...
            'rfDeadTime', 100e-6, ...
            'adcDeadTime', 10e-6);
    else
        sys = mr.opts( ...
            'MaxGrad',   28,   'GradUnit', 'mT/m', ...
            'MaxSlew',   200,  'SlewUnit', 'T/m/s', ...
            'B0', 3.0, ...
            'rfRasterTime', 2e-6, ...
            'gradRasterTime', 20e-6, ...
            'adcRasterTime', 2e-6, ...
            'blockDurationRaster', 20e-6);
    end
end
