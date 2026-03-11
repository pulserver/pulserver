%% generate_test_sequences.m
%
% Iterative rebuild of segmentation test generation.
% Phase 1 scope:
%   - Build one basic GRE 2D case
%   - Export minimal segmentation-focused truth
%   - Keep placeholders for later TR/safety/freqmod truth

clear; clc;
import mr.*

write_gre(true, 1, 1);
write_gre(true, 3, 1);
write_gre(true, 1, 3);
write_gre(true, 3, 3);

write_mprage(true, 1, 1);


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

    out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');

    tb = TruthBuilder(seq, sys);
    tb.setBlocksPerTR(4);
    tb.setSegments([4], [1]);
    tb.setNumAverages(num_averages);
    tb.export(out_dir, base);
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
    rf180 = mr.makeBlockPulse(pi, ...
        'Duration', 20.0e-3, ...
        'use', 'excitation', ... % should be 'inversion', but this way we get timing
        'system', sys);

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

    % Pre/rewinder templates
    gx_pre = mr.makeTrapezoid('x', 'Area', -gx_full.area/2, 'Duration', 1.0e-3, 'system', sys);
    gx_spoil_area = 4 / slice_thickness;

    % Bridged spoiler that starts at gx flat amplitude for continuity.
    gx_spoil = mr.makeExtendedTrapezoidArea('x', gx_full.amplitude, 0, gx_spoil_area, sys);
    delayTR = mr.makeDelay(0.5e-3);

    pe_areas = ((0:Ny-1) - floor(Ny/2)) / fov;
    max_pe_area = max(abs(pe_areas));
    gy_phase = mr.makeTrapezoid('y', 'Area', max_pe_area, 'Duration', 1.0e-3, 'system', sys);

    seq = mr.Sequence(sys);

    rf_center = mr.calcRfCenter(rf);
    rf_phase = 0.0;
    rf_inc = 0.0;

    % Main imaging loop: PE -> slices (slice is inner loop).
    for sl = 1:num_slices
        slc_shift = (sl - 1 - (num_slices - 1) / 2);
        
        seq.addBlock(rf180);
        seq.addBlock(gz_spoil);
        
        for pe = 1:Ny
            if max_pe_area > 0
                yscale = pe_areas(pe) / max_pe_area;
            else
                yscale = 0;
            end
        
            rf_inc = mod(rf_inc + rf_spoil_inc, 360.0);
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

    out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');

    tb = TruthBuilder(seq, sys);
    tb.setBlocksPerTR(2 + 4 * Ny + 1);
    tb.setSegments([2, 4, 1], [1, Ny, 1]);
    tb.setNumAverages(num_averages);
    tb.export(out_dir, base);
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
