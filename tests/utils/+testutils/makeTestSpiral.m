function [gxCells, gyCells, adc, k] = makeTestSpiral(sys, interleaves, nSamples, fov)
%MAKETESTSPIRAL  Generate rotated spiral readout gradients and ADC event.
%
%   [gxCells, gyCells, adc, k] = makeTestSpiral(sys, interleaves, nSamples, fov)
%
%   Returns cell arrays of per-interleave gradient events (gx, gy), a
%   single shared ADC event, and the k-space trajectory (2 x nSamples).
%
%   The gradient waveform is oversampled (4x) relative to the ADC for a
%   smoother spiral.  The ADC duration equals the gradient readout
%   duration and is an integer multiple of grad raster, adc raster, and
%   dwell time.  Gradients are zero-padded at the start and hold g(end)
%   at the end (a separate rewinder handles ramp-down).
    warning('OFF', 'mr:restoreShape');
    dt = sys.gradRasterTime;

    % Oversample factor for gradient
    oversample = 4;
    nGradSamples = nSamples * oversample + 1;  % +1 so traj2grad returns exactly nSamples*oversample

    deltak = 1 / fov;
    nTurns = nSamples / (2 * pi);

    % Design trajectory at gradient resolution
    tg = linspace(0, 1, nGradSamples);
    r   = deltak * nTurns * tg;
    phi = 2 * pi * nTurns * tg;
    kg = [r .* cos(phi); r .* sin(phi)];

    [g, ~] = mr.traj2grad(kg, 'RasterTime', dt);
    nGradOut = size(g, 2);  % traj2grad returns nGradSamples-1 points

    % ADC duration = gradient readout duration (nGradOut raster periods),
    % must be integer multiple of grad raster, adc raster, and dwell time
    readout_dur = nGradOut * dt;
    dwell = readout_dur / nSamples;
    adc = mr.makeAdc(nSamples, 'Dwell', dwell);

    % k-space trajectory at ADC resolution (for output)
    ta = linspace(0, 1, nSamples);
    ra   = deltak * nTurns * ta;
    phia = 2 * pi * nTurns * ta;
    k = [ra .* cos(phia); ra .* sin(phia)];

    gxCells = cell(interleaves, 1);
    gyCells = cell(interleaves, 1);

    for i = 1:interleaves
        angle = 2 * pi * (i - 1) / interleaves;
        R = [cos(angle), -sin(angle); sin(angle), cos(angle)];
        grot = R * g;
        gx = [0, grot(1, :), grot(1, end)];
        gy = [0, grot(2, :), grot(2, end)];
        gxCells{i} = mr.makeArbitraryGrad('x', gx, sys, 'first', 0, 'last', gx(end));
        gyCells{i} = mr.makeArbitraryGrad('y', gy, sys, 'first', 0, 'last', gy(end));
    end
end
