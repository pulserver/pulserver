%% generate_gradient_limit_sequences
%
% Generates Pulseq .seq files covering gradient amplitude and slew-rate
% limit edge cases (single-axis and RSS violations).
%

clear; clc;
import mr.*

%% Output directory
scriptDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(scriptDir, '..', 'data');
if ~exist(dataDir, 'dir'), mkdir(dataDir); end

%% ------------------------------------------------------------------------
% 01 — Single-axis gradient amplitude violation
% ------------------------------------------------------------------------

sys = mr.opts('maxGrad', inf);   % 10 T/m limit

seq = mr.Sequence(sys);
gx = makeTrapezoid('x', ...
    'Amplitude', 20, ...        % exceeds limit
    'Duration', 1e-3, ...
    'System', sys);

seq.addBlock(gx);
seq.write(fullfile(dataDir, '01_grad_amplitude_violation.seq'));


%% ------------------------------------------------------------------------
% 02 — Single-axis slew violation
% ------------------------------------------------------------------------

Smax = mr.convert(500.0, 'T/m/s');   % 100 T/m/s limit
sys = mr.opts('maxSlew', Smax);

rise = 1e-3;
amp  = 200 * rise;   % slew = 200 T/m/s > 100

seq = mr.Sequence(sys);
gx = makeTrapezoid('x', ...
    'Amplitude', amp, ...
    'RiseTime', rise, ...
    'FlatTime', 1e-3, ...
    'FallTime', rise, ...
    'System', sys);

seq.addBlock(gx);
seq.write(fullfile(dataDir, '02_slew_violation.seq'));


%% ------------------------------------------------------------------------
% 03 — RSS gradient amplitude violation
% (Each axis below limit, vector magnitude exceeds limit)
% ------------------------------------------------------------------------

sys = mr.opts('maxGrad', inf);

seq = mr.Sequence(sys);

g = 8;   % 8 < 10 individually

gx = makeTrapezoid('x', 'Amplitude', g, 'Duration', 1e-3, 'System', sys);
gy = makeTrapezoid('y', 'Amplitude', g, 'Duration', 1e-3, 'System', sys);

% sqrt(8^2 + 8^2) ≈ 11.3 > 10
seq.addBlock(gx, gy);
seq.write(fullfile(dataDir, '03_grad_rss_violation.seq'));


%% ------------------------------------------------------------------------
% 04 — RSS slew violation
% ------------------------------------------------------------------------

Smax = mr.convert(500.0, 'T/m/s');
sys = mr.opts('maxSlew', Smax);

rise = 1e-3;
amp  = 80 * rise;   % individual slew = 80 < 100

seq = mr.Sequence(sys);

gx = makeTrapezoid('x', ...
    'Amplitude', amp, ...
    'RiseTime', rise, ...
    'FlatTime', 1e-3, ...
    'FallTime', rise, ...
    'System', sys);

gy = makeTrapezoid('y', ...
    'Amplitude', amp, ...
    'RiseTime', rise, ...
    'FlatTime', 1e-3, ...
    'FallTime', rise, ...
    'System', sys);

% sqrt(80^2 + 80^2) ≈ 113 > 100
seq.addBlock(gx, gy);
seq.write(fullfile(dataDir, '04_slew_rss_violation.seq'));

fprintf('All gradient limit test sequences generated.\n');