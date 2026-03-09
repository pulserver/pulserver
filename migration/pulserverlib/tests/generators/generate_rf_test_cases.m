%% generate_rf_safety_sequences
%
% Generates Pulseq .seq files covering RF safety cases.
%

clear; clc;
import mr.*

%% Output directory
scriptDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(scriptDir, '..', 'data');
if ~exist(dataDir, 'dir'), mkdir(dataDir); end

%% ------------------------------------------------------------------------
% Gradient definitions
% ------------------------------------------------------------------------

gx_trap = makeTrapezoid('x', 'Area', 1000, 'Duration', 1e-3);

%% ------------------------------------------------------------------------
% Basic rf parameters calculation
% ------------------------------------------------------------------------
rf180 = makeBlockPulse(pi,'Duration', 1e-3, 'use', 'excitation');
seq = mr.Sequence();
seq.addBlock(rf180);
seq.write(fullfile(dataDir, '00_basic_rfstat.seq'));

%% ------------------------------------------------------------------------
% Periodic rf amplitude sequence (e.g., MRF)
% ------------------------------------------------------------------------
rf180 = makeBlockPulse(pi,'Duration', 1e-3, 'use', 'inversion');
rf90 = makeBlockPulse(0.5 * pi,'Duration', 1e-3, 'use', 'excitation');
signal = rf90.signal;
seq = mr.Sequence();
seq.addBlock(rf180);
for n = 1:10
    rf90.signal = 0.1 * n * signal;
    seq.addBlock(rf90);
    seq.addBlock(gx_trap);
end
seq.addBlock(rf180);
for n = 1:10
    rf90.signal = 0.1 * n * signal;
    seq.addBlock(rf90);
    seq.addBlock(gx_trap);
end
seq.write(fullfile(dataDir, '01_rfamp_ok_mrfingerprinting.seq'));

%% ------------------------------------------------------------------------
% Nonperiodic rf amplitude sequence (VFA) - error (must be splitted in multiple SPGR subsequences)
% ------------------------------------------------------------------------
rf90 = makeBlockPulse(0.5 * pi,'Duration', 1e-3, 'use', 'excitation');
signal = rf90.signal;
seq = mr.Sequence();
for n = 1:10
    rf90.signal = 0.1 * n * signal;
    seq.addBlock(rf90);
    seq.addBlock(gx_trap);
    seq.addBlock(rf90);
    seq.addBlock(gx_trap);
    seq.addBlock(rf90);
    seq.addBlock(gx_trap);
    seq.addBlock(rf90);
    seq.addBlock(gx_trap);
end
seq.write(fullfile(dataDir, '02_rfamp_fail_vfa.seq'));

%% ------------------------------------------------------------------------
% Periodic rf shim sequence (e.g., PnP-MRF)
% ------------------------------------------------------------------------
rf180 = makeBlockPulse(pi,'Duration', 1e-3, 'use', 'excitation');
rf90 = makeBlockPulse(0.5 * pi,'Duration', 1e-3, 'use', 'excitation');

dalpha = 2 * pi / 16; % 16-TxCh coil
cpMode = makeRfShim(exp(-1i * (0:15) * dalpha));
gradMode = makeRfShim(exp(-1i * 2 * (0:15) * dalpha));

seq = mr.Sequence();
seq.addBlock(rf180);
for n = 1:10
    if mod(n, 2) == 1
        mode = cpMode;
    else
        mode = gradMode;
    end
    seq.addBlock(rf90, mode);
    seq.addBlock(gx_trap);
end
seq.addBlock(rf180);
for n = 1:10
    if mod(n, 2) == 1
        mode = cpMode;
    else
        mode = gradMode;
    end
    seq.addBlock(rf90, mode);
    seq.addBlock(gx_trap);
end
seq.write(fullfile(dataDir, '03_rfshim_ok_pnpmrfingerprinting.seq'));

%% ------------------------------------------------------------------------
% Nonperiodic rf shim sequence - error (must be splitted in multiple Shimmed subsequences)
% ------------------------------------------------------------------------
rf90 = makeBlockPulse(0.5 * pi,'Duration', 1e-3, 'use', 'excitation');

dalpha = 2 * pi / 16; % 16-TxCh coil
cpMode = makeRfShim(exp(-1i * (0:15) * dalpha));
gradMode = makeRfShim(exp(-1i * 2 * (0:15) * dalpha));

seq = mr.Sequence();
for n = 1:10
    seq.addBlock(rf90, cpMode);
    seq.addBlock(gx_trap);
end
for n = 1:10
    seq.addBlock(rf90, gradMode);
    seq.addBlock(gx_trap);
end
seq.write(fullfile(dataDir, '04_rfshim_fail_gre.seq'));


fprintf('All sequences generated.\n');