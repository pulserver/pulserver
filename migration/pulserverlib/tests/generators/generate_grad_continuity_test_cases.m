%% generate_gradient_discontinuity_sequences
%
% Generates Pulseq .seq files covering gradient continuity edge cases.
%
% IMPORTANT:
% Before running this script, comment out the gradient continuity
% check inside:
%
%   mr.Sequence/setBlock.m (L1032-1096)
%
% This script intentionally creates sequences that violate continuity.
% The resulting .seq files are meant to be validated by the Pulseq C library.
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

gx_extended = makeExtendedTrapezoid('x', ...
    'Amplitude', [0, 100000, 0], ...
    'Times',     [0, 1e-4, 2e-4]);

gx_extended_delay = makeExtendedTrapezoid('x', ...
    'Amplitude', [0, 100000, 0], ...
    'Times',     [1e-4, 2e-4, 3e-4]);

gx_endshigh = makeExtendedTrapezoid('x', ...
    'Amplitude', [0, 100000, 100000], ...
    'Times',     [0, 1e-4, 2e-4]);

gx_startshigh = makeExtendedTrapezoid('x', ...
    'Amplitude', [100000, 100000, 0], ...
    'Times',     [0, 1e-4, 2e-4]);

gx_startshigh2 = makeExtendedTrapezoid('x', ...
    'Amplitude', [200000, 100000, 0], ...
    'Times',     [0, 1e-4, 2e-4]);

gx_allhigh = makeExtendedTrapezoid('x', ...
    'Amplitude', [100000, 100000, 100000], ...
    'Times',     [0, 1e-4, 2e-4]);

delay = makeDelay(1e-3);

%% ------------------------------------------------------------------------
% Rotation helpers
% ------------------------------------------------------------------------

identity = mr.makeRotation(eye(3));

theta = deg2rad(90);
R0 = [cos(theta), -sin(theta), 0];
R1 = [sin(theta),  cos(theta), 0];
R2 = [0, 0, 1];
rotmat = mr.makeRotation([R0; R1; R2]);

%% ------------------------------------------------------------------------
% Non-rotated cases
% ------------------------------------------------------------------------

seq = mr.Sequence();
seq.addBlock(gx_trap);
seq.addBlock(gx_extended);
seq.addBlock(gx_trap);
seq.write(fullfile(dataDir, '01_ok_trap_extended_trap.seq'));

seq = mr.Sequence();
seq.addBlock(gx_trap);
seq.addBlock(gx_startshigh);
seq.write(fullfile(dataDir, '02_fail_trap_then_startshigh.seq'));

seq = mr.Sequence();
seq.addBlock(gx_startshigh);
seq.write(fullfile(dataDir, '03_fail_startshigh_first.seq'));

seq = mr.Sequence();
seq.addBlock(delay);
seq.addBlock(gx_allhigh);
seq.write(fullfile(dataDir, '04_fail_delay_then_allhigh.seq'));

seq = mr.Sequence();
seq.addBlock(gx_extended_delay);
seq.write(fullfile(dataDir, '05_ok_extended_with_delay.seq'));

seq = mr.Sequence();
seq.addBlock(delay);
seq.addBlock(gx_startshigh);
seq.write(fullfile(dataDir, '06_fail_delay_then_startshigh.seq'));

seq = mr.Sequence();
seq.addBlock(gx_endshigh);
seq.addBlock(gx_startshigh2);
seq.write(fullfile(dataDir, '07_fail_nonconnecting.seq'));

%% ------------------------------------------------------------------------
% Rotated cases
% ------------------------------------------------------------------------

seq = mr.Sequence();
seq.addBlock(gx_trap, identity);
seq.addBlock(gx_extended, identity);
seq.addBlock(gx_trap, identity);
seq.write(fullfile(dataDir, '08_ok_rot_identity.seq'));

seq = mr.Sequence();
seq.addBlock(gx_trap, identity);
seq.addBlock(gx_startshigh, identity);
seq.write(fullfile(dataDir, '09_fail_rot_identity.seq'));

seq = mr.Sequence();
seq.addBlock(gx_startshigh, identity);
seq.write(fullfile(dataDir, '10_fail_rot_first_block.seq'));

seq = mr.Sequence();
seq.addBlock(delay);
seq.addBlock(gx_allhigh, identity);
seq.write(fullfile(dataDir, '11_fail_rot_allhigh.seq'));

seq = mr.Sequence();
seq.addBlock(gx_extended_delay, identity);
seq.write(fullfile(dataDir, '12_ok_rot_extended_delay.seq'));

seq = mr.Sequence();
seq.addBlock(delay);
seq.addBlock(gx_startshigh, identity);
seq.write(fullfile(dataDir, '13_fail_rot_delay_then_startshigh.seq'));

seq = mr.Sequence();
seq.addBlock(gx_endshigh, identity);
seq.addBlock(gx_startshigh2, identity);
seq.write(fullfile(dataDir, '14_fail_rot_nonconnecting.seq'));

seq = mr.Sequence();
seq.addBlock(gx_endshigh, rotmat);
seq.addBlock(gx_startshigh, rotmat);
seq.write(fullfile(dataDir, '15_ok_rot_same_rotation.seq'));

seq = mr.Sequence();
seq.addBlock(gx_endshigh, identity);
seq.addBlock(gx_startshigh, rotmat);
seq.write(fullfile(dataDir, '16_fail_rot_diff_rotation_1.seq'));

seq = mr.Sequence();
seq.addBlock(gx_endshigh, rotmat);
seq.addBlock(gx_startshigh, identity);
seq.write(fullfile(dataDir, '17_fail_rot_diff_rotation_2.seq'));

fprintf('All sequences generated.\n');