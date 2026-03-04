%% generate_once_flag_sequences
%
% Generates Pulseq .seq files to test proper usage of ONCE flag.
%
% 13 test cases:
%   01: single TR valid
%   02: dual TR valid
%   03: triple TR valid
%   04: degenerate prep/cooldown
%   05: prep too long
%   06: cooldown too long
%   07: ONCE in middle, invalid
%   08: multipass valid [P,M,M,C]×3
%   09: multipass valid prep only
%   10: multipass valid cooldown only
%   11: multipass multi-TR
%   12: multipass fail diff main
%   13: multipass fail diff length

clear; clc;
import mr.*

%% Output directory
scriptDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(scriptDir, '..', 'data');
if ~exist(dataDir, 'dir'), mkdir(dataDir); end


%% ------------------------------------------------------------------------
% Gradient definitions
% ------------------------------------------------------------------------
rf = makeBlockPulse(0.01 * pi,'Duration', 1e-5, 'use', 'excitation');
gx_rup1 = makeExtendedTrapezoid('x', ...
    'Amplitude', [0, 100000], ...
    'Times',     [0, 1e-4]);
gx_rup1_long = makeExtendedTrapezoid('x', ...
    'Amplitude', [0, 100000], ...
    'Times',     [0, 200.0e-3]);
gx_flat1 = makeExtendedTrapezoid('x', ...
    'Amplitude', [100000, 100000], ...
    'Times',     [0, 1e-4]);
gx_rup2 = makeExtendedTrapezoid('x', ...
    'Amplitude', [100000, 200000], ...
    'Times',     [0, 1e-4]);
gx_flat2 = makeExtendedTrapezoid('x', ...
    'Amplitude', [200000, 200000], ...
    'Times',     [0, 1e-4]);
gx_rdown1 = makeExtendedTrapezoid('x', ...
    'Amplitude', [100000, 0], ...
    'Times',     [0, 1e-4]);
gx_rdown1_long = makeExtendedTrapezoid('x', ...
    'Amplitude', [100000, 0], ...
    'Times',     [0, 200.0e-3]);
gx_rdown2 = makeExtendedTrapezoid('x', ...
    'Amplitude', [200000, 0], ...
    'Times',     [0, 1e-4]);

lblOnce0 = mr.makeLabel('SET','ONCE', 0);

%% ------------------------------------------------------------------------
% 01: Single TR, valid case (first TR is also last TR)
% Merged: ONCE=0 into first main block (rf + gx_flat1)
% ------------------------------------------------------------------------

seq = mr.Sequence();
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));
seq.addBlock(rf, gx_flat1, lblOnce0);
seq.addBlock(gx_flat1);
seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2));
seq.write(fullfile(dataDir, '01_single_tr_valid_once.seq'));

%% ------------------------------------------------------------------------
% 02: Double TR, valid case
% Merged: ONCE=0 into first main block
% ------------------------------------------------------------------------

seq = mr.Sequence();
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));
seq.addBlock(rf, gx_flat1, lblOnce0);
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2));
seq.write(fullfile(dataDir, '02_dual_tr_valid_once.seq'));

%% ------------------------------------------------------------------------
% 03: Triple TR (same as N-TRs), valid case
% Merged: ONCE=0 into first main block
% ------------------------------------------------------------------------

seq = mr.Sequence();
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));
seq.addBlock(rf, gx_flat1, lblOnce0);
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2));
seq.write(fullfile(dataDir, '03_multi_tr_valid_once.seq'));

%% ------------------------------------------------------------------------
% 04: Triple TR (same as N-TRs), degenerate prep-cooldown
% This matches original test 04 with merged ONCE=0.
% Prep pattern == first main TR, cooldown pattern == last main TR.
% ------------------------------------------------------------------------

seq = mr.Sequence();
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_rdown1);
seq.addBlock(gx_rup1, lblOnce0);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_rdown1);
seq.addBlock(gx_rup1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_rdown1);
seq.addBlock(gx_rup1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_rdown1);
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 2));
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_rdown1);
seq.write(fullfile(dataDir, '04_multi_tr_valid_once_degenerate.seq'));

%% ------------------------------------------------------------------------
% 05: Prep too long (was 08)
% Merged: ONCE=0 into first main block
% ------------------------------------------------------------------------

seq = mr.Sequence();
seq.addBlock(gx_rup1_long, mr.makeLabel('SET','ONCE', 1));
seq.addBlock(rf, gx_flat1, lblOnce0);
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2));
seq.write(fullfile(dataDir, '05_prep_too_long.seq'));

%% ------------------------------------------------------------------------
% 06: Cooldown too long (was 09)
% Merged: ONCE=0 into first main block
% ------------------------------------------------------------------------

seq = mr.Sequence();
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));
seq.addBlock(rf, gx_flat1, lblOnce0);
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(gx_rdown1_long, mr.makeLabel('SET','ONCE', 2));
seq.write(fullfile(dataDir, '06_cooldown_too_long.seq'));

%% ------------------------------------------------------------------------
% 07: Invalid: ONCE in the middle (was 10)
% Two ONCE=0 blocks merged into their respective next main blocks.
% The once=1 sections produce periods of different lengths / block-ID
% patterns → ERR_INVALID_ONCE_FLAGS.
% ------------------------------------------------------------------------

seq = mr.Sequence();
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));
seq.addBlock(rf, gx_flat1, lblOnce0);
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_flat1, mr.makeLabel('SET','ONCE', 1));
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_flat1, lblOnce0);
seq.addBlock(gx_flat1);
seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2));
seq.write(fullfile(dataDir, '07_multi_tr_nonvalid_once_in_the_middle.seq'));

%% ------------------------------------------------------------------------
% 08: Valid multipass: [P, M, M, C] x 3 passes (was 11)
% Simplest complete multipass case: each pass has prep + 2 main + cooldown.
% ONCE=0 on first main block clears the ONCE flag.
% After folding: num_passes=3, num_prep=1, num_cooldown=1, 2 main blocks.
% Pass boundaries at blocks 0, 4, 8 (transitions back to once=1).
% ------------------------------------------------------------------------

seq = mr.Sequence();
for pass = 1:3
    seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));       % P: ramp 0→100k
    seq.addBlock(rf, gx_flat1, mr.makeLabel('SET','ONCE', 0));  % M: clears once
    seq.addBlock(gx_flat1);                                      % M
    seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2));     % C: ramp 100k→0
end
seq.write(fullfile(dataDir, '08_multipass_valid_prep_cooldown.seq'));

%% ------------------------------------------------------------------------
% 09: Valid multipass: [P, M, M] x 3 passes   (prep only, no cooldown)
%     (was 12)
% Tests the branch where has_cooldown=0 (ONCE=2 never appears in labelset).
% After folding: num_passes=3, num_prep=1, num_cooldown=0.
% Pass boundaries at blocks 0, 3, 6 (transitions back to once=1).
% ------------------------------------------------------------------------

seq = mr.Sequence();
for pass = 1:3
    seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));       % P: ramp 0→100k
    seq.addBlock(rf, gx_flat1, mr.makeLabel('SET','ONCE', 0));  % M: excitation
    seq.addBlock(gx_rdown1);                                     % M: ramp 100k→0
end
seq.write(fullfile(dataDir, '09_multipass_valid_prep_only.seq'));

%% ------------------------------------------------------------------------
% 10: Valid multipass: [M, M, C] x 3 passes   (cooldown only, no prep)
%     (was 13)
% Tests the branch where has_prep=0 (ONCE=1 never appears in labelset).
% After folding: num_passes=3, num_prep=0, num_cooldown=1.
% Pass boundaries at blocks 0, 3, 6 (transitions back to once=0).
% ------------------------------------------------------------------------

seq = mr.Sequence();
for pass = 1:3
    seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 0));       % M: ramp 0→100k
    seq.addBlock(rf, gx_flat1);                                  % M: excitation
    seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2));     % C: ramp 100k→0
end
seq.write(fullfile(dataDir, '10_multipass_valid_cooldown_only.seq'));

%% ------------------------------------------------------------------------
% 11: Valid multipass: [P, M, M, M, M, C] x 3 passes  (multi-TR per pass)
%     (was 15)
% Each pass has 4 main blocks = 2 TRs (each TR = [rf+gx_flat1, gx_flat1]).
% After folding: num_passes=3, num_prep=1, num_cooldown=1, num_trs=2
% Tests that TR identification works correctly within a multipass period.
% ------------------------------------------------------------------------

seq = mr.Sequence();
for pass = 1:3
    seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));       % P: ramp 0→100k
    seq.addBlock(rf, gx_flat1, mr.makeLabel('SET','ONCE', 0));  % M: TR1 excitation
    seq.addBlock(gx_flat1);                                      % M: TR1 readout
    seq.addBlock(rf, gx_flat1);                                  % M: TR2 excitation
    seq.addBlock(gx_flat1);                                      % M: TR2 readout
    seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2));     % C: ramp 100k→0
end
seq.write(fullfile(dataDir, '11_multipass_valid_multi_tr.seq'));

%% ------------------------------------------------------------------------
% 12: Invalid multipass: different main block types across passes (was 16)
% Pass 1: [P, rf+gx_flat1, C]     -- main block has RF
% Pass 2: [P, gx_flat1,    C]     -- main block has NO RF (different def)
% No valid period → PULSEQLIB_ERR_INVALID_ONCE_FLAGS
% ------------------------------------------------------------------------

seq = mr.Sequence();
% Pass 1
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));           % P
seq.addBlock(rf, gx_flat1, mr.makeLabel('SET','ONCE', 0));      % M: with RF
seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2));         % C
% Pass 2 (different main block)
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));           % P
seq.addBlock(gx_flat1, mr.makeLabel('SET','ONCE', 0));          % M': no RF → different block def
seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2));         % C
seq.write(fullfile(dataDir, '12_multipass_fail_diff_main.seq'));

%% ------------------------------------------------------------------------
% 13: Invalid multipass: different pass lengths (was 17)
% Pass 1: [P, M, M, C]  (4 blocks)
% Pass 2: [P, M, C]     (3 blocks)
% Pass lengths differ (4 vs 3) → ERR_INVALID_ONCE_FLAGS
% ------------------------------------------------------------------------

seq = mr.Sequence();
% Pass 1 (4 blocks)
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));           % P
seq.addBlock(rf, gx_flat1, mr.makeLabel('SET','ONCE', 0));      % M
seq.addBlock(gx_flat1);                                          % M
seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2));         % C
% Pass 2 (3 blocks — shorter)
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));           % P
seq.addBlock(rf, gx_flat1, mr.makeLabel('SET','ONCE', 0));      % M
seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2));         % C
seq.write(fullfile(dataDir, '13_multipass_fail_diff_length.seq'));

fprintf('All sequences generated.\n')