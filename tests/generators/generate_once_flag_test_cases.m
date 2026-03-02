%% generate_once_flag_sequences
%
% Generates Pulseq .seq files to test proper usage of ONCE flag.

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

%% ------------------------------------------------------------------------
% Single TR, valid case (first TR is also last TR)
% ------------------------------------------------------------------------

seq = mr.Sequence();
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1)); % we label the first as preparing to exclude them if the sequence is repeated
seq.addBlock(mr.makeLabel('SET','ONCE', 0)); % remove preparing block label
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2)); % we also label this block as the exit block, which excludes it from all but last repetitions if the sequence is repeated
seq.write(fullfile(dataDir, '01_single_tr_valid_once.seq'));

%% ------------------------------------------------------------------------
% Double TR, valid case
% ------------------------------------------------------------------------

seq = mr.Sequence();
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1)); % we label the first as preparing to exclude them if the sequence is repeated
seq.addBlock(mr.makeLabel('SET','ONCE', 0)); % remove preparing block label
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2)); % we also label this block as the exit block, which excludes it from all but last repetitions if the sequence is repeated
seq.write(fullfile(dataDir, '02_dual_tr_valid_once.seq'));

%% ------------------------------------------------------------------------
% Triple TR (same as N-TRs), valid case
% ------------------------------------------------------------------------

seq = mr.Sequence();
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1)); % we label the first as preparing to exclude them if the sequence is repeated
seq.addBlock(mr.makeLabel('SET','ONCE', 0)); % remove preparing block label
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2)); % we also label this block as the exit block, which excludes it from all but last repetitions if the sequence is repeated
seq.write(fullfile(dataDir, '03_multi_tr_valid_once.seq'));

%% ------------------------------------------------------------------------
% Triple TR (same as N-TRs), degenerate prep-cooldown
% ------------------------------------------------------------------------

seq = mr.Sequence();
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_rdown1);
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 0));
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
% Triple TR (same as N-TRs), valid case - prep only
% ------------------------------------------------------------------------

seq = mr.Sequence();
seq.addBlock(mr.makeDelay(0.1e-3), mr.makeLabel('SET','ONCE', 1)); % we label the first as preparing to exclude them if the sequence is repeated
seq.addBlock(mr.makeLabel('SET','ONCE', 0)); % remove preparing block label
seq.addBlock(gx_rup1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(gx_rdown1);
seq.write(fullfile(dataDir, '05_multi_tr_once_prep_only.seq'));

%% ------------------------------------------------------------------------
% Triple TR (same as N-TRs), valid case - cooldown only
% ------------------------------------------------------------------------

seq = mr.Sequence();
seq.addBlock(gx_rup1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(gx_rdown1);
seq.addBlock(mr.makeDelay(0.1e-3), mr.makeLabel('SET','ONCE', 2)); % we also label this block as the exit block, which excludes it from all but last repetitions if the sequence is repeated
seq.write(fullfile(dataDir, '06_multi_tr_once_cooldown_only.seq'));

%% ------------------------------------------------------------------------
% Nonvalid case (first TR is also last TR)
% ------------------------------------------------------------------------

seq = mr.Sequence();
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1)); % we label the first as preparing to exclude them if the sequence is repeated
seq.addBlock(mr.makeLabel('SET','ONCE', 0)); % remove preparing block label
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(gx_rup2);
seq.addBlock(gx_flat2);
seq.addBlock(gx_rdown2, mr.makeLabel('SET','ONCE', 2)); % we also label this block as the exit block, which excludes it from all but last repetitions if the sequence is repeated
seq.write(fullfile(dataDir, '07_single_tr_nonvalid_once.seq'));


%% ------------------------------------------------------------------------
% Triple TR (same as N-TRs), nonvalid case - prep too long
% ------------------------------------------------------------------------

seq = mr.Sequence();
seq.addBlock(gx_rup1_long, mr.makeLabel('SET','ONCE', 1)); % we label the first as preparing to exclude them if the sequence is repeated
seq.addBlock(mr.makeLabel('SET','ONCE', 0)); % remove preparing block label
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2)); % we also label this block as the exit block, which excludes it from all but last repetitions if the sequence is repeated
seq.write(fullfile(dataDir, '08_prep_too_long.seq'));

%% ------------------------------------------------------------------------
% Triple TR (same as N-TRs), nonvalid case - cooldown too long
% ------------------------------------------------------------------------

seq = mr.Sequence();
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1)); % we label the first as preparing to exclude them if the sequence is repeated
seq.addBlock(mr.makeLabel('SET','ONCE', 0)); % remove preparing block label
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(gx_rdown1_long, mr.makeLabel('SET','ONCE', 2)); % we also label this block as the exit block, which excludes it from all but last repetitions if the sequence is repeated
seq.write(fullfile(dataDir, '09_cooldown_too_long.seq'));

%% ------------------------------------------------------------------------
% Invalid: Once in the middle of sequence (non-identical inner loop periods)
% Note: once in the middle is allowed if all inner loop repetitions are
% structurally identical.  This case is invalid because the once=1
% sections produce periods of different lengths / block-ID patterns.
% ------------------------------------------------------------------------

seq = mr.Sequence();
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1)); % we label the first as preparing to exclude them if the sequence is repeated
seq.addBlock(mr.makeLabel('SET','ONCE', 0)); % remove preparing block label
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_flat1, mr.makeLabel('SET','ONCE', 1));
seq.addBlock(gx_flat1);
seq.addBlock(mr.makeLabel('SET','ONCE', 0)); % remove preparing block label
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_flat1);
seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2)); % we also label this block as the exit block, which excludes it from all but last repetitions if the sequence is repeated
seq.write(fullfile(dataDir, '10_multi_tr_nonvalid_once_in_the_middle.seq'));

%% ------------------------------------------------------------------------
% Valid multipass: [P, M, M, C] x 3 passes
% Simplest complete multipass case: each pass has prep + 2 main + cooldown.
% ONCE=0 on first main block clears the ONCE flag.
% After folding: num_passes=3, num_prep=1, num_cooldown=1, 2 main blocks
% Period detection: whole-sequence tiling at pl=4, trailing=0.
% ------------------------------------------------------------------------

seq = mr.Sequence();
for pass = 1:3
    seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));       % P: ramp 0→100k
    seq.addBlock(rf, gx_flat1, mr.makeLabel('SET','ONCE', 0));  % M: clears once
    seq.addBlock(gx_flat1);                                      % M
    seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2));     % C: ramp 100k→0
end
seq.write(fullfile(dataDir, '11_multipass_valid_prep_cooldown.seq'));

%% ------------------------------------------------------------------------
% Valid multipass: [P, M, M] x 3 passes   (prep only, no cooldown)
% Tests the branch where has_cooldown=0 (ONCE=2 never appears in labelset).
% After folding: num_passes=3, num_prep=1, num_cooldown=0
% Period detection: trailing=0 (no trailing once==2), pl=3 tiles directly.
% ------------------------------------------------------------------------

seq = mr.Sequence();
for pass = 1:3
    seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));       % P: ramp 0→100k
    seq.addBlock(rf, gx_flat1, mr.makeLabel('SET','ONCE', 0));  % M: excitation
    seq.addBlock(gx_rdown1);                                     % M: ramp 100k→0
end
seq.write(fullfile(dataDir, '12_multipass_valid_prep_only.seq'));

%% ------------------------------------------------------------------------
% Valid multipass: [M, M, C] x 3 passes   (cooldown only, no prep)
% Tests the branch where has_prep=0 (ONCE=1 never appears in labelset).
% After folding: num_passes=3, num_prep=0, num_cooldown=1
% Period detection: whole-sequence tiling at pl=3, trailing=0.
% ------------------------------------------------------------------------

seq = mr.Sequence();
for pass = 1:3
    seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 0));       % M: ramp 0→100k
    seq.addBlock(rf, gx_flat1);                                  % M: excitation
    seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2));     % C: ramp 100k→0
end
seq.write(fullfile(dataDir, '13_multipass_valid_cooldown_only.seq'));

%% ------------------------------------------------------------------------
% Valid multipass: [P, M, M] x 2 passes + trailing [C]
% Exercises the folding path where trailing cooldown blocks (once_flag==2)
% are separated before period detection:
%   trailing = 1 (the final delay/ONCE=2 block)
%   effective = 6 blocks → period pl=3 tiles twice → num_passes=2
% Folded result: [P, M, M, C] with num_passes=2, num_prep=1, num_cool=1.
% Also tests the 2-pass minimum (num_reps == 2).
% ------------------------------------------------------------------------

seq = mr.Sequence();
for pass = 1:2
    seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));       % P: ramp 0→100k
    seq.addBlock(rf, gx_flat1, mr.makeLabel('SET','ONCE', 0));  % M: excitation
    seq.addBlock(gx_rdown1);                                     % M: ramp 100k→0
end
seq.addBlock(mr.makeDelay(0.1e-3), mr.makeLabel('SET','ONCE', 2));  % trailing C
seq.write(fullfile(dataDir, '14_multipass_valid_trailing_cooldown.seq'));

%% ------------------------------------------------------------------------
% Valid multipass: [P, M, M, M, M, C] x 3 passes  (multi-TR per pass)
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
seq.write(fullfile(dataDir, '15_multipass_valid_multi_tr.seq'));

%% ------------------------------------------------------------------------
% Invalid multipass: different main block types across passes
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
seq.write(fullfile(dataDir, '16_multipass_fail_diff_main.seq'));

%% ------------------------------------------------------------------------
% Invalid multipass: different pass lengths
% Pass 1: [P, M, M, C]  (4 blocks)
% Pass 2: [P, M, C]     (3 blocks)
% Total 7 blocks, 7 is prime → no valid period → ERR_INVALID_ONCE_FLAGS
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
seq.write(fullfile(dataDir, '17_multipass_fail_diff_length.seq'));

fprintf('All sequences generated.\n')