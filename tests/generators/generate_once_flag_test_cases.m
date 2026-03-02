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
% Valid multipass: 3 identical [P, clear, M, M, C] passes
% (valid counterpart of case 10 — once flags in the middle form identical
%  repeating periods; C library folds into 1 period with num_passes=3)
% ------------------------------------------------------------------------

seq = mr.Sequence();
for pass = 1:3
    seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));
    seq.addBlock(mr.makeLabel('SET','ONCE', 0));
    seq.addBlock(rf, gx_flat1);
    seq.addBlock(gx_flat1);
    seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2));
end
seq.write(fullfile(dataDir, '11_multi_tr_valid_once_in_the_middle.seq'));

%% ------------------------------------------------------------------------
% Valid multipass, cooldown only: 3 identical [M, M, C] passes
% No prep blocks. Each pass has 2 main blocks and 1 cooldown block.
% After folding: num_passes=3, num_prep=0, num_cooldown=1
% ------------------------------------------------------------------------

seq = mr.Sequence();
for pass = 1:3
    seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 0));
    seq.addBlock(rf, gx_flat1);
    seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2));
end
seq.write(fullfile(dataDir, '12_multipass_valid_cooldown_only.seq'));

%% ------------------------------------------------------------------------
% Valid multipass, prep + cooldown: 3 identical [P, M, M, C] passes
% Each pass has 1 prep, 2 main, 1 cooldown block.
% After folding: num_passes=3, num_prep=1, num_cooldown=1
% ------------------------------------------------------------------------

seq = mr.Sequence();
for pass = 1:3
    seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));
    seq.addBlock(rf, gx_flat1, mr.makeLabel('SET','ONCE', 0));
    seq.addBlock(gx_flat1);
    seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2));
end
seq.write(fullfile(dataDir, '13_multipass_valid_prep_cooldown.seq'));

%% ------------------------------------------------------------------------
% Invalid multipass: different cooldown block types across passes
% [M, M, C1,  M, M, C2]  where C1 != C2 (C2 has extra RF event)
% No valid period found -> PULSEQLIB_ERR_INVALID_ONCE_FLAGS
% ------------------------------------------------------------------------

seq = mr.Sequence();
% Pass 1
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 0));
seq.addBlock(rf, gx_flat1);
seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2));       % C1: no RF
% Pass 2 (different cooldown block)
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 0));
seq.addBlock(rf, gx_flat1);
seq.addBlock(rf, gx_rdown1, mr.makeLabel('SET','ONCE', 2));   % C2: has RF -> different block ID
seq.write(fullfile(dataDir, '14_multipass_fail_diff_cooldown.seq'));

%% ------------------------------------------------------------------------
% Invalid multipass: different prep block types across passes
% [P1, M, M,  P2, M, M]  where P1 != P2 (P2 has extra RF event)
% No valid period found -> PULSEQLIB_ERR_INVALID_ONCE_FLAGS
% ------------------------------------------------------------------------

seq = mr.Sequence();
% Pass 1
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));          % P1: no RF
seq.addBlock(rf, gx_flat1, mr.makeLabel('SET','ONCE', 0));
seq.addBlock(gx_rdown1);
% Pass 2 (different prep block)
seq.addBlock(rf, gx_rup1, mr.makeLabel('SET','ONCE', 1));      % P2: has RF -> different block ID
seq.addBlock(rf, gx_flat1, mr.makeLabel('SET','ONCE', 0));
seq.addBlock(gx_rdown1);
seq.write(fullfile(dataDir, '15_multipass_fail_diff_prep.seq'));

%% ------------------------------------------------------------------------
% Invalid multipass: different prep types with cooldown
% [P1, M, M, C,  P2, M, M, C]  where P1 != P2
% No valid period found -> PULSEQLIB_ERR_INVALID_ONCE_FLAGS
% ------------------------------------------------------------------------

seq = mr.Sequence();
% Pass 1
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));          % P1: no RF
seq.addBlock(rf, gx_flat1, mr.makeLabel('SET','ONCE', 0));
seq.addBlock(gx_flat1);
seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2));
% Pass 2 (different prep block)
seq.addBlock(rf, gx_rup1, mr.makeLabel('SET','ONCE', 1));      % P2: has RF -> different block ID
seq.addBlock(rf, gx_flat1, mr.makeLabel('SET','ONCE', 0));
seq.addBlock(gx_flat1);
seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2));
seq.write(fullfile(dataDir, '16_multipass_fail_diff_prep_with_cooldown.seq'));

%% ------------------------------------------------------------------------
% Invalid multipass: different cooldown types with prep
% [P, M, M, C1,  P, M, M, C2]  where C1 != C2
% No valid period found -> PULSEQLIB_ERR_INVALID_ONCE_FLAGS
% ------------------------------------------------------------------------

seq = mr.Sequence();
% Pass 1
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));
seq.addBlock(rf, gx_flat1, mr.makeLabel('SET','ONCE', 0));
seq.addBlock(gx_flat1);
seq.addBlock(gx_rdown1, mr.makeLabel('SET','ONCE', 2));        % C1: no RF
% Pass 2 (different cooldown block)
seq.addBlock(gx_rup1, mr.makeLabel('SET','ONCE', 1));
seq.addBlock(rf, gx_flat1, mr.makeLabel('SET','ONCE', 0));
seq.addBlock(gx_flat1);
seq.addBlock(rf, gx_rdown1, mr.makeLabel('SET','ONCE', 2));    % C2: has RF -> different block ID
seq.write(fullfile(dataDir, '17_multipass_fail_diff_cooldown_with_prep.seq'));

fprintf('All sequences generated.\n')