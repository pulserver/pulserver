%% generate_once_flag_sequences
%
% Generates Pulseq .seq files to test proper usage of ONCE flag.

clear; clc;
import mr.*


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
seq.write('01_single_tr_valid_once.seq');

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
seq.write('02_dual_tr_valid_once.seq');

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
seq.write('03_multi_tr_valid_once.seq');

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
seq.write('04_multi_tr_valid_once_degenerate.seq');

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
seq.write('05_multi_tr_once_prep_only.seq');

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
seq.write('06_multi_tr_once_cooldown_only.seq');

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
seq.write('07_single_tr_nonvalid_once.seq');


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
seq.write('08_prep_too_long.seq');

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
seq.write('09_cooldown_too_long.seq');

%% ------------------------------------------------------------------------
% Invalid: Once in the middle of sequence
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
seq.write('10_multi_tr_nonvalid_once_in_the_middle.seq');

fprintf('All sequences generated.\n');