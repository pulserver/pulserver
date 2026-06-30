%% generate_rf_safety_sequences
%
% Generates Pulseq .seq files covering RF safety cases.
%

clear; clc;

%% Output directory
scriptDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(scriptDir, '..', 'expected');
if ~exist(dataDir, 'dir'), mkdir(dataDir); end

%% ------------------------------------------------------------------------
% Gradient definitions
% ------------------------------------------------------------------------

gx_trap = mr.makeTrapezoid('x', 'Area', 1000, 'Duration', 1e-3);

%% ------------------------------------------------------------------------
% Basic rf parameters calculation
% ------------------------------------------------------------------------
rf180 = mr.makeBlockPulse(pi,'Duration', 1e-3, 'use', 'excitation');
seq = mr.Sequence();
seq.addBlock(rf180);
seq.write(fullfile(dataDir, '00_basic_rfstat.seq'));

%% ------------------------------------------------------------------------
% Periodic rf amplitude sequence (e.g., MRF)
% ------------------------------------------------------------------------
rf180 = mr.makeBlockPulse(pi,'Duration', 1e-3, 'use', 'inversion');
rf90 = mr.makeBlockPulse(0.5 * pi,'Duration', 1e-3, 'use', 'excitation');
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
rf90 = mr.makeBlockPulse(0.5 * pi,'Duration', 1e-3, 'use', 'excitation');
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
rf180 = mr.makeBlockPulse(pi,'Duration', 1e-3, 'use', 'excitation');
rf90 = mr.makeBlockPulse(0.5 * pi,'Duration', 1e-3, 'use', 'excitation');

dalpha = 2 * pi / 16; % 16-TxCh coil
cpMode = mr.makeRfShim(exp(-1i * (0:15) * dalpha));
gradMode = mr.makeRfShim(exp(-1i * 2 * (0:15) * dalpha));

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
rf90 = mr.makeBlockPulse(0.5 * pi,'Duration', 1e-3, 'use', 'excitation');

dalpha = 2 * pi / 16; % 16-TxCh coil
cpMode = mr.makeRfShim(exp(-1i * (0:15) * dalpha));
gradMode = mr.makeRfShim(exp(-1i * 2 * (0:15) * dalpha));

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

%% ------------------------------------------------------------------------
% Canonical full-pass RF fixture for getter coverage.
%   Single pass with RF in prep and cooldown once sections.
%   When loaded with num_averages=3, canonical RF traversal must expand to:
%     prep(once) + 3x main + cooldown(once)
% ------------------------------------------------------------------------
lbl_once1 = mr.makeLabel('SET', 'ONCE', 1);
lbl_once0 = mr.makeLabel('SET', 'ONCE', 0);
lbl_once2 = mr.makeLabel('SET', 'ONCE', 2);

rf_prep = mr.makeBlockPulse(0.25 * pi, 'Duration', 1e-3, 'use', 'excitation');
rf_main = mr.makeBlockPulse(0.5 * pi,  'Duration', 1e-3, 'use', 'excitation');
rf_cool = mr.makeBlockPulse(pi,        'Duration', 1e-3, 'use', 'refocusing');

seq = mr.Sequence();
seq.addBlock(rf_prep, lbl_once1);
seq.addBlock(rf_main, gx_trap, lbl_once0);
seq.addBlock(rf_main, gx_trap);
seq.addBlock(rf_cool, lbl_once2);
seq.write(fullfile(dataDir, '05_rfprep_ok_canonical_fullpass.seq'));

%% ------------------------------------------------------------------------
% Invalid sequence: multipass with variable (non-periodic) RF across passes
%   Two complete prep→main→cooldown passes — different RF amplitude each.
% Expected error: PULSEQLIB_ERR_CONSISTENCY_RF_PERIODIC (-561)
% ------------------------------------------------------------------------
rf_pass1 = mr.makeBlockPulse(0.30 * pi, 'Duration', 1e-3, 'use', 'excitation');
rf_pass2 = mr.makeBlockPulse(0.45 * pi, 'Duration', 1e-3, 'use', 'excitation');
delay_prep = mr.makeDelay(1e-3);
delay_cool_short = mr.makeDelay(10e-3);

seq = mr.Sequence();
% --- Pass 1 ---
seq.addBlock(delay_prep, lbl_once1);
seq.addBlock(rf_pass1, gx_trap, lbl_once0);
seq.addBlock(rf_pass1, gx_trap);
seq.addBlock(rf_pass1, gx_trap);
seq.addBlock(rf_pass1, gx_trap);
seq.addBlock(delay_cool_short, lbl_once2);
% --- Pass 2 (different RF amplitude) ---
seq.addBlock(delay_prep, lbl_once1);
seq.addBlock(rf_pass2, gx_trap, lbl_once0);
seq.addBlock(rf_pass2, gx_trap);
seq.addBlock(rf_pass2, gx_trap);
seq.addBlock(rf_pass2, gx_trap);
seq.addBlock(delay_cool_short, lbl_once2);
seq.write(fullfile(dataDir, '06_rfprep_fail_multipass_variable.seq'));

%% ------------------------------------------------------------------------
% 8-channel CP mode block pulse: target for quadrature channel combination.
%   Per-channel shim weight magnitude = 1/sqrt(8), phases = CP spacing.
%   Under quadrature (RSS) combination: sqrt(sum(A_ch^2)) = 500 Hz,
%   matching the single-channel 1 ms 180-degree baseline (case 00).
%   Base pulse: pi rad, 1 ms — same geometry as case 00.
% ------------------------------------------------------------------------
rf_8ch_base = mr.makeBlockPulse(pi, 'Duration', 1e-3, 'use', 'excitation');
dalpha_8ch  = 2 * pi / 8;
shimWeights = (1 / sqrt(8)) * exp(-1i * (0:7) * dalpha_8ch);
mode_8ch    = mr.makeRfShim(shimWeights);
seq = mr.Sequence();
seq.addBlock(rf_8ch_base, mode_8ch);
seq.write(fullfile(dataDir, '07_rfstat_cp_8ch_180.seq'));

%% ------------------------------------------------------------------------
% Checksum corruption fixture:
%   Copy gre_2d_1sl_1avg.seq, corrupt a numeric data line in the body,
%   leave [SIGNATURE] section unchanged → MD5 mismatch on verification.
%   Expected error when loaded with verify_signature=1:
%     PULSEQLIB_ERR_SIGNATURE_MISMATCH (-54)
% ------------------------------------------------------------------------
srcFile = fullfile(dataDir, 'gre_2d_1sl_1avg.seq');
dstFile = fullfile(dataDir, 'gre_2d_1sl_1avg_corrupted.seq');

if exist(srcFile, 'file')
    fid_in  = fopen(srcFile, 'r');
    content = {};
    raw = fgetl(fid_in);
    while ischar(raw)
        content{end + 1} = raw; %#ok<AGROW>
        raw = fgetl(fid_in);
    end
    fclose(fid_in);

    in_sig    = false;
    corrupted = false;
    for k = 1 : numel(content)
        if strcmp(strtrim(content{k}), '[SIGNATURE]')
            in_sig = true;
        end
        if ~in_sig && ~corrupted
            % First line that begins with a digit (numeric data row)
            if ~isempty(regexp(content{k}, '^\s*\d', 'once'))
                content{k} = [content{k}, '1'];  % append '1' to corrupt value
                corrupted   = true;
            end
        end
    end

    if corrupted
        fid_out = fopen(dstFile, 'w');
        for k = 1 : numel(content)
            fprintf(fid_out, '%s\n', content{k});
        end
        fclose(fid_out);
        fprintf('Generated corrupted signature fixture: %s\n', dstFile);
    else
        fprintf('WARNING: No corruptible line found in %s\n', srcFile);
    end
else
    fprintf('NOTE: %s not found; skipping checksum fixture.\n', srcFile);
    fprintf('      Run generate_test_sequences.m first, then re-run this script.\n');
end

fprintf('All sequences generated.\n');