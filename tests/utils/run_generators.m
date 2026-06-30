function run_generators()
this_dir = fileparts(mfilename('fullpath'));
scripts_dir = fullfile(this_dir, 'scripts');
pulseq_matlab_dir = fullfile(this_dir, 'pulseq', 'matlab');

addpath(this_dir);
addpath(scripts_dir);

if isfolder(pulseq_matlab_dir) || isdir(pulseq_matlab_dir)
    addpath(pulseq_matlab_dir);
else
    error('Pulseq MATLAB library not found at: %s', pulseq_matlab_dir);
end

scripts = dir(fullfile(scripts_dir, 'generate_*.m'));
names = {scripts.name};
[~, script_order] = sort(names);

for idx = 1:numel(script_order)
    script_name = scripts(script_order(idx)).name;
    script_path = fullfile(scripts_dir, script_name);
    fprintf('Running %s\n', script_name);
    run_generator_script(script_path);
end

% Post-pass: build pulseqlib cache .pge alongside each *valid* .seq fixture.
% Fixtures named *_fail_* or *_corrupted are intentional negative-test inputs
% (consumed by ctests for safety/IO assertions); pulseqlib_read is expected
% to reject them, so we skip them here.
write_cache_for_fixtures(this_dir);
end

function run_generator_script(script_path)
run(script_path);
end

function write_cache_for_fixtures(utils_dir)
expected_dir = fullfile(utils_dir, '..', '..', '..', 'pulserverlib-tests', 'expected');
expected_dir = char(expected_dir);
if exist(expected_dir, 'dir') ~= 7
    fprintf('write_cache: expected/ dir not found at %s; skipping cache build.\n', expected_dir);
    return;
end

cli = fullfile(utils_dir, '..', '..', '..', 'mrdserver', 'tests', 'cpp', ...
               'write_cache_cli', 'write_cache');
cli = char(cli);
if exist(cli, 'file') ~= 2
    fprintf(['write_cache: CLI not found at %s; skipping cache build. ', ...
             'Build it via the instructions in mrdserver/tests/cpp/write_cache_cli/.\n'], cli);
    return;
end

seqs = dir(fullfile(expected_dir, '*.seq'));
% Explicit skip list: fixtures that pulseqlib_read cannot accept under the
% CLI's default options. Each entry has a justification.
explicit_skip = { ...
    '05_rfprep_ok_canonical_fullpass.seq' ... % requires NEX=3 (load_seq_with_averages); single-NEX read fails consistency check (rc=-560)
};
n_ok = 0; n_skip = 0; n_fail = 0;
for i = 1:numel(seqs)
    name = seqs(i).name;
    stem = name(1:end-4);  % strip .seq
    if ~isempty(strfind(name, '_fail_')) || ~isempty(strfind(stem, '_corrupted')) ...
            || any(strcmp(name, explicit_skip))
        n_skip = n_skip + 1;
        continue;
    end
    seq_path = fullfile(expected_dir, name);
    cmd = sprintf('"%s" "%s"', cli, seq_path);
    [status, output] = system(cmd);
    if status == 0
        n_ok = n_ok + 1;
    else
        n_fail = n_fail + 1;
        fprintf('write_cache FAIL %s: %s\n', name, strtrim(output));
    end
end
fprintf('write_cache: %d cached, %d skipped (negative fixtures), %d failed\n', ...
        n_ok, n_skip, n_fail);
end

run_generators();
