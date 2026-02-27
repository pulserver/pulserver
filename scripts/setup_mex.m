% setup_mex.m — Build the pulseqlib MEX gateway for MATLAB.
%
%   Run from the repository root:
%       >> run('scripts/setup_mex.m')
%
%   Or from any directory (set PULSERVER_ROOT first):
%       >> PULSERVER_ROOT = '/path/to/pulserver';
%       >> run(fullfile(PULSERVER_ROOT, 'scripts', 'setup_mex.m'))
%
%   Prerequisites:
%     - MATLAB R2018a or later (modern C++ MEX API)
%     - A supported C/C++ compiler (gcc, clang, MSVC)
%
%   Output:
%     matlab/+pulserver/private/pulseqlib_mex.<mexext>

%% ---------- locate repository root ----------
thisScript = mfilename('fullpath');
scriptsDir = fileparts(thisScript);
rootDir    = fileparts(scriptsDir);

csrcDir    = fullfile(rootDir, 'csrc');
mexSrcDir  = fullfile(rootDir, 'matlab', '+pulserver', 'private');
outDir     = mexSrcDir;  % MEX binary goes next to the .cpp

%% ---------- collect source files ----------
% C library sources
cSources = { ...
    fullfile(csrcDir, 'pulseqlib_cache.c'), ...
    fullfile(csrcDir, 'pulseqlib_core.c'), ...
    fullfile(csrcDir, 'pulseqlib_dedup.c'), ...
    fullfile(csrcDir, 'pulseqlib_error.c'), ...
    fullfile(csrcDir, 'pulseqlib_freqmod.c'), ...
    fullfile(csrcDir, 'pulseqlib_getters.c'), ...
    fullfile(csrcDir, 'pulseqlib_math.c'), ...
    fullfile(csrcDir, 'pulseqlib_parse.c'), ...
    fullfile(csrcDir, 'pulseqlib_safety.c'), ...
    fullfile(csrcDir, 'pulseqlib_structure.c'), ...
    fullfile(csrcDir, 'pulseqlib_waveforms.c'), ...
    fullfile(csrcDir, 'external_kiss_fft.c'), ...
    fullfile(csrcDir, 'external_kiss_fftr.c'), ...
    fullfile(csrcDir, 'external_md5.c'), ...
};

% MEX C++ gateway
mexSource = fullfile(mexSrcDir, 'pulseqlib_mex.cpp');

%% ---------- compiler flags ----------
% Include path for C headers
incFlag = ['-I' csrcDir];

%% ---------- build ----------
fprintf('Building pulseqlib MEX gateway...\n');
fprintf('  Source dir : %s\n', csrcDir);
fprintf('  MEX source : %s\n', mexSource);
fprintf('  Output dir : %s\n', outDir);

% Combine all args
allSources = [cSources, {mexSource}];

try
    mex('-R2018a', ...
        '-outdir', outDir, ...
        incFlag, ...
        allSources{:});
    fprintf('Build successful: %s\n', ...
        fullfile(outDir, ['pulseqlib_mex.' mexext]));
catch ME
    fprintf(2, 'Build failed:\n%s\n', ME.message);
    rethrow(ME);
end

%% ---------- verify ----------
% Quick smoke test: the MEX file should exist
mexFile = fullfile(outDir, ['pulseqlib_mex.' mexext]);
assert(exist(mexFile, 'file') == 3, ...
    'MEX file not found after build: %s', mexFile);

fprintf('Done. Add the matlab/ folder to your MATLAB path:\n');
fprintf('  >> addpath(''%s'')\n', fullfile(rootDir, 'matlab'));
fprintf('Then use:  sc = pulserver.SequenceCollection(''scan_001.seq'');\n');
