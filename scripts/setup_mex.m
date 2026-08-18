% setup_mex.m — Build the pulseg MEX gateway for MATLAB.
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
%     matlab/+pulserver/private/pulseg_mex.<mexext>

%% ---------- locate repository root ----------
thisScript = mfilename('fullpath');
scriptsDir = fileparts(thisScript);
rootDir    = fileparts(scriptsDir);

src/cDir    = fullfile(rootDir, 'src/c');
mexSrcDir  = fullfile(rootDir, 'matlab', '+pulserver', 'private');
outDir     = mexSrcDir;  % MEX binary goes next to the .cpp

%% ---------- collect source files ----------
% C library sources
cSources = { ...
    fullfile(src/cDir, 'pulseg_cache.c'), ...
    fullfile(src/cDir, 'pulseg_core.c'), ...
    fullfile(src/cDir, 'pulseg_dedup.c'), ...
    fullfile(src/cDir, 'pulseg_error.c'), ...
    fullfile(src/cDir, 'pulseg_freqmod.c'), ...
    fullfile(src/cDir, 'pulseg_getters.c'), ...
    fullfile(src/cDir, 'pulseg_math.c'), ...
    fullfile(src/cDir, 'pulseg_parse.c'), ...
    fullfile(src/cDir, 'pulseg_safety.c'), ...
    fullfile(src/cDir, 'pulseg_structure.c'), ...
    fullfile(src/cDir, 'pulseg_waveforms.c'), ...
    fullfile(src/cDir, 'external_kiss_fft.c'), ...
    fullfile(src/cDir, 'external_kiss_fftr.c'), ...
    fullfile(src/cDir, 'external_md5.c'), ...
};

% MEX C++ gateway
mexSource = fullfile(mexSrcDir, 'pulseg_mex.cpp');

%% ---------- compiler flags ----------
% Include path for C headers
incFlag = ['-I' src/cDir];

%% ---------- build ----------
fprintf('Building pulseg MEX gateway...\n');
fprintf('  Source dir : %s\n', src/cDir);
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
        fullfile(outDir, ['pulseg_mex.' mexext]));
catch ME
    fprintf(2, 'Build failed:\n%s\n', ME.message);
    rethrow(ME);
end

%% ---------- verify ----------
% Quick smoke test: the MEX file should exist
mexFile = fullfile(outDir, ['pulseg_mex.' mexext]);
assert(exist(mexFile, 'file') == 3, ...
    'MEX file not found after build: %s', mexFile);

fprintf('Done. Add the matlab/ folder to your MATLAB path:\n');
fprintf('  >> addpath(''%s'')\n', fullfile(rootDir, 'matlab'));
fprintf('Then use:  sc = pulserver.SequenceCollection(''scan_001.seq'');\n');
