## Regression checks for bundled Python resolution in bridge host code.
import ../../bridge/pypulseq_host
import std/os
import std/times

# Mirror the strdefine from pypulseq_host so -d:pythonVenvPath=... is visible here.
const pythonVenvPath {.strdefine.} = ""

# Bundled Python resolution should work for installer-style and dev-style layouts.
let layoutRoot = getTempDir() / ("pulserver_host_layout_test_" & $epochTime())
discard existsOrCreateDir(layoutRoot)
let installerBin = layoutRoot / "bin"
discard existsOrCreateDir(installerBin)
let installerPython = layoutRoot / "python"
discard existsOrCreateDir(installerPython)
# Must have lib/ + bin/ so resolveBundledPythonHome recognises it as bpkStandalone.
discard existsOrCreateDir(installerPython / "lib")
discard existsOrCreateDir(installerPython / "bin")
doAssert resolveBundledPythonHome(installerBin, "").home == installerPython,
  "Expected installer layout to resolve to <root>/python"

let devPython = installerBin / "python"
discard existsOrCreateDir(devPython)
discard existsOrCreateDir(devPython / "lib")
discard existsOrCreateDir(devPython / "bin")
doAssert resolveBundledPythonHome(installerBin, "").home == devPython,
  "Expected dev layout (<exe_dir>/python) to take precedence"

# Verify the canonical pythonVenvPath (EnvPulserver) takes priority over relative
# fallbacks when it exists.  run_tests.sh compiles with -d:pythonVenvPath pointing
# to a copy of the bundle venv so this always runs in CI.
when pythonVenvPath != "":
  if dirExists(pythonVenvPath):
    # Use a fake exeDir with no relative python/ dir so only the canonical path matches.
    let fakeExeDir = getTempDir() / ("pulserver_fake_exe_" & $epochTime())
    discard existsOrCreateDir(fakeExeDir)
    let canonical = resolveBundledPythonHome(fakeExeDir)
    doAssert canonical.home == pythonVenvPath,
      "Expected canonical venv (pythonVenvPath) to be picked first: got " & canonical.home
    doAssert canonical.kind == bpkVenv,
      "Expected canonical path to be detected as venv (pyvenv.cfg must exist)"
    # Verify that packages installed in the canonical venv are actually importable.
    # This confirms the bridge would load from EnvPulserver, not the system Python.
    let pyBin = pythonVenvPath / "bin" / "python3"
    let (output, code) = gorgeEx(pyBin & " -c 'import numpy, scipy, pypulseq, pulserver; print(\"ok\")'")
    doAssert code == 0,
      "Expected packages in canonical venv to be importable, but got:\n" & output
    echo "Canonical venv test passed: packages importable from " & pythonVenvPath

echo "Bundled Python resolution checks passed!"
