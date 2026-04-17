## Quick test that isPyNone works for Python None objects.
import nimpy
import ../../bridge/pypulseq_host
import std/os
import std/times

let builtins = pyBuiltinsModule()
let none = builtins.getAttr("None")
let notNone = builtins.callMethod("int", 42)

doAssert isPyNone(none), "Python None should be detected as None"
doAssert not isPyNone(notNone), "Python int(42) should not be None"

# Bundled Python resolution should work for installer-style and dev-style layouts.
let layoutRoot = getTempDir() / ("pulserver_host_layout_test_" & $epochTime())
discard existsOrCreateDir(layoutRoot)
let installerBin = layoutRoot / "bin"
discard existsOrCreateDir(installerBin)
let installerPython = layoutRoot / "python"
discard existsOrCreateDir(installerPython)
doAssert resolveBundledPythonHome(installerBin).home == installerPython,
  "Expected installer layout to resolve to <root>/python"

let devPython = installerBin / "python"
discard existsOrCreateDir(devPython)
doAssert resolveBundledPythonHome(installerBin).home == devPython,
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

# Test via the actual plugin using a path derived from this source file.
const thisFile = currentSourcePath()
let repoRoot = thisFile.parentDir.parentDir.parentDir
let pluginPath = repoRoot / "bridge" / "tests" / "test_plugin.py"
let plugin = loadPyPlugin(pluginPath)
let defaultProt = plugin.module.callMethod("get_default_protocol", none)
let validation = plugin.module.callMethod("validate_protocol", none, defaultProt)
doAssert validation["valid"].to(bool), "Default protocol should validate successfully"
doAssert not isPyNone(validation["duration"]), "Duration should be present for valid protocol"

echo "All isPyNone tests passed!"
