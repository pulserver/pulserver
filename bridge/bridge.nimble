# Package

version       = "0.1.0"
author        = "Matteo Cencini"
description   = "Python/MATLAB bridge hosts for pulserver sequence plugins (via nimpulseqgui)"
license       = "MIT"
bin           = @["pypulseq_host", "matlab_host"]
srcDir        = "."

# Dependencies

requires "nim >= 2.0.0"
requires "nimpy >= 0.2.0"
requires "nimpulseqgui >= 0.1.0"  # stock, unmodified (installed from GitHub in test bootstrap)

# Tasks

task test, "Run bridge test suite":
  exec "nim r ../tests/nim/test_bridge_common.nim"
  exec "nim r ../tests/nim/test_isPyNone.nim"
