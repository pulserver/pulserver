# Package

version       = "0.1.0"
author        = "the Pulserver authors"
description   = "Python/MATLAB bridge hosts for pulserver sequence plugins (via nimpulseqgui)"
license       = "MIT"
bin           = @["pypulseq_host", "matlab_host"]
srcDir        = "."

# Dependencies

requires "nim >= 2.0.0"

# Exact versions, with nimble.lock beside this file pinning the whole
# transitive tree to a commit and a checksum. The pulseq Nim packages are all
# pre-1.0, where a minor bump is allowed to break, and the bridge is built
# against upstream unmodified -- so the version it was built against is the
# only one it is known to work with.
requires "nimpy == 0.2.1"
requires "nimpulseqgui == 0.1.0"

# Tasks

task test, "Run bridge test suite":
  let flags = getEnv("NIMFLAGS", "")
  # nimble.lock puts nimble in lock mode, which takes the global package store
  # off nim's search path. The entry point below lives outside this package, so
  # it never reads config.nims/nimble.paths -- the locked paths have to go on
  # the command line. `nimble setup` writes nimble.paths; without it this falls
  # back to the global store, which is what an unlocked checkout wants anyway.
  var paths = ""
  if fileExists("nimble.paths"):
    for line in readFile("nimble.paths").splitLines():
      if line.startsWith("--path:"):
        paths.add " " & line
  exec "nim r" & paths & " " & flags & " ../../tests/nim/test_bridge_common.nim"
