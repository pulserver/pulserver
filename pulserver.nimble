# Package

version       = "0.1.0"
author        = "the Pulserver authors"
description   = "Top-level Nimble manifest for pulserver bridge tasks"
license       = "MIT"
srcDir        = "."

# Dependencies

requires "nim >= 2.0.0"

# Tasks

task test_nim, "Run top-level Nim bridge tests":
  exec "bash tests/nim/run_tests.sh"

task build_bridge, "Build bridge hosts with Nimble":
  exec "cd src/nim && nimble build -y"
