## matlab_host — pre-compiled generic host for MATLAB-based Pulseq plugins.
##
## Loads a user-supplied ``.m`` script at runtime via ``--script`` flag.
## The MATLAB script provides the three sequence callbacks
## (``get_default_protocol``, ``validate_protocol``, ``make_sequence``).
##
## **Phase 5 stub** — full implementation will use either:
##   - MATLAB Engine API for C (subprocess) in development
##   - MCR (MATLAB Compiler Runtime) for deployment
##
## For now this binary prints a not-yet-implemented message and exits.

import bridge_common
import std/os

proc main() =
  echo "matlab_host: not yet implemented (Phase 5)."
  echo "Usage: matlab_host --script <plugin.m> [nimpulseqgui flags...]"
  quit(1)

when isMainModule:
  main()
