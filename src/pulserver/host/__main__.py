"""Answer one design call: ``python -m pulserver.host CALL ...``, as ``pulserver design`` does."""

import sys

from ._command import main

if __name__ == "__main__":
    sys.exit(main())
