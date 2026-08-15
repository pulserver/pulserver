from __future__ import annotations

import sys
from pathlib import Path

# The lane helpers (_lanes.py) import by bare name whatever directory pytest
# runs from.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
