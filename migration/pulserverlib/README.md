# pulserverlib

This scaffold contains the high-level Python and MATLAB interfaces over Pulseq core libraries.

## Included from monorepo
- `python/pulserver/core/`
- `python/pulserver/__init__.py`
- `matlab/`
- `pyproject.toml`, `setup.py`, `CMakeLists.txt`
- `tests/pytests/`

## Notes
- Designed to consume `pulserverlib-source` as a submodule/dependency.
- Python layout and scikit-build backend are kept for extension builds.
