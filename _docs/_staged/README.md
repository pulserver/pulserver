# Staged for rewriting

Excluded from the Sphinx build (see `exclude_patterns` in `docs/conf.py`) and
in no toctree.

These pages are written around the example sequence plugins, which are being
rebuilt on the current `SequenceModule`; the sources they were written against
are staged the same way under
`python/pulserver/design/_disabled/`. They are kept because their *structure*
— what a plugin tutorial has to cover, in what order — is worth rewriting
from. Rewrite a page against the shipped API, then move it back into
`docs/tutorials/` or `docs/how-to/` and add it to the toctree in
`docs/tutorials.md` or `docs/how_to.md`.
