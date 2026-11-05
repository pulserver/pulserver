# Releases

Versions are derived from Git tags by `setuptools_scm`. Pushing a tag `vX.Y.Z`
runs the Release workflow, which builds the wheels (Linux x86-64, macOS
arm64 and Windows x86-64, CPython 3.10 to 3.14) and the source distribution, publishes them to
PyPI through trusted publishing, and creates a GitHub release with Sigstore
signatures.

The Docs workflow publishes the documentation to the `gh-pages` branch, one
directory per version: `latest` from `main`, `vX.Y.Z` for each release tag, and
`stable`, a copy of the newest release. `scripts/publish_docs.py` places a
built site into that tree, keeps the `--keep` newest releases beside `latest`
and `stable`, and rewrites the root redirect and the `versions.json` the
version switcher reads.
