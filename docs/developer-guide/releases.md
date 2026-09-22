# Releases

Versions are derived from Git tags by `setuptools_scm`. Pushing a tag
`vX.Y.Z` runs the release workflow, which builds the wheels and the source
distribution, publishes them to PyPI through trusted publishing, and creates a
GitHub release with Sigstore signatures.

The Docs workflow builds the documentation for every pull request and publishes
it on pushes to `main` and on release tags. The site holds one directory per
version: `latest` for `main`, `vX.Y.Z` for each release, and `stable`, a copy
of the newest release.
