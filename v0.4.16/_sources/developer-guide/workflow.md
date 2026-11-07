# Contribution workflow

## Branches

1. Fork `pulserver/pulserver` on GitHub and clone the fork.
2. Add `https://github.com/pulserver/pulserver.git` as the `upstream` remote.
3. Update your local `main` from `upstream` and create a focused branch.
4. Commit tested changes and push the branch to your fork.
5. Open a pull request against `pulserver/pulserver:main`.

Keep commits reviewable, without unrelated formatting or prose changes. A
change to sequence design belongs in
[pypulseqpp](https://github.com/pulserver/pypulseqpp), and one to a
reconstruction algorithm in [bartorch](https://github.com/mcencini/bartorch).

## Pull requests

The repository's pull-request template asks for what changed, why, the exact
validation commands with their results, and the documentation impact. New
behaviour comes with a regression test where practical.

## Continuous integration

| Workflow | Runs on | Checks |
| --- | --- | --- |
| Style | every pull request, `main` | `scripts/format_and_lint.sh --check` |
| Tests | every pull request, `main` | the suite on Linux and macOS at Python 3.10 and 3.14; on Linux, also the coil models, bartorch reconstructions and console sequences |
| Docs | a maintainer's `/build-docs` comment on a pull request, `main`, release tags | the documentation build with every gallery script executed; publishes from `main` and tags only |
| Console | pull requests that change `docker/`, the console or the proxy | one exam scanned through the virtual scanner's console, the proxy and Gadgetron in Docker Compose |
| Image | pull requests that change `docker/`, `src/` or the build, `main` | the Docker image, scanned through its own console; published from `main` |
| Fields | on demand | the virtual coils' field maps and VOPs, solved with mariepy and published as an image |
| CodeQL | every pull request, `main` | static analysis |
| Release | release tags | wheels, PyPI and the GitHub release ({doc}`releases`) |

## Code of conduct

Issues, discussions and pull requests follow the root
[`CODE_OF_CONDUCT.md`](https://github.com/pulserver/pulserver/blob/main/CODE_OF_CONDUCT.md).
