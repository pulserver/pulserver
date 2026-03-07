# Migration Scaffolds

This directory contains initial scaffolds for the planned repository split.

## Repositories scaffolded
1. `pulserverlib-source`
2. `pulserverlib`
3. `gadgetronlib-source`
4. `gadgetronlib`
5. `pulserverlib-interface`
6. `pulserverlib-tools`
7. `gadgetron-tools`
8. `pulserver-interpreter` (public placeholder only)

## Scope of this scaffold
- Creates folder layout for each target repo.
- Copies selected relevant files from the current monorepo.
- Adds baseline hygiene files:
  - `.gitignore`
  - `.gitattributes`
  - `.pre-commit-config.yaml`
  - `.github/prompts/`
- Adds per-repo `README.md` notes.

## Next migration steps
- Trim `.gitignore` and workflow files per repo role.
- Create repo-specific CI workflows under `.github/workflows/`.
- Refine package metadata (`pyproject.toml`) for each split target.
- Decide history-preserving extraction per repo and execute with `git-filter-repo` where needed.
