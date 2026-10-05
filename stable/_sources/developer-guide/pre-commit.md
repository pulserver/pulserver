# Pre-commit

```bash
pre-commit install
pre-commit run --all-files
```

The local hook runs `scripts/format_and_lint.sh`, the script the Style workflow
runs with `--check`. Test fixtures under `tests/fixtures/` are excluded from
the whitespace hooks.

A deliberate one-off bypass uses `git commit --no-verify`; one hook can be
skipped with `SKIP=<hook-id> git commit`. A bypass does not bypass CI and must
not be used to conceal a failing check.
