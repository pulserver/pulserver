---
name: write-documentation
description: Write, move or audit pulserver documentation — docstrings, user-guide, explanation, API and developer-guide pages, and the README. Use before creating or substantially modifying any documentation, and when deciding which documentation type a piece of material belongs to.
---

# Write documentation

Two documents govern documentation and both are binding. Read them before
writing:

- `docs/developer-guide/documentation.md` — what belongs in each form of
  documentation and how each is written.
- `docs/developer-guide/terminology.md` — terminology, register, units,
  frames, safety language and source-of-truth rules.

The `Docstrings` and `Comments` sections of `AGENTS.md` govern docstrings and
comments.

## Choose the type first

| Location | Type | Answers |
|---|---|---|
| `docs/user-guide/` | How-to | How do I install, run and extend pulserver? |
| `docs/explanations/` | Conceptual explanation | Why does it work this way? |
| `docs/api/` | Reference, from the docstrings | What exactly does this object do? |
| `docs/developer-guide/` | Contributor procedure | How is this repository developed? |
| `README.md` | Project summary and documentation landing page | What is this, and where is the rest? |

Material in the wrong type is moved, not deleted.

## Verify before writing

State nothing the code does not do. Read the implementation, its callers and
its tests, and the C or C++ behind it when there is any. For the file formats
and wire protocols, the authorities are the code in `src/pulserver/protocol/`,
`src/c/include/pulseg/` and `src/pulserver/recon/_runtime/`. Flag a discrepancy
you cannot resolve rather than guessing.

## Mechanics

- A runnable example on a user-guide page is a `pycon` block with `>>>`
  prompts, ending with a blank line before the closing fence; it is executed
  by `tests/test_docs.py`. An example that needs a running service or a data
  file is a plain `python` block.
- An API page lists its subpackage's public names in `| Object | Description |`
  tables of `{obj}` links under a `currentmodule` directive.
  `docs/api_objects.py` writes the per-object stubs from those tables into
  `docs/generated/`, which is not tracked.
- An explanation page ends with a `## See also` list linking the how-to and
  the API pages it relates to.
- A new page is added to the table and the hidden toctree of its section's
  `index.md`.

## Validate

```bash
pytest -q tests/test_docs.py tests/test_docstrings.py
bash scripts/build_docs.sh
```

Open `docs/build/html/index.html` and check the rendered pages and the sidebar.
