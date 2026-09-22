---
name: write-documentation
description: Write, move or audit pulserver documentation — docstrings, user-guide, explanation, API and developer-guide pages, and the README. Use before creating or substantially modifying any documentation, and when deciding which documentation type a piece of material belongs to.
---

# Write documentation

`docs/developer-guide/documentation.md` governs the documentation, and the
`Docstrings`, `Comments` and `Documentation style` sections of `AGENTS.md`
govern docstrings and comments. Read them before writing.

## Choose the type first

| Location | Type | Answers |
|---|---|---|
| `docs/user-guide/` | How-to | How do I install, run and extend pulserver? |
| `docs/explanations/` | Conceptual explanation | Why does it work this way? |
| `docs/api/` | Reference, from the docstrings | What exactly does this object do? |
| `docs/developer-guide/` | Contributor procedure | How is this repository developed? |
| `README.md` | Project summary | What is this, and where is the rest? |

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
- An API page lists its subpackage's public names in `autosummary` blocks,
  three-space indented one per line; the stubs are generated into
  `docs/generated/`, which is not tracked.
- A new page is added to the table and the hidden toctree of its section's
  `index.md`.

## Validate

```bash
pytest -q tests/test_docs.py tests/test_docstrings.py
bash scripts/build_docs.sh
```

Open `docs/build/html/index.html` and check the rendered pages and the sidebar.
