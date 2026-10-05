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
| `docs/explanations/` | Conceptual explanation, one flat list in data-flow order | Why does it work this way? |
| `gallery/01-course/` | Course lessons, read in order | What does a new user need to work independently? |
| `gallery/02-tours/` | Tours: applications, advanced branches, specialised workflows | What does a representative workflow look like? |
| `docs/api/` | Reference, from the docstrings | What exactly does this object do? |
| `docs/developer-guide/` | Contributor procedure | How is this repository developed? |
| `docs/developer-guide/internals/` | Implementation design | How is it built, and what does a playout rely on? |
| `README.md` | Project summary and documentation landing page | What is this, and where is the rest? |

Material in the wrong type is moved, not deleted. Implementation design found
on an explanation page — file layouts, memory budgets, tolerances, test
inventories — moves to `docs/developer-guide/internals/`, and the explanation
links to it.

A lesson belongs in the course only if it is on the shortest coherent path
that gives a new user the framework's core mental model and enough practical
competence to work independently; anything else is a Tour. A lesson has an
introduction, a *Learning objectives* list and previous/next lines; a Tour
opens with its objective and a prerequisites line and has neither.

The IR is written in PulSeg's terms — base block, virtual segment, segment
instance, execution stream — and *repetition* is the period
`pypulseqpp.Sequence.repetition()` reports. Never "repeating unit", never
"app" for a sequence function (`terminology.md`).

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
  the API pages it relates to. One with more than one `##` section opens with
  a TL;DR admonition, ```` ```{admonition} TL;DR ```` and `:class: tldr`,
  directly under its title; no landing, API or example page has one.
- A figure takes its style from `docs/figure_style.py`; a script sets no font
  size, DPI or colour literal and imports the colours it needs. A static
  figure is a light/dark SVG pair written by `scripts/make_artwork.py` and
  shown with `only-light` and `only-dark`.
- A new page is added to the table and the hidden toctree of its section's
  `index.md`.
- A new gallery script goes into its section's directory under `gallery/`,
  named with a numeric prefix, and gets a row and a toctree entry on the
  section's landing page under `docs/examples/` (`course.md` or `tours.md`). A new section is a directory
  with `README.rst` and `_gallery_header.md`, an entry in `GALLERY_SECTIONS`,
  and a landing page listed in `docs/examples/index.md`.

## Validate

```bash
pytest -q tests/test_docs.py tests/test_docstrings.py tests/test_docstring_defaults.py
bash scripts/build_docs.sh
```

Open `docs/build/html/index.html` and check the rendered pages in the light
and the dark theme, and the sidebar: User guide, Developer guide, Explanations,
Examples (Course, then Tours), API, Misc.
