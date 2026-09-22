# Documentation

The documentation is written for MR scientists who run or extend pulserver. It
is built with Sphinx from Markdown (MyST) pages and the docstrings of the
package.

## Documentation types

| Location | Type | Answers |
| --- | --- | --- |
| `docs/user-guide/` | How-to | How do I install, run and extend pulserver? |
| `docs/explanations/` | Conceptual explanation | Why does it work this way? |
| `docs/api/` | Reference | What exactly does this object do? |
| `docs/developer-guide/` | Contributor procedure | How is this repository developed? |
| `README.md` | Project summary | What is this, and where is the rest? |

The prose style of one type is not carried into another. A how-to page states
steps and the facts a reader needs to follow them, and links to the explanation
rather than repeating it. An explanation page describes a design and its
reasons, not a procedure. The API reference is extracted from the docstrings;
an API page adds only a one-paragraph orientation and the object tables.

## Writing

- Use the vocabulary of pulse sequences, MRD and the Pulseq format, not of
  software architecture. Keep established terms: *repetition time*, *readout*,
  *encoding space*, *trajectory*, *revision*.
- State units, frames and rasters wherever a value crosses an interface.
- Write dry, declarative prose. Do not personify services, plugins or files.
- Keep four layers apart: the Pulseq representation, pulserver's orchestration,
  the engines (pypulseqpp, bartorch), and the scanner interpreter.
- Do not print a measured number that varies across releases or hardware.
- Verify a statement against the implementation and its tests. Existing prose
  is not evidence.

## Examples

A code block a reader could run is written as a doctest (a `pycon` block with
`>>>` prompts) and executed by `tests/test_docs.py`, so it cannot drift from the
code. Blocks that need a running service or a data file are plain `python`
blocks and are not executed.

## API pages

Each subpackage has a page in `docs/api/` whose `autosummary` tables list its
public names; `tests/test_docs.py` fails when a name in a subpackage's
`__all__` is missing from its page. autosummary writes the per-object pages
into `docs/generated/`, which is not tracked.

## Building

```bash
bash scripts/build_docs.sh
```

The build treats every warning as an error and writes `docs/build/html/`. On
`main` and on release tags, the Docs workflow publishes the pages to GitHub
Pages, one directory per version: `latest` for `main`, the tag for a release,
and `stable` for the newest release.
