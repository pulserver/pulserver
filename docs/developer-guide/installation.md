# Editable installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,doc]'
```

The `dev` extra contains the lint and test tools and the `doc` extra the
documentation tools, including sphinx-gallery and matplotlib for the examples,
which are executed while the pages are built. The editable install compiles the extension into the build
directory; a change to `src/cpp/` or `src/c/` is compiled again on the next
import.
